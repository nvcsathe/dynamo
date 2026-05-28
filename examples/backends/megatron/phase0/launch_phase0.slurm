#!/bin/bash
# Phase-0 Megatron+Dynamo launcher: 1 node, 8B model, TP=2, aggregated decode,
# token streaming. Brings up the Megatron coordinator + the Dynamo Megatron
# worker on the same node, both writing logs to the shared filesystem.
#
# The Dynamo frontend (HTTP server, router) must be running separately and
# pointing at the same NATS/etcd as this worker. See README.md.
#
#SBATCH --job-name=megatron-dynamo-phase0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# REQUIRED OVERRIDES (pass via --export or sbatch CLI):
#   MEGATRON_WORKTREE   absolute path to the Megatron-LM checkout
#   DYNAMO_WORKTREE     absolute path to the dynamo checkout (this repo)
#   MODEL_CHECKPOINT    path to the 8B Megatron checkpoint (--load)
#   TOKENIZER_MODEL     HF model id for the tokenizer (e.g. meta-llama/Llama-3.1-8B)
#   SERVED_MODEL_NAME   model name to advertise to clients (often == TOKENIZER_MODEL)
#   ETCD_ENDPOINTS      etcd URL(s) the Dynamo runtime should attach to
#   NATS_SERVER         NATS URL the Dynamo runtime should attach to

set -euo pipefail

: "${MEGATRON_WORKTREE:?must be set}"
: "${DYNAMO_WORKTREE:?must be set}"
: "${MODEL_CHECKPOINT:?must be set}"
: "${TOKENIZER_MODEL:?must be set}"
: "${SERVED_MODEL_NAME:=${TOKENIZER_MODEL}}"
: "${ETCD_ENDPOINTS:?must be set}"
: "${NATS_SERVER:?must be set}"

mkdir -p logs

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29500}
export NNODES=${SLURM_NNODES}
export GPUS_PER_NODE=2
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

# Pre-Blackwell with TP=2 requires this. On Blackwell it is a no-op.
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Dynamo runtime endpoints.
export ETCD_ENDPOINTS NATS_SERVER

COORD_LOG="logs/coordinator-${SLURM_JOB_ID}.out"
WORKER_LOG="logs/dynamo-worker-${SLURM_JOB_ID}.out"

###############################################################################
# Step 1: launch the Megatron coordinator (TP=2). It prints
#   MEGATRON_COORDINATOR_ADDR=tcp://<host>:<port>
# on its first stdout line after the coordinator binds. We tail that file
# until we see the address.
###############################################################################
cd "$MEGATRON_WORKTREE"

srun --ntasks=1 --nodes=1 --gpus-per-node="$GPUS_PER_NODE" --output="$COORD_LOG" \
  bash -c '
    uv run python -m torch.distributed.run \
      --nnodes='"${NNODES}"' \
      --nproc-per-node='"${GPUS_PER_NODE}"' \
      --node-rank=0 \
      --master-addr='"${MASTER_ADDR}"' \
      --master-port='"${MASTER_PORT}"' \
      tools/run_dynamic_text_generation_server.py \
        --frontend dynamo \
        --inference-coordinator-port 5555 \
        --tensor-model-parallel-size 2 \
        --pipeline-model-parallel-size 1 \
        --load "'"${MODEL_CHECKPOINT}"'" \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model "'"${TOKENIZER_MODEL}"'" \
        --bf16 \
        --use-flash-attn
  ' &
COORD_PID=$!

echo "Coordinator launched (pid=$COORD_PID), waiting for address..."
COORD_ADDR=""
for _ in $(seq 1 60); do
  if [[ -f "$COORD_LOG" ]]; then
    COORD_ADDR=$(grep -m1 -oP 'MEGATRON_COORDINATOR_ADDR=\K.*' "$COORD_LOG" || true)
    [[ -n "$COORD_ADDR" ]] && break
  fi
  if ! kill -0 "$COORD_PID" 2>/dev/null; then
    echo "ERROR: coordinator process exited before printing address; see $COORD_LOG" >&2
    exit 1
  fi
  sleep 5
done
if [[ -z "$COORD_ADDR" ]]; then
  echo "ERROR: timed out waiting for MEGATRON_COORDINATOR_ADDR in $COORD_LOG" >&2
  kill "$COORD_PID" 2>/dev/null || true
  exit 1
fi
echo "Got coordinator: $COORD_ADDR"

###############################################################################
# Step 2: launch the Dynamo Megatron worker pinned to no GPU. It just speaks
# ZMQ to the coordinator and NATS/etcd to the Dynamo control plane.
###############################################################################
cd "$DYNAMO_WORKTREE"

srun --ntasks=1 --nodes=1 --overlap --output="$WORKER_LOG" \
  bash -c '
    python -m dynamo.megatron \
      --coordinator-addr "'"${COORD_ADDR}"'" \
      --model "'"${TOKENIZER_MODEL}"'" \
      --served-model-name "'"${SERVED_MODEL_NAME}"'" \
      --context-length 4096
  ' &
WORKER_PID=$!

# Cleanup hook: tearing down the worker also stops the coordinator.
trap 'kill "$WORKER_PID" "$COORD_PID" 2>/dev/null || true' EXIT INT TERM

echo "Dynamo worker launched (pid=$WORKER_PID). Logs:"
echo "  coordinator: $COORD_LOG"
echo "  worker:      $WORKER_LOG"

wait "$WORKER_PID"
WORKER_EXIT=$?
kill "$COORD_PID" 2>/dev/null || true
wait "$COORD_PID" 2>/dev/null || true
exit "$WORKER_EXIT"
