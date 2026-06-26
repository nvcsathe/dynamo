#!/usr/bin/env bash
# Orchestrates the disaggregated stack on one node:
#
#     NATS + etcd
#     │
#     ├── Megatron coordinator [PREFILL]  (torchrun TP=$TP_PREFILL, GPUs 0..N-1)
#     │     - --frontend dynamo --disagg-role prefill
#     │     - --kv-transfer-listen-addr 0.0.0.0:$NIXL_PORT_PREFILL
#     │
#     ├── Megatron coordinator [DECODE]   (torchrun TP=$TP_DECODE, GPUs N..2N-1)
#     │     - --frontend dynamo --disagg-role decode --enable-prefix-caching
#     │     - --kv-transfer-listen-addr 0.0.0.0:$NIXL_PORT_DECODE
#     │
#     ├── dynamo.megatron worker [PREFILL]  --role prefill   --coordinator-addr <prefill coord>
#     ├── dynamo.megatron worker [DECODE]   --role decode    --coordinator-addr <decode coord>
#     └── dynamo.frontend
#
# Outputs:
#   - /tmp/phase3.env       : env-vars sourced by test scripts
#   - /tmp/{nats,etcd,coordinator-prefill,coordinator-decode,worker-prefill,
#           worker-decode,frontend}.log
#   - On stdout: "PHASE3_READY" once the stack is healthy.
#
# Decode requires prefix caching for imported KV blocks.

set -uo pipefail

# Override image UCX defaults so NIXL can register and transfer VRAM safely.
export UCX_TLS="${UCX_TLS_OVERRIDE:-cuda_ipc,cuda_copy,tcp,shm,cma,self}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE_OVERRIDE:-n}"
export UCX_LOG_LEVEL="${UCX_LOG_LEVEL_OVERRIDE:-info}"
export UCX_LOG_FILE="${UCX_LOG_FILE_OVERRIDE:-/tmp/ucx_%p.log}"

STAGE="${STAGE:-/lustre/fsw/portfolios/nemotron/users/csathe}"
MODEL_DIR="${MODEL_DIR:-llama3.1-8b-mcore}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$STAGE/models/${MODEL_DIR}}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama-3.1-8b-instruct}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"

# Disagg topology. Heterogeneous TP and PP are supported: prefill and decode
# may use different TP and/or PP values. Total GPU count is
# (TP_PREFILL×PP_PREFILL) + (TP_DECODE×PP_DECODE).
TP_PREFILL="${TP_PREFILL:-1}"
TP_DECODE="${TP_DECODE:-1}"
PP_PREFILL="${PP_PREFILL:-1}"
PP_DECODE="${PP_DECODE:-1}"

HTTP_PORT="${HTTP_PORT:-8100}"
COORD_PORT_PREFILL="${COORD_PORT_PREFILL:-5555}"
COORD_PORT_DECODE="${COORD_PORT_DECODE:-5556}"
NIXL_PORT_PREFILL="${NIXL_PORT_PREFILL:-7000}"
NIXL_PORT_DECODE="${NIXL_PORT_DECODE:-7001}"
MASTER_PORT_PREFILL="${MASTER_PORT_PREFILL:-29500}"
MASTER_PORT_DECODE="${MASTER_PORT_DECODE:-29501}"

export NATS_SERVER="nats://127.0.0.1:4222"
export ETCD_ENDPOINTS="http://127.0.0.1:2379"
export HF_HOME="${HF_HOME:-${STAGE}/hf-cache}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=127.0.0.1

LOG_DIR="${LOG_DIR:-/tmp}"
PIDS=()

log()  { printf '[orchestrate %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { log "FATAL: $*" >&2; cleanup; exit 1; }

cleanup() {
    log "cleaning up..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for() {
    local desc="$1" max="$2"; shift 2
    local elapsed=0
    while ! "$@" >/dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [[ $elapsed -ge $max ]]; then
            log "TIMEOUT waiting for $desc after ${max}s"
            return 1
        fi
    done
    log "ready: $desc (${elapsed}s)"
}

###############################################################################
# 1. NATS + etcd
###############################################################################
log "starting NATS..."
nats-server --jetstream --store_dir /tmp/nats-jetstream --port 4222 -m 8222 \
    > "$LOG_DIR/nats.log" 2>&1 &
PIDS+=($!)

log "starting etcd..."
etcd --data-dir /tmp/etcd-data \
     --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://0.0.0.0:2379 \
     > "$LOG_DIR/etcd.log" 2>&1 &
PIDS+=($!)

wait_for "nats /healthz"  30 curl -sf http://127.0.0.1:8222/healthz || die "nats never healthy"
wait_for "etcd /health"   30 curl -sf http://127.0.0.1:2379/health  || die "etcd never healthy"

###############################################################################
# 2. Two Megatron coordinators (prefill + decode)
###############################################################################
# Model-architecture flags. Override for non-Llama-3.1-8B.
MODEL_ARGS=(
    --ckpt-format torch_dist
    --use-checkpoint-args
    --disable-bias-linear
    --transformer-impl transformer_engine
    --normalization RMSNorm
    --group-query-attention --num-query-groups 8
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --rotary-percent 1.0
    --rotary-base 500000
    --use-rope-scaling
    --use-rotary-position-embeddings
    --swiglu
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --max-position-embeddings 131072
    --seq-length 8192
    --micro-batch-size 1
    --bf16
)

# Compute GPU assignments. Prefill gets GPUs [0, TP_PREFILL×PP_PREFILL);
# decode gets the next TP_DECODE×PP_DECODE GPUs.
PREFILL_NPROC=$((TP_PREFILL * PP_PREFILL))
DECODE_NPROC=$((TP_DECODE  * PP_DECODE))
PREFILL_GPUS=$(seq -s, 0 $((PREFILL_NPROC - 1)))
DECODE_GPUS=$(seq -s, $PREFILL_NPROC $((PREFILL_NPROC + DECODE_NPROC - 1)))

log "starting Megatron PREFILL coordinator (TP=$TP_PREFILL PP=$PP_PREFILL, GPUs=$PREFILL_GPUS)..."
(
    cd /opt/megatron-lm
    CUDA_VISIBLE_DEVICES="$PREFILL_GPUS" exec python -m torch.distributed.run \
        --nnodes=1 --nproc-per-node="$PREFILL_NPROC" --node-rank=0 \
        --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT_PREFILL" \
        tools/run_dynamic_text_generation_server.py \
            --frontend dynamo \
            --disagg-role prefill \
            --kv-transfer-listen-addr "0.0.0.0:$NIXL_PORT_PREFILL" \
            --inference-coordinator-port "$COORD_PORT_PREFILL" \
            --tensor-model-parallel-size "$TP_PREFILL" \
            --pipeline-model-parallel-size "$PP_PREFILL" \
            --load "$MODEL_CHECKPOINT" \
            --tokenizer-type HuggingFaceTokenizer \
            --tokenizer-model "$TOKENIZER_MODEL" \
            "${MODEL_ARGS[@]}"
) > "$LOG_DIR/coordinator-prefill.log" 2>&1 &
PIDS+=($!)

log "starting Megatron DECODE coordinator (TP=$TP_DECODE PP=$PP_DECODE, GPUs=$DECODE_GPUS)..."
(
    cd /opt/megatron-lm
    CUDA_VISIBLE_DEVICES="$DECODE_GPUS" exec python -m torch.distributed.run \
        --nnodes=1 --nproc-per-node="$DECODE_NPROC" --node-rank=0 \
        --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT_DECODE" \
        tools/run_dynamic_text_generation_server.py \
            --frontend dynamo \
            --disagg-role decode \
            --kv-transfer-listen-addr "0.0.0.0:$NIXL_PORT_DECODE" \
            --inference-coordinator-port "$COORD_PORT_DECODE" \
            --tensor-model-parallel-size "$TP_DECODE" \
            --pipeline-model-parallel-size "$PP_DECODE" \
            --inference-dynamic-batching-prefix-caching \
            --load "$MODEL_CHECKPOINT" \
            --tokenizer-type HuggingFaceTokenizer \
            --tokenizer-model "$TOKENIZER_MODEL" \
            "${MODEL_ARGS[@]}"
) > "$LOG_DIR/coordinator-decode.log" 2>&1 &
PIDS+=($!)

wait_for "prefill coordinator announced" 600 \
    grep -q "MEGATRON_COORDINATOR_ADDR=" "$LOG_DIR/coordinator-prefill.log" \
    || die "prefill coordinator never announced (see $LOG_DIR/coordinator-prefill.log)"
wait_for "decode coordinator announced" 600 \
    grep -q "MEGATRON_COORDINATOR_ADDR=" "$LOG_DIR/coordinator-decode.log" \
    || die "decode coordinator never announced (see $LOG_DIR/coordinator-decode.log)"

PREFILL_COORD_ADDR=$(grep -m1 -oP 'MEGATRON_COORDINATOR_ADDR=\K\S+' "$LOG_DIR/coordinator-prefill.log")
DECODE_COORD_ADDR=$(grep -m1 -oP 'MEGATRON_COORDINATOR_ADDR=\K\S+' "$LOG_DIR/coordinator-decode.log")
log "prefill coordinator at $PREFILL_COORD_ADDR"
log "decode coordinator at $DECODE_COORD_ADDR"

###############################################################################
# 3. Two Dynamo Megatron workers
###############################################################################
log "starting Dynamo PREFILL worker..."
python -m dynamo.megatron \
    --role prefill \
    --coordinator-addr "$PREFILL_COORD_ADDR" \
    --model "$TOKENIZER_MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --context-length "$CONTEXT_LENGTH" \
    > "$LOG_DIR/worker-prefill.log" 2>&1 &
PIDS+=($!)

log "starting Dynamo DECODE worker..."
python -m dynamo.megatron \
    --role decode \
    --coordinator-addr "$DECODE_COORD_ADDR" \
    --model "$TOKENIZER_MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --context-length "$CONTEXT_LENGTH" \
    > "$LOG_DIR/worker-decode.log" 2>&1 &
PIDS+=($!)

# Both workers are "ready" when their _core binding registers the model card.
wait_for "prefill worker registered" 300 \
    grep -q "Registered base model" "$LOG_DIR/worker-prefill.log" \
    || die "prefill worker never registered (see $LOG_DIR/worker-prefill.log)"
wait_for "decode worker registered" 300 \
    grep -q "Registered base model" "$LOG_DIR/worker-decode.log" \
    || die "decode worker never registered (see $LOG_DIR/worker-decode.log)"

###############################################################################
# 4. Dynamo frontend
###############################################################################
log "starting Dynamo frontend on :$HTTP_PORT..."
python -m dynamo.frontend \
    --http-port "$HTTP_PORT" \
    --request-plane nats \
    --event-plane nats \
    > "$LOG_DIR/frontend.log" 2>&1 &
PIDS+=($!)

wait_for "frontend exposes $SERVED_MODEL_NAME" 60 \
    bash -c "curl -sf http://127.0.0.1:$HTTP_PORT/v1/models | grep -q '$SERVED_MODEL_NAME'" \
    || die "frontend never exposed model (see $LOG_DIR/frontend.log)"

###############################################################################
# 5. Publish env + block
###############################################################################
cat > /tmp/phase3.env <<ENV
export NATS_SERVER="$NATS_SERVER"
export ETCD_ENDPOINTS="$ETCD_ENDPOINTS"
export MEGATRON_PREFILL_COORDINATOR_ADDR="$PREFILL_COORD_ADDR"
export MEGATRON_DECODE_COORDINATOR_ADDR="$DECODE_COORD_ADDR"
export PHASE3_FRONTEND_URL="http://127.0.0.1:$HTTP_PORT"
export PHASE3_MODEL_NAME="$SERVED_MODEL_NAME"
export PHASE3_LOG_DIR="$LOG_DIR"
export PHASE3_PREFILL_LOG="$LOG_DIR/coordinator-prefill.log"
export PHASE3_DECODE_LOG="$LOG_DIR/coordinator-decode.log"
ENV

log "all components healthy. test endpoint: http://127.0.0.1:$HTTP_PORT"
echo "PHASE3_READY"

wait -n "${PIDS[@]}"
log "a component exited; tearing down"
exit 1
