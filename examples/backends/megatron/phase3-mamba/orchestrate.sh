#!/usr/bin/env bash
# Hybrid (Mamba) disaggregation orchestrator — Nemotron-3 Nano.
#
# Validates that the Mamba conv/ssm recurrent state is handed off from prefill
# to decode over NIXL alongside the attention KV cache. Without that transfer a
# hybrid model silently decodes from ZERO Mamba state and produces wrong tokens
# (see kv_transfer.py + dynamic_engine._import_mamba_handoff).
#
#     NATS + etcd
#     ├── Megatron coordinator [PREFILL]   (TP=1 PP=1, GPU 0)  --disagg-role prefill
#     ├── Megatron coordinator [DECODE]    (TP=1 PP=1, GPU 1)  --disagg-role decode
#     ├── dynamo.megatron worker [PREFILL] --role prefill
#     ├── dynamo.megatron worker [DECODE]  --role decode
#     ├── dynamo.frontend                  (disagg stack, :$HTTP_PORT)
#     └── [optional baseline, WITH_BASELINE=1]
#         ├── Megatron coordinator [AGG]   (TP=1 PP=1, GPU 2)  --disagg-role aggregated
#         ├── dynamo.megatron worker [AGG] --role aggregated
#         └── dynamo.frontend              (baseline stack, :$HTTP_PORT_AGG)
#
# The baseline is a normal single-engine (non-disagg) server of the SAME model.
# verify_mamba.sh greedy-decodes the same prompt through both stacks and diffs
# the token text — the only check that reliably catches lost Mamba state, since
# attention-only checks pass even when conv/ssm state is zeroed.
#
# Mamba transfer is matched TP=1/PP=1 only (enforced in setup_kv_transfer); the
# conv/ssm state has no TP/PP reshard plan yet.
#
# Outputs: /tmp/phase3_mamba.env + per-component logs in $LOG_DIR.
# Emits "PHASE3_MAMBA_READY" on stdout once healthy.

set -uo pipefail

export UCX_TLS="${UCX_TLS_OVERRIDE:-cuda_ipc,cuda_copy,tcp,shm,cma,self}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE_OVERRIDE:-n}"
export UCX_LOG_LEVEL="${UCX_LOG_LEVEL_OVERRIDE:-info}"
export UCX_LOG_FILE="${UCX_LOG_FILE_OVERRIDE:-/tmp/ucx_%p.log}"

STAGE="${STAGE:-/lustre/fsw/portfolios/nemotron/users/csathe}"
# Nemotron-3 Nano mcore (torch_dist) checkpoint, its pretrained source, and the
# tokenizer — the exact artifacts the Nano v3 functional test consumes. These
# already live on the cluster; override only if you staged your own copy.
#   MODEL_CHECKPOINT      -> --load (the mcore dist checkpoint to serve)
#   PRETRAINED_CHECKPOINT -> --pretrained-checkpoint
#   TOKENIZER_MODEL       -> --tokenizer-model (a vocab.json, not an HF repo id)
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/nemotron-3-nano-30b}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/nanov3}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nemotron3-nano}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
# Dynamic-batching KV buffer budget (per engine). Nano v3 test uses 70.
INFER_BUFFER_GB="${INFER_BUFFER_GB:-70}"

# Mamba / prefix-cache budgets. BOTH prefill and decode need the Mamba state
# cache: prefill commits block-boundary conv/ssm snapshots into it (the source
# of the handoff) and decode restores them. Bump MAMBA_GB if you see
# "No Mamba slots available" / "No evictable Mamba cache slots".
PREFIX_CACHE="${PREFIX_CACHE:-1}"
MAMBA_GB="${MAMBA_GB:-4.0}"

# Bring up the non-disagg reference stack for the gold-standard token diff.
WITH_BASELINE="${WITH_BASELINE:-1}"

# GPUs: prefill=0, decode=1, baseline=2.
GPU_PREFILL="${GPU_PREFILL:-0}"
GPU_DECODE="${GPU_DECODE:-1}"
GPU_BASELINE="${GPU_BASELINE:-2}"

HTTP_PORT="${HTTP_PORT:-8100}"
HTTP_PORT_AGG="${HTTP_PORT_AGG:-8101}"
COORD_PORT_PREFILL="${COORD_PORT_PREFILL:-5555}"
COORD_PORT_DECODE="${COORD_PORT_DECODE:-5556}"
COORD_PORT_AGG="${COORD_PORT_AGG:-5557}"
NIXL_PORT_PREFILL="${NIXL_PORT_PREFILL:-7000}"
NIXL_PORT_DECODE="${NIXL_PORT_DECODE:-7001}"
MASTER_PORT_PREFILL="${MASTER_PORT_PREFILL:-29500}"
MASTER_PORT_DECODE="${MASTER_PORT_DECODE:-29501}"
MASTER_PORT_AGG="${MASTER_PORT_AGG:-29502}"

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
        sleep 2; elapsed=$((elapsed + 2))
        if [[ $elapsed -ge $max ]]; then log "TIMEOUT waiting for $desc after ${max}s"; return 1; fi
    done
    log "ready: $desc (${elapsed}s)"
}

log "Hybrid disagg (Mamba transfer): prefill GPU=$GPU_PREFILL, decode GPU=$GPU_DECODE; baseline=$WITH_BASELINE (GPU=$GPU_BASELINE)"

###############################################################################
# Model args — Nemotron-3 Nano (hybrid Mamba-2 + attention + MoE).
#
# Taken verbatim from the Nano v3 functional test, MINUS the flags the launch
# helper already supplies (--load, --tokenizer-model, --tensor/pipeline-model-
# parallel-size). Architecture comes from the checkpoint (--use-checkpoint-args).
#
# EP is pinned to 1: this Mamba-transfer test runs one rank per role (matched
# TP=1/PP=1, single GPU each). The functional test uses EP=${WORLD_SIZE} to
# shard experts across GPUs, but multi-rank-per-role isn't wired for the Mamba
# handoff yet, so keep a single process per coordinator here.
#
# Override the whole block with MODEL_ARGS_OVERRIDE="--foo ... --bar ...".
###############################################################################
if [[ -n "${MODEL_ARGS_OVERRIDE:-}" ]]; then
    # shellcheck disable=SC2206
    MODEL_ARGS=( $MODEL_ARGS_OVERRIDE )
else
    MODEL_ARGS=(
        --model-provider mamba
        --pretrained-checkpoint "$PRETRAINED_CHECKPOINT"
        --use-checkpoint-args
        --dist-ckpt-strictness log_unexpected
        --expert-model-parallel-size 1
        --expert-tensor-parallel-size 1
        --moe-router-score-function sigmoid
        --moe-router-enable-expert-bias
        --moe-router-topk-scaling-factor 2.5
        --moe-token-dispatcher-type alltoall
        --moe-grouped-gemm
        --moe-router-dtype fp32
        --moe-shared-expert-overlap
        --seq-length 73728
        --max-position-embeddings 73728
        --inference-max-seq-length 73728
        --transformer-impl inference_optimized
        --inference-grouped-gemm-backend vllm
        --inference-use-synchronous-zmq-collectives
        --inference-dynamic-batching-buffer-size-gb "$INFER_BUFFER_GB"
        --micro-batch-size 1
    )
fi

PREFIX_ARGS=()
if [[ "$PREFIX_CACHE" == "1" ]]; then
    PREFIX_ARGS=(
        --inference-dynamic-batching-prefix-caching
        --inference-dynamic-batching-prefix-caching-mamba-gb "$MAMBA_GB"
    )
fi

# Launch one Megatron coordinator. Args: role gpu master_port coord_port log
# [extra args...]
launch_coordinator() {
    local role="$1" gpu="$2" master_port="$3" coord_port="$4" logf="$5"; shift 5
    log "starting Megatron $role coordinator (GPU=$gpu)..."
    (
        cd /opt/megatron-lm
        CUDA_VISIBLE_DEVICES="$gpu" exec python -m torch.distributed.run \
            --nnodes=1 --nproc-per-node=1 --node-rank=0 \
            --master-addr="$MASTER_ADDR" --master-port="$master_port" \
            tools/run_dynamic_text_generation_server.py \
                --frontend dynamo --disagg-role "$role" \
                --inference-coordinator-port "$coord_port" \
                --tensor-model-parallel-size 1 \
                --pipeline-model-parallel-size 1 \
                --load "$MODEL_CHECKPOINT" \
                --tokenizer-model "$TOKENIZER_MODEL" \
                "${MODEL_ARGS[@]}" "$@"
    ) > "$logf" 2>&1 &
    PIDS+=($!)
}

await_coordinator() {
    local role="$1" logf="$2"
    wait_for "$role coordinator announced" 900 \
        grep -q "MEGATRON_COORDINATOR_ADDR=" "$logf" \
        || die "$role coordinator never announced (see $logf)"
    grep -m1 -oP 'MEGATRON_COORDINATOR_ADDR=\K\S+' "$logf"
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
# 2. Disagg coordinators (prefill + decode). Both need the Mamba state cache.
###############################################################################
launch_coordinator prefill "$GPU_PREFILL" "$MASTER_PORT_PREFILL" "$COORD_PORT_PREFILL" \
    "$LOG_DIR/coordinator-prefill.log" \
    --kv-transfer-listen-addr "0.0.0.0:$NIXL_PORT_PREFILL" \
    "${PREFIX_ARGS[@]}"

launch_coordinator decode "$GPU_DECODE" "$MASTER_PORT_DECODE" "$COORD_PORT_DECODE" \
    "$LOG_DIR/coordinator-decode.log" \
    --kv-transfer-listen-addr "0.0.0.0:$NIXL_PORT_DECODE" \
    "${PREFIX_ARGS[@]}"

PREFILL_COORD_ADDR=$(await_coordinator prefill "$LOG_DIR/coordinator-prefill.log")
DECODE_COORD_ADDR=$(await_coordinator decode "$LOG_DIR/coordinator-decode.log")
log "prefill coordinator at $PREFILL_COORD_ADDR; decode coordinator at $DECODE_COORD_ADDR"

###############################################################################
# 3. Disagg Dynamo workers + frontend
###############################################################################
log "starting Dynamo PREFILL worker..."
python -m dynamo.megatron --role prefill --coordinator-addr "$PREFILL_COORD_ADDR" \
    --model "$TOKENIZER_MODEL" --served-model-name "$SERVED_MODEL_NAME" \
    --context-length "$CONTEXT_LENGTH" > "$LOG_DIR/worker-prefill.log" 2>&1 &
PIDS+=($!)
log "starting Dynamo DECODE worker..."
python -m dynamo.megatron --role decode --coordinator-addr "$DECODE_COORD_ADDR" \
    --model "$TOKENIZER_MODEL" --served-model-name "$SERVED_MODEL_NAME" \
    --context-length "$CONTEXT_LENGTH" > "$LOG_DIR/worker-decode.log" 2>&1 &
PIDS+=($!)
wait_for "prefill worker registered" 300 \
    grep -q "Registered base model" "$LOG_DIR/worker-prefill.log" \
    || die "prefill worker never registered (see $LOG_DIR/worker-prefill.log)"
wait_for "decode worker registered" 300 \
    grep -q "Registered base model" "$LOG_DIR/worker-decode.log" \
    || die "decode worker never registered (see $LOG_DIR/worker-decode.log)"

log "starting Dynamo frontend (disagg) on :$HTTP_PORT..."
python -m dynamo.frontend --http-port "$HTTP_PORT" \
    --request-plane nats --event-plane nats > "$LOG_DIR/frontend.log" 2>&1 &
PIDS+=($!)
wait_for "frontend exposes $SERVED_MODEL_NAME" 60 \
    bash -c "curl -sf http://127.0.0.1:$HTTP_PORT/v1/models | grep -q '$SERVED_MODEL_NAME'" \
    || die "frontend never exposed model (see $LOG_DIR/frontend.log)"

###############################################################################
# 4. Optional aggregated baseline stack (own coordinator + worker + frontend).
###############################################################################
BASELINE_URL=""
if [[ "$WITH_BASELINE" == "1" ]]; then
    launch_coordinator aggregated "$GPU_BASELINE" "$MASTER_PORT_AGG" "$COORD_PORT_AGG" \
        "$LOG_DIR/coordinator-agg.log"
    AGG_COORD_ADDR=$(await_coordinator aggregated "$LOG_DIR/coordinator-agg.log")
    log "baseline coordinator at $AGG_COORD_ADDR"

    log "starting Dynamo AGG worker..."
    DYN_NAMESPACE=baseline python -m dynamo.megatron --role aggregated \
        --coordinator-addr "$AGG_COORD_ADDR" \
        --model "$TOKENIZER_MODEL" --served-model-name "$SERVED_MODEL_NAME" \
        --context-length "$CONTEXT_LENGTH" > "$LOG_DIR/worker-agg.log" 2>&1 &
    PIDS+=($!)
    wait_for "agg worker registered" 300 \
        grep -q "Registered base model" "$LOG_DIR/worker-agg.log" \
        || die "agg worker never registered (see $LOG_DIR/worker-agg.log)"

    log "starting Dynamo frontend (baseline) on :$HTTP_PORT_AGG..."
    DYN_NAMESPACE=baseline python -m dynamo.frontend --http-port "$HTTP_PORT_AGG" \
        --request-plane nats --event-plane nats > "$LOG_DIR/frontend-agg.log" 2>&1 &
    PIDS+=($!)
    wait_for "baseline frontend exposes $SERVED_MODEL_NAME" 60 \
        bash -c "curl -sf http://127.0.0.1:$HTTP_PORT_AGG/v1/models | grep -q '$SERVED_MODEL_NAME'" \
        || die "baseline frontend never exposed model (see $LOG_DIR/frontend-agg.log)"
    BASELINE_URL="http://127.0.0.1:$HTTP_PORT_AGG"
fi

###############################################################################
# 5. Publish env + block
###############################################################################
cat > /tmp/phase3_mamba.env <<ENV
export PHASE3_FRONTEND_URL="http://127.0.0.1:$HTTP_PORT"
export PHASE3_BASELINE_URL="$BASELINE_URL"
export PHASE3_MODEL_NAME="$SERVED_MODEL_NAME"
export PHASE3_LOG_DIR="$LOG_DIR"
export PHASE3_PREFILL_LOG="$LOG_DIR/coordinator-prefill.log"
export PHASE3_DECODE_LOG="$LOG_DIR/coordinator-decode.log"
ENV

log "all components healthy."
log "  disagg:   http://127.0.0.1:$HTTP_PORT"
[[ -n "$BASELINE_URL" ]] && log "  baseline: $BASELINE_URL"
echo "PHASE3_MAMBA_READY"
wait -n "${PIDS[@]}"
log "a component exited; tearing down"
exit 1
