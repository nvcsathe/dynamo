#!/usr/bin/env bash
# Phase-3 host wrapper. Run from inside an existing salloc'd shell on the
# login node. srun's into the dynamo-megatron container (enroot/pyxis) and
# invokes orchestrate.sh, which brings up the two-engine (prefill + decode)
# disagg topology.
#
# Required env:
#   SLURM_JOB_ID   set automatically by salloc
#   DMG_SQSH       absolute path to the dynamo-megatron container sqsh.
#                  The container must have `nixl` Python + native runtime
#                  importable — confirm with:
#                      python -c "from nixl._api import nixl_agent"
#                  inside the container before relying on this launcher.
#   STAGE          lustre staging dir holding the model checkpoint + hf-cache
#
# Optional env (passed through to orchestrate.sh):
#   MODEL_CHECKPOINT, TOKENIZER_MODEL, SERVED_MODEL_NAME, CONTEXT_LENGTH
#   TP_PREFILL, TP_DECODE                          (must be equal, default 1)
#   HTTP_PORT
#   COORD_PORT_PREFILL, COORD_PORT_DECODE          (default 5555, 5556)
#   NIXL_PORT_PREFILL, NIXL_PORT_DECODE            (default 7000, 7001)
#   MASTER_PORT_PREFILL, MASTER_PORT_DECODE        (default 29500, 29501)
#   MEGATRON_LOCAL_DEV
#
# UCX transport / logging (force-set in this script, use _OVERRIDE suffix to change):
#   UCX_TLS_OVERRIDE            transport allow-list (default: cuda_ipc,cuda_copy,cma,shm,self)
#                               TCP is intentionally excluded — it cannot handle VRAM addresses.
#                               The NGC base image bakes in UCX_TLS=tcp; this script overrides it.
#   UCX_MEMTYPE_CACHE_OVERRIDE  set to "n" by default to avoid early-init misclassification
#   UCX_LOG_LEVEL_OVERRIDE      UCX log verbosity (default: info — set to warn once stable)
#   UCX_LOG_FILE_OVERRIDE       log path; UCX expands %p to PID (default: /tmp/ucx_%p.log)
#
# If MEGATRON_LOCAL_DEV is set, the host directory it points to is mounted
# over /opt/megatron-lm so you can edit InferenceClient / coordinator /
# kv_transfer.py code without rebuilding the image.

set -euo pipefail

: "${SLURM_JOB_ID:?must run inside a salloc allocation}"
: "${DMG_SQSH:?DMG_SQSH must be set (path to the dynamo-megatron sqsh on lustre)}"
: "${STAGE:?STAGE must be set (lustre staging dir)}"

[[ -f "$DMG_SQSH" ]] || { echo "DMG_SQSH not found: $DMG_SQSH" >&2; exit 1; }
[[ -d "$STAGE"   ]] || { echo "STAGE not found: $STAGE"        >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default to the dynamo checkout this launcher belongs to. Override with
# DYNAMO_LOCAL_DEV=/some/other/path if you want to mount a different
# checkout (e.g. one that has Phase-3 dynamo code while this launcher
# was vendored from somewhere older).
DYNAMO_ROOT="${DYNAMO_LOCAL_DEV:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
[[ -d "$DYNAMO_ROOT" ]] || { echo "DYNAMO_ROOT not a dir: $DYNAMO_ROOT" >&2; exit 1; }
echo "[launch] dynamo source mount: $DYNAMO_ROOT -> /workspace"

MOUNTS="$STAGE:$STAGE,$DYNAMO_ROOT:/workspace"
if [[ -n "${MEGATRON_LOCAL_DEV:-}" ]]; then
    [[ -d "$MEGATRON_LOCAL_DEV" ]] || { echo "MEGATRON_LOCAL_DEV not a dir: $MEGATRON_LOCAL_DEV" >&2; exit 1; }
    MOUNTS="$MOUNTS,$MEGATRON_LOCAL_DEV:/opt/megatron-lm"
    echo "[launch] live Megatron mount: $MEGATRON_LOCAL_DEV -> /opt/megatron-lm"
fi

# Forward env that orchestrate.sh consumes. HF_TOKEN is needed for gated
# repos like meta-llama/Llama-3.1-8B.
EXPORT_VARS="STAGE,HF_HOME,HF_TOKEN,MODEL_CHECKPOINT,MODEL_DIR,TOKENIZER_MODEL,SERVED_MODEL_NAME"
EXPORT_VARS="$EXPORT_VARS,CONTEXT_LENGTH,TP_PREFILL,TP_DECODE,HTTP_PORT"
EXPORT_VARS="$EXPORT_VARS,COORD_PORT_PREFILL,COORD_PORT_DECODE"
EXPORT_VARS="$EXPORT_VARS,NIXL_PORT_PREFILL,NIXL_PORT_DECODE"
EXPORT_VARS="$EXPORT_VARS,MASTER_PORT_PREFILL,MASTER_PORT_DECODE"

# UCX transport constraints and diagnostic logging.
#
# The NGC base image bakes in UCX_TLS=tcp to keep NCCL off UCX-RDMA paths.
# That setting is fatal for NIXL: uct_tcp_ep_am_bcopy on the prefill side
# tries to CPU-memcpy from VRAM addresses when handling decode's GET request
# → SIGSEGV on aarch64. We must force-override it unconditionally (no :=
# setdefault — that silently skips if the var is already set to "tcp").
#
# Passing the override both as an exported shell var and via srun --env
# ensures pyxis applies it on top of the image ENV layer.
#
# UCX_LOG_LEVEL=info + UCX_LOG_FILE captures which transport UCX selects.
# Grep the log after a run: grep -iE 'tls|cuda|tcp|selected' /tmp/ucx_*.log
# Set UCX_LOG_LEVEL=warn once cuda_icp is confirmed as the chosen transport.
UCX_TLS="${UCX_TLS_OVERRIDE:-cuda_ipc,cuda_copy,tcp,shm,cma,self}"
UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE_OVERRIDE:-n}"
UCX_LOG_LEVEL="${UCX_LOG_LEVEL_OVERRIDE:-info}"
UCX_LOG_FILE="${UCX_LOG_FILE_OVERRIDE:-/tmp/ucx_%p.log}"
export UCX_TLS UCX_MEMTYPE_CACHE UCX_LOG_LEVEL UCX_LOG_FILE

echo "[launch] container: $DMG_SQSH"
echo "[launch] mounts:    $MOUNTS"
echo "[launch] UCX_TLS=$UCX_TLS  UCX_LOG_FILE=$UCX_LOG_FILE"
echo "[launch] expect 'PHASE3_READY' on stdout when ready"
echo

exec srun \
    --jobid="$SLURM_JOB_ID" --overlap \
    --container-image="$DMG_SQSH" \
    --container-name=dmg \
    --container-mounts="$MOUNTS" \
    --container-workdir=/workspace \
    --export="ALL,$EXPORT_VARS,UCX_TLS=$UCX_TLS,UCX_MEMTYPE_CACHE=$UCX_MEMTYPE_CACHE,UCX_LOG_LEVEL=$UCX_LOG_LEVEL,UCX_LOG_FILE=$UCX_LOG_FILE" \
    bash /workspace/examples/backends/megatron/phase3/orchestrate.sh
