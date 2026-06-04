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
DYNAMO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

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

echo "[launch] container: $DMG_SQSH"
echo "[launch] mounts:    $MOUNTS"
echo "[launch] expect 'PHASE3_READY' on stdout when ready"
echo

exec srun \
    --jobid="$SLURM_JOB_ID" --overlap \
    --container-image="$DMG_SQSH" \
    --container-name=dmg \
    --container-mounts="$MOUNTS" \
    --container-workdir=/workspace \
    --export="ALL,$EXPORT_VARS" \
    bash /workspace/examples/backends/megatron/phase3/orchestrate.sh
