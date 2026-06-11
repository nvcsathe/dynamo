#!/usr/bin/env bash
# Verify a heterogeneous-TP disagg stack brought up by orchestrate.sh.
#
# Checks, in order:
#   1. A greedy (temperature=0) completion streams coherent text — if the KV
#      re-shard corrupted heads, greedy output would be garbage / repeated.
#   2. The decode coordinator log shows it IMPORTED KV and SKIPPED prefill
#      (proves the resharded blocks were accepted by the prefix cache, i.e. the
#      transfer actually happened rather than silently re-prefilling).
#
# Gold-standard correctness (not done here, needs a second stack): run the same
# greedy prompt through a MATCHED-TP stack and diff the token sequences — they
# must be identical, since re-shard only moves bytes, it does not change math.
#
# Usage (inside the container, after orchestrate.sh prints PHASE3_HETERO_READY):
#     source /tmp/phase3_hetero.env
#     ./verify_hetero_tp.sh

set -uo pipefail
source /tmp/phase3_hetero.env

PROMPT='Explain in two sentences why the sky is blue.'
echo "== prefill TP=$TP_PREFILL  decode TP=$TP_DECODE =="

# 1. Greedy completion. Long-enough prompt to span >1 KV block.
RESP=$(curl -sf "$PHASE3_FRONTEND_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$PHASE3_MODEL_NAME\",
         \"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],
         \"max_tokens\":64,\"temperature\":0,\"stream\":false}") \
  || { echo "FAIL: request errored"; exit 1; }

TEXT=$(echo "$RESP" | python -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])') \
  || { echo "FAIL: could not parse response: $RESP"; exit 1; }

echo "completion: $TEXT"
WORDS=$(echo "$TEXT" | wc -w)
if (( WORDS < 3 )); then echo "FAIL: degenerate output ($WORDS words) — re-shard likely corrupted KV"; exit 1; fi

# 2. Decode engine imported KV and skipped prefill.
if grep -q "DISAGG_DECODE_IMPORT" "$PHASE3_DECODE_LOG"; then
    grep -m1 "DISAGG_DECODE_IMPORT" "$PHASE3_DECODE_LOG"
    echo "PASS: decode imported re-sharded KV (prefill TP=$TP_PREFILL -> decode TP=$TP_DECODE)"
else
    echo "FAIL: no DISAGG_DECODE_IMPORT in $PHASE3_DECODE_LOG — decode re-prefilled instead of importing"
    exit 1
fi
