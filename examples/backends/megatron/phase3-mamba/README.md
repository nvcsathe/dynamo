# Phase-3 hybrid disagg — Mamba state transfer (Nemotron-3 Nano)

Disaggregated prefill→decode for **hybrid (Mamba-2 + attention) models**. Hybrid
models keep two kinds of per-request state:

| state | where it lives | transferred by |
| --- | --- | --- |
| attention KV cache | block-indexed `memory_buffer` | `KvTransferAgent` (already) |
| Mamba conv/ssm state | block-indexed slot pool in `MambaSlotAllocator` | **new** conv/ssm `KvTransferAgent`s |

Before this change, only attention KV was handed off; the decode engine restored
Mamba state from its own (empty) cache, fell back to **zeroing** it, and decoded
hybrid models from zero recurrent state — silently wrong output. This example
transfers the Mamba conv/ssm state over the same NIXL path and verifies it.

## What changed (core)

`Megatron-LM/megatron/core/inference/engines/dynamic_engine.py`:

- `setup_kv_transfer` — for hybrid models, brings up two extra NIXL agents over
  the Mamba cache's `conv_states` / `ssm_states` slot pools (layout
  `[num_mamba_layers, max_slots, …]`, which `KvTransferAgent` already understands
  with the slot as the "block"). Guarded to **matched TP=1/PP=1** — conv/ssm
  state has no TP/PP reshard plan yet, so anything else fails loudly at launch.
- `_capture_handoff_meta` (prefill) — nests a `"mamba"` entry inside `kv_meta`
  listing, per handoff block with a committed snapshot, its `[position, slot]`.
  Rides the existing handoff transport verbatim (no coordinator/transport change).
- `add_request_with_kv_handoff` + `_import_mamba_handoff` (decode) — pulls the
  conv/ssm slots over NIXL into freshly allocated local slots, then registers
  them in `block_to_slot` + `hash_to_block_id` so the **existing** Mamba
  prefix-cache restore path takes over (`_find_mamba_match_count` →
  `_compute_prefix_match` sets `prefix_skip_tokens` to the matched Mamba block
  boundary → `restore_to_live`). The short tail past that boundary is
  re-prefilled, advancing the recurrence to the exact handoff position — no
  off-by-one, no zeroed state.

Emits `DISAGG_DECODE_MAMBA_IMPORT request_id=… mamba_blocks=…` on the decode
coordinator log.

## Run

From an `salloc`'d shell on the login node, use the wrapper — it srun's into the
container and binds the checkpoint paths for you:

```bash
export DMG_SQSH=/lustre/.../dynamo-megatron.sqsh   # container with nixl
export STAGE=/lustre/fsw/portfolios/nemotron/users/csathe
bash launch_mamba.sh           # prints PHASE3_MAMBA_READY
```

Or, if you're already inside the container with the checkpoint mounted:

```bash
# node with >=3 GPUs (prefill, decode, baseline)
bash orchestrate.sh            # prints PHASE3_MAMBA_READY
```

Then, in a second shell inside the same container:

```bash
source /tmp/phase3_mamba.env
bash verify_mamba.sh
```

### Mounting the checkpoint

The Nano artifacts live under `/lustre/fsw/portfolios/llmservice/...`, a
different portfolio than `$STAGE`, so a single `$STAGE` bind doesn't reach them.
`launch_mamba.sh` binds each artifact's directory into the container at the
**same absolute path** (read-only) via pyxis `--container-mounts`, so
orchestrate.sh's `--load` / `--pretrained-checkpoint` / `--tokenizer-model`
resolve unchanged. It dedupes shared parent dirs (ckpt + pretrained under the
same `users/` dir mount once). Point at your own copy by exporting
`MODEL_CHECKPOINT` / `PRETRAINED_CHECKPOINT` / `TOKENIZER_MODEL`, or add extra
binds with `EXTRA_MOUNTS="src:dst,src2:dst2"`.

`orchestrate.sh` defaults `MODEL_CHECKPOINT`, `PRETRAINED_CHECKPOINT` and
`TOKENIZER_MODEL` to the cluster-staged Nemotron-3 Nano v3 artifacts (the same
ones the Nano v3 functional test uses), and its `MODEL_ARGS` mirror that test
(`--model-provider mamba`, MoE + `transformer-impl inference_optimized`, etc.).
Override any of them to point at your own copy. The checkpoint is consumed only
by `orchestrate.sh` (via `--load`); `verify_mamba.sh` just hits the running HTTP
endpoint and needs nothing checkpoint-related.

It brings up the disagg stack (prefill GPU0 + decode GPU1) and, by default
(`WITH_BASELINE=1`), a non-disagg **aggregated** reference of the same model on
GPU2 for the gold-standard token diff. Set `WITH_BASELINE=0` to skip it (verify
then only checks the import markers, not correctness).

`verify_mamba.sh` uses `/v1/completions` (raw prompt) rather than chat
completions because the Nano tokenizer is a plain `vocab.json` with no chat
template.

EP is pinned to 1 (one rank per role). The functional test shards experts with
`--expert-model-parallel-size ${WORLD_SIZE}`; multi-rank-per-role isn't wired
for the Mamba handoff yet, so keep a single process per coordinator here.

## How verify proves correctness

`verify_mamba.sh`:

1. Confirms the decode log shows `DISAGG_DECODE_IMPORT` **and**
   `DISAGG_DECODE_MAMBA_IMPORT` with `mamba_blocks >= 1` (else the prompt was too
   short to commit a Mamba block boundary and the transfer path was untested →
   FAIL).
2. Greedy-decodes the same prompt through the disagg stack and the aggregated
   baseline and asserts the token text is **byte-identical**. Greedy decoding is
   deterministic and the handoff only moves bytes, so any divergence means the
   transferred state is wrong. An attention-only coherence check would pass even
   with zeroed Mamba state — this diff is what actually catches the bug.

## Key knobs

| env | default | meaning |
| --- | --- | --- |
| `MODEL_CHECKPOINT` | `…/ksanthanam/nemotron-3-nano-30b` | mcore checkpoint served via `--load` |
| `PRETRAINED_CHECKPOINT` | `…/ksanthanam/nanov3` | `--pretrained-checkpoint` source |
| `TOKENIZER_MODEL` | `…/multiMixV8….vocab.json` | tokenizer vocab passed to `--tokenizer-model` |
| `INFER_BUFFER_GB` | `70` | dynamic-batching KV buffer budget per engine |
| `MAMBA_GB` | `4.0` | Mamba state-cache budget (both prefill + decode) |
| `PREFIX_CACHE` | `1` | enable prefix caching (required for the handoff path) |
| `WITH_BASELINE` | `1` | also launch the aggregated reference for the token diff |
| `GPU_PREFILL`/`GPU_DECODE`/`GPU_BASELINE` | `0`/`1`/`2` | GPU assignment |
| `MODEL_ARGS_OVERRIDE` | _(unset)_ | replace the entire Nemotron arg block |

## Scope / limitations

- **Matched TP=1 / PP=1 only.** Mamba conv/ssm state is sharded across TP (head
  group) and PP (layer); unlike attention KV heads there is no reshard plan for
  it yet. Hybrid + (TP>1 or PP>1) with KV transfer raises `NotImplementedError`
  at launch rather than producing wrong output. Extending to heterogeneous
  TP/PP requires a conv/ssm reshard plan analogous to `kv_reshard_plan.py`.
- Both prefill and decode must run with `--inference-dynamic-batching-prefix-caching`
  and `--inference-dynamic-batching-prefix-caching-mamba-gb`: prefill is the
  source of the block-boundary snapshots, decode restores them.
