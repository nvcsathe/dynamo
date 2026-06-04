# Phase-3 — Minimal disaggregated serving end-to-end

This directory mirrors `phase0/`, but brings up the full disagg topology and
adds the one assertion that actually proves disagg is working: greppable
markers in the prefill + decode engine logs.

## Building the container

Phase-3 needs NIXL in the image. The Megatron Dockerfile template was
updated to install the `nixl` PyPI wheel and wire `LD_LIBRARY_PATH` to the
auditwheel-bundled `libnixl.so` directory. Rebuild before the first run:

```bash
# From the dynamo repo root (host with docker + adequate disk).
cd /path/to/dynamo

# 1. Render the Megatron Dockerfile, pointing MEGATRON_REF at your fork's
#    Phase-3 branch so the engine code with --disagg-role / --kv-transfer-listen-addr
#    is the one that gets git clone'd into /opt/megatron-lm.
container/render.py \
    --framework megatron \
    --target runtime \
    --output-short-filename

# 2. Build. The MEGATRON_REPO + MEGATRON_REF overrides only matter if you do
#    NOT plan to mount a live Megatron-LM checkout via MEGATRON_LOCAL_DEV at
#    launch time; if you do, the baked clone is irrelevant.
docker build \
    -f container/rendered.Dockerfile \
    --build-arg MEGATRON_REPO=https://github.com/<your-fork>/Megatron-LM.git \
    --build-arg MEGATRON_REF=feature/dynamo-disagg-phase3 \
    -t dynamo:phase3-megatron-runtime \
    .

# 3. Convert to sqsh for pyxis on the cluster.
enroot import -o dynamo-megatron+phase3-arm64.sqsh \
    dockerd://dynamo:phase3-megatron-runtime

# 4. Push to lustre and update DMG_SQSH in your launch invocation.
rsync -av --progress dynamo-megatron+phase3-arm64.sqsh \
    csathe@cluster:/lustre/.../csathe/
```

If you only edited the Megatron *Python* code (not the requirements or the
Dockerfile), you do **not** need to rebuild — bind-mount the host checkout
via `MEGATRON_LOCAL_DEV` (see [Run](#run)).

### Verifying NIXL landed in the image

After the build but before pushing to the cluster:

```bash
docker run --rm dynamo:phase3-megatron-runtime python -c \
    "from nixl._api import nixl_agent; print('nixl OK')"
```

And confirm the native lib + UCX paths landed:

```bash
docker run --rm dynamo:phase3-megatron-runtime sh -c \
    'ls /opt/nvidia/nvda_nixl/lib64/libnixl* && ls /usr/local/ucx/lib/libucp* && echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"'
```

Expect `libnixl.so*` under `/opt/nvidia/nvda_nixl/lib64/`, `libucp.so*` under
`/usr/local/ucx/lib/`, and `LD_LIBRARY_PATH` containing both prefixes.

## Topology

```
                              ┌── Megatron PREFILL engine (TP=N, GPUs 0..N-1)
                              │     - role=prefill, NIXL :7000
                              │
nats + etcd ── dynamo.frontend
                              │
                              └── Megatron DECODE engine  (TP=N, GPUs N..2N-1)
                                    - role=decode, NIXL :7001
                                    - prefix caching ENABLED
```

A request flows: frontend → `PrefillRouter` → prefill worker → prefill engine
runs 1-token prefill, pins KV blocks, returns `disaggregated_params` → frontend
forwards `prefill_result` → decode worker → decode engine NIXL-pulls the KV
blocks, registers their hashes, then `add_request` finds them as
prefix-matched → only 1 token of prefill on the decode side → tokens stream
back.

## Run

Inside the container, after building / staging the model checkpoint:

```bash
# 1. Bring up the stack (blocks; tail logs in another pane).
bash orchestrate.sh

# 2. In another pane:
bash test_phase3.sh
# or
pytest -q test_phase3.py
```

By default both engines run TP=1 → 2 GPUs total. Set `TP_PREFILL=$N
TP_DECODE=$N` to scale. They MUST match (the NIXL transfer pairs rank N to
rank N).

## Configuration knobs

| Var | Default | Notes |
| --- | --- | --- |
| `MODEL_CHECKPOINT` | `$STAGE/models/llama3.1-8b-instruct-mcore` | mcore-format checkpoint |
| `TOKENIZER_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | HF id for tokenizer (Instruct variant — base has no `chat_template`, see `[[project_phase0_stack_quirks]]`) |
| `TP_PREFILL` / `TP_DECODE` | `1` | Must match. PP=1 hard-coded. |
| `COORD_PORT_PREFILL` / `COORD_PORT_DECODE` | `5555` / `5556` | Coordinator ZMQ. |
| `NIXL_PORT_PREFILL` / `NIXL_PORT_DECODE` | `7000` / `7001` | NIXL listen sockets — must be distinct. |
| `HTTP_PORT` | `8100` | Frontend. |

## What the test asserts

1. `/v1/models` lists the served model.
2. A streaming chat completion returns deltas and finishes. (Tokens coming
   back is strong evidence the whole pipeline works — broken NIXL or broken
   KV import would produce garbage or crash, not a coherent stream.)
3. `DISAGG_PREFILL_HANDOFF request_id=... pinned_blocks=N` is present in
   `coordinator-prefill.log`. Proves the prefill engine pinned blocks and
   emitted `disaggregated_params`.
4. `DISAGG_DECODE_IMPORT request_id=... prompt_tokens=... imported_blocks=N
   hashes_registered=M` is present in `coordinator-decode.log`. Proves the
   decode engine took the NIXL-import path **and registered hashes so the
   prefix-cache match could skip prefill**. `hashes_registered > 0` is the
   load-bearing assertion — without it, NIXL might transfer but prefill
   would still re-run.

If only (1) and (2) pass but (3) or (4) fail, the deployment is using the
Phase-0 aggregated path even though the workers report `--role prefill` /
`--role decode`. Most likely cause: the frontend's `PrefillRouter` didn't
activate — check that both workers registered under their expected NATS
subjects (`dynamo.prefill.generate` and `dynamo.backend.generate`).

## Known limitations

- **One node only.** Multi-node will work in principle (NIXL handles RDMA);
  the orchestrator just doesn't lay it out. Extend with srun for that.
- **TP/PP must match.** PP=1 is hard-coded; lifting this requires per-layer
  NIXL routing on the decode side.
- **Single DP per role.** Multi-DP-aware routing of the handoff payload is
  a follow-up.
- **The "partial-block-at-tail" case still does ~1 block of prefill work
  on decode.** Block hashes only cover whole blocks, so any tokens beyond
  the last whole-block boundary go through the normal prefill path. For
  block_size=64 and a 200-token prompt, that's ~8 tokens of redundant
  compute — acceptable.

## What's NOT yet here (follow-ups)

- Phase-1 (per-step metrics PUB) and Phase-2 (KV-event PUB) — stashed under
  `disagg/.stashed_phase1_phase2/`. Independent of Phase-3 and can be
  re-applied any time.
- Block-release on decode failure: the handler emits `RELEASE_KV` only after
  a successful stream completion. A torn TCP connection mid-decode leaves
  prefill blocks pinned until engine restart. See the `release_handoff_blocks`
  TODO in `engines/dynamic_engine.py`.
