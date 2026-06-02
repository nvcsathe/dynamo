# Megatron + Dynamo — Phase 0

Brings up an 8B Megatron model behind a Dynamo worker on a single node with
TP=2. Aggregated decode, token streaming. No KV events, no metrics, no
disaggregated prefill / decode. Those land in later phases.

Everything runs inside the `chaitrasathe/dynamo-megatron:phase0-arm64`
container under enroot/pyxis. There is no bare-metal path.

## Files in this directory

| File | Role |
|---|---|
| `launch_phase0.sh` | Host wrapper. `srun`s into the container and invokes `orchestrate.sh`. |
| `orchestrate.sh`   | Inside-container orchestrator. Brings up NATS + etcd + coordinator + worker + frontend, waits for health, writes `/tmp/phase0.env`, blocks until Ctrl-C. |
| `test_phase0.sh`   | Curl smoke test against an already-up stack. |
| `test_phase0.py`   | Pytest functional test against an already-up stack. |

## What's in scope

Five processes, all running inside one container:

1. **NATS + etcd** — baked into the image, started first by `orchestrate.sh`.
2. **Megatron coordinator + engine (rank 0..1):** `torchrun` launches
   `tools/run_dynamic_text_generation_server.py --frontend dynamo`. Binds a
   ZMQ ROUTER on `:5555` and prints
   `MEGATRON_COORDINATOR_ADDR=tcp://<host>:5555` once ready.
3. **Dynamo Megatron worker:** `python -m dynamo.megatron` connects to the
   coordinator, registers the model on etcd, serves `dynamo.backend.generate`.
4. **Dynamo frontend:** `python -m dynamo.frontend` — OpenAI-compatible HTTP
   on `:8080`.

## Prerequisites

- The container image (`chaitrasathe/dynamo-megatron:phase0-arm64`) built and
  imported under enroot — see "Building the image" below.
- 8B Megatron checkpoint (e.g. Llama-3.1-8B converted via Megatron-Bridge) on
  lustre at `$STAGE/models/Llama-3.1-8B-mcore`. See conversion notes in the
  parent `README.md` of `examples/backends/megatron/`.
- HuggingFace tokenizer matching the Megatron tokenizer used at training.
  The Dynamo frontend tokenizes with this — must match what Megatron was
  trained with, or generation will be garbage.

## Building the image

Build on a Docker-capable host (typically the login node), push to a
registry, import to enroot on the cluster.

```bash
# 1. Render the Dockerfile (Megatron's pin: pytorch:26.04-py3 on cuda12.9).
cd container
python render.py --framework megatron --target runtime \
    --platform linux/arm64 --cuda-version 12.9 \
    --output-short-filename

# 2. Build and push.
docker buildx build --platform linux/arm64 \
    --build-arg MEGATRON_REPO=https://github.com/<fork>/Megatron-LM.git \
    --build-arg MEGATRON_REF=dynamo-integration \
    -t <user>/dynamo-megatron:phase0-arm64 \
    --push \
    -f rendered.Dockerfile ..

# 3. On the cluster (login node): import to a lustre-resident squashfs.
cd $STAGE
enroot import docker://<user>/dynamo-megatron:phase0-arm64
# Produces: <user>+dynamo-megatron+phase0-arm64.sqsh
```

`enroot import` reads `~/.config/enroot/.credentials` for private-image
pulls. The expected format:

```
machine auth.docker.io login <dockerhub-user> password <PAT>
machine registry-1.docker.io login <dockerhub-user> password <PAT>
```

Both endpoints are required. If a fresh `srun` on a new node 401s when
fetching the image, your credentials file isn't on shared storage and that
node hasn't seen it — put it on lustre and set `ENROOT_CONFIG_PATH` to
point at the directory holding it.

## One-time per-session setup

```bash
export DMG_SQSH=/lustre/fsw/portfolios/nemotron/users/csathe/chaitrasathe+dynamo-megatron+phase0-arm64.sqsh
export STAGE=/lustre/fsw/portfolios/nemotron/users/csathe
mkdir -p $STAGE/hf-cache    # first time only
```

## Bring it up

```bash
salloc --nodes=1 --gpus-per-node=2 --time=02:00:00 --account=<a> --partition=<p>
tmux

# Pane A — orchestrator (blocks until Ctrl-C)
bash examples/backends/megatron/phase0/launch_phase0.sh
# Wait for "PHASE0_READY" on stdout.

# Pane B — run tests against the running stack
srun --jobid=$SLURM_JOB_ID --overlap --container-name=dmg --pty bash
bash /workspace/examples/backends/megatron/phase0/test_phase0.sh
# or:
pytest -q /workspace/examples/backends/megatron/phase0/test_phase0.py
```

Optional knobs (env vars consumed by `orchestrate.sh`):

| Var | Default | What it does |
|---|---|---|
| `TP` | `2` | Tensor-parallel size for the coordinator. |
| `MODEL_CHECKPOINT` | `$STAGE/models/Llama-3.1-8B-mcore` | Megatron checkpoint path. |
| `TOKENIZER_MODEL` | `meta-llama/Llama-3.1-8B` | HF id for both Megatron's `--tokenizer-model` and Dynamo's `--model`. |
| `SERVED_MODEL_NAME` | `llama-3.1-8b` | Name advertised to clients. |
| `CONTEXT_LENGTH` | `4096` | Advertised on the model card. |
| `HTTP_PORT` | `8080` | Frontend bind port. |
| `COORD_PORT` | `5555` | Coordinator ROUTER bind port. |
| `MEGATRON_LOCAL_DEV` | unset | Path to a host Megatron-LM checkout. If set, the wrapper bind-mounts it over `/opt/megatron-lm` for live editing without rebuilding. |

For fast iteration on the streaming protocol, swap to **Llama-3.2-1B**,
`TP=1`:

```bash
TP=1 \
MODEL_CHECKPOINT=$STAGE/models/Llama-3.2-1B-mcore \
TOKENIZER_MODEL=meta-llama/Llama-3.2-1B \
SERVED_MODEL_NAME=llama-3.2-1b \
  bash examples/backends/megatron/phase0/launch_phase0.sh
```

The integration code is identical at 1B and 8B; restart cost drops from
~30–60 s to ~5 s.

## Smoke test

Once `PHASE0_READY` appears, from any pane attached to the container:

```bash
curl -s http://127.0.0.1:8080/v1/models | jq .

curl -N http://127.0.0.1:8080/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"model":"llama-3.1-8b",
         "messages":[{"role":"user","content":"Hello in one short sentence."}],
         "stream":true,"max_tokens":64}'
```

Expected: token-by-token SSE chunks. The Megatron engine emits one
`ENGINE_REPLY_PARTIAL` per generation step; the Dynamo worker translates each
into a streamed `{"token_ids": [...]}` chunk; the frontend detokenizes and
streams `delta.content` back to curl.

## Debugging

The orchestrator tees every component to a log file in the container's
`/tmp/`. Tail any of these from an attached pane:

| Log | Component |
|---|---|
| `/tmp/nats.log` | NATS server |
| `/tmp/etcd.log` | etcd |
| `/tmp/coordinator.log` | Megatron coordinator (torchrun stdout/stderr, including `MEGATRON_COORDINATOR_ADDR=...`) |
| `/tmp/worker.log` | `dynamo.megatron` |
| `/tmp/frontend.log` | `dynamo.frontend` |

If the orchestrator dies with `FATAL: <something>`, the named log has the
underlying error.

### Restarting a single component without re-loading the model

The expensive process is the coordinator (model load). To restart only the
worker or the frontend without dropping the model:

```bash
# In any attached pane:
pkill -f 'python -m dynamo.megatron'                  # or dynamo.frontend
# Re-run just that line from orchestrate.sh by hand:
python -m dynamo.megatron --coordinator-addr "$MEGATRON_COORDINATOR_ADDR" \
    --model "$PHASE0_MODEL_NAME" ... 2>&1 | tee /tmp/worker.log
```

`/tmp/phase0.env` was written by the orchestrator with all the right
addresses, so `source /tmp/phase0.env` first.

### Live Megatron-LM iteration

The image clones Megatron at build time. To edit `inference_client.py`,
`headers.py`, or the coordinator without rebuilding:

```bash
MEGATRON_LOCAL_DEV=$STAGE/Megatron-LM \
  bash examples/backends/megatron/phase0/launch_phase0.sh
```

The host checkout shadows `/opt/megatron-lm` inside the container; restart
the coordinator (or the whole stack) to pick up edits.

## Common gotchas

- **Coordinator address never appears in `/tmp/coordinator.log`.** Tokenizer
  download failure (check `HF_HOME` reachability), `--tensor-model-parallel-size`
  divisibility against the checkpoint, or OOM during model load.
- **Worker exits immediately.** Most often the `--coordinator-addr` was
  captured before the coordinator finished binding (the orchestrator's
  health checks should prevent this, but if you restarted the worker by
  hand, re-grep `/tmp/coordinator.log` for the latest address).
- **Streaming stalls after first token.** Check `/tmp/coordinator.log` for
  engine errors. Most stalls come from a request being submitted without
  `sampling_params.streaming=True` — the handler always sets this, so it
  should not happen in Phase 0, but it is the first place to look.
- **Port collisions on rerun.** If you Ctrl-C the orchestrator and
  immediately re-run, `:5555` (coordinator) or `:8080` (frontend) may be in
  TIME_WAIT for ~30 s. Either wait, or set `COORD_PORT` / `HTTP_PORT` to
  different values for the retry.
- **Stale model in etcd after a hard kill.** If the worker dies without
  graceful shutdown its etcd entry lingers until the lease expires (~30 s).
  The frontend may keep advertising a model that no longer answers. If a
  curl hangs after a worker restart, wait the lease out or
  `etcdctl get --prefix /dynamo` to inspect.
- **Tokenizer mismatch is silent.** If `TOKENIZER_MODEL` doesn't match the
  HF id Megatron was trained with, generation will be garbage but nothing
  will error. The orchestrator passes the same value to both Megatron's
  `--tokenizer-model` and the Dynamo worker's `--model`, so if you override
  one, override both.
- **`HF_HOME` on lustre is mandatory for new model ids.** Without it, the
  Megatron coordinator and the frontend download tokenizer files into
  ephemeral container storage. Cold-start downloads can be slow on lustre —
  pre-populate by running `huggingface-cli download meta-llama/Llama-3.1-8B`
  on the login node with `HF_HOME=$STAGE/hf-cache` set.
- **`enroot import` 401s.** See the credentials file format in "Building
  the image" above.
