# Megatron + Dynamo — Phase 0 Launcher

Brings up an 8B Megatron model behind a Dynamo worker on a single node with
TP=2. Aggregated decode, token streaming. No KV events, no metrics, no
disaggregated prefill / decode. Those land in later phases.

## What this gives you

Three processes:

1. **Megatron coordinator + engine (rank 0..1):** `torchrun` launches
   `tools/run_dynamic_text_generation_server.py --frontend dynamo`. The
   coordinator binds a ZMQ ROUTER on `:5555` and prints
   `MEGATRON_COORDINATOR_ADDR=tcp://<host>:5555` on stdout once ready.
2. **Dynamo Megatron worker (no GPU):** `python -m dynamo.megatron` connects
   to the coordinator address scraped from step 1, registers the model on
   etcd, and serves the `dynamo.backend.generate` endpoint.
3. **Dynamo frontend (external):** the HTTP/router process that sits in front
   of `(2)`. **Not** brought up by this script — launch it separately on a
   reachable host with the same NATS / etcd endpoints.

## Prerequisites

- Megatron-LM checkout containing the `feature/dynamo-streaming` changes
  (`ENGINE_REPLY_PARTIAL` header, `InferenceClient.add_request_streaming`,
  `tools/run_dynamic_text_generation_server.py --frontend dynamo`).
- Dynamo checkout containing this directory and `components/src/dynamo/megatron/`.
- NATS + etcd already running on hosts reachable from the compute node.
- 8B Megatron checkpoint (e.g. Llama-3.1-8B converted via Megatron tools) on a
  shared filesystem.
- HuggingFace tokenizer matching the Megatron tokenizer used at training time.
  The Dynamo frontend uses this tokenizer to convert prompts to token IDs
  before they hit the worker.
- Either: the **container image** below (recommended — one image with both
  Megatron-LM and Dynamo baked in), OR a host venv where you've already run
  `uv sync --extra training` in the Megatron worktree so its `.venv` exists
  on the shared filesystem.

## Container image (one image, both stacks baked in)

The Dynamo container build supports `--framework megatron`, mirroring the
trtllm and vllm images: NGC PyTorch base + Megatron-LM cloned at build time +
Dynamo wheels + nats/etcd binaries. From the dynamo checkout root:

```bash
cd container

# Render the Dockerfile (Megatron's pin: pytorch:26.04-py3 on cuda12.9).
python render.py --framework megatron --target runtime \
    --platform linux/arm64 --cuda-version 12.9 \
    --output-short-filename

# Build, pointing MEGATRON_REPO/MEGATRON_REF at the feature branch with the
# streaming patches. Drop --build-arg overrides once they're merged upstream.
docker buildx build \
    --platform linux/arm64 \
    --build-arg MEGATRON_REPO=https://github.com/<your-fork>/Megatron-LM.git \
    --build-arg MEGATRON_REF=feature/dynamo-streaming \
    -t dynamo-megatron:phase0 \
    -f rendered.Dockerfile \
    ..
```

For local Megatron development (changing the streaming protocol without
pushing commits), build once at any ref then mount your live worktree over
`/opt/megatron-lm` at run time:

```bash
docker run --rm -it --gpus all \
    -v /path/to/local/Megatron-LM:/opt/megatron-lm \
    -v /shared/models:/shared/models:ro \
    --network host \
    dynamo-megatron:phase0
```

The clone in the image stays as the fallback; the bind mount shadows it. Both
the coordinator and the Dynamo worker live in this image — launch them as
separate `docker run` invocations (or `srun` steps on Slurm, see below) all
pointing at `dynamo-megatron:phase0`.

## Submit

```bash
sbatch \

## Interactive bring-up (recommended for first run)

When iterating on the integration code, use `INTERACTIVE.md` instead of the
sbatch — `salloc` + tmux lets you restart the worker / frontend without
re-loading the 8B checkpoint.

## Submit

```bash
sbatch \
  --account=<acct> --partition=<partition> \
  --export=ALL,\
MEGATRON_WORKTREE=/path/to/Megatron-LM,\
DYNAMO_WORKTREE=/path/to/dynamo,\
MODEL_CHECKPOINT=/path/to/llama-3.1-8b-mcore,\
TOKENIZER_MODEL=meta-llama/Llama-3.1-8B,\
SERVED_MODEL_NAME=llama-3.1-8b,\
ETCD_ENDPOINTS=http://etcd-host:2379,\
NATS_SERVER=nats://nats-host:4222 \
  launch_phase0.slurm
```

Logs land under `logs/`:

- `logs/megatron-dynamo-phase0-<jobid>.out` — sbatch driver script.
- `logs/coordinator-<jobid>.out` — Megatron engine stdout, including the
  `MEGATRON_COORDINATOR_ADDR=...` line we grep for.
- `logs/dynamo-worker-<jobid>.out` — Dynamo worker stdout.

## Smoke test

Once the worker registers (look for the `register_model` success line in
`dynamo-worker-<jobid>.out`), hit the frontend:

```bash
curl -N http://<frontend-host>/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
        "model": "llama-3.1-8b",
        "messages": [{"role":"user","content":"Hello in one short sentence."}],
        "stream": true,
        "max_tokens": 64
      }'
```

Expected: token-by-token SSE chunks. The Megatron engine emits one
`ENGINE_REPLY_PARTIAL` per generation step; the Dynamo worker translates each
into a streamed `{"token_ids": [...]}` chunk; the frontend detokenizes and
streams `delta.content` back to curl.

## Known limitations (Phase 0)

- **No KV-aware routing.** The Dynamo worker does not publish KV events yet,
  so the frontend can only route round-robin / least-loaded.
- **No metrics.** No `ForwardPassMetrics`. The Dynamo router has zero load
  visibility into Megatron.
- **No disagg.** Single worker handles both prefill and decode for each
  request. No NIXL handoff.
- **Tokenizer must match.** Dynamo's frontend tokenizes with the HF tokenizer
  specified by `--model`. Megatron's engine receives the resulting token IDs
  verbatim and uses them as `prompt_tokens` — if Megatron was trained with a
  different tokenizer, generation will be garbage. Always pass the same HF
  tokenizer id to both sides.
- **Single-node only.** Multi-node coordinator + multi-node worker pools come
  with Phase 5.

## Troubleshooting

- **Address never appears:** `coordinator-<jobid>.out` has the engine startup
  trace. Common causes: tokenizer download failure, missing
  `--tensor-model-parallel-size` divisibility, OOM during model load.
- **Worker exits immediately:** `dynamo-worker-<jobid>.out` will show the
  cause. Most often: NATS/etcd unreachable, or `megatron-core` missing from
  the Dynamo worker's Python env.
- **Streaming stalls after first token:** check that the coordinator log
  shows `coordinator_streaming` NVTX scopes firing (with NVTX off, just check
  for absence of errors). Most stalls come from a request being submitted
  without `sampling_params.streaming=True` — which our handler always sets,
  so this should not happen in Phase 0, but it is the first place to look.
