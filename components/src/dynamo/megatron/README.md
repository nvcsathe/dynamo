# Dynamo Megatron Backend (Phase 0)

This package is a Dynamo worker that proxies streaming generation requests to
a separately-launched [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
`DataParallelInferenceCoordinator`. The Dynamo worker does **not** own
Megatron's forward-pass loop — it is a thin client, mirroring the way Dynamo's
vLLM backend talks to vLLM's `AsyncLLM` / EngineCore child process.

**Status:** Phase 0 — aggregated decode, streaming tokens. No KV events, no
metrics publishing, no disaggregated prefill/decode split. Those land in later
phases.

## Layout

```
dynamo/megatron/
  __init__.py
  __main__.py             # `python -m dynamo.megatron`
  args.py                 # CLI Config
  engine_client.py        # Async wrapper around megatron InferenceClient
  handlers.py             # DecodeWorkerHandler — request translation + streaming
  main.py                 # worker() — endpoint registration + serve loop
  tests/
    test_handler_smoke.py # Smoke tests with a fake engine client
```

## Dependencies

- `megatron-core` must be importable in this worker's Python environment. The
  worker imports `megatron.core.inference.inference_client.InferenceClient`
  and `megatron.core.inference.sampling_params.SamplingParams` directly. Install
  Megatron-LM from the `feature/dynamo-streaming` branch (or any branch that
  has the `add_request_streaming` API and the `ENGINE_REPLY_PARTIAL` header).
- `pyzmq`, `msgpack` — pulled in transitively by `megatron-core`.
- The Dynamo runtime (`dynamo.runtime`, `dynamo.llm`) — already a peer
  dependency for any Dynamo backend.

## End-to-end flow (Phase 0)

1. Launch the Megatron coordinator on the model node(s):

   ```
   torchrun --nproc-per-node 2 tools/run_dynamic_text_generation_server.py \
       --frontend dynamo \
       --inference-coordinator-port 5555 \
       --tensor-model-parallel-size 2 \
       --load <megatron-checkpoint> \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model <hf-model-id>
       # ... other Megatron flags
   ```

   The launcher prints `MEGATRON_COORDINATOR_ADDR=tcp://<host>:<port>` to
   stdout. Capture it.

2. Launch the Dynamo Megatron worker against that coordinator:

   ```
   python -m dynamo.megatron \
       --coordinator-addr "$MEGATRON_COORDINATOR_ADDR" \
       --model <hf-model-id> \
       --context-length 4096
   ```

   The HF model id is what the Dynamo frontend uses to fetch the tokenizer /
   `config.json`. **It must match the tokenizer the Megatron engine was built
   with**, otherwise tokenization will diverge.

3. Launch a Dynamo frontend (separate process) and hit it:

   ```
   curl -N http://<frontend>/v1/chat/completions \
       -H 'content-type: application/json' \
       -d '{"model": "<hf-model-id>", "messages": [...], "stream": true}'
   ```

   Tokens should stream back. The path is: frontend tokenizes → tokens to
   `dynamo.backend.generate` → Megatron worker handler → InferenceClient
   `add_request_streaming` → Megatron coordinator → engine emits
   `ENGINE_REPLY_PARTIAL` per step → handler yields `{token_ids: [...]}` per
   chunk back to the frontend.

## Running the smoke tests

The smoke tests in `tests/test_handler_smoke.py` mock out
`MegatronEngineClient` so they do not require a coordinator or Megatron model.
They run in any Python environment that has Dynamo's Python deps installed.

The tests have **not** been executed in the dev environment — they need to be
run inside the Dynamo container (which has `dynamo.runtime`, `dynamo.llm`, etc.
on the import path).

Inside the container:

```bash
# All smoke tests:
pytest -q components/src/dynamo/megatron/tests/

# Single test:
pytest -q components/src/dynamo/megatron/tests/test_handler_smoke.py::test_handler_streams_partials_then_finish
```

If you don't have the full Dynamo container handy, the smoke tests can be run
with just `pytest` + `pytest-asyncio` + `megatron-core` if you stub the
`dynamo.runtime`, `dynamo.common.utils.runtime`, and `dynamo.llm` imports — but
that's only worthwhile if you're iterating on the handler logic in isolation.

## What's not tested yet (follow-up)

- **End-to-end with a real coordinator.** A test that:
  1. Spawns a minimal in-process fake coordinator (just a ZMQ ROUTER that
     accepts `CONNECT`/`SUBMIT_REQUEST` and replies with
     `ENGINE_REPLY_PARTIAL` frames followed by `ENGINE_REPLY`).
  2. Builds a real `MegatronEngineClient` against it.
  3. Drives a handler request through and asserts the streamed responses.

  This is the strongest signal short of bringing up Megatron itself, and it's
  the test that catches the wire-protocol bugs the unit tests can't.

- **Frontend integration.** Once the smoke tests pass, the next step is to run
  the Phase-0 sbatch launcher (see `examples/backends/megatron/phase0/`
  pending) end-to-end and verify token streaming through curl. That's a manual
  validation step, not a unit test.

## Migrating to later phases

Each follow-up phase adds files to this package without restructuring the
existing ones:

- **Phase 1 (metrics):** add `publisher.py` that subscribes to a new ZMQ PUB
  emitted by the coordinator carrying scheduler + KV block counts, then
  forwards to `WorkerMetricsPublisher` (Rust → NATS). The handler itself does
  not change.
- **Phase 2 (KV events):** extend Megatron's `KVBlockAllocator` to emit
  `BlockStored` / `BlockRemoved` over a ZMQ PUB; add `kv_event_publisher.py`
  here that bridges to `dynamo._core.KvEventPublisher`. Independent of the
  handler.
- **Phase 3 (disagg):** add `MegatronPrefillWorker` / `MegatronDecodeWorker`
  handlers and split `main.py` into two role-aware entrypoints.
- **Phase 4 (NIXL):** add `kv_connector.py` registering the Megatron KV
  buffer with NIXL and exposing handle metadata for the decode side.
