# Dynamo Megatron Backend

`dynamo.megatron` owns a complete Megatron inference engine. It launches the
local TP/PP/EP rank group, constructs `MegatronAsyncLLM` on every rank, and
registers one Dynamo endpoint from global rank zero only. It does not launch
Megatron's HTTP server and does not connect to an operator-managed coordinator.

Each registered endpoint is one independently routable logical `DP=1`
replica. For MoE models this is measured by expert-DP because regular DP
includes the EP-overlapped ranks. Dynamo routing and Planner scaling therefore
operate on complete engines rather than individual model-parallel ranks.

## Launch

Pass Dynamo launcher arguments before `--` and normal Megatron arguments after
it. `--nproc-per-node` is required and is not inferred from parallelism flags.

```bash
python -m dynamo.megatron \
  --role aggregated \
  --model Qwen/Qwen3-8B \
  --served-model-name Qwen/Qwen3-8B \
  --nproc-per-node 4 \
  --megatron-root /opt/megatron-lm \
  -- \
  --load /models/qwen3-8b-megatron \
  --tensor-model-parallel-size 4 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model Qwen/Qwen3-8B \
  --inference-dynamic-batching \
  --inference-dynamic-batching-prefix-caching
```

The parent process re-executes itself through `torch.distributed.run`. Every
rank initializes and loads Megatron; rank zero then starts the common Dynamo
`Worker`. Registration happens only after the coordinator, model-parallel
ranks, and any NIXL agent are ready.

For disaggregated serving, launch separate prefill and decode components:

```bash
python -m dynamo.megatron \
  --role prefill \
  --component prefill \
  --model Qwen/Qwen3-8B \
  --nproc-per-node 4 \
  --coordinator-host 10.0.0.12 \
  --kv-transfer-listen-addr 10.0.0.12:7000 \
  -- <Megatron model and inference arguments>

python -m dynamo.megatron \
  --role decode \
  --component backend \
  --model Qwen/Qwen3-8B \
  --nproc-per-node 4 \
  --coordinator-host 10.0.0.13 \
  --kv-transfer-listen-addr 10.0.0.13:7000 \
  -- <Megatron model and inference arguments>
```

The coordinator address is private handoff metadata used to release
prefill-owned blocks after decode imports them. It is not a Dynamo routing
identity. Both addresses must be reachable from the peer component.

## Runtime Contract

- `MegatronAsyncLLM.generate_stream()` supplies incremental token IDs directly
  from `ENGINE_REPLY_PARTIAL` and the terminal `ENGINE_REPLY`.
- Prefill runs a zero-token request, pins complete prompt blocks, and returns
  NIXL metadata through `disaggregated_params`.
- Decode imports those blocks before generation and releases the source after
  the first post-import engine output.
- Stream closure and Dynamo cancellation send `ABORT_REQUEST` to the exact
  Megatron request.
- Prefix block events are emitted only by rank zero and fed into Dynamo's
  `PushSource`; decode replicas publish load snapshots but no routing events.
- Shutdown unregisters the endpoint, drains active requests and pinned
  handoffs up to `--drain-timeout`, then stops all ranks.

The initial implementation supports one node per engine. Multi-node
rendezvous and gang lifecycle are intentionally deferred.

## Tests

Run CPU-facing unit tests in an environment containing both Dynamo and
Megatron dependencies:

```bash
pytest -q components/src/dynamo/megatron/tests
pytest -q tests/unit_tests/inference/test_async_llm_streaming.py \
  tests/unit_tests/inference/test_kv_allocator_observers.py
```

The DGD examples in `examples/backends/megatron/deploy` allocate all local
rank GPUs to each component replica and demonstrate load-based Planner
scaling.
