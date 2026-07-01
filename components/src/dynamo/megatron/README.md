# Dynamo Megatron Backend

`dynamo.megatron` owns a complete Megatron inference engine. The registered
Dynamo worker is a lightweight parent process: during `start()` it launches a
private TP/PP/EP rank group and Megatron coordinator, then connects through
`InferenceClient`. The child ranks contain no Dynamo worker code and the
coordinator is never shared or supplied by an operator.

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

The parent follows the common `LLMEngine` entry point and launches
`megatron.inference.dynamic_server` through `torch.distributed.run`. Every
child rank initializes and loads Megatron, while the parent waits for an
atomic readiness advertisement and coordinator metadata. Dynamo registration
happens only after the coordinator, model-parallel ranks, and any NIXL agent
are ready.

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

## Docker build and validation runbook

This refactor changes both the Dynamo checkout and Megatron-LM. The image
build uses the local Dynamo checkout as its build context, but clones
Megatron-LM from `MEGATRON_REPO` at `MEGATRON_REF`. Push the matching
Megatron-LM commit to a reachable branch before producing a deployable image;
otherwise the parent and coordinator protocol versions will not match.

### Build the image

From the Dynamo repository root:

```bash
export IMAGE=dynamo:megatron-owned-backend
export MEGATRON_REPO=https://github.com/<your-fork>/Megatron-LM.git
export MEGATRON_REF=<branch-or-tag-containing-this-refactor>

# The base image comes from nvcr.io; authenticate first if needed.
docker login nvcr.io

container/render.py \
  --framework megatron \
  --target runtime \
  --output-short-filename

DOCKER_BUILDKIT=1 docker build \
  -f container/rendered.Dockerfile \
  --build-arg MEGATRON_REPO="$MEGATRON_REPO" \
  --build-arg MEGATRON_REF="$MEGATRON_REF" \
  -t "$IMAGE" \
  .
```

Confirm that the image contains both sides of the new boundary:

```bash
docker run --rm "$IMAGE" python -c '
from dynamo.megatron.llm_engine import MegatronLLMEngine
from megatron.inference.dynamic_server import main
from megatron.core.inference.headers import Headers
assert hasattr(Headers, "ENGINE_STATUS")
print("Dynamo parent and Megatron engine service are present")
'
```

For a temporary development run with unpushed Megatron Python changes, mount
the checkout at the path used by its editable installation:

```bash
export MEGATRON_LOCAL_DEV=/absolute/path/to/Megatron-LM
# Add to the docker run commands below:
#   -v "$MEGATRON_LOCAL_DEV:/opt/megatron-lm"
```

Rebuild whenever the Dynamo parent changes. Use the bind mount only for
development; deployable and Planner-scaled images must bake in a matching
Megatron commit.

### Launch an aggregated Docker smoke stack

The phase-0 orchestrator starts NATS, etcd, one owned Megatron worker, and the
Dynamo frontend. `STAGE` must contain the checkpoint and HF cache expected by
`examples/backends/megatron/phase0/orchestrate.sh`.

```bash
export STAGE=/absolute/path/to/model/staging

docker run --rm -d \
  --name dynamo-megatron-agg \
  --gpus all \
  --ipc=host \
  --network=host \
  -v "$STAGE:$STAGE" \
  -e STAGE="$STAGE" \
  -e HF_TOKEN \
  -e TP=1 \
  "$IMAGE" \
  bash -lc 'bash /workspace/examples/backends/megatron/phase0/orchestrate.sh'

# Wait for PHASE0_READY, then press Ctrl-C; the container keeps running.
docker logs -f dynamo-megatron-agg
docker exec dynamo-megatron-agg \
  bash /workspace/examples/backends/megatron/phase0/test_phase0.sh

docker stop dynamo-megatron-agg
```

Override `MODEL_CHECKPOINT`, `TOKENIZER_MODEL`, `SERVED_MODEL_NAME`, `TP`, and
`CONTEXT_LENGTH` with `-e NAME=value` as needed. `TP` is the number of GPUs
owned by this one logical worker replica.

### Launch the disaggregated Docker smoke stack

The phase-3 stack starts independent prefill and decode Dynamo workers. Each
worker launches its own coordinator and rank group. The defaults use one GPU
for prefill and one for decode.

```bash
docker run --rm -d \
  --name dynamo-megatron-disagg \
  --gpus all \
  --ipc=host \
  --network=host \
  -v "$STAGE:$STAGE" \
  -e STAGE="$STAGE" \
  -e HF_TOKEN \
  -e TP_PREFILL=1 \
  -e TP_DECODE=1 \
  "$IMAGE" \
  bash -lc 'bash /workspace/examples/backends/megatron/phase3/orchestrate.sh'

# Wait for PHASE3_READY, then press Ctrl-C; the container keeps running.
docker logs -f dynamo-megatron-disagg
docker exec dynamo-megatron-disagg \
  bash /workspace/examples/backends/megatron/phase3/test_phase3.sh

docker stop dynamo-megatron-disagg
```

For heterogeneous layouts, set `TP_PREFILL`, `PP_PREFILL`, `TP_DECODE`, and
`PP_DECODE`. The GPU count is
`TP_PREFILL*PP_PREFILL + TP_DECODE*PP_DECODE`.

### Run the focused tests in Docker

The runtime image may not include pytest. Install test-only dependencies in an
ephemeral container and run both repositories' focused suites there:

```bash
docker run --rm \
  --gpus all \
  --ipc=host \
  "$IMAGE" \
  bash -lc '
    uv pip install pytest pytest-asyncio

    cd /workspace
    pytest -q components/src/dynamo/megatron/tests

    cd /opt/megatron-lm
    pytest -q \
      tests/unit_tests/inference/test_inference_client.py \
      tests/unit_tests/inference/test_inference_client_streaming.py \
      tests/unit_tests/inference/test_data_parallel_inference_coordinator.py \
      tests/unit_tests/inference/test_dynamic_server.py \
      tests/unit_tests/inference/test_async_llm_streaming.py \
      tests/unit_tests/inference/test_kv_allocator_observers.py
  '
```

To test unpushed Megatron changes, add
`-v "$MEGATRON_LOCAL_DEV:/opt/megatron-lm"` to this command.

### Exercise Planner scaling

Push the image to a registry visible to the cluster, replace every
`mainContainer.image` in
`examples/backends/megatron/deploy/disagg_planner.yaml`, and apply it to a
cluster with the Dynamo operator and Planner installed:

```bash
docker tag "$IMAGE" <registry>/dynamo-megatron:<tag>
docker push <registry>/dynamo-megatron:<tag>

kubectl apply -f examples/backends/megatron/deploy/disagg_planner.yaml
kubectl get dgd megatron-disagg-planner -w
```

Each Planner replica creates a pod running `python -m dynamo.megatron`; that
parent launches a private coordinator/rank group and registers one new Dynamo
endpoint. No pre-existing coordinator service is required.

## Runtime Contract

- `InferenceClient` supplies incremental token IDs directly from
  `ENGINE_REPLY_PARTIAL` and the terminal `ENGINE_REPLY`.
- Prefill runs a zero-token request, pins complete prompt blocks, and returns
  NIXL metadata through `disaggregated_params`.
- Decode imports those blocks before generation and releases the source after
  the first post-import engine output.
- Stream closure and Dynamo cancellation send `ABORT_REQUEST` to the exact
  Megatron request.
- Prefix block events and scheduler snapshots cross the private coordinator's
  telemetry channel and are published by the parent as logical DP rank zero.
- Shutdown unregisters the endpoint, drains active requests and pinned
  handoffs up to `--drain-timeout`, then stops all ranks.

The initial implementation supports one node per engine. Multi-node
rendezvous and gang lifecycle are intentionally deferred.

The DGD examples in `examples/backends/megatron/deploy` allocate all local
rank GPUs to each component replica and demonstrate load-based Planner
scaling.
