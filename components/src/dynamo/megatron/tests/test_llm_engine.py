from __future__ import annotations

from types import SimpleNamespace

import pytest

from dynamo.megatron.args import Config
from dynamo.megatron.llm_engine import MegatronLLMEngine, build_sampling_params


def _config(role="aggregated"):
    return Config(
        model="model",
        served_model_name="served",
        namespace="dynamo",
        component="prefill" if role == "prefill" else "backend",
        endpoint="generate",
        discovery_backend="etcd",
        request_plane="nats",
        event_plane="nats",
        role=role,
        nproc_per_node=1,
        coordinator_host=None,
        coordinator_port=None,
        kv_transfer_listen_addr="127.0.0.1:7000" if role != "aggregated" else None,
        megatron_root="/opt/megatron-lm",
        drain_timeout=0.1,
        megatron_argv=["--load", "/checkpoint"],
    )


def test_sampling_params_maps_greedy_and_limits():
    params = build_sampling_params(
        {
            "token_ids": [1],
            "sampling_options": {"temperature": 0.0, "top_p": 0.9},
            "stop_conditions": {"max_tokens": 7},
        }
    )
    assert params.top_k == 1
    assert params.top_p == 0.0
    assert params.num_tokens_to_generate == 7


@pytest.mark.asyncio
async def test_start_advertises_real_engine_capacity():
    metadata = SimpleNamespace(
        context_length=8192,
        block_size_tokens=64,
        total_kv_blocks=1024,
        max_requests=32,
        max_tokens=4096,
        coordinator_address="tcp://127.0.0.1:5555",
        role="aggregated",
    )
    llm = SimpleNamespace(
        metadata=metadata,
        context=SimpleNamespace(enable_prefix_caching=True),
        add_kv_event_listener=lambda callback: None,
        add_metrics_listener=lambda callback: None,
    )

    async def initialize():
        return llm

    engine = MegatronLLMEngine(_config(), initialize)
    result = await engine.start(1)
    assert result.total_kv_blocks == 1024
    assert result.kv_cache_block_size == 64
    assert result.data_parallel_size == 1


class _Context:
    def __init__(self, request_id="dynamo-request"):
        self.request_id = request_id

    def id(self):
        return self.request_id


@pytest.mark.asyncio
async def test_decode_health_probe_bypasses_kv_handoff():
    handoff_called = False

    async def output_stream(on_request_started):
        on_request_started(31)
        yield SimpleNamespace(token_ids=[9], finished=True)

    def generate_stream(_tokens, _params, on_request_started):
        return output_stream(on_request_started)

    def generate_stream_with_kv_handoff(*_args, **_kwargs):
        nonlocal handoff_called
        handoff_called = True
        raise AssertionError("health probe must not import KV")

    engine = MegatronLLMEngine(_config("decode"), lambda: None)
    engine.llm = SimpleNamespace(
        generate_stream=generate_stream,
        generate_stream_with_kv_handoff=generate_stream_with_kv_handoff,
    )
    request = {
        "token_ids": [1],
        "_HEALTH_CHECK": True,
        "sampling_options": {},
        "stop_conditions": {"max_tokens": 1},
    }

    chunks = [chunk async for chunk in engine.generate(request, _Context())]

    assert chunks[-1]["token_ids"] == [9]
    assert chunks[-1]["finish_reason"] == "length"
    assert not handoff_called


@pytest.mark.asyncio
async def test_abort_uses_megatron_request_id_recorded_for_context():
    aborted = []

    async def abort(request_id):
        aborted.append(request_id)

    engine = MegatronLLMEngine(_config(), lambda: None)
    engine.llm = SimpleNamespace(abort=abort)
    engine._request_ids["dynamo-request"] = 77

    await engine.abort(_Context())

    assert aborted == [77]


def test_metrics_snapshot_includes_scheduler_load():
    published = []
    publisher = SimpleNamespace(
        publish=lambda rank, snapshot: published.append((rank, snapshot))
    )
    engine = MegatronLLMEngine(_config(), lambda: None)
    engine.attach_snapshot_publisher(publisher)

    engine._on_metrics(
        {
            "allocated_blocks": 25,
            "total_blocks": 100,
            "allocated_utilization": 0.25,
            "prefix_cache_hit_rate": 0.5,
            "active_request_count": 3,
            "waiting_request_count": 2,
        }
    )

    rank, snapshot = published[0]
    assert rank == 0
    assert snapshot.active_requests == 3
    assert snapshot.waiting_requests == 2
