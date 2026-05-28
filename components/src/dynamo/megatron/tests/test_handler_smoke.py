# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the Megatron decode handler.

These tests verify the handler-to-engine-client wiring without bringing up an
actual Megatron coordinator. The MegatronEngineClient is replaced with a fake
that yields a deterministic chunk sequence so we can assert the handler's
translation behavior.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator

import pytest

from dynamo.megatron.args import Config
from dynamo.megatron.handlers import DecodeWorkerHandler

pytestmark = pytest.mark.asyncio


def _make_config() -> Config:
    return Config(
        coordinator_addr="tcp://127.0.0.1:0",
        model="dummy/model",
        served_model_name="dummy/model",
        context_length=2048,
        namespace="dynamo",
        component="backend",
        endpoint="generate",
        discovery_backend="etcd",
        request_plane="nats",
        event_plane="nats",
    )


class _FakeEngineClient:
    """Yields a fixed sequence of chunks for handler tests."""

    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        self.last_token_ids: list[int] | None = None
        self.last_sampling_params: Any = None

    async def generate(self, token_ids: list[int], sampling_params: Any) -> AsyncGenerator:
        self.last_token_ids = token_ids
        self.last_sampling_params = sampling_params
        for chunk in self.chunks:
            yield chunk


async def test_handler_streams_partials_then_finish():
    """Two partial chunks followed by a final chunk: handler emits one token_ids dict per chunk."""
    engine = _FakeEngineClient(
        [
            {"new_tokens": [10, 11], "finished": False},
            {"new_tokens": [12], "finished": False},
            {"new_tokens": [13], "finished": True, "reply": {}},
        ]
    )
    handler = DecodeWorkerHandler(_make_config(), engine)

    request = {
        "token_ids": [1, 2, 3],
        "sampling_options": {"temperature": 0.7, "top_p": 0.9},
        "stop_conditions": {"max_tokens": 16},
    }

    responses = []
    async for resp in handler.generate(request, context=None):
        responses.append(resp)

    assert [r["token_ids"] for r in responses] == [[10, 11], [12], [13]]
    assert "finish_reason" not in responses[0]
    assert "finish_reason" not in responses[1]
    assert responses[2]["finish_reason"] == "stop"

    # Sampling params correctly translated.
    assert engine.last_token_ids == [1, 2, 3]
    assert engine.last_sampling_params.temperature == pytest.approx(0.7)
    assert engine.last_sampling_params.top_p == pytest.approx(0.9)
    assert engine.last_sampling_params.num_tokens_to_generate == 16


async def test_handler_rejects_empty_prompt():
    handler = DecodeWorkerHandler(_make_config(), _FakeEngineClient([]))
    with pytest.raises(ValueError, match="token_ids"):
        async for _ in handler.generate({"token_ids": []}, context=None):
            pass
