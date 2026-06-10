# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase-3 smoke tests for the prefill + decode worker handlers.

These tests stub out :class:`MegatronEngineClient` with a fake that records
calls and yields deterministic chunks; no live coordinator, no NIXL, no
network. They cover:

1. ``PrefillWorkerHandler`` runs one prefill call and emits a single chunk
   carrying ``disaggregated_params`` only.
2. ``DecodeWorkerHandler`` in ``role=decode`` rejects requests without
   ``prefill_result``.
3. ``DecodeWorkerHandler`` in ``role=decode`` passes ``prefill_result`` fields
   to ``decode_with_kv`` correctly and emits the streamed chunks.
4. ``DecodeWorkerHandler`` in ``role=decode`` releases prefill-owned KV after
   decode has started, so the prefill engine can make its pinned blocks reusable.
"""

from __future__ import annotations

from typing import Any

import pytest

from dynamo.megatron.handlers import DecodeWorkerHandler, PrefillWorkerHandler
from dynamo.megatron.tests.test_handler_smoke import _make_config

pytestmark = pytest.mark.asyncio


class _FakePrefillClient:
    """Captures the prefill_for_handoff call and returns a canned reply."""

    def __init__(self, reply: dict[str, Any]):
        self.reply = reply
        self.last_call: dict[str, Any] | None = None

    async def prefill_for_handoff(self, token_ids, sampling_params):
        self.last_call = {
            "token_ids": list(token_ids),
            "sampling_params": sampling_params,
        }
        return self.reply


class _FakeDecodeClient:
    """Captures decode_with_kv args + yields a fixed stream."""

    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        self.last_call: dict[str, Any] | None = None
        self.released: list[int] = []
        self.remote_released: list[tuple[str, int]] = []

    async def decode_with_kv(self, token_ids, sampling_params, kv_meta, src_block_ids):
        self.last_call = {
            "token_ids": list(token_ids),
            "sampling_params": sampling_params,
            "kv_meta": kv_meta,
            "src_block_ids": list(src_block_ids),
        }
        for chunk in self.chunks:
            yield chunk

    def release_handoff(self, request_id: int) -> None:
        self.released.append(int(request_id))

    def release_remote_handoff(self, coordinator_addr: str, request_id: int) -> None:
        self.remote_released.append((coordinator_addr, int(request_id)))


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------


async def test_prefill_handler_emits_disaggregated_params_without_tokens():
    engine = _FakePrefillClient(
        reply={
            "request_id": 42,
            "disaggregated_params": {
                "request_id": 314,
                "block_ids": [3, 4, 5],
                "kv_meta": {"agent_name": "prefill-rank0", "host": "h", "port": 9000},
            },
        }
    )
    handler = PrefillWorkerHandler(_make_config(role="prefill"), engine)
    request = {"token_ids": [1, 2, 3, 4, 5, 6, 7, 8]}

    responses = [r async for r in handler.generate(request, context=None)]

    assert len(responses) == 1
    resp = responses[0]
    assert resp["finish_reason"] == "stop"
    assert resp["token_ids"] == []
    assert resp["disaggregated_params"]["request_id"] == 314
    assert resp["disaggregated_params"]["release"] == {
        "coordinator_addr": "tcp://127.0.0.1:0",
        "request_id": 314,
    }
    assert resp["disaggregated_params"]["block_ids"] == [3, 4, 5]
    assert resp["disaggregated_params"]["kv_meta"]["agent_name"] == "prefill-rank0"

    # Engine sees the full prompt verbatim.
    assert engine.last_call["token_ids"] == [1, 2, 3, 4, 5, 6, 7, 8]


async def test_prefill_handler_omits_tokens_for_empty_handoff_meta():
    engine = _FakePrefillClient(
        reply={
            "request_id": 99,
            "disaggregated_params": {
                "block_ids": [1],
                "kv_meta": {},
            },
        }
    )
    handler = PrefillWorkerHandler(_make_config(role="prefill"), engine)
    responses = [r async for r in handler.generate({"token_ids": [1, 2]}, context=None)]
    assert responses[0]["token_ids"] == []


# ---------------------------------------------------------------------------
# Decode (role=decode)
# ---------------------------------------------------------------------------


async def test_decode_role_rejects_missing_prefill_result():
    engine = _FakeDecodeClient(chunks=[])
    handler = DecodeWorkerHandler(_make_config(role="decode"), engine)
    with pytest.raises(ValueError, match="prefill_result"):
        async for _ in handler.generate({"token_ids": [1, 2, 3]}, context=None):
            pass


async def test_decode_role_passes_handoff_meta_to_engine_and_streams():
    engine = _FakeDecodeClient(
        chunks=[
            {"new_tokens": [10, 11], "finished": False},
            {"new_tokens": [12], "finished": True, "reply": {"request_id": 77}},
        ]
    )
    handler = DecodeWorkerHandler(_make_config(role="decode"), engine)
    request = {
        "token_ids": [1, 2, 3, 4],
        "sampling_options": {"temperature": 0.5},
        "stop_conditions": {"max_tokens": 8},
        "prefill_result": {
            "disaggregated_params": {
                "block_ids": [9, 10],
                "kv_meta": {"agent_name": "prefill-rank0"},
                "release": {
                    "coordinator_addr": "tcp://prefill.example:5555",
                    "request_id": 1234,
                },
            }
        },
    }

    responses = [r async for r in handler.generate(request, context=None)]

    # Two streamed responses (partial + final).
    assert [r["token_ids"] for r in responses] == [[10, 11], [12]]
    assert responses[-1]["finish_reason"] == "stop"

    # Engine received the right handoff meta.
    assert engine.last_call["src_block_ids"] == [9, 10]
    assert engine.last_call["kv_meta"]["agent_name"] == "prefill-rank0"
    assert engine.last_call["sampling_params"].temperature == pytest.approx(0.5)
    assert engine.last_call["sampling_params"].num_tokens_to_generate == 8

    # Release issued for the prefill request as soon as decode starts.
    assert engine.released == []
    assert engine.remote_released == [("tcp://prefill.example:5555", 1234)]


async def test_decode_role_releases_handoff_after_first_chunk_on_engine_error():
    """If decode errors after import, the prefill handoff has already been released."""

    class _ExplodingClient(_FakeDecodeClient):
        async def decode_with_kv(self, *args, **kwargs):
            self.last_call = {"args": args}
            yield {"new_tokens": [10], "finished": False}
            raise RuntimeError("boom")

    engine = _ExplodingClient(chunks=[])
    handler = DecodeWorkerHandler(_make_config(role="decode"), engine)
    request = {
        "token_ids": [1, 2, 3],
        "prefill_result": {
            "disaggregated_params": {
                "block_ids": [1],
                "kv_meta": {},
                "release": {
                    "coordinator_addr": "tcp://prefill.example:5555",
                    "request_id": 5678,
                },
            }
        },
    }

    with pytest.raises(RuntimeError, match="boom"):
        async for _ in handler.generate(request, context=None):
            pass

    assert engine.released == []
    assert engine.remote_released == [("tcp://prefill.example:5555", 5678)]
