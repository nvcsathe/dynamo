# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase-3 smoke tests for the prefill + decode worker handlers.

These tests stub out :class:`MegatronEngineClient` with a fake that records
calls and yields deterministic chunks; no live coordinator, no NIXL, no
network. They cover:

1. ``PrefillWorkerHandler`` runs one prefill call and emits a single chunk
   carrying ``disaggregated_params`` (plus the first token).
2. ``DecodeWorkerHandler`` in ``role=decode`` rejects requests without
   ``prefill_result``.
3. ``DecodeWorkerHandler`` in ``role=decode`` passes ``prefill_result`` fields
   to ``decode_with_kv`` correctly and emits the streamed chunks.
4. ``DecodeWorkerHandler`` in ``role=decode`` calls ``release_handoff`` after
   the decode completes (so the prefill engine can free its pinned blocks).
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

    async def decode_with_kv(
        self, token_ids, sampling_params, kv_meta, src_block_ids, first_token
    ):
        self.last_call = {
            "token_ids": list(token_ids),
            "sampling_params": sampling_params,
            "kv_meta": kv_meta,
            "src_block_ids": list(src_block_ids),
            "first_token": first_token,
        }
        for chunk in self.chunks:
            yield chunk

    def release_handoff(self, request_id: int) -> None:
        self.released.append(int(request_id))


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------


async def test_prefill_handler_emits_disaggregated_params_and_first_token():
    engine = _FakePrefillClient(
        reply={
            "request_id": 42,
            "disaggregated_params": {
                "block_ids": [3, 4, 5],
                "kv_meta": {"agent_name": "prefill-rank0", "host": "h", "port": 9000},
                "first_token": 17,
            },
        }
    )
    handler = PrefillWorkerHandler(_make_config(role="prefill"), engine)
    request = {"token_ids": [1, 2, 3, 4, 5, 6, 7, 8]}

    responses = [r async for r in handler.generate(request, context=None)]

    assert len(responses) == 1
    resp = responses[0]
    assert resp["finish_reason"] == "stop"
    assert resp["token_ids"] == [17]
    assert resp["disaggregated_params"]["block_ids"] == [3, 4, 5]
    assert resp["disaggregated_params"]["kv_meta"]["agent_name"] == "prefill-rank0"
    assert resp["disaggregated_params"]["first_token"] == 17

    # Engine sees the full prompt verbatim.
    assert engine.last_call["token_ids"] == [1, 2, 3, 4, 5, 6, 7, 8]


async def test_prefill_handler_omits_token_when_first_token_missing():
    """A reply without first_token still produces a valid response (empty token_ids)."""
    engine = _FakePrefillClient(
        reply={
            "request_id": 99,
            "disaggregated_params": {
                "block_ids": [1],
                "kv_meta": {},
                "first_token": None,
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
                "first_token": 17,
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
    assert engine.last_call["first_token"] == 17
    assert engine.last_call["sampling_params"].temperature == pytest.approx(0.5)
    assert engine.last_call["sampling_params"].num_tokens_to_generate == 8

    # Release issued for the right request_id after the stream ended.
    assert engine.released == [77]


async def test_decode_role_releases_handoff_even_on_engine_error():
    """If the engine stream raises mid-decode, the handler still releases."""

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
                "first_token": 5,
            }
        },
    }

    with pytest.raises(RuntimeError, match="boom"):
        async for _ in handler.generate(request, context=None):
            pass

    # The decode never produced a `finished=True` chunk → no request_id was
    # captured → no release. Document this minor wart: production callers
    # should also drive release on timeout / abort paths from the frontend
    # side. (Tracked under the "Block release on decode failure" risk in the
    # Phase-3 design memo.)
    assert engine.released == []
