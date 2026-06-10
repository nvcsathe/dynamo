# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin async wrapper around Megatron's :class:`InferenceClient`.

The Dynamo worker does NOT own Megatron's forward-pass loop. It connects as a
client to a separately-launched Megatron coordinator (vLLM-style separation,
mirroring how Dynamo's vllm backend talks to vLLM's EngineCore child process).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


class MegatronEngineClient:
    """Async-iterator wrapper around InferenceClient.add_request_streaming."""

    def __init__(self, coordinator_addr: str):
        # Stash the address as a string so start()'s log line is greppable;
        # the raw ZMQ socket repr doesn't show the connect target.
        self._coordinator_addr = coordinator_addr
        self._client = InferenceClient(coordinator_addr, deserialize=False)
        self._release_clients: dict[str, InferenceClient] = {}
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._client.start()
        self._started = True
        logger.info(
            "MegatronEngineClient connected to coordinator at %s",
            self._coordinator_addr,
        )

    async def prefill_for_handoff(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> dict[str, Any]:
        """Phase-3 prefill: submit a prefill-only request with do_kv_handoff=True.

        Returns the final reply dict once Megatron has populated KV blocks
        and pinned them. The reply's ``disaggregated_params`` field carries
        ``{request_id, block_ids, kv_meta}`` for the decode peer.

        This is non-streaming — prefill produces no tokens, so we just
        consume the iterator to completion and return the final frame.
        """
        if not self._started:
            raise RuntimeError("MegatronEngineClient.start() must be called first")
        # Prefill emits a single ENGINE_REPLY (no partials). Force handoff on
        # and leave first-token generation to the decode worker.
        sampling_params.do_kv_handoff = True
        sampling_params.streaming = False
        sampling_params.num_tokens_to_generate = 0

        iterator = self._client.add_request_streaming(token_ids, sampling_params)
        # Even with streaming=False, add_request_streaming wires a queue and
        # delivers the final reply through `final`. Loop until we see it.
        async for item in iterator:
            if "final" in item:
                reply = item["final"]
                # `reply` is a serialized DynamicInferenceRequest dict (because
                # InferenceClient was constructed with deserialize=False).
                # disaggregated_params rides as a top-level field.
                return reply
        raise RuntimeError("MegatronEngineClient.prefill_for_handoff: stream ended without final")

    async def decode_with_kv(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams,
        kv_meta: dict[str, Any],
        src_block_ids: list[int],
    ) -> AsyncIterator[dict[str, Any]]:
        """Phase-3 decode: submit a request that imports KV from a prefill peer.

        Streams tokens identically to :meth:`generate`; the only difference is
        the engine-side wire header (SUBMIT_REQUEST_WITH_KV) and the extra
        ``kv_meta`` / ``src_block_ids`` payload.
        """
        if not self._started:
            raise RuntimeError("MegatronEngineClient.start() must be called first")
        iterator = self._client.add_request_with_kv_handoff(
            token_ids,
            sampling_params,
            kv_meta,
            src_block_ids,
        )
        emitted_count = 0
        async for item in iterator:
            if "partial" in item:
                new_tokens = item["partial"]["new_tokens"]
                emitted_count += len(new_tokens)
                yield {"new_tokens": new_tokens, "finished": False}
            elif "final" in item:
                reply = item["final"]
                generated = reply.get("generated_tokens") or []
                tail = list(generated[emitted_count:])
                yield {"new_tokens": tail, "finished": True, "reply": reply}

    def release_handoff(self, request_id: int) -> None:
        """Tell the coordinator to release blocks pinned by the prefill engine."""
        if not self._started:
            return
        self._client.release_handoff(request_id)

    def release_remote_handoff(self, coordinator_addr: str, request_id: int) -> None:
        """Release prefill-owned KV blocks through the coordinator that owns them."""
        if coordinator_addr == self._coordinator_addr:
            self.release_handoff(request_id)
            return

        client = self._release_clients.get(coordinator_addr)
        if client is None:
            client = InferenceClient(coordinator_addr, deserialize=False)
            client.start()
            self._release_clients[coordinator_addr] = client
            logger.info(
                "MegatronEngineClient connected release channel to prefill coordinator at %s",
                coordinator_addr,
            )
        client.release_handoff(request_id)

    async def generate(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> AsyncIterator[dict[str, Any]]:
        """Submit a streaming generation request.

        Yields one dict per ENGINE_REPLY_PARTIAL frame and one final dict on the
        terminating ENGINE_REPLY. Each yielded dict has shape:

        - ``{"new_tokens": list[int], "finished": False}`` for partials.
        - ``{"new_tokens": list[int], "finished": True, "reply": <full dict>}``
          for the final reply. ``new_tokens`` on the final frame contains any
          tokens generated since the last partial (may be empty if the engine
          already emitted them as a partial).
        """
        if not self._started:
            raise RuntimeError("MegatronEngineClient.start() must be called first")

        iterator = self._client.add_request_streaming(token_ids, sampling_params)
        emitted_count = 0
        async for item in iterator:
            if "partial" in item:
                new_tokens = item["partial"]["new_tokens"]
                emitted_count += len(new_tokens)
                yield {"new_tokens": new_tokens, "finished": False}
            elif "final" in item:
                reply = item["final"]
                generated = reply.get("generated_tokens") or []
                tail = list(generated[emitted_count:])
                yield {"new_tokens": tail, "finished": True, "reply": reply}

    def stop(self) -> None:
        for client in self._release_clients.values():
            client.stop()
        self._release_clients.clear()
        if self._started:
            self._client.stop()
            self._started = False
