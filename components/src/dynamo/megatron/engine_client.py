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
        self._client = InferenceClient(coordinator_addr, deserialize=False)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._client.start()
        self._started = True
        logger.info("MegatronEngineClient connected to %s", self._client.socket)

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
        if self._started:
            self._client.stop()
            self._started = False
