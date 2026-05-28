# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request handler that streams tokens from a Megatron coordinator to a Dynamo client.

Phase 0 only implements aggregated decode against a single coordinator. There
is no prefill / decode split, no KV publishing, no metrics. Those layers will
slot into the same handler interface later.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator

from megatron.core.inference.sampling_params import SamplingParams

from dynamo.megatron.args import Config
from dynamo.megatron.engine_client import MegatronEngineClient

logger = logging.getLogger(__name__)


def _build_sampling_params(request: dict[str, Any]) -> SamplingParams:
    """Translate a Dynamo preprocessed request into Megatron SamplingParams.

    The Dynamo frontend (token-based path) provides ``sampling_options`` and
    ``stop_conditions`` blocks. Phase 0 wires temperature, top_p, top_k, and
    max_tokens; everything else falls back to Megatron defaults.
    """
    sampling_opts = request.get("sampling_options") or {}
    stop_conditions = request.get("stop_conditions") or {}

    params = SamplingParams()
    if (temperature := sampling_opts.get("temperature")) is not None:
        params.temperature = float(temperature)
    if (top_p := sampling_opts.get("top_p")) is not None:
        params.top_p = float(top_p)
    if (top_k := sampling_opts.get("top_k")) is not None:
        params.top_k = int(top_k)
    if (max_tokens := stop_conditions.get("max_tokens")) is not None:
        params.num_tokens_to_generate = int(max_tokens)
    return params


class DecodeWorkerHandler:
    """Aggregated decode handler. Streams tokens from Megatron to Dynamo."""

    def __init__(self, config: Config, engine_client: MegatronEngineClient):
        self.config = config
        self.engine_client = engine_client

    async def generate(
        self, request: dict[str, Any], context: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        token_ids = list(request.get("token_ids") or [])
        if not token_ids:
            raise ValueError("Megatron backend requires token_ids in the request")

        sampling_params = _build_sampling_params(request)
        logger.debug(
            "Megatron handler: %d input tokens, max_new=%s, streaming=True",
            len(token_ids),
            sampling_params.num_tokens_to_generate,
        )

        async for chunk in self.engine_client.generate(token_ids, sampling_params):
            response: dict[str, Any] = {"token_ids": chunk["new_tokens"]}
            if chunk["finished"]:
                # Phase 0 does not propagate Megatron's finish reason granularly.
                # The frontend treats absence of further chunks as completion;
                # this flag makes intent explicit for downstream consumers.
                response["finish_reason"] = "stop"
            yield response
