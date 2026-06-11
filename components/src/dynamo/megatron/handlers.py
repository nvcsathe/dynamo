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

    # temperature==0 means greedy. Megatron's HTTP endpoints (completions.py /
    # chat_completions.py) translate this to top_k=1, which TorchSampling
    # short-circuits to argmax. The Dynamo path bypasses those endpoints, so we
    # must replicate the conversion here. Without it torch_sampling does
    # `logits.div_(0.0)` -> inf -> softmax -> nan, and torch.multinomial aborts
    # with "probability tensor contains either inf, nan or element < 0".
    if params.temperature == 0.0:
        params.top_k = 1
        params.top_p = 0.0
    return params


class DecodeWorkerHandler:
    """Aggregated or disagg-decode handler.

    In ``role=aggregated`` mode the handler streams tokens from Megatron as in
    Phase 0. In ``role=decode`` mode the handler requires ``prefill_result`` to
    be present in the request and routes through
    :meth:`MegatronEngineClient.decode_with_kv` so the engine can NIXL-pull the
    prefill peer's KV state before generating.
    """

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
        prefill_result = request.get("prefill_result")

        if self.config.role == "decode":
            if not prefill_result or not isinstance(prefill_result, dict):
                raise ValueError(
                    "Megatron decode worker received a request without "
                    "prefill_result; expected the frontend's PrefillRouter to "
                    "populate it via a Prefill peer."
                )
            disagg = prefill_result.get("disaggregated_params") or {}
            kv_meta = disagg.get("kv_meta") or {}
            src_block_ids = list(disagg.get("block_ids") or [])
            release = disagg.get("release") or {}
            release_coordinator_addr = release.get("coordinator_addr")
            release_request_id = release.get("request_id")
            logger.debug(
                "Megatron decode handler: %d input tokens, %d imported KV blocks",
                len(token_ids),
                len(src_block_ids),
            )
            released_prefill = False
            async for chunk in self.engine_client.decode_with_kv(
                token_ids,
                sampling_params,
                kv_meta,
                src_block_ids,
            ):
                if (
                    not released_prefill
                    and release_coordinator_addr is not None
                    and release_request_id is not None
                ):
                    self.engine_client.release_remote_handoff(
                        str(release_coordinator_addr),
                        int(release_request_id),
                    )
                    released_prefill = True
                    logger.debug(
                        "Megatron decode handler released prefill KV blocks "
                        "request_id=%s via %s",
                        release_request_id,
                        release_coordinator_addr,
                    )

                response: dict[str, Any] = {"token_ids": chunk["new_tokens"]}
                if chunk["finished"]:
                    response["finish_reason"] = "stop"
                yield response
            return

        # Aggregated mode — Phase-0 path, unchanged.
        logger.debug(
            "Megatron aggregated handler: %d input tokens, max_new=%s, streaming=True",
            len(token_ids),
            sampling_params.num_tokens_to_generate,
        )

        async for chunk in self.engine_client.generate(token_ids, sampling_params):
            response: dict[str, Any] = {"token_ids": chunk["new_tokens"]}
            if chunk["finished"]:
                response["finish_reason"] = "stop"
            yield response


class PrefillWorkerHandler:
    """Phase-3 prefill handler. Runs one prefill step, returns disaggregated_params.

    The frontend's :class:`PrefillRouter` consumes the response's
    ``disaggregated_params`` field and ships it to a decode peer via the
    standard ``prefill_result`` envelope.
    """

    def __init__(self, config: Config, engine_client: MegatronEngineClient):
        self.config = config
        self.engine_client = engine_client

    async def generate(
        self, request: dict[str, Any], context: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        token_ids = list(request.get("token_ids") or [])
        if not token_ids:
            raise ValueError("Megatron prefill backend requires token_ids in the request")

        sampling_params = _build_sampling_params(request)
        reply = await self.engine_client.prefill_for_handoff(token_ids, sampling_params)
        disaggregated_params = dict(reply.get("disaggregated_params") or {})
        prefill_request_id = disaggregated_params.get("request_id", reply.get("request_id"))
        if prefill_request_id is not None:
            disaggregated_params["release"] = {
                "coordinator_addr": self.config.coordinator_addr,
                "request_id": int(prefill_request_id),
            }

        # PrefillRouter looks at top-level disaggregated_params; the rest of
        # the dict mirrors the frontend's expected ChatCompletion-ish shape so
        # this can also be debugged directly via curl.
        response: dict[str, Any] = {
            "token_ids": [],
            "disaggregated_params": disaggregated_params,
            "finish_reason": "stop",
        }
        yield response
