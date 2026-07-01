# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from megatron.core.inference.apis import MegatronAsyncLLM
    from megatron.core.inference.sampling_params import SamplingParams

from dynamo._core import Context
from dynamo.common.backend.disagg import require_prefill_result
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
)
from dynamo.common.backend.health_check import (
    bos_token_id_or,
    build_health_check_payload,
    is_probe,
)
from dynamo.common.backend.publisher import ComponentSnapshot, KvEventSource, PushSource
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode
from dynamo.llm import KvEventPublisher, ModelInput
from dynamo.megatron.args import Config

logger = logging.getLogger(__name__)


def build_sampling_params(request: GenerateRequest) -> SamplingParams:
    from megatron.core.inference.sampling_params import SamplingParams

    sampling = request.get("sampling_options") or {}
    stop = request.get("stop_conditions") or {}
    params = SamplingParams()
    if sampling.get("temperature") is not None:
        params.temperature = float(sampling["temperature"])
    if sampling.get("top_p") is not None:
        params.top_p = float(sampling["top_p"])
    if sampling.get("top_k") is not None:
        params.top_k = max(0, int(sampling["top_k"]))
    if stop.get("max_tokens") is not None:
        params.num_tokens_to_generate = int(stop["max_tokens"])
    if stop.get("stop"):
        params.stop_words = list(stop["stop"])
    stop_token_ids = list(stop.get("stop_token_ids") or [])
    if stop.get("ignore_eos"):
        params.termination_id = -1
    elif stop_token_ids:
        params.termination_id = int(stop_token_ids[0])
        if len(stop_token_ids) > 1:
            logger.warning(
                "Megatron supports one termination token; ignoring %d additional IDs",
                len(stop_token_ids) - 1,
            )
    output = request.get("output_options") or {}
    token_logprobs = output.get("logprobs")
    prompt_logprobs = output.get("prompt_logprobs")
    params.top_n_logprobs = max(
        int(token_logprobs or 0), int(prompt_logprobs or 0)
    )
    params.return_log_probs = token_logprobs is not None
    params.skip_prompt_log_probs = prompt_logprobs is None
    params.add_attributes({})
    if params.temperature == 0.0:
        params.top_k = 1
        params.top_p = 0.0
    return params


class MegatronLLMEngine(LLMEngine):
    """Unified Dynamo backend for one self-owned Megatron DP replica."""

    def __init__(
        self,
        config: Config,
        initialize_llm: Callable[[], Awaitable["MegatronAsyncLLM"]],
    ) -> None:
        self.config = config
        self._initialize_llm = initialize_llm
        self.llm: Optional["MegatronAsyncLLM"] = None
        self._snapshot_publisher: Any = None
        self._kv_queue: queue.Queue[tuple[str, dict]] = queue.Queue()
        self._publisher_stop = threading.Event()
        self._publisher_thread: Optional[threading.Thread] = None
        self._release_clients: dict[str, Any] = {}
        self._request_ids: dict[str, int] = {}

    @classmethod
    async def from_args(cls, argv=None):
        raise RuntimeError(
            "MegatronLLMEngine must be created by the distributed "
            "dynamo.megatron entrypoint"
        )

    @staticmethod
    def worker_config(config: Config) -> WorkerConfig:
        mode = {
            "aggregated": DisaggregationMode.AGGREGATED,
            "prefill": DisaggregationMode.PREFILL,
            "decode": DisaggregationMode.DECODE,
        }[config.role]
        return WorkerConfig(
            namespace=config.namespace,
            component=config.component,
            endpoint=config.endpoint,
            model_name=config.model,
            served_model_name=config.served_model_name,
            model_input=ModelInput.Tokens,
            discovery_backend=config.discovery_backend,
            request_plane=config.request_plane,
            event_plane=config.event_plane,
            disaggregation_mode=mode,
            enable_kv_routing=True,
        )

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id
        self.llm = await self._initialize_llm()
        metadata = self.llm.metadata
        if self.config.role != "decode" and self.llm.context.enable_prefix_caching:
            self.llm.add_kv_event_listener(self._on_kv_event)
        self.llm.add_metrics_listener(self._on_metrics)
        return EngineConfig(
            model=self.config.model,
            served_model_name=self.config.served_model_name,
            context_length=metadata.context_length,
            kv_cache_block_size=metadata.block_size_tokens,
            total_kv_blocks=metadata.total_kv_blocks,
            max_num_seqs=metadata.max_requests,
            max_num_batched_tokens=metadata.max_tokens,
            data_parallel_size=1,
            data_parallel_start_rank=0,
            runtime_data={"role": self.config.role},
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        if self.llm is None:
            raise RuntimeError("Megatron engine is not initialized")
        token_ids = list(request.get("token_ids") or [])
        if not token_ids:
            raise ValueError("Megatron backend requires token_ids")
        params = build_sampling_params(request)
        n = int((request.get("sampling_options") or {}).get("n") or 1)
        if n != 1:
            raise ValueError("Megatron Dynamo backend currently supports sampling n=1")
        context_id = str(context.id())

        def remember_request(request_id: int) -> None:
            self._request_ids[context_id] = request_id

        # Health probes exercise the local model without cross-worker handoff.
        if is_probe(request):
            stream = self.llm.generate_stream(
                token_ids, params, on_request_started=remember_request
            )
            try:
                async for chunk in self._stream_chunks(stream, token_ids, params):
                    yield chunk
            finally:
                self._request_ids.pop(context_id, None)
            return

        if self.config.role == "prefill":
            try:
                result = await self.llm.prefill_for_handoff(
                    token_ids, params, on_request_started=remember_request
                )
                disagg = dict(result.disaggregated_params or {})
                disagg["release"] = {
                    "coordinator_addr": self.llm.metadata.coordinator_address,
                    "request_id": int(disagg.get("request_id", result.request_id)),
                }
                yield {
                    "token_ids": [],
                    "index": 0,
                    "finish_reason": "stop",
                    "completion_usage": {
                        "prompt_tokens": len(token_ids),
                        "completion_tokens": 0,
                        "total_tokens": len(token_ids),
                    },
                    "disaggregated_params": disagg,
                }
            finally:
                self._request_ids.pop(context_id, None)
            return

        stream = None
        release: dict[str, Any] = {}
        if self.config.role == "decode":
            prefill = require_prefill_result(request, DisaggregationMode.DECODE)
            disagg = prefill.get("disaggregated_params") or {}
            release = disagg.get("release") or {}
            stream = self.llm.generate_stream_with_kv_handoff(
                token_ids,
                params,
                disagg.get("kv_meta") or {},
                list(disagg.get("block_ids") or []),
                on_request_started=remember_request,
            )
        else:
            stream = self.llm.generate_stream(
                token_ids, params, on_request_started=remember_request
            )

        released = False
        try:
            async for chunk in self._stream_chunks(stream, token_ids, params):
                if not released and self.config.role == "decode":
                    address = release.get("coordinator_addr")
                    request_id = release.get("request_id")
                    if address is not None and request_id is not None:
                        self._release_remote_handoff(str(address), int(request_id))
                        released = True
                yield chunk
        finally:
            self._request_ids.pop(context_id, None)

    async def _stream_chunks(
        self, stream, prompt_token_ids: list[int], params: SamplingParams
    ) -> AsyncGenerator[GenerateChunk, None]:
        completion_tokens = 0
        async for output in stream:
            completion_tokens += len(output.token_ids)
            chunk: GenerateChunk = {"token_ids": output.token_ids, "index": 0}
            if output.finished:
                max_tokens = params.num_tokens_to_generate
                chunk["finish_reason"] = (
                    "length"
                    if max_tokens is not None and completion_tokens >= max_tokens
                    else "stop"
                )
                chunk["completion_usage"] = {
                    "prompt_tokens": len(prompt_token_ids),
                    "completion_tokens": completion_tokens,
                    "total_tokens": len(prompt_token_ids) + completion_tokens,
                }
            yield chunk

    def _release_remote_handoff(self, address: str, request_id: int) -> None:
        from megatron.core.inference.inference_client import InferenceClient

        client = self._release_clients.get(address)
        if client is None:
            client = InferenceClient(address, deserialize=False)
            client.start()
            self._release_clients[address] = client
        client.release_handoff(request_id)

    async def abort(self, context: Context) -> None:
        request_id = self._request_ids.pop(str(context.id()), None)
        if request_id is not None and self.llm is not None:
            await self.llm.abort(request_id)

    async def drain(self) -> None:
        if self.llm is None:
            return
        deadline = asyncio.get_running_loop().time() + self.config.drain_timeout
        while self.llm.active_request_count or self.llm.pinned_handoff_count:
            if asyncio.get_running_loop().time() >= deadline:
                logger.warning("Timed out draining Megatron requests and KV handoffs")
                return
            await asyncio.sleep(0.05)

    async def cleanup(self) -> None:
        self._publisher_stop.set()
        if self._publisher_thread is not None:
            self._publisher_thread.join(timeout=2)
            self._publisher_thread = None
        for client in self._release_clients.values():
            client.stop()
        self._release_clients.clear()
        if self.llm is not None:
            await self.llm.shutdown()
            self.llm = None

    async def health_check_payload(self) -> dict[str, Any] | None:
        if self.llm is None:
            return None
        return build_health_check_payload(bos_token_id_or(self.llm.controller.tokenizer))

    async def kv_event_sources(self) -> list[KvEventSource]:
        if self.config.role == "decode" or self.llm is None:
            return []
        if not self.llm.context.enable_prefix_caching:
            return []
        return [PushSource(on_ready=self._start_publisher_thread, dp_rank=0)]

    def component_metrics_dp_ranks(self) -> list[int]:
        return [0]

    def attach_snapshot_publisher(self, publisher: Any) -> None:
        self._snapshot_publisher = publisher

    def _on_kv_event(self, kind: str, payload: dict) -> None:
        self._kv_queue.put((kind, payload))

    def _start_publisher_thread(self, publisher: KvEventPublisher) -> None:
        self._publisher_thread = threading.Thread(
            target=self._publish_loop,
            args=(publisher,),
            daemon=True,
            name="megatron-kv-publisher",
        )
        self._publisher_thread.start()

    def _publish_loop(self, publisher: KvEventPublisher) -> None:
        while not self._publisher_stop.is_set():
            try:
                kind, payload = self._kv_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if kind == "stored":
                    publisher.publish_stored(**payload)
                elif kind == "removed":
                    publisher.publish_removed(**payload)
                elif kind == "cleared":
                    publisher.publish_all_cleared()
            except Exception:
                logger.exception("Failed to publish Megatron KV event %s", kind)

    def _on_metrics(self, stats: dict) -> None:
        if self._snapshot_publisher is None:
            return
        total = int(stats.get("total_blocks") or 0)
        used = int(stats.get("allocated_blocks") or 0)
        self._snapshot_publisher.publish(
            0,
            ComponentSnapshot(
                kv_used_blocks=used,
                kv_total_blocks=total,
                gpu_cache_usage=float(stats.get("allocated_utilization") or 0.0),
                kv_cache_hit_rate=stats.get("prefix_cache_hit_rate"),
                dp_rank=0,
                active_requests=int(stats.get("active_request_count") or 0),
                waiting_requests=int(stats.get("waiting_request_count") or 0),
            ),
        )
