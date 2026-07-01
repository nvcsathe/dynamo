from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.monitoring.worker_info import build_worker_info_from_defaults


def test_megatron_backend_uses_unified_worker_component_names():
    config = PlannerConfig(
        backend="megatron",
        pre_deployment_sweeping_mode="none",
        optimization_target="load",
        prefill_scale_up_queue_tokens=10000,
        prefill_scale_down_queue_tokens=2000,
        decode_scale_up_kv_rate=80,
        decode_scale_down_kv_rate=40,
    )
    prefill = build_worker_info_from_defaults("megatron", SubComponentType.PREFILL)
    decode = build_worker_info_from_defaults("megatron", SubComponentType.DECODE)

    assert config.backend == "megatron"
    assert (prefill.k8s_name, prefill.component_name) == (
        "MegatronPrefillWorker",
        "prefill",
    )
    assert (decode.k8s_name, decode.component_name) == (
        "MegatronDecodeWorker",
        "backend",
    )
