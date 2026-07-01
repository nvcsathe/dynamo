# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Self-launching distributed entrypoint for the Dynamo Megatron backend."""

from __future__ import annotations

import os
import sys

import torch
import uvloop

from dynamo.common.backend.worker import Worker
from dynamo.megatron.args import Config, parse_args
from dynamo.megatron.llm_engine import MegatronLLMEngine

_CHILD_ENV = "DYNAMO_MEGATRON_RANK_PROCESS"


def _under_torchrun(env: dict[str, str] | None = None) -> bool:
    values = os.environ if env is None else env
    return values.get(_CHILD_ENV) == "1" or (
        "RANK" in values and "LOCAL_RANK" in values
    )


def _torchrun_command(argv: list[str], nproc_per_node: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={nproc_per_node}",
        "--module",
        "dynamo.megatron",
        *argv,
    ]


def _bootstrap(argv: list[str]) -> None:
    config = parse_args(argv)
    if not os.path.isdir(config.megatron_root):
        raise FileNotFoundError(f"Megatron root does not exist: {config.megatron_root}")
    os.chdir(config.megatron_root)
    env = os.environ.copy()
    env[_CHILD_ENV] = "1"
    os.execvpe(
        sys.executable,
        _torchrun_command(argv, config.nproc_per_node),
        env,
    )


def _extra_megatron_args(parser):
    from megatron.inference.utils import add_inference_args
    from megatron.post_training.arguments import add_modelopt_args

    return add_inference_args(add_modelopt_args(parser))


def _logical_replica_group(args, pg_collection):
    """Return the group that counts independently complete model replicas.

    Regular DP includes EP-overlapped ranks because dense layers are replicated
    across them. For MoE models, expert-DP instead counts complete expert sets.
    """
    if getattr(args, "expert_model_parallel_size", 1) > 1:
        return pg_collection.expt_dp
    return pg_collection.dp


async def _initialize_llm(config: Config):
    from megatron.core.inference.apis import MegatronAsyncLLM
    from megatron.core.utils import get_pg_size
    from megatron.inference.utils import (
        get_inference_config_from_model_and_args,
        get_model_for_inference,
    )
    from megatron.training import get_args
    from megatron.training.arguments import parse_and_validate_args
    from megatron.training.initialize import initialize_megatron
    from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer

    original_argv = sys.argv
    try:
        sys.argv = ["dynamo.megatron", *config.megatron_argv]
        parse_and_validate_args(
            extra_args_provider=_extra_megatron_args,
            args_defaults={"no_load_rng": True, "no_load_optim": True},
        )
    finally:
        sys.argv = original_argv

    initialize_megatron()
    args = get_args()
    args.return_log_probs = True
    model = get_model_for_inference()
    tokenizer = build_tokenizer(args)
    inference_config = get_inference_config_from_model_and_args(model, args)
    llm = MegatronAsyncLLM(
        model=model,
        tokenizer=tokenizer,
        inference_config=inference_config,
        use_coordinator=True,
        coordinator_host=config.coordinator_host,
        coordinator_port=config.coordinator_port,
    )
    try:
        replica_group = _logical_replica_group(args, llm.engine.pg_collection)
        replica_count = get_pg_size(replica_group)
        if replica_count != 1:
            raw_dp_size = get_pg_size(llm.engine.pg_collection.dp)
            raise ValueError(
                "dynamo.megatron requires one complete model replica per worker; "
                f"got logical DP={replica_count}, regular DP={raw_dp_size}, "
                f"EP={args.expert_model_parallel_size}"
            )
        if config.role in ("prefill", "decode"):
            llm.engine.setup_kv_transfer(
                role=config.role,
                listen_addr=config.kv_transfer_listen_addr,
            )
        return llm
    except BaseException:
        await llm.shutdown()
        raise


async def _distributed_worker(config: Config) -> None:
    rank = int(os.environ.get("RANK", "0"))
    llm = None
    if rank == 0:
        engine = MegatronLLMEngine(config, lambda: _initialize_llm(config))
        worker = Worker(engine, MegatronLLMEngine.worker_config(config))
        try:
            await worker.run()
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        return

    try:
        llm = await _initialize_llm(config)
        await llm.wait_for_shutdown()
    finally:
        if llm is not None:
            await llm.shutdown()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not _under_torchrun():
        _bootstrap(argv)
        return
    config = parse_args(argv)
    uvloop.run(_distributed_worker(config))


if __name__ == "__main__":
    main()
