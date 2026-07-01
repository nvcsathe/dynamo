from __future__ import annotations

from types import SimpleNamespace

import pytest

from dynamo.megatron.args import parse_args
from dynamo.megatron.main import (
    _logical_replica_group,
    _torchrun_command,
    _under_torchrun,
)


def test_parse_args_splits_dynamo_and_megatron_arguments():
    config = parse_args(
        [
            "--role",
            "aggregated",
            "--model",
            "model-meta",
            "--nproc-per-node",
            "2",
            "--",
            "--load",
            "/checkpoints/model",
            "--tensor-model-parallel-size",
            "2",
        ]
    )
    assert config.component == "backend"
    assert config.nproc_per_node == 2
    assert config.megatron_argv == [
        "--load",
        "/checkpoints/model",
        "--tensor-model-parallel-size",
        "2",
    ]


def test_disaggregated_role_requires_transfer_and_coordinator_addresses():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--role",
                "prefill",
                "--model",
                "model-meta",
                "--nproc-per-node",
                "1",
                "--",
                "--load",
                "/checkpoint",
            ]
        )

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--role",
                "decode",
                "--model",
                "model-meta",
                "--nproc-per-node",
                "1",
                "--kv-transfer-listen-addr",
                "127.0.0.1:7000",
                "--",
                "--load",
                "/checkpoint",
            ]
        )


def test_torchrun_command_reexecutes_module():
    command = _torchrun_command(["--model", "m"], 4)
    assert command[1:4] == ["-m", "torch.distributed.run", "--standalone"]
    assert "--nproc-per-node=4" in command
    assert command[-3:] == ["dynamo.megatron", "--model", "m"]


def test_recursion_guard_accepts_internal_or_external_torchrun_env():
    assert _under_torchrun({"DYNAMO_MEGATRON_RANK_PROCESS": "1"})
    assert _under_torchrun({"RANK": "0", "LOCAL_RANK": "0"})
    assert not _under_torchrun({})


def test_logical_replica_group_uses_expert_dp_for_nano_ep_topology():
    groups = SimpleNamespace(dp=object(), expt_dp=object())

    nano_args = SimpleNamespace(expert_model_parallel_size=2)
    dense_args = SimpleNamespace(expert_model_parallel_size=1)

    nano_group = _logical_replica_group(nano_args, groups)
    group_sizes = {groups.dp: 2, groups.expt_dp: 1}

    assert nano_group is groups.expt_dp
    assert group_sizes[nano_group] == 1
    assert _logical_replica_group(dense_args, groups) is groups.dp
