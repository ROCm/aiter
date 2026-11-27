# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which gfx1250 TP4 kernel takes fused_allreduce_rmsnorm(_quant): the Gluon
all_reduce_rmsnorm_per_token_quant, the Lamport one-shot (FP8 in the same kernel),
the Gluon all_reduce_rmsnorm_quant, the barrier fused kernel or the split path. No
GPU or process group is needed: the production gates run on a stub communicator and
the kernels are mocks.
"""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from aiter.dist.device_communicators import communicator_cuda as cc
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

CALLS = torch.zeros(2 * 8192 + 1, dtype=torch.int32)  # all_reduce_init(4 MiB)
CAP = cc._GLUON_AR_NORM_MAX_TOKENS


@pytest.mark.parametrize(
    "case,m,n,lamport,expected",
    [
        ("ok", 16, 6152, False, True),
        ("ok", CAP, 6144, True, True),
        ("ok", CAP + 1, 6144, True, False),  # Lamport is faster past CAP rows
        ("ok", CAP + 1, 6144, False, True),
        ("ok", 340, 6152, False, True),  # 4 MiB
        ("ok", 341, 6152, False, False),
        ("ok", 16, 6148, False, False),  # hidden % 8
        ("ok", 1, 32768, False, False),  # more than 32 waves
        ("ok", 8192, 8, False, True),  # every chunk of the inboxes
        ("ok", 8193, 8, False, False),
        ("fp16", 16, 6152, False, False),
        ("strided", 16, 6152, False, False),
        ("off", 16, 6152, False, False),
        ("misaligned", 16, 6152, False, True),  # copied: no rank leaves the path
    ],
)
def test_gluon_ar_norm_gates(case, m, n, lamport, expected):
    dtype = torch.float16 if case == "fp16" else torch.bfloat16
    x = torch.zeros(m * n + 1, dtype=dtype)[1:].view(m, n)
    if case != "misaligned":
        x = torch.zeros((m, n), dtype=dtype)
    residual = torch.zeros((m, 2 * n), dtype=dtype)[:, ::2]
    if case != "strided":
        residual = torch.zeros_like(x)
    comm = SimpleNamespace(gluon_ar_plain=None if case == "off" else ("in", CALLS))
    weight = torch.ones(n, dtype=dtype)
    assert cc._use_gluon_ar_norm(comm, x, residual, weight, lamport) == expected


@pytest.mark.parametrize(
    "quant,m,n,emit_bf16,gluon,expected",
    [
        (False, 1, 6144, False, True, "gluon_norm"),
        (False, CAP, 6144, False, True, "gluon_norm"),
        (False, CAP + 1, 6144, False, True, "lamport"),
        (False, 16, 6144, False, False, "lamport"),
        (False, 16, 4096, False, True, "gluon_fused"),  # power-of-2 hidden
        (False, 340, 6152, False, True, "gluon_norm"),  # not a Lamport hidden
        (False, 341, 6152, False, True, "custom_fused"),  # past 4 MiB
        (True, 1, 6144, False, True, "gluon_norm"),
        (True, CAP + 1, 6144, False, True, "lamport"),
        (True, 16, 6144, True, True, "lamport"),  # only Lamport has the bf16 out
        (True, 16, 6144, False, False, "lamport"),
        (True, 16, 4096, False, True, "gluon_norm"),
        (True, 682, 6144, False, True, "lamport"),  # 8 MiB, the TP4 Lamport cap
        (True, 683, 6144, False, True, "custom_fused"),
        (True, 513, 8192, False, True, "split"),
        (True, 341, 6152, False, True, "split"),
    ],
)
def test_ar_norm_precedence(quant, m, n, emit_bf16, gluon, expected):
    x = torch.zeros((m, n), dtype=torch.bfloat16)
    weight = torch.ones(n, dtype=torch.bfloat16)
    scale = torch.ones((m, 1))
    ca = SimpleNamespace(
        disabled=False,
        has_old_arch_fused_kernels=False,
        _is_gfx1250=True,
        _lamport_max_bytes=8 << 20,
        _fly_rms=None,
        should_custom_ar=lambda *args: True,
        lamport_fused_ar_rms=Mock(return_value=(x, x)),
        lamport_fused_ar_rms_quant=Mock(
            side_effect=lambda *args: (x, x, scale, x)[: 4 if args[-1] else 3]
        ),
        custom_fused_ar_rms=Mock(return_value=(x, x)),
        custom_fused_ar_rms_quant=Mock(return_value=(x, x, scale)),
    )
    for name in (
        "can_fuse_ar_rms",
        "should_fuse_ar_rms",
        "should_lamport_ar_rms",
        "should_fly_allreduce_rmsnorm",
    ):
        setattr(ca, name, MethodType(getattr(CustomAllreduce, name), ca))
    comm = SimpleNamespace(
        ca_comm=ca,
        qr_comm=None,
        world_size=4,
        cpu_group="group",
        gluon_ar=("fused_inboxes", "fused_calls") if gluon else None,
        gluon_ar_plain=("inboxes", CALLS) if gluon else None,
        _ar_quant_max_bytes=-1,
        _ar_quant_no_prefill_max_bytes=-1,
        _ar_1stage_max_bytes=-1,
        _ar_1stage_override=None,
        all_reduce=Mock(return_value=x),
        fused_allreduce_rmsnorm=Mock(return_value=(x, x)),
    )
    epilogue = Mock(return_value=(x, x, scale) if quant else (x, x))
    gluon_fused = Mock(return_value=(x, x))
    with patch(
        "aiter.ops.triton.comms.all_reduce.all_reduce_rmsnorm_per_token_quant",
        epilogue,
    ), patch(
        "aiter.ops.triton.comms.fused.all_reduce_rmsnorm_quant.all_reduce_rmsnorm_quant",
        gluon_fused,
    ), patch(
        "aiter.rmsnorm2d_fwd_with_add"
    ) as split_norm, patch.object(
        cc, "get_hip_quant", return_value=Mock(return_value=(x, scale))
    ):
        if quant:
            result = cc.CudaCommunicator.fused_allreduce_rmsnorm_quant(
                comm, x, x, weight, 1e-6, emit_bf16=emit_bf16
            )
            assert len(result) == (4 if emit_bf16 else 3)
        else:
            cc.CudaCommunicator.fused_allreduce_rmsnorm(comm, x, x, weight, 1e-6)
    taken = {
        "gluon_norm": epilogue.called,
        "gluon_fused": gluon_fused.called,
        "lamport": ca.lamport_fused_ar_rms.called
        or ca.lamport_fused_ar_rms_quant.called,
        "custom_fused": ca.custom_fused_ar_rms.called
        or ca.custom_fused_ar_rms_quant.called,
        "split": split_norm.called or comm.fused_allreduce_rmsnorm.called,
    }
    assert taken == {path: path == expected for path in taken}, taken
    if expected == "gluon_norm":
        assert epilogue.call_args.args[3:] == (1e-6, False)
        kwargs = epilogue.call_args.kwargs
        assert kwargs.pop("calls") is CALLS
        assert kwargs == {"quant": quant, "group": "group", "inboxes": "inboxes"}
    if expected == "lamport" and quant:
        args = ca.lamport_fused_ar_rms_quant.call_args.args
        assert args[3:] == (1e-6, False, emit_bf16)
