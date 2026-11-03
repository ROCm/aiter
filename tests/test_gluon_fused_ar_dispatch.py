# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import aiter
from aiter.dist.device_communicators.communicator_cuda import (
    CudaCommunicator,
    _GLUON_AR_MAX_BYTES,
    _supports_gluon_fused_ar_hidden,
)

_GLUON_MODULE = "aiter.ops.triton.comms.fused.all_reduce_rmsnorm_quant"


@contextmanager
def _fake_gluon(return_value):
    module = ModuleType(_GLUON_MODULE)
    launch = Mock(return_value=return_value)
    module.all_reduce_rmsnorm_quant = launch
    with patch.dict(sys.modules, {_GLUON_MODULE: module}):
        yield launch


def _inputs(rows=2, hidden=6144, dtype=torch.bfloat16):
    x = torch.zeros((rows, hidden), dtype=dtype)
    residual = torch.zeros_like(x)
    weight = torch.zeros(hidden, dtype=dtype)
    return x, residual, weight


def test_gluon_fused_ar_hidden_policy():
    assert _supports_gluon_fused_ar_hidden(256)
    assert _supports_gluon_fused_ar_hidden(2048)
    assert _supports_gluon_fused_ar_hidden(6144)
    assert not _supports_gluon_fused_ar_hidden(128)
    assert not _supports_gluon_fused_ar_hidden(6143)
    assert not _supports_gluon_fused_ar_hidden(7168)


def test_hidden_6144_padding_preserves_reductions_and_fits_inbox():
    hidden_pad = 1 << (6144 - 1).bit_length()
    assert hidden_pad == 8192

    valid = torch.ones(6144)
    valid[::3] = -2.0
    padded = torch.cat((valid, torch.zeros(hidden_pad - valid.numel())))
    assert torch.equal(padded[:6144], valid)
    assert padded[6144:].count_nonzero() == 0
    assert padded.square().sum() / 6144 == valid.square().mean()
    assert padded.abs().max() == valid.abs().max()

    max_rows = _GLUON_AR_MAX_BYTES // (6144 * 2)
    triples = (max_rows + 2) // 3
    chunks = (_GLUON_AR_MAX_BYTES + 8 * 64 - 1) // (8 * 64)
    slot_words = 4 * 64 * chunks
    assert 2 * hidden_pad * triples <= slot_words


@pytest.mark.parametrize("rows", [1, 4, 16, 64])
@pytest.mark.parametrize("hidden", [2048, 6144])
def test_unquantized_dispatches_to_one_gluon_call(rows, hidden):
    x, residual, weight = _inputs(rows, hidden)
    expected = (object(), object())
    comm = SimpleNamespace(
        gluon_ar=("inboxes", "calls"),
        cpu_group="cpu_group",
    )

    with _fake_gluon(expected) as launch:
        result = CudaCommunicator.fused_allreduce_rmsnorm(
            comm, x, residual, weight, 1e-6, gemma_norm=True
        )

    assert result is expected
    launch.assert_called_once_with(
        x,
        residual,
        weight,
        1e-6,
        True,
        False,
        "cpu_group",
        "inboxes",
        "calls",
    )


@pytest.mark.parametrize("rows", [1, 4, 16, 64])
@pytest.mark.parametrize("emit_bf16", [False, True])
def test_per_token_quant_dispatches_to_one_gluon_call(rows, emit_bf16):
    x, residual, weight = _inputs(rows)
    q, res_out, scale, norm = object(), object(), object(), object()
    kernel_result = (norm, res_out, q, scale) if emit_bf16 else (q, res_out, scale)
    expected = (q, res_out, scale, norm) if emit_bf16 else kernel_result
    comm = SimpleNamespace(
        gluon_ar=("inboxes", "calls"),
        cpu_group="cpu_group",
    )

    with _fake_gluon(kernel_result) as launch:
        result = CudaCommunicator.fused_allreduce_rmsnorm_quant(
            comm,
            x,
            residual,
            weight,
            1e-6,
            quant_type="per_token",
            emit_bf16=emit_bf16,
            gemma_norm=True,
        )

    assert result == expected
    launch.assert_called_once_with(
        x,
        residual,
        weight,
        1e-6,
        True,
        True,
        "cpu_group",
        "inboxes",
        "calls",
        per_token_quant=True,
        return_norm=emit_bf16,
    )


def test_unquantized_unsupported_hidden_keeps_split_fallback():
    x, residual, weight = _inputs(hidden=7168)
    ar_out = x.clone()
    comm = SimpleNamespace(
        gluon_ar=("inboxes", "calls"),
        ca_comm=None,
        qr_comm=None,
        world_size=4,
        _ar_1stage_override=None,
        _ar_1stage_max_kb=-1,
        all_reduce=Mock(return_value=ar_out),
    )

    def fake_rmsnorm(out, ar, res_in, res_out, w, eps, gemma, _):
        assert ar is ar_out
        assert res_in is residual
        assert w is weight
        assert eps == 1e-6
        assert gemma is True
        out.copy_(ar)
        res_out.copy_(res_in)

    with patch.object(aiter, "rmsnorm2d_fwd_with_add", fake_rmsnorm):
        out, res_out = CudaCommunicator.fused_allreduce_rmsnorm(
            comm, x, residual, weight, 1e-6, gemma_norm=True
        )

    comm.all_reduce.assert_called_once_with(x, prefill_support=False)
    assert torch.equal(out, ar_out)
    assert torch.equal(res_out, residual)


def test_per_token_quant_without_gluon_keeps_existing_fallback():
    x, residual, weight = _inputs()
    norm = x.clone()
    res_out = residual.clone()
    q, scale = object(), object()
    quant = Mock(return_value=(q, scale))
    fused_norm = Mock(return_value=(norm, res_out))
    comm = SimpleNamespace(
        gluon_ar=None,
        ca_comm=None,
        qr_comm=None,
        _ar_quant_max_bytes=-1,
        _ar_quant_no_prefill_max_bytes=-1,
        fused_allreduce_rmsnorm=fused_norm,
    )

    with patch(
        "aiter.dist.device_communicators.communicator_cuda.get_hip_quant",
        return_value=quant,
    ):
        result = CudaCommunicator.fused_allreduce_rmsnorm_quant(
            comm, x, residual, weight, 1e-6, gemma_norm=True
        )

    assert result == (q, res_out, scale)
    fused_norm.assert_called_once_with(
        x, residual, weight, 1e-6, False, gemma_norm=True
    )
    quant.assert_called_once()
    assert quant.call_args.args == (norm,)
