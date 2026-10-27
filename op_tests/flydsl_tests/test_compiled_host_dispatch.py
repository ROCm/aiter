# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from unittest.mock import patch

import flydsl.compiler as flyc
import flydsl.expr as fx
import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.mega_moe import quant
from aiter.ops.flydsl.kernels.mqa_logits import qsa_flydsl_layout as qsa
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="requires gfx950 MX quantization",
)


@pytest.mark.parametrize("scale_low", [16, 96])
def test_scale_sort_reuses_cache_across_buffer_ranks(monkeypatch, scale_low):
    monkeypatch.setattr(quant, "_SCALE_SORT_CACHE", {})
    generator = torch.Generator(device="cuda").manual_seed(6946)
    compiled = None
    with patch.object(flyc, "compile", wraps=flyc.compile) as compile_spy:
        for tokens, flat in ((2, False), (3, True), (2, False)):
            scale = torch.randint(
                scale_low,
                scale_low + 16,
                (tokens, 8),
                dtype=torch.uint8,
                device="cuda",
                generator=generator,
            )
            ids = torch.arange(32, dtype=torch.int32, device="cuda") % tokens
            valid = torch.tensor([32], dtype=torch.int32, device="cuda")
            output = torch.empty((32, 8), dtype=torch.uint8, device="cuda")
            # Every argument is a raw buffer in this kernel. Different shaped
            # aliases must keep the same byte interpretation and compiled ABI.
            quant.mxfp4_moe_scale_sort(
                output if flat else output.view(32, 2, 4),
                scale.view(-1) if flat else scale,
                ids if flat else ids.view(16, 2),
                valid if flat else valid.view(1, 1),
                tokens,
                256,
            )
            torch.cuda.synchronize()
            # n32k4 scale layout: columns grouped by four, 16-row halves
            # interleaved in each dword, then the second four columns.
            expected = scale[ids.long()].view(2, 16, 2, 4)
            expected = expected.permute(3, 1, 2, 0).contiguous().view(32, 8)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            launcher = quant._SCALE_SORT_CACHE[256]
            if compiled is None:
                compiled = launcher._cf
            assert launcher._cf is compiled
        assert compile_spy.call_count == 1


def test_qsa_reuses_cache_across_metadata_ranks_and_divisors(monkeypatch):
    monkeypatch.setattr(
        qsa._qsa_logits_layout_launch, "_cf_by_specialization", {}, raising=False
    )
    generator = torch.Generator(device="cuda").manual_seed(6946)
    with patch.object(flyc, "compile", wraps=flyc.compile) as compile_spy:
        for rows, shaped, divisor in (
            (300, True, 8.0),
            (513, False, 8.0),
            (300, False, 16.0),
            (513, True, 8.0),
        ):
            columns = rows // 4
            pages = (columns + 15) // 16
            q = torch.randn(
                (rows, 4, 128),
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            k = torch.randn(
                (pages, 16, 1, 128),
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            table = torch.arange(pages, dtype=torch.int32, device="cuda").view(1, -1)
            requests = torch.zeros(rows, dtype=torch.int32, device="cuda")
            positions = torch.arange(rows, dtype=torch.int32, device="cuda")
            contexts = torch.tensor([rows], dtype=torch.int32, device="cuda")
            logits = torch.empty((rows, columns), dtype=torch.float32, device="cuda")
            visible = torch.empty(rows, dtype=torch.int32, device="cuda")
            starts = torch.empty_like(visible)
            qsa.qsa_paged_mqa_logits_flydsl(
                q,
                k,
                table,
                requests.view(-1, 1) if shaped else requests,
                positions.view(-1, 1) if shaped else positions,
                contexts.view(1, 1) if shaped else contexts,
                4,
                divisor,
                logits,
                visible.view(-1, 1) if shaped else visible,
                starts.view(-1, 1) if shaped else starts,
            )
            torch.cuda.synchronize()
            keys = k.view(-1, 128)[:columns].float().T
            expected = sum((q[:, h].float() @ keys).relu() for h in range(4))
            expected /= divisor
            bounds = (positions + 1) // 4
            mask = torch.arange(columns, device="cuda")[None, :] < bounds[:, None]
            torch.testing.assert_close(
                logits[mask], expected[mask], rtol=0.005, atol=0.005
            )
            torch.testing.assert_close(visible, bounds, rtol=0, atol=0)
            assert torch.count_nonzero(starts) == 0
        assert compile_spy.call_count == 2


@pytest.mark.parametrize("key", [None, ("configuration",)])
@pytest.mark.parametrize("compile_only", ["1", "true"])
def test_compile_only_does_not_dispatch_a_warm_callable(monkeypatch, key, compile_only):
    launcher = quant.build_per_1x32_mx_quant_module(256, "fp4")
    x = torch.randn((2, 256), dtype=torch.bfloat16, device="cuda")
    y = torch.empty((2, 128), dtype=torch.uint8, device="cuda")
    scales = torch.empty((2, 8), dtype=torch.uint8, device="cuda")
    args = x, y, scales, 2, 1, fx.Stream(torch.cuda.current_stream().cuda_stream)
    _run_compiled(launcher, *args, specialization_key=key)
    torch.cuda.synchronize()
    compiled = launcher._cf if key is None else launcher._cf_by_specialization[key]

    def fail_dispatch(_):
        pytest.fail("compile-only invoked the cached GPU dispatcher")

    with monkeypatch.context() as mode:
        mode.setenv("COMPILE_ONLY", compile_only)
        mode.setattr(compiled._call_state._tls, "dispatch", fail_dispatch)
        _run_compiled(launcher, *args, specialization_key=key)
    assert (
        launcher._cf if key is None else launcher._cf_by_specialization[key]
    ) is compiled
    # The execution entry survives compile-only and can dispatch again.
    y.fill_(255)
    _run_compiled(launcher, *args, specialization_key=key)
    torch.cuda.synchronize()
    assert torch.any(y != 255)
