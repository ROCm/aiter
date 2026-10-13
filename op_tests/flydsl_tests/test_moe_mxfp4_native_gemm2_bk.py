# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Native MXMOE gemm2 at BK=128 and on pre-padded (``w2.inter_real``) weights.

Kimi-K3 TP8 has inter_dim=384, which only the BK=128 native gemm2 can contract
unpadded. Pre-padded weights (inter_dim 384 -> 512, ``w2.inter_real=384``) take
the native gemm2 with ``D_INTER_REAL``, which must skip the K-pad halves.

Every case runs the same native gemm1, so the stage1 activation quantization is
shared and only gemm2 differs. The v2 layout gemm2 on the unpadded weights is the
baseline; the torch reference guards against both being wrong together.

Run:
    pytest op_tests/flydsl_tests/test_moe_mxfp4_native_gemm2_bk.py -q
"""

import pytest
import torch
import torch.nn.functional as F

import aiter
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, torch_moe_stage1, torch_moe_stage2
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import e8m0_shuffle

_SKIP = pytest.mark.skipif(
    get_gfx() != "gfx950", reason="native MXMOE kernels require gfx950"
)

# Kimi-K3 TP8: the MXMOE aux kernels are codegen'd per (E, topk, model_dim).
TOKEN, MODEL_DIM, INTER_DIM, PADDED_INTER, E, TOPK, BM = 8, 3584, 384, 512, 896, 16, 16
# Inline (f16in) activation quant, as in the Kimi-K3 tuned rows.
KN1 = f"flydsl_mxmoe_g1_a4w4_{BM}x128x256_f16in_hpf_nt_situv2"
KN2_V2 = f"flydsl_moe2_layout_afp4_wfp4_bf16_t{BM}x128x128_atomic_sbm{BM}"


def _rel_l2(actual, ref):
    return ((actual.float() - ref.float()).norm() / ref.float().norm()).item()


def _quant_shuffle(w):
    tq = aiter.get_torch_quant(QuantType.per_1x32)
    q, s = tq(w, quant_dtype=dtypes.fp4x2)
    q = q.view(*w.shape[:-1], w.shape[-1] // 2)
    s = s.view(*w.shape[:-1], w.shape[-1] // 32)
    s0, s1, _ = s.shape
    shuffled = (
        shuffle_weight(q.view(dtypes.fp4x2), (16, 16)),
        e8m0_shuffle(s.view(s0 * s1, -1)).view(s0, s1, -1),
    )
    return q, s, shuffled


def _run(monkeypatch, inp, w1, w2, topk_weights, topk_ids, kn2, inter_real=None):
    _, _, (w1_sh, w1_ss) = _quant_shuffle(w1)
    _, _, (w2_sh, w2_ss) = _quant_shuffle(w2)
    if inter_real is not None:
        w2_sh.inter_real = inter_real
    metadata = fm._make_mxfp4_metadata(KN1, kn2, GateMode.SEPARATED, 0, block_m=BM)
    monkeypatch.setattr(fm, "get_2stage_cfgs", lambda *a, **k: metadata)
    return fused_moe(
        inp,
        w1_sh,
        w2_sh,
        topk_weights,
        topk_ids,
        w1_scale=w1_ss,
        w2_scale=w2_ss,
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Situv2,
        doweight_stage1=False,
    )


@_SKIP
@pytest.mark.parametrize(
    "kn2, padded",
    [
        (f"flydsl_mxmoe_g2_a4w4_{BM}x256x128_atomic", False),
        (f"flydsl_mxmoe_g2_a4w4_{BM}x256x128_atomic", True),
        (f"flydsl_mxmoe_g2_a4w4_{BM}x256x256_atomic", True),
    ],
)
def test_native_gemm2_matches_v2_gemm2(monkeypatch, kn2, padded):
    torch.manual_seed(0)
    dev = "cuda"
    dtype = dtypes.bf16
    inp = torch.randn((TOKEN, MODEL_DIM), dtype=dtype, device=dev) / 10
    w1 = torch.randn((E, 2 * INTER_DIM, MODEL_DIM), dtype=dtype, device=dev) / 10
    w2 = torch.randn((E, MODEL_DIM, INTER_DIM), dtype=dtype, device=dev) / 10
    score = torch.randn((TOKEN, E), dtype=dtype, device=dev)
    topk_weights, topk_ids = fused_topk(inp, score, TOPK, True)

    # a4w4 reference: the kernels quantize both activations to MXFP4 too.
    tq = aiter.get_torch_quant(QuantType.per_1x32)
    w1_q, w1_s, _ = _quant_shuffle(w1)
    w2_q, w2_s, _ = _quant_shuffle(w2)
    a1_q, a1_s = tq(inp, quant_dtype=dtypes.fp4x2)
    ref1 = torch_moe_stage1(
        a1_q,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        a1_scale=a1_s,
        w1_scale=w1_s,
        doweight=False,
    )
    a2_q, a2_s = tq(ref1, quant_dtype=dtypes.fp4x2)
    ref = torch_moe_stage2(
        a2_q.view(TOKEN, TOPK, -1),
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_s,
        a2_scale=a2_s,
        doweight=True,
    )
    baseline = _run(monkeypatch, inp, w1, w2, topk_weights, topk_ids, KN2_V2)

    if padded:
        pad = PADDED_INTER - INTER_DIM
        gate, up = w1.chunk(2, dim=1)
        w1 = torch.cat([F.pad(gate, (0, 0, 0, pad)), F.pad(up, (0, 0, 0, pad))], 1)
        w2 = F.pad(w2, (0, pad))
    out = _run(
        monkeypatch,
        inp,
        w1,
        w2,
        topk_weights,
        topk_ids,
        kn2,
        inter_real=INTER_DIM if padded else None,
    )

    assert not out.isnan().any().item()
    # Same gemm1 and inputs: only gemm2's accumulation order differs.
    assert _rel_l2(out, baseline) < 1e-2, _rel_l2(out, baseline)
    assert _rel_l2(out, ref) < 5e-2, _rel_l2(out, ref)
