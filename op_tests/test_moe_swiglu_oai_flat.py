# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx950 SwiGLU OAI MXFP4 FLAT 1-stage asm fused-MoE regression (Minimax M3).

Shape: model_dim=6144, inter_dim=768, E=129, topk=5, bf16 X, MXFP4 weights
(a16w4), tokens 1/2/4/8/16 -- the buckets in
aiter/configs/model_configs/minimax_m3_a16w4_tuned_fmoe.csv.

Default run (CI, ``python3 op_tests/test_moe_swiglu_oai_flat.py``):
  * dispatch: each token bucket resolves to the expected kernel
    (16x32 / 16x128 asm FLAT, or the FlyDSL 2-stage row), and the asm row is
    discarded for calls the kernel cannot honour (bias, non-default limit).
  * accuracy: fused_moe output vs. a torch reference that mirrors the path's
    quantization. The asm kernels quantize X and the SwiGLU output to MXFP4
    in-kernel, so their reference quantizes both too; the FlyDSL a16w4 path
    keeps bf16 activations, so its reference does as well.
"""

import importlib
import math

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    fused_moe,
    fused_topk,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.test_common import checkAllclose

# sys.modules lookup: ``aiter.fused_moe`` as an attribute may be the function.
fm = importlib.import_module("aiter.fused_moe")

MODEL_DIM = 6144
INTER_DIM = 768
EXPERTS = 129
TOPK = 5
TOKENS = (1, 2, 4, 8, 16)
CU_NUM = 256

K_16X32 = "_ZN5aiter53fmoe_bf16_pertokenMXfp4_g1u1_flat_vs_swiglu_oai_16x32E"
K_16X128 = "_ZN5aiter56fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_swiglu_oai_16x128E"

# Expected dispatch from minimax_m3_a16w4_tuned_fmoe.csv; None = FlyDSL 2-stage.
EXPECTED_KERNEL = {1: K_16X32, 2: K_16X32, 4: K_16X32, 8: K_16X128, 16: None}

# logits_diff = 1 - cosine similarity, the accuracy gate for every path.
# Measured on MI355X: asm 16x32 ~9e-5, asm 16x128 ~3e-5, FlyDSL ~7e-6. A wrong
# activation (e.g. plain silu(gate) * up) or a corrupted tile is >= 1e-2.
#
# Elementwise allclose is not used for the asm rows: their MXFP4 intermediate
# is quantized from fp32 sums accumulated in a different order than torch's,
# so values near an fp4 rounding boundary land one step apart. That moves
# ~10-35% of outputs by more than 0.01 (max ~0.4 on outputs up to ~7) while the
# result as a whole is unchanged.
LOGITS_DIFF_TOL = 2e-3


# ---------------------------------------------------------------------------
# Data and references
# ---------------------------------------------------------------------------
class Weights:
    def __init__(self, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        quant = aiter.get_torch_quant(QuantType.per_1x32)
        # Scaled so gate/up have std ~3: the sigmoid and both +-7 clamps are hit.
        w1 = (
            torch.randn(
                (EXPERTS, 2 * INTER_DIM, MODEL_DIM), generator=gen, device="cuda"
            )
            * (3.0 / math.sqrt(MODEL_DIM))
        ).to(dtypes.bf16)
        w2 = (
            torch.randn((EXPERTS, MODEL_DIM, INTER_DIM), generator=gen, device="cuda")
            / math.sqrt(INTER_DIM)
        ).to(dtypes.bf16)
        w1_qt, self.w1_scale = quant(w1, quant_dtype=dtypes.fp4x2)
        w2_qt, self.w2_scale = quant(w2, quant_dtype=dtypes.fp4x2)
        del w1, w2
        self.w1_qt = w1_qt.view(EXPERTS, 2 * INTER_DIM, MODEL_DIM // 2)
        self.w2_qt = w2_qt.view(EXPERTS, MODEL_DIM, INTER_DIM // 2)
        # a16w4 GGUU layout. For gate_up=False this is byte-identical to
        # shuffle_weight(w, (16, 16)) + e8m0_shuffle, which the asm FLAT kernels
        # read, and it is what the FlyDSL a16w4 rows expect.
        self.w1 = shuffle_weight_a16w4(self.w1_qt, 16, False)
        self.w2 = shuffle_weight_a16w4(self.w2_qt, 16, False)
        self.w1_scale_shuf = shuffle_scale_a16w4(self.w1_scale, EXPERTS, False)
        self.w2_scale_shuf = shuffle_scale_a16w4(self.w2_scale, EXPERTS, False)


def make_tokens(token, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((token, MODEL_DIM), generator=gen, device="cuda").to(dtypes.bf16)
    score = torch.randn((token, EXPERTS), generator=gen, device="cuda").to(dtypes.bf16)
    topk_weights, topk_ids = fused_topk(x, score, TOPK, True)
    return x, topk_weights, topk_ids


def reference(x, topk_weights, topk_ids, w, quantize_activations):
    """Torch MoE reference with OAI SwiGLU (alpha=1.702, limit=7.0).

    quantize_activations=True mirrors the asm FLAT kernels: X and the fp32
    SwiGLU output are MXFP4-quantized (per 1x32, e8m0 RoundUp) before each GEMM.
    False mirrors the FlyDSL a16w4 path, which keeps bf16 activations.
    """
    quant = aiter.get_torch_quant(QuantType.per_1x32)
    token = x.shape[0]
    if quantize_activations:
        a1, a1_scale = quant(x, quant_dtype=dtypes.fp4x2)
    else:
        a1, a1_scale = x, None
    out1 = torch_moe_stage1(
        a1,
        w.w1_qt,
        w.w2_qt,
        topk_weights,
        topk_ids,
        dtype=dtypes.fp32 if quantize_activations else dtypes.bf16,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        a1_scale=a1_scale,
        w1_scale=w.w1_scale,
        swiglu_limit=7.0,
    )
    if quantize_activations:
        a2, a2_scale = quant(out1, quant_dtype=dtypes.fp4x2)
    else:
        a2, a2_scale = out1, None
    return torch_moe_stage2(
        a2.view(token, TOPK, -1),
        w.w1_qt,
        w.w2_qt,
        topk_weights,
        topk_ids,
        dtype=dtypes.bf16,
        quant_type=QuantType.per_1x32,
        w2_scale=w.w2_scale,
        a2_scale=a2_scale,
        doweight=True,
    )


def logits_diff(ref, out):
    x, y = ref.double(), out.double()
    return float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())


# ---------------------------------------------------------------------------
# Dispatch capture
# ---------------------------------------------------------------------------
class CaptureDispatch:
    """Record the get_2stage_cfgs call made by fused_moe and its result."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        self._orig = fm.get_2stage_cfgs

        def wrapper(*args, **kwargs):
            metadata = self._orig(*args, **kwargs)
            self.calls.append((args, kwargs, metadata))
            return metadata

        fm.get_2stage_cfgs = wrapper
        return self

    def __exit__(self, *exc):
        fm.get_2stage_cfgs = self._orig

    @property
    def last(self):
        assert self.calls, "fused_moe did not call get_2stage_cfgs"
        return self.calls[-1]


def asm_kernel_of(metadata):
    """Kernel name of an asm 1-stage dispatch, or None for anything else."""
    if not metadata.run_1stage:
        return None
    stage1 = metadata.stage1
    assert getattr(stage1, "func", None) is fm.fused_moe_1stage, stage1
    return stage1.keywords.get("kernelName")


def run_fused_moe(x, topk_weights, topk_ids, w, **extra):
    return fused_moe(
        x,
        w.w1,
        w.w2,
        topk_weights,
        topk_ids,
        w1_scale=w.w1_scale_shuf,
        w2_scale=w.w2_scale_shuf,
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Swiglu,
        gate_mode=GateMode.SEPARATED.value,
        **extra,
    )


def check_token(token, w, expected_kernel, seed=0):
    """Run one bucket and assert dispatch and accuracy.

    Returns (out, ref, captured get_2stage_cfgs (args, kwargs), asm kernel name
    or None, logits_diff).
    """
    x, topk_weights, topk_ids = make_tokens(token, seed + token)
    with CaptureDispatch() as cap:
        out = run_fused_moe(x, topk_weights, topk_ids, w)
    torch.cuda.synchronize()
    args, kwargs, metadata = cap.last
    kernel = asm_kernel_of(metadata)
    assert kernel == expected_kernel, (
        f"token={token}: dispatched {kernel or 'non-asm (2-stage)'}, "
        f"expected {expected_kernel or 'FlyDSL 2-stage'}"
    )
    if expected_kernel is not None:
        assert metadata.flat, f"token={token}: asm row must be FLAT"
    ref = reference(
        x, topk_weights, topk_ids, w, quantize_activations=kernel is not None
    )
    assert torch.isfinite(out).all(), f"token={token}: non-finite output"
    diff = logits_diff(ref, out)
    label = f"token={token} {kernel or 'flydsl'} logits_diff={diff:.2e}"
    if kernel is None:
        checkAllclose(ref, out, msg=label)
    else:
        delta = (out.float() - ref.float()).abs()
        aiter.logger.info(
            f"{label} max_abs_delta={delta.max().item():.3f} "
            f"mean_abs_delta={delta.mean().item():.4f} "
            f"(MXFP4 activations: elementwise allclose not applicable)"
        )
    assert diff < LOGITS_DIFF_TOL, (
        f"token={token}: logits_diff {diff:.3e} >= {LOGITS_DIFF_TOL:.0e} "
        f"({kernel or 'flydsl'})"
    )
    return out, ref, (args, kwargs), kernel, diff


def check_guard(dispatch_call, **override):
    """Re-resolve a captured asm dispatch with one argument changed."""
    args, kwargs = dispatch_call
    kwargs = {**kwargs, **override}
    try:
        metadata = fm.get_2stage_cfgs(*args, **kwargs)
    except NotImplementedError:
        return  # rejected outright: also never reaches the asm kernel
    assert asm_kernel_of(metadata) not in (
        K_16X32,
        K_16X128,
    ), f"asm SwiGLU OAI row must be discarded for {override}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _skip_reason():
    if not torch.cuda.is_available():
        return "no GPU"
    if get_gfx() != "gfx950":
        return f"needs gfx950, got {get_gfx()}"
    if get_cu_num() != CU_NUM:
        return f"tuned rows are for cu_num={CU_NUM}, got {get_cu_num()}"
    return None


def test_swiglu_oai_flat_minimax_m3():
    reason = _skip_reason()
    if reason:
        print(f"[skip] test_swiglu_oai_flat_minimax_m3: {reason}")
        return
    w = Weights()
    asm_dispatch = None
    for token in TOKENS:
        _, _, dispatch, kernel, _ = check_token(token, w, EXPECTED_KERNEL[token])
        if kernel is not None and asm_dispatch is None:
            asm_dispatch = dispatch
    assert asm_dispatch is not None
    # The asm kernels hard-code alpha/limit and take no bias: such calls must
    # fall back instead of silently computing a different function.
    check_guard(asm_dispatch, has_stage1_bias=True)
    check_guard(asm_dispatch, has_stage2_bias=True)
    check_guard(asm_dispatch, swiglu_limit=5.0)
    print("[pass] test_swiglu_oai_flat_minimax_m3")


if __name__ == "__main__":
    torch.set_default_device("cuda")
    test_swiglu_oai_flat_minimax_m3()
