# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Do the MoE GEMMs still address the right expert once the weights pass 4 GB?

``buffer_load``'s voffset is 32-bit, so a kernel spanning a whole weight tensor with
one buffer resource and folding the expert offset into that offset wraps as soon as
an expert's byte offset passes 2**32, silently reading a different expert's weights.
E=896 with model_dim=3584 and inter_dim=3072 makes w1 9.2 GB and w2 4.6 GB.

A torch reference cannot exist at these shapes: it dequantizes fp4 to f32, and
``w1 = 2 * w2`` always holds here, so ``w2 > 4 GB`` forces a >68 GB reference. That
is also why the bug survived -- the existing harnesses cannot reach these shapes.

So calibrate a twin instead: give one weight code to a LOW expert (whose offset
never wraps) and the same code to a HIGH expert (whose offset does), background code
elsewhere, and route every token to one expert at a time. Correct addressing makes
``out(high) == out(low)`` exactly; a wrap collapses ``out(high)`` to the background
level. Outputs are only ever compared against outputs that should be identical, so
stage1's activation nonlinearity never has to be inverted.
"""

from __future__ import annotations

import pytest
import torch

from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

# One rank holding every expert, which is what makes the weight tensors big
# enough to trip the 32-bit offset.
E, MODEL_DIM, INTER_DIM = 896, 3584, 3072
TOKEN, TOPK, BLOCK_M = 32, 1, 32
DTYPE = dtypes.bf16

# e2m1 codes, sign bit clear: 1 -> 0.5, 7 -> 6.0. Both nibbles of a byte carry the
# same code, so every element of an expert decodes to one value and the output
# becomes a fingerprint of *which* expert was actually read.
BG_CODE = 1
HI_CODE = 7

# w1 needs 9.2 GB, and shuffle_weight_a16w4 makes a second copy while it runs.
_NEED_GIB = 26.0


def _free_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _total = torch.cuda.mem_get_info()
    return free / 2**30


_SKIP = pytest.mark.skipif(
    get_gfx() not in ("gfx950",) or not is_flydsl_available(),
    reason="gfx950 FlyDSL required",
)
_SKIP_MEM = pytest.mark.skipif(
    _free_gib() < _NEED_GIB,
    reason=f"needs ~{_NEED_GIB:.0f} GiB free VRAM, have {_free_gib():.1f} GiB",
)


def _byte(code: int) -> int:
    return (code << 4) | code


def _wraps(per_expert_bytes: int, expert: int) -> bool:
    return expert * per_expert_bytes > 2**32


def _run_one_expert(launch, target: int) -> float:
    """Route every token to ``target`` and reduce the output to one number."""
    topk_ids = torch.full((TOKEN, TOPK), target, dtype=torch.int32, device="cuda")
    topk_w = torch.ones((TOKEN, TOPK), dtype=torch.float32, device="cuda")
    sorted_ids, sorted_w, sorted_eids, num_valid, _ = moe_sorting(
        topk_ids, topk_w, E, MODEL_DIM, DTYPE, BLOCK_M
    )
    out = launch(sorted_ids, sorted_eids, num_valid, sorted_w)
    torch.cuda.synchronize()
    return out.float().abs().mean().item()


def _assert_addressing(outs: dict[int, float], bg_e: int, low_e: int, high_e: int):
    bg, low, high = outs[bg_e], outs[low_e], outs[high_e]

    # Guard against a degenerate pass: if the background and the calibration expert
    # produced the same output, the fingerprint carries no information and an
    # "out(high) == out(low)" match would prove nothing.
    assert abs(low - bg) > 0.1 * max(bg, 1e-9), (
        f"probe is degenerate: background expert {bg_e} ({bg:.4f}) and calibration "
        f"expert {low_e} ({low:.4f}) are indistinguishable, so the test cannot tell "
        f"a correct read from a wrapped one"
    )

    wrapped = abs(high - bg) <= 0.1 * max(bg, 1e-9)
    assert abs(high - low) <= 1e-6 * max(low, 1e-9), (
        f"expert {high_e} read the wrong weights: got {high:.4f}, expected "
        f"{low:.4f} (same weight code as expert {low_e})"
        + (
            f" -- it matches the background level {bg:.4f}, i.e. the byte offset "
            f"wrapped past 2**32"
            if wrapped
            else ""
        )
    )


@_SKIP
@_SKIP_MEM
def test_a16w4_stage1_past_4gib():
    """w1 = [E, 2*inter, model] = 9.2 GB; wraps from expert 390 on."""
    per_expert = (2 * INTER_DIM) * MODEL_DIM // 2
    bg_e, low_e, high_e = 50, 100, 400
    assert not _wraps(per_expert, low_e), "calibration expert must not wrap"
    assert _wraps(per_expert, high_e), "test expert must wrap"

    w1 = torch.empty(
        (E, 2 * INTER_DIM, MODEL_DIM // 2), dtype=torch.uint8, device="cuda"
    )
    w1.fill_(_byte(BG_CODE))
    for e in (low_e, high_e):
        w1[e].fill_(_byte(HI_CODE))
    w1_shuf = shuffle_weight_a16w4(w1, 16, False)
    # The whole premise is "expert e decodes to one value", so the shuffle must not
    # move data across experts.
    for e in (bg_e, low_e, high_e):
        assert torch.unique(w1_shuf[e]).numel() == 1, f"shuffle mixed expert {e}"
    del w1
    torch.cuda.empty_cache()

    scale = torch.full(
        (E * 2 * INTER_DIM, MODEL_DIM // 32), 127, dtype=torch.uint8, device="cuda"
    )
    scale_shuf = shuffle_scale_a16w4(scale, E, False)
    del scale
    torch.cuda.empty_cache()

    inp = torch.full((TOKEN, MODEL_DIM), 1.0, dtype=DTYPE, device="cuda")

    def launch(sids, seids, nvalid, sw):
        return flydsl_moe_stage1(
            a=inp,
            w1=w1_shuf,
            sorted_token_ids=sids,
            sorted_expert_ids=seids,
            num_valid_ids=nvalid,
            topk=TOPK,
            tile_m=BLOCK_M,
            tile_n=256,
            tile_k=256,
            a_dtype="bf16",
            b_dtype="fp4",
            out_dtype="bf16",
            w1_scale=scale_shuf,
            a1_scale=None,
            sorted_weights=sw,
        )

    try:
        outs = {e: _run_one_expert(launch, e) for e in (bg_e, low_e, high_e)}
        _assert_addressing(outs, bg_e, low_e, high_e)
    finally:
        # Rebound rather than deleted: `launch` closes over both, so a del would leave
        # the closure referring to an unbound name.
        w1_shuf = scale_shuf = None
        torch.cuda.empty_cache()


def _inter_pad(inter_dim: int) -> int:
    return ((inter_dim + 255) // 256 * 256) - inter_dim


@_SKIP
@_SKIP_MEM
@pytest.mark.parametrize("a_dtype", ["bf16", "fp4"], ids=["a16w4", "a4w4"])
@pytest.mark.parametrize(
    "inter_dim",
    # 3072: the real TP1 shape, K a multiple of the tile, no padding.
    # 1408: padded (1408 % 256 == 128) *and* still over the size threshold, so the
    #       raw-pointer path runs with the k-tail clamping active. Worth covering
    #       separately: the buffer path bounds-checks in hardware and the raw
    #       pointer does not, so anything that leant on out-of-range-reads-zero
    #       for the padded tail would only break here.
    [3072, 1408],
    ids=["i3072", "i1408_padded"],
)
def test_stage2_past_4gib(a_dtype, inter_dim):
    """w2 = [E, model, inter]; 4.6 GB at inter=3072, wrapping from expert 780 on.

    Both activation dtypes are covered because they reach different builders:
    bf16 -> compile_mixed_moe_gemm2_a16w4, fp4 -> compile_mixed_moe_gemm2_common.
    """
    per_expert = MODEL_DIM * inter_dim // 2
    bg_e, low_e, high_e = 50, 100, 800
    assert not _wraps(per_expert, low_e), "calibration expert must not wrap"
    # Only the big shape actually wraps. The padded one is here for a different
    # reason: its total size still puts the kernel on the raw-pointer path, so it
    # checks that path stays correct while the padded-k tail clamping is active.
    # Both cases share the same assertion, which is the point -- two experts with
    # identical weights must give identical output either way.

    w2 = torch.empty((E, MODEL_DIM, inter_dim // 2), dtype=torch.uint8, device="cuda")
    w2.fill_(_byte(BG_CODE))
    for e in (low_e, high_e):
        w2[e].fill_(_byte(HI_CODE))
    w2_shuf = shuffle_weight_a16w4(w2, 16, False)
    for e in (bg_e, low_e, high_e):
        assert torch.unique(w2_shuf[e]).numel() == 1, f"shuffle mixed expert {e}"
    del w2
    torch.cuda.empty_cache()

    scale = torch.full(
        (E * MODEL_DIM, inter_dim // 32), 127, dtype=torch.uint8, device="cuda"
    )
    scale_shuf = shuffle_scale_a16w4(scale, E, False)
    del scale
    torch.cuda.empty_cache()

    if a_dtype == "bf16":
        # a16w4 stage2 takes the intermediate buffer already in sorted-row layout and
        # atomic-scatters into a caller-provided, zeroed output, so both are built per
        # launch from the sorted ids rather than up front as (TOKEN, TOPK, K).
        a2 = None
        a2_scale = None
    else:
        # 0x22 = both nibbles e2m1 code 2 (=1.0), so the activation is constant too.
        a2 = torch.full(
            (TOKEN, TOPK, inter_dim // 2), 0x22, dtype=torch.uint8, device="cuda"
        )
        a2_scale = torch.full(
            (TOKEN * TOPK, inter_dim // 32), 127, dtype=torch.uint8, device="cuda"
        )

    def launch(sids, seids, nvalid, sw):
        a2s = a2_scale
        if a2s is not None:
            from aiter.utility.fp4_utils import moe_mxfp4_sort

            a2s = moe_mxfp4_sort(
                a2_scale.view(TOKEN, TOPK, -1),
                sorted_ids=sids,
                num_valid_ids=nvalid,
                token_num=TOKEN,
                block_size=BLOCK_M,
            )
        if a_dtype == "bf16":
            states = torch.full(
                (sids.numel(), inter_dim), 1.0, dtype=DTYPE, device="cuda"
            )
            out = torch.zeros((TOKEN, MODEL_DIM), dtype=DTYPE, device="cuda")
        else:
            states, out = a2, None
        return flydsl_moe_stage2(
            inter_states=states,
            out=out,
            w2=w2_shuf,
            sorted_token_ids=sids,
            sorted_expert_ids=seids,
            num_valid_ids=nvalid,
            topk=TOPK,
            tile_m=BLOCK_M,
            tile_n=128 if a_dtype == "bf16" else 256,
            tile_k=256,
            a_dtype=a_dtype,
            b_dtype="fp4",
            out_dtype="bf16",
            mode="atomic",
            w2_scale=scale_shuf,
            a2_scale=a2s,
            sorted_weights=sw,
            inter_dim_pad=_inter_pad(inter_dim),
        )

    try:
        outs = {e: _run_one_expert(launch, e) for e in (bg_e, low_e, high_e)}
        _assert_addressing(outs, bg_e, low_e, high_e)
    finally:
        w2_shuf = scale_shuf = None
        torch.cuda.empty_cache()
