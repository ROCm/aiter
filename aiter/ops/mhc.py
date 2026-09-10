# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import math

import torch
from torch import Tensor

from aiter import dtypes

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_cu_num, get_gfx_runtime
from ..jit.utils.torch_guard import torch_compile_guard


@compile_ops("module_mhc", develop=True)
def mhc_pre_gemm_sqrsum(
    out: Tensor,
    sqrsum: Tensor,
    x: Tensor,
    fn: Tensor,
    tile_k: int = 128,  # 64 or 128
    w_preshuffle_bf16: int = 0,  # 1: fn is pre-packed BF16 hi/lo from mhc_pre_convert_fn
) -> None: ...


@compile_ops("module_mhc", develop=True)
def mhc_pre_convert_fn(
    fn_packed: Tensor,  # (hc_mult3, hc_hidden_size) int32 out
    fn: Tensor,  # (hc_mult3, hc_hidden_size) fp32 in
) -> None: ...


def mhc_fn_kb(arch: str | None = None) -> int:
    """k elements per hi/lo interleave block in the packed BF16 ``fn`` layout.

    A lane's fn fragment must cover a whole block, i.e.
    ``vec_tile = tile_k / (warp_size / 16) >= kb``. wave32 reaches 16 at tile_k=32
    and its WMMA path consumes a whole 16-float block as two verbatim fragments;
    wave64 only reaches 8 there, so it uses 8-element blocks and keeps tile_k=32.

    MUST stay in lockstep with ``mhc_fn_kb`` in csrc/kernels/mhc_kernels.cu.
    """
    if arch is None:
        arch = get_gfx_runtime()
    return 16 if arch == "gfx1250" else 8


def mhc_shuffle_fn(fn: torch.Tensor, arch: str | None = None) -> torch.Tensor:
    """Pack FP32 ``fn`` into the layout the ``w_preshuffle_bf16`` GEMM reads.

    Each FP32 weight is split into ``hi = bf16(fn)`` and ``lo = bf16(fn - fp32(hi))``;
    bf16 shares fp32's 8-bit exponent, so ``lo`` keeps its magnitude and
    ``hi*x + lo*x`` reconstructs the FP32 product at BF16 MFMA rate.

    The pair is stored block-interleaved. Viewing the (hc_mult3, hc_hidden_size) int32
    output as uint16, for ``kb = mhc_fn_kb()``::

        hi(k) at  row * 2 * hc_hidden_size + (k // kb) * 2 * kb + (k % kb)
        lo(k) at  the same index + kb

    so one block of ``kb`` consecutive k occupies ``kb`` floats -- hi in the first
    half, lo in the second. Element count and row stride are unchanged.

    ``arch`` defaults to the runtime GPU. Pass it explicitly only to build a layout
    for a different target than the one this process is running on.
    """
    assert fn.dtype == torch.float32, f"fn must be fp32, got {fn.dtype}"
    kb = mhc_fn_kb(arch)
    n_row, k = fn.shape
    assert k % kb == 0, f"hc_hidden_size {k} must be divisible by mhc_fn_kb {kb}"
    fn = fn.contiguous()
    hi = fn.to(torch.bfloat16)
    lo = (fn - hi.to(torch.float32)).to(torch.bfloat16)
    out16 = torch.empty(n_row, 2 * k, dtype=torch.int16, device=fn.device)
    blocks = out16.view(n_row, k // kb, 2, kb)
    blocks[:, :, 0, :] = hi.view(torch.int16).view(n_row, k // kb, kb)
    blocks[:, :, 1, :] = lo.view(torch.int16).view(n_row, k // kb, kb)
    return out16.view(torch.int32)


# Back-compat name; the HIP kernel and this must produce identical bytes.
mhc_pre_convert_fn_ref = mhc_shuffle_fn


@compile_ops("module_mhc", develop=True)
def mhc_pre_big_fuse(
    post_mix: Tensor,
    comb_mix: Tensor,
    layer_input: Tensor,
    gemm_out_mul: Tensor,
    gemm_out_sqrsum: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    residual: Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
    res_preshuffle: int = 0,
) -> None: ...


@compile_ops("module_mhc", develop=True)
def mhc_pre_big_fuse_rmsnorm(
    post_mix: Tensor,
    comb_mix: Tensor,
    out: Tensor,
    gemm_out_mul: Tensor,
    gemm_out_sqrsum: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    residual: Tensor,
    norm_weight: Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
    res_preshuffle: int = 0,
) -> None: ...


# Pre-shuffled residual layout selected independently with res_preshuffle=1
# (kernel-side tile width: mhc_res_ks).
#   resS[k // KS][head][row][k % KS]  <-  res[row][head][k]
# The tensor keeps its (m, hc_mult, hidden_size) shape and element count; only the
# order of the elements inside the buffer changes. Both the gemm's residual read and
# its next_residual write use this layout, so it stays internal to a stack of layers:
# only the first residual in and the last one out need converting.
MHC_RES_KS = 32


def _check_mhc_res_preshuffle_arch(shuffled: bool, arch: str) -> None:
    if shuffled and arch != "gfx1250":
        raise ValueError(
            f"res_preshuffle=1 is only supported on gfx1250, got {arch}; "
            "use res_preshuffle=0 (BF16 compute is controlled independently)"
        )


def mhc_res_shuffle(residual: torch.Tensor) -> torch.Tensor:
    """res[row][head][k] -> resS[k//KS][head][row][k%KS], same shape."""
    m, hc_mult, hidden_size = residual.shape
    assert hidden_size % MHC_RES_KS == 0
    return (
        residual.view(m, hc_mult, hidden_size // MHC_RES_KS, MHC_RES_KS)
        .permute(2, 1, 0, 3)
        .contiguous()
        .view(m, hc_mult, hidden_size)
    )


def mhc_res_unshuffle(shuffled: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`mhc_res_shuffle`."""
    m, hc_mult, hidden_size = shuffled.shape
    assert hidden_size % MHC_RES_KS == 0
    return (
        shuffled.view(hidden_size // MHC_RES_KS, hc_mult, m, MHC_RES_KS)
        .permute(2, 1, 0, 3)
        .contiguous()
        .view(m, hc_mult, hidden_size)
    )


@functools.lru_cache(maxsize=1024)
def get_mhc_pre_splitk(
    m: int, hc_hidden_size: int, w_preshuffle_bf16: bool = False
) -> tuple[int, int]:
    prefetch_stages = 2
    tile_m = 16 * 4
    num_cu = get_cu_num()
    arch = get_gfx_runtime()
    # Vector LDS weight reads shorten the gfx1250 BF16 loop. More split-K
    # workgroups improve latency hiding for large M without changing FP32 tuning.
    prefill_tg_factor = (
        8
        if w_preshuffle_bf16
        and arch == "gfx1250"
        and num_cu == 256
        and m >= 2048
        and hc_hidden_size in (4 * 4096, 4 * 7168)
        else 4
    )
    tile_k_tg_dict = (
        {
            128: 2 * num_cu,
            64: 4 * num_cu,
        }
        if arch.startswith("gfx9")
        else {
            64: prefill_tg_factor * num_cu,
        }
    )
    selected_splitk = 1
    selected_tile_k = 64
    num_tg_m = (m + tile_m - 1) // tile_m
    selected_score = num_tg_m / (num_cu * tile_k_tg_dict[selected_tile_k])
    selected_score = selected_score / math.ceil(selected_score)
    for tile_k, meanwhile_tg in tile_k_tg_dict.items():
        if (hc_hidden_size % tile_k) != 0:
            continue
        for splitk in range(1, num_cu + 1):
            if hc_hidden_size % (splitk * tile_k) != 0 or (hc_hidden_size // splitk) < (
                tile_k * prefetch_stages
            ):
                continue
            num_tg = num_tg_m * splitk
            score = num_tg / meanwhile_tg
            score = score / math.ceil(score)
            if selected_score < score:
                selected_splitk = splitk
                selected_tile_k = tile_k
                selected_score = score
            # print(f"{selected_score=} {selected_splitk=} {selected_tile_k=} {score=} {splitk=} {tile_k=}")
            if num_tg > meanwhile_tg * 2:
                break

    return selected_splitk, selected_tile_k


def _mhc_fused_valid_splitk(hidden_size, tile_k, num_cu, prefetch_stages=2):
    return [
        sk
        for sk in range(1, num_cu + 1)
        if hidden_size % (sk * tile_k) == 0
        and (hidden_size // sk) >= tile_k * prefetch_stages
    ]


def _mhc_fused_fill_splitk(m, valid_splitk, num_cu):
    """Pick the split-k whose total grid best fills the device a few waves deep.

    Empirically the optimum sits near total_blocks ~= 32*num_cu, i.e.
    splitk ~= 32*num_cu/m, snapped (geometrically) to the nearest valid divisor.
    """
    ideal = max(1.0, 32.0 * num_cu / m)
    return min(valid_splitk, key=lambda sk: (abs(math.log(sk) - math.log(ideal)), -sk))


def _mhc_fused_config_gfx950_256(m, hidden_size, num_cu):
    tile_k = 64 if m >= 2 * hidden_size else 32
    if hidden_size % tile_k != 0:
        tile_k = 32 if tile_k == 64 else 64

    valid = _mhc_fused_valid_splitk(hidden_size, tile_k, num_cu)
    if not valid:
        return 1, 16, 32, tile_k
    splitk = _mhc_fused_fill_splitk(m, valid, num_cu)

    tile_n = 32  # tile_n=16 never wins on this chip
    if tile_k == 32:
        # large-m underfill: geom fill at split_k 2..4 leaves ~1 wave; a 2nd
        # K-reduction wave measured faster. Excludes geom>=8 (small m) and geom=1.
        if 2 <= splitk <= 4 and (2 * splitk) in valid:
            splitk = 2 * splitk
        tile_m = 32 if (m + 31) // 32 * splitk >= num_cu else 16
    else:  # tile_k == 64: tile_m=32 would overflow LDS, keep 16
        tile_m = 16
        # the compute-bound wide-k path wants >=2 K-reduction waves; fill gives
        # sk=1 at this m but sk=2 is ~2-5% faster (measured m>=8192).
        if splitk < 2 and 2 in valid:
            splitk = 2
    return splitk, tile_m, tile_n, tile_k


def _mhc_fused_config_gfx942_80(m, hidden_size, num_cu):
    tile_k = 32 if (hidden_size <= 4096 and m <= 128) else 64
    if hidden_size % tile_k != 0:
        tile_k = 32 if tile_k == 64 else 64

    valid = _mhc_fused_valid_splitk(hidden_size, tile_k, num_cu)
    if not valid:
        return 1, 16, 32, tile_k

    tile_n = 32  # tile_n=16 never wins on this chip
    tile_m = 16  # tile_m=32 (fn-reuse) never wins on this low-CU part
    if tile_k == 64:
        # deep fill: optimum ~= 12.8*num_cu total blocks; small m saturates the cap.
        m_blocks = (m + 15) // 16
        ideal = max(1.0, 12.8 * num_cu / m_blocks)
        splitk = min(valid, key=lambda sk: (abs(math.log(sk) - math.log(ideal)), -sk))
        if splitk < 2 and 2 in valid:  # large-m underfill; sk>=2 measured faster
            splitk = 2
    else:  # tile_k == 32 small-problem path: shallow ~2-wave fill
        splitk = _mhc_fused_fill_splitk(m, valid, num_cu)
    return splitk, tile_m, tile_n, tile_k


def _mhc_fused_config_gfx1250_256(m, hidden_size, num_cu):
    """Tuned on the pair (this gemm + the mhc_pre_big_fuse reduction that always
    follows it), not on the gemm alone. The reduction reads (split_k, m, hc_mult3)
    and (split_k, m), so its cost grows with split_k: tuning the gemm in isolation
    picks split_k up to 4x too deep and loses ~1.06x end-to-end at m<=512.

    Thresholds below were measured at num_cu=256 only; they are written relative to
    num_cu because they are occupancy-driven, not because the scaling was verified.
    """
    tile_k = 32 if hidden_size % 32 == 0 else 64
    # k_loop >= 1 is enough: the LDS/fn ring guards every prefetch against k_loop
    # (verified bit-exact against split_k=1), and small m wants a deep split.
    valid = _mhc_fused_valid_splitk(hidden_size, tile_k, num_cu, prefetch_stages=1)
    if not valid:
        return 1, 16, 32, tile_k

    tile_n = 32  # re-measured over tile_n in {16, 32}: 16 never wins

    # split_k, snapped DOWN to a legal divisor -- never to the geometrically nearest
    # one, which on a coarse lattice overshoots badly (hidden=5120, m=512: split_k=80
    # is 1.48x slower than 40).
    if m <= num_cu:
        # launch-bound: the optimum holds split_k*m ~= 32*num_cu. Deeper stops paying
        # (the reduction grows with split_k), shallower starves the gemm grid.
        target = max(1, 32 * num_cu // m)
    elif m <= 2 * num_cu:
        target = num_cu // 8  # short plateau between the two regimes
    else:
        # the optimum tracks the largest power of two <= 128*num_cu/m; flooring to a
        # power of two matters (a plain geometric snap overshoots at every m that is
        # not one, e.g. m=3072 wants 8, not 14)
        ideal = int(max(1.0, 128.0 * num_cu / m))
        target = 1 << (ideal.bit_length() - 1)
    below = [sk for sk in valid if sk <= target]
    splitk = max(below) if below else min(valid)

    # tile_m. Small m keeps 16: with the pair's shallower split_k the fn-reuse grid no
    # longer fills the device (this is the opposite of what tuning the gemm alone says).
    # tile_m=64 pays off (up to 1.07x at m=8192/16384) only where its grid tiles whole
    # waves exactly; at m=6144 or 12288, where it does not, it loses up to 1.09x.
    if m <= 1.5 * num_cu:
        tile_m = 16
    elif m >= 16 * num_cu and m % 64 == 0 and ((m // 64) * splitk) % num_cu == 0:
        tile_m = 64
    else:
        tile_m = 32
    return splitk, tile_m, tile_n, tile_k


def _mhc_fused_bf16_shuffled_config_gfx1250_256(m, hidden_size, config):
    """GEMM + RMSNorm reduction tuning for packed weights and shuffled residuals.

    Measured on gfx1250/256 CU for N=4096/7168. Keep the prior policy for
    unaligned/tail shapes; the tuned ranges cover M=32..16384 in steps of 32.
    Plain residuals and FP32 GEMM retain their independent existing policy.
    """
    if hidden_size not in (4096, 7168) or not 32 <= m <= 16384 or m % 32:
        return config
    split_k, tile_m, tile_n, tile_k = config
    if hidden_size == 7168:
        if m <= 96:
            split_k, tile_m = 112, 16
        elif m <= 256:
            split_k, tile_m = 112, 32
        elif m <= 384 or 512 < m <= 640:
            split_k, tile_m = 56, 32
        elif 640 < m <= 1024:
            # The direct-store pipeline favors tile_m=32 in this range.
            split_k, tile_m = 32, 32
    elif 128 < m <= 384 or 512 < m <= 640:
        split_k, tile_m = 64, 32
    # Extend tile_m=64 to M>=2048 only when the grid fills whole CU waves.
    if m >= 2048 and m % 64 == 0 and ((m // 64) * split_k) % 256 == 0:
        tile_m = 64
    return split_k, tile_m, tile_n, tile_k


def _mhc_fused_config_default(m, hidden_size, num_cu):
    """Generic fallback for untuned chips: pick (split_k, tile_k) by the occupancy
    scoring search (how many thread-groups fit vs. how many the device can run at
    once), with tile_m fixed at 16 (single mfma band, no fn-reuse path) to avoid the
    tile_m=32 regression seen on low-CU parts that aren't tuned yet. tile_n is then
    chosen to fill the grid."""
    prefetch_stages = 2
    tile_m = 16
    # thread-groups the device can keep in flight per tile_k (smaller tile_k => more)
    tile_k_tg_dict = {
        64: 2 * num_cu,
        32: 4 * num_cu,
    }
    num_tg_m = (m + tile_m - 1) // tile_m

    selected_splitk = 1
    selected_tile_k = 32 if hidden_size % 32 == 0 else 64
    selected_score = -1.0
    for tile_k, meanwhile_tg in tile_k_tg_dict.items():
        if hidden_size % tile_k != 0:
            continue
        for splitk in range(1, num_cu + 1):
            if hidden_size % (splitk * tile_k) != 0 or (hidden_size // splitk) < (
                tile_k * prefetch_stages
            ):
                continue
            num_tg = num_tg_m * splitk
            # occupancy fill ratio, penalize the partial last wave (closer to 1 = better)
            score = num_tg / meanwhile_tg
            score = score / math.ceil(score)
            if score > selected_score:
                selected_splitk = splitk
                selected_tile_k = tile_k
                selected_score = score
            if num_tg > meanwhile_tg * 2:
                break

    m_blocks = (m + tile_m - 1) // tile_m
    tile_n = 16 if num_cu * 2 > m_blocks * selected_splitk else 32
    return selected_splitk, tile_m, tile_n, selected_tile_k


# Per-chip tuned config registry, keyed by (gfx_arch, cu_num).
# Each entry: (m, hidden_size, num_cu) -> (splitk, tile_m, tile_n, tile_k).
_MHC_FUSED_POST_PRE_CONFIG = {
    ("gfx950", 256): _mhc_fused_config_gfx950_256,
    ("gfx942", 80): _mhc_fused_config_gfx942_80,
    ("gfx1250", 256): _mhc_fused_config_gfx1250_256,
}


@functools.lru_cache(maxsize=1024)
def get_mhc_fused_post_pre_config(
    m: int,
    hidden_size: int,
    w_preshuffle_bf16: bool = False,
    res_preshuffle: bool = False,
) -> tuple[int, int, int, int]:
    """Select (split_k, tile_m, tile_n, tile_k) for the fused post+pre GEMM.

    Looks up a per-chip tuned policy keyed by (gfx_arch, cu_num); falls back to a
    conservative default for untuned chips. Packed BF16 with shuffled residuals
    has separate GEMM + reduction tuning. K = hidden_size per stream.
    """
    num_cu = get_cu_num()
    try:
        arch = get_gfx_runtime()
    except Exception:  # noqa: BLE001
        arch = "unknown"
    _check_mhc_res_preshuffle_arch(res_preshuffle, arch)
    policy = _MHC_FUSED_POST_PRE_CONFIG.get((arch, num_cu), _mhc_fused_config_default)
    split_k, tile_m, tile_n, tile_k = policy(m, hidden_size, num_cu)
    if (
        w_preshuffle_bf16
        and res_preshuffle
        and arch == "gfx1250"
        and num_cu == 256
        and 1 <= m <= 1024
    ):
        # Direct residual stores change the best GEMM + reduction configuration.
        # Keep the packed decode policy separate from FP32 and prefill tuning.
        if hidden_size == 7168 and 512 < m <= 768:
            tile_m = 16
        elif hidden_size == 4096 and 960 < m:
            split_k = 16
    if w_preshuffle_bf16 and res_preshuffle and arch == "gfx1250" and num_cu == 256:
        return _mhc_fused_bf16_shuffled_config_gfx1250_256(
            m, hidden_size, (split_k, tile_m, tile_n, tile_k)
        )
    return split_k, tile_m, tile_n, tile_k


def mhc_pre_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,  # if 0, only do pre for hc_head
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    large_m_splitk: bool = False,
    w_preshuffle_bf16: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = residual.size(0)
    hc_mult = residual.size(1)
    hidden_size = residual.size(2)
    device = residual.device
    post_mix = torch.empty(m, hc_mult, 1, dtype=dtypes.fp32, device=device)
    comb_mix = torch.empty(m, hc_mult, hc_mult, dtype=dtypes.fp32, device=device)
    layer_input = torch.empty(m, hidden_size, dtype=dtypes.bf16, device=device)
    return post_mix, comb_mix, layer_input


@torch_compile_guard(mutates_args=[], gen_fake=mhc_pre_fake)
def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,  # if 0, only do pre for hc_head
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    large_m_splitk: bool = False,
    w_preshuffle_bf16: int = 0,  # 1: fn is pre-packed BF16 hi/lo from mhc_pre_convert_fn
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = residual.size(0)
    hc_mult = residual.size(1)
    hidden_size = residual.size(2)
    hc_mult3 = fn.size(0)
    assert hc_mult3 == hc_mult * 2 + hc_mult * hc_mult or (
        hc_mult3 == hc_mult and sinkhorn_repeat == 0
    )
    hc_hidden_size = hc_mult * hidden_size
    # The tuned packed-weight policy targets the standard four-stream pre GEMM.
    packed_config = bool(w_preshuffle_bf16) and hc_mult == 4 and hc_mult3 == 24
    if large_m_splitk:
        selected_splitk, selected_tile_k = get_mhc_pre_splitk_large_m(
            m, hc_hidden_size, w_preshuffle_bf16=packed_config
        )
    else:
        selected_splitk, selected_tile_k = get_mhc_pre_splitk(
            m, hc_hidden_size, w_preshuffle_bf16=packed_config
        )
    device = residual.device
    out_pad = torch.empty(
        selected_splitk, m, (hc_mult3 + 31) // 32 * 32, dtype=dtypes.fp32, device=device
    )
    out = out_pad[:, :, :hc_mult3]
    sqrsum = torch.empty(selected_splitk, m, dtype=dtypes.fp32, device=device)
    # The flag selects packed weights and BF16 compute; residual keeps its plain layout.
    mhc_pre_gemm_sqrsum(
        out,
        sqrsum,
        residual,
        fn,
        selected_tile_k,
        w_preshuffle_bf16=w_preshuffle_bf16,
    )
    # out = out.sum(0)
    # sqrsum = sqrsum.sum(0)

    post_mix = torch.empty(m, hc_mult, 1, dtype=dtypes.fp32, device=device)
    comb_mix = torch.empty(m, hc_mult, hc_mult, dtype=dtypes.fp32, device=device)
    layer_input = torch.empty(m, hidden_size, dtype=dtypes.bf16, device=device)
    if norm_weight is not None:
        mhc_pre_big_fuse_rmsnorm(
            post_mix,
            comb_mix,
            layer_input,
            out,
            sqrsum,
            hc_scale,
            hc_base,
            residual,
            norm_weight,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            norm_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    else:
        mhc_pre_big_fuse(
            post_mix,
            comb_mix,
            layer_input,
            out,
            sqrsum,
            hc_scale,
            hc_base,
            residual,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )

    return post_mix, comb_mix, layer_input


@compile_ops("module_mhc", develop=True)
def mhc_post(
    out: Tensor,
    x: Tensor,
    residual: Tensor,
    post_layer_mix: Tensor,
    comb_res_mix: Tensor,
    store_nt: int = -1,
) -> None: ...


def get_mhc_pre_splitk_large_m(
    m: int, hc_hidden_size: int, w_preshuffle_bf16: bool = False
) -> tuple[int, int]:
    """Split-K policy for gfx950 large-M post_pre kernel (M > 1024)."""
    if get_gfx_runtime() == "gfx950" and m >= 8192 and hc_hidden_size % (8 * 64) == 0:
        return 8, 64
    return get_mhc_pre_splitk(
        m, hc_hidden_size, w_preshuffle_bf16=w_preshuffle_bf16
    )


@compile_ops("module_mhc", develop=True)
def mhc_fused_post_pre_gemm_sqrsum(
    gemm_out_mul: Tensor,
    gemm_out_sqrsum: Tensor,
    next_residual: Tensor,
    layer_input: Tensor,
    residual_in: Tensor,
    post_layer_mix: Tensor,
    comb_res_mix: Tensor,
    fn: Tensor,
    tile_m: int = 16,  # 16, 32 or 64
    tile_n: int = 32,  # 16 or 32
    tile_k: int = 32,  # 32 or 64
    w_preshuffle_bf16: int = 0,  # 0: FP32; 1: packed BF16 hi/lo
    res_preshuffle: int = 0,  # 0: plain; 1: shuffled residual (gfx1250 only)
) -> None: ...


def mhc_fused_post_pre_fake(
    layer_input: torch.Tensor,
    residual_in: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    force_fused: bool = False,
    w_preshuffle_bf16: bool = False,  # True: packed BF16 hi/lo
    res_preshuffle: bool = False,  # True: shuffled residual (gfx1250 only)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m = layer_input.size(0)
    hc_mult = residual_in.size(1)
    hidden_size = residual_in.size(2)
    device = layer_input.device
    post_mix = torch.empty(m, hc_mult, 1, dtype=dtypes.fp32, device=device)
    comb_mix = torch.empty(m, hc_mult, hc_mult, dtype=dtypes.fp32, device=device)
    layer_input_out = torch.empty(m, hidden_size, dtype=dtypes.bf16, device=device)
    next_residual = torch.empty_like(residual_in)
    return post_mix, comb_mix, layer_input_out, next_residual


@torch_compile_guard(mutates_args=[], gen_fake=mhc_fused_post_pre_fake)
def mhc_fused_post_pre_large_m(
    layer_input: torch.Tensor,
    residual_in: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    w_preshuffle_bf16: bool = False,  # True: packed BF16 hi/lo
    res_preshuffle: bool = False,  # plain residuals only for this entry point
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """gfx950 large-M post+pre (M > 1024): upstream ``mhc_post`` + ``mhc_pre``."""
    if res_preshuffle:
        raise ValueError(
            "mhc_fused_post_pre_large_m requires res_preshuffle=0; "
            "shuffled residuals require mhc_fused_post_pre on gfx1250"
        )
    m = residual_in.size(0)

    if post_layer_mix.ndim == 3 or not post_layer_mix.is_contiguous():
        post_layer_mix = post_layer_mix.contiguous()
    if not comb_res_mix.is_contiguous():
        comb_res_mix = comb_res_mix.contiguous()
    if not residual_in.is_contiguous():
        residual_in = residual_in.contiguous()
    if not layer_input.is_contiguous():
        layer_input = layer_input.contiguous()
    if not fn.is_contiguous():
        fn = fn.contiguous()
    if norm_weight is not None and not norm_weight.is_contiguous():
        norm_weight = norm_weight.contiguous()

    next_residual = torch.empty_like(residual_in)
    post_store_nt = 0 if m > 8 * get_cu_num() else -1
    mhc_post(
        next_residual,
        layer_input,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        post_store_nt,
    )
    post_mix, comb_mix, layer_input_out = mhc_pre(
        next_residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_weight,
        norm_eps,
        large_m_splitk=True,
        w_preshuffle_bf16=int(w_preshuffle_bf16),
    )
    return post_mix, comb_mix, layer_input_out, next_residual


@torch_compile_guard(mutates_args=[], gen_fake=mhc_fused_post_pre_fake)
def mhc_fused_post_pre(
    layer_input: torch.Tensor,
    residual_in: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    force_fused: bool = False,
    w_preshuffle_bf16: bool = False,  # True: packed BF16 hi/lo
    res_preshuffle: bool = False,  # True: shuffled residual (gfx1250 only)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused mhc_post + next mhc_pre (HIP), mirroring ``mhc_pre`` with post-step inputs.

    Post step (from preceding layer's pre):
        ``layer_input`` (attn/ffn output), ``residual_in``, ``post_layer_mix``, ``comb_res_mix``.

    Pre step (next layer): same ``fn`` / ``hc_scale`` / ``hc_base`` as ``mhc_pre``.

    Returns ``(post_mix, comb_mix, layer_input_out, next_residual)`` -- next pre mixes,
    folded layer input, and the new residual stream for the following layer's post.

    ``force_fused``: when True, select the fused HIP path, except for the existing
    gfx950 large-M post+pre specialization with plain residuals. When False
    (default), larger ``m`` with plain residuals falls back to ``mhc_post`` +
    ``mhc_pre`` (threshold depends on the detected GPU arch).
    Shuffled residuals always use the fused kernel, which understands that layout.

    ``w_preshuffle_bf16=True`` selects BF16 hi/lo compute and requires ``fn`` from
    ``mhc_pre_convert_fn``. ``res_preshuffle=True`` independently declares that
    ``residual_in`` is shuffled (gfx1250 only); ``next_residual`` uses the same
    layout. Neither flag converts inputs. For BF16 with plain residuals, pass
    ``w_preshuffle_bf16=True, res_preshuffle=False``.

    Both flags are boolean, default to False, and are controlled independently.
    """
    m = layer_input.size(0)
    hc_mult = residual_in.size(1)
    hidden_size = residual_in.size(2)
    arch = get_gfx_runtime()
    _check_mhc_res_preshuffle_arch(res_preshuffle, arch)
    fused_m_upper_bound = {
        "gfx950": 1024,
        "gfx942": 128,
        "gfx1250": 1024,
    }.get(arch, 1024)

    # Plain post/pre fallbacks cannot consume shuffled buffers. BF16 arithmetic
    # itself is supported by both paths and must not prevent the fallback.
    if not force_fused and not res_preshuffle and m >= fused_m_upper_bound:
        next_residual = torch.empty_like(residual_in)
        mhc_post(
            next_residual,
            layer_input,
            residual_in,
            post_layer_mix,
            comb_res_mix,
        )
        post_mix, comb_mix, layer_input_out = mhc_pre(
            next_residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_weight,
            norm_eps,
            w_preshuffle_bf16=int(w_preshuffle_bf16),
        )
        return post_mix, comb_mix, layer_input_out, next_residual

    if force_fused and not res_preshuffle and arch == "gfx950" and m > fused_m_upper_bound:
        return mhc_fused_post_pre_large_m(
            layer_input,
            residual_in,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_weight,
            norm_eps,
            w_preshuffle_bf16=w_preshuffle_bf16,
            res_preshuffle=False,
        )

    assert layer_input.shape == (
        m,
        hidden_size,
    ), f"layer_input shape mismatch: expected ({m}, {hidden_size}), got {tuple(layer_input.shape)}"
    assert residual_in.shape == (m, hc_mult, hidden_size), (
        f"residual_in shape mismatch: expected ({m}, {hc_mult}, {hidden_size}), "
        f"got {tuple(residual_in.shape)}"
    )
    hc_hidden_size = hc_mult * hidden_size
    hc_mult3 = fn.size(0)
    assert hc_mult3 == hc_mult * 2 + hc_mult * hc_mult or (
        hc_mult3 == hc_mult and sinkhorn_repeat == 0
    )
    assert fn.size(1) == hc_hidden_size

    if post_layer_mix.ndim == 3:
        post_layer_mix = post_layer_mix.squeeze(-1)
    assert post_layer_mix.shape == (
        m,
        hc_mult,
    ), f"post_layer_mix shape mismatch: expected ({m}, {hc_mult}), got {tuple(post_layer_mix.shape)}"
    assert comb_res_mix.shape == (m, hc_mult, hc_mult), (
        f"comb_res_mix shape mismatch: expected ({m}, {hc_mult}, {hc_mult}), "
        f"got {tuple(comb_res_mix.shape)}"
    )

    selected_splitk, selected_tile_m, selected_tile_n, selected_tile_k = (
        get_mhc_fused_post_pre_config(
            m,
            hidden_size,
            w_preshuffle_bf16=w_preshuffle_bf16,
            res_preshuffle=res_preshuffle,
        )
    )
    n_splits = selected_splitk
    device = layer_input.device

    gemm_out_pad = torch.empty(
        n_splits, m, (hc_mult3 + 31) // 32 * 32, dtype=dtypes.fp32, device=device
    )
    gemm_out = gemm_out_pad[:, :, :hc_mult3]
    gemm_out_sqrsum = torch.empty(n_splits, m, dtype=dtypes.fp32, device=device)
    next_residual = torch.empty_like(residual_in)

    mhc_fused_post_pre_gemm_sqrsum(
        gemm_out,
        gemm_out_sqrsum,
        next_residual,
        layer_input,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        fn,
        selected_tile_m,
        selected_tile_n,
        selected_tile_k,
        w_preshuffle_bf16=int(w_preshuffle_bf16),
        res_preshuffle=int(res_preshuffle),
    )

    post_mix = torch.empty(m, hc_mult, 1, dtype=dtypes.fp32, device=device)
    comb_mix = torch.empty(m, hc_mult, hc_mult, dtype=dtypes.fp32, device=device)
    layer_input_out = torch.empty(m, hidden_size, dtype=dtypes.bf16, device=device)
    if norm_weight is not None:
        mhc_pre_big_fuse_rmsnorm(
            post_mix,
            comb_mix,
            layer_input_out,
            gemm_out,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            next_residual,
            norm_weight,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            norm_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            res_preshuffle=int(res_preshuffle),
        )
    else:
        mhc_pre_big_fuse(
            post_mix,
            comb_mix,
            layer_input_out,
            gemm_out,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            next_residual,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            res_preshuffle=int(res_preshuffle),
        )

    return post_mix, comb_mix, layer_input_out, next_residual
