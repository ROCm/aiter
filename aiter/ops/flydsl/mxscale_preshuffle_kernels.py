# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host dispatcher for the FlyDSL MXFP4/MXFP6/MXFP8 preshuffle GEMM (gfx950 MFMA).

Operand convention (all preshuffling is caller-side, as for the weight itself):
    A       : [M, K]   row-major, NOT preshuffled  (fp8/fp6 = 1 byte/code, fp4 = 2 codes/byte)
    B       : preshuffled via aiter.ops.shuffle.shuffle_weight(., (16, 16))  (fp4 or fp8 weight)
    a_scale : blockscale -> aiter.ops.shuffle.shuffle_scale_blockscale_a(a_1x128, K)
              MX         -> per-1x32 E8M0, shuffle_scale_a16w4'd
    b_scale : blockscale -> aiter.ops.shuffle.shuffle_scale_blockscale_b(b_128x128, N, K)
              MX         -> per-1x32 E8M0, shuffle_scale_a16w4'd
    Out     : [M, N]   bf16 / fp16

``shuffle_scale_blockscale_a`` takes the LOGICAL ``(M, K//128)`` scale, not the
transposed-byte ``x_scale_t`` spelling the ck/asm blockscale kernels read. The two
are indistinguishable by shape/stride/dtype, so handing it the latter silently
computes garbage.

``run_gemm_a8w8_mxscale_preshuffle_gfx950`` is the entry point that
``gemm_a8w8_blockscale_bpreshuffle`` and the tuner share. It VALIDATES the scale
buffers rather than shuffling them: b_scale is a weight, shuffled once at
weight-prep time, and a_scale is an activation, shuffled per call next to
quantization -- so neither belongs on the GEMM's critical path.
"""

from __future__ import annotations

import functools

import torch

from aiter.ops.flydsl.utils import is_flydsl_available

_OUT_DTYPE_STR = {torch.bfloat16: "bf16", torch.float16: "fp16"}


@functools.cache
def _gemm_exe(_cfg):
    import flydsl.compiler as flyc

    from .kernels.mxscale_preshuffle import launch_gemm

    return flyc.jit(launch_gemm.func)


@functools.cache
def _reduce_exe(_cfg):
    import flydsl.compiler as flyc

    from .kernels.mxscale_preshuffle import launch_splitk_reduce

    return flyc.jit(launch_splitk_reduce.func)


def flydsl_mxscale_preshuffle_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    Out: torch.Tensor,
    *,
    a_dtype: str,
    b_dtype: str = "fp4",
    tile_m: int,
    tile_n: int,
    tile_k: int,
    waves_per_eu: int = 0,
    xcd_swizzle: int = 0,
    split_k: int = 1,
    blockscale: bool = True,
    stream=None,
) -> torch.Tensor:
    """Run the gfx950 MXFP4/6/8 preshuffle GEMM. a8w8 = a_dtype="fp8", b_dtype="fp8".

    A is [M, K]; N is taken from Out ([M, N]); K from A. Returns Out.

    split_k>1 splits the K reduction across grid.z: each split writes an fp32
    partial slab to a scratch tmp[split_k, M, N], then a reduce kernel sums the
    slabs into Out (bf16/fp16). Helps small-M / large-K (low-occupancy) shapes.

    blockscale selects the scale format and **defaults to True** -- the coarse
    blockscale path is the one this op is tuned for. It is a8w8-only and needs
    N%128==0; fp4/fp6 operands (a4w4 / a6w4) must pass blockscale=False.

    * blockscale=True: a_scale/b_scale are the *compact-shuffled* blockscale buffers
      the caller built via aiter.ops.shuffle.shuffle_scale_blockscale_a/_b from the
      coarse E8M0 (A 1x128 [rows, K//128], B 128x128 [N//128, K//128]).
    * blockscale=False: a_scale/b_scale are per-1x32 E8M0 run through
      `shuffle_scale_a16w4`.

    Either way the op does NOT repack -- the caller pre-shuffles. For blockscale the
    kernel broadcasts to the 1x32 scaled-MFMA via the scale load address. Prepare the
    static B-scale at weight-prep time to keep it off the per-call path; the per-token
    A-scale is a plain reshape+permute, so build it on device (a host round-trip
    costs more than this GEMM).
    """
    if not is_flydsl_available():
        raise RuntimeError(
            "flydsl is not available; cannot run mxscale_preshuffle GEMM"
        )

    from .kernels.tensor_shim import _run_compiled, ptr_arg

    # Logical K: fp4 A packs 2 codes/byte (A last dim = K//2); fp6/fp8 A = 1 byte/code.
    if a_dtype not in ("fp4", "fp6", "fp8"):
        raise ValueError(
            f"unsupported a_dtype {a_dtype!r}; expected 'fp4', 'fp6', or 'fp8'"
        )
    if b_dtype not in ("fp4", "fp8"):
        raise ValueError(f"unsupported b_dtype {b_dtype!r}; expected 'fp4' or 'fp8'")

    M = int(A.shape[0])
    K = int(A.shape[-1]) * (2 if a_dtype == "fp4" else 1)
    N = int(Out.shape[-1])
    if N % int(tile_n) != 0:
        raise ValueError(f"N ({N}) is not a multiple of tile_n ({tile_n})")
    if K % int(tile_k) != 0:
        raise ValueError(f"K ({K}) is not a multiple of tile_k ({tile_k})")
    if K % 128 != 0:
        raise ValueError(
            f"K ({K}) must be a multiple of 128 for MXFP microscale; got {K}"
        )
    out_dtype = _OUT_DTYPE_STR.get(Out.dtype)
    if out_dtype is None:
        raise ValueError(
            f"unsupported Out dtype {Out.dtype}; expected bfloat16 or float16"
        )

    # blockscale is the default path; an unsupported shape is an error rather than
    # a silent downgrade, so a4w4/a6w4 callers have to opt out explicitly.
    if blockscale:
        if a_dtype != "fp8" or b_dtype != "fp8":
            raise ValueError(
                f"blockscale is a8w8-only; got a_dtype={a_dtype!r} b_dtype={b_dtype!r}"
            )
        if N % 128 != 0:
            raise ValueError(f"blockscale requires N ({N}) to be a multiple of 128")
    # a_scale/b_scale are already compact-shuffled by the caller
    # (shuffle_scale_blockscale_a/_b). No per-call repack here.
    bs_mode = "ab" if blockscale else "none"

    st = stream if stream is not None else torch.cuda.current_stream()

    split_k = int(split_k)
    if split_k > 1:
        # split-K legality (same constraints the tuner enforces in fits_shape):
        # per-split K must be a whole number of tile_k tiles AND 256-K scale chunks.
        k_per_split = K // split_k
        if K % split_k != 0 or k_per_split % int(tile_k) != 0 or k_per_split % 256 != 0:
            raise ValueError(
                f"illegal split_k={split_k} for K={K}, tile_k={tile_k}: "
                f"K/split_k ({k_per_split}) must be a multiple of tile_k and 256"
            )

    # Constexpr tail of launch_gemm -- the same for the direct and the split-K launch
    # apart from k_batch, and it doubles as the per-config compiled-kernel cache key.
    cfg = (
        N,
        K,
        int(tile_m),
        int(tile_n),
        int(tile_k),
        a_dtype,
        out_dtype,
        b_dtype,
        1,  # batch
        -1,  # a_row_stride     ) each <0 keeps the contiguous
        -1,  # a_batch_stride   ) [B,M,*] default; this op never
        -1,  # sca_row_stride   ) overrides them, the batched
        -1,  # sca_batch_stride ) callers do
        -1,  # c_row_stride
        -1,  # c_batch_stride
        int(waves_per_eu),
        int(xcd_swizzle),
        split_k,  # k_batch
        bs_mode,  # blockscale
    )
    gemm_exe = _gemm_exe(cfg)

    if split_k == 1:
        _run_compiled(
            gemm_exe,
            ptr_arg(Out),
            ptr_arg(A),
            ptr_arg(B),
            ptr_arg(a_scale),
            ptr_arg(b_scale),
            M,
            N,
            st,
            *cfg,
        )
        return Out

    # split-K: GEMM -> fp32 partial slabs tmp[split_k, M, N] -> fused fp32 reduce -> Out.
    tmp = torch.empty((split_k, M, N), dtype=torch.float32, device=A.device)
    _run_compiled(
        gemm_exe,
        ptr_arg(tmp),
        ptr_arg(A),
        ptr_arg(B),
        ptr_arg(a_scale),
        ptr_arg(b_scale),
        M,
        N,
        st,
        *cfg,
    )
    _run_compiled(
        _reduce_exe((split_k, out_dtype)),
        ptr_arg(tmp),
        ptr_arg(Out),
        (M * N) // 2,  # n_out_dw (2 out elems per dword)
        M * N,  # slab_stride_dw (fp32: 1 dword/elem)
        st,
        split_k,
        out_dtype,
    )
    return Out


_HEURISTIC_MIN_TILE_N = 32


@functools.lru_cache(maxsize=1024)
def _warn_untuned(M, N, K, a_dtype, b_dtype, kernel_name):
    from aiter import logger

    logger.warning(
        f"[flydsl mxpsh] no tuned row for M={M} N={N} K={K} {a_dtype}/{b_dtype}; "
        f"falling back to the heuristic tile '{kernel_name}'. It picks a grid that "
        f"fills the CUs, but nothing about this shape was measured -- tune it with "
        f"csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py "
        f"--preshuffle --libtype flydsl."
    )


@functools.lru_cache(maxsize=1024)
def _heuristic_tile(a_dtype, b_dtype, M, N, K):
    from aiter.jit.utils.chip_info import get_cu_num

    from .gemm_tune.flydsl_gemm_mxscale_preshuffle_common import candidates_for

    cands = [ki for _, ki in candidates_for(a_dtype, b_dtype, M, N, K)]
    if not cands:
        return None

    def _wg(ki):
        return -(-M // ki.tile_m) * (N // ki.tile_n) * ki.split_k

    pool = [ki for ki in cands if ki.tile_n >= _HEURISTIC_MIN_TILE_N] or cands
    filled = [ki for ki in pool if _wg(ki) >= get_cu_num()]
    if filled:
        pool = filled
    else:  # N too narrow, or split-K legality leaves nothing that fills the CUs
        best_wg = max(_wg(ki) for ki in pool)
        pool = [ki for ki in pool if _wg(ki) == best_wg]

    target_m = min(max((M + 31) // 32 * 32, 32), 128)
    return max(
        pool,
        key=lambda ki: (
            -ki.split_k,
            ki.tile_k,
            ki.tile_n,
            -abs(ki.tile_m - target_m),
            -ki.waves_per_eu,
        ),
    )


def run_gemm_a8w8_mxscale_preshuffle_gfx950(XQ, WQ, x_scale, w_scale, Out, kernel_name):
    """Dispatch the gfx950 mxpsh GEMM by tuned kernelName.

    ``x_scale`` / ``w_scale`` must ALREADY be shuffled by
    ``aiter.ops.shuffle.shuffle_scale_blockscale_a`` / ``_b`` -- caller-side prep,
    exactly like ``shuffle_weight`` for B. B's scale is a weight, so it is shuffled
    once at weight-prep time; A's is an activation, so it is shuffled per call
    alongside quantization. Both are flat fp8_e8m0 buffers.
    """
    from .gemm_tune.flydsl_gemm_mxscale_preshuffle_common import parse_kernel_name

    p = parse_kernel_name(kernel_name)
    if p is None:
        raise ValueError(
            f"[FlyDSL gfx950 mxpsh] unrecognized kernelName {kernel_name!r}"
        )
    if (p["a_dtype"], p["b_dtype"]) != ("fp8", "fp8"):
        raise ValueError(
            f"[FlyDSL gfx950 mxpsh] {kernel_name!r} is not an a8w8 kernel; "
            f"gemm_a8w8_blockscale_bpreshuffle only serves fp8/fp8"
        )

    M, K = int(XQ.shape[0]), int(XQ.shape[-1])
    N = int(Out.shape[-1])
    k_chunks = (K + 255) // 256
    want_a = (M + 31) // 32 * 32 * 2 * k_chunks
    want_b = (N // 128) * 4 * k_chunks
    for name, s, want, helper in (
        ("x_scale", x_scale, want_a, "shuffle_scale_blockscale_a(x_scale, K)"),
        ("w_scale", w_scale, want_b, "shuffle_scale_blockscale_b(w_scale, N, K)"),
    ):
        if s.dim() != 1 or s.numel() != want:
            raise RuntimeError(
                f"[FlyDSL gfx950 mxpsh] {name} must be the shuffled flat buffer "
                f"({want} elements, 1-D), got {tuple(s.shape)}. Prepare it with "
                f"aiter.ops.shuffle.{helper}."
            )

    return flydsl_mxscale_preshuffle_gemm(
        XQ,
        WQ,
        x_scale,
        w_scale,
        Out,
        a_dtype="fp8",
        b_dtype="fp8",
        tile_m=p["tile_m"],
        tile_n=p["tile_n"],
        tile_k=p["tile_k"],
        waves_per_eu=p["waves_per_eu"],
        xcd_swizzle=p["xcd_swizzle"],
        split_k=p["split_k"],
        blockscale=True,
    )
