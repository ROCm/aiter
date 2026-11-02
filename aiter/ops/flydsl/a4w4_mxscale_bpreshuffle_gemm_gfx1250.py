# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 FlyDSL backend for the MXFP4 (a4w4) bpreshuffle GEMM.

``A [M, K / 2]`` and ``B [N, K / 2]`` are packed fp4x2 (E2M1); ``B`` is 16x16
preshuffled (``shuffle_weight``). Both E8M0 block scales are 32x4 shuffled
(``shuffle_mxfp8fp4_scale``) and shaped ``[pad32(rows), K / 32]``, the layout the
gfx1250 ASM F4GEMM reads too. A is row-major (no A preshuffle).
"""

from __future__ import annotations

import functools
import re

import torch
from torch import Tensor

GENERAL_NAME_PREFIX = "flydsl_a4w4_bpreshuffle_wmma"
COMPUTE_NAME_PREFIX = "flydsl_a4w4_bpreshuffle_compute_wmma"
_NAME_SUFFIX_RE = (
    r"t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"mw(?P<m_warp>\d+)_nw(?P<n_warp>\d+)_"
    r"nb(?P<num_buffers>\d+)_sk(?P<split_k>\d+)_"
    r"cm(?P<cluster_m>\d+)_cn(?P<cluster_n>\d+)"
    r"(?P<fused_splitk>_fsk)?"
)
_GENERAL_NAME_RE = re.compile(rf"^{re.escape(GENERAL_NAME_PREFIX)}_{_NAME_SUFFIX_RE}$")
_COMPUTE_NAME_RE = re.compile(rf"^{re.escape(COMPUTE_NAME_PREFIX)}_{_NAME_SUFFIX_RE}$")

_OUT_DTYPE_NAME = {torch.bfloat16: "bf16", torch.float16: "f16"}
_MAX_SPLIT_K = 8
_MAX_CLUSTER = 16
SCALE_BLOCK = 32
# Geometry of kernels/splitk_fused_epilogue_gfx1250.py (``vec``, ``UNROLL``, ``MAX_PARTIAL_VECTORS``),
# repeated here because this module must import without FlyDSL.
_EPILOGUE_VEC = 4
_EPILOGUE_UNROLL = 32
_EPILOGUE_MAX_PARTIAL_VECTORS = 128

# The kernels read A / B / the scales and write Out as 16-byte vectors (TDM and
# buffer copies), so every base address and row pitch must be a multiple of this.
ALIGN_BYTES = 16

_launch_general = None
_launch_compute = None
_run_compiled = None
_flyc = None
_fx = None


def _lazy_import():
    global _launch_general, _launch_compute, _run_compiled, _flyc, _fx
    if _launch_general is not None:
        return
    import flydsl.compiler as flyc_mod
    import flydsl.expr as fx_mod

    from .kernels.gemm_a4w4_256x256_gfx1250 import launch_gemm_a4w4_256x256
    from .kernels.gemm_a4w4_gfx1250 import launch_gemm_a4w4
    from .kernels.tensor_shim import _run_compiled as run_compiled

    _launch_general = launch_gemm_a4w4
    _launch_compute = launch_gemm_a4w4_256x256
    _run_compiled = run_compiled
    _flyc = flyc_mod
    _fx = fx_mod


def _ptr_arg(t: Tensor):
    # Declare the alignment the kernels rely on (the kernel recasts the output pointer
    # to bf16/fp32, which needs more than the byte alignment an untyped pointer has).
    return _flyc.from_c_void_p(_fx.Uint8, t.data_ptr(), assumed_align=ALIGN_BYTES)


def a4w4_operands_aligned(
    A: Tensor, B: Tensor, A_scale: Tensor, B_scale: Tensor
) -> bool:
    """Whether A / B / the scales meet the kernels' 16-byte alignment, so a caller
    can fall back to another kernel instead of reading past a misaligned view.

    The scales are read in 32-row groups of K bytes (aligned, as K % 128 == 0), not
    by their K / 32 byte row stride, so only their base addresses are checked."""
    if any(t.data_ptr() % ALIGN_BYTES for t in (A, B, A_scale, B_scale)):
        return False
    return A.dim() != 2 or A.stride(0) % ALIGN_BYTES == 0


def is_compute_a4w4_kernel_name(name: str) -> bool:
    return _COMPUTE_NAME_RE.fullmatch(name) is not None


def is_a4w4_flydsl_kernel_name(name: str) -> bool:
    return bool(name) and (
        _COMPUTE_NAME_RE.fullmatch(name) is not None
        or _GENERAL_NAME_RE.fullmatch(name) is not None
    )


@functools.lru_cache(maxsize=1024)
def parse_a4w4_kernel_name(name: str) -> dict | None:
    """Parse a general or compute-bound a4w4 kernelName, None if it is neither."""
    match = _COMPUTE_NAME_RE.fullmatch(name) or _GENERAL_NAME_RE.fullmatch(name)
    if match is None:
        return None
    groups = match.groupdict()
    fused_splitk = groups.pop("fused_splitk") is not None
    cfg = {key: int(value) for key, value in groups.items()}
    cfg["fused_splitk"] = fused_splitk
    cfg["compute_bound"] = _COMPUTE_NAME_RE.fullmatch(name) is not None
    return cfg


def a4w4_kernel_name(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    *,
    m_warp: int = 2,
    n_warp: int = 2,
    num_buffers: int = 4,
    split_k: int = 1,
    cluster_m: int = 1,
    cluster_n: int = 1,
    compute_bound: bool = True,
) -> str:
    """The tuned-CSV kernelName of one launch configuration."""
    prefix = COMPUTE_NAME_PREFIX if compute_bound else GENERAL_NAME_PREFIX
    fused = "_fsk" if compute_bound and split_k > 1 else ""
    return (
        f"{prefix}_t{tile_m}x{tile_n}x{tile_k}_mw{m_warp}_nw{n_warp}"
        f"_nb{num_buffers}_sk{split_k}_cm{cluster_m}_cn{cluster_n}{fused}"
    )


def _require_scale(scale: Tensor, rows: int, k_blocks: int, name: str) -> Tensor:
    if scale.dim() != 2 or scale.shape[0] < rows or scale.shape[1] != k_blocks:
        raise RuntimeError(
            f"[FlyDSL gfx1250 a4w4] {name} must be [>= {rows}, {k_blocks}] 32x4 "
            f"shuffled e8m0, got {tuple(scale.shape)}"
        )
    if scale.element_size() != 1:
        raise RuntimeError(f"[FlyDSL gfx1250 a4w4] {name} must be 1-byte e8m0")
    if not scale.is_contiguous():
        raise RuntimeError(f"[FlyDSL gfx1250 a4w4] {name} must be contiguous")
    return scale


def fused_splitk_partial_numel(
    M: int,
    N: int,
    tile_m: int,
    tile_n: int,
    cluster_m: int,
    cluster_n: int,
    split_k: int,
) -> int:
    """fp32 scratch elements the clustered fused split-K epilogue exchanges."""
    m_tiles = -(-M // tile_m)
    padded_m = -(-m_tiles // cluster_m) * cluster_m
    padded_n = -(-N // (tile_n * cluster_n)) * cluster_n
    scratch_rows = tile_m // 2 if split_k == 2 else tile_m
    return padded_m * padded_n * split_k * scratch_rows * tile_n


def a4w4_fit_cluster_m(cfg: dict, M: int) -> int | None:
    """The cluster_m to launch with for this M, None when no cluster fits.

    A cluster must not hold workgroups that own no M row (they can deadlock it), so the
    tuned cluster_m is kept only when whole cluster rows tile M; otherwise the largest
    smaller cluster_m that does is used, as the a8w8 wrapper does. A compute kernel still
    needs a real cluster (cluster_m * cluster_n >= 2)."""
    m_tiles = -(-M // cfg["tile_m"])
    cn = cfg["cluster_n"]
    for cm in range(cfg["cluster_m"], 0, -1):
        if m_tiles % cm == 0 and (not cfg["compute_bound"] or cm * cn >= 2):
            return cm
    return None


def _splitk_epilogue_error(
    tile_m: int, tile_n: int, split_k: int, block: int
) -> str | None:
    """Why the fused split-K epilogue cannot reduce a tile, None when it can."""
    lanes = tile_n // _EPILOGUE_VEC
    if tile_n % _EPILOGUE_VEC or lanes == 0 or block % lanes or block % 128:
        return f"split-K epilogue needs tile_n/4={lanes} to divide the {block}-thread block"
    rows_per_iter = block // lanes
    reduce_m = tile_m // split_k
    unroll = min(
        _EPILOGUE_UNROLL,
        _EPILOGUE_MAX_PARTIAL_VECTORS // (block // 128) // split_k,
        reduce_m // rows_per_iter,
    )
    if unroll < 1 or reduce_m % (rows_per_iter * unroll):
        return (
            f"split-K epilogue would skip rows: {reduce_m} rows per split, "
            f"{rows_per_iter} per iteration x unroll {unroll}"
        )
    waves = block // 32
    if reduce_m % waves or (tile_m - reduce_m) % waves:
        return f"split-K epilogue stores {reduce_m} / {tile_m - reduce_m} rows over {waves} waves"
    return None


def a4w4_launch_error(cfg: dict, M: int, N: int, K: int) -> str | None:
    """Why the kernel a parsed kernelName names cannot launch for (M, N, K), None when it can.

    One rule set for the launcher, the tuned-row lookup and the tuner. It mirrors the
    asserts of the two FlyDSL launchers. An M below tile_m is one partial M tile: the A /
    scale-A loads and the C store are bounded by M, and :func:`a4w4_fit_cluster_m` keeps
    cluster_m at 1 there, so no workgroup owns no M row.
    """
    tile_m, tile_n, tile_k = cfg["tile_m"], cfg["tile_n"], cfg["tile_k"]
    mw, nw, nb = cfg["m_warp"], cfg["n_warp"], cfg["num_buffers"]
    sk, cm, cn = cfg["split_k"], cfg["cluster_m"], cfg["cluster_n"]
    if M < 1 or N % 16 != 0 or K % 128 != 0:
        return (
            "needs M >= 1, N % 16 == 0 and K % 128 == 0 (the 32x4 scale layout), "
            f"got M={M}, N={N}, K={K}"
        )
    if a4w4_fit_cluster_m(cfg, M) is None:
        return f"no cluster_m <= {cfg['cluster_m']} tiles M={M} in whole cluster rows"
    if not (1 <= cm < 16 and 1 <= cn < 16) or cm * cn * sk > _MAX_CLUSTER:
        return f"cluster {cm}x{cn} with split_k={sk} exceeds one {_MAX_CLUSTER}-workgroup cluster"
    if sk not in (1, 2, 4, 8) or (sk > 1 and tile_m % sk):
        return f"split_k={sk} must be 1/2/4/8 and divide tile_m={tile_m}"
    if sk > 1:
        epilogue_error = _splitk_epilogue_error(tile_m, tile_n, sk, 32 * mw * nw)
        if epilogue_error is not None:
            return epilogue_error
    if cfg["compute_bound"]:
        if (mw, nw, nb) != (2, 2, 4) or tile_m not in (128, 192, 256):
            return "unsupported compute tile profile"
        if sk > 1 and tile_m & (tile_m - 1):
            return f"compute fused split-K needs a 2^n tile_m, got {tile_m}"
        if tile_n not in (128, 256) or tile_k not in (128, 256):
            return "unsupported compute tile profile"
        if cm * cn < 2:
            return f"compute needs a real cluster, got {cm}x{cn}"
        if N % (tile_n * cn) != 0:
            return f"N={N} must be a multiple of tile_n*cluster_n={tile_n * cn}"
        if K % (tile_k * sk) != 0 or K // sk < tile_k:
            return f"K={K} must be a multiple of tile_k*split_k={tile_k * sk}"
        if sk > 1 and not cfg["fused_splitk"]:
            return "compute split_k > 1 needs the fused epilogue (_fsk kernelName)"
        return None
    if tile_m % (16 * mw) or tile_n % (32 * nw):
        return f"tile {tile_m}x{tile_n} does not divide into {mw}x{nw} warps"
    if (tile_m // mw) % 32 and not (tile_m == 16 and mw == 1):
        return f"warp tile rows {tile_m // mw} must be a multiple of 32 (or tile_m=16, 1 warp row)"
    if tile_k not in (128, 256, 512, 1024) or not 2 <= nb <= 4:
        return f"unsupported tile_k={tile_k} / num_buffers={nb}"
    if sk > 1 and tile_m // sk < mw * nw * 128 // tile_n:
        return f"split_k={sk} leaves too few rows per split for the epilogue"
    return None


def a4w4_kernel_launchable(kernel_name: str, M: int, N: int, K: int) -> bool:
    """Whether the tuned kernelName can launch for (M, N, K); see :func:`a4w4_launch_error`."""
    cfg = parse_a4w4_kernel_name(kernel_name)
    return cfg is not None and a4w4_launch_error(cfg, M, N, K) is None


def _launch_specialization(
    cfg: dict, M: int, cluster_m: int, out_is_f16: int, bounded_m: bool | None
):
    """(launcher, constexpr arguments) of a kernel config for this M.

    ``bounded_m`` is the compute kernel's masked-M variant, needed when M does not fill
    whole tiles; None derives it from M."""
    tile_m, tile_n, tile_k = cfg["tile_m"], cfg["tile_n"], cfg["tile_k"]
    m_warp, n_warp, nb = cfg["m_warp"], cfg["n_warp"], cfg["num_buffers"]
    split_k, cluster_n = cfg["split_k"], cfg["cluster_n"]
    common = (tile_m, tile_n, tile_k, m_warp, n_warp, out_is_f16, nb)
    if cfg["compute_bound"]:
        if bounded_m is None:
            bounded_m = bool(M % tile_m or (-(-M // tile_m)) % cluster_m)
        return _launch_compute, (
            *common,
            cluster_m,
            cluster_n,
            split_k,
            False,  # a_preshuffle
            split_k > 1,  # fused_splitk
            bounded_m,
        )
    return _launch_general, (*common, cluster_m, cluster_n, split_k)


def preload_a4w4_kernel(
    kernel_name: str,
    M: int,
    N: int,
    K: int,
    out_dtype: torch.dtype = torch.bfloat16,
    bounded_m: bool | None = None,
) -> None:
    """Compile a kernel config into FlyDSL's disk cache without launching it.

    The runtime arguments (M, N, K, pointers) are not part of the compiled artifact, so one
    compile serves every shape the config can launch for. ``bounded_m`` picks the compute
    kernel's masked-M variant (None: derived from M)."""
    cfg = parse_a4w4_kernel_name(kernel_name)
    if cfg is None:
        raise ValueError(
            f"[FlyDSL gfx1250 a4w4] unrecognised kernelName {kernel_name!r}"
        )
    err = a4w4_launch_error(cfg, M, N, K)
    if err is not None:
        raise RuntimeError(f"[FlyDSL gfx1250 a4w4] {kernel_name!r}: {err}")
    _lazy_import()
    from .kernels.tensor_shim import _preload_compiled

    out_is_f16 = 1 if _OUT_DTYPE_NAME[out_dtype] == "f16" else 0
    launch, specialization = _launch_specialization(
        cfg, M, a4w4_fit_cluster_m(cfg, M), out_is_f16, bounded_m
    )
    null = lambda: _flyc.from_c_void_p(_fx.Uint8, 0, assumed_align=ALIGN_BYTES)
    runtime_args = (
        null(),
        null(),
        null(),
        null(),
        null(),
        M,
        _fx.Stream(0),
        N,
        K,
        K,
        N,
        null(),
    )
    _preload_compiled(launch, *runtime_args, *specialization)


def run_gemm_a4w4_mxscale_bpreshuffle_gfx1250(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    Out: Tensor,
    kernel_name: str,
) -> Tensor:
    """``Out = dequant(A) @ dequant(B).T`` on the FlyDSL kernel named by a tuned
    row; writes into ``Out`` ([M, N] bf16 / fp16) and returns it.

    A: ``[M, K / 2]`` fp4x2, B: ``[N, K / 2]`` fp4x2 16x16-preshuffled,
    A_scale / B_scale: e8m0 ``[pad32(M), K / 32]`` / ``[pad32(N), K / 32]`` in 32x4
    shuffled storage.
    """
    cfg = parse_a4w4_kernel_name(kernel_name)
    if cfg is None:
        raise ValueError(
            f"[FlyDSL gfx1250 a4w4] unrecognised kernelName {kernel_name!r}"
        )
    tile_m, tile_n = cfg["tile_m"], cfg["tile_n"]
    split_k, cluster_m, cluster_n = cfg["split_k"], cfg["cluster_m"], cfg["cluster_n"]

    if A.dim() != 2 or B.dim() != 2 or Out.dim() != 2:
        raise RuntimeError(
            "[FlyDSL gfx1250 a4w4] A/B/Out must be 2-D, got "
            f"{tuple(A.shape)}, {tuple(B.shape)}, {tuple(Out.shape)}"
        )
    if A.element_size() != 1 or B.element_size() != 1:
        raise RuntimeError("[FlyDSL gfx1250 a4w4] A/B must be packed 1-byte fp4x2")
    M, N, K = A.shape[0], B.shape[0], A.shape[1] * 2
    if B.shape[1] * 2 != K:
        raise RuntimeError(
            f"[FlyDSL gfx1250 a4w4] K mismatch: A has K={K}, B has K={B.shape[1] * 2}"
        )
    if tuple(Out.shape) != (M, N):
        raise RuntimeError(
            f"[FlyDSL gfx1250 a4w4] Out must be [{M}, {N}], got {tuple(Out.shape)}"
        )
    out_dtype = _OUT_DTYPE_NAME.get(Out.dtype)
    if out_dtype is None:
        raise RuntimeError(
            f"[FlyDSL gfx1250 a4w4] unsupported out dtype {Out.dtype}; expected bf16/fp16"
        )
    err = a4w4_launch_error(cfg, M, N, K)
    if err is not None:
        raise RuntimeError(f"[FlyDSL gfx1250 a4w4] {kernel_name!r}: {err}")
    cluster_m = a4w4_fit_cluster_m(cfg, M)
    if A.stride(1) != 1 or Out.stride(1) != 1:
        raise RuntimeError("[FlyDSL gfx1250 a4w4] A and Out rows must be contiguous")
    if not B.is_contiguous():  # the kernels take no B stride, only K
        raise RuntimeError("[FlyDSL gfx1250 a4w4] B must be contiguous")
    if split_k > 1 and Out.stride(0) != N and (Out.stride(0) & 7 or N & 7):
        raise RuntimeError(
            "[FlyDSL gfx1250 a4w4] split_k > 1 with padded Out rows needs N and "
            f"stride(0) to be multiples of 8, got N={N}, stride(0)={Out.stride(0)}"
        )

    if not a4w4_operands_aligned(A, B, A_scale, B_scale):
        raise RuntimeError(
            f"[FlyDSL gfx1250 a4w4] A, B and the scales need {ALIGN_BYTES}-byte aligned "
            "base addresses, and A an aligned row pitch"
        )
    if (
        Out.data_ptr() % ALIGN_BYTES
        or (Out.stride(0) * Out.element_size()) % ALIGN_BYTES
    ):
        raise RuntimeError(
            f"[FlyDSL gfx1250 a4w4] Out needs a {ALIGN_BYTES}-byte aligned base address "
            f"and row pitch, got stride(0)={Out.stride(0)}"
        )

    k_blocks = K // SCALE_BLOCK
    a_scale = _require_scale(A_scale, (M + 31) // 32 * 32, k_blocks, "A_scale")
    b_scale = _require_scale(B_scale, (N + 31) // 32 * 32, k_blocks, "B_scale")

    _lazy_import()
    torch_stream = torch.cuda.current_stream(device=A.device)
    stream = _fx.Stream(torch_stream)
    ldc = Out.stride(0)
    lda = A.stride(0) * 2  # the kernels take A's row stride in K elements
    if split_k > 1:
        partials = torch.empty(
            fused_splitk_partial_numel(
                M, N, tile_m, tile_n, cluster_m, cluster_n, split_k
            ),
            dtype=torch.float32,
            device=Out.device,
        )
    else:
        partials = Out
    out_is_f16 = 1 if out_dtype == "f16" else 0
    runtime_args = (
        _ptr_arg(partials),
        _ptr_arg(A),
        _ptr_arg(B),
        _ptr_arg(a_scale),
        _ptr_arg(b_scale),
        M,
        stream,
        N,
        K,
        lda,
        ldc,
        _ptr_arg(Out),
    )
    launch, specialization = _launch_specialization(
        cfg, M, cluster_m, out_is_f16, bounded_m=None
    )
    _run_compiled(
        launch, *runtime_args, *specialization, specialization_key=specialization
    )
    return Out


__all__ = [
    "ALIGN_BYTES",
    "COMPUTE_NAME_PREFIX",
    "GENERAL_NAME_PREFIX",
    "a4w4_fit_cluster_m",
    "a4w4_kernel_launchable",
    "a4w4_kernel_name",
    "a4w4_launch_error",
    "a4w4_operands_aligned",
    "fused_splitk_partial_numel",
    "is_a4w4_flydsl_kernel_name",
    "is_compute_a4w4_kernel_name",
    "parse_a4w4_kernel_name",
    "preload_a4w4_kernel",
    "run_gemm_a4w4_mxscale_bpreshuffle_gfx1250",
]
