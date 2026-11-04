# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL GEMM APIs."""

from __future__ import annotations

import functools
import re

import flydsl.expr as fx
import torch
from torch import Tensor

from aiter import logger
from aiter.jit.utils.chip_info import get_gfx
from aiter.utility.graph_alloc import persistent_alloc

from .gemm_a16w16_gfx1250 import gemm_a16w16 as gemm_a16w16_gfx1250

# Tile candidates, validity rules and split-K buffer sizes live in the tune module so
# the tuner and this op agree.
from .gemm_tune import flydsl_gemm_a8w8_blockscale_bpreshuffle_common as bs_tune
from .kernels.gemm_a16w16_gfx950 import (
    SPLIT_K_SEMAPHORE_MAX_LEN,
    gemm_a16w16,
)
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "SPLIT_K_SEMAPHORE_MAX_LEN",
    "flydsl_gemm_a8w8_blockscale_bpreshuffle",
    "flydsl_hgemm",
    "flydsl_hgemm_kernel_name",
    "flydsl_preshuffle_gemm_a8",
    "get_flydsl_hgemm_kernel_params",
]


_HGEMM_KERNEL_RE = re.compile(
    r"^flydsl_hgemm_"
    r"a(?P<dtype>bf16|fp16|f16)_w(?P=dtype)"
    r"_(?P<out_dtype>bf16|fp16|f16|fp32)_"
    r"t(?P<block_m>\d+)x(?P<block_n>\d+)x(?P<block_k>\d+)"
    r"x(?P<stages>\d+)_ks(?P<split_k>\d+)_"
    r"w(?P<m_waves>\d+)x(?P<n_waves>\d+)x(?P<k_waves>\d+)_"
    r"bias(?P<has_bias>[01])_ktail(?P<has_k_tail>[01])_"
    r"gm(?P<group_m>\d+)_p(?P<policy>ft|ht|hti)_"
    r"(?P<target_gfx>gfx[0-9a-z]+(?:-[0-9a-z]+)*)$"
)


def _normalize_dtype_name(dtype: str | torch.dtype) -> str:
    if dtype in ("f16", "fp16", torch.float16):
        return "fp16"
    if dtype in ("bf16", torch.bfloat16):
        return "bf16"
    if dtype in ("f32", "fp32", torch.float32):
        return "fp32"
    raise ValueError(f"Unsupported FlyDSL HGEMM dtype: {dtype!r}")


def flydsl_hgemm_kernel_name(
    *,
    dtype: str | torch.dtype,
    out_dtype: str | torch.dtype,
    config: dict,
    has_bias: bool,
    target_gfx: str | None = None,
) -> str:
    """Build the stable kernel name persisted in tuned GEMM CSV files."""

    dtype_name = _normalize_dtype_name(dtype)
    out_dtype_name = _normalize_dtype_name(out_dtype)
    if dtype_name not in ("bf16", "fp16"):
        raise ValueError(f"Unsupported input dtype for HGEMM: {dtype_name}")
    if out_dtype_name not in (dtype_name, "fp32"):
        raise ValueError(
            f"Unsupported output dtype {out_dtype_name} for input {dtype_name}"
        )

    policy = "ht" if config["use_half_tile_interleaved"] else "ft"
    name = (
        f"flydsl_hgemm_a{dtype_name}_w{dtype_name}_{out_dtype_name}_"
        f"t{config['block_m']}x{config['block_n']}x{config['block_k']}"
        f"x{config['stages']}_ks{config['split_k']}_"
        f"w{config['m_waves']}x{config['n_waves']}x{config['k_waves']}_"
        f"bias{int(has_bias)}_ktail0_gm{config['group_m']}_p{policy}_"
        f"{target_gfx or get_gfx()}"
    )
    return name


def get_flydsl_hgemm_kernel_params(name: str) -> dict | None:
    """Parse a tuned FlyDSL HGEMM kernel name into the current config schema."""

    match = _HGEMM_KERNEL_RE.fullmatch(name)
    if match is None or match.group("has_k_tail") != "0":
        return None

    dtype = _normalize_dtype_name(match.group("dtype"))
    out_dtype = _normalize_dtype_name(match.group("out_dtype"))
    if out_dtype not in (dtype, "fp32"):
        return None

    return {
        "block_m": int(match.group("block_m")),
        "block_n": int(match.group("block_n")),
        "block_k": int(match.group("block_k")),
        "stages": int(match.group("stages")),
        "split_k": int(match.group("split_k")),
        "m_waves": int(match.group("m_waves")),
        "n_waves": int(match.group("n_waves")),
        "k_waves": int(match.group("k_waves")),
        "group_m": int(match.group("group_m")),
        "use_half_tile_interleaved": match.group("policy") in ("ht", "hti"),
        "has_bias": match.group("has_bias") == "1",
        "dtype": dtype,
        "out_dtype": out_dtype,
        "target_gfx": match.group("target_gfx"),
    }


def flydsl_hgemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    stages: int = 2,
    split_k: int = 1,
    m_waves: int = 2,
    n_waves: int = 2,
    k_waves: int = 1,
    group_m: int = 0,
    policy: str = "ft",
    out_dtype: torch.dtype | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Run the A16W16 kernel for this arch with AITER's ``B[N, K]`` convention."""

    if policy not in ("ft", "ht", "hti"):
        raise ValueError(f"Unsupported FlyDSL HGEMM policy: {policy!r}")
    launch_stream = (
        torch.cuda.current_stream(device=a.device) if stream is None else stream
    )
    if launch_stream.device != a.device:
        raise ValueError(f"`stream` must be on {a.device}, got {launch_stream.device}")

    gfx = get_gfx()
    if gfx in ("gfx1250", "gfx1250-strict"):
        if k_waves != 1:
            raise ValueError("The gfx1250 FlyDSL A16W16 kernel supports k_waves=1 only")
        with torch.cuda.stream(launch_stream):
            return gemm_a16w16_gfx1250(
                a,
                b,
                bias=bias,
                dtype=out_dtype or a.dtype,
                y=out,
                tile_m=block_m,
                tile_n=block_n,
                tile_k=block_k,
                m_warp=m_waves,
                n_warp=n_waves,
                num_buffers=stages,
                split_k=split_k,
                main_loop_unroll=policy in ("ht", "hti"),
            )
    if gfx != "gfx950":
        raise RuntimeError(
            "The FlyDSL A16W16 kernel currently supports gfx950 and gfx1250 only"
        )

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    user_kwargs = {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "stages": stages,
        "split_k": split_k,
        "m_waves": m_waves,
        "n_waves": n_waves,
        "k_waves": k_waves,
        "group_m": group_m,
        "use_half_tile_interleaved": policy in ("ht", "hti"),
    }
    return gemm_a16w16(
        a,
        b.t(),
        out=out,
        bias=bias,
        user_kwargs=user_kwargs,
        stream=launch_stream,
        layout="nt",
        out_dtype=out_dtype,
    )


# ---------------------------------------------------------------------------
# FlyDSL preshuffle GEMM kernel management
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _get_compile_fn():
    """Import the preshuffle compiler on first use."""
    from .kernels.preshuffle_gemm import compile_preshuffle_gemm

    logger.info("[FlyDSL] loaded preshuffle GEMM compiler")
    return compile_preshuffle_gemm


# Fixed size rather than one buffer per shape: a shape-keyed cache grows without
# limit and can evict a buffer a captured CUDA graph still points at. The bounds
# come from k_split_candidates, which keeps tile_count under CU_NUM and
# k_split * tile_count at four per CU.
# Mirrors preshuffle_gemm.PRESHUFFLE_M_MAX; duplicated to avoid importing the
# compiler module before the preshuffle path is selected.
PRESHUFFLE_M_MAX = 65536
PRESHUFFLE_FLAT_BUFFER_LIMIT_BYTES = 1 << 32

PRESHUFFLE_SPLIT_K_MAX_TILES = 256
PRESHUFFLE_SPLIT_K_MAX_TILE_ELEMS = 32 * 128
PRESHUFFLE_SPLIT_K_WORKSPACE_ELEMS = (
    4 * PRESHUFFLE_SPLIT_K_MAX_TILES * PRESHUFFLE_SPLIT_K_MAX_TILE_ELEMS
)


@functools.lru_cache(maxsize=128)
def _get_preshuffle_split_buffers(
    device: torch.device,
    stream: torch.cuda.Stream,
) -> tuple[Tensor, Tensor]:
    # Safe to reuse: launches on a stream are ordered and the reduction hands
    # the semaphore back zeroed.
    with persistent_alloc(device):
        workspace = torch.empty(
            PRESHUFFLE_SPLIT_K_WORKSPACE_ELEMS, dtype=torch.float32, device=device
        )
        semaphore = torch.zeros(
            PRESHUFFLE_SPLIT_K_MAX_TILES, dtype=torch.int32, device=device
        )
    return workspace, semaphore


def _check_preshuffle_flat_buffer_capacity(
    m: int,
    n: int,
    k: int,
    a_elem_bytes: int,
    b_elem_bytes: int,
    out_elem_bytes: int,
) -> None:
    """Keep flat AMD buffer descriptors and their i32 offsets below 4 GiB."""
    buffer_bytes = {
        "A": m * k * a_elem_bytes,
        "B": n * k * b_elem_bytes,
        "output": m * n * out_elem_bytes,
    }
    for name, size in buffer_bytes.items():
        if size >= PRESHUFFLE_FLAT_BUFFER_LIMIT_BYTES:
            raise RuntimeError(
                f"[FlyDSL] preshuffle {name} buffer needs {size} bytes; "
                "flat buffer descriptors require fewer than 4 GiB"
            )


def _check_preshuffle_split_capacity(
    m: int, n: int, tile_m: int, tile_n: int, split_k: int
) -> None:
    tiles = ((m + tile_m - 1) // tile_m) * ((n + tile_n - 1) // tile_n)
    if tiles > PRESHUFFLE_SPLIT_K_MAX_TILES:
        raise RuntimeError(
            f"[FlyDSL] split_k needs {tiles} tile semaphores, "
            f"more than {PRESHUFFLE_SPLIT_K_MAX_TILES}"
        )
    elems = split_k * m * n
    if elems > PRESHUFFLE_SPLIT_K_WORKSPACE_ELEMS:
        raise RuntimeError(
            f"[FlyDSL] split_k needs a {elems}-element fp32 workspace, "
            f"more than {PRESHUFFLE_SPLIT_K_WORKSPACE_ELEMS}"
        )


def flydsl_preshuffle_gemm_a8(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    use_async_copy: int = 0,
    waves_per_eu: int = 0,
    xcd_swizzle: int = 0,
    lds_stage: int = 2,
    enable_scheduler: bool = True,
    split_k: int = 1,
) -> Tensor:
    """Compile and run FlyDSL preshuffle GEMM, optionally with fp32 split-K."""
    compile_fn = _get_compile_fn()
    from aiter.utility import dtypes

    m, k = XQ.shape[0], XQ.shape[-1]
    n = WQ.shape[0]

    if m > PRESHUFFLE_M_MAX:
        raise RuntimeError(
            f"[FlyDSL] M ({m}) exceeds {PRESHUFFLE_M_MAX}; the preshuffle kernel "
            f"views A and C through a layout bounded by that many rows."
        )
    if n % 16 != 0:
        raise RuntimeError(
            f"[FlyDSL] N ({n}) must be a multiple of 16 for preshuffled B."
        )
    if n % tile_n != 0 and split_k > 1:
        raise RuntimeError(
            f"[FlyDSL] ragged N ({n}) does not support split_k ({split_k})."
        )
    if split_k < 1 or k % split_k != 0:
        raise RuntimeError(
            f"[FlyDSL] K ({k}) must be divisible by split_k ({split_k})."
        )
    if (k // split_k) % tile_k != 0:
        raise RuntimeError(
            f"[FlyDSL] K/split_k ({k // split_k}) is not a multiple of "
            f"tile_k ({tile_k}). "
            f"Arguments not supported! Skipping gemm!"
        )

    if XQ.dtype == dtypes.fp8:
        in_dtype = "fp8"
    elif XQ.dtype == torch.int8:
        in_dtype = "int8"
    else:
        raise ValueError(f"[FlyDSL] unsupported input dtype {XQ.dtype}")

    wpe = None if waves_per_eu <= 0 else waves_per_eu

    if Out.dtype == torch.bfloat16:
        out_dtype = "bf16"
    elif Out.dtype == torch.float16:
        out_dtype = "fp16"
    else:
        raise ValueError(
            f"[FlyDSL] unsupported output dtype {Out.dtype}; "
            "expected torch.bfloat16 or torch.float16"
        )
    _check_preshuffle_flat_buffer_capacity(
        m,
        n,
        k,
        XQ.element_size(),
        WQ.element_size(),
        Out.element_size(),
    )

    exe = compile_fn(
        N=n,
        K=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        use_async_copy=bool(use_async_copy),
        waves_per_eu=wpe,
        enable_scheduler=bool(enable_scheduler),
        xcd_swizzle=int(xcd_swizzle),
        lds_stage=int(lds_stage),
        split_k=int(split_k),
    )

    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    out_contig = Out.contiguous()
    # FlyDSL's preshuffle kernel requires an arg_bias slot (used only when
    # epilogue != "none"). Pass an empty tensor as a placeholder for the
    # default epilogue="none" path.
    dummy_bias = torch.empty(0, dtype=Out.dtype, device=Out.device)
    if split_k > 1:
        _check_preshuffle_split_capacity(m, n, tile_m, tile_n, split_k)
        workspace, semaphore = _get_preshuffle_split_buffers(
            Out.device, torch.cuda.current_stream(device=Out.device)
        )
    else:
        workspace = out_contig
        # dtype is part of the executable's cache signature, so this must match
        # what the AOT pre-compile passes or every non-split-K kernel misses it.
        semaphore = torch.empty(0, dtype=torch.int32, device=Out.device)
    # The layout-API launcher takes fx.Tensor args, so pass flat tensors
    # directly rather than raw pointers.
    _run_compiled(
        exe,
        workspace.view(-1),
        out_contig.view(-1),
        semaphore,
        _as_i8(XQ.contiguous()).view(-1),
        _as_i8(WQ.contiguous()).view(-1),
        x_scale.contiguous().view(-1),
        w_scale.contiguous().view(-1),
        dummy_bias,
        m,
        n,
        fx.Stream(torch.cuda.current_stream()),
    )
    if out_contig is not Out:
        Out.copy_(out_contig)

    return Out


# FlyDSL blockscale bpreshuffle GEMM kernel management


@functools.lru_cache(maxsize=1)
def _get_blockscale_compile_fn():
    """Import the block-scale compiler on first use."""
    from .kernels.gemm_blockscale_preshuffle import compile_blockscale_preshuffle_gemm

    logger.info("[FlyDSL] loaded blockscale bpreshuffle GEMM compiler")
    return compile_blockscale_preshuffle_gemm


# tile_is_valid plans LDS and queries the device limit, so cache it.
_blockscale_tile_is_valid = functools.lru_cache(maxsize=1024)(bs_tune.tile_is_valid)


@functools.lru_cache(maxsize=1024)
def _compile_flydsl_blockscale(n: int, k: int, config, out_dtype: str):
    """Cached compile. M is not part of the key: the kernel takes it at runtime."""
    return _get_blockscale_compile_fn()(
        N=n,
        K=k,
        tile_m=config.tile_m,
        tile_n=config.tile_n,
        tile_k=config.tile_k,
        out_dtype=out_dtype,
        use_cshuffle_epilog=config.use_cshuffle_epilog,
        use_async_copy=config.use_async_copy,
        num_waves=config.num_waves,
        split_k=config.split_k,
    )


@functools.cache
def _get_blockscale_split_buffers(
    device: torch.device, stream: torch.cuda.Stream
) -> tuple[Tensor, Tensor]:
    """The split-K workspace and arrival counters for one (device, stream).

    Never evicted or regrown: a captured graph keeps these addresses, so the pair must
    outlive every graph that used it. The kernel leaves every counter at zero, so
    nothing resets them between launches or replays.
    """
    with persistent_alloc(device):
        workspace = torch.empty(
            bs_tune.SPLIT_K_WORKSPACE_ELEMS, dtype=torch.float32, device=device
        )
        semaphore = torch.zeros(
            bs_tune.SPLIT_K_SEMAPHORE_SLOTS, dtype=torch.int32, device=device
        )
    return workspace, semaphore


def flydsl_gemm_a8w8_blockscale_bpreshuffle(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    *,
    use_cshuffle_epilog: bool = False,
    use_async_copy: bool = False,
    num_waves: int = 4,
    split_k: int = 1,
) -> Tensor:
    """Compile (cached) and run the FlyDSL blockscale bpreshuffle GEMM.

    The caller supplies the layouts gemm_a8w8_blockscale_bpreshuffle requires: WQ
    preshuffled with shuffle_weight(w, layout=(16, 16)) and x_scale K-major.

    Matches gemm_a8w8_blockscale within fp8 block-scale tolerance, not bit for bit.
    With split_k > 1 the fp32 partials are summed in a fixed order, so results are
    deterministic but differ from the unsplit kernel in the last bits.

    A split kernel needs split_k * M * N fp32 workspace elements and one zeroed Int32
    counter per output tile, ceil(M / tile_m) * (N / tile_n). This wrapper owns one
    fixed pair per (device, current stream), SPLIT_K_WORKSPACE_ELEMS and
    SPLIT_K_SEMAPHORE_SLOTS large (16 MiB and 16 KiB), allocated on the first split
    call on that stream and kept for the life of the process. A shape that does not
    fit raises RuntimeError; the tuner never offers one.

    * Calls on one stream reuse the pair in launch order; the kernel returns every
      counter to zero before it exits.
    * Calls on two streams get two pairs and may run concurrently.
    * Under CUDA graph capture the capture stream's pair is recorded into the graph.
      Replaying a graph is a race if, at the same time, the capture stream or another
      replay of a graph captured on it runs a split kernel; replays in one stream order
      are safe. On torch older than 2.10 the first split call on the capture stream
      must precede the capture, as for every persistent_alloc buffer.
    * A launch aborted mid-kernel can leave counters non-zero, which breaks every later
      split call on that stream.
    """
    from aiter.utility import dtypes

    m, k = XQ.shape[0], XQ.shape[-1]
    n = WQ.shape[0]
    if not bs_tune.offsets_fit(m, n, k):
        raise RuntimeError(
            f"[FlyDSL] M={m}, N={n}, K={k} needs offsets past 2**31; the block-scale "
            f"kernel requires M * K, N * K and M * N below {bs_tune.OFFSET_LIMIT}."
        )

    # uint8-typed fp8 bytes are accepted by the CK and triton paths, so accept them
    # here too.
    if XQ.dtype == torch.uint8:
        XQ = XQ.view(dtypes.fp8)
    if WQ.dtype == torch.uint8:
        WQ = WQ.view(dtypes.fp8)
    if XQ.dtype != dtypes.fp8 or WQ.dtype != dtypes.fp8:
        raise RuntimeError(
            f"[FlyDSL] blockscale GEMM needs fp8 inputs, got {XQ.dtype} and {WQ.dtype}"
        )
    if tuple(WQ.shape) != (n, k):
        raise RuntimeError(
            f"[FlyDSL] WQ must be (N, K) = ({n}, {k}), got {tuple(WQ.shape)}"
        )
    if x_scale.dtype != torch.float32 or w_scale.dtype != torch.float32:
        raise RuntimeError(
            f"[FlyDSL] blockscale GEMM needs fp32 scales, got {x_scale.dtype} and "
            f"{w_scale.dtype}"
        )
    scale_k = k // bs_tune.SCALE_BLOCK
    scale_n = -(-n // bs_tune.SCALE_BLOCK)
    if x_scale.numel() != m * scale_k or w_scale.numel() != scale_n * scale_k:
        raise RuntimeError(
            f"[FlyDSL] scales must hold M * K/128 = {m * scale_k} and "
            f"ceil(N/128) * K/128 = {scale_n * scale_k} elements, got "
            f"{x_scale.numel()} and {w_scale.numel()}"
        )
    if out.dtype == torch.bfloat16:
        out_dtype = "bf16"
    elif out.dtype == torch.float16:
        out_dtype = "fp16"
    else:
        raise RuntimeError(
            f"[FlyDSL] unsupported output dtype {out.dtype}; "
            f"expected torch.bfloat16 or torch.float16"
        )

    config = bs_tune.kernelInstance(
        tile_m, tile_n, tile_k, num_waves, use_async_copy, use_cshuffle_epilog, split_k
    )
    if not _blockscale_tile_is_valid(config, n, k):
        raise RuntimeError(
            f"[FlyDSL] kernel {config.name} is invalid for N={n}, K={k}. "
            f"Arguments not supported! Skipping gemm!"
        )
    if not bs_tune.split_k_buffers_fit(m, n, tile_m, tile_n, split_k):
        raise RuntimeError(
            f"[FlyDSL] split_k={split_k} at M={m}, N={n} needs {split_k * m * n} fp32 "
            f"workspace elements and {-(-m // tile_m) * (n // tile_n)} tile counters; "
            f"the split-K buffers hold {bs_tune.SPLIT_K_WORKSPACE_ELEMS} and "
            f"{bs_tune.SPLIT_K_SEMAPHORE_SLOTS}."
        )

    exe = _compile_flydsl_blockscale(n, k, config, out_dtype)

    # The kernel indexes scale_a as kb * M + row, so x_scale must be K-major bytes:
    # contiguous, or an (M, K/128) view with stride (1, M), which transposes back
    # without a copy. .contiguous() on other strides would silently re-order it.
    if x_scale.dim() == 2 and x_scale.stride(0) == 1 and x_scale.size(1) > 1:
        x_scale = x_scale.t()
    if not x_scale.is_contiguous():
        raise RuntimeError(
            f"[FlyDSL] x_scale must be contiguous K-major bytes or an (M, K/128) view "
            f"with stride (1, M); got shape {tuple(x_scale.shape)}, "
            f"stride {x_scale.stride()}"
        )
    x_scale_flat = x_scale.view(-1)
    if m == 0:
        return out

    # Unlike flydsl_preshuffle_gemm_a8, the launcher takes fx.Tensor, not pointers.
    out_contig = out if out.is_contiguous() else out.contiguous()
    stream = torch.cuda.current_stream()
    split_args = (
        _get_blockscale_split_buffers(stream.device, stream) if split_k > 1 else ()
    )
    _run_compiled(
        exe,
        out_contig,
        XQ.contiguous(),
        WQ.contiguous(),
        x_scale_flat,
        w_scale.contiguous().view(-1),
        m,
        *split_args,
        fx.Stream(stream),
    )
    if out_contig is not out:
        out.copy_(out_contig)

    return out
