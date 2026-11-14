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
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.utility.graph_alloc import capture_scoped_cache, persistent_alloc

from .gemm_a16w16_band import band_stream_name_supported, gemm_a16w16_band
from .gemm_a16w16_gfx1250 import gemm_a16w16 as gemm_a16w16_gfx1250
from .gemm_a16w16_ring import gemm_a16w16_ring
from .kernels.gemm_a16w16_gfx950 import (
    SPLIT_K_SEMAPHORE_MAX_LEN,
    gemm_a16w16,
)
from .kernels.gemm_a16w16_ring_gfx950 import ring_tile_error
from .kernels.gemm_a16w16_skinny_gfx950 import gemm_a16w16_skinny
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "RING_POLICY",
    "SKINNY_POLICY",
    "SPLIT_K_SEMAPHORE_MAX_LEN",
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
    r"gm(?P<group_m>\d+)(?:_stg(?P<k_stagger>\d+))?(?P<b_non_temporal>_bnt)?_p(?P<policy>ft|ht|hti|sk|bs|rg|tdm|tdmi|tdma|tdmia)_"
    r"(?P<target_gfx>gfx[0-9a-z]+(?:-[0-9a-z]+)*)$"
)

# The skinny policy (kernels/gemm_a16w16_skinny_gfx950.py) reuses the HGEMM name
# fields: t<block_m>x<block_n>x<block_k>x<stages> is its tile and LDS ring depth,
# w1x1x<k_waves> the waves splitting K inside a workgroup. It has no split-K,
# bias or fp32 output, and runs on gfx950 only.
SKINNY_POLICY = "sk"
# The ring policy (kernels/gemm_a16w16_ring_gfx950.py): t<block_m>x<block_n>x<block_k>x<stages> is its
# tile and LDS ring depth, w<m>x<n>x1 its waves, gm its XCD tile grouping. Same-dtype output only, no
# split-K or bias, gfx950 only.
RING_POLICY = "rg"
TDM_POLICIES = {
    "tdm": (False, False),
    "tdmi": (True, False),
    "tdma": (False, True),
    "tdmia": (True, True),
}


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

    if config.get("policy") in (SKINNY_POLICY, RING_POLICY, "bs", *TDM_POLICIES):
        policy = config["policy"]
    else:
        policy = "ht" if config["use_half_tile_interleaved"] else "ft"
    k_stagger = config.get("k_stagger", 0)
    gfx = target_gfx or get_gfx_runtime()
    _validate_k_stagger(k_stagger, gfx, policy)
    b_non_temporal = config.get("b_non_temporal", False)
    _validate_b_non_temporal(b_non_temporal, gfx, policy)
    stagger_suffix = f"_stg{k_stagger}" if k_stagger else ""
    memory_suffix = "_bnt" if b_non_temporal else ""
    name = (
        f"flydsl_hgemm_a{dtype_name}_w{dtype_name}_{out_dtype_name}_"
        f"t{config['block_m']}x{config['block_n']}x{config['block_k']}"
        f"x{config['stages']}_ks{config['split_k']}_"
        f"w{config['m_waves']}x{config['n_waves']}x{config['k_waves']}_"
        f"bias{int(has_bias)}_ktail0_gm{config['group_m']}{stagger_suffix}{memory_suffix}_p{policy}_"
        f"{gfx}"
    )
    if (
        policy in ("bs", RING_POLICY, *TDM_POLICIES)
        and get_flydsl_hgemm_kernel_params(name) is None
    ):
        raise ValueError("unsupported FlyDSL kernel configuration")
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
    policy = match.group("policy")
    k_stagger = int(match.group("k_stagger") or 0)
    b_non_temporal = match.group("b_non_temporal") is not None
    if (k_stagger or b_non_temporal) and (
        match.group("target_gfx") != "gfx950" or policy in (SKINNY_POLICY, RING_POLICY)
    ):
        return None
    if policy == SKINNY_POLICY and (
        dtype != "bf16"
        or out_dtype != "bf16"
        or match.group("split_k") != "1"
        or match.group("m_waves") != "1"
        or match.group("n_waves") != "1"
        or match.group("has_bias") != "0"
        or match.group("group_m") != "0"
        or match.group("target_gfx") != "gfx950"
    ):
        return None

    params = {
        "block_m": int(match.group("block_m")),
        "block_n": int(match.group("block_n")),
        "block_k": int(match.group("block_k")),
        "stages": int(match.group("stages")),
        "split_k": int(match.group("split_k")),
        "m_waves": int(match.group("m_waves")),
        "n_waves": int(match.group("n_waves")),
        "k_waves": int(match.group("k_waves")),
        "group_m": int(match.group("group_m")),
        "k_stagger": k_stagger,
        "b_non_temporal": b_non_temporal,
        "use_half_tile_interleaved": policy in ("ht", "hti"),
        "policy": policy,
        "has_bias": match.group("has_bias") == "1",
        "dtype": dtype,
        "out_dtype": out_dtype,
        "target_gfx": match.group("target_gfx"),
    }

    if policy in TDM_POLICIES:
        interleave, atomic = TDM_POLICIES[policy]
        if (
            params["target_gfx"] != "gfx1250"
            or params["out_dtype"] != params["dtype"]
            or params["k_waves"] != 1
            or (atomic and (dtype != "bf16" or params["split_k"] == 1))
        ):
            return None
        from .kernels.gemm_a16w16_tdm_gfx1250 import (
            GEMM_DTYPE_BF16,
            GEMM_DTYPE_FP16,
            validate_gemm_gfx1250_config,
        )

        try:
            validate_gemm_gfx1250_config(
                params["block_m"],
                params["block_n"],
                params["block_k"] * params["split_k"] * max(1, params["stages"] - 1),
                params["block_m"],
                params["block_n"],
                params["block_k"],
                params["stages"],
                params["m_waves"],
                params["n_waves"],
                params["group_m"],
                GEMM_DTYPE_BF16 if dtype == "bf16" else GEMM_DTYPE_FP16,
                16,
                16,
                32,
                True,
                params["split_k"],
                interleave,
                atomic,
            )
        except ValueError:
            return None
        params.update(interleave=interleave, split_k_atomic=atomic)
    if policy == "bs" and not band_stream_name_supported(params):
        return None
    if policy == RING_POLICY and (
        out_dtype != dtype
        or params["split_k"] != 1
        or params["k_waves"] != 1
        or params["has_bias"]
        or params["target_gfx"] != "gfx950"
        or ring_tile_error(
            params["block_m"],
            params["block_n"],
            params["block_k"],
            params["stages"],
            params["m_waves"],
            params["n_waves"],
        )
    ):
        return None
    return params


def _validate_k_stagger(k_stagger, gfx, policy):
    if not isinstance(k_stagger, int) or isinstance(k_stagger, bool) or k_stagger < 0:
        raise ValueError("k_stagger must be a non-negative integer")
    if k_stagger and (gfx != "gfx950" or policy in (SKINNY_POLICY, RING_POLICY)):
        raise ValueError("k_stagger supports gfx950 ft/ht/hti/bs policies only")


def _validate_b_non_temporal(value, gfx, policy):
    if not isinstance(value, bool):
        raise TypeError("b_non_temporal must be a bool")
    if value and (gfx != "gfx950" or policy in (SKINNY_POLICY, RING_POLICY)):
        raise ValueError("b_non_temporal supports gfx950 ft/ht/hti/bs policies only")


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
    k_stagger: int = 0,
    b_non_temporal: bool = False,
) -> torch.Tensor:
    """Run the A16W16 kernel with AITER's ``B[N, K]`` convention.

    On gfx950 ft/ht/hti, ``k_stagger`` rotates each workgroup's K traversal
    by ``(block_idx.x * k_stagger) % partition_k_tiles`` tiles, within its
    split-K partition. Zero preserves the original traversal. Tune this
    per shape: spreading concurrent reads can help streaming GEMMs but can
    hurt cache reuse. Rotation changes floating-point summation order, not
    accumulation precision or the split-K reduction protocol.
    ``b_non_temporal`` streams B's async loads with the non-temporal cache
    hint. It is independent of k_stagger; tune both, since shared B can
    benefit from caching. Both options default off.
    ``policy="tdm"/"tdmi"`` selects workspace TDM, plain/interleaved;
    ``"tdma"/"tdmia"`` selects BF16 packed-atomic split-K on gfx1250.
    All four use the same-dtype output and the existing weight[N,K] convention.
    ``policy="bs"`` selects the BF16 band-stream decode kernel for the
    validated 256-CU gfx950 M/N/K and configuration catalog. It requires
    no bias, split_k=1 and BF16 output. As on the legacy FT path, strided
    A/B are made contiguous before launch. Other unsupported inputs raise
    rather than silently selecting a different kernel.
    """

    if policy not in (
        "ft",
        "ht",
        "hti",
        SKINNY_POLICY,
        RING_POLICY,
        "bs",
        *TDM_POLICIES,
    ):
        raise ValueError(f"Unsupported FlyDSL HGEMM policy: {policy!r}")
    launch_stream = (
        torch.cuda.current_stream(device=a.device) if stream is None else stream
    )
    if launch_stream.device != a.device:
        raise ValueError(f"`stream` must be on {a.device}, got {launch_stream.device}")

    gfx = get_gfx_runtime()
    _validate_k_stagger(k_stagger, gfx, policy)
    _validate_b_non_temporal(b_non_temporal, gfx, policy)
    if policy in TDM_POLICIES:
        if gfx != "gfx1250":
            raise RuntimeError("TDM HGEMM policies require gfx1250")
        if k_waves != 1 or out_dtype not in (None, a.dtype):
            raise ValueError(
                "TDM HGEMM requires k_waves=1 and matching input/output dtype"
            )
        from .gemm_a16w16_tdm import gemm_a16w16_tdm

        interleave, atomic = TDM_POLICIES[policy]
        with torch.cuda.stream(launch_stream):
            # The tuned op accepts strided A/B. Normalize only layouts the
            # NT TDM contract cannot represent, before launching on this stream.
            if a.stride(1) != 1 or a.stride(0) * a.element_size() % 16:
                a = a.contiguous()
            if b.stride(1) != 1 or b.stride(0) * b.element_size() % 16:
                b = b.contiguous()
            if bias is not None and not bias.is_contiguous():
                bias = bias.contiguous()
            return gemm_a16w16_tdm(
                a,
                b.t(),
                out,
                bias,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                stages=stages,
                split_k=split_k,
                m_waves=m_waves,
                n_waves=n_waves,
                group_m=group_m,
                layout="nt",
                interleave=interleave,
                split_k_atomic=atomic,
                stream=launch_stream,
            )
    if policy == "bs":
        # A tuned row can serve strided inputs too. Preserve the legacy
        # public HGEMM normalization; only the device kernel requires
        # contiguous A/B. Copies must precede its launch on the same stream.
        with torch.cuda.stream(launch_stream):
            if not a.is_contiguous():
                a = a.contiguous()
            if not b.is_contiguous():
                b = b.contiguous()
        return gemm_a16w16_band(
            a,
            b,
            out,
            bias,
            out_dtype,
            launch_stream,
            {
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "stages": stages,
                "split_k": split_k,
                "m_waves": m_waves,
                "n_waves": n_waves,
                "k_waves": k_waves,
                "group_m": group_m,
                "k_stagger": k_stagger,
                "b_non_temporal": b_non_temporal,
            },
        )
    if policy == RING_POLICY:
        if gfx != "gfx950":
            raise RuntimeError("The FlyDSL ring HGEMM policy supports gfx950 only")
        if bias is not None or split_k != 1 or k_waves != 1:
            raise ValueError(
                "The FlyDSL ring HGEMM policy requires split_k=1, k_waves=1 and no bias"
            )
        if out_dtype not in (None, a.dtype):
            raise ValueError("The FlyDSL ring HGEMM policy writes the input dtype")
        with torch.cuda.stream(launch_stream):
            if not a.is_contiguous():
                a = a.contiguous()
            if not b.is_contiguous():
                b = b.contiguous()
        return gemm_a16w16_ring(
            a,
            b,
            out,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            stages=stages,
            m_waves=m_waves,
            n_waves=n_waves,
            group_m=group_m,
            stream=launch_stream,
        )
    if policy == SKINNY_POLICY:
        if gfx != "gfx950":
            raise RuntimeError("The FlyDSL skinny HGEMM policy supports gfx950 only")
        if bias is not None or split_k != 1 or m_waves != 1 or n_waves != 1 or group_m:
            raise ValueError(
                "The FlyDSL skinny HGEMM policy requires split_k=1, m_waves=1, "
                "n_waves=1, group_m=0 and no bias"
            )
        if out_dtype not in (None, a.dtype):
            raise ValueError("The FlyDSL skinny HGEMM policy writes the input dtype")
        return gemm_a16w16_skinny(
            a,
            b,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            stages=stages,
            k_waves=k_waves,
            out=out,
            stream=launch_stream,
        )
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
        "k_stagger": k_stagger,
        "b_non_temporal": b_non_temporal,
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


@functools.cache
def _get_preshuffle_split_workspace(
    device: torch.device,
    stream: torch.cuda.Stream,
) -> Tensor:
    # Fully overwritten by every launch, so it needs no initialization and is
    # shared with captures. Never evicted: a captured graph may point at it.
    with persistent_alloc(device):
        return torch.empty(
            PRESHUFFLE_SPLIT_K_WORKSPACE_ELEMS, dtype=torch.float32, device=device
        )


@capture_scoped_cache
def _get_preshuffle_split_semaphore(
    device: torch.device,
    stream: torch.cuda.Stream,
) -> Tensor:
    # Safe to reuse: launches on a stream are ordered and the reduction hands
    # the semaphore back zeroed.
    with persistent_alloc(device):
        return torch.zeros(
            PRESHUFFLE_SPLIT_K_MAX_TILES, dtype=torch.int32, device=device
        )


def _get_preshuffle_split_buffers(
    device: torch.device,
    stream: torch.cuda.Stream,
) -> tuple[Tensor, Tensor]:
    return (
        _get_preshuffle_split_workspace(device, stream),
        _get_preshuffle_split_semaphore(device, stream),
    )


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
