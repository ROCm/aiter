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
from aiter.jit.utils.chip_info import (
    get_cu_num,
    get_gfx,
    get_gfx_runtime,
)
from aiter.utility.graph_alloc import capture_scoped_cache, persistent_alloc

from .gemm_a16w16_band import band_stream_name_supported, gemm_a16w16_band
from .gemm_a16w16_gfx1250 import gemm_a16w16 as gemm_a16w16_gfx1250
from .kernels.gemm_a16w16_gfx950 import (
    SPLIT_K_SEMAPHORE_MAX_LEN,
    gemm_a16w16,
)
from .kernels.gemm_a16w16_skinny_gfx950 import gemm_a16w16_skinny
from .kernels.gemm_decode_block_mfma import compile_gemm_decode_block_mfma_bf16
from .kernels.gemm_decode_common import (
    DECODE_MAX_M,
    ActivationSource,
    BlockMfmaDecodeConfig,
    ContractionMode,
    DecodeConfig,
    OutputRounding,
    ReductionMode,
    WaveDecodeConfig,
    gemm_decode_kernel_name,
    get_decode_arch_traits,
    iter_gemm_decode_configs,
    parse_gemm_decode_kernel_name,
)
from .kernels.gemm_decode_wave import compile_gemm_decode_wave_bf16
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "SKINNY_POLICY",
    "SPLIT_K_SEMAPHORE_MAX_LEN",
    "ActivationSource",
    "BlockMfmaDecodeConfig",
    "ContractionMode",
    "DecodeConfig",
    "OutputRounding",
    "ReductionMode",
    "WaveDecodeConfig",
    "check_flydsl_decode_row",
    "compile_gemm_decode_bf16",
    "flydsl_decode_gemm",
    "flydsl_hgemm",
    "flydsl_hgemm_kernel_name",
    "flydsl_preshuffle_gemm_a8",
    "gemm_decode_bf16",
    "gemm_decode_kernel_name",
    "get_decode_arch_traits",
    "get_flydsl_decode_kernel_params",
    "get_flydsl_hgemm_kernel_params",
    "iter_gemm_decode_configs",
    "parse_gemm_decode_kernel_name",
]


_HGEMM_KERNEL_RE = re.compile(
    r"^flydsl_hgemm_"
    r"a(?P<dtype>bf16|fp16|f16)_w(?P=dtype)"
    r"_(?P<out_dtype>bf16|fp16|f16|fp32)_"
    r"t(?P<block_m>\d+)x(?P<block_n>\d+)x(?P<block_k>\d+)"
    r"x(?P<stages>\d+)_ks(?P<split_k>\d+)_"
    r"w(?P<m_waves>\d+)x(?P<n_waves>\d+)x(?P<k_waves>\d+)_"
    r"bias(?P<has_bias>[01])_ktail(?P<has_k_tail>[01])_"
    r"gm(?P<group_m>\d+)(?:_stg(?P<k_stagger>\d+))?(?P<b_non_temporal>_bnt)?_p(?P<policy>ft|ht|hti|sk|bs)_"
    r"(?P<target_gfx>gfx[0-9a-z]+(?:-[0-9a-z]+)*)$"
)

# The skinny policy (kernels/gemm_a16w16_skinny_gfx950.py) reuses the HGEMM name
# fields: t<block_m>x<block_n>x<block_k>x<stages> is its tile and LDS ring depth,
# w1x1x<k_waves> the waves splitting K inside a workgroup. It has no split-K,
# bias or fp32 output, and runs on gfx950 only.
SKINNY_POLICY = "sk"


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

    if config.get("policy") in (SKINNY_POLICY, "bs"):
        policy = config["policy"]
    else:
        policy = "ht" if config["use_half_tile_interleaved"] else "ft"
    k_stagger = config.get("k_stagger", 0)
    gfx = target_gfx or get_gfx()
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
    if policy == "bs" and get_flydsl_hgemm_kernel_params(name) is None:
        raise ValueError("unsupported band-stream kernel configuration")
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
        match.group("target_gfx") != "gfx950" or policy == SKINNY_POLICY
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

    if policy == "bs" and not band_stream_name_supported(params):
        return None
    return params


def _validate_k_stagger(k_stagger, gfx, policy):
    if not isinstance(k_stagger, int) or isinstance(k_stagger, bool) or k_stagger < 0:
        raise ValueError("k_stagger must be a non-negative integer")
    if k_stagger and (gfx != "gfx950" or policy == SKINNY_POLICY):
        raise ValueError("k_stagger supports gfx950 ft/ht/hti/bs policies only")


def _validate_b_non_temporal(value, gfx, policy):
    if not isinstance(value, bool):
        raise TypeError("b_non_temporal must be a bool")
    if value and (gfx != "gfx950" or policy == SKINNY_POLICY):
        raise ValueError("b_non_temporal supports gfx950 ft/ht/hti/bs policies only")


def _normalize_launch_stream(
    device: torch.device,
    stream: torch.cuda.Stream | None,
) -> torch.cuda.Stream:
    launch_stream = (
        torch.cuda.current_stream(device=device) if stream is None else stream
    )
    if launch_stream.device != device:
        raise ValueError(f"`stream` must be on {device}, got {launch_stream.device}")
    return launch_stream


def _validate_decode_bias(
    a: torch.Tensor,
    bias: torch.Tensor | None,
    n: int,
) -> None:
    if bias is None:
        return
    if bias.dim() != 1:
        raise ValueError(f"`bias` must be 1D, got bias.dim={bias.dim()}")
    if bias.shape != (n,):
        raise ValueError(f"`bias` must have shape {(n,)}, got {tuple(bias.shape)}")
    if bias.dtype != a.dtype:
        raise ValueError(
            f"`bias` dtype must match input dtype, got {bias.dtype=} {a.dtype=}"
        )
    if bias.device != a.device:
        raise ValueError(f"`bias` must be on {a.device}, got {bias.device}")


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
    ``policy="bs"`` selects the BF16 band-stream decode kernel for the
    validated 256-CU gfx950 M/N/K and configuration catalog. It requires
    no bias, split_k=1 and BF16 output. As on the legacy FT path, strided
    A/B are made contiguous before launch. Other unsupported inputs raise
    rather than silently selecting a different kernel.
    """

    if policy not in ("ft", "ht", "hti", SKINNY_POLICY, "bs"):
        raise ValueError(f"Unsupported FlyDSL HGEMM policy: {policy!r}")
    launch_stream = _normalize_launch_stream(a.device, stream)

    gfx = get_gfx_runtime()
    _validate_k_stagger(k_stagger, gfx, policy)
    _validate_b_non_temporal(b_non_temporal, gfx, policy)
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


def _byte_ranges(tensor: torch.Tensor) -> list[tuple[int, int]]:
    """The byte ranges a decode operand occupies.

    A packed tensor is one range. A rank-2 tensor with a unit column stride
    and a row stride above its width (A, a column slice of a wider
    activation) is one range per row, so the gaps between rows stay free.
    """
    begin = tensor.data_ptr()
    size = tensor.element_size()
    if tensor.dim() == 2 and tensor.shape[0] > 1:
        rows, cols = tensor.shape
        row_stride = tensor.stride(0)
        if tensor.stride(1) == 1 and row_stride > cols:
            return [
                (
                    begin + row * row_stride * size,
                    begin + (row * row_stride + cols) * size,
                )
                for row in range(rows)
            ]
    return [(begin, begin + tensor.numel() * size)]


def _overlaps(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """Check if two tensors overlap in memory, row by row for a strided A."""
    return any(
        lhs_begin < rhs_end and rhs_begin < lhs_end
        for lhs_begin, lhs_end in _byte_ranges(lhs)
        for rhs_begin, rhs_end in _byte_ranges(rhs)
    )


def validate_gemm_decode_tensors(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    bias: torch.Tensor | None = None,
    arch: str | None = None,
) -> tuple[int, int, int]:
    """Validate the real-tensor ABI shared by both kernel families.

    B and C must be packed row-major. A may have any row stride ``>= K`` with a
    unit column stride, e.g. a column slice of a wider activation; the stride
    of a size-1 M dimension is ignored, as PyTorch does.

    This runs per launch on the decode path, so it is deliberately flat: no
    dicts, no tuple building.
    """
    for name, t in (("A", A), ("B", B), ("C", C)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if t.dim() != 2:
            raise ValueError(f"{name} must be rank 2, got rank {t.dim()}")
        if t.dtype != torch.bfloat16:
            raise ValueError(f"{name} must have dtype torch.bfloat16")
        if t.device.type != "cuda":
            raise ValueError(f"{name} must be on a CUDA/ROCm device")

    m, k = A.shape
    n, b_k = B.shape
    if not (1 <= m <= DECODE_MAX_M):
        raise ValueError(f"decode GEMM supports exact M in [1, {DECODE_MAX_M}]")
    if n <= 0 or k <= 0:
        raise ValueError("decode GEMM requires positive N and K")
    if b_k != k:
        raise ValueError(f"B must have shape ({n}, {k}), got {tuple(B.shape)}")
    if A.device != B.device or A.device != C.device:
        raise ValueError("A, B, and C must be on the same device")
    _validate_decode_bias(A, bias, n)

    for name, t, rows, cols in (("A", A, m, k), ("B", B, n, k), ("C", C, m, n)):
        shape = t.shape
        if shape[0] != rows or shape[1] != cols:
            raise ValueError(
                f"{name} must have shape {(rows, cols)}, got {tuple(shape)}"
            )
        stride = t.stride()
        if name == "A":
            if stride[1] != 1 or (rows > 1 and stride[0] < cols):
                raise ValueError(
                    "A must be row-major with a unit column stride and a row "
                    f"stride of at least K, got strides {tuple(stride)}"
                )
        elif stride[0] != cols or stride[1] != 1:
            raise ValueError(f"{name} must use packed row-major storage")

    if _overlaps(C, A) or _overlaps(C, B):
        raise ValueError("C must not overlap A or B")
    if bias is not None and _overlaps(C, bias):
        raise ValueError("C must not overlap bias")
    gfx = get_gfx_runtime() if arch is None else arch
    if gfx not in ("gfx942", "gfx950"):
        raise ValueError(f"decode GEMM requires gfx942 or gfx950, got {gfx}")
    return m, n, k


def compile_gemm_decode_bf16(
    m: int,
    n: int,
    k: int,
    config: DecodeConfig,
    *,
    arch: str,
    num_cus: int | None = None,
    has_bias: bool = False,
    strided_a: bool = False,
):
    """Compile one exact unified ``(arch, M, N, K, config)`` identity.

    ``strided_a`` selects the variant that accepts an A row stride other than K.
    """
    if isinstance(config, WaveDecodeConfig):
        return compile_gemm_decode_wave_bf16(
            m,
            n,
            k,
            config,
            arch,
            has_bias=has_bias,
            strided_a=strided_a,
        )
    if isinstance(config, BlockMfmaDecodeConfig):
        return compile_gemm_decode_block_mfma_bf16(
            m,
            n,
            k,
            config,
            arch,
            num_cus=num_cus,
            has_bias=has_bias,
            strided_a=strided_a,
        )
    raise TypeError(f"unsupported decode config type: {type(config).__name__}")


def gemm_decode_bf16(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    config: DecodeConfig,
    stream: torch.cuda.Stream | None = None,
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch one Wave or BlockMFMA configuration for the inferred exact shape."""
    runtime_arch = get_gfx_runtime()
    m, n, k = validate_gemm_decode_tensors(
        A,
        B,
        C,
        bias=bias,
        arch=runtime_arch,
    )
    if bias is not None and not bias.is_contiguous():
        raise ValueError("bias must be contiguous")
    launch_stream = _normalize_launch_stream(A.device, stream)
    if m == 1 and A.stride() != (k, 1):
        # A size-1 row has no meaningful row stride; give it the packed one so
        # the packed kernel serves it (zero copy).
        A = A.as_strided((1, k), (k, 1))
    strided_a = A.stride(0) != k
    launcher = compile_gemm_decode_bf16(
        m,
        n,
        k,
        config,
        arch=runtime_arch,
        num_cus=get_cu_num(),
        has_bias=bias is not None,
        strided_a=strided_a,
    )
    launcher(A, B, C, bias=bias, stream=fx.Stream(launch_stream))
    return C


def get_flydsl_decode_kernel_params(name: str) -> dict | None:
    """Parse a decode kernel name; None if it is not a valid decode name."""
    try:
        arch, m, n, k, config, has_bias = parse_gemm_decode_kernel_name(name)
    except ValueError:
        return None
    return {
        "arch": arch,
        "m": m,
        "n": n,
        "k": k,
        "config": config,
        "has_bias": has_bias,
    }


def check_flydsl_decode_row(
    params: dict,
    m: int,
    n: int,
    k: int,
    arch: str,
    has_bias: bool,
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
    scaled: bool,
    preshuffled: bool,
) -> str | None:
    """None if a decode row fits the call, otherwise the reason it does not."""
    if (in_dtype, out_dtype) != (torch.bfloat16, torch.bfloat16):
        return f"decode is BF16 only, call is {in_dtype}->{out_dtype}"
    if scaled or preshuffled:
        return "decode supports neither scaling nor preshuffled weights"
    got = (params["arch"], params["m"], params["n"], params["k"], params["has_bias"])
    want = (arch, m, n, k, has_bias)
    if got != want:
        return f"kernel identity {got} does not match the call {want}"
    return None


def flydsl_decode_gemm(
    inp: torch.Tensor,
    weights: torch.Tensor,
    kernel_name: str,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch the exact-M/N/K decode kernel that a tuned row names."""
    params = get_flydsl_decode_kernel_params(kernel_name)
    if params is None:
        raise ValueError(f"invalid FlyDSL decode kernel name: {kernel_name!r}")
    expected = (
        get_gfx_runtime(),
        int(inp.shape[0]),
        int(weights.shape[0]),
        int(inp.shape[1]),
    )
    kernel = (params["arch"], params["m"], params["n"], params["k"])
    if kernel != expected:
        raise ValueError(
            "FlyDSL decode tuned kernel does not match the runtime "
            f"exact identity: kernel={kernel}, runtime={expected}"
        )
    if params["has_bias"] != (bias is not None):
        raise ValueError("FlyDSL decode kernel bias identity does not match launch")
    if out is None:
        out = torch.empty(
            (params["m"], params["n"]), dtype=torch.bfloat16, device=inp.device
        )
    return gemm_decode_bf16(inp, weights, out, params["config"], bias=bias)


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
