"""Dedicated small-M bf16 HGEMM kernel path.

This module intentionally stays separate from `hgemm.py`. The generic HGEMM
kernel and this small-M path share the same split-K contract and both still
take `m` as a runtime value, but this path is no longer just a different
parameter point of one template:

- `TILE_M=16` and `BLOCK_M_WARPS=1` are hard-wired so the block spends its
  wave budget on N/K work instead of over-parallelizing the tiny M dimension.
  Concretely, the block only covers one 16-row M tile and avoids launching
  extra M-side warps whose useful work would quickly disappear once `m` is
  much smaller than a generic HGEMM tile.
- Warp mapping is specialized for tiny-M shapes: warps do not spread across
  the M dimension like the generic kernel, and more of the wave budget is used
  to cover N-side work. In the hot path this shows up as `warp_m_idx = 0` and
  `warp_n_idx = wid * WARP_N`, so the whole block behaves like "one small M
  slice, many N workers" instead of a more balanced 2D warp decomposition.
- The kernel adds small-M-specific wide-N mechanisms:
  `N_TILE_REPEAT` for non-`B_TO_LDS` multi-tile accumulation and
  `PERSISTENT_N_TILES` for the `B_TO_LDS` persistent-N path. The first lets one
  block reuse the same loaded A fragments while accumulating several N tiles in
  registers; the second lets a `B_TO_LDS` block stay on a small group of N
  tiles longer so the cost of setting up the tiny-M tile is amortized over more
  useful N-side work.
- The `B_TO_LDS` hot loop is tuned separately with an explicit unroll knob and
  a dedicated wide-N scheduler, rather than reusing the generic `hgemm.py`
  scheduling structure. `B_TO_LDS_UNROLL` controls how many K iterations are
  pipelined per outer step, and the wide-N scheduler adjusts the DS/VMEM/MFMA
  issue pattern so LDS reads, async B loads, and matrix instructions stay
  better balanced for these skinny-M / wide-N shapes.

In practice, the main optimization goal here is to improve decode-like GEMMs
where M is tiny while N/K stay large: reduce wasted M-side parallelism, reuse
the loaded A tile across more N work, and give wide-N shapes a more specialized
schedule than the generic HGEMM kernel.
"""

import functools
import re
from itertools import product

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP

from aiter.ops.flydsl.kernels import vector

from .splitk_hgemm import (
    OnlineScheduler,
    SPLIT_K_SEMAPHORE_MAX_LEN,
    WmmaHalf_m16n16k16,
    WmmaHalf_m16n16k32,
    swizzle_xor16,
)
from .tensor_shim import GTensor, _to_raw, get_dtype_in_kernel

__all__ = [
    "LDS_STAGING_DIRECT",
    "LDS_STAGING_OPTIONS",
    "LDS_STAGING_VGPR",
    "SMALL_M_KERNEL_MAX",
    "SMALL_M_SUPPORTED_ARCHS",
    "compile_small_m_hgemm_kernel",
    "iter_small_m_registry_configs",
    "parse_small_m_kernel_name",
    "small_m_arch_params",
    "small_m_kernel_name",
    "small_m_lds_bytes",
    "small_m_max_lds_bytes",
    "small_m_tile_k_is_swizzle_safe",
    "validate_small_m_kernel_config",
]

SMALL_M_KERNEL_MAX = 17
SMALL_M_SUPPORTED_ARCHS = frozenset({"gfx942", "gfx950"})
TILE_M = 16
BLOCK_M_WARPS = 1
STAGES = 2
WARP_SIZE = 64
DTYPE_BYTES = 2
LDG_VEC_SIZE = 8
LDS_STAGING_DIRECT = "direct"
LDS_STAGING_VGPR = "vgpr"
LDS_STAGING_OPTIONS = (LDS_STAGING_DIRECT, LDS_STAGING_VGPR)

# Expand the original small-M catalog with the additional cases that proved
# useful during the deeper exhaustive search, instead of maintaining separate
# compact/exhaustive modes.
SMALL_M_TILE_K_OPTIONS = (32, 64, 128, 256)
SMALL_M_MAX_SPLIT_K = 32
SMALL_M_TILE_N_OPTIONS = (
    32,
    64,
    96,
    128,
    160,
    192,
    224,
    256,
    384,
    512,
    768,
    1024,
)
SMALL_M_GFX942_EXTRA_TILE_N_OPTIONS = (16,)
SMALL_M_NON_B_TO_LDS_WAVES_PER_EU_OPTIONS = (0, 2, 4)
# Keep 0 for narrow B_TO_LDS shapes where it remains a real candidate, and
# canonicalize only the wide-N B_TO_LDS duplicates at registry emission time.
SMALL_M_B_TO_LDS_WAVES_PER_EU_OPTIONS = (0, 2, 4)
SMALL_M_B_TO_LDS_UNROLL_OPTIONS = (8, 16)
SMALL_M_N_TILE_REPEAT_OPTIONS = (1, 2, 4)
SMALL_M_PERSISTENT_N_TILE_OPTIONS = (2, 4, 8)
SMALL_M_BASE_BLOCK_N_WARPS = (1, 2, 3, 4)
SMALL_M_REPEAT_BLOCK_N_WARPS = (1, 2)
SMALL_M_B_TO_LDS_BLOCK_N_WARPS = (1, 2, 3, 4)
SMALL_M_PERSISTENT_BLOCK_N_WARPS = (2, 3, 4)


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _align_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def _require_small_m_arch(arch: str) -> None:
    if arch not in SMALL_M_SUPPORTED_ARCHS:
        supported = ", ".join(sorted(SMALL_M_SUPPORTED_ARCHS))
        raise ValueError(
            f"unsupported small-M architecture {arch!r}; "
            f"supported architectures: {supported}"
        )


def small_m_arch_params(arch: str) -> dict:
    """Return the native MFMA and direct-to-LDS parameters for ``arch``."""
    _require_small_m_arch(arch)
    if arch == "gfx942":
        return {
            "wmma_cls": WmmaHalf_m16n16k16,
            "mfma_per_warp_k": 2,
            "direct_dma_bytes": 4,
        }
    return {
        "wmma_cls": WmmaHalf_m16n16k32,
        "mfma_per_warp_k": 1,
        "direct_dma_bytes": 16,
    }


def small_m_max_lds_bytes(arch: str) -> int:
    """Return the addressable per-workgroup LDS budget for ``arch``."""
    _require_small_m_arch(arch)
    return SMEM_CAPACITY_MAP[arch]


def small_m_tile_k_is_swizzle_safe(tile_k: int) -> bool:
    """Whether XOR-16 remains a permutation within each staged K row."""
    if tile_k < 32 or tile_k % 32 != 0:
        return False
    k_blocks16 = tile_k * DTYPE_BYTES // 16
    return k_blocks16 > 0 and (k_blocks16 & (k_blocks16 - 1)) == 0


def small_m_lds_bytes(*, tile_n: int, tile_k: int, b_to_lds: bool) -> int:
    """Mirror the SharedStorage A/C alias and optional B field footprint."""
    a_lds_bytes = max(
        STAGES * TILE_M * tile_k * DTYPE_BYTES,
        TILE_M * tile_n * DTYPE_BYTES,
    )
    if not b_to_lds:
        return a_lds_bytes
    return _align_up(a_lds_bytes, 16) + STAGES * tile_n * tile_k * DTYPE_BYTES


def _small_m_tile_n_options(arch: str) -> tuple[int, ...]:
    _require_small_m_arch(arch)
    if arch == "gfx942":
        return tuple(
            sorted(SMALL_M_GFX942_EXTRA_TILE_N_OPTIONS + SMALL_M_TILE_N_OPTIONS)
        )
    return SMALL_M_TILE_N_OPTIONS


def _small_m_tile_k_options(k: int) -> tuple[int, ...]:
    return tuple(
        tile_k
        for tile_k in SMALL_M_TILE_K_OPTIONS
        if any(
            k % split_k == 0 and (k // split_k) % tile_k == 0
            for split_k in range(1, SMALL_M_MAX_SPLIT_K + 1)
        )
    )


def _small_m_split_k_options(k: int, tile_k: int) -> tuple[int, ...]:
    return tuple(
        split_k
        for split_k in range(1, SMALL_M_MAX_SPLIT_K + 1)
        if k % split_k == 0 and (k // split_k) % tile_k == 0
    )


def small_m_kernel_name(
    dtype: str,
    *,
    target_gfx: str,
    tile_n: int,
    tile_k: int,
    split_k: int,
    block_n_warps: int,
    n_tile_repeat: int,
    persistent_n_tiles: int,
    waves_per_eu: int,
    b_to_lds_unroll: int,
    b_to_lds: bool,
    has_bias: bool,
    lds_staging: str = LDS_STAGING_DIRECT,
) -> str:
    _require_small_m_arch(target_gfx)
    if lds_staging not in LDS_STAGING_OPTIONS:
        raise ValueError(f"unrecognized LDS staging policy: {lds_staging!r}")
    if target_gfx != "gfx942" and lds_staging != LDS_STAGING_DIRECT:
        raise ValueError(f"LDS staging {lds_staging!r} is not supported for {target_gfx}")
    name = (
        f"smallm_hgemm_{dtype}_{target_gfx}_{TILE_M}x{tile_n}x{tile_k}_S{STAGES}TN_AS"
        f"_BNW{block_n_warps}"
    )
    if n_tile_repeat > 1:
        name += f"_NR{n_tile_repeat}"
    if persistent_n_tiles > 1:
        name += f"_PN{persistent_n_tiles}"
    if split_k > 1:
        name += f"_SPK{split_k}"
    if b_to_lds:
        name += "_BS"
    if waves_per_eu > 0:
        name += f"_WPE{waves_per_eu}"
    if b_to_lds:
        if b_to_lds_unroll > 0:
            name += f"_UR{b_to_lds_unroll}"
    if has_bias:
        name += "_BIAS"
    name += f"_LDS{lds_staging.upper()}"
    return name


_SMALL_M_KERNEL_NAME_RE = re.compile(
    r"^smallm_hgemm_(?P<dtype>bf16)_(?P<target_gfx>gfx\d+)_"
    r"16x(?P<tile_n>[1-9]\d*)x"
    r"(?P<tile_k>[1-9]\d*)_S2TN_AS_BNW(?P<block_n_warps>[1-9]\d*)"
    r"(?:_NR(?P<n_tile_repeat>(?:[2-9]|[1-9]\d+)))?"
    r"(?:_PN(?P<persistent_n_tiles>(?:[2-9]|[1-9]\d+)))?"
    r"(?:_SPK(?P<split_k>(?:[2-9]|[1-9]\d+)))?"
    r"(?P<b_to_lds>_BS)?"
    r"(?:_WPE(?P<waves_per_eu>[1-9]\d*))?"
    r"(?:_UR(?P<b_to_lds_unroll>[1-9]\d*))?"
    r"(?P<has_bias>_BIAS)?"
    r"_LDS(?P<lds_staging>DIRECT|VGPR)$"
)


def parse_small_m_kernel_name(kernel_name: str) -> dict:
    """Parse a canonical small-M kernel name into its compile-time config."""
    match = _SMALL_M_KERNEL_NAME_RE.fullmatch(kernel_name)
    if match is None:
        raise ValueError(f"invalid FlyDSL small-M kernel name: {kernel_name!r}")

    groups = match.groupdict()
    config = {
        "kernel_family": "small_m",
        "target_gfx": groups["target_gfx"],
        "stage": STAGES,
        "tile_m": TILE_M,
        "tile_n": int(groups["tile_n"]),
        "tile_k": int(groups["tile_k"]),
        "split_k": int(groups["split_k"] or 1),
        "block_m_warps": BLOCK_M_WARPS,
        "block_n_warps": int(groups["block_n_warps"]),
        "block_k_warps": 1,
        "n_tile_repeat": int(groups["n_tile_repeat"] or 1),
        "persistent_n_tiles": int(groups["persistent_n_tiles"] or 1),
        "waves_per_eu": int(groups["waves_per_eu"] or 0),
        "b_to_lds_unroll": int(groups["b_to_lds_unroll"] or 0),
        "async_copy": True,
        "b_to_lds": groups["b_to_lds"] is not None,
        "c_to_lds": False,
        "dtype": groups["dtype"],
        "out_dtype": groups["dtype"],
        "has_bias": groups["has_bias"] is not None,
        "lds_staging": groups["lds_staging"].lower(),
    }
    if config["b_to_lds_unroll"] and not config["b_to_lds"]:
        raise ValueError("small-M kernel name encodes unroll without B-to-LDS")
    if config["b_to_lds"] and config["b_to_lds_unroll"] == 0:
        raise ValueError("small-M B-to-LDS kernel name must encode its effective unroll")
    wide_n_b_to_lds = (
        config["b_to_lds"]
        and config["n_tile_repeat"] == 1
        and config["tile_n"] >= 128
        and config["block_n_warps"] >= 2
    )
    if wide_n_b_to_lds and config["waves_per_eu"] == 0:
        raise ValueError(
            "wide-N small-M B-to-LDS kernel name must encode effective waves-per-EU"
        )

    validation_n = config["tile_n"] * max(
        config["n_tile_repeat"], config["persistent_n_tiles"]
    )
    validation_k = config["tile_k"] * config["split_k"]
    try:
        _validate_small_m_registry_config(
            1,
            validation_n,
            validation_k,
            tile_n=config["tile_n"],
            tile_k=config["tile_k"],
            split_k=config["split_k"],
            block_n_warps=config["block_n_warps"],
            n_tile_repeat=config["n_tile_repeat"],
            persistent_n_tiles=config["persistent_n_tiles"],
            waves_per_eu=config["waves_per_eu"],
            b_to_lds_unroll=config["b_to_lds_unroll"],
            b_to_lds=config["b_to_lds"],
            lds_staging=config["lds_staging"],
            arch=config["target_gfx"],
        )
    except ValueError as error:
        raise ValueError(
            f"illegal FlyDSL small-M kernel name: {kernel_name!r}"
        ) from error

    canonical_name = small_m_kernel_name(
        config["dtype"],
        target_gfx=config["target_gfx"],
        tile_n=config["tile_n"],
        tile_k=config["tile_k"],
        split_k=config["split_k"],
        block_n_warps=config["block_n_warps"],
        n_tile_repeat=config["n_tile_repeat"],
        persistent_n_tiles=config["persistent_n_tiles"],
        waves_per_eu=config["waves_per_eu"],
        b_to_lds_unroll=config["b_to_lds_unroll"],
        b_to_lds=config["b_to_lds"],
        has_bias=config["has_bias"],
        lds_staging=config["lds_staging"],
    )
    if canonical_name != kernel_name:
        raise ValueError(
            f"non-canonical FlyDSL small-M kernel name: {kernel_name!r}; "
            f"expected {canonical_name!r}"
        )
    return config


def _validate_small_m_registry_config(
    m: int,
    n: int,
    k: int,
    *,
    tile_n: int,
    tile_k: int,
    split_k: int,
    block_n_warps: int,
    n_tile_repeat: int,
    persistent_n_tiles: int,
    waves_per_eu: int,
    b_to_lds_unroll: int,
    b_to_lds: bool,
    lds_staging: str,
    arch: str,
) -> None:
    del waves_per_eu

    _require_small_m_arch(arch)
    if lds_staging not in LDS_STAGING_OPTIONS:
        raise ValueError
    if arch != "gfx942" and lds_staging != LDS_STAGING_DIRECT:
        raise ValueError
    if not (1 <= m < SMALL_M_KERNEL_MAX):
        raise ValueError
    if tile_n < 1 or not small_m_tile_k_is_swizzle_safe(tile_k):
        raise ValueError
    if block_n_warps < 1 or split_k < 1:
        raise ValueError
    if n_tile_repeat < 1 or persistent_n_tiles < 1:
        raise ValueError
    if b_to_lds_unroll < 0:
        raise ValueError
    if tile_n % (block_n_warps * 16) != 0:
        raise ValueError
    if n_tile_repeat > 1:
        if b_to_lds:
            raise ValueError
        classic_repeat = block_n_warps == 1 and tile_n == 64
        wave_repeat = n_tile_repeat == 2 and block_n_warps == 2 and tile_n == 192
        if not (classic_repeat or wave_repeat):
            raise ValueError
    if persistent_n_tiles > 1 and (
        not b_to_lds or n_tile_repeat != 1 or tile_n < 128 or block_n_warps < 2
    ):
        raise ValueError
    if n < tile_n or n % tile_n != 0:
        raise ValueError
    if persistent_n_tiles > n // tile_n:
        raise ValueError
    if k % split_k != 0:
        raise ValueError
    ks = k // split_k
    if ks < tile_k or ks % tile_k != 0:
        raise ValueError

    if split_k > 1:
        counters = _ceil_div(m, TILE_M) * (n // tile_n)
        if counters > SPLIT_K_SEMAPHORE_MAX_LEN:
            raise ValueError

    if (
        small_m_lds_bytes(tile_n=tile_n, tile_k=tile_k, b_to_lds=b_to_lds)
        > small_m_max_lds_bytes(arch)
    ):
        raise ValueError


def validate_small_m_kernel_config(
    m: int,
    n: int,
    k: int,
    config: dict,
    *,
    arch: str,
) -> None:
    """Validate a parsed small-M config against its CSV/runtime shape."""
    if arch not in SMALL_M_SUPPORTED_ARCHS:
        raise ValueError(
            "FlyDSL small-M currently supports "
            f"{', '.join(sorted(SMALL_M_SUPPORTED_ARCHS))}, got {arch!r}"
        )
    if config.get("target_gfx") != arch:
        raise ValueError(
            "FlyDSL small-M config architecture mismatch: "
            f"name={config.get('target_gfx')!r}, runtime={arch!r}"
        )
    try:
        _validate_small_m_registry_config(
            m,
            n,
            k,
            tile_n=config["tile_n"],
            tile_k=config["tile_k"],
            split_k=config["split_k"],
            block_n_warps=config["block_n_warps"],
            n_tile_repeat=config["n_tile_repeat"],
            persistent_n_tiles=config["persistent_n_tiles"],
            waves_per_eu=config["waves_per_eu"],
            b_to_lds_unroll=config["b_to_lds_unroll"],
            b_to_lds=config["b_to_lds"],
            lds_staging=config["lds_staging"],
            arch=arch,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"illegal FlyDSL small-M config for M={m}, N={n}, K={k}"
        ) from error


def _small_m_registry_variants():
    variants = []
    seen_variants = set()

    def add_variant(
        *,
        block_n_warps: int,
        b_to_lds: bool,
        n_tile_repeat: int = 1,
        persistent_n_tiles: int = 1,
        waves_per_eu: int = 0,
        b_to_lds_unroll: int = 0,
    ) -> None:
        variant = {
            "block_m_warps": BLOCK_M_WARPS,
            "block_n_warps": block_n_warps,
            "b_to_lds": b_to_lds,
            "n_tile_repeat": n_tile_repeat,
            "persistent_n_tiles": persistent_n_tiles,
            "waves_per_eu": waves_per_eu,
            "b_to_lds_unroll": b_to_lds_unroll,
        }
        variant_key = tuple(sorted(variant.items()))
        if variant_key in seen_variants:
            return
        seen_variants.add(variant_key)
        variants.append(variant)

    for block_n_warps in SMALL_M_BASE_BLOCK_N_WARPS:
        for waves_per_eu in SMALL_M_NON_B_TO_LDS_WAVES_PER_EU_OPTIONS:
            add_variant(
                block_n_warps=block_n_warps,
                b_to_lds=False,
                waves_per_eu=waves_per_eu,
            )

    for n_tile_repeat in SMALL_M_N_TILE_REPEAT_OPTIONS[1:]:
        for block_n_warps in SMALL_M_REPEAT_BLOCK_N_WARPS:
            for waves_per_eu in SMALL_M_NON_B_TO_LDS_WAVES_PER_EU_OPTIONS:
                add_variant(
                    block_n_warps=block_n_warps,
                    b_to_lds=False,
                    n_tile_repeat=n_tile_repeat,
                    waves_per_eu=waves_per_eu,
                )

    for block_n_warps in SMALL_M_B_TO_LDS_BLOCK_N_WARPS:
        for waves_per_eu in SMALL_M_B_TO_LDS_WAVES_PER_EU_OPTIONS:
            for b_to_lds_unroll in SMALL_M_B_TO_LDS_UNROLL_OPTIONS:
                add_variant(
                    block_n_warps=block_n_warps,
                    b_to_lds=True,
                    waves_per_eu=waves_per_eu,
                    b_to_lds_unroll=b_to_lds_unroll,
                )

    for persistent_n_tiles in SMALL_M_PERSISTENT_N_TILE_OPTIONS:
        for block_n_warps in SMALL_M_PERSISTENT_BLOCK_N_WARPS:
            for waves_per_eu in SMALL_M_B_TO_LDS_WAVES_PER_EU_OPTIONS:
                for b_to_lds_unroll in SMALL_M_B_TO_LDS_UNROLL_OPTIONS:
                    add_variant(
                        block_n_warps=block_n_warps,
                        b_to_lds=True,
                        persistent_n_tiles=persistent_n_tiles,
                        waves_per_eu=waves_per_eu,
                        b_to_lds_unroll=b_to_lds_unroll,
                    )

    return tuple(variants)


def _canonicalize_small_m_registry_config(config: dict) -> dict:
    """Match registry metadata to the effective compile-time kernel settings."""
    canonical = dict(config)
    wide_n_b_to_lds = (
        canonical["b_to_lds"]
        and canonical["n_tile_repeat"] == 1
        and canonical["tile_n"] >= 128
        and canonical["block_n_warps"] >= 2
    )
    if canonical["b_to_lds"]:
        if canonical["b_to_lds_unroll"] <= 0:
            canonical["b_to_lds_unroll"] = 8
        if canonical["waves_per_eu"] <= 0 and wide_n_b_to_lds:
            canonical["waves_per_eu"] = 2
    return canonical


def iter_small_m_registry_configs(
    dtype: str,
    out_dtype: str,
    *,
    m: int,
    n: int,
    k: int,
):
    if dtype != "bf16" or out_dtype != "bf16":
        return

    gpu_arch = get_rocm_arch()
    if gpu_arch not in SMALL_M_SUPPORTED_ARCHS or not (
        1 <= m < SMALL_M_KERNEL_MAX
    ):
        return

    seen_configs = set()
    staging_options = (
        LDS_STAGING_OPTIONS if gpu_arch == "gfx942" else (LDS_STAGING_DIRECT,)
    )
    for tile_n in _small_m_tile_n_options(gpu_arch):
        for tile_k in _small_m_tile_k_options(k):
            split_k_options = _small_m_split_k_options(k, tile_k)
            if not split_k_options:
                continue
            for split_k in split_k_options:
                for variant, lds_staging in product(
                    _small_m_registry_variants(), staging_options
                ):
                    config = {
                        "kernel_family": "small_m",
                        "stage": STAGES,
                        "tile_m": TILE_M,
                        "tile_n": tile_n,
                        "tile_k": tile_k,
                        "split_k": split_k,
                        "block_m_warps": BLOCK_M_WARPS,
                        "block_n_warps": variant["block_n_warps"],
                        "n_tile_repeat": variant["n_tile_repeat"],
                        "persistent_n_tiles": variant["persistent_n_tiles"],
                        "waves_per_eu": variant["waves_per_eu"],
                        "b_to_lds_unroll": variant["b_to_lds_unroll"],
                        "async_copy": True,
                        "b_to_lds": variant["b_to_lds"],
                        "lds_staging": lds_staging,
                        "c_to_lds": False,
                        "dtype": dtype,
                        "out_dtype": out_dtype,
                        "target_gfx": gpu_arch,
                    }
                    try:
                        _validate_small_m_registry_config(
                            m,
                            n,
                            k,
                            tile_n=config["tile_n"],
                            tile_k=config["tile_k"],
                            split_k=config["split_k"],
                            block_n_warps=config["block_n_warps"],
                            n_tile_repeat=config["n_tile_repeat"],
                            persistent_n_tiles=config["persistent_n_tiles"],
                            waves_per_eu=config["waves_per_eu"],
                            b_to_lds_unroll=config["b_to_lds_unroll"],
                            b_to_lds=config["b_to_lds"],
                            lds_staging=config["lds_staging"],
                            arch=gpu_arch,
                        )
                    except ValueError:
                        continue
                    config = _canonicalize_small_m_registry_config(config)
                    config_key = tuple(sorted(config.items()))
                    if config_key in seen_configs:
                        continue
                    seen_configs.add(config_key)
                    yield config


@functools.lru_cache(maxsize=1024)
def compile_small_m_hgemm_kernel(
    dtype: str,
    n: int,
    k: int,
    *,
    ARCH: str | None = None,
    TILE_N: int = 128,
    TILE_K: int = 64,
    SPLIT_K: int = 1,
    BLOCK_N_WARPS: int = 2,
    N_TILE_REPEAT: int = 1,
    PERSISTENT_N_TILES: int = 1,
    WAVES_PER_EU_HINT: int = 0,
    B_TO_LDS_UNROLL: int = 0,
    B_TO_LDS: bool = False,
    HAS_BIAS: bool = False,
    LDS_STAGING: str = LDS_STAGING_DIRECT,
):
    if dtype != "bf16":
        raise ValueError(f"`small_m_hgemm.py` only supports bf16, got {dtype!r}")
    if SPLIT_K < 1:
        raise ValueError(f"SPLIT_K must be >= 1, got {SPLIT_K}")
    if LDS_STAGING not in LDS_STAGING_OPTIONS:
        raise ValueError(
            f"LDS_STAGING must be one of {LDS_STAGING_OPTIONS}, got {LDS_STAGING!r}"
        )

    GPU_ARCH = get_rocm_arch() if ARCH is None else ARCH
    _require_small_m_arch(GPU_ARCH)
    if GPU_ARCH != "gfx942" and LDS_STAGING != LDS_STAGING_DIRECT:
        raise ValueError(
            f"LDS_STAGING={LDS_STAGING!r} is only implemented for gfx942; "
            f"{GPU_ARCH} keeps the direct global-to-LDS path"
        )

    ARCH_PARAMS = small_m_arch_params(GPU_ARCH)
    WMMA_IMPL = ARCH_PARAMS["wmma_cls"](dtype)
    DMA_BYTES = ARCH_PARAMS["direct_dma_bytes"]
    MFMA_PER_WARP_K = ARCH_PARAMS["mfma_per_warp_k"]
    DIRECT_TO_LDS = LDS_STAGING == LDS_STAGING_DIRECT
    MAX_LDS_BYTES = small_m_max_lds_bytes(GPU_ARCH)
    BLOCK_K = TILE_K
    IS_SPLIT_K = SPLIT_K > 1
    if not small_m_tile_k_is_swizzle_safe(BLOCK_K):
        raise ValueError(
            f"TILE_K={TILE_K} is not supported by the XOR-16 LDS swizzle"
        )
    if k % SPLIT_K != 0:
        raise ValueError(f"K={k} is not divisible by SPLIT_K={SPLIT_K}")
    ks = k // SPLIT_K
    if ks < BLOCK_K or ks % BLOCK_K != 0:
        raise ValueError(
            f"K/SPLIT_K={ks} is not a positive multiple of TILE_K={BLOCK_K}; "
            "this shape has no legal small-M K schedule"
        )

    WMMA_M = WMMA_IMPL.WMMA_M
    WMMA_N = WMMA_IMPL.WMMA_N
    WMMA_K = WMMA_IMPL.WMMA_K
    WMMA_A_FRAG_VALUES = WMMA_IMPL.WMMA_A_FRAG_VALUES
    WMMA_B_FRAG_VALUES = WMMA_IMPL.WMMA_B_FRAG_VALUES
    WMMA_C_FRAG_VALUES = WMMA_IMPL.WMMA_C_FRAG_VALUES
    WARP_ATOM_M = WMMA_M
    WARP_ATOM_N = WMMA_N
    WARP_ATOM_K = WMMA_K * MFMA_PER_WARP_K
    BLOCK_K_LOOPS = ks // BLOCK_K
    WARP_K_STEPS = BLOCK_K // WARP_ATOM_K
    assert (BLOCK_K % WARP_ATOM_K == 0) and (WARP_K_STEPS >= 1)

    BLOCK_THREADS = BLOCK_N_WARPS * WARP_SIZE
    WARP_M_STEPS = TILE_M // BLOCK_M_WARPS // WARP_ATOM_M
    WARP_N_STEPS = TILE_N // BLOCK_N_WARPS // WARP_ATOM_N
    assert WARP_M_STEPS == 1
    assert (WARP_N_STEPS >= 1) and (TILE_N % (BLOCK_N_WARPS * WARP_ATOM_N) == 0)

    WARP_M = WARP_M_STEPS * WARP_ATOM_M
    WARP_N = WARP_N_STEPS * WARP_ATOM_N
    BLOCK_M = BLOCK_M_WARPS * WARP_M
    BLOCK_N = BLOCK_N_WARPS * WARP_N
    assert BLOCK_M == TILE_M
    if n < BLOCK_N or n % BLOCK_N != 0:
        raise ValueError(
            f"N={n} is not a positive multiple of the block-N width {BLOCK_N}"
        )
    BLOCK_N_TILES = n // BLOCK_N
    if N_TILE_REPEAT > 1:
        if B_TO_LDS:
            raise ValueError("wide-N repeat path only supports B_TO_LDS=False")
        classic_repeat = BLOCK_N_WARPS == 1 and TILE_N == 64
        wave_repeat = N_TILE_REPEAT == 2 and BLOCK_N_WARPS == 2 and TILE_N == 192
        if not (classic_repeat or wave_repeat):
            raise ValueError(
                "wide-N repeat path requires either the classic "
                "(BLOCK_N_WARPS=1, TILE_N=64, N_TILE_REPEAT>1) setup or the "
                "wave-specialized (N_TILE_REPEAT=2, BLOCK_N_WARPS=2, TILE_N=192) setup"
            )
    if PERSISTENT_N_TILES > 1:
        if not B_TO_LDS:
            raise ValueError("persistent-N path requires B_TO_LDS=True")
        if N_TILE_REPEAT != 1:
            raise ValueError("persistent-N path requires N_TILE_REPEAT=1")
        if TILE_N < 128:
            raise ValueError("persistent-N path currently requires TILE_N >= 128")
        if BLOCK_N_WARPS < 2:
            raise ValueError("persistent-N path currently requires BLOCK_N_WARPS >= 2")
        if PERSISTENT_N_TILES > BLOCK_N_TILES:
            raise ValueError(
                "persistent-N path requires PERSISTENT_N_TILES <= total N tiles; "
                f"got {PERSISTENT_N_TILES} > {BLOCK_N_TILES}"
            )
    PERSISTENT_N = PERSISTENT_N_TILES > 1
    WIDE_N_B_TO_LDS = (
        B_TO_LDS and N_TILE_REPEAT == 1 and TILE_N >= 128 and BLOCK_N_WARPS >= 2
    )
    WAVES_PER_EU = (
        int(WAVES_PER_EU_HINT)
        if const_expr(WAVES_PER_EU_HINT > 0)
        else (2 if const_expr(WIDE_N_B_TO_LDS) else 0)
    )
    EFFECTIVE_B_TO_LDS_UNROLL = (
        int(B_TO_LDS_UNROLL) if const_expr(B_TO_LDS_UNROLL > 0) else 8
    )

    BLOCK_MK_SIZE = BLOCK_M * BLOCK_K
    BLOCK_NK_SIZE = BLOCK_N * BLOCK_K
    BLOCK_MN_SIZE = BLOCK_M * BLOCK_N
    LDG_A_X_THREADS = BLOCK_K // LDG_VEC_SIZE
    LDG_B_X_THREADS = BLOCK_K // LDG_VEC_SIZE
    LDG_C_X_THREADS = BLOCK_N // LDG_VEC_SIZE
    assert BLOCK_MK_SIZE % LDG_VEC_SIZE == 0
    assert BLOCK_NK_SIZE % LDG_VEC_SIZE == 0
    assert BLOCK_MN_SIZE % LDG_VEC_SIZE == 0
    LDG_A_TOTAL_VECS = BLOCK_MK_SIZE // LDG_VEC_SIZE
    LDG_B_TOTAL_VECS = BLOCK_NK_SIZE // LDG_VEC_SIZE
    LDG_C_TOTAL_VECS = BLOCK_MN_SIZE // LDG_VEC_SIZE
    LDG_REG_A_COUNT = _ceil_div(LDG_A_TOTAL_VECS, BLOCK_THREADS)
    LDG_REG_B_COUNT = _ceil_div(LDG_B_TOTAL_VECS, BLOCK_THREADS)
    LDG_REG_C_COUNT = _ceil_div(LDG_C_TOTAL_VECS, BLOCK_THREADS)
    assert (LDG_REG_A_COUNT >= 1) and (LDG_REG_B_COUNT >= 1) and (LDG_REG_C_COUNT >= 1)

    BLOCK_K_BYTES = BLOCK_K * DTYPE_BYTES

    # LDS layout: C output (and the split-K arrival counter) alias the A tile
    # region; B has its own field only on the B_TO_LDS path.
    A_FIELD_ELEMS = max(STAGES * BLOCK_M * BLOCK_K, BLOCK_M * BLOCK_N)
    B_FIELD_ELEMS = STAGES * BLOCK_N * BLOCK_K if B_TO_LDS else 0
    LDS_BYTES = (A_FIELD_ELEMS + B_FIELD_ELEMS) * DTYPE_BYTES
    if LDS_BYTES > MAX_LDS_BYTES:
        raise ValueError(
            f"small-M config needs {LDS_BYTES} B of LDS but {GPU_ARCH} "
            f"provides {MAX_LDS_BYTES} B"
        )
    assert LDS_BYTES == small_m_lds_bytes(
        tile_n=TILE_N, tile_k=TILE_K, b_to_lds=B_TO_LDS
    )
    fx_dtype = fx.BFloat16
    if B_TO_LDS:

        @fx.struct
        class SharedStorage:
            a_lds: fx.Array[fx_dtype, A_FIELD_ELEMS, 16]
            b_lds: fx.Array[fx_dtype, B_FIELD_ELEMS, 16]

    else:

        @fx.struct
        class SharedStorage:
            a_lds: fx.Array[fx_dtype, A_FIELD_ELEMS, 16]

    LDG_ASYNC_VEC_SIZE = DMA_BYTES // DTYPE_BYTES
    LDG_A_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
    LDG_B_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
    assert BLOCK_MK_SIZE % LDG_ASYNC_VEC_SIZE == 0
    assert BLOCK_NK_SIZE % LDG_ASYNC_VEC_SIZE == 0
    LDG_A_TOTAL_VECS_AS = BLOCK_MK_SIZE // LDG_ASYNC_VEC_SIZE
    LDG_B_TOTAL_VECS_AS = BLOCK_NK_SIZE // LDG_ASYNC_VEC_SIZE
    LDG_REG_A_COUNT_AS = _ceil_div(LDG_A_TOTAL_VECS_AS, BLOCK_THREADS)
    LDG_REG_B_COUNT_AS = _ceil_div(LDG_B_TOTAL_VECS_AS, BLOCK_THREADS)
    STAGE_VMEM_A_COUNT = LDG_REG_A_COUNT_AS if DIRECT_TO_LDS else LDG_REG_A_COUNT
    STAGE_VMEM_B_COUNT = LDG_REG_B_COUNT_AS if DIRECT_TO_LDS else LDG_REG_B_COUNT
    STAGE_DSWR_A_COUNT = 0 if DIRECT_TO_LDS else LDG_REG_A_COUNT
    STAGE_DSWR_B_COUNT = 0 if DIRECT_TO_LDS else LDG_REG_B_COUNT

    KERNEL_NAME = small_m_kernel_name(
        dtype,
        target_gfx=GPU_ARCH,
        tile_n=TILE_N,
        tile_k=TILE_K,
        split_k=SPLIT_K,
        block_n_warps=BLOCK_N_WARPS,
        n_tile_repeat=N_TILE_REPEAT,
        persistent_n_tiles=PERSISTENT_N_TILES,
        waves_per_eu=WAVES_PER_EU,
        b_to_lds_unroll=EFFECTIVE_B_TO_LDS_UNROLL if const_expr(B_TO_LDS) else 0,
        b_to_lds=B_TO_LDS,
        has_bias=HAS_BIAS,
        lds_staging=LDS_STAGING,
    )

    @flyc.kernel
    def small_m_hgemm_kernel(
        C: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        BIAS: fx.Pointer,
        m: fx.Int32,
        semaphore: fx.Pointer,
        signal: fx.Pointer,
    ):
        dtype_ = get_dtype_in_kernel(dtype)
        _ptr_type = ir.Type.parse("!llvm.ptr<1>")
        _i64_type = T.i64
        c_zero_d = arith.constant(0.0, type=dtype_)
        acc_init = arith.constant_vector(0.0, T.vec(WMMA_C_FRAG_VALUES, T.f32))
        zero_a_vec = vector.broadcast(T.vec(LDG_VEC_SIZE, dtype_), c_zero_d)
        zero_a_async_vec = vector.broadcast(T.vec(LDG_ASYNC_VEC_SIZE, dtype_), c_zero_d)

        A_ = GTensor(A, dtype=dtype_, shape=(-1, k))
        B_ = GTensor(B, dtype=dtype_, shape=(n, k))
        C_ = GTensor(C, dtype=dtype_, shape=(-1, n))
        BIAS_ = GTensor(BIAS, dtype=dtype_, shape=(n,))

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_lds_ptr = lds.a_lds.ptr
        a_lds_i64 = fx.Int64(fx.ptrtoint(a_lds_ptr))
        if const_expr(B_TO_LDS):
            b_lds_ptr = lds.b_lds.ptr
            b_lds_i64 = fx.Int64(fx.ptrtoint(b_lds_ptr))

        # LDS accessors: linear element offsets mirroring the old STensor shapes.
        # as_/bs_ = (stage, row, col) over (STAGES, BLOCK*, BLOCK_K); cs_ =
        # (row, col) over (BLOCK_M, BLOCK_N) aliasing the A field; the split-K
        # arrival counter reinterprets the A field as i32.
        def as_store(stage, row, col, value):
            elem_off = (
                fx.Int64(stage) * (BLOCK_M * BLOCK_K)
                + fx.Int64(row) * BLOCK_K
                + fx.Int64(col)
            )
            fx.ptr_store(value, a_lds_ptr + elem_off)

        def as_load(stage, row, col, vec_size):
            elem_off = (
                fx.Int64(stage) * (BLOCK_M * BLOCK_K)
                + fx.Int64(row) * BLOCK_K
                + fx.Int64(col)
            )
            return fx.ptr_load(
                a_lds_ptr + elem_off,
                result_type=fx.Vector.make_type(vec_size, fx_dtype),
            )

        def bs_load(stage, row, col, vec_size):
            elem_off = (
                fx.Int64(stage) * (BLOCK_N * BLOCK_K)
                + fx.Int64(row) * BLOCK_K
                + fx.Int64(col)
            )
            return fx.ptr_load(
                b_lds_ptr + elem_off,
                result_type=fx.Vector.make_type(vec_size, fx_dtype),
            )

        def bs_store(stage, row, col, value):
            elem_off = (
                fx.Int64(stage) * (BLOCK_N * BLOCK_K)
                + fx.Int64(row) * BLOCK_K
                + fx.Int64(col)
            )
            fx.ptr_store(value, b_lds_ptr + elem_off)

        def cs_store_scalar(row, col, value):
            elem_off = fx.Int64(row) * BLOCK_N + fx.Int64(col)
            fx.ptr_store(value, a_lds_ptr + elem_off)

        def cs_load_vec(row, col, vec_size):
            elem_off = fx.Int64(row) * BLOCK_N + fx.Int64(col)
            return fx.ptr_load(
                a_lds_ptr + elem_off,
                result_type=fx.Vector.make_type(vec_size, fx_dtype),
            )

        if const_expr(IS_SPLIT_K):
            bc_i32_ptr = fx.recast_iter(fx.Int32, a_lds_ptr)

        tid = fx.Int32(fx.thread_idx.x)
        wid = tid // WARP_SIZE
        w_tid = tid % WARP_SIZE
        block_m_idx = fx.block_idx.x
        block_n_group_idx = fx.Index(fx.block_idx.y)
        ks_idx = fx.Index(fx.block_idx.z)
        ks_begin = arith.index_cast(T.i32, ks_idx * ks)
        block_n_tiles = n // BLOCK_N
        tile_group = PERSISTENT_N_TILES if const_expr(PERSISTENT_N) else N_TILE_REPEAT

        m_offset = fx.Index(block_m_idx * BLOCK_M)
        tile_block_n_indices = [
            block_n_group_idx * fx.Index(tile_group) + fx.Index(tile_i)
            for tile_i in range_constexpr(tile_group)
        ]
        tile_n_offsets = [
            tile_block_n_idx * fx.Index(BLOCK_N)
            for tile_block_n_idx in tile_block_n_indices
        ]
        tile_actives = [
            arith.cmpi(
                arith.CmpIPredicate.ult,
                tile_block_n_idx,
                fx.Index(block_n_tiles),
            )
            for tile_block_n_idx in tile_block_n_indices
        ]
        tile_signal_indices = [
            fx.block_idx.x * fx.Int32(block_n_tiles)
            + arith.index_cast(T.i32, tile_block_n_idx)
            for tile_block_n_idx in tile_block_n_indices
        ]
        k_blocks16 = fx.Int32(BLOCK_K_BYTES // 16)

        warp_m_idx = fx.Int32(0)
        warp_n_idx = wid * WARP_N
        ldmatrix_a_m_idx = w_tid % WMMA_M
        ldmatrix_a_k_vec_idx = w_tid // WMMA_M * WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K
        ldmatrix_b_n_idx = w_tid % WMMA_N
        ldmatrix_b_k_vec_idx = w_tid // WMMA_N * WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K

        A_FRAGS_LEN = WARP_K_STEPS * WARP_M_STEPS
        B_FRAGS_LEN = WARP_K_STEPS * WARP_N_STEPS
        C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS
        B_FRAG_T = T.vec(WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K, dtype_)
        zero_b_frag = vector.broadcast(B_FRAG_T, c_zero_d)
        c_frags = [acc_init] * (C_FRAGS_LEN * N_TILE_REPEAT)

        def zero_c_tile(c_g, bias_g, tile_n_offset):
            zero_vec = vector.broadcast(T.vec(LDG_VEC_SIZE, dtype_), c_zero_d)
            for i in range_constexpr(LDG_REG_C_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_C_X_THREADS
                n_local_idx = global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE
                row_idx = m_offset + fx.Index(m_local_idx)
                init_vec = zero_vec
                if const_expr(HAS_BIAS):
                    init_vec = bias_g.vec_load(
                        (tile_n_offset + n_local_idx,), LDG_VEC_SIZE
                    )
                cond_boundary = arith.cmpi(
                    arith.CmpIPredicate.ult, row_idx, fx.Index(m)
                )
                cond_boundary_if = scf.IfOp(cond_boundary, results_=[], has_else=False)
                with ir.InsertionPoint(cond_boundary_if.then_block):
                    c_g.vec_store(
                        (row_idx, tile_n_offset + n_local_idx), init_vec, LDG_VEC_SIZE
                    )
                    scf.YieldOp([])

        def get_llvm_ptr(ptr, offset, dtype_bytes):
            base_ptr = arith.index_cast(_i64_type, fx.ptrtoint(ptr))
            byte_offset = arith.index_cast(
                T.i64, fx.Index(offset) * fx.Index(dtype_bytes)
            )
            llvm_ptr = llvm.AddOp(
                base_ptr, byte_offset, llvm.IntegerOverflowFlags(0)
            ).result
            llvm_ptr = llvm.IntToPtrOp(_ptr_type, llvm_ptr).result
            return llvm_ptr._value if hasattr(llvm_ptr, "_value") else llvm_ptr

        def prepare_split_k_tile(c_g, bias_g, tile_n_offset, tile_signal_idx):
            is_t0_cond = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(tid), fx.Index(0))
            is_t0_cond_if = scf.IfOp(is_t0_cond, results_=[], has_else=False)
            with ir.InsertionPoint(is_t0_cond_if.then_block):
                semaphore_ptr = get_llvm_ptr(semaphore, tile_signal_idx, 4)
                prev = llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    semaphore_ptr,
                    arith.constant(1, type=T.i32),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                ).result
                fx.ptr_store(prev, bc_i32_ptr)
                scf.YieldOp([])
            gpu.barrier()
            arrive_idx = fx.Index(fx.ptr_load(bc_i32_ptr))

            first_arrival = arith.cmpi(arith.CmpIPredicate.eq, arrive_idx, fx.Index(0))
            first_arrival_if = scf.IfOp(first_arrival, results_=[], has_else=False)
            with ir.InsertionPoint(first_arrival_if.then_block):
                zero_c_tile(c_g, bias_g, tile_n_offset)
                llvm.InlineAsmOp(
                    None,
                    [],
                    "s_waitcnt vmcnt(0)",
                    "",
                    has_side_effects=True,
                )
                gpu.barrier()
                is_t0_cond_if = scf.IfOp(is_t0_cond, results_=[], has_else=False)
                with ir.InsertionPoint(is_t0_cond_if.then_block):
                    signal_ptr = get_llvm_ptr(signal, tile_signal_idx, 4)
                    llvm.AtomicRMWOp(
                        llvm.AtomicBinOp.xchg,
                        signal_ptr,
                        arith.constant(1, type=T.i32),
                        llvm.AtomicOrdering.release,
                        syncscope="agent",
                        alignment=4,
                    )
                    scf.YieldOp([])
                gpu.barrier()
                scf.YieldOp([])

        def split_k_barrier(tile_signal_idx):
            init_cur = arith.constant(0, type=T.i32)
            w = scf.WhileOp([T.i32], [init_cur])
            before = ir.Block.create_at_start(w.before, [T.i32])
            after = ir.Block.create_at_start(w.after, [T.i32])
            with ir.InsertionPoint(before):
                cur = before.arguments[0]
                need_wait = arith.CmpIOp(
                    arith.CmpIPredicate.eq, cur, arith.constant(0, type=T.i32)
                ).result
                scf.ConditionOp(need_wait, [cur])
            with ir.InsertionPoint(after):
                signal_ptr = get_llvm_ptr(signal, tile_signal_idx, 4)
                data = llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    signal_ptr,
                    arith.constant(0, type=T.i32),
                    llvm.AtomicOrdering.acquire,
                    syncscope="agent",
                    alignment=4,
                ).result
                scf.YieldOp([data])
            rocdl.sched_barrier(0)
            gpu.barrier()

            is_t0_cond = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(tid), fx.Index(0))
            is_t0_cond_if = scf.IfOp(is_t0_cond, results_=[T.i32], has_else=True)
            with ir.InsertionPoint(is_t0_cond_if.then_block):
                semaphore_ptr = get_llvm_ptr(semaphore, tile_signal_idx, 4)
                arrive_idx = llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    semaphore_ptr,
                    arith.constant(1, type=T.i32),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                ).result
                scf.YieldOp([arrive_idx])
            with ir.InsertionPoint(is_t0_cond_if.else_block):
                scf.YieldOp([arith.constant(0, type=T.i32)])

            last_departure = arith.cmpi(
                arith.CmpIPredicate.eq,
                is_t0_cond_if.results[0],
                arith.constant(2 * SPLIT_K - 1, type=T.i32),
            )
            last_departure_if = scf.IfOp(last_departure, results_=[], has_else=False)
            with ir.InsertionPoint(last_departure_if.then_block):
                zero = arith.constant(0, type=T.i32)
                semaphore_ptr = get_llvm_ptr(semaphore, tile_signal_idx, 4)
                signal_ptr = get_llvm_ptr(signal, tile_signal_idx, 4)
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.xchg,
                    semaphore_ptr,
                    zero,
                    llvm.AtomicOrdering.release,
                    syncscope="agent",
                    alignment=4,
                )
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.xchg,
                    signal_ptr,
                    zero,
                    llvm.AtomicOrdering.release,
                    syncscope="agent",
                    alignment=4,
                )
                scf.YieldOp([])
            gpu.barrier()

        def ldg_a(k_offset):
            vecs = []
            for i in range_constexpr(LDG_REG_A_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS
                k_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                row_idx = m_offset + fx.Index(m_local_idx)
                col_idx = fx.Index(k_offset + k_local_idx)
                slot_valid = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    fx.Index(global_tid),
                    fx.Index(LDG_A_TOTAL_VECS),
                )
                valid_row = arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(m))
                can_load = arith.andi(slot_valid, valid_row)
                load_if = scf.IfOp(
                    can_load,
                    results_=[T.vec(LDG_VEC_SIZE, dtype_)],
                    has_else=True,
                )
                with ir.InsertionPoint(load_if.then_block):
                    scf.YieldOp([A_.vec_load((row_idx, col_idx), LDG_VEC_SIZE)])
                with ir.InsertionPoint(load_if.else_block):
                    scf.YieldOp([zero_a_vec])
                vecs.append(load_if.results[0])
            return vecs

        def sts_a(vecs, lds_stage):
            for i in range_constexpr(LDG_REG_A_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS
                k_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                col_in_bytes = k_local_idx * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
                slot_valid = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    fx.Index(global_tid),
                    fx.Index(LDG_A_TOTAL_VECS),
                )
                store_if = scf.IfOp(slot_valid, results_=[], has_else=False)
                with ir.InsertionPoint(store_if.then_block):
                    as_store(
                        lds_stage, m_local_idx, col_in_bytes // DTYPE_BYTES, vecs[i]
                    )
                    scf.YieldOp([])

        def ldg_sts_a_async(k_offset, lds_stage):
            for i in range_constexpr(LDG_REG_A_COUNT_AS):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS_AS
                k_local_idx = global_tid % LDG_A_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
                col_in_bytes = k_local_idx * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
                row_idx = m_offset + fx.Index(m_local_idx)
                col_idx = fx.Index(k_offset + col_in_bytes // DTYPE_BYTES)
                slot_valid = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    fx.Index(global_tid),
                    fx.Index(LDG_A_TOTAL_VECS_AS),
                )
                slot_if = scf.IfOp(slot_valid, results_=[], has_else=False)
                with ir.InsertionPoint(slot_if.then_block):
                    valid_row = arith.cmpi(
                        arith.CmpIPredicate.ult, row_idx, fx.Index(m)
                    )
                    cond_if = scf.IfOp(valid_row, results_=[], has_else=True)
                    with ir.InsertionPoint(cond_if.then_block):
                        global_offset = (
                            A_.linear_offset((row_idx, col_idx)) * DTYPE_BYTES
                        )
                        global_offset = arith.index_cast(T.i32, global_offset)
                        lds_elem_off = (
                            fx.Index(lds_stage) * (BLOCK_M * BLOCK_K)
                            + fx.Index(m_local_idx) * BLOCK_K
                            + fx.Index(k_local_idx)
                        )
                        lds_byte_off = arith.index_cast(
                            T.i64, lds_elem_off * fx.Index(DTYPE_BYTES)
                        )
                        lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                        lds_addr_ = rocdl.readfirstlane(
                            T.i64, a_lds_i64 + fx.Int64(lds_byte_off)
                        )
                        lds_ptr = llvm.inttoptr(lds_ptr_type, lds_addr_)
                        rocdl.raw_ptr_buffer_load_lds(
                            A_.rsrc,
                            lds_ptr,
                            arith.constant(DMA_BYTES, type=T.i32),
                            global_offset,
                            arith.constant(0, type=T.i32),
                            arith.constant(0, type=T.i32),
                            arith.constant(1, type=T.i32),
                        )
                        scf.YieldOp([])
                    with ir.InsertionPoint(cond_if.else_block):
                        as_store(lds_stage, m_local_idx, k_local_idx, zero_a_async_vec)
                        scf.YieldOp([])
                    scf.YieldOp([])

        def lds_matrix_a(lds_stage):
            s = fx.Index(lds_stage)
            a_frags = [0] * A_FRAGS_LEN
            for ii in range_constexpr(WARP_M_STEPS):
                warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
                for kk in range_constexpr(WARP_K_STEPS):
                    warp_atom_k_idx = kk * WARP_ATOM_K
                    row = warp_atom_m_idx + ldmatrix_a_m_idx
                    col_in_bytes = (
                        warp_atom_k_idx + ldmatrix_a_k_vec_idx
                    ) * DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                    vec = as_load(
                        s,
                        row,
                        col_in_bytes // DTYPE_BYTES,
                        WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K,
                    )
                    a_frags[kk * WARP_M_STEPS + ii] = vec
            return a_frags

        def ldg_matrix_b(k_offset, tile_n_offset):
            vecs = []
            for kk in range_constexpr(WARP_K_STEPS):
                warp_atom_k_idx = kk * WARP_ATOM_K
                for ii in range_constexpr(WARP_N_STEPS):
                    warp_atom_n_idx = warp_n_idx + ii * WARP_ATOM_N
                    n_idx = tile_n_offset + warp_atom_n_idx + ldmatrix_b_n_idx
                    k_idx = k_offset + warp_atom_k_idx + ldmatrix_b_k_vec_idx
                    vec = B_.vec_load(
                        (n_idx, k_idx), WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K
                    )
                    vecs.append(vec)
            return vecs

        def maybe_ldg_matrix_b(k_offset, tile_n_offset, tile_active):
            if const_expr(N_TILE_REPEAT == 1):
                return ldg_matrix_b(k_offset, tile_n_offset)
            load_if = scf.IfOp(
                tile_active,
                results_=[B_FRAG_T] * B_FRAGS_LEN,
                has_else=True,
            )
            with ir.InsertionPoint(load_if.then_block):
                scf.YieldOp(ldg_matrix_b(k_offset, tile_n_offset))
            with ir.InsertionPoint(load_if.else_block):
                scf.YieldOp([zero_b_frag] * B_FRAGS_LEN)
            return list(load_if.results)

        def split_mfma_k_halves(frag):
            """Split one logical K32 fragment into native gfx942 K16 operands."""
            pair = vector.bitcast(T.i64x2, frag)
            halves = []
            for half in range_constexpr(2):
                part = vector.extract(
                    pair, static_position=[half], dynamic_position=[]
                )
                halves.append(
                    vector.bitcast(
                        T.f16x4,
                        vector.from_elements(T.vec(1, T.i64), [part]),
                    )
                )
            return halves

        def block_mma_sync(a_frags, b_frags, c_frags):
            c_frags_new = [cx for cx in c_frags]
            for kk in range_constexpr(WARP_K_STEPS):
                for ii in range_constexpr(WARP_M_STEPS):
                    a_frag = a_frags[kk * WARP_M_STEPS + ii]
                    if const_expr(MFMA_PER_WARP_K == 2):
                        a_halves = split_mfma_k_halves(a_frag)
                    for jj in range_constexpr(WARP_N_STEPS):
                        b_frag = b_frags[kk * WARP_N_STEPS + jj]
                        c_idx = ii * WARP_N_STEPS + jj
                        if const_expr(MFMA_PER_WARP_K == 2):
                            b_halves = split_mfma_k_halves(b_frag)
                            acc = c_frags_new[c_idx]
                            for half in range_constexpr(2):
                                acc = WMMA_IMPL(a_halves[half], b_halves[half], acc)
                            c_frags_new[c_idx] = acc
                        else:
                            c_frags_new[c_idx] = WMMA_IMPL(
                                a_frag, b_frag, c_frags_new[c_idx]
                            )
            return c_frags_new

        def store_split_k_tile(c_tensor, c_g, tile_n_offset):
            out_raw = c_tensor
            out_base_int = arith.index_cast(_i64_type, fx.ptrtoint(out_raw))
            for i in range_constexpr(LDG_REG_C_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = fx.Index(global_tid // LDG_C_X_THREADS)
                n_local_idx = fx.Index(global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE)
                m_global_idx = m_offset + m_local_idx
                n_global_idx = tile_n_offset + n_local_idx
                cond_boundary = arith.cmpi(
                    arith.CmpIPredicate.ult, m_global_idx, fx.Index(m)
                )
                cond_boundary_if = scf.IfOp(cond_boundary, results_=[], has_else=False)
                with ir.InsertionPoint(cond_boundary_if.then_block):
                    pk_val = cs_load_vec(m_local_idx, n_local_idx, LDG_VEC_SIZE)
                    linear_bytes_offset = (
                        c_g.linear_offset((m_global_idx, n_global_idx)) * DTYPE_BYTES
                    )
                    vec2_ty = T.vec(2, dtype_)
                    for vec_idx in range_constexpr(LDG_VEC_SIZE // 2):
                        e0 = vector.extract(
                            pk_val,
                            static_position=[vec_idx * 2],
                            dynamic_position=[],
                        )
                        e1 = vector.extract(
                            pk_val,
                            static_position=[vec_idx * 2 + 1],
                            dynamic_position=[],
                        )
                        pair = vector.from_elements(vec2_ty, [e0, e1])
                        pair_byte_offset = arith.index_cast(
                            T.i64,
                            linear_bytes_offset + fx.Index(vec_idx * 2 * DTYPE_BYTES),
                        )
                        pair_addr_i64 = llvm.AddOp(
                            out_base_int,
                            pair_byte_offset,
                            llvm.IntegerOverflowFlags(0),
                        ).result
                        pair_ptr = llvm.IntToPtrOp(_ptr_type, pair_addr_i64).result
                        pair_ptr_v = (
                            pair_ptr._value if hasattr(pair_ptr, "_value") else pair_ptr
                        )
                        pair_v = pair._value if hasattr(pair, "_value") else pair
                        llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.fadd,
                            pair_ptr_v,
                            pair_v,
                            llvm.AtomicOrdering.monotonic,
                            syncscope="agent",
                            alignment=4,
                        )
                    scf.YieldOp([])

        def store_c_tile(bias_g, c_g, tile_n_offset):
            for i in range_constexpr(LDG_REG_C_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = fx.Index(global_tid // LDG_C_X_THREADS)
                n_local_idx = fx.Index(global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE)
                m_global_idx = m_offset + m_local_idx
                cond_boundary = arith.cmpi(
                    arith.CmpIPredicate.ult, m_global_idx, fx.Index(m)
                )
                cond_boundary_if = scf.IfOp(cond_boundary, results_=[], has_else=False)
                with ir.InsertionPoint(cond_boundary_if.then_block):
                    vec = cs_load_vec(m_local_idx, n_local_idx, LDG_VEC_SIZE)
                    if const_expr(HAS_BIAS):
                        bias_vec = bias_g.vec_load(
                            (tile_n_offset + n_local_idx,), LDG_VEC_SIZE
                        )
                        vec = vec + bias_vec
                    c_g.vec_store(
                        (m_global_idx, tile_n_offset + n_local_idx), vec, LDG_VEC_SIZE
                    )
                    scf.YieldOp([])

        stmatrix_c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG_VALUES
        stmatrix_c_n_idx = w_tid % WMMA_N

        def write_c_frags_to_lds(tile_c_frags_):
            for ii in range_constexpr(WARP_M_STEPS):
                warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
                for jj in range_constexpr(WARP_N_STEPS):
                    warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
                    for kk in range_constexpr(WMMA_C_FRAG_VALUES):
                        lds_m_idx = fx.Index(
                            warp_atom_m_idx + stmatrix_c_m_vec_idx + kk
                        )
                        lds_n_idx = fx.Index(warp_atom_n_idx + stmatrix_c_n_idx)
                        val = vector.extract(
                            tile_c_frags_[ii * WARP_N_STEPS + jj],
                            static_position=[kk],
                            dynamic_position=[],
                        )
                        cs_store_scalar(lds_m_idx, lds_n_idx, val.truncf(dtype_))

        if const_expr(IS_SPLIT_K and not B_TO_LDS):
            for tile_i in range_constexpr(N_TILE_REPEAT):
                tile_init_if = scf.IfOp(
                    tile_actives[tile_i], results_=[], has_else=False
                )
                with ir.InsertionPoint(tile_init_if.then_block):
                    prepare_split_k_tile(
                        C_,
                        BIAS_,
                        tile_n_offsets[tile_i],
                        tile_signal_indices[tile_i],
                    )
                    scf.YieldOp([])

        if const_expr(B_TO_LDS):

            def ldg_sts_b_async(k_offset, lds_stage, tile_n_offset):
                for i in range_constexpr(LDG_REG_B_COUNT_AS):
                    global_tid = BLOCK_THREADS * i + tid
                    n_local_idx = global_tid // LDG_B_X_THREADS_AS
                    k_local_idx = global_tid % LDG_B_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
                    col_in_bytes = k_local_idx * DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(n_local_idx, col_in_bytes, k_blocks16)
                    col_idx = fx.Index(k_offset + col_in_bytes // DTYPE_BYTES)
                    slot_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        fx.Index(global_tid),
                        fx.Index(LDG_B_TOTAL_VECS_AS),
                    )
                    slot_if = scf.IfOp(slot_valid, results_=[], has_else=False)
                    with ir.InsertionPoint(slot_if.then_block):
                        global_offset = B_.linear_offset(
                            (tile_n_offset + fx.Index(n_local_idx), col_idx)
                        )
                        global_offset = arith.index_cast(
                            T.i32, global_offset * DTYPE_BYTES
                        )
                        lds_elem_off = (
                            fx.Index(lds_stage) * (BLOCK_N * BLOCK_K)
                            + fx.Index(n_local_idx) * BLOCK_K
                            + fx.Index(k_local_idx)
                        )
                        lds_byte_off = arith.index_cast(
                            T.i64, lds_elem_off * fx.Index(DTYPE_BYTES)
                        )
                        lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                        lds_addr_ = rocdl.readfirstlane(
                            T.i64, b_lds_i64 + fx.Int64(lds_byte_off)
                        )
                        lds_ptr = llvm.inttoptr(lds_ptr_type, lds_addr_)
                        rocdl.raw_ptr_buffer_load_lds(
                            B_.rsrc,
                            lds_ptr,
                            arith.constant(DMA_BYTES, type=T.i32),
                            global_offset,
                            arith.constant(0, type=T.i32),
                            arith.constant(0, type=T.i32),
                            arith.constant(1, type=T.i32),
                        )
                        scf.YieldOp([])

            def ldg_b(k_offset, tile_n_offset):
                vecs = []
                for i in range_constexpr(LDG_REG_B_COUNT):
                    global_tid = BLOCK_THREADS * i + tid
                    n_local_idx = global_tid // LDG_B_X_THREADS
                    k_local_idx = global_tid % LDG_B_X_THREADS * LDG_VEC_SIZE
                    row_idx = tile_n_offset + fx.Index(n_local_idx)
                    col_idx = fx.Index(k_offset + k_local_idx)
                    slot_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        fx.Index(global_tid),
                        fx.Index(LDG_B_TOTAL_VECS),
                    )
                    load_if = scf.IfOp(
                        slot_valid,
                        results_=[T.vec(LDG_VEC_SIZE, dtype_)],
                        has_else=True,
                    )
                    with ir.InsertionPoint(load_if.then_block):
                        scf.YieldOp([B_.vec_load((row_idx, col_idx), LDG_VEC_SIZE)])
                    with ir.InsertionPoint(load_if.else_block):
                        scf.YieldOp([zero_a_vec])
                    vecs.append(load_if.results[0])
                return vecs

            def sts_b(vecs, lds_stage):
                for i in range_constexpr(LDG_REG_B_COUNT):
                    global_tid = BLOCK_THREADS * i + tid
                    n_local_idx = global_tid // LDG_B_X_THREADS
                    k_local_idx = global_tid % LDG_B_X_THREADS * LDG_VEC_SIZE
                    col_in_bytes = k_local_idx * DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(
                        n_local_idx, col_in_bytes, k_blocks16
                    )
                    slot_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        fx.Index(global_tid),
                        fx.Index(LDG_B_TOTAL_VECS),
                    )
                    store_if = scf.IfOp(slot_valid, results_=[], has_else=False)
                    with ir.InsertionPoint(store_if.then_block):
                        bs_store(
                            lds_stage,
                            n_local_idx,
                            col_in_bytes // DTYPE_BYTES,
                            vecs[i],
                        )
                        scf.YieldOp([])

            def lds_matrix_b(lds_stage):
                s = fx.Index(lds_stage)
                b_frags = [0] * B_FRAGS_LEN
                for ii in range_constexpr(WARP_N_STEPS):
                    warp_atom_n_idx = warp_n_idx + ii * WARP_ATOM_N
                    for kk in range_constexpr(WARP_K_STEPS):
                        warp_atom_k_idx = kk * WARP_ATOM_K
                        row = warp_atom_n_idx + ldmatrix_b_n_idx
                        col_in_bytes = (
                            warp_atom_k_idx + ldmatrix_b_k_vec_idx
                        ) * DTYPE_BYTES
                        col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                        vec = bs_load(
                            s,
                            row,
                            col_in_bytes // DTYPE_BYTES,
                            WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K,
                        )
                        b_frags[kk * WARP_N_STEPS + ii] = vec
                return b_frags

            def stage_ab_load(k_offset, tile_n_offset):
                if const_expr(DIRECT_TO_LDS):
                    return None
                return ldg_a(k_offset), ldg_b(k_offset, tile_n_offset)

            def stage_ab_commit(regs, k_offset, lds_stage, tile_n_offset):
                if const_expr(DIRECT_TO_LDS):
                    ldg_sts_a_async(k_offset, lds_stage)
                    ldg_sts_b_async(k_offset, lds_stage, tile_n_offset)
                else:
                    sts_a(regs[0], lds_stage)
                    sts_b(regs[1], lds_stage)

            def run_b_to_lds_tile(tile_n_offset, tile_signal_idx):
                c_frags_local = [acc_init] * C_FRAGS_LEN
                if const_expr(IS_SPLIT_K):
                    prepare_split_k_tile(C_, BIAS_, tile_n_offset, tile_signal_idx)

                stage_ab_commit(
                    stage_ab_load(ks_begin, tile_n_offset),
                    ks_begin,
                    0,
                    tile_n_offset,
                )
                gpu.barrier()

                def hot_loop_scheduler():
                    MFMA_TOTAL = (
                        WARP_K_STEPS * WARP_M_STEPS * WARP_N_STEPS * MFMA_PER_WARP_K
                    )
                    LDG_TOTAL = STAGE_VMEM_A_COUNT + STAGE_VMEM_B_COUNT
                    DS_WRITE_TOTAL = STAGE_DSWR_A_COUNT + STAGE_DSWR_B_COUNT
                    if const_expr(WIDE_N_B_TO_LDS):
                        for _ in range_constexpr(WARP_K_STEPS * WARP_M_STEPS):
                            rocdl.sched_dsrd(1)
                        for _ in range_constexpr(WARP_K_STEPS * WARP_N_STEPS):
                            rocdl.sched_dsrd(1)
                        for _ in range_constexpr(STAGE_VMEM_A_COUNT):
                            rocdl.sched_vmem(1)
                            rocdl.sched_mfma(2)
                        for _ in range_constexpr(STAGE_VMEM_B_COUNT):
                            rocdl.sched_vmem(1)
                            rocdl.sched_mfma(2)
                        remaining = max(MFMA_TOTAL - LDG_TOTAL * 2, 0)
                        for _ in range_constexpr(remaining):
                            rocdl.sched_mfma(1)
                    else:
                        for _ in range_constexpr(WARP_K_STEPS * WARP_M_STEPS):
                            rocdl.sched_dsrd(1)
                        for _ in range_constexpr(WARP_K_STEPS * WARP_N_STEPS):
                            rocdl.sched_dsrd(1)
                        for _ in range_constexpr(LDG_TOTAL):
                            rocdl.sched_vmem(1)
                            rocdl.sched_mfma(2)
                        remaining = max(MFMA_TOTAL - LDG_TOTAL * 2, 0)
                        for _ in range_constexpr(remaining):
                            rocdl.sched_mfma(1)
                    for _ in range_constexpr(DS_WRITE_TOTAL):
                        rocdl.sched_dswr(1)
                    rocdl.sched_barrier(0)

                UNROLL = EFFECTIVE_B_TO_LDS_UNROLL
                init_state = [ks_begin, arith.constant(0, index=True)] + c_frags_local
                for bki, state in range(0, BLOCK_K_LOOPS - 1, UNROLL, init=init_state):
                    k_offset = state[0]
                    current_stage = fx.Index(state[1])
                    c_frags_local = state[2 : 2 + C_FRAGS_LEN]
                    for unroll_i in range_constexpr(UNROLL):
                        cond = arith.cmpi(
                            arith.CmpIPredicate.ult,
                            fx.Index(bki + unroll_i),
                            fx.Index(BLOCK_K_LOOPS - 1),
                        )
                        cond_if = scf.IfOp(
                            cond,
                            results_=[T.vec(WMMA_C_FRAG_VALUES, T.f32)] * C_FRAGS_LEN
                            + [T.index, T.i32],
                            has_else=True,
                        )
                        with ir.InsertionPoint(cond_if.then_block):
                            next_stage = 1 - current_stage
                            a_frags = lds_matrix_a(current_stage)
                            b_frags = lds_matrix_b(current_stage)
                            stage_regs = stage_ab_load(
                                k_offset + BLOCK_K, tile_n_offset
                            )
                            if const_expr(DIRECT_TO_LDS):
                                stage_ab_commit(
                                    stage_regs,
                                    k_offset + BLOCK_K,
                                    next_stage,
                                    tile_n_offset,
                                )
                            c_frags_new = block_mma_sync(
                                a_frags, b_frags, c_frags_local
                            )
                            if const_expr(not DIRECT_TO_LDS):
                                stage_ab_commit(
                                    stage_regs,
                                    k_offset + BLOCK_K,
                                    next_stage,
                                    tile_n_offset,
                                )
                            hot_loop_scheduler()
                            gpu.barrier()
                            k_offset_next = k_offset + fx.Int32(BLOCK_K)
                            current_stage_next = 1 - current_stage
                            scf.YieldOp(
                                c_frags_new
                                + [_to_raw(current_stage_next), k_offset_next]
                            )
                        with ir.InsertionPoint(cond_if.else_block):
                            scf.YieldOp(
                                c_frags_local + [_to_raw(current_stage), k_offset]
                            )
                        c_frags_local = [cond_if.results[i] for i in range(C_FRAGS_LEN)]
                        current_stage = cond_if.results[C_FRAGS_LEN]
                        k_offset = cond_if.results[C_FRAGS_LEN + 1]
                    results = yield [k_offset, current_stage] + c_frags_local
                current_stage = results[1]
                c_frags_local = results[2 : 2 + C_FRAGS_LEN]
                a_frags = lds_matrix_a(current_stage)
                b_frags = lds_matrix_b(current_stage)
                c_frags_local = block_mma_sync(a_frags, b_frags, c_frags_local)

                write_c_frags_to_lds(c_frags_local)
                gpu.barrier()
                if const_expr(IS_SPLIT_K):
                    split_k_barrier(tile_signal_idx)
                    store_split_k_tile(C, C_, tile_n_offset)
                else:
                    store_c_tile(BIAS_, C_, tile_n_offset)
                gpu.barrier()

            for tile_i in range_constexpr(tile_group):
                tile_exec_if = scf.IfOp(
                    tile_actives[tile_i], results_=[], has_else=False
                )
                with ir.InsertionPoint(tile_exec_if.then_block):
                    run_b_to_lds_tile(
                        tile_n_offsets[tile_i], tile_signal_indices[tile_i]
                    )
                    scf.YieldOp([])
        else:
            sts_a(ldg_a(ks_begin), 0)
            gpu.barrier()
            a_frags = lds_matrix_a(0)
            b_frags = []
            for tile_i in range_constexpr(N_TILE_REPEAT):
                b_frags.extend(
                    maybe_ldg_matrix_b(
                        ks_begin,
                        tile_n_offsets[tile_i],
                        tile_actives[tile_i],
                    )
                )
            rocdl.sched_barrier(0)

            def hot_loop_scheduler():
                MFMA_TOTAL = (
                    N_TILE_REPEAT
                    * WARP_K_STEPS
                    * WARP_M_STEPS
                    * WARP_N_STEPS
                    * MFMA_PER_WARP_K
                )
                LDG_TOTAL = (
                    STAGE_VMEM_A_COUNT
                    + N_TILE_REPEAT * WARP_K_STEPS * WARP_N_STEPS
                )
                avg_mfma_count = (MFMA_TOTAL + LDG_TOTAL - 1) // LDG_TOTAL
                mfma_sched = OnlineScheduler(MFMA_TOTAL, MFMA_TOTAL)
                ldg_sched = OnlineScheduler(LDG_TOTAL, LDG_TOTAL)
                for _ in range_constexpr(LDG_TOTAL):
                    rocdl.sched_vmem(ldg_sched.consume(1))
                    rocdl.sched_mfma(mfma_sched.consume(avg_mfma_count))
                for _ in range_constexpr(STAGE_DSWR_A_COUNT):
                    rocdl.sched_dswr(1)
                rocdl.sched_barrier(0)

            TOTAL_C_FRAGS_LEN = C_FRAGS_LEN * N_TILE_REPEAT
            TOTAL_B_FRAGS_LEN = B_FRAGS_LEN * N_TILE_REPEAT
            init_state = (
                [ks_begin, arith.constant(0, index=True)] + c_frags + a_frags + b_frags
            )
            for _, state in range(1, BLOCK_K_LOOPS, init=init_state):
                k_offset = state[0]
                current_stage = fx.Index(state[1])
                next_stage = 1 - current_stage
                c_frags = state[2 : 2 + TOTAL_C_FRAGS_LEN]
                a_frags = state[
                    2 + TOTAL_C_FRAGS_LEN : 2 + TOTAL_C_FRAGS_LEN + A_FRAGS_LEN
                ]
                b_frags = state[
                    2
                    + TOTAL_C_FRAGS_LEN
                    + A_FRAGS_LEN : 2
                    + TOTAL_C_FRAGS_LEN
                    + A_FRAGS_LEN
                    + TOTAL_B_FRAGS_LEN
                ]
                if const_expr(DIRECT_TO_LDS):
                    a_stage_regs = None
                    ldg_sts_a_async(k_offset + BLOCK_K, next_stage)
                else:
                    a_stage_regs = ldg_a(k_offset + BLOCK_K)
                b_frags_next = []
                c_frags_next = []
                for tile_i in range_constexpr(N_TILE_REPEAT):
                    b_start = tile_i * B_FRAGS_LEN
                    c_start = tile_i * C_FRAGS_LEN
                    b_frags_next.extend(
                        maybe_ldg_matrix_b(
                            k_offset + BLOCK_K,
                            tile_n_offsets[tile_i],
                            tile_actives[tile_i],
                        )
                    )
                    c_frags_next.extend(
                        block_mma_sync(
                            a_frags,
                            b_frags[b_start : b_start + B_FRAGS_LEN],
                            c_frags[c_start : c_start + C_FRAGS_LEN],
                        )
                    )
                c_frags = c_frags_next
                if const_expr(not DIRECT_TO_LDS):
                    sts_a(a_stage_regs, next_stage)
                hot_loop_scheduler()
                gpu.barrier()
                a_frags_next = lds_matrix_a(next_stage)
                k_offset = k_offset + fx.Int32(BLOCK_K)
                rocdl.sched_barrier(0)
                results = (
                    yield [k_offset, next_stage] + c_frags + a_frags_next + b_frags_next
                )
            c_frags = results[2 : 2 + TOTAL_C_FRAGS_LEN]
            a_frags = results[
                2 + TOTAL_C_FRAGS_LEN : 2 + TOTAL_C_FRAGS_LEN + A_FRAGS_LEN
            ]
            b_frags = results[
                2
                + TOTAL_C_FRAGS_LEN
                + A_FRAGS_LEN : 2
                + TOTAL_C_FRAGS_LEN
                + A_FRAGS_LEN
                + TOTAL_B_FRAGS_LEN
            ]
            c_frags_next = []
            for tile_i in range_constexpr(N_TILE_REPEAT):
                b_start = tile_i * B_FRAGS_LEN
                c_start = tile_i * C_FRAGS_LEN
                c_frags_next.extend(
                    block_mma_sync(
                        a_frags,
                        b_frags[b_start : b_start + B_FRAGS_LEN],
                        c_frags[c_start : c_start + C_FRAGS_LEN],
                    )
                )
            c_frags = c_frags_next

            tile_c_frags = [
                c_frags[tile_i * C_FRAGS_LEN : (tile_i + 1) * C_FRAGS_LEN]
                for tile_i in range_constexpr(N_TILE_REPEAT)
            ]

            for tile_i in range_constexpr(N_TILE_REPEAT):
                tile_store_if = scf.IfOp(
                    tile_actives[tile_i], results_=[], has_else=False
                )
                with ir.InsertionPoint(tile_store_if.then_block):
                    write_c_frags_to_lds(tile_c_frags[tile_i])
                    gpu.barrier()
                    if const_expr(IS_SPLIT_K):
                        split_k_barrier(tile_signal_indices[tile_i])
                        store_split_k_tile(C, C_, tile_n_offsets[tile_i])
                    else:
                        store_c_tile(BIAS_, C_, tile_n_offsets[tile_i])
                    gpu.barrier()
                    scf.YieldOp([])

    @flyc.jit
    def launch_small_m_hgemm_kernel(
        C: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        BIAS: fx.Pointer,
        m: fx.Int32,
        semaphore: fx.Pointer,
        signal: fx.Pointer,
        stream: fx.Stream,
    ):
        ctx = CompilationContext.get_current()
        if const_expr(WAVES_PER_EU > 0):
            for op in ctx.gpu_module_body.operations:
                if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                        T.i32, int(WAVES_PER_EU)
                    )

        bm = (m + BLOCK_M - 1) // BLOCK_M
        tile_group = PERSISTENT_N_TILES if const_expr(PERSISTENT_N) else N_TILE_REPEAT
        bn = (n // BLOCK_N + tile_group - 1) // tile_group
        small_m_hgemm_kernel._func.__name__ = KERNEL_NAME
        small_m_hgemm_kernel(C, A, B, BIAS, m, semaphore, signal).launch(
            grid=(bm, bn, SPLIT_K),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_small_m_hgemm_kernel
