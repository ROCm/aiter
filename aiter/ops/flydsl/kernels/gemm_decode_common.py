# SPDX-License-Identifier: MIT

"""Shared configuration, layouts, and numeric primitives for BF16 decode GEMM."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
import re
from typing import Iterator, TypeAlias

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as arith_dialect
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops, vector
from aiter.ops.flydsl.utils import addressable_lds_bytes_for_gfx

from .tensor_shim import _to_raw as raw


# Host configuration, validation, naming, and enumeration.
WAVE_SIZE = 64
MFMA_K = 4
BF16_BYTES = 2
SIGNED_INT32_MAX = (1 << 31) - 1
CACHE_POLICY_DEFAULT = 0
CACHE_POLICY_NON_TEMPORAL = 0x2
_CACHE_POLICY_MASK = 0x13


def validate_cache_policy(cache_policy: int) -> None:
    """Reject cache-policy bits that gfx942 lowering would silently discard."""
    if not isinstance(cache_policy, int):
        raise TypeError("cache policy must be an integer")
    if cache_policy < 0 or cache_policy & ~_CACHE_POLICY_MASK:
        raise ValueError(f"unsupported cache policy: {cache_policy:#x}")


class OutputRounding(str, Enum):
    RNE = "rne"
    STOCHASTIC = "stochastic"


class ReductionMode(str, Enum):
    DPP = "dpp"
    BPERMUTE = "bpermute"


class ContractionMode(str, Enum):
    SCALAR_F32 = "scalar"
    PACKED_F32 = "packed"
    DOT2_BF16 = "dot2"


class ActivationSource(str, Enum):
    GLOBAL = "global"
    FULL_LDS = "lds"


@dataclass(frozen=True)
class DecodeArchTraits:
    arch: str
    max_lds_bytes: int
    supports_dot2: bool
    supports_stochastic: bool
    supports_wide_b_load: bool
    supports_two_stage: bool
    max_estimated_live_vgprs: int


ARCH_TRAITS = {
    "gfx942": DecodeArchTraits(
        arch="gfx942",
        max_lds_bytes=64 * 1024,
        supports_dot2=False,
        supports_stochastic=False,
        supports_wide_b_load=True,
        supports_two_stage=False,
        max_estimated_live_vgprs=192,
    ),
    "gfx950": DecodeArchTraits(
        arch="gfx950",
        max_lds_bytes=160 * 1024,
        supports_dot2=True,
        supports_stochastic=True,
        supports_wide_b_load=True,
        supports_two_stage=True,
        max_estimated_live_vgprs=192,
    ),
}


def get_decode_arch_traits(arch: str) -> DecodeArchTraits:
    try:
        return ARCH_TRAITS[arch]
    except KeyError as error:
        raise ValueError(
            f"BF16 decode GEMM requires gfx942 or gfx950, got {arch}"
        ) from error


@dataclass(frozen=True)
class WaveDecodeConfig:
    """One-wave/no-LDS policy axes."""

    m_per_wave: int
    n_per_wave: int = 1
    kvec: int = 8
    prefetch_depth: int = 0
    waves_per_eu: int = 4
    b_cache_modifier: int = CACHE_POLICY_DEFAULT
    reduction: ReductionMode = ReductionMode.DPP
    contraction: ContractionMode = ContractionMode.SCALAR_F32
    output_rounding: OutputRounding = OutputRounding.RNE

    def validate(self, *, m: int, n: int, k: int, arch: str) -> None:
        traits = get_decode_arch_traits(arch)
        _validate_problem(m, n, k)
        if self.m_per_wave not in range(1, 6) or m % self.m_per_wave:
            raise ValueError(
                "wave policy requires m_per_wave to exactly divide M without "
                f"padding or a masked M tail, got M={m}, m_per_wave={self.m_per_wave}"
            )
        if self.n_per_wave not in (1, 2, 4):
            raise ValueError("n_per_wave must be 1, 2, or 4")
        if n % self.n_per_wave:
            raise ValueError("N must be divisible by n_per_wave")
        if self.kvec not in (2, 4, 8):
            raise ValueError("kvec must be 2, 4, or 8")
        if self.prefetch_depth not in (0, 1, 2):
            raise ValueError("prefetch_depth must be 0, 1, or 2")
        if arch == "gfx942" and self.prefetch_depth == 2:
            raise ValueError("two-stage wave prefetch is gfx950-only")
        if self.waves_per_eu not in (1, 2, 4):
            raise ValueError("waves_per_eu must be 1, 2, or 4")
        validate_cache_policy(self.b_cache_modifier)
        if self.contraction == ContractionMode.DOT2_BF16 and not traits.supports_dot2:
            raise ValueError("dot2 BF16 contraction requires gfx950")
        if arch == "gfx950" and self.contraction != ContractionMode.DOT2_BF16:
            raise ValueError("gfx950 wave policy uses native dot2 BF16 contraction")
        if self.output_rounding == OutputRounding.STOCHASTIC:
            if not traits.supports_stochastic:
                raise ValueError("stochastic BF16 conversion requires gfx950")
        if self.reduction == ReductionMode.DPP and k % self.kvec:
            raise ValueError("DPP reduction requires K divisible by kvec")
        validate_wave_i32_addressing(m, n, k, self)


@dataclass(frozen=True)
class BlockMfmaDecodeConfig:
    """Multi-wave 4x4x4 BF16 MFMA policy axes.

    ``workgroups_per_cu`` is a grid-cap multiplier, not a promise that this
    many workgroups can reside simultaneously on each CU. Actual residency
    also depends on compiler-assigned registers and device scheduling.
    """

    waves_per_workgroup: int = 8
    columns_per_wave: int = 1
    activation_source: ActivationSource = ActivationSource.GLOBAL
    b_load_width: int = MFMA_K
    k_unroll: int = 1
    prefetch_stages: int = 1
    persistent_n: bool = False
    workgroups_per_cu: int = 1
    waves_per_eu: int = 0
    b_cache_modifier: int = CACHE_POLICY_DEFAULT
    output_rounding: OutputRounding = OutputRounding.RNE

    def validate(self, *, m: int, n: int, k: int, arch: str) -> None:
        traits = get_decode_arch_traits(arch)
        _validate_problem(m, n, k)
        if self.waves_per_workgroup not in (4, 8, 12, 16):
            raise ValueError("waves_per_workgroup must be 4, 8, 12, or 16")
        if self.columns_per_wave not in (1, 2, 4):
            raise ValueError("columns_per_wave must be 1, 2, or 4")
        if self.b_load_width not in (4, 8):
            raise ValueError("b_load_width must be 4 or 8")
        if self.b_load_width == 8 and not traits.supports_wide_b_load:
            raise ValueError("wide B loads require gfx950")
        if self.k_unroll not in (1, 2):
            raise ValueError("k_unroll must be 1 or 2")
        if self.prefetch_stages not in (1, 2):
            raise ValueError("prefetch_stages must be 1 or 2")
        if self.prefetch_stages == 2 and not traits.supports_two_stage:
            raise ValueError("two-stage BlockMFMA prefetch requires gfx950")
        if self.prefetch_stages == 2 and m * self.columns_per_wave > 12:
            raise ValueError("two-stage prefetch exceeds the accumulator budget")
        if self.workgroups_per_cu not in (1, 2, 4):
            raise ValueError("workgroups_per_cu must be 1, 2, or 4")
        if self.persistent_n:
            if self.activation_source != ActivationSource.FULL_LDS:
                raise ValueError(
                    "N persistence requires full-A LDS staging so A is loaded "
                    "once and safely reused across N turns"
                )
        elif self.workgroups_per_cu != 1:
            raise ValueError(
                "workgroups_per_cu only applies to N-persistent BlockMFMA"
            )
        if self.waves_per_eu not in (0, 1, 2, 4):
            raise ValueError("waves_per_eu must be 0, 1, 2, or 4")
        validate_cache_policy(self.b_cache_modifier)
        validate_block_mfma_i32_addressing(m, n, k, self)
        if self.output_rounding == OutputRounding.STOCHASTIC:
            if not traits.supports_stochastic:
                raise ValueError("stochastic BF16 conversion requires gfx950")
        if self.activation_source == ActivationSource.FULL_LDS:
            required = block_mfma_lds_bytes(m, k)
            lds_limit = addressable_lds_bytes_for_gfx(arch)
            if required > lds_limit:
                raise ValueError(
                    f"full A LDS requires {required} bytes, exceeding the "
                    f"{lds_limit}-byte {arch} limit"
                )
        # Keep the exact five-row/c4 body below the validated accumulator budget.
        if m * self.columns_per_wave > 20:
            raise ValueError("BlockMFMA accumulator tile exceeds the register budget")
        estimated_vgprs = block_mfma_estimated_live_vgprs(m, self)
        if estimated_vgprs > traits.max_estimated_live_vgprs:
            raise ValueError(
                "BlockMFMA live prefetch state exceeds the conservative "
                f"{arch} register budget: estimated {estimated_vgprs} VGPRs > "
                f"{traits.max_estimated_live_vgprs}"
            )


DecodeConfig: TypeAlias = WaveDecodeConfig | BlockMfmaDecodeConfig


def _validate_problem(m: int, n: int, k: int) -> None:
    if not 1 <= m <= 5:
        raise ValueError("BF16 decode GEMM supports exact M in [1, 5]")
    if n <= 0 or k <= 0:
        raise ValueError("BF16 decode GEMM requires positive N and K")


def _require_signed_i32(family: str, name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{family} {name}={value} must be non-negative")
    if value > SIGNED_INT32_MAX:
        raise ValueError(
            f"{family} {name}={value} exceeds signed-32-bit addressing limit "
            f"{SIGNED_INT32_MAX}"
        )


def _validate_common_i32_addressing(
    family: str,
    m: int,
    n: int,
    k: int,
) -> None:
    """Validate packed BF16 resource sizes and row/column offset products."""
    values = {
        "M": m,
        "N": n,
        "K": k,
        "A row stride bytes": k * BF16_BYTES,
        "B row stride bytes": k * BF16_BYTES,
        "C row stride bytes": n * BF16_BYTES,
        "A element extent": m * k,
        "B element extent": n * k,
        "C element extent": m * n,
        "A byte extent": m * k * BF16_BYTES,
        "B byte extent": n * k * BF16_BYTES,
        "C byte extent": m * n * BF16_BYTES,
        "A max row*K": (m - 1) * k,
        "B max column*K": (n - 1) * k,
        "C max row*N": (m - 1) * n,
        "A max element offset": m * k - 1,
        "B max element offset": n * k - 1,
        "C max element offset": m * n - 1,
    }
    for name, value in values.items():
        _require_signed_i32(family, name, value)


def validate_wave_i32_addressing(
    m: int,
    n: int,
    k: int,
    config: WaveDecodeConfig,
) -> None:
    """Reject shapes whose generated Wave address/grid math can overflow i32."""
    family = "Wave"
    _validate_common_i32_addressing(family, m, n, k)
    k_tile = WAVE_SIZE * config.kvec
    full_tiles = k // k_tile
    tail_start = full_tiles * k_tile
    has_tail = k % k_tile != 0
    rounded_k_boundary = (
        tail_start + k_tile - 1 if has_tail else k - 1
    )
    row_blocks = m // config.m_per_wave
    column_blocks = n // config.n_per_wave
    values = {
        "K vector tile": k_tile,
        "full K tiles": full_tiles,
        "tail start": tail_start,
        "lane*K vector maximum": (WAVE_SIZE - 1) * config.kvec,
        "rounded K vector boundary": rounded_k_boundary,
        "row grid dimension": row_blocks,
        "column grid dimension": column_blocks,
        "max row block index": row_blocks - 1,
        "max column block index": column_blocks - 1,
        "max row base": (row_blocks - 1) * config.m_per_wave,
        "max column base": (column_blocks - 1) * config.n_per_wave,
        "max row index": m - 1,
        "max column index": n - 1,
        "A max vector load offset": m * k - 1,
        "B max vector load offset": n * k - 1,
        "C max store offset": m * n - 1,
    }
    for name, value in values.items():
        _require_signed_i32(family, name, value)


def validate_block_mfma_i32_addressing(
    m: int,
    n: int,
    k: int,
    config: BlockMfmaDecodeConfig,
) -> None:
    """Reject shapes whose generated BlockMFMA address math can overflow i32."""
    family = "BlockMFMA"
    _validate_common_i32_addressing(family, m, n, k)
    tile_columns = config.waves_per_workgroup * config.columns_per_wave
    staged_k = block_mfma_staged_k(k)
    logical_workgroups = (n + tile_columns - 1) // tile_columns
    max_tile_column = logical_workgroups * tile_columns - 1
    values = {
        "staged K": staged_k,
        "staged-A element extent": m * staged_k,
        "logical workgroups": logical_workgroups,
        "max logical tile column": max_tile_column,
        "tile column stride": tile_columns,
    }
    for name, value in values.items():
        _require_signed_i32(family, name, value)


def validate_block_mfma_grid_i32(
    n: int,
    config: BlockMfmaDecodeConfig,
    *,
    num_cus: int,
) -> tuple[int, int, int]:
    """Validate config-dependent persistent grid/turn i32 intermediates."""
    if not isinstance(num_cus, int) or num_cus <= 0:
        raise ValueError(f"num_cus must be a positive integer, got {num_cus!r}")
    tile_columns = config.waves_per_workgroup * config.columns_per_wave
    logical_workgroups = (n + tile_columns - 1) // tile_columns
    grid_cap = num_cus * config.workgroups_per_cu
    grid_workgroups = min(logical_workgroups, grid_cap)
    persistent_turns = (
        logical_workgroups + grid_workgroups - 1
    ) // grid_workgroups
    column_stride = grid_workgroups * tile_columns
    max_turn_column = (
        (persistent_turns - 1) * column_stride
        + (grid_workgroups - 1) * tile_columns
        + tile_columns
        - 1
    )
    for name, value in {
        "num_cus": num_cus,
        "grid-cap workgroups": grid_cap,
        "grid workgroups": grid_workgroups,
        "persistent turns": persistent_turns,
        "persistent tile stride": column_stride,
        "max persistent tile column": max_turn_column,
    }.items():
        _require_signed_i32("BlockMFMA", name, value)
    return grid_workgroups, persistent_turns, column_stride


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def block_mfma_staged_k(k: int) -> int:
    padding = MFMA_K if k % (WAVE_SIZE * MFMA_K) else 0
    return _align_up(k + padding, 8)


def block_mfma_lds_bytes(m: int, k: int) -> int:
    return m * block_mfma_staged_k(k) * BF16_BYTES


def block_mfma_estimated_live_vgprs(
    m: int,
    config: BlockMfmaDecodeConfig,
) -> int:
    """Conservative bound for accumulators plus simultaneously prefetched data."""
    accumulator_vgprs = 4 * m * config.columns_per_wave
    vector_vgprs = config.b_load_width // 2
    live_prefetch_vgprs = (
        config.k_unroll
        * config.prefetch_stages
        * (m + config.columns_per_wave)
        * vector_vgprs
    )
    return accumulator_vgprs + live_prefetch_vgprs + 48


_DEFAULT_CUS_BY_ARCH = {"gfx942": 304, "gfx950": 256}
_NON_TEMPORAL_WEIGHT_BYTES = 8 * 1024 * 1024
_BLOCK_TILE_PRESETS = (
    (4, 1),
    (8, 1),
    (8, 2),
    (12, 1),
    (16, 1),
    (16, 2),
    (16, 4),
)


def _decode_cache_options(n: int, k: int) -> tuple[int, ...]:
    if n * k * BF16_BYTES >= _NON_TEMPORAL_WEIGHT_BYTES:
        return CACHE_POLICY_DEFAULT, CACHE_POLICY_NON_TEMPORAL
    return (CACHE_POLICY_DEFAULT,)


def _wave_prefetch_options(k: int, arch: str) -> tuple[int, ...]:
    options = [0]
    if k >= WAVE_SIZE * 4:
        options.append(1)
    if arch == "gfx950" and k >= WAVE_SIZE * 8:
        options.append(2)
    return tuple(options)


def _block_pipeline_presets(arch: str) -> tuple[tuple[int, int, int], ...]:
    presets = [(4, 1, 1), (8, 1, 1), (8, 2, 1)]
    if arch == "gfx950":
        presets.append((8, 2, 2))
    return tuple(presets)


def _persistent_grid_options(
    n: int,
    config: BlockMfmaDecodeConfig,
    *,
    num_cus: int,
) -> tuple[int, ...]:
    """Keep grid caps whose final turn wastes at most one quarter of the grid."""
    tile_columns = config.waves_per_workgroup * config.columns_per_wave
    logical_workgroups = (n + tile_columns - 1) // tile_columns
    if logical_workgroups < 2:
        return ()
    options = []
    seen_grids = set()
    for workgroups_per_cu in (1, 2, 4):
        grid = min(logical_workgroups, num_cus * workgroups_per_cu)
        if grid in seen_grids:
            continue
        turns = (logical_workgroups + grid - 1) // grid
        scheduled = grid * turns
        if scheduled - logical_workgroups <= max(1, grid // 4):
            options.append(workgroups_per_cu)
            seen_grids.add(grid)
    return tuple(options)


def iter_gemm_decode_configs(
    m: int,
    n: int,
    k: int,
    arch: str,
    *,
    num_cus: int | None = None,
) -> Iterator[DecodeConfig]:
    """Enumerate the bounded production tuning catalog.

    The supported catalog includes every exact-M Wave tile, direct BlockMFMA
    tiles from the measured 4/8/12/16-wave families, and N-persistent
    BlockMFMA whenever full-A LDS is legal. It intentionally does not form the
    raw Cartesian product of every axis: prefetch depth is K-aware, the
    non-temporal cache variant is reserved for weights of at least 8 MiB, and
    persistent grid caps with more than 25% final-turn waste are omitted.
    Stable names remain capable of representing any individually legal config.
    """
    _validate_problem(m, n, k)
    get_decode_arch_traits(arch)
    if num_cus is None:
        num_cus = _DEFAULT_CUS_BY_ARCH[arch]
    if not isinstance(num_cus, int) or num_cus <= 0:
        raise ValueError(f"num_cus must be a positive integer, got {num_cus!r}")

    seen: set[DecodeConfig] = set()

    def emit(config: DecodeConfig) -> Iterator[DecodeConfig]:
        try:
            config.validate(m=m, n=n, k=k, arch=arch)
        except ValueError:
            return
        if config not in seen:
            seen.add(config)
            yield config

    contractions = (
        (ContractionMode.SCALAR_F32, ContractionMode.PACKED_F32)
        if arch == "gfx942"
        else (ContractionMode.DOT2_BF16,)
    )
    exact_m_divisors = tuple(mp for mp in range(1, m + 1) if m % mp == 0)
    for values in product(
        exact_m_divisors,
        (2, 4, 8),
        (1, 2, 4),
        _wave_prefetch_options(k, arch),
        (2, 4),
        _decode_cache_options(n, k),
        tuple(ReductionMode),
        contractions,
    ):
        mp, kvec, np, prefetch, wpe, cache, reduction, contraction = values
        yield from emit(
            WaveDecodeConfig(
                m_per_wave=mp,
                n_per_wave=np,
                kvec=kvec,
                prefetch_depth=prefetch,
                waves_per_eu=wpe,
                b_cache_modifier=cache,
                reduction=reduction,
                contraction=contraction,
            )
        )

    pipelines = _block_pipeline_presets(arch)
    for values in product(
        _BLOCK_TILE_PRESETS,
        tuple(ActivationSource),
        pipelines,
        (0, 2),
        _decode_cache_options(n, k),
    ):
        (waves, columns), a_source, (b_width, unroll, prefetch), wpe, cache = values
        yield from emit(
            BlockMfmaDecodeConfig(
                waves_per_workgroup=waves,
                columns_per_wave=columns,
                activation_source=a_source,
                b_load_width=b_width,
                k_unroll=unroll,
                prefetch_stages=prefetch,
                waves_per_eu=wpe,
                b_cache_modifier=cache,
            )
        )

    # Persistence is useful beyond one historical cell, but only when full-A
    # staging fits. Keep two representative load pipelines and N-aware grid
    # caps; direct variants above remain available for every legal shape.
    persistent_pipelines = (pipelines[0], pipelines[-1])
    for (waves, columns), (b_width, unroll, prefetch), wpe in product(
        _BLOCK_TILE_PRESETS,
        persistent_pipelines,
        (0, 2),
    ):
        base = BlockMfmaDecodeConfig(
            waves_per_workgroup=waves,
            columns_per_wave=columns,
            activation_source=ActivationSource.FULL_LDS,
            b_load_width=b_width,
            k_unroll=unroll,
            prefetch_stages=prefetch,
            persistent_n=True,
            waves_per_eu=wpe,
        )
        try:
            base.validate(m=m, n=n, k=k, arch=arch)
        except ValueError:
            continue
        for grid in _persistent_grid_options(n, base, num_cus=num_cus):
            yield from emit(
                BlockMfmaDecodeConfig(
                    waves_per_workgroup=waves,
                    columns_per_wave=columns,
                    activation_source=ActivationSource.FULL_LDS,
                    b_load_width=b_width,
                    k_unroll=unroll,
                    prefetch_stages=prefetch,
                    persistent_n=True,
                    workgroups_per_cu=grid,
                    waves_per_eu=wpe,
                )
            )


_NAME_RE = re.compile(
    r"^flydsl_decode_a(?P<arch>gfx\d+)_m(?P<m>\d+)_n(?P<n>\d+)_k(?P<k>\d+)_"
    r"(?P<body>.+)$"
)


def gemm_decode_kernel_name(
    arch: str,
    m: int,
    n: int,
    k: int,
    config: DecodeConfig,
    *,
    has_bias: bool = False,
) -> str:
    config.validate(m=m, n=n, k=k, arch=arch)
    prefix = f"flydsl_decode_a{arch}_m{m}_n{n}_k{k}_"
    if isinstance(config, WaveDecodeConfig):
        name = prefix + (
            f"pwave_mp{config.m_per_wave}_np{config.n_per_wave}_kv{config.kvec}_"
            f"pf{config.prefetch_depth}_we{config.waves_per_eu}_"
            f"cp{config.b_cache_modifier}_rd{config.reduction.value}_"
            f"ct{config.contraction.value}_or{config.output_rounding.value}"
        )
    else:
        name = prefix + (
            f"pblock_ww{config.waves_per_workgroup}_cw{config.columns_per_wave}_"
            f"as{config.activation_source.value}_bl{config.b_load_width}_"
            f"ku{config.k_unroll}_pf{config.prefetch_stages}_"
            f"pn{int(config.persistent_n)}_g{config.workgroups_per_cu}_"
            f"we{config.waves_per_eu}_cp{config.b_cache_modifier}_"
            f"or{config.output_rounding.value}"
        )
    return name + ("_BIAS" if has_bias else "")


def parse_gemm_decode_kernel_name(
    name: str,
) -> tuple[str, int, int, int, DecodeConfig, bool]:
    match = _NAME_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid FlyDSL decode kernel name: {name!r}")
    arch = match.group("arch")
    m, n, k = (int(match.group(field)) for field in ("m", "n", "k"))
    body = match.group("body")
    has_bias = body.endswith("_BIAS")
    if has_bias:
        body = body.removesuffix("_BIAS")
    if body.startswith("pwave_"):
        wave = re.fullmatch(
            r"pwave_mp(\d+)_np(\d+)_kv(\d+)_pf(\d+)_we(\d+)_cp(\d+)_"
            r"rd([a-z]+)_ct([a-z0-9]+)_or([a-z]+)",
            body,
        )
        if wave is None:
            raise ValueError(f"invalid wave decode kernel name: {name!r}")
        config: DecodeConfig = WaveDecodeConfig(
            m_per_wave=int(wave.group(1)),
            n_per_wave=int(wave.group(2)),
            kvec=int(wave.group(3)),
            prefetch_depth=int(wave.group(4)),
            waves_per_eu=int(wave.group(5)),
            b_cache_modifier=int(wave.group(6)),
            reduction=ReductionMode(wave.group(7)),
            contraction=ContractionMode(wave.group(8)),
            output_rounding=OutputRounding(wave.group(9)),
        )
    else:
        block = re.fullmatch(
            r"pblock_ww(\d+)_cw(\d+)_as([a-z]+)_bl(\d+)_ku(\d+)_"
            r"pf(\d+)_pn([01])_g(\d+)_we(\d+)_cp(\d+)_or([a-z]+)",
            body,
        )
        if block is None:
            raise ValueError(f"invalid BlockMFMA decode kernel name: {name!r}")
        config = BlockMfmaDecodeConfig(
            waves_per_workgroup=int(block.group(1)),
            columns_per_wave=int(block.group(2)),
            activation_source=ActivationSource(block.group(3)),
            b_load_width=int(block.group(4)),
            k_unroll=int(block.group(5)),
            prefetch_stages=int(block.group(6)),
            persistent_n=bool(int(block.group(7))),
            workgroups_per_cu=int(block.group(8)),
            waves_per_eu=int(block.group(9)),
            b_cache_modifier=int(block.group(10)),
            output_rounding=OutputRounding(block.group(11)),
        )
    config.validate(m=m, n=n, k=k, arch=arch)
    canonical = gemm_decode_kernel_name(
        arch,
        m,
        n,
        k,
        config,
        has_bias=has_bias,
    )
    if canonical != name:
        raise ValueError(
            f"non-canonical FlyDSL decode kernel name: {name!r}; "
            f"expected {canonical!r}"
        )
    return arch, m, n, k, config, has_bias


# Layout, address, and data-movement helpers.
def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def make_buffer_matrix(
    tensor,
    rows: int,
    columns: int,
):
    """Give a packed BF16 global operand an exact two-dimensional view."""
    buffer = fx.rocdl.make_buffer_tensor(
        tensor,
        max_size=False,
        num_records_bytes=rows * columns * BF16_BYTES,
    )
    return fx.make_view(
        fx.get_iter(buffer),
        fx.make_layout((rows, columns), (columns, 1)),
    )


def make_buffer_vector(tensor, length: int):
    """Give a packed BF16 global operand an exact one-dimensional view."""
    buffer = fx.rocdl.make_buffer_tensor(
        tensor,
        max_size=False,
        num_records_bytes=length * BF16_BYTES,
    )
    return fx.make_view(
        fx.get_iter(buffer),
        fx.make_layout(length, 1),
    )


def make_vector_view(
    tensor,
    row,
    column,
    row_stride: int,
    width: int,
):
    """Create a contiguous register-width view at a matrix coordinate."""
    # FlyDSL 0.3.0 crd2idx normalizes dynamic coordinates. These coordinates
    # are already proven in bounds, so keep the physical offset explicit at
    # this address-generation boundary.
    element = fx.Int32(row) * fx.Int32(row_stride) + fx.Int32(column)
    pointer = fx.add_offset(fx.get_iter(tensor), fx.make_int_tuple(element))
    return fx.make_view(pointer, fx.make_layout(width, 1))


def load_vector(
    tensor,
    row,
    column,
    row_stride: int,
    width: int,
    cache_modifier: int = 0,
):
    """Load one logical row vector while preserving explicit cache controls."""
    if cache_modifier:
        resource = fx.rocdl.get_buffer_rsrc(fx.get_iter(tensor))
        element = fx.Int32(row) * fx.Int32(row_stride) + fx.Int32(column)
        return buffer_ops.buffer_load(
            resource,
            element,
            vec_width=width,
            dtype=tensor.dtype,
            cache_modifier=cache_modifier,
        )
    return make_vector_view(tensor, row, column, row_stride, width).load()


def load_scalar(
    tensor,
    row,
    column,
    row_stride: int,
    cache_modifier: int = 0,
):
    """Load one logical matrix element with optional cache controls."""
    element = fx.Int32(row) * fx.Int32(row_stride) + fx.Int32(column)
    if cache_modifier:
        resource = fx.rocdl.get_buffer_rsrc(fx.get_iter(tensor))
        return buffer_ops.buffer_load(
            resource,
            element,
            vec_width=1,
            dtype=tensor.dtype,
            cache_modifier=cache_modifier,
        )
    pointer = fx.add_offset(fx.get_iter(tensor), fx.make_int_tuple(element))
    return fx.make_view(pointer, fx.make_layout(1, 1))[0]


def wave_lane_coordinates(thread_id, waves: int):
    """Map a workgroup thread id to ``(wave, lane)``."""
    return fx.idx2crd(
        thread_id,
        fx.make_layout((waves, WAVE_SIZE), (WAVE_SIZE, 1)),
    ).unpack()


def padded_row_coordinates(
    slot,
    rows: int,
    values_per_row: int,
):
    """Map a cooperative slot to a padded physical row and row-local value.

    The physical row mode is power-of-two padded so FlyDSL's coordinate
    normalization lowers to a mask instead of integer remainder. The staging
    predicate limits slots to the logical ``rows`` and leaves padded rows
    unreachable.
    """
    return fx.idx2crd(
        slot,
        fx.make_layout(
            (_next_power_of_two(rows), values_per_row),
            (values_per_row, 1),
        ),
    ).unpack()


def k_element(
    chunk,
    lane,
    value,
    chunks: int,
    width: int,
):
    return fx.Int32(
        fx.get_scalar(
            fx.crd2idx(
                (chunk, lane, value),
                fx.make_layout(
                    (chunks, WAVE_SIZE, width),
                    (WAVE_SIZE * width, width, 1),
                ),
            )
        )
    )


# Compiler-sensitive numeric, MFMA, DPP, and tail helpers.
def const_bf16(value: float = 0.0):
    return arith_dialect.ConstantOp(
        ir.BF16Type.get(),
        ir.FloatAttr.get(ir.BF16Type.get(), value),
    ).result


def const_f32(value: float = 0.0):
    return arith_dialect.ConstantOp(
        ir.F32Type.get(),
        ir.FloatAttr.get(ir.F32Type.get(), value),
    ).result


def add_f32(lhs, rhs):
    return arith_dialect.AddFOp(raw(lhs), raw(rhs)).result


def add_bias_f32(value, bias_tensor, column):
    """Add one BF16 column bias to an FP32 accumulator before conversion."""
    bias_f32 = arith_dialect.ExtFOp(T.f32, raw(bias_tensor[column])).result
    return add_f32(value, bias_f32)


def pack_bf16x2(lo, hi):
    lo_i16 = ArithValue(raw(lo)).bitcast(T.i16)
    hi_i16 = ArithValue(raw(hi)).bitcast(T.i16)
    lo_i32 = ArithValue(lo_i16).extui(T.i32)
    hi_i32 = ArithValue(hi_i16).extui(T.i32)
    return ArithValue(lo_i32) | (ArithValue(hi_i32) << fx.Int32(16))


def unpack_bf16x2_f32(packed):
    packed = ArithValue(raw(packed))
    lo_bits = (packed & fx.Int32(0xFFFF)) << fx.Int32(16)
    hi_bits = packed & fx.Int32(0xFFFF0000)
    return (
        raw(ArithValue(lo_bits).bitcast(T.f32)),
        raw(ArithValue(hi_bits).bitcast(T.f32)),
    )


def prepare_pair(packed, contraction: ContractionMode):
    if contraction == ContractionMode.DOT2_BF16:
        return raw(packed)
    expanded = unpack_bf16x2_f32(packed)
    if contraction == ContractionMode.PACKED_F32:
        return vector.from_elements(T.vec(2, T.f32), list(expanded))
    return expanded


def zero_wave_accumulator(contraction: ContractionMode):
    if contraction == ContractionMode.PACKED_F32:
        return arith.constant_vector(0.0, T.vec(2, T.f32))
    return const_f32()


def contract_pair(accumulator, a_pair, b_pair, contraction: ContractionMode):
    if contraction == ContractionMode.DOT2_BF16:
        return llvm.inline_asm(
            ir.F32Type.get(),
            [raw(accumulator), raw(a_pair), raw(b_pair)],
            "v_dot2_f32_bf16 $0, $2, $3, $1",
            "=v,0,v,v",
            has_side_effects=False,
        )
    if contraction == ContractionMode.PACKED_F32:
        return llvm.inline_asm(
            ir.VectorType.get([2], ir.F32Type.get()),
            [raw(accumulator), raw(a_pair), raw(b_pair)],
            "v_pk_fma_f32 $0, $2, $3, $1",
            "=v,0,v,v",
            has_side_effects=False,
        )
    accumulator = llvm.intr_fma(a_pair[0], b_pair[0], raw(accumulator))
    return llvm.intr_fma(a_pair[1], b_pair[1], accumulator)


def dpp_add_f32(value, control: str):
    return llvm.inline_asm(
        ir.F32Type.get(),
        [raw(value), raw(value), raw(value)],
        f"s_nop 3\n\tv_add_f32 $0, $2, $3 {control} bound_ctrl:0",
        "=v,0,v,v",
        has_side_effects=False,
    )


def wavefront_reduce_sum_f32(value):
    for shift in (8, 4, 2, 1):
        value = dpp_add_f32(value, f"row_shr:{shift}")
    value = dpp_add_f32(value, "row_bcast:15")
    return dpp_add_f32(value, "row_bcast:31")


def bpermute_reduce_sum_f32(value, lane):
    value = llvm.inline_asm(
        ir.F32Type.get(),
        [raw(value)],
        "s_nop 3\n\tv_mov_b32 $0, $1",
        "=v,v",
        has_side_effects=False,
    )
    for stage in range_constexpr(6):
        partner = lane ^ fx.Int32(1 << stage)
        value_i32 = ArithValue(raw(value)).bitcast(T.i32)
        peer_i32 = fx.rocdl.ds_bpermute(
            T.i32,
            partner * fx.Int32(4),
            value_i32,
        )
        value = add_f32(value, ArithValue(peer_i32).bitcast(T.f32))
    return value


def reduce_wave_accumulator(accumulator, lane, contraction, reduction):
    use_dpp = reduction == ReductionMode.DPP
    if contraction != ContractionMode.PACKED_F32:
        return (
            wavefront_reduce_sum_f32(accumulator)
            if use_dpp
            else bpermute_reduce_sum_f32(accumulator, lane)
        )
    lo = vector.extract(
        accumulator,
        static_position=[0],
        dynamic_position=[],
    )
    hi = vector.extract(
        accumulator,
        static_position=[1],
        dynamic_position=[],
    )
    if use_dpp:
        lo = wavefront_reduce_sum_f32(lo)
        hi = wavefront_reduce_sum_f32(hi)
    else:
        lo = bpermute_reduce_sum_f32(lo, lane)
        hi = bpermute_reduce_sum_f32(hi, lane)
    return add_f32(lo, hi)


def convert_bf16(value, element, rounding: OutputRounding):
    if rounding == OutputRounding.RNE:
        return arith_dialect.TruncFOp(T.bf16, raw(value)).result
    seed = (
        ArithValue(raw(element)) * fx.Int32(0x45D9F3B)
    ) ^ (ArithValue(raw(element)) << fx.Int32(16)) ^ fx.Int32(0x27D4EB2D)
    converted = llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [raw(value), raw(seed)],
        "v_cvt_sr_bf16_f32 $0, $1, $2",
        "=v,v,v",
        has_side_effects=False,
    )
    return ArithValue(converted).trunci(T.i16).bitcast(T.bf16)


def store_bf16(
    value,
    tensor,
    row,
    column,
    row_stride: int,
    rounding: OutputRounding,
) -> None:
    element = fx.Int32(row) * fx.Int32(row_stride) + fx.Int32(column)
    output = convert_bf16(value, element, rounding)
    tensor[row, column] = output


def mfma_4x4x4_bf16(a_fragment, b_fragment, accumulator):
    """Use the shared native atom; FlyDSL has no matching high-level MMA atom."""
    a_i16 = vector.bitcast(T.vec(4, T.i16), a_fragment)
    b_i16 = vector.bitcast(T.vec(4, T.i16), b_fragment)
    return fx.rocdl.mfma_f32_4x4x4bf16_1k_(
        T.vec(4, T.f32),
        raw(a_i16),
        raw(b_i16),
        raw(accumulator),
        0,
        0,
        0,
    )


def bf16x4_slice(fragment, fragment_index: int):
    return vector.extract_strided_slice(
        T.vec(MFMA_K, T.bf16),
        raw(fragment),
        [fragment_index * MFMA_K],
        [MFMA_K],
        [1],
    )


def dpp_move_f32(value, control: int):
    return fx.rocdl.update_dpp(
        T.f32,
        raw(value),
        raw(value),
        control,
        0xF,
        0xF,
        True,
    )


def reduce_mfma_scalar(accumulator):
    components = [
        vector.extract(accumulator, static_position=[i], dynamic_position=[])
        for i in range_constexpr(4)
    ]
    result = components[0]
    result = add_f32(result, dpp_move_f32(components[1], 0x101))
    result = add_f32(result, dpp_move_f32(components[2], 0x102))
    result = add_f32(result, dpp_move_f32(components[3], 0x103))
    result = add_f32(result, dpp_move_f32(result, 0x104))
    result = add_f32(result, dpp_move_f32(result, 0x108))
    result = dpp_move_f32(result, 0x11F)
    result = add_f32(result, dpp_move_f32(result, 0x142))
    return add_f32(result, dpp_move_f32(result, 0x143))


def masked_bf16_vector(
    tensor,
    row,
    column_base,
    width: int,
    row_size: int,
    cache_modifier: int = 0,
):
    """Load a compile-time BF16 vector with safe N/K tail masking."""
    zero = const_bf16()
    values = []
    for offset in range_constexpr(width):
        column = column_base + fx.Int32(offset)
        valid = column < fx.Int32(row_size)
        safe_column = ArithValue(raw(valid)).select(column, fx.Int32(0))
        loaded = load_scalar(
            tensor,
            row,
            safe_column,
            row_size,
            cache_modifier,
        )
        values.append(ArithValue(raw(valid)).select(loaded, zero))
    return vector.from_elements(T.vec(width, T.bf16), values)
