# SPDX-License-Identifier: MIT

"""Configuration, architecture traits, and registry for BF16 decode GEMM."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
import re
from typing import Iterator, TypeAlias

from .gemm_decode_common import (
    CACHE_POLICY_DEFAULT,
    CACHE_POLICY_NON_TEMPORAL,
    validate_cache_policy,
)

WAVE_SIZE = 64
MFMA_K = 4
BF16_BYTES = 2
SIGNED_INT32_MAX = (1 << 31) - 1


class DecodePolicy(str, Enum):
    WAVE = "wave"
    BLOCK_MFMA = "block"


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

    @property
    def policy(self) -> DecodePolicy:
        return DecodePolicy.WAVE

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

    @property
    def policy(self) -> DecodePolicy:
        return DecodePolicy.BLOCK_MFMA

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
            if required > traits.max_lds_bytes:
                raise ValueError(
                    f"full A LDS requires {required} bytes, exceeding the "
                    f"{traits.max_lds_bytes}-byte {arch} limit"
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
    return _align_up(m * block_mfma_staged_k(k) * BF16_BYTES, 128)


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


def conservative_wave_config(m: int, n: int, k: int, arch: str) -> WaveDecodeConfig:
    """Return one legality-only fallback without a shape-performance ladder."""
    get_decode_arch_traits(arch)
    kvec = 2
    reduction = ReductionMode.DPP if k % kvec == 0 else ReductionMode.BPERMUTE
    contraction = (
        ContractionMode.DOT2_BF16
        if arch == "gfx950"
        else ContractionMode.SCALAR_F32
    )
    config = WaveDecodeConfig(
        m_per_wave=m,
        n_per_wave=1,
        kvec=kvec,
        prefetch_depth=0,
        waves_per_eu=4,
        reduction=reduction,
        contraction=contraction,
    )
    config.validate(m=m, n=n, k=k, arch=arch)
    return config


def iter_gemm_decode_configs(
    m: int,
    n: int,
    k: int,
    arch: str,
) -> Iterator[DecodeConfig]:
    """Enumerate, deduplicate, and legality-prune all production-equivalent axes."""
    _validate_problem(m, n, k)
    get_decode_arch_traits(arch)
    seen: set[DecodeConfig] = set()
    contractions = (
        (ContractionMode.SCALAR_F32, ContractionMode.PACKED_F32)
        if arch == "gfx942"
        else (ContractionMode.DOT2_BF16,)
    )
    wave_prefetch = (0, 1) if arch == "gfx942" else (0, 1, 2)
    reductions = (ReductionMode.DPP, ReductionMode.BPERMUTE)
    exact_m_divisors = tuple(mp for mp in range(1, m + 1) if m % mp == 0)
    for values in product(
        exact_m_divisors,
        (2, 4, 8),
        (1, 2, 4),
        wave_prefetch,
        (1, 2, 4),
        (CACHE_POLICY_DEFAULT, CACHE_POLICY_NON_TEMPORAL),
        reductions,
        contractions,
    ):
        mp, kvec, np, prefetch, wpe, cache, reduction, contraction = values
        config = WaveDecodeConfig(
            m_per_wave=mp,
            n_per_wave=np,
            kvec=kvec,
            prefetch_depth=prefetch,
            waves_per_eu=wpe,
            b_cache_modifier=cache,
            reduction=reduction,
            contraction=contraction,
        )
        try:
            config.validate(m=m, n=n, k=k, arch=arch)
        except ValueError:
            continue
        if config not in seen:
            seen.add(config)
            yield config

    b_widths = (4, 8)
    stages = (1, 2) if arch == "gfx950" else (1,)
    for values in product(
        (4, 8, 12, 16),
        (1, 2, 4),
        tuple(ActivationSource),
        b_widths,
        (1, 2),
        stages,
        (0, 1, 2, 4),
        (CACHE_POLICY_DEFAULT, CACHE_POLICY_NON_TEMPORAL),
    ):
        waves, columns, a_source, b_width, unroll, prefetch, wpe, cache = values
        config = BlockMfmaDecodeConfig(
            waves_per_workgroup=waves,
            columns_per_wave=columns,
            activation_source=a_source,
            b_load_width=b_width,
            k_unroll=unroll,
            prefetch_stages=prefetch,
            waves_per_eu=wpe,
            b_cache_modifier=cache,
        )
        try:
            config.validate(m=m, n=n, k=k, arch=arch)
        except ValueError:
            continue
        if config not in seen:
            seen.add(config)
            yield config

    # Keep persistence targeted to the measured parity gaps. Names can still
    # request any legal shape/config explicitly without multiplying the broad
    # tuner catalog for unrelated cells.
    if m in (3, 4) and (n, k) == (2304, 1536):
        for values in product(
            (4, 8, 16),
            (1, 2),
            b_widths,
            (1, 2),
            stages,
            (1, 2),
            (0, 2, 4),
        ):
            waves, columns, b_width, unroll, prefetch, grid, wpe = values
            config = BlockMfmaDecodeConfig(
                waves_per_workgroup=waves,
                columns_per_wave=columns,
                activation_source=ActivationSource.FULL_LDS,
                b_load_width=b_width,
                k_unroll=unroll,
                prefetch_stages=prefetch,
                persistent_n=True,
                workgroups_per_cu=grid,
                waves_per_eu=wpe,
                b_cache_modifier=CACHE_POLICY_DEFAULT,
            )
            try:
                config.validate(m=m, n=n, k=k, arch=arch)
            except ValueError:
                continue
            if config not in seen:
                seen.add(config)
                yield config


_NAME_RE = re.compile(
    r"^flydsl_decode_v1_a(?P<arch>gfx\d+)_m(?P<m>\d+)_n(?P<n>\d+)_k(?P<k>\d+)_"
    r"(?P<body>.+)$"
)


def gemm_decode_kernel_name(
    arch: str,
    m: int,
    n: int,
    k: int,
    config: DecodeConfig,
) -> str:
    config.validate(m=m, n=n, k=k, arch=arch)
    prefix = f"flydsl_decode_v1_a{arch}_m{m}_n{n}_k{k}_"
    if isinstance(config, WaveDecodeConfig):
        return prefix + (
            f"pwave_mp{config.m_per_wave}_np{config.n_per_wave}_kv{config.kvec}_"
            f"pf{config.prefetch_depth}_we{config.waves_per_eu}_"
            f"cp{config.b_cache_modifier}_rd{config.reduction.value}_"
            f"ct{config.contraction.value}_or{config.output_rounding.value}"
        )
    return prefix + (
        f"pblock2_ww{config.waves_per_workgroup}_cw{config.columns_per_wave}_"
        f"as{config.activation_source.value}_bl{config.b_load_width}_"
        f"ku{config.k_unroll}_pf{config.prefetch_stages}_"
        f"pn{int(config.persistent_n)}_g{config.workgroups_per_cu}_"
        f"we{config.waves_per_eu}_cp{config.b_cache_modifier}_"
        f"or{config.output_rounding.value}"
    )


def parse_gemm_decode_kernel_name(
    name: str,
) -> tuple[str, int, int, int, DecodeConfig]:
    match = _NAME_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid FlyDSL decode kernel name: {name!r}")
    arch = match.group("arch")
    m, n, k = (int(match.group(field)) for field in ("m", "n", "k"))
    body = match.group("body")
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
            r"pblock2_ww(\d+)_cw(\d+)_as([a-z]+)_bl(\d+)_ku(\d+)_"
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
    return arch, m, n, k, config
