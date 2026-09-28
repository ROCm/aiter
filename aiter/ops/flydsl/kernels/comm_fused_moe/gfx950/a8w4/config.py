# SPDX-License-Identifier: Apache-2.0
"""Configuration and workspace layouts for gfx950 A8W4 comm-fused MoE."""

from dataclasses import dataclass

BLOCK = 256
SLOTS = 2
PRODUCER_COUNTER_STRIDE = 64
DIRECT_BLOCK = 512
DIRECT_GRID_CAP = 80

SUPPORTED_TP_SIZES = (2, 4, 8)


@dataclass(frozen=True, slots=True)
class Shape:
    """Model dimensions specialized into one generated kernel."""

    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    tp_size: int = 8
    add_shared: bool = True

    def __post_init__(self):
        for name, value in (
            ("model_dim", self.model_dim),
            ("inter_dim", self.inter_dim),
            ("experts", self.experts),
            ("topk", self.topk),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.model_dim % 32:
            raise ValueError(
                "model_dim must be divisible by the 32-column MXFP8 scale "
                f"group, got {self.model_dim}"
            )
        if self.inter_dim % 128:
            raise ValueError(
                "inter_dim must be divisible by the 128-element shuffled "
                f"A8W4 K block, got {self.inter_dim}"
            )
        if self.tp_size not in SUPPORTED_TP_SIZES:
            raise ValueError(
                "gfx950 A8W4 GEMM2/TP currently requires TP in "
                f"{SUPPORTED_TP_SIZES}, "
                f"got {self.tp_size}"
            )
        if not isinstance(self.add_shared, bool):
            raise TypeError(f"add_shared must be bool, got {self.add_shared!r}")

    @property
    def tag(self) -> str:
        tag = (
            f"h{self.model_dim}_i{self.inter_dim}_e{self.experts}"
            f"_k{self.topk}_tp{self.tp_size}"
        )
        return tag if self.add_shared else f"{tag}_no_shared"


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _mxfp8_scale_bytes(payload_bytes: int, vector_width: int) -> int:
    if vector_width == 8:
        return payload_bytes // 32
    return payload_bytes * 4 // vector_width


@dataclass(frozen=True)
class MegakernelConfig:
    """Configuration for the single-launch GEMM2 + TP collective kernel."""

    shape: Shape
    m: int
    tile_m: int = 16
    tile_n: int = 256
    tile_k: int = 128
    sort_block_m: int = 32
    compute_groups: int = 96
    block_threads: int = BLOCK
    vector_width: int = 16
    waves_per_eu: int = 0
    b_cache_modifier: int = 0
    local_load_cache_modifier: int = 1
    remote_load_cache_modifier: int = 1
    gather_load_cache_modifier: int = -1
    remote_store_cache_modifier: int = 0
    n_tile_cohort: int = 0
    collective: str = "direct"
    service_groups: int = 1
    service_tile_group: int = 1
    flat_producer_grid: bool = False
    sorted_input: bool = False
    wide_partial_scales: bool = False
    coherent_direct_handoff: bool = False

    def __post_init__(self):
        if self.m <= 0:
            raise ValueError(f"m must be positive, got {self.m}")
        if self.vector_width not in (8, 16):
            raise ValueError(
                "production GEMM2 TP megakernel requires vector_width 8 or 16"
            )
        if self.sort_block_m <= 0:
            raise ValueError(f"sort_block_m must be positive, got {self.sort_block_m}")
        if self.tile_m <= 0 or self.tile_n <= 0 or self.tile_k <= 0:
            raise ValueError(
                "tile sizes must be positive, got "
                f"{(self.tile_m, self.tile_n, self.tile_k)}"
            )
        if self.tile_m not in (16, 32, 64, 128):
            raise ValueError(f"unsupported tile_m={self.tile_m}")
        if self.tile_n not in (128, 256, 512):
            raise ValueError(f"unsupported tile_n={self.tile_n}")
        if self.tile_k not in (128, 256):
            raise ValueError(f"unsupported tile_k={self.tile_k}")
        if self.shape.model_dim % self.tile_n:
            raise ValueError(
                f"model_dim={self.shape.model_dim} must be divisible by "
                f"tile_n={self.tile_n}"
            )
        if self.shape.inter_dim % self.tile_k:
            raise ValueError(
                f"inter_dim={self.shape.inter_dim} must be divisible by "
                f"tile_k={self.tile_k}"
            )
        if self.sort_block_m % self.tile_m:
            raise ValueError(
                "sort_block_m must be divisible by tile_m, got "
                f"sort_block_m={self.sort_block_m}, tile_m={self.tile_m}"
            )
        if self.tile_n % self.vector_width:
            raise ValueError(
                "tile_n must be divisible by vector_width, got "
                f"tile_n={self.tile_n}, vector_width={self.vector_width}"
            )
        if self.b_cache_modifier not in (0, 2):
            raise ValueError(
                "MXMoE GEMM2 producer requires b_cache_modifier 0 or 2, got "
                f"{self.b_cache_modifier}"
            )
        if not 0 <= self.waves_per_eu <= 10:
            raise ValueError(
                f"waves_per_eu must be in [0, 10], got {self.waves_per_eu}"
            )
        if self.compute_groups <= 0:
            raise ValueError(
                f"compute_groups must be positive, got {self.compute_groups}"
            )
        if self.block_threads != BLOCK:
            raise ValueError(
                f"MXMoE GEMM2 producer requires block_threads={BLOCK}, "
                f"got {self.block_threads}"
            )
        num_waves = self.block_threads // 64
        if self.tile_n % (num_waves * 16):
            raise ValueError(
                "tile_n must provide an integral number of 16-column MFMA "
                "tiles per wave, got "
                f"tile_n={self.tile_n}, block_threads={self.block_threads}"
            )
        if self.local_load_cache_modifier not in (0, 1, 2, 3):
            raise ValueError("local_load_cache_modifier must be in [0, 3]")
        if self.remote_load_cache_modifier not in (0, 1, 2, 3):
            raise ValueError("remote_load_cache_modifier must be in [0, 3]")
        if self.gather_load_cache_modifier not in (-1, 0, 1, 2, 3):
            raise ValueError("gather_load_cache_modifier must be -1 or in [0, 3]")
        if self.remote_store_cache_modifier not in (0, 1, 2, 3):
            raise ValueError("remote_store_cache_modifier must be in [0, 3]")
        if self.n_tile_cohort < 0:
            raise ValueError(
                f"n_tile_cohort must be non-negative, got {self.n_tile_cohort}"
            )
        if self.n_tile_cohort and self.n_tiles % self.n_tile_cohort:
            raise ValueError(
                "n_tile_cohort must divide n_tiles, got "
                f"n_tile_cohort={self.n_tile_cohort}, n_tiles={self.n_tiles}"
            )
        if self.collective not in (
            "direct",
            "rsag",
            "rs_broadcast",
        ):
            raise ValueError(
                "collective must be 'direct', 'rsag', or 'rs_broadcast', got "
                f"{self.collective!r}"
            )
        if not 1 <= self.service_groups <= self.compute_groups:
            raise ValueError(
                "service_groups must be in [1, compute_groups], got "
                f"service_groups={self.service_groups}, "
                f"compute_groups={self.compute_groups}"
            )
        if self.collective != "rsag" and self.service_groups != 1:
            raise ValueError(
                "direct and rs_broadcast collectives require service_groups=1"
            )
        if self.collective == "rsag" and (
            self.service_groups not in (1, 2, 4, 8)
            or self.service_groups > self.shape.tp_size
            or self.shape.tp_size % self.service_groups
        ):
            raise ValueError(
                "rsag service_groups must be a supported divisor of TP, got "
                f"service_groups={self.service_groups}, "
                f"TP={self.shape.tp_size}"
            )
        if self.service_tile_group <= 0 or self.n_tiles % self.service_tile_group:
            raise ValueError(
                "service_tile_group must be a positive divisor of n_tiles, got "
                f"service_tile_group={self.service_tile_group}, "
                f"n_tiles={self.n_tiles}"
            )
        if self.service_tile_group > 1 and (
            self.collective != "rsag" or self.service_groups == 1
        ):
            raise ValueError(
                "grouped service synchronization requires collective='rsag' "
                "and service_groups > 1"
            )
        if self.flat_producer_grid and self.collective == "direct":
            raise ValueError("flat_producer_grid requires a dynamic collective path")
        if self.flat_producer_grid and self.n_tile_cohort:
            raise ValueError(
                "flat_producer_grid and n_tile_cohort are mutually exclusive"
            )
        if self.collective != "rsag" and self.gather_load_cache_modifier != -1:
            raise ValueError("gather_load_cache_modifier only applies to rsag")
        if self.collective == "direct" and self.remote_store_cache_modifier != 0:
            raise ValueError("remote_store_cache_modifier does not apply to direct")
        if self.wide_partial_scales and (
            self.collective != "direct" or self.tile_n != 256
        ):
            raise ValueError(
                "wide_partial_scales requires collective='direct' and tile_n=256"
            )
        if self.coherent_direct_handoff and (
            self.collective != "direct" or not self.single_pass_direct
        ):
            raise ValueError(
                "coherent_direct_handoff requires a single-pass direct collective"
            )
        if self.uses_rsag and self.m % self.shape.tp_size:
            raise ValueError(
                f"m={self.m} must be divisible by TP={self.shape.tp_size} "
                f"for collective={self.collective!r}"
            )
        if (
            self.uses_rsag
            and (self.m * self.tile_n // self.vector_width) % self.shape.tp_size
        ):
            raise ValueError("rsag vector items must divide evenly across TP ranks")

    @property
    def n_tiles(self) -> int:
        return self.shape.model_dim // self.tile_n

    @property
    def uses_rsag(self) -> bool:
        return self.collective in ("rsag", "rs_broadcast")

    @property
    def shared_bf16_partials(self) -> bool:
        return self.collective == "rs_broadcast"

    @property
    def single_pass_direct(self) -> bool:
        return bool(
            self.collective == "direct"
            and self.m * self.tile_n // self.vector_width <= self.block_threads
        )

    @property
    def producer_rows(self) -> int:
        route_rows = self.m * self.shape.topk
        # Sorting pads each non-empty expert independently to sort_block_m.
        max_sort_blocks = (
            route_rows
            if route_rows <= self.shape.experts
            else self.shape.experts
            + (route_rows - self.shape.experts) // self.sort_block_m
        )
        return max_sort_blocks * self.sort_block_m // self.tile_m

    @property
    def payload_bytes(self) -> int:
        return self.m * self.shape.model_dim * 2

    @property
    def output_rows(self) -> int:
        return self.m

    @property
    def partial_bytes(self) -> int:
        return _align_up(self.partial_payload_bytes + self.partial_scale_bytes, 16)

    @property
    def partial_payload_bytes(self) -> int:
        return self.m * self.shape.model_dim

    @property
    def partial_scale_bytes(self) -> int:
        if self.shared_bf16_partials:
            return 0
        if self.wide_partial_scales:
            return self.partial_payload_bytes // 8
        return _mxfp8_scale_bytes(self.partial_payload_bytes, self.vector_width)

    @property
    def reduced_payload_bytes(self) -> int:
        if not self.uses_rsag:
            return 0
        element_bytes = 2 if self.shared_bf16_partials else 1
        return self.m * self.shape.model_dim * element_bytes // self.shape.tp_size

    @property
    def reduced_scale_bytes(self) -> int:
        if not self.uses_rsag or self.shared_bf16_partials:
            return 0
        return _mxfp8_scale_bytes(self.reduced_payload_bytes, self.vector_width)

    @property
    def reduced_shard_bytes(self) -> int:
        return _align_up(self.reduced_payload_bytes + self.reduced_scale_bytes, 16)

    @property
    def reduced_offset(self) -> int:
        return SLOTS * self.partial_bytes

    @property
    def route_offset(self) -> int:
        return self.reduced_offset + SLOTS * self.reduced_shard_bytes

    @property
    def route_bytes(self) -> int:
        return self.m * self.shape.topk * self.shape.model_dim * 2

    @property
    def output_offset(self) -> int:
        return self.route_offset + self.route_bytes

    @property
    def producer_done_offset(self) -> int:
        return self.output_offset + self.payload_bytes

    @property
    def epoch_offset(self) -> int:
        return self.gather_service_done_offset + self.n_tiles * 8

    @property
    def service_done_offset(self) -> int:
        return self.producer_done_offset + self.n_tiles * PRODUCER_COUNTER_STRIDE

    @property
    def reduce_done_offset(self) -> int:
        return self.service_done_offset + self.n_tiles * 8

    @property
    def gather_service_done_offset(self) -> int:
        return self.reduce_done_offset + self.n_tiles * 8

    @property
    def rank_ready_offset(self) -> int:
        return self.epoch_offset + self.n_tiles * 8

    @property
    def flat_base_offset(self) -> int:
        return _align_up(
            self.gather_done_offset
            + (self.n_tiles * self.shape.tp_size * 4 if self.uses_rsag else 0),
            8,
        )

    @property
    def gather_done_offset(self) -> int:
        return self.owner_ready_offset + (
            self.n_tiles * self.shape.tp_size * 4
            if self.uses_rsag
            else self.n_tiles * 4
        )

    @property
    def owner_ready_offset(self) -> int:
        return self.reduced_collective_ready_offset + self.n_tiles * 4

    @property
    def reduced_collective_ready_offset(self) -> int:
        return self.collective_ready_offset + self.n_tiles * 4

    @property
    def collective_ready_offset(self) -> int:
        return self.rank_ready_offset + self.n_tiles * self.shape.tp_size * 4

    @property
    def workspace_bytes(self) -> int:
        return _align_up(self.flat_base_offset + 8, 256)


@dataclass(frozen=True)
class WindowConfig:
    shape: Shape
    m: int
    tile_m: int
    tile_n: int
    tile_k: int
    sort_block_m: int
    window: int
    local_workers: int
    reduce_scatter_grid: int
    all_gather_grid: int
    sorted_input: bool = False
    gather_output: bool = True
    collective_order: str = "reduce_first"
    local_load_cache_modifier: int = 2
    local_rows_per_cta: int = 1
    producer_spatial_partition: int = 0
    producer_spatial_alt_partition: int = 0
    producer_spatial_switch_rows: int = 0
    # Window partial and reduced handoffs independently select per-group
    # dynamic MXFP4 or MXFP8 payloads.
    partial_payload_bits: int = 8
    reduced_payload_bits: int = 8
    reduce_scatter_load_cache_modifier: int = 2
    all_gather_load_cache_modifier: int = 2
    drain_reduce_scatter_grid: int = 0
    drain_all_gather_grid: int = 0

    def __post_init__(self):
        self._validate_geometry()
        self._validate_payload()
        self._validate_collective()

    def _validate_geometry(self) -> None:
        if self.m <= 0:
            raise ValueError(f"m must be positive, got {self.m}")
        for name, value in (
            ("tile_m", self.tile_m),
            ("tile_n", self.tile_n),
            ("tile_k", self.tile_k),
            ("sort_block_m", self.sort_block_m),
            ("window", self.window),
            ("local_workers", self.local_workers),
            ("reduce_scatter_grid", self.reduce_scatter_grid),
            ("all_gather_grid", self.all_gather_grid),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.local_load_cache_modifier not in (0, 1, 2, 3):
            raise ValueError("local_load_cache_modifier must be in [0, 3]")
        if self.local_rows_per_cta not in (1, 2):
            raise ValueError("local_rows_per_cta must be 1 or 2")
        if self.producer_spatial_partition < 0:
            raise ValueError("producer_spatial_partition must be non-negative")
        if self.producer_spatial_partition and (
            self.producer_spatial_partition < 101
            or self.producer_spatial_partition % 100 == 0
        ):
            raise ValueError(
                "producer_spatial_partition must encode positive group and "
                "M01 values"
            )
        if bool(self.producer_spatial_alt_partition) != bool(
            self.producer_spatial_switch_rows
        ):
            raise ValueError(
                "producer_spatial_alt_partition and "
                "producer_spatial_switch_rows must be enabled together"
            )
        if self.producer_spatial_alt_partition:
            if not self.producer_spatial_partition:
                raise ValueError(
                    "adaptive spatial partition requires a primary partition"
                )
            if (
                self.producer_spatial_alt_partition < 101
                or self.producer_spatial_alt_partition % 100 == 0
            ):
                raise ValueError(
                    "producer_spatial_alt_partition must encode positive group "
                    "and M01 values"
                )
            if (
                self.producer_spatial_alt_partition % 100
                != self.producer_spatial_partition % 100
            ):
                raise ValueError("adaptive spatial partitions must use the same M01")
        if self.tile_m not in (16, 32, 64, 128):
            raise ValueError(f"unsupported tile_m={self.tile_m}")
        if self.tile_n not in (128, 256, 512):
            raise ValueError(f"unsupported tile_n={self.tile_n}")
        if self.tile_k not in (128, 256):
            raise ValueError(f"unsupported tile_k={self.tile_k}")
        divisibility = (
            ("m", self.m, "tp_size", self.shape.tp_size),
            ("model_dim", self.shape.model_dim, "window", self.window),
            ("window", self.window, "tile_n", self.tile_n),
            ("window", self.window, "MXFP8 group", 32),
            ("inter_dim", self.shape.inter_dim, "tile_k", self.tile_k),
            ("sort_block_m", self.sort_block_m, "tile_m", self.tile_m),
        )
        for value_name, value, divisor_name, divisor in divisibility:
            if divisor <= 0 or value % divisor:
                raise ValueError(
                    f"{value_name}={value} must be divisible by "
                    f"{divisor_name}={divisor}"
                )
        if self.phase_count < 3:
            raise ValueError(
                "window pipeline requires at least three phases, got "
                f"model_dim={self.shape.model_dim}, window={self.window}"
            )
        if self.local_workers > self.compute_workers:
            raise ValueError(
                "local_workers cannot exceed the compute grid, got "
                f"local_workers={self.local_workers}, "
                f"compute_workers={self.compute_workers}"
            )

    def _validate_payload(self) -> None:
        # Window communication deliberately supports only dynamic
        # MXFP4/MXFP8. Do not reintroduce fixed-scale or FP6 handoffs.
        if self.partial_payload_bits not in (4, 8):
            raise ValueError("partial_payload_bits must be 4 or 8")
        if self.reduced_payload_bits not in (4, 8):
            raise ValueError("reduced_payload_bits must be 4 or 8")
        if self.reduced_payload_bits != 8 and not self.gather_output:
            raise ValueError("compressed reduced payload requires gather_output=True")

    def _validate_collective(self) -> None:
        if self.drain_reduce_scatter_grid < 0:
            raise ValueError(
                "drain_reduce_scatter_grid must be non-negative, got "
                f"{self.drain_reduce_scatter_grid}"
            )
        if self.drain_all_gather_grid < 0:
            raise ValueError(
                "drain_all_gather_grid must be non-negative, got "
                f"{self.drain_all_gather_grid}"
            )
        if self.drain_all_gather_grid and not self.gather_output:
            raise ValueError("drain_all_gather_grid requires gather_output=True")
        if self.collective_order not in (
            "reduce_first",
            "alternating",
            "gather_first",
        ):
            raise ValueError(
                "collective_order must be 'reduce_first', 'alternating', or "
                f"'gather_first', got {self.collective_order!r}"
            )
        if self.collective_order != "reduce_first" and not self.gather_output:
            raise ValueError("non-default collective_order requires gather_output=True")
        if self.reduce_scatter_load_cache_modifier not in (0, 1, 2, 3):
            raise ValueError("reduce_scatter_load_cache_modifier must be in [0, 3]")
        if self.all_gather_load_cache_modifier not in (0, 1, 2, 3):
            raise ValueError("all_gather_load_cache_modifier must be in [0, 3]")
        source_count = self.shape.tp_size - 1
        if self.gather_output and self.all_gather_grid % source_count:
            raise ValueError(
                "all_gather_grid must be a multiple of TP-1, got "
                f"all_gather_grid={self.all_gather_grid}, TP={self.shape.tp_size}"
            )
        if (
            self.gather_output
            and self.drain_all_gather_grid
            and self.drain_all_gather_grid % source_count
        ):
            raise ValueError(
                "drain_all_gather_grid must be a multiple of TP-1, got "
                f"drain_all_gather_grid={self.drain_all_gather_grid}, "
                f"TP={self.shape.tp_size}"
            )
        service_grid = max(
            self.reduce_scatter_grid,
            self.all_gather_grid if self.gather_output else 0,
        )
        if self.compute_workers < 2 * service_grid:
            raise ValueError(
                "paired window services require at least two compute workers "
                "per service worker, got "
                f"compute_workers={self.compute_workers}, service_grid={service_grid}"
            )

    @property
    def shard_rows(self) -> int:
        return self.m // self.shape.tp_size

    @property
    def output_rows(self) -> int:
        return self.m if self.gather_output else self.shard_rows

    @property
    def phase_count(self) -> int:
        return self.shape.model_dim // self.window

    @property
    def tiles_per_window(self) -> int:
        return self.window // self.tile_n

    @property
    def route_row_bytes(self) -> int:
        return self.window + self.window // 8

    @property
    def groups_per_row(self) -> int:
        return self.window // 32

    @property
    def partial_row_bytes(self) -> int:
        return self.window * self.partial_payload_bits // 8

    @property
    def partial_payload_bytes(self) -> int:
        return self.m * self.partial_row_bytes

    @property
    def partial_scale_bytes(self) -> int:
        return self.m * self.groups_per_row

    @property
    def partial_epoch_offset(self) -> int:
        return self.partial_payload_bytes + self.partial_scale_bytes

    @property
    def partial_buffer_bytes(self) -> int:
        return _align_up(self.partial_epoch_offset + 8, 256)

    @property
    def reduced_row_bytes(self) -> int:
        return self.window * self.reduced_payload_bits // 8

    @property
    def reduced_epoch_offset(self) -> int:
        return self.shard_rows * self.reduced_row_bytes

    @property
    def reduced_buffer_bytes(self) -> int:
        return _align_up(self.reduced_epoch_offset + 8, 256)

    @property
    def reduced_scale_bytes(self) -> int:
        return self.shard_rows * self.groups_per_row

    @property
    def compute_workers(self) -> int:
        max_sorted = (
            self.m * self.shape.topk
            + self.shape.experts * self.sort_block_m
            - self.shape.topk
        )
        sort_blocks = (max_sorted + self.sort_block_m - 1) // self.sort_block_m
        producer_rows = sort_blocks * self.sort_block_m // self.tile_m
        return producer_rows * self.tiles_per_window


@dataclass(frozen=True)
class DirectConfig:
    """Ordinary Stage2 followed by direct collective postprocessing."""

    shape: Shape
    m: int
    sort_block_m: int
    grid_cap: int = DIRECT_GRID_CAP
    vector_width: int = 8

    def __post_init__(self):
        if not self.shape.add_shared:
            raise ValueError("direct collective requires add_shared=True")
        for name, value in (
            ("m", self.m),
            ("sort_block_m", self.sort_block_m),
            ("grid_cap", self.grid_cap),
            ("vector_width", self.vector_width),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.vector_width not in (4, 8):
            raise ValueError(
                "direct vector_width must be 4 or 8, got " f"{self.vector_width}"
            )
        if self.m % self.shape.tp_size:
            raise ValueError(f"m={self.m} must be divisible by TP={self.shape.tp_size}")
        if self.shape.model_dim % self.vector_width:
            raise ValueError(
                f"model_dim={self.shape.model_dim} must be divisible by "
                f"vector_width={self.vector_width}"
            )

    @property
    def output_rows(self) -> int:
        return self.m // self.shape.tp_size

    @property
    def requires_exact_m(self) -> bool:
        return True

    @property
    def vectors(self) -> int:
        return self.output_rows * self.shape.model_dim // self.vector_width

    @property
    def grid(self) -> int:
        return min(
            self.grid_cap,
            (self.vectors + DIRECT_BLOCK - 1) // DIRECT_BLOCK,
        )

    @property
    def partial_bytes(self) -> int:
        return self.m * self.shape.model_dim * 2

    @property
    def ready_offset(self) -> int:
        return _align_up(self.partial_bytes, 256)

    @property
    def done_offset(self) -> int:
        return self.ready_offset + self.grid * self.shape.tp_size * 8

    @property
    def workspace_bytes(self) -> int:
        return _align_up(
            self.done_offset + self.grid * self.shape.tp_size * 8,
            256,
        )


PipelineConfig = MegakernelConfig | WindowConfig | DirectConfig
