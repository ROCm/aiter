# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""RDNA decode and prefill attention over the paged KV cache, for
gfx1150..gfx1153 and RDNA4 (gfx1200, gfx1201): the kernels of
``csrc/kernels/rdna_attn``, ported from ROCm/vllm's ``ROCM_RDNA_ATTN``.

Each entry of ``csrc/kernels/rdna_attn/rdna_{decode,prefill}_configs.h`` is
one build of a kernel, with its shape and launch knobs as compile-time
constants; those of ``rdna_decode_batch_configs.h`` serve a batch. Each header
has a table per GPU family: gfx1151's (1536 VGPRs a SIMD lane), gfx1150's
(1024), which gfx1152 and gfx1153 also use, and RDNA4's gfx1201's, which
gfx1200 also uses.

``module_rdna_attn`` lists the entries of a device's tables and builds in
seconds: it holds no kernel. The kernels of one head dim and dtype are a
module of their own, ``module_rdna_attn_d<head dim>_<fp16|bf16>``, built the
first time one of them launches.

    variant = variant_for(hq, hkv, d, m, page_size, 1, 0, q.dtype)
    scratch = make_scratch(variant, q.device)
    load(variant).decode_attn(q, kv_cache, block_table, out, *scratch,
                              seq_lens, scale)

Decodes of up to ``MAX_M`` query tokens per sequence run on the decode builds
(``variant_for``, ``load``), everything else on the prefill builds
(``prefill_variant_for``, ``load_prefill``; a batch through
``prefill_batch_variant_for`` and ``plan_prefill``), and an encoder's
non-causal attention on ``encoder_forward``. A lookup returns None for a shape
without a build.
"""

import functools
from dataclasses import dataclass
from typing import List, NamedTuple, Optional

import numpy as np
import torch

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_list

MD_NAME = "module_rdna_attn"

# The GPU families (Arch in rdna_attn_types.h) by arch.
_FAMILY_OF_ARCH = {
    "gfx1150": "GFX1150",
    "gfx1151": "GFX1151",
    "gfx1152": "GFX1150",
    "gfx1153": "GFX1150",
    "gfx1200": "GFX1201",
    "gfx1201": "GFX1201",
}


def _build_families() -> List[str]:
    """The GPU families a module is built for: those of GPU_ARCHS, or of the
    GPUs of this machine."""
    return sorted({_FAMILY_OF_ARCH[a] for a in get_gfx_list() if a in _FAMILY_OF_ARCH})


def unit_module_name(head_size: int, dtype: torch.dtype) -> str:
    """The module holding the kernels of one head dim and dtype."""
    return f"{MD_NAME}_d{head_size}_{'bf16' if dtype == torch.bfloat16 else 'fp16'}"


@functools.cache
def _unit_build_args(head_size: int, dtype: torch.dtype) -> dict:
    """The build of unit_module_name (csrc/kernels/rdna_attn/rdna_attn_unit.cuh)."""
    flags = [
        "-DENABLE_CK=0",
        f"-DRDNA_ATTN_HEAD_DIM={head_size}",
        f"-DRDNA_ATTN_BF16={int(dtype == torch.bfloat16)}",
        # A build leaves some values unused.
        "-Wno-unused-variable",
    ]
    flags += [f"-DRDNA_ATTN_{family}=1" for family in _build_families()]
    return {
        "md_name": unit_module_name(head_size, dtype),
        "flags_extra_hip": flags,
    }


def _unit_of_q(q: torch.Tensor, *args, **kwargs) -> dict:
    return dict(_unit_build_args(q.shape[-1], q.dtype))


@compile_ops(MD_NAME)
def rdna_decode_configs(device_id: int) -> List[int]: ...


@compile_ops(MD_NAME)
def rdna_decode_batch_configs(device_id: int) -> List[int]: ...


@compile_ops(MD_NAME)
def rdna_prefill_configs(device_id: int) -> List[int]: ...


@compile_ops(MD_NAME)
def rdna_prefill_batch_configs(device_id: int) -> List[int]: ...


@compile_ops(MD_NAME, gen_func=_unit_of_q, develop=True)
def rdna_decode_attn(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    out: torch.Tensor,
    acc: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,  # noqa: E741
    cnt: torch.Tensor,
    seq_lens: torch.Tensor,
    config: int,
    scale: float,
    mm_ranges: Optional[torch.Tensor] = None,
    batched: bool = False,
) -> None: ...


@compile_ops(MD_NAME, gen_func=_unit_of_q, develop=True)
def rdna_prefill_attn(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    out: torch.Tensor,
    partial_o: torch.Tensor,
    partial_ml: torch.Tensor,
    counters: torch.Tensor,
    seq_lens: torch.Tensor,
    config: int,
    max_seq_len: int,
    scale: float,
    mm_ranges: Optional[torch.Tensor] = None,
    plan: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    head_group: int = 0,
) -> None: ...


@compile_ops(MD_NAME, gen_func=_unit_of_q, develop=True)
def rdna_encoder_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    partial_o: torch.Tensor,
    partial_ml: torch.Tensor,
    counters: torch.Tensor,
    seq_lens: torch.Tensor,
    config: int,
    max_seq_len: int,
    scale: float,
    plan: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    head_group: int = 0,
) -> None: ...


def _device_id(device: Optional[torch.device] = None) -> int:
    """The index of a CUDA device; of the current one by default."""
    if device is None:
        return torch.cuda.current_device()
    device = torch.device(device)
    return torch.cuda.current_device() if device.index is None else device.index


def device_family(device: Optional[torch.device] = None) -> Optional[str]:
    """The GPU family whose tables serve a device (the current one by
    default): "GFX1151", "GFX1150" or "GFX1201", or None on other GPUs."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(_device_id(device))
    name = getattr(props, "gcnArchName", "")
    return next((f for arch, f in _FAMILY_OF_ARCH.items() if arch in name), None)


@dataclass(frozen=True)
class KernelVariant:
    """One decode build: its shape and launch knobs."""

    head_size: int
    num_q_heads: int
    num_kv_heads: int
    max_query_len: int
    page_size: int
    layout: int  # 0 = NHD, 1 = HND
    # Most KV segments per (kv head, row group).
    max_segments: int = 1
    # Workgroups sharing a kv head's query rows.
    row_groups: int = 1
    # Fewest KV blocks per segment.
    min_segment_blocks: int = 1
    # Waves per workgroup.
    waves: int = 8
    # Waves splitting the head dim of a key tile; 0 is the kernel default.
    head_dim_split: int = 0
    # Prefetch a second tile per wave.
    prefetch: int = 0
    # Stage V in LDS beside K.
    v_in_lds: int = 0
    # Per-q-head dot products instead of WMMA.
    dot_product: int = 0
    dtype: torch.dtype = torch.float16
    # Sliding window in keys; 0 is full causal attention.
    window: int = 0
    # An entry of kDecodeBatchConfigs: a batch, one sequence per grid.y.
    batched: int = 0

    def __post_init__(self) -> None:
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"kernel is built for fp16 or bf16, not {self.dtype}")

    @property
    def name(self) -> str:
        """The DecodeConfig fields joined by "_"."""
        kernel = "decode_batch" if self.batched else "decode"
        return f"rdna_{kernel}_" + "_".join(map(str, config_fields(self)))

    @property
    def rows_padded(self) -> int:
        """Query rows per workgroup, padded to 16."""
        gqa = self.num_q_heads // self.num_kv_heads
        rows = gqa * self.max_query_len // self.row_groups
        return -(-rows // 16) * 16

    def scratch_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        """Shapes of the split-KV partials (acc, m/l) and the counter count."""
        if self.dot_product:
            rows = self.num_q_heads * self.max_segments * self.max_query_len
            cnt = self.num_q_heads
        else:
            rows = (
                self.num_kv_heads
                * self.row_groups
                * self.max_segments
                * self.rows_padded
            )
            # Per (kv head, row group): arrivals, generation, two masks.
            cnt = 4 * self.num_kv_heads * self.row_groups
        return (rows, self.head_size), (rows,), cnt


# DecodeConfig's fields (csrc/kernels/rdna_attn/rdna_decode_configs.h) in order,
# as KernelVariant fields; the dtype is 0 for fp16 and 1 for bf16 (Dtype in
# rdna_attn_types.h). Every build is HND.
DECODE_CONFIG_FIELDS = (
    "head_size",
    "num_q_heads",
    "num_kv_heads",
    "window",
    "max_query_len",
    "dtype",
    "page_size",
    "max_segments",
    "row_groups",
    "min_segment_blocks",
    "waves",
    "head_dim_split",
    "prefetch",
    "v_in_lds",
    "dot_product",
)
_HND = 1


def config_fields(variant: KernelVariant) -> list[int]:
    """The variant's DecodeConfig fields, in order."""
    return [
        int(variant.dtype == torch.bfloat16) if f == "dtype" else getattr(variant, f)
        for f in DECODE_CONFIG_FIELDS
    ]


def _lookup_key(v: KernelVariant) -> tuple:
    return (
        v.num_q_heads,
        v.num_kv_heads,
        v.head_size,
        v.max_query_len,
        v.page_size,
        v.window,
        v.dtype,
        v.batched,
    )


def _built_fields(op, fields: tuple[str, ...]) -> list[dict[str, int]]:
    """The entries of a table of the current device as fields, by index; none
    on a GPU without tables."""
    if device_family() is None:
        return []
    flat = op(_device_id())
    n = len(fields)
    return [
        dict(zip(fields, flat[i : i + n], strict=True)) for i in range(0, len(flat), n)
    ]


def _built_configs(op, fields: tuple[str, ...]) -> list[dict]:
    """_built_fields with the dtype as a torch dtype."""
    configs = _built_fields(op, fields)
    for f in configs:
        f["dtype"] = torch.bfloat16 if f["dtype"] else torch.float16
    return configs


@functools.cache
def _builds(batched: int = 0) -> list[KernelVariant]:
    """The decode builds of the current device's tables, by index."""
    op = rdna_decode_batch_configs if batched else rdna_decode_configs
    return [
        KernelVariant(**f, layout=_HND, batched=batched)
        for f in _built_configs(op, DECODE_CONFIG_FIELDS)
    ]


@functools.cache
def _by_key() -> dict[tuple, KernelVariant]:
    return {_lookup_key(v): v for v in variants()}


def variants() -> list[KernelVariant]:
    """Every decode build of the current device's tables."""
    return _builds(0) + _builds(1)


def variant_for(
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    max_query_len: int,
    page_size: int,
    layout: int,
    window: int,
    dtype: torch.dtype,
    batch: bool = False,
) -> KernelVariant | None:
    """The build for one call, or None."""
    if layout != _HND:
        return None
    return _by_key().get(
        (
            num_q_heads,
            num_kv_heads,
            head_size,
            max_query_len,
            page_size,
            window,
            dtype,
            int(batch),
        )
    )


class _Builtin:
    """A decode build, by its index in its table."""

    def __init__(self, index: int, batched: int):
        self.index = index
        self.batched = bool(batched)

    def decode_attn(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        out: torch.Tensor,
        acc: torch.Tensor,
        m: torch.Tensor,
        l: torch.Tensor,  # noqa: E741
        cnt: torch.Tensor,
        seq_lens: torch.Tensor,
        scale: float,
        mm_ranges: Optional[torch.Tensor] = None,
    ) -> None:
        """Launch the build: q and out (sequences * max_query_len, Hq, D),
        the scratch of make_scratch, int32 block_table and seq_lens."""
        rdna_decode_attn(
            q,
            kv_cache,
            block_table,
            out,
            acc,
            m,
            l,
            cnt,
            seq_lens,
            self.index,
            scale,
            mm_ranges,
            self.batched,
        )


_loaded: dict[KernelVariant, _Builtin | None] = {}


def load(variant: KernelVariant) -> _Builtin | None:
    """The variant's launcher, or None if the device's tables lack it. Its
    module is built at the first launch."""
    if variant not in _loaded:
        builds = _builds(variant.batched)
        _loaded[variant] = (
            _Builtin(builds.index(variant), variant.batched)
            if variant in builds
            else None
        )
    return _loaded[variant]


def make_scratch(
    variant: KernelVariant, device: torch.device, max_seqs: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate the split-KV scratch, outside CUDA-graph capture; max_seqs
    sequences of it for a batched build."""
    acc_shape, ml_shape, cnt = variant.scratch_shapes()
    lead = (max_seqs,) if max_seqs else ()
    opts = {"dtype": torch.float32, "device": device}
    return (
        torch.empty(lead + acc_shape, **opts),
        torch.empty(lead + ml_shape, **opts),
        torch.empty(lead + ml_shape, **opts),
        # Zeroed once; every launch leaves them zeroed.
        torch.zeros(lead + (cnt,), dtype=torch.int32, device=device),
    )


def scratch_bytes(variant: KernelVariant) -> int:
    """Scratch bytes per sequence of a batch."""
    acc_shape, ml_shape, cnt = variant.scratch_shapes()
    return 4 * (acc_shape[0] * acc_shape[1] + 2 * ml_shape[0] + cnt)


def expected_kv_cache_strides(variant: KernelVariant) -> tuple[int, int, int]:
    """Strides of (block, kv head, token) of the HND cache the kernel assumes:
    (NB, HKV, PAGE_SIZE, 2D).
    """
    kv_row = 2 * variant.head_size
    page = variant.page_size * variant.num_kv_heads * kv_row
    return page, variant.page_size * kv_row, kv_row


# Most query tokens the decode kernel serves.
MAX_M = 8

# Head dims the decode kernel supports.
SUPPORTED_HEAD_SIZES = (32, 48, 64, 72, 80, 96, 128, 256, 512)


# ---------------------------------------------------------------- prefill
#
# A shape has one build per step of M; a build serves M from its min_query_len
# up to the next one's, and the first also every smaller M.

# Fewest query tokens the prefill kernel serves.
PREFILL_MIN_M = MAX_M + 1


@functools.cache
def prefill_target_workgroups(device: torch.device) -> int:
    """Workgroups below which the prefill kernel splits the KV (one per SIMD)."""
    return 4 * torch.cuda.get_device_properties(device).multi_processor_count


# PrefillConfig's fields (csrc/kernels/rdna_attn/rdna_prefill_configs.h) in order,
# as PrefillVariant fields, the dtype as for decode.
PREFILL_CONFIG_FIELDS = (
    "head_size",
    "num_q_heads",
    "num_kv_heads",
    "window",
    "dtype",
    "page_size",
    "min_query_len",
    "waves",
    "key_tile",
    "head_dim_split",
    "v_t",
    "double_buffer",
    "key_waves",
    "max_segments",
    "head_group",
    "causal",
    "row_tiles",
)


@dataclass(frozen=True)
class PrefillVariant:
    """One prefill build: its shape, smallest M and launch knobs."""

    head_size: int
    num_q_heads: int
    num_kv_heads: int
    window: int
    page_size: int
    min_query_len: int
    waves: int
    key_tile: int
    head_dim_split: int
    v_t: int
    double_buffer: int
    key_waves: int
    max_segments: int
    head_group: int
    dtype: torch.dtype = torch.float16
    # 0: every query token sees every key of its sequence; a build of its own.
    causal: int = 1
    # 16-row tiles per wave, 1 or (non-causal) 2.
    row_tiles: int = 1

    @property
    def name(self) -> str:
        """The PrefillConfig fields joined by "_"."""
        return "rdna_prefill_" + "_".join(map(str, prefill_config_fields(self)))

    @property
    def block_rows(self) -> int:
        """Query rows per workgroup."""
        return (
            self.waves // (self.head_dim_split * self.key_waves) * 16 * self.row_tiles
        )

    def scratch_rows(self, device: torch.device) -> int:
        """Most split-KV partial rows a launch may write."""
        return 2 * prefill_target_workgroups(device) * self.block_rows


def prefill_config_fields(variant: PrefillVariant) -> list[int]:
    """The variant's PrefillConfig fields, in order."""
    return [
        int(variant.dtype == torch.bfloat16) if f == "dtype" else getattr(variant, f)
        for f in PREFILL_CONFIG_FIELDS
    ]


@functools.cache
def _prefill_builds() -> list[PrefillVariant]:
    """The prefill builds of the current device's tables, by index."""
    return [
        PrefillVariant(**f)
        for f in _built_configs(rdna_prefill_configs, PREFILL_CONFIG_FIELDS)
    ]


@functools.cache
def _prefill_table() -> dict[tuple, list[PrefillVariant]]:
    """The builds by M per (Hq, Hkv, D, page, window, dtype, causal)."""
    table: dict[tuple, list[PrefillVariant]] = {}
    for v in _prefill_builds():
        key = (
            v.num_q_heads,
            v.num_kv_heads,
            v.head_size,
            v.page_size,
            v.window,
            v.dtype,
            v.causal,
        )
        table.setdefault(key, []).append(v)
    for by_m in table.values():
        by_m.sort(key=lambda v: v.min_query_len)
    return table


def prefill_variants(
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    page_size: int,
    window: int,
    dtype: torch.dtype,
    causal: bool = True,
) -> list[PrefillVariant]:
    """Every row of one shape, sorted by M."""
    return _prefill_table().get(
        (num_q_heads, num_kv_heads, head_size, page_size, window, dtype, int(causal)),
        [],
    )


def prefill_variant_for(
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    page_size: int,
    layout: int,
    window: int,
    dtype: torch.dtype,
    num_tokens: int,
    causal: bool = True,
) -> PrefillVariant | None:
    """The row for num_tokens query tokens, or None."""
    if layout != _HND or num_tokens < PREFILL_MIN_M:
        return None
    rows = prefill_variants(
        num_q_heads, num_kv_heads, head_size, page_size, window, dtype, causal
    )
    served = [v for v in rows if v.min_query_len <= num_tokens]
    return served[-1] if served else (rows[0] if rows else None)


# ---------------------------------------------------------------- prefill batches
#
# A batch runs on the build and head group kPrefillBatchConfigs picks per shape
# and cell of (mean query tokens, mean keys per token).

# PrefillBatchConfig's fields in order.
PREFILL_BATCH_CONFIG_FIELDS = (
    "head_size",
    "num_q_heads",
    "num_kv_heads",
    "window",
    "page_size",
    "min_mean_m",
    "min_keys",
    "min_query_len",
    "head_group",
    "causal",
)
# The cells' lower bounds.
PREFILL_BATCH_MEAN_M = (1, 16, 64, 256, 1024, 4096)
PREFILL_BATCH_KEYS = (0, 512, 2048, 8192)


def prefill_batch_features(
    query_start_loc: np.ndarray,
    seq_lens: np.ndarray,
    window: int,
    causal: bool = True,
) -> tuple[float, float]:
    """(mean query tokens, mean keys per token) of a batch, padding left out."""
    q_lens = np.diff(query_start_loc.astype(np.int64))
    seq_lens = seq_lens[: q_lens.shape[0]].astype(np.int64)
    live = q_lens > 0
    q_lens, seq_lens = q_lens[live], np.maximum(seq_lens[live], q_lens[live])
    tokens = int(q_lens.sum())
    if not tokens:
        return 0.0, 0.0
    keys = seq_lens - q_lens + (q_lens + 1) / 2 if causal else seq_lens
    if window:
        keys = np.minimum(keys, window)
    return tokens / q_lens.shape[0], float((q_lens * keys).sum() / tokens)


@functools.cache
def _prefill_batch_table() -> dict[tuple, list[tuple[int, int, int, int]]]:
    """Per (Hq, Hkv, D, page, window, causal), its cells (min mean M, min keys,
    min_query_len, head group)."""
    table: dict[tuple, list[tuple[int, int, int, int]]] = {}
    for f in _built_fields(rdna_prefill_batch_configs, PREFILL_BATCH_CONFIG_FIELDS):
        key = (
            f["num_q_heads"],
            f["num_kv_heads"],
            f["head_size"],
            f["page_size"],
            f["window"],
            f["causal"],
        )
        table.setdefault(key, []).append(
            (f["min_mean_m"], f["min_keys"], f["min_query_len"], f["head_group"])
        )
    return table


def prefill_batch_cell(
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    page_size: int,
    window: int,
    mean_m: float,
    keys: float,
    causal: bool = True,
) -> tuple[int, int] | None:
    """(min_query_len, head group) of the cell a batch falls in, or None."""
    cells = _prefill_batch_table().get(
        (num_q_heads, num_kv_heads, head_size, page_size, window, int(causal))
    )
    if not cells:
        return None
    steps = sorted({c[0] for c in cells})
    m_step = max((x for x in steps if x <= mean_m), default=steps[0])
    row = [c for c in cells if c[0] == m_step]
    below = [c for c in row if c[1] <= keys]
    cell = max(below, key=lambda c: c[1]) if below else min(row, key=lambda c: c[1])
    return cell[2], cell[3]


def prefill_batch_variant_for(
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    page_size: int,
    layout: int,
    window: int,
    dtype: torch.dtype,
    query_start_loc: np.ndarray,
    seq_lens: np.ndarray,
    causal: bool = True,
) -> tuple[PrefillVariant | None, int]:
    """The build and head group for a prefill batch: its cell's, else the
    build of its tokens."""
    mean_m, keys = prefill_batch_features(query_start_loc, seq_lens, window, causal)
    cell = prefill_batch_cell(
        num_q_heads, num_kv_heads, head_size, page_size, window, mean_m, keys, causal
    )
    rows = prefill_variants(
        num_q_heads, num_kv_heads, head_size, page_size, window, dtype, causal
    )
    if cell is not None and layout == _HND:
        row = next((v for v in rows if v.min_query_len == cell[0]), None)
        if row is not None:
            return row, cell[1]
    tokens = int(query_start_loc[-1] - query_start_loc[0])
    v = prefill_variant_for(
        num_q_heads,
        num_kv_heads,
        head_size,
        page_size,
        layout,
        window,
        dtype,
        max(tokens, PREFILL_MIN_M),
        causal,
    )
    return v, 0


# The kernel's kMinSegmentSteps.
_MIN_SEGMENT_STEPS = 16
# Most segments a plan splits one tile into.
PLAN_MAX_SEGMENTS = 16


def plan_prefill(
    query_start_loc: np.ndarray,
    seq_lens: np.ndarray,
    variant: PrefillVariant,
    window: int,
    target_workgroups: int,
    head_group: int = 0,
) -> tuple[np.ndarray, int]:
    """The work list of a prefill launch over a batch.

    An entry is one (sequence, tile, KV segment), run for every kv head,
    heaviest first. Tiles heavier than the mean load per SIMD are split.

    Args:
        query_start_loc: The batch's query starts.
        seq_lens: Each sequence's length, or an upper bound.
        variant: The build that runs the launch.
        window: Sliding window in keys, 0 for full causal (or non-causal).
        target_workgroups: One per SIMD (prefill_target_workgroups).
        head_group: Kv heads to interleave (0: the build's).

    Returns:
        The int32 [entries, 4] list {seq, tile | split tile << 16,
        segment | segments << 16, first partials slot or -1}, and the head
        group (all kv heads if a tile splits).

    """
    g = variant.num_q_heads // variant.num_kv_heads
    rows, key_tile = variant.block_rows, variant.key_tile
    q_lens = np.diff(query_start_loc.astype(np.int64))
    nseq = q_lens.shape[0]
    context = np.maximum(seq_lens[:nseq].astype(np.int64), q_lens) - q_lens
    tiles = (q_lens * g + rows - 1) // rows
    # Last (heaviest) tile first.
    seq = np.repeat(np.arange(nseq), tiles)
    starts = np.cumsum(tiles) - tiles
    tile = np.repeat(starts + tiles - 1, tiles) - np.arange(seq.shape[0])
    first = tile * rows // g
    last = np.minimum(q_lens[seq] - 1, ((tile + 1) * rows - 1) // g)
    # Non-causal: every token sees its whole sequence.
    key_end = context[seq] + last + 1 if variant.causal else context[seq] + q_lens[seq]
    key_begin = (
        np.maximum(0, context[seq] + first - window + 1) // key_tile * key_tile
        if window
        else 0
    )
    steps = (key_end - key_begin + key_tile - 1) // key_tile

    load = max(
        variant.num_kv_heads * int(steps.sum()) / target_workgroups,
        float(_MIN_SEGMENT_STEPS),
    )
    split = steps > load
    segments = np.where(
        split,
        np.clip(
            np.minimum(-(-steps // load), steps // _MIN_SEGMENT_STEPS),
            1,
            PLAN_MAX_SEGMENTS,
        ),
        1,
    ).astype(np.int64)
    split &= segments > 1
    split_index = np.cumsum(split) - 1
    split_slots = np.where(split, segments, 0)
    slot = np.where(split, np.cumsum(split_slots) - split_slots, -1)
    assert int(split.sum()) * variant.num_kv_heads < target_workgroups
    assert int(split_slots.sum()) * variant.num_kv_heads < 2 * target_workgroups
    assert tile.size == 0 or int(tile.max()) < 1 << 16

    owner = np.repeat(np.arange(tile.shape[0]), segments)
    segment = np.arange(owner.shape[0]) - np.repeat(
        np.cumsum(segments) - segments, segments
    )
    plan = np.stack(
        [
            seq[owner],
            tile[owner] | np.where(split[owner], split_index[owner], 0) << 16,
            segment | segments[owner] << 16,
            slot[owner],
        ],
        axis=1,
    )
    order = np.argsort(-steps[owner] / segments[owner], kind="stable")
    if split.any():
        head_group = variant.num_kv_heads
    return np.ascontiguousarray(plan[order], dtype=np.int32), head_group


class _BuiltinPrefill:
    """A prefill build, by its index in its table."""

    def __init__(self, index: int):
        self.index = index

    def prefill_attn(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        out: torch.Tensor,
        partial_o: torch.Tensor,
        partial_ml: torch.Tensor,
        counters: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        scale: float,
        mm_ranges: Optional[torch.Tensor] = None,
        plan: Optional[torch.Tensor] = None,
        query_start_loc: Optional[torch.Tensor] = None,
        head_group: int = 0,
    ) -> None:
        """Launch the causal build over one sequence, or over a batch with
        plan_prefill's plan and its query starts."""
        rdna_prefill_attn(
            q,
            kv_cache,
            block_table,
            out,
            partial_o,
            partial_ml,
            counters,
            seq_lens,
            self.index,
            max_seq_len,
            scale,
            mm_ranges,
            plan,
            query_start_loc,
            head_group,
        )

    def encoder_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        partial_o: torch.Tensor,
        partial_ml: torch.Tensor,
        counters: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        scale: float,
        plan: Optional[torch.Tensor] = None,
        query_start_loc: Optional[torch.Tensor] = None,
        head_group: int = 0,
    ) -> None:
        """Launch the non-causal build over Q, K and V read in place."""
        rdna_encoder_attn(
            q,
            k,
            v,
            out,
            partial_o,
            partial_ml,
            counters,
            seq_lens,
            self.index,
            max_seq_len,
            scale,
            plan,
            query_start_loc,
            head_group,
        )


_prefill_loaded: dict[PrefillVariant, _BuiltinPrefill | None] = {}


def load_prefill(variant: PrefillVariant) -> _BuiltinPrefill | None:
    """The variant's launcher, or None if the device's tables lack it. Its
    module is built at the first launch."""
    if variant not in _prefill_loaded:
        builds = _prefill_builds()
        _prefill_loaded[variant] = (
            _BuiltinPrefill(builds.index(variant)) if variant in builds else None
        )
    return _prefill_loaded[variant]


def make_prefill_scratch(
    variants: list[PrefillVariant], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split-KV scratch (O, max/sum, counters) for any of `variants`."""
    rows = max(v.scratch_rows(device) for v in variants)
    head_size = max(v.head_size for v in variants)
    opts = {"dtype": torch.float32, "device": device}
    return (
        torch.empty(rows * head_size, **opts),
        torch.empty(rows * 2, **opts),
        torch.zeros(
            prefill_target_workgroups(device), dtype=torch.int32, device=device
        ),
    )


# ---------------------------------------------------------------- encoder
#
# An encoder's segments, each token seeing every key of its segment, in one
# launch of the non-causal prefill builds, which read Q, K and V in place.

# The page size the non-causal builds are listed under; they read no pages.
ENCODER_PAGE_SIZE = 16
# Per (shape, device): prefill scratch, sized for every non-causal build.
_ENCODER_SCRATCH: dict = {}


def encoder_unsupported(q: torch.Tensor, num_kv_heads: int) -> str | None:
    """Why encoder_forward cannot serve this call, or None."""
    if q.dtype not in (torch.float16, torch.bfloat16):
        return f"dtype {q.dtype}"
    hq, d = q.shape[1], q.shape[2]
    if d not in SUPPORTED_HEAD_SIZES:
        return f"head size {d}"
    if hq % num_kv_heads:
        return "q heads must divide evenly over kv heads"
    if not prefill_variants(hq, num_kv_heads, d, ENCODER_PAGE_SIZE, 0, q.dtype, False):
        return f"no non-causal build for {hq}/{num_kv_heads}/{d} {q.dtype}"
    return None


class _EncoderLayout(NamedTuple):
    """How one list of segment lengths runs, on the device."""

    variant: PrefillVariant
    seq_lens: torch.Tensor
    # The plan and query starts; None for one segment.
    plan: torch.Tensor | None
    query_starts: torch.Tensor | None
    head_group: int
    longest: int


# A video gives a list per clip shape; the entries are a few small tensors.
@functools.lru_cache(maxsize=128)
def _encoder_layout(
    seg_lens: tuple[int, ...],
    hq: int,
    hkv: int,
    d: int,
    dtype: torch.dtype,
    dev: torch.device,
) -> _EncoderLayout:
    def to_dev(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.int32)).to(dev)

    lens = np.asarray(seg_lens, dtype=np.int64)
    shape = (hq, hkv, d, ENCODER_PAGE_SIZE)
    if len(lens) == 1:
        # The first row also serves fewer than PREFILL_MIN_M tokens.
        variant = prefill_variant_for(
            *shape, _HND, 0, dtype, max(int(lens[0]), PREFILL_MIN_M), causal=False
        )
        assert variant is not None, "encoder_unsupported checks the builds"
        return _EncoderLayout(variant, to_dev(lens), None, None, 0, int(lens[0]))
    starts = np.concatenate([[0], np.cumsum(lens)])
    variant, head_group = prefill_batch_variant_for(
        *shape, _HND, 0, dtype, starts, lens, causal=False
    )
    assert variant is not None, "encoder_unsupported checks the builds"
    plan, head_group = plan_prefill(
        starts, lens, variant, 0, prefill_target_workgroups(dev), head_group
    )
    return _EncoderLayout(
        variant,
        to_dev(lens),
        to_dev(plan),
        to_dev(starts),
        head_group,
        int(lens.max()),
    )


def _heads_contiguous(x: torch.Tensor) -> torch.Tensor:
    """x, or a copy with each token's heads contiguous."""
    return x if x.stride(2) == 1 and x.stride(1) == x.shape[2] else x.contiguous()


def encoder_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seg_lens: list[int],
    scale: float,
) -> torch.Tensor:
    """Attention of each segment's tokens over its keys, no mask.

    Args:
        q: (tokens, Hq, D), the segments one after the other.
        k: (tokens, Hkv, D).
        v: (tokens, Hkv, D).
        seg_lens: Each segment's tokens, on the host.
        scale: Softmax scale.

    Returns:
        (tokens, Hq, D). `encoder_unsupported` must have returned None.

    """
    dev, dtype = q.device, q.dtype
    out = torch.empty(q.shape, dtype=dtype, device=dev)
    if not seg_lens:
        return out
    hq, d, hkv = q.shape[1], q.shape[2], k.shape[1]
    lay = _encoder_layout(tuple(seg_lens), hq, hkv, d, dtype, dev)
    q, k, v = _heads_contiguous(q), _heads_contiguous(k), _heads_contiguous(v)
    prefill = load_prefill(lay.variant)
    assert prefill is not None
    shape = (hq, hkv, d, ENCODER_PAGE_SIZE, 0, dtype)
    scratch = _ENCODER_SCRATCH.get((shape, dev))
    if scratch is None or scratch[1].numel() < 2 * lay.variant.scratch_rows(dev):
        scratch = make_prefill_scratch(
            [*prefill_variants(*shape, causal=False), lay.variant], dev
        )
        _ENCODER_SCRATCH[(shape, dev)] = scratch
    if lay.plan is None or lay.plan.shape[0]:
        prefill.encoder_attn(
            q,
            k,
            v,
            out,
            *scratch,
            lay.seq_lens,
            lay.longest,
            scale,
            plan=lay.plan,
            query_start_loc=lay.query_starts,
            head_group=lay.head_group,
        )
    return out
