# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms
import torch
from flydsl.runtime.device import get_rocm_arch
from mori.shmem import mori_shmem_create_tensor

from .communication_ops_utils import GeometryTuningTable
from .flydsl_dispatch_combine_intranode_kernel import (
    make_combine_jit,
    make_dispatch_jit,
)

# Reject unsupported token dtypes at construction, not deep in JIT codegen.
_SUPPORTED_TOK_DTYPES = (
    torch.bfloat16,
    torch.float32,
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float4_e2m1fn_x2,
)

_SUPPORTED_QUANT_TYPES = ("none", "fp8_direct_cast")
_SUPPORTED_STAGE2_P2P_QUANT_TYPES = ("none", "fp8_blockwise_1x32")

_MAX_INTRANODE_NPES = 8

# Kernel streams 16-byte chunks (v4i32); sets ``token_bytes`` alignment.
_TOK_BYTES_ALIGN = 16

_DEFAULT_DISPATCH_BLOCK_NUM = 128
_DEFAULT_DISPATCH_WARP_NUM = 4
_DEFAULT_COMBINE_BLOCK_NUM = 128
_DEFAULT_COMBINE_WARP_NUM = 8

logger = logging.getLogger(__name__)

# Per-shape tuning JSONs (schema: flydsl_{arch}_{model}_{kernel}_ep{n}.json).
_TUNING_CONFIGS_DIR = Path(__file__).resolve().parent / "mega_moe_tuning_config"


@functools.cache
def _device_cu_count(device_index):
    """CU count (ROCm ``multiProcessorCount``) for the resident-block bound."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _check_block_num_resident(phase, block_num):
    """Hard-cap block_num at #CU: the phase-ending grid-wide barrier needs all
    blocks co-resident, so block_num > #CU risks a spin-wait deadlock."""
    num_cu = _device_cu_count(torch.cuda.current_device())
    if block_num > num_cu:
        raise ValueError(
            f"{phase}: block_num={block_num} exceeds device CU count ({num_cu}). "
            f"The kernel uses a grid-wide barrier requiring all blocks to be "
            f"co-resident; block_num > #CU risks a spin-wait deadlock when surplus "
            f"blocks cannot be scheduled. Keep block_num <= {num_cu}."
        )


def _resolve_launch_geometry(
    phase, block_num, warp_num_per_block, table, num_tokens, default_bn, default_wpb
):
    """Geometry resolution: explicit per-call override > table lookup by token
    count > cfg default. Explicit override needs BOTH values (>0) or neither."""
    has_bn = block_num is not None
    has_wpb = warp_num_per_block is not None
    if has_bn != has_wpb:
        raise ValueError(
            f"{phase}: block_num and warp_num_per_block must be passed together "
            f"or both omitted (got block_num={block_num}, warp_num_per_block={warp_num_per_block})"
        )
    if has_bn:
        if block_num <= 0 or warp_num_per_block <= 0:
            raise ValueError(
                f"{phase}: launch geometry must be positive, "
                f"got block_num={block_num}, warp_num_per_block={warp_num_per_block}"
            )
        geom, source = (block_num, warp_num_per_block), "explicit"
    else:
        hit = table.lookup(phase, num_tokens) if table is not None else None
        if hit is not None:
            geom, source = hit, "config"
        else:
            geom, source = (default_bn, default_wpb), "default"
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "%s launch geometry: block_num=%d warp_num_per_block=%d "
            "(source=%s, num_tokens=%d)",
            phase,
            geom[0],
            geom[1],
            source,
            num_tokens,
        )
    return geom


# torch dtype -> tuning-file ``dtype`` string (fp8_ocp == OCP E4M3).
_TUNING_DTYPE_NAME = {
    torch.float4_e2m1fn_x2: "fp4",
    torch.float8_e4m3fn: "fp8_ocp",
    torch.float8_e4m3fnuz: "fp8_ocp",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}


def dtype_to_tuning_name(dt):
    try:
        return _TUNING_DTYPE_NAME[dt]
    except KeyError:
        raise ValueError(f"no tuning-table dtype name for {dt}")


def _detect_gpu_model(device_index=0):
    """GPU model substring (e.g. ``"mi355x"``) for tuning-file selection."""
    try:
        name = torch.cuda.get_device_properties(device_index).name.lower()
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r"\bmi\d+\w*", name)
    return m.group(0) if m else None


def resolve_tuning_config_path(
    ep_size, *, kernel_type="IntraNode", gpu_arch=None, gpu_model=None
):
    if not _TUNING_CONFIGS_DIR.is_dir():
        return None
    if gpu_arch is None:
        try:
            gpu_arch = str(get_rocm_arch() or "")
        except Exception:  # noqa: BLE001
            gpu_arch = None
    if gpu_model is None:
        gpu_model = _detect_gpu_model()
    suffix = f"_{kernel_type}_ep{ep_size}.json"
    candidates = [
        p for p in _TUNING_CONFIGS_DIR.glob(f"flydsl_*{suffix}") if p.is_file()
    ]
    if not candidates:
        return None

    def _score(p):
        n = p.name
        return (
            1 if (gpu_arch and gpu_arch in n) else 0,
            1 if (gpu_model and gpu_model in n) else 0,
        )

    candidates.sort(key=lambda p: (_score(p), p.name), reverse=True)
    return candidates[0]


def build_geometry_tuning_table_for_config(cfg, path=None):
    """Build a :class:`GeometryTuningTable` for ``cfg``'s shape (tuning JSON
    auto-resolved from ``cfg.world_size`` when ``path`` omitted); None on miss."""
    if path is None:
        path = resolve_tuning_config_path(cfg.world_size)
    if path is None or not Path(path).is_file():
        return None
    try:
        dispatch_dtype_name = dtype_to_tuning_name(cfg.dispatch_dtype)
    except ValueError:
        return None
    table = GeometryTuningTable.from_tuning_file(
        str(path),
        dtype=dispatch_dtype_name,
        hidden_dim=cfg.hidden_dim,
        zero_copy=cfg.zero_copy,
        topk=cfg.num_experts_per_token,
        local_expert_num=cfg.num_experts_per_rank,
        combine_dtype="bf16",
    )
    if not table.dispatch and not table.combine:
        logger.warning(
            "Tuning file %s has no rule matching shape "
            "(dtype=%s, hidden=%d, topk=%d, local_experts=%d, zero_copy=%s); "
            "using static geometry defaults.",
            path,
            dispatch_dtype_name,
            cfg.hidden_dim,
            cfg.num_experts_per_token,
            cfg.num_experts_per_rank,
            cfg.zero_copy,
        )
        return None
    logger.info(
        "Loaded FlyDSL launch-geometry tuning from %s "
        "(dispatch=%d rules, combine=%d rules)",
        path,
        len(table.dispatch),
        len(table.combine),
    )
    return table


def _dtype_elem_size(dt):
    """Raw storage size in bytes. fp4x2 returns 1 (two fp4 per byte)."""
    return torch.tensor([], dtype=dt).element_size()


def _is_fp4_dtype(dt):
    return dt == torch.float4_e2m1fn_x2


def _token_bytes_for(dt, hidden_dim):
    """Per-row payload bytes; parametrised on launch-time dtype so dispatch
    / combine can use independent dtypes (mori parity)."""
    if _is_fp4_dtype(dt):
        return hidden_dim // 2
    return hidden_dim * _dtype_elem_size(dt)


def _token_view_dim_for(dt, hidden_dim):
    """Per-row torch view trailing dim (fp4 packs 2 elements per byte)."""
    return hidden_dim // 2 if _is_fp4_dtype(dt) else hidden_dim


@dataclass
class FlyDSLDispatchCombineConfig:
    rank: int
    world_size: int
    hidden_dim: int
    max_num_inp_token_per_rank: int
    num_experts_per_rank: int
    num_experts_per_token: int
    data_type: torch.dtype | None = None
    # User-pinned launch geometry, resolved ONCE per op (not per call):
    # pinned value (block_num+warp must be set together) > tuning table >
    # ``default_*`` fallback above. None => not pinned.
    dispatch_block_num: int | None = None
    dispatch_warp_num_per_block: int | None = None
    combine_block_num: int | None = None
    combine_warp_num_per_block: int | None = None
    tuning_table: GeometryTuningTable | None = None
    tuning_config_path: str | None = None
    # Per-token dispatch scale count and element size.
    scale_dim: int = 0
    scale_type_size: int = 0
    enable_std_moe: bool = False
    zero_copy: bool = False
    quant_type: str | None = None
    max_total_recv_tokens: int = 0
    max_token_type_size: int = 0
    dispatch_dtype: torch.dtype | None = None
    combine_dtype: torch.dtype | None = None
    combine_quant_type: str | None = None
    stage2_p2p_quant: str = "none"
    enable_group_major: bool = False
    gm_unit_size: int = 0
    gm_scheme: str = "fixedslot"
    gm_compact: bool = False

    def __post_init__(self):
        if self.data_type is not None and (
            self.dispatch_dtype not in (None, self.data_type)
            or self.combine_dtype not in (None, self.data_type)
        ):
            raise ValueError("data_type conflicts with dispatch_dtype or combine_dtype")
        self.dispatch_dtype = self.dispatch_dtype or self.data_type or torch.bfloat16
        self.combine_dtype = self.combine_dtype or self.data_type or torch.bfloat16
        self.data_type = (
            self.dispatch_dtype if self.dispatch_dtype == self.combine_dtype else None
        )
        if self.quant_type is not None and self.combine_quant_type not in (
            None,
            self.quant_type,
        ):
            raise ValueError("quant_type conflicts with combine_quant_type")
        self.combine_quant_type = self.combine_quant_type or self.quant_type or "none"
        self.quant_type = self.combine_quant_type

    @property
    def is_fp4(self):
        return self.data_type == torch.float4_e2m1fn_x2

    @property
    def elem_size(self):
        return _dtype_elem_size(self.data_type or self.dispatch_dtype)

    @property
    def token_bytes(self):
        return _token_bytes_for(self.data_type or self.dispatch_dtype, self.hidden_dim)

    @property
    def token_view_dim(self):
        return _token_view_dim_for(
            self.data_type or self.dispatch_dtype, self.hidden_dim
        )

    @property
    def max_token_bytes(self):
        return (
            self.hidden_dim * self.max_token_type_size
            if self.max_token_type_size > 0
            else self.token_bytes
        )

    @property
    def dispatch_is_fp4(self):
        return self.dispatch_dtype == torch.float4_e2m1fn_x2

    @property
    def combine_is_fp4(self):
        return self.combine_dtype == torch.float4_e2m1fn_x2

    @property
    def dispatch_elem_size(self):
        return _dtype_elem_size(self.dispatch_dtype)

    @property
    def combine_elem_size(self):
        return _dtype_elem_size(self.combine_dtype)

    @property
    def dispatch_token_bytes(self):
        return _token_bytes_for(self.dispatch_dtype, self.hidden_dim)

    @property
    def combine_token_bytes(self):
        return _token_bytes_for(self.combine_dtype, self.hidden_dim)

    @property
    def dispatch_token_view_dim(self):
        return _token_view_dim_for(self.dispatch_dtype, self.hidden_dim)

    @property
    def combine_token_view_dim(self):
        return _token_view_dim_for(self.combine_dtype, self.hidden_dim)

    @property
    def max_recv(self):
        return self.world_size * self.max_num_inp_token_per_rank

    @property
    def effective_max_recv_per_rank(self):
        if self.max_total_recv_tokens <= 0:
            return self.max_num_inp_token_per_rank
        per_rank = (self.max_total_recv_tokens + self.world_size - 1) // self.world_size
        return min(per_rank, self.max_num_inp_token_per_rank)

    @property
    def effective_max_recv(self):
        """Return the destination receive-slot count."""
        return self.world_size * self.effective_max_recv_per_rank

    @property
    def scale_bytes(self):
        return self.scale_dim * self.scale_type_size


def build_p2p_table(t, rank, npes, dev):
    """i64[npes] table of intra-node P2P pointers to ``t`` on every peer (self incl.)."""
    tbl = torch.zeros(npes, dtype=torch.int64, device=dev)
    for pe in range(npes):
        tbl[pe] = ms.shmem_ptr_p2p(t.data_ptr(), rank, pe)
    return tbl


class FlyDSLDispatchGroupMajorOp:
    """Own expert-major dispatch buffers and metadata for fused stage1."""

    def __init__(
        self,
        *,
        rank,
        world_size,
        hidden_dim,
        max_tok_per_rank,
        experts_per_rank,
        topk,
        data_type,
        unit_size,
        scale_dim,
        scale_type_size=1,
        compact=False,
    ):
        assert world_size <= 8
        # Compact mode uses a count-first layout without per-expert reservations.
        self.compact = bool(compact)
        self.rank = rank
        self.npes = world_size
        self.hidden = hidden_dim
        self.mtpr = max_tok_per_rank
        self.epr = experts_per_rank
        self.topk = topk
        self.dtype = data_type
        self.unit = unit_size
        # Fixed-slot capacity covers the all-to-one routing case.
        self.max_tokens_per_expert = world_size * max_tok_per_rank
        self.ll_cap = (
            (self.max_tokens_per_expert + unit_size - 1) // unit_size
        ) * unit_size
        self.scale_dim = scale_dim
        self.scale_type_size = scale_type_size
        self.scale_bytes = scale_dim * scale_type_size
        self.scale_n_i32 = (self.scale_bytes + 3) // 4 if self.scale_bytes > 0 else 0
        self._dev = torch.device("cuda", rank)
        self.row_bytes = _token_bytes_for(data_type, hidden_dim)
        self.row_view = _token_view_dim_for(data_type, hidden_dim)

        # Compact capacity includes worst-case routes plus per-expert padding.
        if self.compact:
            num_valid_max = (
                world_size * max_tok_per_rank * topk + experts_per_rank * unit_size
            )
        else:
            num_valid_max = experts_per_rank * self.ll_cap + 256
        self.num_valid_max = int(num_valid_max)
        self.max_blocks = (self.num_valid_max + unit_size - 1) // unit_size

        self._alloc()
        ms.shmem_barrier_all()
        self._build_p2p()

    def _sym(self, shape, dtype):
        t = mori_shmem_create_tensor(shape, dtype)
        t.zero_()
        return t

    def _alloc(self):
        npes, epr = self.npes, self.epr
        nvm = self.num_valid_max
        self.done2 = self._sym((npes,), torch.int32)
        self.running = self._sym((epr,), torch.int32)
        self.ll_count = self._sym((epr,), torch.int32)
        self.rx_em = self._sym((nvm * self.row_bytes,), torch.int8)
        self.scale_em = self._sym((max(1, nvm * self.scale_n_i32),), torch.int32)
        self.idx_em = self._sym((nvm,), torch.int32)
        self.wts_em = self._sym((nvm,), torch.float32)
        self.srcmap_em = self._sym((nvm,), torch.int32)
        self.gb1 = torch.zeros(1, dtype=torch.int64, device=self._dev)
        self.sorted_expert_ids = torch.zeros(
            self.max_blocks, dtype=torch.int32, device=self._dev
        )
        self.tile_row_base = torch.zeros(
            self.max_blocks, dtype=torch.int32, device=self._dev
        )
        self.num_valid = torch.zeros(2, dtype=torch.int32, device=self._dev)
        # Fused dispatch writes the distinct receive count consumed by stage2.
        self.total_recv = torch.zeros(
            1, dtype=torch.int32, device=self._dev
        )  # local; kernel accumulates
        self.dest_ctr = torch.zeros(
            self.npes, dtype=torch.int32, device=self._dev
        )  # local send-count/dest
        self.recv_num = self._sym(
            (self.npes,), torch.int32
        )  # symmetric: peers signal here
        self.compact_base = None
        self.done2c = None
        self.gb_cnt = None
        self.meta2 = None
        self.write_cursor = None
        if self.compact:
            self.compact_base = self._sym(
                (epr,), torch.int32
            )  # dense prefix-sum base per local expert
            self.done2c = self._sym(
                (npes,), torch.int32
            )  # COUNT cross-PE done-barrier (vs done2 WRITE)
            self.gb_cnt = torch.zeros(
                1, dtype=torch.int64, device=self._dev
            )  # count-pass grid-arrival counter
            self.meta2 = torch.zeros(
                1, dtype=torch.int32, device=self._dev
            )  # payload-ready flag
            self.write_cursor = self._sym(
                (epr,), torch.int32
            )  # phase-2 write cursor (reset only at kernel end)
            self.done2cb = self._sym(
                (npes,), torch.int32
            )  # cross-PE#1b: gate compact_base read in phase-2
            _te = npes * epr
            self.local_hist = torch.zeros(
                _te, dtype=torch.int32, device=self._dev
            )  # per-launch reset
            self.bigcnt = self._sym((npes * _te,), torch.int32)  # [src][ge] all-gather
            self.cnt_done = self._sym((npes,), torch.int32)  # all-gather epoch barrier
            self.my_base = torch.zeros(
                _te, dtype=torch.int32, device=self._dev
            )  # my tokens' base in each dest
            self.local_cursor = torch.zeros(
                _te, dtype=torch.int32, device=self._dev
            )  # per-launch reset (write cursor)

    def _p2p_table(self, t):
        return build_p2p_table(t, self.rank, self.npes, self._dev)

    def _build_p2p(self):
        self.p2p_done2 = self._p2p_table(self.done2)
        self.p2p_running = self._p2p_table(self.running)
        self.p2p_rx_em = self._p2p_table(self.rx_em)
        self.p2p_scale_em = self._p2p_table(self.scale_em)
        self.p2p_idx_em = self._p2p_table(self.idx_em)
        self.p2p_wts_em = self._p2p_table(self.wts_em)
        self.p2p_srcmap_em = self._p2p_table(self.srcmap_em)
        self.p2p_recv_num = self._p2p_table(
            self.recv_num
        )  # Plan A: cross-PE recv-count signal target
        if self.compact:
            self.p2p_compact_base = self._p2p_table(self.compact_base)
            self.p2p_done2c = self._p2p_table(self.done2c)
            self.p2p_write_cursor = self._p2p_table(self.write_cursor)
            self.p2p_done2cb = self._p2p_table(self.done2cb)
            self.p2p_bigcnt = self._p2p_table(self.bigcnt)
            self.p2p_cnt_done = self._p2p_table(self.cnt_done)

    def reset_counters(self):
        """No-op: the per-expert count reset is folded into the kernel post-pass."""
        return

    def _ll_views(self):
        rx_dtype = torch.float4_e2m1fn_x2 if _is_fp4_dtype(self.dtype) else self.dtype
        rx_em_view = self.rx_em.view(rx_dtype).view(self.num_valid_max, self.row_view)
        scale_em_view = self.scale_em.view(torch.uint8).view(
            self.num_valid_max, max(1, self.scale_n_i32 * 4)
        )[:, : self.scale_bytes]
        scale_em_i32 = self.scale_em.view(self.num_valid_max, max(1, self.scale_n_i32))
        return {
            "rx_em": rx_em_view,
            "scale_em": scale_em_view,
            "scale_em_i32": scale_em_i32,
            "idx_em": self.idx_em,
            "wts_em": self.wts_em,
            "srcmap_em": self.srcmap_em,
            "sorted_expert_ids": self.sorted_expert_ids,
            "tile_row_base": self.tile_row_base,
            "num_valid": self.num_valid,
            "dedup_rx": None,
        }


class FlyDSLDispatchCombineIntraNodeOp:

    def __init__(self, config):
        self.cfg = config
        self._check_config()
        if config.tuning_table is None:
            self.load_tuning_config(config.tuning_config_path)
        self._dev = torch.device("cuda", config.rank)
        r = config.rank

        self._alloc_buffers()
        ms.shmem_barrier_all()

        npes = config.world_size
        _p2p_srcs = {
            "_p2p_tok_off": self.shmem_tok_off,
            "_p2p_tis": self.shmem_tok_id_to_src,
            "_p2p_out_wts": self.shmem_disp_out_wts,
            "_p2p_out_idx": self.shmem_disp_out_idx,
            "_p2p_out_tok": self.shmem_disp_out_tok,
            "_p2p_recv_num": self.shmem_recv_tok_num,
            "_p2p_out_scales": self.shmem_out_scales,
            "_p2p_comb_inp": self.shmem_comb_inp_tok,
            "_p2p_comb_inp_wts": self.shmem_comb_inp_wts,
            "_p2p_xdb_mem": self.shmem_xdev_bar_mem,
        }
        for _attr, _src in _p2p_srcs.items():
            setattr(self, _attr, build_p2p_table(_src, r, npes, self._dev))

        # Dispatch (encode) and combine (decode, Stage 3) must agree on this.
        self._effective_max_recv = config.effective_max_recv

        self._disp_jit_cache: dict[tuple[torch.dtype, int, int], Any] = {}
        self._disp_compiled_cache: dict[tuple[torch.dtype, int, int], Any] = {}
        self._comb_jit_cache: dict[
            tuple[torch.dtype, bool, bool, bool, int, int], Any
        ] = {}
        self._comb_compiled_cache: dict[
            tuple[torch.dtype, bool, bool, bool, int, int], Any
        ] = {}

        # Start at 1: a zero flag would satisfy the first wait and skip the sync.
        self._xdev_flag = torch.ones(1, dtype=torch.int64, device=self._dev)

        _fx_srcs = {
            "_fx_out_tok": self.shmem_disp_out_tok,
            "_fx_out_idx": self.shmem_disp_out_idx,
            "_fx_tok_off": self.shmem_tok_off,
            "_fx_recv_num": self.shmem_recv_tok_num,
            "_fx_dest_ctr": self.dest_pe_ctr,
            "_fx_disp_bar": self.disp_bar,
            "_fx_tok_map": self.dest_tok_map,
            "_fx_out_shmem_tok_id_to_src": self.shmem_tok_id_to_src,
            "_fx_out_total_recv": self.total_recv,
            "_fx_comb_inp": self.shmem_comb_inp_tok,
            "_fx_comb_out": self.shmem_comb_out_tok,
            "_fx_xdb_mem": self.shmem_xdev_bar_mem,
            "_fx_xdev_flag": self._xdev_flag,
            "_fx_comb_bar": self.comb_bar,
            "_fx_trecv": self.total_recv,
            "_fx_p2p_tok_off": self._p2p_tok_off,
            "_fx_p2p_out_tok_id_to_src": self._p2p_tis,
            "_fx_p2p_out_wts": self._p2p_out_wts,
            "_fx_p2p_out_idx": self._p2p_out_idx,
            "_fx_p2p_out_tok": self._p2p_out_tok,
            "_fx_p2p_recv_num": self._p2p_recv_num,
            "_fx_p2p_out_scales": self._p2p_out_scales,
            "_fx_out_scales": self.shmem_out_scales,
            "_fx_p2p_comb_inp": self._p2p_comb_inp,
            "_fx_p2p_comb_inp_wts": self._p2p_comb_inp_wts,
            "_fx_p2p_xdb_mem": self._p2p_xdb_mem,
            "_fx_comb_inp_wts": self.shmem_comb_inp_wts,
            "_fx_comb_out_wts": self.shmem_comb_out_wts,
            "_fx_packed_recv_count": self.packed_recv_count,
            "_fx_packed_recv_src_info": self.packed_recv_src_info,
            "_fx_disp_tok_map": self.disp_tok_to_ep_slot_map,
            "_fx_disp_grid_bar": self.disp_grid_bar,
            "_fx_disp_out_wts": self.shmem_disp_out_wts,
        }
        for _attr, _src in _fx_srcs.items():
            setattr(self, _attr, fx.Int64(_src.data_ptr()))

        self._gm = None
        if getattr(config, "enable_group_major", False):
            self._gm = FlyDSLDispatchGroupMajorOp(
                rank=config.rank,
                world_size=config.world_size,
                hidden_dim=config.hidden_dim,
                max_tok_per_rank=config.max_num_inp_token_per_rank,
                experts_per_rank=config.num_experts_per_rank,
                topk=config.num_experts_per_token,
                data_type=config.dispatch_dtype,
                unit_size=config.gm_unit_size,
                scale_dim=config.scale_dim,
                scale_type_size=config.scale_type_size,
                compact=config.gm_compact,
            )
            # Fused dispatch and combine share one receive-count buffer.
            self._gm.total_recv = self.total_recv
            ms.shmem_barrier_all()

        self._fx_tis = self._fx_out_shmem_tok_id_to_src
        ms.shmem_barrier_all()

    def load_tuning_config(self, path=None):
        """Build and attach the geometry tuning table for this op's shape."""
        self.cfg.tuning_table = build_geometry_tuning_table_for_config(self.cfg, path)
        return self.cfg.tuning_table

    def _alloc_buffers(self):
        cfg = self.cfg
        npes = cfg.world_size
        k = cfg.num_experts_per_token
        mt = cfg.max_num_inp_token_per_rank
        mr = cfg.effective_max_recv
        mr_worst = cfg.max_recv

        # Size combine input for both sender-major and fused dest_lid*topk layouts.
        mr_worst_inp = max(mr_worst, mt * k)

        legacy_tb = (
            cfg.hidden_dim * cfg.max_token_type_size
            if cfg.max_token_type_size > 0
            else 0
        )
        disp_tb = max(cfg.dispatch_token_bytes, legacy_tb)
        comb_tb = max(cfg.combine_token_bytes, legacy_tb)
        tok_i16_mr = (mr * disp_tb + 1) // 2  # dispatch output (dispatch_dtype)
        tok_i16_mr_worst = (
            mr_worst_inp * comb_tb + 1
        ) // 2  # combine input, worst-case (combine_dtype)
        tok_i16_mt = (mt * comb_tb + 1) // 2  # combine output (combine_dtype)

        self.shmem_disp_out_tok = mori_shmem_create_tensor((tok_i16_mr,), torch.int16)
        self.shmem_disp_out_wts = mori_shmem_create_tensor((mr * k,), torch.float32)
        self.shmem_disp_out_idx = mori_shmem_create_tensor((mr * k,), torch.int32)
        scale_total = mr * cfg.scale_bytes if cfg.scale_bytes > 0 else 1
        self.shmem_out_scales = mori_shmem_create_tensor((scale_total,), torch.int8)
        self.shmem_tok_off = mori_shmem_create_tensor((1,), torch.int32)
        self.shmem_recv_tok_num = mori_shmem_create_tensor((npes,), torch.int32)
        self.shmem_tok_id_to_src = mori_shmem_create_tensor((mr,), torch.int32)
        self.shmem_comb_inp_tok = mori_shmem_create_tensor(
            (tok_i16_mr_worst,), torch.int16
        )
        self.shmem_comb_inp_wts = mori_shmem_create_tensor(
            (mr_worst * k,), torch.float32
        )
        self.shmem_comb_out_tok = mori_shmem_create_tensor((tok_i16_mt,), torch.int16)
        self.shmem_comb_out_wts = mori_shmem_create_tensor((mt * k,), torch.float32)
        self.shmem_xdev_bar_mem = mori_shmem_create_tensor((npes,), torch.int64)

        # shmem_malloc is uninitialized; zero what combine reads.
        self.shmem_tok_id_to_src.zero_()
        self.shmem_comb_inp_tok.zero_()
        self.shmem_comb_inp_wts.zero_()
        self.shmem_xdev_bar_mem.zero_()

        self.dest_pe_ctr = torch.zeros(npes, dtype=torch.int32, device=self._dev)
        self.disp_bar = torch.zeros(1, dtype=torch.int32, device=self._dev)
        self.comb_bar = torch.zeros(1, dtype=torch.int32, device=self._dev)
        self.total_recv = torch.zeros(1, dtype=torch.int32, device=self._dev)
        sentinel = cfg.world_size * mr
        self.dest_tok_map = torch.full(
            (mt * k,), sentinel, dtype=torch.int32, device=self._dev
        )

        if cfg.enable_std_moe:
            epr = cfg.num_experts_per_rank
            max_tok_per_expert = cfg.max_recv
            self.packed_recv_count = torch.zeros(
                epr, dtype=torch.int32, device=self._dev
            )
            self.packed_recv_src_info = torch.zeros(
                epr * max_tok_per_expert, dtype=torch.int32, device=self._dev
            )
            self.disp_tok_to_ep_slot_map = torch.full(
                (mr * k,), -1, dtype=torch.int64, device=self._dev
            )
            self.disp_grid_bar = torch.zeros(1, dtype=torch.int64, device=self._dev)
        else:
            self.packed_recv_count = torch.zeros(1, dtype=torch.int32, device=self._dev)
            self.packed_recv_src_info = torch.zeros(
                1, dtype=torch.int32, device=self._dev
            )
            self.disp_tok_to_ep_slot_map = torch.zeros(
                1, dtype=torch.int64, device=self._dev
            )
            self.disp_grid_bar = torch.zeros(1, dtype=torch.int64, device=self._dev)

    def _check_config(self):
        """Static check of ``self.cfg``; runs before any GPU alloc."""
        cfg = self.cfg

        if not isinstance(cfg.rank, int) or cfg.rank < 0:
            raise ValueError(f"rank must be a non-negative int, got {cfg.rank!r}")
        if not isinstance(cfg.world_size, int) or cfg.world_size <= 0:
            raise ValueError(
                f"world_size must be a positive int, got {cfg.world_size!r}"
            )
        if cfg.rank >= cfg.world_size:
            raise ValueError(f"rank({cfg.rank}) must be < world_size({cfg.world_size})")
        if cfg.world_size > _MAX_INTRANODE_NPES:
            raise ValueError(
                f"world_size={cfg.world_size} exceeds intranode limit "
                f"_MAX_INTRANODE_NPES={_MAX_INTRANODE_NPES} (single-node GPU count); "
                "use an inter-node dispatch/combine op for world_size > 8"
            )

        if cfg.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {cfg.hidden_dim}")
        if cfg.max_num_inp_token_per_rank <= 0:
            raise ValueError(
                f"max_num_inp_token_per_rank must be positive, got {cfg.max_num_inp_token_per_rank}"
            )
        if cfg.num_experts_per_rank <= 0:
            raise ValueError(
                f"num_experts_per_rank must be positive, got {cfg.num_experts_per_rank}"
            )
        if cfg.num_experts_per_token <= 0:
            raise ValueError(
                f"num_experts_per_token must be positive, got {cfg.num_experts_per_token}"
            )
        if cfg.max_token_type_size < 0:
            raise ValueError(
                f"max_token_type_size must be non-negative, got {cfg.max_token_type_size}"
            )
        # k <= 64: ballot only covers 64 warp lanes.
        if cfg.num_experts_per_token > 64:
            raise ValueError(
                f"num_experts_per_token={cfg.num_experts_per_token} exceeds the warp-lane budget (64)"
            )

        if cfg.dispatch_dtype not in _SUPPORTED_TOK_DTYPES:
            raise ValueError(
                f"dispatch_dtype={cfg.dispatch_dtype} not supported. Supported: {_SUPPORTED_TOK_DTYPES}"
            )
        if cfg.combine_dtype not in _SUPPORTED_TOK_DTYPES:
            raise ValueError(
                f"combine_dtype={cfg.combine_dtype} not supported. Supported: {_SUPPORTED_TOK_DTYPES}"
            )
        if cfg.combine_quant_type not in _SUPPORTED_QUANT_TYPES:
            raise ValueError(
                f"combine_quant_type={cfg.combine_quant_type!r} not supported. Supported: {_SUPPORTED_QUANT_TYPES}"
            )
        if cfg.stage2_p2p_quant not in _SUPPORTED_STAGE2_P2P_QUANT_TYPES:
            raise ValueError(
                f"stage2_p2p_quant={cfg.stage2_p2p_quant!r} not supported. "
                f"Supported: {_SUPPORTED_STAGE2_P2P_QUANT_TYPES}"
            )
        if cfg.stage2_p2p_quant != "none" and cfg.combine_dtype != torch.bfloat16:
            raise ValueError("stage2_p2p_quant requires combine_dtype=torch.bfloat16")
        if cfg.stage2_p2p_quant != "none" and cfg.combine_quant_type != "none":
            raise ValueError(
                "stage2_p2p_quant and combine_quant_type are mutually exclusive"
            )
        if (
            cfg.combine_quant_type == "fp8_direct_cast"
            and cfg.combine_dtype != torch.bfloat16
        ):
            raise ValueError(
                f"combine_quant_type='fp8_direct_cast' requires combine_dtype=bfloat16 "
                f"(external dtype), got {cfg.combine_dtype}"
            )

        if cfg.combine_quant_type == "fp8_direct_cast" and cfg.enable_std_moe:
            raise NotImplementedError(
                "combine_quant_type='fp8_direct_cast' is not supported with "
                "enable_std_moe=True (asymmetric I/O dtypes not yet wired)"
            )

        if cfg.combine_quant_type == "fp8_direct_cast" and cfg.zero_copy:
            raise ValueError(
                "combine_quant_type='fp8_direct_cast' is incompatible with "
                "zero_copy=True (use_external_inp_buf=False): zero_copy "
                "skips Stage 1 where the bf16->fp8 cast lives."
            )

        for _tag, _tb, _dt in (
            ("dispatch", cfg.dispatch_token_bytes, cfg.dispatch_dtype),
            ("combine", cfg.combine_token_bytes, cfg.combine_dtype),
        ):
            if _tb % _TOK_BYTES_ALIGN != 0:
                raise ValueError(
                    f"{_tag} token row bytes ({_tb}) must be a multiple of "
                    f"{_TOK_BYTES_ALIGN} for v4i32 vector loads; check hidden_dim "
                    f"({cfg.hidden_dim}) and {_tag}_dtype ({_dt})"
                )

        # max_total_recv_tokens: 0 disables; >0 needs every rank >= 1 slot.
        if cfg.max_total_recv_tokens < 0:
            raise ValueError(
                f"max_total_recv_tokens must be non-negative, got {cfg.max_total_recv_tokens}"
            )
        if cfg.max_total_recv_tokens > 0:
            lo = cfg.world_size
            hi = cfg.world_size * cfg.max_num_inp_token_per_rank
            if cfg.max_total_recv_tokens < lo:
                raise ValueError(
                    f"max_total_recv_tokens={cfg.max_total_recv_tokens} < "
                    f"world_size={lo}; every rank must receive at least one slot"
                )
            if cfg.max_total_recv_tokens > hi:
                import warnings

                warnings.warn(
                    f"max_total_recv_tokens={cfg.max_total_recv_tokens} exceeds the "
                    f"worst case {hi} (= world_size * max_num_inp_token_per_rank); "
                    f"clamping to {hi}.  effective_max_recv_per_rank will be "
                    f"{cfg.max_num_inp_token_per_rank} (M).",
                    stacklevel=2,
                )

        if cfg.scale_dim < 0 or cfg.scale_type_size < 0:
            raise ValueError(
                f"scale_dim/scale_type_size must be non-negative, got "
                f"({cfg.scale_dim}, {cfg.scale_type_size})"
            )
        if (cfg.scale_dim == 0) != (cfg.scale_type_size == 0):
            raise ValueError(
                "scale_dim and scale_type_size must be both zero or both "
                f"positive, got ({cfg.scale_dim}, {cfg.scale_type_size})"
            )

        # expert_id (= dest_pe * experts_per_rank + local) must fit i32.
        total_experts = cfg.world_size * cfg.num_experts_per_rank
        if total_experts > (1 << 31) - 1:
            raise ValueError(
                f"total experts ({cfg.world_size} * {cfg.num_experts_per_rank} = {total_experts}) "
                "exceeds int32 range"
            )

        self._check_lds_capacity()

    def _check_lds_capacity(self):
        """Reject configs whose combine-kernel LDS overflows the GPU."""
        from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP

        cfg = self.cfg

        # Mirror make_combine_jit's LDS layout: two 8B-aligned i64[npes] tables.
        def _align(p, a):
            return (p + a - 1) // a * a

        ptr = 0
        ptr = _align(ptr, 8) + cfg.world_size * 8
        ptr = _align(ptr, 8) + cfg.world_size * 8
        lds_bytes = max(_align(ptr, 128), 128)

        arch = get_rocm_arch()
        limit = SMEM_CAPACITY_MAP.get(arch)
        if limit is not None and lds_bytes > limit:
            raise RuntimeError(
                f"combine kernel LDS needs {lds_bytes} B "
                f"(2 x i64[world_size={cfg.world_size}] P2P tables + 128 B arena), "
                f"but device {arch} provides only {limit} B"
            )

    def _check_tensor_device(self, name, t):
        if not torch.is_tensor(t):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t)}")
        if not t.is_cuda:
            raise ValueError(f"{name} must live on CUDA, got device={t.device}")
        if t.device.index != self.cfg.rank:
            raise ValueError(
                f"{name}.device={t.device} does not match cfg.rank={self.cfg.rank} "
                f"(expected cuda:{self.cfg.rank})"
            )

    def _check_dispatch_inputs(self, input, weights, scales, indices, packed_recv_x):
        cfg = self.cfg
        self._check_tensor_device("input", input)
        self._check_tensor_device("weights", weights)
        self._check_tensor_device("indices", indices)

        if input.dim() != 2:
            raise ValueError(
                f"input must be 2-D (cur_tok, hidden_dim), got shape {tuple(input.shape)}"
            )
        cur_tok = input.shape[0]
        if cur_tok > cfg.max_num_inp_token_per_rank:
            raise ValueError(
                f"input rows={cur_tok} exceeds cfg.max_num_inp_token_per_rank="
                f"{cfg.max_num_inp_token_per_rank} (would OOB-write into shmem)"
            )
        # Shared buffers are sized for the configured dispatch dtype.
        if input.dtype != cfg.dispatch_dtype:
            raise ValueError(
                f"dispatch input.dtype={input.dtype} != cfg.dispatch_dtype="
                f"{cfg.dispatch_dtype}; buffers are sized for the pinned dispatch dtype"
            )
        expected_hdim = cfg.dispatch_token_view_dim
        if input.shape[1] != expected_hdim:
            raise ValueError(
                f"input.shape[1]={input.shape[1]} != expected {expected_hdim} "
                f"(hidden_dim={cfg.hidden_dim}, dispatch_dtype={cfg.dispatch_dtype})"
            )

        if weights.dim() != 2:
            raise ValueError(
                f"weights must be 2-D (cur_tok, k), got shape {tuple(weights.shape)}"
            )
        if weights.shape != (cur_tok, cfg.num_experts_per_token):
            raise ValueError(
                f"weights.shape={tuple(weights.shape)} != expected "
                f"({cur_tok}, {cfg.num_experts_per_token})"
            )
        if weights.dtype != torch.float32:
            raise ValueError(f"weights.dtype={weights.dtype} must be torch.float32")

        if indices.dim() != 2:
            raise ValueError(
                f"indices must be 2-D (cur_tok, k), got shape {tuple(indices.shape)}"
            )
        if indices.shape != (cur_tok, cfg.num_experts_per_token):
            raise ValueError(
                f"indices.shape={tuple(indices.shape)} != expected "
                f"({cur_tok}, {cfg.num_experts_per_token})"
            )
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"indices.dtype={indices.dtype} must be int32 or int64")

        # scales all-or-none: tensor + scale_dim>0 + scale_type_size>0, or none.
        scales_configured = cfg.scale_bytes > 0
        if (scales is not None) != scales_configured:
            raise ValueError(
                "dispatch scales all-or-none contract violated: "
                f"scales={'provided' if scales is not None else 'None'} but "
                f"cfg.scale_dim={cfg.scale_dim} / cfg.scale_type_size={cfg.scale_type_size} "
                f"{'enable' if scales_configured else 'disable'} the scales path. "
                "Pass all three (scales tensor + cfg.scale_dim>0 + cfg.scale_type_size>0) "
                "or none of them."
            )
        if scales is not None:
            self._check_tensor_device("scales", scales)
            if scales.dim() != 2:
                raise ValueError(f"scales must be 2-D, got shape {tuple(scales.shape)}")
            row_bytes = scales.shape[1] * scales.element_size()
            if scales.shape[0] != cur_tok or row_bytes != cfg.scale_bytes:
                raise ValueError(
                    f"scales row-bytes={row_bytes} (shape={tuple(scales.shape)}, "
                    f"elem={scales.element_size()}B) does not match cfg.scale_bytes="
                    f"{cfg.scale_bytes}; expected ({cur_tok}, ...) totalling "
                    f"{cfg.scale_bytes}B per row"
                )

        if packed_recv_x is not None:
            self._check_tensor_device("packed_recv_x", packed_recv_x)
            if not cfg.enable_std_moe:
                raise ValueError(
                    "packed_recv_x is only consumed when cfg.enable_std_moe=True"
                )
            expected_rows = cfg.num_experts_per_rank * cfg.max_recv
            if packed_recv_x.shape[0] != expected_rows:
                raise ValueError(
                    f"packed_recv_x.shape[0]={packed_recv_x.shape[0]} != "
                    f"num_experts_per_rank * max_recv = {expected_rows}"
                )

    def _check_combine_inputs(
        self, input, weights, indices, packed_recv_x, strict_input_dtype: bool = True
    ):
        cfg = self.cfg
        self._check_tensor_device("input", input)

        if input.dim() != 2:
            raise ValueError(
                f"combine input must be 2-D, got shape {tuple(input.shape)}"
            )
        # The fused skip-stage1 path treats input as a placeholder.
        if strict_input_dtype and input.dtype != cfg.combine_dtype:
            raise ValueError(
                f"combine input.dtype={input.dtype} != cfg.combine_dtype="
                f"{cfg.combine_dtype}; buffers are sized for the pinned combine dtype"
            )
        view_dtype = input.dtype if strict_input_dtype else cfg.combine_dtype
        expected_hdim = _token_view_dim_for(view_dtype, cfg.hidden_dim)
        if input.shape[1] != expected_hdim:
            raise ValueError(
                f"combine input.shape[1]={input.shape[1]} != expected "
                f"{expected_hdim} (hidden_dim={cfg.hidden_dim}, dtype={view_dtype})"
            )
        if input.shape[0] > cfg.max_recv:
            raise ValueError(
                f"combine input rows={input.shape[0]} exceeds max_recv={cfg.max_recv}"
            )

        if weights is not None:
            self._check_tensor_device("weights", weights)
            if weights.dim() != 2 or weights.shape[1] != cfg.num_experts_per_token:
                raise ValueError(
                    f"combine weights must be (max_recv, {cfg.num_experts_per_token}), "
                    f"got shape {tuple(weights.shape)}"
                )
            if weights.dtype != torch.float32:
                raise ValueError(
                    f"combine weights.dtype={weights.dtype} must be torch.float32"
                )

        if indices is not None:
            self._check_tensor_device("indices", indices)
            if indices.dim() != 2 or indices.shape[1] != cfg.num_experts_per_token:
                raise ValueError(
                    f"combine indices must be (max_recv, {cfg.num_experts_per_token}), "
                    f"got shape {tuple(indices.shape)}"
                )
            if indices.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    f"combine indices.dtype={indices.dtype} must be int32/int64"
                )

        if packed_recv_x is not None:
            self._check_tensor_device("packed_recv_x", packed_recv_x)
            if not cfg.enable_std_moe:
                raise ValueError(
                    "packed_recv_x is only consumed when cfg.enable_std_moe=True"
                )

    def _get_dispatch_jit(self, d_dtype, block_num, warp_num_per_block):
        key = (d_dtype, block_num, warp_num_per_block)
        if key not in self._disp_jit_cache:
            cfg = self.cfg
            self._disp_jit_cache[key] = make_dispatch_jit(
                rank=cfg.rank,
                npes=cfg.world_size,
                experts_per_rank=cfg.num_experts_per_rank,
                experts_per_token=cfg.num_experts_per_token,
                hidden_dim=cfg.hidden_dim,
                max_tok_per_rank=cfg.max_num_inp_token_per_rank,
                block_num=block_num,
                warp_num_per_block=warp_num_per_block,
                data_type=d_dtype,
                scale_dim=cfg.scale_dim,
                scale_type_size=cfg.scale_type_size,
                enable_std_moe=cfg.enable_std_moe,
                max_recv=self._effective_max_recv,
            )
        return self._disp_jit_cache[key]

    def dispatch(
        self,
        input,
        weights,
        scales,
        indices,
        packed_recv_x=None,
    ):
        """Intranode dispatch. Launch geometry is resolved from cfg:
        cfg.dispatch_block_num/warp (user-pinned) > tuning table > default."""
        self._check_dispatch_inputs(input, weights, scales, indices, packed_recv_x)
        cfg = self.cfg
        d_dtype = input.dtype
        inp_cur_tok = input.shape[0]
        bn, wpb = _resolve_launch_geometry(
            "dispatch",
            cfg.dispatch_block_num,
            cfg.dispatch_warp_num_per_block,
            cfg.tuning_table,
            inp_cur_tok,
            _DEFAULT_DISPATCH_BLOCK_NUM,
            _DEFAULT_DISPATCH_WARP_NUM,
        )
        _check_block_num_resident("dispatch", bn)
        disp_key = (d_dtype, bn, wpb)
        self._last_inp_cur_tok = inp_cur_tok
        stream = torch.cuda.current_stream()
        inp_c = input if input.is_contiguous() else input.contiguous()
        wts_c = weights if weights.is_contiguous() else weights.contiguous()
        idx_c = (
            indices
            if (indices.dtype == torch.int32 and indices.is_contiguous())
            else indices.to(torch.int32).contiguous()
        )

        sc_ptr = scales.data_ptr() if scales is not None else 0
        prx_ptr = packed_recv_x.data_ptr() if packed_recv_x is not None else 0

        if cfg.enable_std_moe:
            self.packed_recv_count.zero_()

        _std_args = (
            self._fx_out_tok,
            self._fx_out_idx,
            self._fx_out_shmem_tok_id_to_src,
            fx.Int64(prx_ptr),
            self._fx_packed_recv_count if cfg.enable_std_moe else fx.Int64(0),
            self._fx_packed_recv_src_info,
            self._fx_disp_tok_map,
            self._fx_disp_grid_bar,
        )

        disp_fn = self._get_dispatch_jit(d_dtype, bn, wpb)
        disp_compiled = self._disp_compiled_cache.get(disp_key)
        if disp_compiled is None:
            args = (
                fx.Int64(inp_c.data_ptr()),
                fx.Int64(idx_c.data_ptr()),
                fx.Int64(wts_c.data_ptr()),
                self._fx_tok_map,
                self._fx_tok_off,
                self._fx_dest_ctr,
                self._fx_disp_bar,
                self._fx_recv_num,
                self._fx_out_total_recv,
                self._fx_p2p_tok_off,
                self._fx_p2p_out_tok,
                self._fx_p2p_out_tok_id_to_src,
                self._fx_p2p_out_idx,
                self._fx_p2p_out_wts,
                self._fx_p2p_recv_num,
                fx.Int64(sc_ptr),
                self._fx_p2p_out_scales,
                *_std_args,
                inp_cur_tok,
                stream,
            )
            disp_compiled = flyc.compile(disp_fn, *args)
            self._disp_compiled_cache[disp_key] = disp_compiled
        else:
            disp_compiled(
                inp_c.data_ptr(),
                idx_c.data_ptr(),
                wts_c.data_ptr(),
                self._fx_tok_map,
                self._fx_tok_off,
                self._fx_dest_ctr,
                self._fx_disp_bar,
                self._fx_recv_num,
                self._fx_out_total_recv,
                self._fx_p2p_tok_off,
                self._fx_p2p_out_tok,
                self._fx_p2p_out_tok_id_to_src,
                self._fx_p2p_out_idx,
                self._fx_p2p_out_wts,
                self._fx_p2p_recv_num,
                sc_ptr,
                self._fx_p2p_out_scales,
                *_std_args,
                inp_cur_tok,
                stream,
            )

        mr = cfg.effective_max_recv
        k = cfg.num_experts_per_token

        out_token_bytes = _token_bytes_for(d_dtype, cfg.hidden_dim)
        out_view_dim = _token_view_dim_for(d_dtype, cfg.hidden_dim)
        out_tok = (
            self.shmem_disp_out_tok.view(torch.int8)[: mr * out_token_bytes]
            .view(d_dtype)
            .view(mr, out_view_dim)
        )
        out_wts = self.shmem_disp_out_wts.view(mr, k)
        out_idx = self.shmem_disp_out_idx.view(mr, k)
        out_scales = None
        if cfg.scale_bytes > 0:
            out_scales = (
                self.shmem_out_scales[: mr * cfg.scale_bytes]
                .view(scales.dtype)
                .view(mr, scales.shape[1])
            )

        result = (out_tok, out_wts, out_scales, out_idx, self.total_recv)
        if cfg.enable_std_moe:
            epr = cfg.num_experts_per_rank
            result = result + (
                self.packed_recv_count[:epr],
                self.packed_recv_src_info,
            )
        return result

    def _resolve_cur_tok(self, cur_tok, who):
        """cur_tok explicit or the last dispatch()'s input count; validated in range."""
        if cur_tok is None:
            cur_tok = getattr(self, "_last_inp_cur_tok", None)
            if cur_tok is None:
                raise ValueError(
                    f"{who} requires an explicit cur_tok or a preceding dispatch() on this op "
                    "(cur_tok defaults to the dispatch input.shape[0])."
                )
        mt = self.cfg.max_num_inp_token_per_rank
        if cur_tok < 0 or cur_tok > mt:
            raise ValueError(
                f"cur_tok={cur_tok} out of range [0, max_num_inp_token_per_rank={mt}]"
            )
        return cur_tok

    def _run_combine_kernel(
        self, cache, key, fn, inp_ptr, wts_ptr, prx_ptr, cur_tok, stream
    ):
        """Compile once and reuse the cached combine launcher."""
        fixed = (
            self._fx_comb_inp,
            self._fx_comb_out,
            self._fx_xdb_mem,
            self._fx_xdev_flag,
            self._fx_tok_map,
            self._fx_comb_bar,
            self._fx_trecv,
            self._fx_out_shmem_tok_id_to_src,
            self._fx_p2p_comb_inp,
            self._fx_p2p_xdb_mem,
        )
        tail = (self._fx_comb_inp_wts, self._fx_comb_out_wts, self._fx_p2p_comb_inp_wts)
        std = (self._fx_disp_tok_map, self._fx_disp_out_wts)
        compiled = cache.get(key)
        if compiled is None:
            cache[key] = flyc.compile(
                fn,
                fx.Int64(inp_ptr),
                *fixed,
                fx.Int64(wts_ptr),
                *tail,
                fx.Int64(prx_ptr),
                *std,
                cur_tok,
                stream,
            )
        else:
            compiled(
                inp_ptr,
                *fixed,
                wts_ptr,
                *tail,
                prx_ptr,
                *std,
                cur_tok,
                stream,
            )

    def _launch_combine(
        self,
        input,
        weights,
        indices,
        packed_recv_x,
        cur_tok,
        enable_weights,
        skip_stage1,
        stage2_p2p_quant=None,
    ):
        """Launch regular or skip-stage1 combine."""
        cfg = self.cfg
        stream = torch.cuda.current_stream()
        # skip_stage1 treats input as a placeholder -> relax the dtype check.
        self._check_combine_inputs(
            input, weights, indices, packed_recv_x, strict_input_dtype=not skip_stage1
        )

        # fp8_direct_cast fires only when cfg asks AND launch dtype is bf16.
        fp8_dc = (
            cfg.combine_quant_type == "fp8_direct_cast"
            and input.dtype == torch.bfloat16
        )
        p2p_quant = (
            cfg.stage2_p2p_quant if stage2_p2p_quant is None else stage2_p2p_quant
        )
        if p2p_quant not in _SUPPORTED_STAGE2_P2P_QUANT_TYPES:
            raise ValueError(
                f"stage2_p2p_quant={p2p_quant!r} not supported. "
                f"Supported: {_SUPPORTED_STAGE2_P2P_QUANT_TYPES}"
            )
        blockwise_fp8 = skip_stage1 and p2p_quant == "fp8_blockwise_1x32"
        if skip_stage1:
            # placeholder input: pre-cast to fp8 so the kernel dtype + out view match.
            if fp8_dc and input.dtype != torch.float8_e4m3fn:
                inp_c = input.to(torch.float8_e4m3fn).contiguous()
            else:
                inp_c = input if input.is_contiguous() else input.contiguous()
            c_dtype = torch.float8_e4m3fn if fp8_dc else input.dtype
        else:
            # Zero-copy contract: peers read shmem_comb_inp_tok; any other pointer is a bug.
            if cfg.zero_copy and input.data_ptr() != self.shmem_comb_inp_tok.data_ptr():
                raise ValueError(
                    "zero_copy mode requires the caller to write into the buffer "
                    "returned by op.get_registered_combine_input_buffer(combine_dtype). "
                    f"Got input.data_ptr()={input.data_ptr():#x} but "
                    f"shmem_comb_inp_tok.data_ptr()={self.shmem_comb_inp_tok.data_ptr():#x}."
                )
            inp_c = input if input.is_contiguous() else input.contiguous()
            c_dtype = input.dtype

        _cur_tok = self._resolve_cur_tok(
            cur_tok, "combine_no_stage1()" if skip_stage1 else "combine()"
        )

        # Resolve geometry on cur_tok, not input.shape[0] (ws*M under zero_copy).
        bn, wpb = _resolve_launch_geometry(
            "combine",
            cfg.combine_block_num,
            cfg.combine_warp_num_per_block,
            cfg.tuning_table,
            _cur_tok,
            _DEFAULT_COMBINE_BLOCK_NUM,
            _DEFAULT_COMBINE_WARP_NUM,
        )
        _check_block_num_resident("combine", bn)

        wts_ptr = (
            self.shmem_disp_out_wts.data_ptr()
            if weights is None
            else weights.data_ptr()
        )

        _prx_ref = None
        if fp8_dc and packed_recv_x is not None:
            # std-MoE expert-major buffer is bf16 upstream but Stage 1 reads fp8.
            _prx_ref = (
                packed_recv_x.view(torch.bfloat16).to(torch.float8_e4m3fn).contiguous()
            )
            prx_ptr = _prx_ref.data_ptr()
        else:
            prx_ptr = packed_recv_x.data_ptr() if packed_recv_x is not None else 0

        key = (
            c_dtype,
            bool(cfg.zero_copy),
            bool(enable_weights),
            bool(fp8_dc),
            bool(blockwise_fp8),
            bn,
            wpb,
            bool(skip_stage1),
        )
        fn = self._comb_jit_cache.get(key)
        if fn is None:
            fn = make_combine_jit(
                rank=cfg.rank,
                npes=cfg.world_size,
                experts_per_token=cfg.num_experts_per_token,
                hidden_dim=cfg.hidden_dim,
                max_tok_per_rank=cfg.max_num_inp_token_per_rank,
                block_num=bn,
                warp_num_per_block=wpb,
                data_type=c_dtype,
                enable_weights=bool(enable_weights),
                enable_std_moe=cfg.enable_std_moe,
                zero_copy=cfg.zero_copy,
                skip_stage1=bool(skip_stage1),
                fp8_direct_cast=bool(fp8_dc),
                blockwise_fp8_transport=bool(blockwise_fp8),
                # Must match dispatch's encoding stride so tok_map decode lines up.
                max_recv=self._effective_max_recv,
            )
            self._comb_jit_cache[key] = fn
        self._run_combine_kernel(
            self._comb_compiled_cache,
            key,
            fn,
            inp_c.data_ptr(),
            wts_ptr,
            prx_ptr,
            _cur_tok,
            stream,
        )

        mt = cfg.max_num_inp_token_per_rank
        k = cfg.num_experts_per_token
        out_token_bytes = _token_bytes_for(c_dtype, cfg.hidden_dim)
        out_view_dim = _token_view_dim_for(c_dtype, cfg.hidden_dim)
        out_tok = (
            self.shmem_comb_out_tok.view(torch.int8)[: mt * out_token_bytes]
            .view(c_dtype)
            .view(mt, out_view_dim)
        )
        out_wts = self.shmem_comb_out_wts.view(mt, k)
        return out_tok, out_wts

    def combine(
        self,
        input,
        weights,
        indices,
        packed_recv_x=None,
        cur_tok=None,
    ):
        """Run intranode combine."""
        return self._launch_combine(
            input,
            weights,
            indices,
            packed_recv_x,
            cur_tok,
            enable_weights=True,
            skip_stage1=False,
        )

    # Gated reserved API: only the fused GEMM2+combine path is contract-safe.
    _ENABLE_COMBINE_NO_STAGE1 = False

    def combine_no_stage1(
        self,
        input,
        weights,
        indices,
        packed_recv_x=None,
        cur_tok=None,
        enable_weights: bool = True,
        stage2_p2p_quant=None,
    ):
        """Run combine after fused GEMM2 has populated the P2P input."""
        if not type(self)._ENABLE_COMBINE_NO_STAGE1:
            raise NotImplementedError(
                "combine_no_stage1 is reserved for the fused GEMM2+combine "
                "path. Use combine(...) instead, or set "
                "FlyDSLDispatchCombineIntraNodeOp._ENABLE_COMBINE_NO_STAGE1=True "
                "after auditing the upstream IPC-ordering contract."
            )
        return self._launch_combine(
            input,
            weights,
            indices,
            packed_recv_x,
            cur_tok,
            enable_weights=enable_weights,
            skip_stage1=True,
            stage2_p2p_quant=stage2_p2p_quant,
        )

    def get_dispatch_src_token_pos(self):
        torch.cuda.synchronize()
        n = int(self.total_recv[0].item())
        return self.shmem_tok_id_to_src[:n].clone()

    def get_registered_combine_input_buffer(self, dtype=None, hidden_dim=-1):
        """Return the registered combine input with the requested view."""
        cfg = self.cfg
        dt = dtype if dtype is not None else cfg.combine_dtype
        if dt != cfg.combine_dtype:
            raise ValueError(
                f"get_registered_combine_input_buffer: dtype={dt} != cfg.combine_dtype="
                f"{cfg.combine_dtype}; buffer is sized for the pinned combine dtype"
            )
        h = hidden_dim if hidden_dim > 0 else _token_view_dim_for(dt, cfg.hidden_dim)
        return self.shmem_comb_inp_tok.view(torch.int8).view(dt).view(-1, h)
