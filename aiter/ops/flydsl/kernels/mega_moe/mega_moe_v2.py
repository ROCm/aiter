# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""MegaMoE v2 fused dispatch, GEMM1, GEMM2, and combine implementation."""

import functools
import os
from pathlib import Path

import mori.shmem as ms
import torch

import flydsl.expr as fx
from flydsl.autotune import Config

from ...utils import autotune_compat as autotune
from ...utils import do_bench_collective
from ..flydsl_dispatch_combine_intranode_op import (
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
)

from .dispatch import DISPATCH_TABLE_SIZE, DispatchSlot
from .quant import per_1x32_mx_quant

__all__ = ["MegaMoEV2"]
_JOINT_AUTOTUNE_SCHEMA = 7
_STAGE2_P2P_QUANT_TYPES = ("auto", "none", "fp8_blockwise_1x32")
_AUTO_FP8_MIN_TOKENS = 1024
_AUTOTUNE_CONFIG_DIR = Path(__file__).with_name("autotune_configs")
if _AUTOTUNE_CONFIG_DIR.is_dir():
    os.environ.setdefault("FLYDSL_AUTOTUNE_CONFIG_DIR", str(_AUTOTUNE_CONFIG_DIR))


def _resolve_stage2_p2p_quant(mode, run_tokens):
    if mode == "auto":
        return "fp8_blockwise_1x32" if run_tokens > _AUTO_FP8_MIN_TOKENS else "none"
    return mode


# fmt: off
def _run_joint_sbm_config(owner, x, scales, wts, topk_ids, stream, slice_output, *, SBM, tune_tokens,
    model_dim, inter_dim, experts, topk, rank, npes, max_tok, sbm_candidates, autotune_schema):
    del model_dim, inter_dim, experts, topk, rank, npes, max_tok, sbm_candidates, autotune_schema
    owner._run_fused_stage1(x, wts, scales, topk_ids, stream=stream, tile_m=SBM)
    owner._joint_output = owner._run_stage2(int(tune_tokens), stream, slice_output)


@functools.lru_cache(maxsize=None)
def _make_joint_sbm_autotuner(tile_m_values):
    configs = [Config(SBM=int(tile_m)) for tile_m in tile_m_values]
    key = [
        "tune_tokens", "model_dim", "inter_dim", "experts", "topk", "npes", "max_tok",
        "sbm_candidates", "autotune_schema",
    ]
    return autotune(
        configs=configs, key=key, warmup=2, rep=7, do_bench=do_bench_collective,
        default=configs[0], artifact_name="mega-moe-v2-joint-sbm", artifact_bundle=True,
    )(_run_joint_sbm_config)
# fmt: on


class MegaMoEV2:
    """Fused dispatch, GEMM1, GEMM2, and combine with one in-flight launch per instance."""

    # fmt: off
    def __init__(self, *, rank: int, world_size: int, model_dim: int, inter_dim: int, experts: int, topk: int,
        quant: str, w1: torch.Tensor, w1_scale: torch.Tensor, w2: torch.Tensor, w2_scale: torch.Tensor,
        max_tok_per_rank: int, tune_tokens: int | None = None, mega_scheme: str = "fixedslot",
        gemm2_tile_m: int = -1, gemm2_tile_n: int = -1, gemm2_tile_k: int = -1,
        enable_fused_stage1: bool = True, enable_fused_stage2: bool = True, gate_mode=None,
        stage1_dispatch_cu: int | None = None, stage1_grid_mult: int | None = None,
        stage1_tile_m_values: tuple[int, ...] | None = None, stage2_p2p_quant: str = "auto"):
    # fmt: on
        if not enable_fused_stage1 or not enable_fused_stage2:
            raise ValueError("MegaMoEV2 requires enable_fused_stage1=True and enable_fused_stage2=True")
        if quant != "a8w4":
            raise ValueError("MegaMoEV2 currently supports quant='a8w4' only")
        if stage2_p2p_quant not in _STAGE2_P2P_QUANT_TYPES:
            raise ValueError(
                "stage2_p2p_quant must be 'auto', 'none', or 'fp8_blockwise_1x32', "
                f"got {stage2_p2p_quant!r}"
            )
        if experts % world_size != 0:
            raise ValueError(f"experts={experts} must be divisible by world_size={world_size}")
        if max_tok_per_rank <= 0 or max_tok_per_rank & (max_tok_per_rank - 1):
            raise ValueError(f"max_tok_per_rank={max_tok_per_rank} must be a power of two")
        mode = getattr(gate_mode, "value", gate_mode)
        if mode not in (None, "interleave"):
            raise ValueError("MegaMoEV2 requires interleaved gate/up weights")
        del gemm2_tile_m, gemm2_tile_n, gemm2_tile_k
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.model_dim = int(model_dim)
        self.inter_dim = int(inter_dim)
        self.experts = int(experts)
        self.epr = int(experts // world_size)
        self.topk = int(topk)
        self.quant = quant
        self.mtpr = int(max_tok_per_rank)
        self.dev = torch.device("cuda", rank)
        self.max_recv = self.world_size * self.mtpr
        self.enable_fused_stage1 = True
        self.enable_fused_stage2 = True
        self.gate_mode = "interleave"
        self.mega_scheme = mega_scheme
        self._stage2_p2p_quant = stage2_p2p_quant
        self._v2_dispatch_cu = None if stage1_dispatch_cu is None else int(stage1_dispatch_cu)
        self._v2_grid_mult = None if stage1_grid_mult is None else int(stage1_grid_mult)
        if stage1_tile_m_values is None:
            token_capacity = int(max_tok_per_rank if tune_tokens is None else tune_tokens)
            stage1_tile_m_values = (32,) if token_capacity < 256 else (32, 64, 128)
        self._v2_tile_m_values = tuple(sorted({int(tile_m) for tile_m in stage1_tile_m_values}))
        if not self._v2_tile_m_values or any(tile_m not in (32, 64, 128) for tile_m in self._v2_tile_m_values):
            raise ValueError("stage1_tile_m_values must contain only 32, 64, and/or 128")
        capacity_tile_m = max(self._v2_tile_m_values)
        max_buffer_bytes = self.epr * self.world_size * self.mtpr * max(self.model_dim, self.inter_dim * 2)
        compact = self.max_recv >= 2048 or max_buffer_bytes >= 3_000_000_000 or self.epr > 64
        self._s1_fixed_slot = not compact
        self._s1_scale_dim = self.model_dim // 32
        # fmt: off
        self.comb_cfg = FlyDSLDispatchCombineConfig(rank=self.rank, world_size=self.world_size,
            hidden_dim=self.model_dim, max_num_inp_token_per_rank=self.mtpr, num_experts_per_rank=self.epr,
            num_experts_per_token=self.topk, combine_dtype=torch.bfloat16,
            dispatch_dtype=torch.float8_e4m3fn, scale_dim=self._s1_scale_dim, scale_type_size=1,
            enable_std_moe=False, enable_group_major=True, gm_unit_size=capacity_tile_m,
            gm_scheme=mega_scheme, gm_compact=compact, max_total_recv_tokens=self.world_size)
        # fmt: on
        self.comb_op = FlyDSLDispatchCombineIntraNodeOp(self.comb_cfg)
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        self.w2 = w2 if w2.is_contiguous() else w2.contiguous()
        self.w2_scale = w2_scale if w2_scale.is_contiguous() else w2_scale.contiguous()
        self._build_fused_stage1(w1, w1_scale)
        self._build_fused_stage2()
        self._joint_sbm_tuner = _make_joint_sbm_autotuner(self._v2_tile_m_values)

    def _build_fused_stage1(self, w1, w1_scale):
        from .mega_moe_stage1 import make_stage1_autotuner

        self.sort_block_m = min(self._v2_tile_m_values)
        self._s1_w1 = w1.contiguous().view(torch.uint8)
        self._s1_w1_scale = w1_scale.contiguous().view(torch.uint8)
        op = self.comb_op._gm
        assert op is not None, "combine op was built without enable_group_major"
        self._s1_op = op
        # Payload capacity follows the largest SBM; metadata covers the smallest candidate.
        metadata_blocks = (op.num_valid_max + self.sort_block_m - 1) // self.sort_block_m
        if metadata_blocks > op.max_blocks:
            op.max_blocks = metadata_blocks
            op.sorted_expert_ids = torch.zeros(metadata_blocks, dtype=torch.int32, device=self.dev)
            op.tile_row_base = torch.zeros(metadata_blocks, dtype=torch.int32, device=self.dev)
        self._s1_nvm = op.num_valid_max
        self._s1_cap = op.ll_cap
        self._s1_epoch_parity = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self._s1_epoch_expected = torch.zeros(2, dtype=torch.int32, device=self.dev)
        self._s1_num_cu = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self._allocate_dispatch_workspace(op)
        self._s1_mega = make_stage1_autotuner(self._v2_dispatch_cu, self._v2_grid_mult, self._v2_tile_m_values)
        self._s1_mega_by_tile = {
            tile_m: make_stage1_autotuner(self._v2_dispatch_cu, self._v2_grid_mult, (tile_m,))
            for tile_m in self._v2_tile_m_values
        }

        v = op._ll_views()
        self._s1_rx = v["rx_em"]
        self._s1_scale_i32 = v["scale_em_i32"]

        inter_dim = self.inter_dim
        a2rows = self._s1_nvm
        self._s1_out = torch.zeros((a2rows, inter_dim), dtype=torch.float8_e4m3fn, device=self.dev)
        prows = ((a2rows + 255) // 256) * 256
        pcols = (((inter_dim // 32) + 7) // 8) * 8
        self._s1_osd = torch.zeros(prows * pcols + inter_dim, dtype=torch.uint8, device=self.dev)
        self._build_v2_disp_table()

    def _allocate_dispatch_workspace(self, op):
        total_experts = self.world_size * self.epr
        workspace = {
            "local_hist": torch.zeros(total_experts, dtype=torch.int32, device=self.dev),
            "local_cursor": torch.zeros(total_experts, dtype=torch.int32, device=self.dev),
            "pair_order": torch.empty(self.mtpr * self.topk, dtype=torch.int32, device=self.dev),
            "pair_base": torch.empty(total_experts, dtype=torch.int32, device=self.dev),
            "pair_ready": torch.zeros(2, dtype=torch.int32, device=self.dev),
            "entry_count": torch.zeros(10, dtype=torch.int64, device=self.dev),
            "epoch_gate": torch.zeros(10, dtype=torch.int32, device=self.dev),
            "pair_order_ready": torch.zeros(2, dtype=torch.int32, device=self.dev),
            "work_head": torch.zeros(8 * 16, dtype=torch.int32, device=self.dev),
            "work_tail": torch.zeros(1, dtype=torch.int32, device=self.dev),
            "expert_tile_end": torch.empty(self.epr, dtype=torch.int32, device=self.dev),
            "active_experts": torch.empty(total_experts, dtype=torch.int32, device=self.dev),
            "active_count": torch.zeros(self.world_size, dtype=torch.int32, device=self.dev),
        }
        workspace["bigcnt"] = op._sym((self.world_size * self.epr,), torch.int32)
        workspace["count_done"] = op._sym((2 * self.world_size,), torch.int32)
        workspace["my_base"] = op._sym((total_experts,), torch.int32)
        workspace["plan_ready"] = op._sym((2 * self.world_size,), torch.int32)
        workspace["payload_ready"] = op._sym((2 * self.epr,), torch.int32)
        ms.shmem_barrier_all()
        workspace["p2p_bigcnt"] = op._p2p_table(workspace["bigcnt"])
        workspace["p2p_count_done"] = op._p2p_table(workspace["count_done"])
        workspace["p2p_my_base"] = op._p2p_table(workspace["my_base"])
        workspace["p2p_plan_ready"] = op._p2p_table(workspace["plan_ready"])
        workspace["p2p_payload_ready"] = op._p2p_table(workspace["payload_ready"])
        self._s1_dispatch_workspace = workspace

    def _build_v2_disp_table(self):
        op = self._s1_op
        workspace = self._s1_dispatch_workspace
        table = [0] * DISPATCH_TABLE_SIZE
        table[DispatchSlot.PAIR_BASE] = workspace["pair_base"].data_ptr()
        table[DispatchSlot.P2P_TOKEN] = op.p2p_rx_em.data_ptr()
        table[DispatchSlot.P2P_SCALE] = op.p2p_scale_em.data_ptr()
        table[DispatchSlot.P2P_WEIGHT] = op.p2p_wts_em.data_ptr()
        table[DispatchSlot.P2P_SRCMAP] = op.p2p_srcmap_em.data_ptr()
        table[DispatchSlot.SORTED_EXPERT] = op.sorted_expert_ids.data_ptr()
        table[DispatchSlot.TILE_ROW_BASE] = op.tile_row_base.data_ptr()
        table[DispatchSlot.NUM_VALID] = op.num_valid.data_ptr()
        table[DispatchSlot.SRCMAP] = op.srcmap_em.data_ptr()
        table[DispatchSlot.LOCAL_HIST] = workspace["local_hist"].data_ptr()
        table[DispatchSlot.COUNT_MATRIX] = workspace["bigcnt"].data_ptr()
        table[DispatchSlot.P2P_COUNT_MATRIX] = workspace["p2p_bigcnt"].data_ptr()
        table[DispatchSlot.COUNT_DONE] = workspace["count_done"].data_ptr()
        table[DispatchSlot.P2P_COUNT_DONE] = workspace["p2p_count_done"].data_ptr()
        table[DispatchSlot.TASK_ROW_BASE] = workspace["my_base"].data_ptr()
        table[DispatchSlot.LOCAL_CURSOR] = workspace["local_cursor"].data_ptr()
        table[DispatchSlot.P2P_PAYLOAD_READY] = workspace["p2p_payload_ready"].data_ptr()
        table[DispatchSlot.PAIR_ORDER] = workspace["pair_order"].data_ptr()
        table[DispatchSlot.P2P_TASK_ROW_BASE] = workspace["p2p_my_base"].data_ptr()
        table[DispatchSlot.P2P_PLAN_READY] = workspace["p2p_plan_ready"].data_ptr()
        table[DispatchSlot.PLAN_READY] = workspace["plan_ready"].data_ptr()
        table[DispatchSlot.PAIR_READY] = workspace["pair_ready"].data_ptr()
        table[DispatchSlot.ENTRY_COUNT] = workspace["entry_count"].data_ptr()
        table[DispatchSlot.EPOCH_GATE] = workspace["epoch_gate"].data_ptr()
        table[DispatchSlot.PAIR_ORDER_READY] = workspace["pair_order_ready"].data_ptr()
        table[DispatchSlot.WORK_HEAD] = workspace["work_head"].data_ptr()
        table[DispatchSlot.WORK_TAIL] = workspace["work_tail"].data_ptr()
        table[DispatchSlot.EXPERT_TILE_END] = workspace["expert_tile_end"].data_ptr()
        table[DispatchSlot.ACTIVE_EXPERTS] = workspace["active_experts"].data_ptr()
        table[DispatchSlot.ACTIVE_COUNT] = workspace["active_count"].data_ptr()
        table[DispatchSlot.RUNNING] = op.running.data_ptr()
        table[DispatchSlot.P2P_RUNNING] = op.p2p_running.data_ptr()
        self._s1_disp = torch.tensor(table, dtype=torch.int64, device=self.dev)

    def _run_fused_stage1(self, x, wts, scales, topk_ids, stream=None, tile_m=None):
        if stream is None:
            stream = fx.Stream(torch.cuda.current_stream())
        cur_tok = int(x.shape[0])
        if cur_tok > self.mtpr:
            raise ValueError(f"run_tokens={cur_tok} > max_tok_per_rank={self.mtpr}")
        if x.dtype != torch.float8_e4m3fn or not x.is_contiguous():
            raise ValueError("x must be contiguous float8_e4m3fn")
        if tuple(x.shape) != (cur_tok, self.model_dim):
            raise ValueError(f"x must have shape ({cur_tok}, {self.model_dim})")
        if wts.dtype != torch.float32 or not wts.is_contiguous():
            raise ValueError("wts must be contiguous float32")
        if tuple(wts.shape) != (cur_tok, self.topk):
            raise ValueError(f"wts must have shape ({cur_tok}, {self.topk})")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        if tuple(topk_ids.shape) != (cur_tok, self.topk):
            raise ValueError(f"topk_ids must have shape ({cur_tok}, {self.topk})")
        if not scales.is_contiguous():
            raise ValueError("scales must be contiguous")
        op = self._s1_op
        joint_config = self._joint_sbm_tuner.last_config
        if tile_m is None and joint_config is not None:
            tile_m = int(joint_config.kwargs["SBM"])
        tuner = self._s1_mega if tile_m is None else self._s1_mega_by_tile[int(tile_m)]
        tile_m_values = self._v2_tile_m_values if tile_m is None else (int(tile_m),)
        metadata_block_m = self.sort_block_m if tile_m is None else int(tile_m)
        # fmt: off
        tuner(self._s1_out, self._s1_rx, self._s1_w1, self._s1_scale_i32, self._s1_w1_scale,
            op.tile_row_base, op.sorted_expert_ids, op.num_valid, self._s1_osd, fx.Int32(self._s1_nvm),
            fx.Int64(self._s1_disp.data_ptr()), fx.Int32(cur_tok), fx.Int64(x.data_ptr()),
            fx.Int64(topk_ids.data_ptr()), fx.Int64(wts.data_ptr()), fx.Int64(scales.data_ptr()),
            fx.Int64(self._s1_epoch_parity.data_ptr()), fx.Int64(self._s1_epoch_expected.data_ptr()),
            stream, model_dim=self.model_dim, inter_dim=self.inter_dim,
            rank=self.rank, experts_per_rank=self.epr, fuse_npes=self.world_size, fuse_topk=self.topk,
            fuse_cap=self._s1_cap, fuse_mtpr=self.mtpr, fuse_scale_dim=self._s1_scale_dim,
            fixed_slot_dispatch=self._s1_fixed_slot, metadata_block_m=metadata_block_m,
            num_cu=self._s1_num_cu,
            tune_tokens=cur_tok, dispatch_constraint=-1 if self._v2_dispatch_cu is None else self._v2_dispatch_cu,
            grid_constraint=-1 if self._v2_grid_mult is None else self._v2_grid_mult,
            tile_m_constraint=",".join(str(v) for v in tile_m_values), autotune_schema=tuner.schema)
        # fmt: on
        self._s1_active_tile_m = int(tuner.last_config.kwargs["sort_block_m"])

    def quantize(self, x_bf16):
        return per_1x32_mx_quant(x_bf16, quant_mode="fp8")

    def _run_joint(self, x, scales, wts, topk_ids, run_tokens, stream, slice_output):
        # fmt: off
        self._joint_sbm_tuner(
            self, x, scales, wts, topk_ids, stream, slice_output, tune_tokens=int(run_tokens),
            model_dim=self.model_dim, inter_dim=self.inter_dim, experts=self.experts, topk=self.topk,
            rank=self.rank, npes=self.world_size, max_tok=self.mtpr,
            sbm_candidates=",".join(str(v) for v in self._v2_tile_m_values),
            autotune_schema=_JOINT_AUTOTUNE_SCHEMA)
        # fmt: on
        return self._joint_output

    def _run_stage2(self, run_tokens, stream, slice_output):
        ret = self._run_fused_stage2(run_tokens, stream)
        out_tok = ret[0] if isinstance(ret, (tuple, list)) else ret
        if out_tok is None:
            cfg = self.comb_cfg
            out_tok = (
                self.comb_op.shmem_comb_out_tok.view(torch.int8)[: self.mtpr * cfg.combine_token_bytes]
                .view(cfg.combine_dtype)
                .view(self.mtpr, cfg.combine_token_view_dim)
            )
        return out_tok[:run_tokens] if slice_output else out_tok

    def forward(self, x_bf16, wts, topk_ids, *, stream=None, slice_output=True):
        run_tokens = int(x_bf16.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        if x_bf16.dtype != torch.bfloat16 or not x_bf16.is_contiguous():
            raise ValueError("x_bf16 must be contiguous bfloat16")
        if wts.dtype != torch.float32 or not wts.is_contiguous():
            raise ValueError("wts must be contiguous float32")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        x_q, scales = self.quantize(x_bf16)
        return self._run_joint(x_q, scales, wts, topk_ids, run_tokens, stream, slice_output)

    def forward_prequant(self, x_q, scales, wts, topk_ids, *, stream=None, slice_output=True):
        run_tokens = int(x_q.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        return self._run_joint(x_q, scales, wts, topk_ids, run_tokens, stream, slice_output)

    forward_bf16 = forward
    __call__ = forward

    def _build_fused_stage2(self):
        from .mega_moe_stage2 import make_gemm2_autotuner

        FlyDSLDispatchCombineIntraNodeOp._ENABLE_COMBINE_NO_STAGE1 = True
        comb_cfg = self.comb_cfg
        dev = torch.device("cuda", comb_cfg.rank)
        k = comb_cfg.num_experts_per_token
        cu_num = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self._g2v2_inter = int(self.inter_dim)
        self._g2v2_hidden = int(comb_cfg.hidden_dim)
        quant_modes = (
            ("none", "fp8_blockwise_1x32")
            if self._stage2_p2p_quant == "auto"
            else (self._stage2_p2p_quant,)
        )
        self._g2_tuners = {}
        self._g2_invariants_by_quant = {}
        for p2p_quant in quant_modes:
            tuner = make_gemm2_autotuner(a_dtype="fp8", p2p_quant_type=p2p_quant)
            p2p_row_nbytes = (
                int(comb_cfg.hidden_dim) + int(comb_cfg.hidden_dim) // 32
                if p2p_quant == "fp8_blockwise_1x32"
                else int(comb_cfg.hidden_dim) * 2
            )
            self._g2_tuners[p2p_quant] = tuner
            self._g2_invariants_by_quant[p2p_quant] = {
                "model_dim": int(comb_cfg.hidden_dim), "inter_dim": int(self.inter_dim),
                "experts": int(comb_cfg.num_experts_per_rank), "topk": int(k), "rank": int(comb_cfg.rank),
                "npes": int(comb_cfg.world_size), "max_tok": int(comb_cfg.max_num_inp_token_per_rank),
                "recv_cap": int(self.max_recv),
                "comb_inp_nbytes": int(comb_cfg.max_num_inp_token_per_rank) * int(k) * p2p_row_nbytes,
                "HIDDEN_MAX": int(comb_cfg.hidden_dim), "INTER_MAX": int(self.inter_dim), "cu_num": int(cu_num),
                "autotune_schema": tuner.schema, "p2p_quant_type": p2p_quant,
            }
        initial_quant = _resolve_stage2_p2p_quant(self._stage2_p2p_quant, 0)
        self._g2_tuner = self._g2_tuners[initial_quant]
        self._g2_invariants = self._g2_invariants_by_quant[initial_quant]
        self._g2_combine_placeholder = torch.empty(
            1, comb_cfg.hidden_dim, dtype=comb_cfg.combine_dtype, device=dev
        )

    def _run_fused_stage2(self, run_tokens, stream=None):
        comb_op = self.comb_op
        op = self._s1_op
        active_tile_m = self._s1_active_tile_m
        if stream is None:
            stream = torch.cuda.current_stream()
        s_fx = fx.Stream(stream.cuda_stream)
        p2p_quant = _resolve_stage2_p2p_quant(self._stage2_p2p_quant, run_tokens)
        self._g2_tuner = self._g2_tuners[p2p_quant]
        self._g2_invariants = self._g2_invariants_by_quant[p2p_quant]
        # fmt: off
        self._g2_tuner(
            fx.Int64(self._s1_out.view(-1).data_ptr()), fx.Int64(self._s1_osd.data_ptr()),
            fx.Int64(self.w2.data_ptr()), fx.Int64(self.w2_scale.data_ptr()),
            fx.Int64(op.sorted_expert_ids.data_ptr()), fx.Int64(op.num_valid.data_ptr()),
            fx.Int64(op.srcmap_em.data_ptr()), fx.Int64(op.wts_em.data_ptr()),
            fx.Int64(op.tile_row_base.data_ptr()), comb_op._fx_p2p_comb_inp, self._s1_nvm,
            self._g2v2_inter, self._g2v2_hidden, s_fx,
            tune_tokens=int(run_tokens), SBM=active_tile_m, **self._g2_invariants)
        # fmt: on
        return comb_op.combine_no_stage1(
            self._g2_combine_placeholder, None, None, cur_tok=run_tokens, enable_weights=False,
            stage2_p2p_quant=p2p_quant,
        )
