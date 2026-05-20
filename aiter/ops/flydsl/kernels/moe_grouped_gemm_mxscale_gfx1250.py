# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Grouped/masked MoE MXScale GEMM helpers for gfx1250.

Initial A8W4 grouped support reuses the tuned gemm_fp8fp4_gfx1250
compile_a8w4_gemm schedule per expert.  The wrapper keeps the grouped/masked
calling convention while the underlying A8W4 GEMM owns TDM/WMMA_SCALE codegen.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.gemm_fp8fp4_gfx1250 import compile_a8w4_gemm, compile_mxfp4_gemm
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled


@dataclass(frozen=True)
class _GroupedA8W4Config:
    model_dim: int
    inter_dim: int
    experts: int
    max_m: int
    tile_m: int
    tile_n: int
    tile_k: int
    m_warp: int
    n_warp: int
    num_buffers: int
    waves_per_eu: Optional[int]
    out_dtype: str
    use_tdm_store: bool
    inst_prefetch: bool
    wave_specialized_tdm: bool
    split_k: int
    cluster_m: int
    cluster_n: int
    use_scale_opsel: bool
    expert_sched_mode: bool
    grouped_persistent_m: bool = True
    persistent_workers: Optional[int] = None
    data_format: str = "a8w4"
    act: str = "silu"


def _validate_common(cfg: _GroupedA8W4Config) -> None:
    if cfg.out_dtype not in ("f16", "bf16"):
        raise ValueError(f"out_dtype must be 'f16' or 'bf16', got {cfg.out_dtype!r}")
    if cfg.num_buffers not in (2, 3, 4):
        raise ValueError(f"num_buffers must be 2, 3 or 4, got {cfg.num_buffers}")
    if cfg.data_format not in ("a8w4", "fp4"):
        raise ValueError(f"data_format must be 'a8w4' or 'fp4', got {cfg.data_format!r}")
    if cfg.model_dim % 32 != 0:
        raise ValueError(f"model_dim must be divisible by 32 for MXScale scales, got {cfg.model_dim}")
    if cfg.inter_dim % 32 != 0:
        raise ValueError(f"inter_dim must be divisible by 32 for MXScale scales, got {cfg.inter_dim}")
    if cfg.tile_k % 128 != 0:
        raise ValueError(f"tile_k must be a multiple of 128 for MXScale WMMA_SCALE, got {cfg.tile_k}")
    if cfg.split_k < 1:
        raise ValueError(f"split_k must be >= 1, got {cfg.split_k}")
    if cfg.act not in ("silu", "swiglu"):
        raise ValueError(f"act must be 'silu' or 'swiglu', got {cfg.act!r}")
    if cfg.grouped_persistent_m and (cfg.cluster_m != 1 or cfg.cluster_n != 1):
        raise ValueError("grouped_persistent_m currently requires cluster_m=cluster_n=1")


def _to_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _make_m_tile_prefix(masked_m: torch.Tensor, cfg: _GroupedA8W4Config) -> torch.Tensor:
    valid_m = masked_m[:cfg.experts].to(dtype=torch.int32)
    valid_m = valid_m.clamp(min=0, max=cfg.max_m)
    valid_tiles = torch.div(
        valid_m + (cfg.tile_m - 1),
        cfg.tile_m,
        rounding_mode="floor",
    )
    prefix = torch.empty((cfg.experts + 1,), device=masked_m.device, dtype=torch.int32)
    prefix[0].zero_()
    torch.cumsum(valid_tiles, dim=0, out=prefix[1:])
    return prefix


def _make_m_tile_map(masked_m: torch.Tensor, cfg: _GroupedA8W4Config) -> torch.Tensor:
    valid_m = masked_m[:cfg.experts].to(dtype=torch.int32)
    valid_m = valid_m.clamp(min=0, max=cfg.max_m)
    valid_tiles = torch.div(
        valid_m + (cfg.tile_m - 1),
        cfg.tile_m,
        rounding_mode="floor",
    ).cpu().tolist()
    max_m_tiles = (cfg.max_m + cfg.tile_m - 1) // cfg.tile_m
    packed = [
        expert * max_m_tiles + local_tile
        for expert, count in enumerate(valid_tiles)
        for local_tile in range(int(count))
    ]
    if not packed:
        packed = [0]
    return torch.tensor(packed, device=masked_m.device, dtype=torch.int32)


def _check_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    if tensor.dim() != rank:
        raise ValueError(f"{name} must be rank-{rank}, got shape={tuple(tensor.shape)}")


def _pack_factors(cfg: _GroupedA8W4Config) -> tuple[int, int]:
    if cfg.data_format == "fp4":
        return 2, 2
    return 1, 2


def _preshuffled_scale_shape(rows: int, k_dim: int, warp_tile: int, tile_k: int) -> tuple[int, int]:
    # Matches tests.kernels.test_gemm_fp8fp4_gfx1250.preshuffle_e8m0_scale.
    k_scale = int(k_dim) // 32
    scale_k_per_tile = int(tile_k) // 32
    if k_scale % scale_k_per_tile != 0:
        raise ValueError(f"K scale columns must be divisible by tile_k/32, got {k_scale} and {scale_k_per_tile}")
    wmma_rep = int(warp_tile) // 16
    if wmma_rep < 1:
        raise ValueError(f"warp_tile must be >= 16, got {warp_tile}")
    if int(rows) % wmma_rep != 0:
        raise ValueError(f"scale rows must be divisible by wmma_rep={wmma_rep}, got {rows}")
    return int(rows) // wmma_rep, k_scale * wmma_rep


def _check_stage1_args(y, x, w, scale_x, scale_w, masked_m, cfg: _GroupedA8W4Config) -> None:
    _check_rank("y", y, 3)
    _check_rank("x", x, 3)
    _check_rank("w", w, 3)
    _check_rank("scale_x", scale_x, 3)
    _check_rank("scale_w", scale_w, 3)
    if tuple(y.shape) != (cfg.experts, cfg.max_m, cfg.inter_dim):
        raise ValueError(f"y shape must be {(cfg.experts, cfg.max_m, cfg.inter_dim)}, got {tuple(y.shape)}")
    pack_a, pack_b = _pack_factors(cfg)
    if tuple(x.shape) != (cfg.experts, cfg.max_m, cfg.model_dim // pack_a):
        raise ValueError(f"x shape must be {(cfg.experts, cfg.max_m, cfg.model_dim // pack_a)}, got {tuple(x.shape)}")
    if tuple(w.shape) != (cfg.experts, 2 * cfg.inter_dim, cfg.model_dim // pack_b):
        raise ValueError(f"w shape must be {(cfg.experts, 2 * cfg.inter_dim, cfg.model_dim // pack_b)}, got {tuple(w.shape)}")
    warp_tile_m = cfg.tile_m // cfg.m_warp
    warp_tile_n = cfg.tile_n // cfg.n_warp
    scale_x_shape = _preshuffled_scale_shape(cfg.max_m, cfg.model_dim, warp_tile_m, cfg.tile_k)
    scale_w_shape = _preshuffled_scale_shape(2 * cfg.inter_dim, cfg.model_dim, warp_tile_n, cfg.tile_k)
    if tuple(scale_x.shape) != (cfg.experts, *scale_x_shape):
        raise ValueError(f"scale_x shape must be {(cfg.experts, *scale_x_shape)}, got {tuple(scale_x.shape)}")
    if tuple(scale_w.shape) != (cfg.experts, *scale_w_shape):
        raise ValueError(f"scale_w shape must be {(cfg.experts, *scale_w_shape)}, got {tuple(scale_w.shape)}")
    if masked_m.numel() < cfg.experts:
        raise ValueError(f"masked_m must contain at least {cfg.experts} entries, got {masked_m.numel()}")


def _apply_gate_up(gate: torch.Tensor, up: torch.Tensor, act: str) -> torch.Tensor:
    if act == "swiglu":
        gate = gate.clamp(max=7.0)
        up = up.clamp(min=-7.0, max=7.0)
        return gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
    return torch.nn.functional.silu(gate) * up


@functools.lru_cache(maxsize=64)
def _compile_stage1_finalize_act(
    *,
    experts: int,
    max_m: int,
    inter_dim: int,
    out_dtype: str,
    act: str,
):
    if out_dtype not in ("f16", "bf16"):
        raise ValueError(f"stage1 finalize supports f16/bf16, got {out_dtype!r}")
    if act not in ("silu", "swiglu"):
        raise ValueError(f"stage1 finalize act must be silu/swiglu, got {act!r}")
    block_threads = 256
    total_elems = int(experts) * int(max_m) * int(inter_dim)
    tmp_stride_e = int(max_m) * int(2 * inter_dim)
    out_stride_e = int(max_m) * int(inter_dim)

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def stage1_finalize_act_kernel(
        arg_y: fx.Tensor,
        arg_tmp: fx.Tensor,
        arg_masked_m: fx.Tensor,
    ):
        elem_ty = T.bf16 if out_dtype == "bf16" else T.f16
        tx = arith.index_cast(T.index, _raw(gpu.thread_id("x")))
        bx = arith.index_cast(T.index, _raw(gpu.block_id("x")))
        linear = bx * arith.index(block_threads) + tx
        linear_i32 = arith.index_cast(T.i32, linear)
        in_range = arith.cmpi(
            arith.CmpIPredicate.ult,
            linear_i32,
            arith.constant(total_elems, type=T.i32),
        )

        y_rsrc = buffer_ops.create_buffer_resource(arg_y, max_size=True)
        tmp_rsrc = buffer_ops.create_buffer_resource(arg_tmp, max_size=True)
        masked_rsrc = buffer_ops.create_buffer_resource(arg_masked_m, max_size=True)

        if_elem = scf.IfOp(in_range, results_=[], has_else=False)
        with ir.InsertionPoint(if_elem.then_block):
            e = linear / arith.index(out_stride_e)
            rem0 = linear - e * arith.index(out_stride_e)
            row = rem0 / arith.index(inter_dim)
            col = rem0 - row * arith.index(inter_dim)

            valid_m = buffer_ops.buffer_load(
                masked_rsrc, arith.index_cast(T.i32, e),
                vec_width=1, dtype=T.i32)
            row_ok = arith.cmpi(
                arith.CmpIPredicate.slt,
                arith.index_cast(T.i32, row),
                valid_m,
            )
            if_row = scf.IfOp(row_ok, results_=[], has_else=False)
            with ir.InsertionPoint(if_row.then_block):
                tmp_base = e * arith.index(tmp_stride_e) \
                    + row * arith.index(2 * inter_dim) + col
                gate_h = buffer_ops.buffer_load(
                    tmp_rsrc, arith.index_cast(T.i32, tmp_base),
                    vec_width=1, dtype=elem_ty)
                up_h = buffer_ops.buffer_load(
                    tmp_rsrc,
                    arith.index_cast(T.i32, tmp_base + arith.index(inter_dim)),
                    vec_width=1, dtype=elem_ty)
                g = gate_h.extf(T.f32)
                u = up_h.extf(T.f32)
                one = arith.constant(1.0, type=T.f32)
                neg_log2e = arith.constant(-1.4426950408889634, type=T.f32)
                if const_expr(act == "swiglu"):
                    limit = arith.constant(7.0, type=T.f32)
                    neg_limit = arith.constant(-7.0, type=T.f32)
                    alpha = arith.constant(1.702, type=T.f32)
                    g = arith.minimumf(g, limit)
                    u = arith.maximumf(arith.minimumf(u, limit), neg_limit)
                    t = g * alpha * neg_log2e
                    emu = llvm.call_intrinsic(
                        T.f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                    sig = llvm.call_intrinsic(
                        T.f32, "llvm.amdgcn.rcp.f32", [one + emu], [], [])
                    out_f = g * sig * (u + one)
                else:
                    t = g * neg_log2e
                    emu = llvm.call_intrinsic(
                        T.f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                    sig = llvm.call_intrinsic(
                        T.f32, "llvm.amdgcn.rcp.f32", [one + emu], [], [])
                    out_f = g * sig * u
                out_h = arith.trunc_f(elem_ty, out_f)
                buffer_ops.buffer_store(out_h, y_rsrc, linear_i32)
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_stage1_finalize_act(
        arg_y: fx.Tensor,
        arg_tmp: fx.Tensor,
        arg_masked_m: fx.Tensor,
        stream: fx.Stream,
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass
        gx = (arith.index(total_elems) + arith.index(block_threads - 1)) \
            / arith.index(block_threads)
        launcher = stage1_finalize_act_kernel(arg_y, arg_tmp, arg_masked_m)
        launcher.launch(
            grid=(_raw(gx), 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_stage1_finalize_act


def _check_stage2_args(y, x, w, scale_x, scale_w, masked_m, cfg: _GroupedA8W4Config) -> None:
    _check_rank("y", y, 3)
    _check_rank("x", x, 3)
    _check_rank("w", w, 3)
    _check_rank("scale_x", scale_x, 3)
    _check_rank("scale_w", scale_w, 3)
    if tuple(y.shape) != (cfg.experts, cfg.max_m, cfg.model_dim):
        raise ValueError(f"y shape must be {(cfg.experts, cfg.max_m, cfg.model_dim)}, got {tuple(y.shape)}")
    pack_a, pack_b = _pack_factors(cfg)
    if tuple(x.shape) != (cfg.experts, cfg.max_m, cfg.inter_dim // pack_a):
        raise ValueError(f"x shape must be {(cfg.experts, cfg.max_m, cfg.inter_dim // pack_a)}, got {tuple(x.shape)}")
    if tuple(w.shape) != (cfg.experts, cfg.model_dim, cfg.inter_dim // pack_b):
        raise ValueError(f"w shape must be {(cfg.experts, cfg.model_dim, cfg.inter_dim // pack_b)}, got {tuple(w.shape)}")
    warp_tile_m = cfg.tile_m // cfg.m_warp
    warp_tile_n = cfg.tile_n // cfg.n_warp
    scale_x_shape = _preshuffled_scale_shape(cfg.max_m, cfg.inter_dim, warp_tile_m, cfg.tile_k)
    scale_w_shape = _preshuffled_scale_shape(cfg.model_dim, cfg.inter_dim, warp_tile_n, cfg.tile_k)
    if tuple(scale_x.shape) != (cfg.experts, *scale_x_shape):
        raise ValueError(f"scale_x shape must be {(cfg.experts, *scale_x_shape)}, got {tuple(scale_x.shape)}")
    if tuple(scale_w.shape) != (cfg.experts, *scale_w_shape):
        raise ValueError(f"scale_w shape must be {(cfg.experts, *scale_w_shape)}, got {tuple(scale_w.shape)}")
    if masked_m.numel() < cfg.experts:
        raise ValueError(f"masked_m must contain at least {cfg.experts} entries, got {masked_m.numel()}")

def _compile_base_a8w4_gemm(
    *,
    K: int,
    N: int,
    cfg: _GroupedA8W4Config,
    stage1_act: str | None = None,
):
    compiler = compile_mxfp4_gemm if cfg.data_format == "fp4" else compile_a8w4_gemm
    return compiler(
        M=cfg.max_m,
        N=N,
        K=K,
        tile_m=cfg.tile_m,
        tile_n=cfg.tile_n,
        tile_k=cfg.tile_k,
        m_warp=cfg.m_warp,
        n_warp=cfg.n_warp,
        num_buffers=cfg.num_buffers,
        waves_per_eu=cfg.waves_per_eu,
        out_dtype=cfg.out_dtype,
        use_tdm_store=cfg.use_tdm_store and cfg.split_k == 1 and stage1_act is None,
        inst_prefetch=cfg.inst_prefetch,
        wave_specialized_tdm=cfg.wave_specialized_tdm and stage1_act is None,
        split_k=cfg.split_k,
        cluster_m=cfg.cluster_m,
        cluster_n=cfg.cluster_n,
        use_scale_opsel=cfg.use_scale_opsel,
        expert_sched_mode=cfg.expert_sched_mode,
        batch_count=cfg.experts,
        grouped_masked_m=True,
        grouped_persistent_m=cfg.grouped_persistent_m,
        persistent_workers=cfg.persistent_workers,
        stage1_act=stage1_act,
    )


@functools.lru_cache(maxsize=128)
def compile_moe_grouped_gemm1_a8w4_masked(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    max_m: int,
    tile_m: int = 16,
    tile_n: int = 64,
    tile_k: int = 128,
    m_warp: int = 1,
    n_warp: int = 2,
    out_dtype: str = "f16",
    num_buffers: int = 2,
    waves_per_eu: int | None = None,
    use_tdm_store: bool = True,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    split_k: int = 1,
    cluster_m: int = 1,
    cluster_n: int = 1,
    use_scale_opsel: bool = False,
    expert_sched_mode: bool = True,
    grouped_persistent_m: bool = True,
    persistent_workers: int | None = None,
    act: str = "silu",
    data_format: str = "a8w4",
):
    cfg = _GroupedA8W4Config(
        model_dim=int(model_dim), inter_dim=int(inter_dim), experts=int(experts),
        max_m=int(max_m), tile_m=int(tile_m), tile_n=int(tile_n), tile_k=int(tile_k),
        m_warp=int(m_warp), n_warp=int(n_warp), num_buffers=int(num_buffers),
        waves_per_eu=waves_per_eu, out_dtype=str(out_dtype), use_tdm_store=bool(use_tdm_store),
        inst_prefetch=bool(inst_prefetch), wave_specialized_tdm=bool(wave_specialized_tdm),
        split_k=int(split_k),
        cluster_m=int(cluster_m), cluster_n=int(cluster_n), use_scale_opsel=bool(use_scale_opsel),
        expert_sched_mode=bool(expert_sched_mode), grouped_persistent_m=bool(grouped_persistent_m),
        persistent_workers=persistent_workers, data_format=str(data_format), act=str(act),
    )
    _validate_common(cfg)
    fused_base = None
    if cfg.split_k == 1:
        fused_base = _compile_base_a8w4_gemm(
            K=cfg.model_dim, N=cfg.inter_dim, cfg=cfg, stage1_act=cfg.act)
    raw_base = _compile_base_a8w4_gemm(
        K=cfg.model_dim, N=2 * cfg.inter_dim, cfg=cfg)
    finalize_act = _compile_stage1_finalize_act(
        experts=cfg.experts,
        max_m=cfg.max_m,
        inter_dim=cfg.inter_dim,
        out_dtype=cfg.out_dtype,
        act=cfg.act,
    )

    def launch(y, x, w, scale_x, scale_w, masked_m, max_m_arg, inter_dim_arg,
               model_dim_arg, experts_arg, *, stream=None, _gemm_events=None,
               _m_tile_prefix=None, _m_tile_map=None, _tmp=None, _skip_epilogue=False,
               _debug_tmp_sentinel=None, _debug_tmp_out=None):
        """If `_gemm_events=(start, end)` is given, those cuda.Events are
        recorded immediately before / after the GEMM kernel launch only, so the
        caller can measure pure GEMM device time excluding prefix-sum and
        gate*act epilogue.

        Diagnostic hooks (off by default):
          - `_debug_tmp_sentinel`: if set to a float, the intermediate `tmp`
            buffer is filled with that value BEFORE the GEMM launch. Cells
            still equal to the sentinel after the launch indicate GEMM did
            not write them.
          - `_debug_tmp_out`: a list-like; if provided, the post-GEMM `tmp`
            (before the silu(gate)*up epilogue) is appended so the caller can
            inspect raw GEMM output statistics.
        """
        if int(max_m_arg) != cfg.max_m or int(inter_dim_arg) != cfg.inter_dim or int(model_dim_arg) != cfg.model_dim or int(experts_arg) != cfg.experts:
            raise ValueError("runtime dimensions must match compile-time grouped A8W4 stage1 config")
        _check_stage1_args(y, x, w, scale_x, scale_w, masked_m, cfg)
        if stream is None:
            stream = torch.cuda.current_stream()
        use_fused_gemm = (
            fused_base is not None
            and _tmp is None
            and not _skip_epilogue
            and _debug_tmp_sentinel is None
            and _debug_tmp_out is None
        )
        tmp = _tmp
        if not use_fused_gemm:
            if tmp is None:
                tmp = torch.empty(
                    (cfg.experts, cfg.max_m, 2 * cfg.inter_dim),
                    device=y.device, dtype=y.dtype)
            if _debug_tmp_sentinel is not None:
                tmp.fill_(float(_debug_tmp_sentinel))
            if cfg.split_k > 1:
                tmp.zero_()
        if cfg.grouped_persistent_m:
            m_tile_prefix = _m_tile_prefix
            if m_tile_prefix is None:
                m_tile_prefix = _make_m_tile_prefix(masked_m, cfg)
            m_tile_map = _m_tile_map
            if m_tile_map is None:
                m_tile_map = _make_m_tile_map(masked_m, cfg)
            if _gemm_events is not None:
                _gemm_events[0].record(stream)
            if use_fused_gemm:
                _run_compiled(
                    fused_base, y, x, w, scale_x, scale_w,
                    masked_m, m_tile_prefix, m_tile_map,
                    cfg.max_m, cfg.inter_dim, stream)
            else:
                _run_compiled(
                    raw_base, tmp, x, w, scale_x, scale_w,
                    masked_m, m_tile_prefix, m_tile_map,
                    cfg.max_m, 2 * cfg.inter_dim, stream)
            if _gemm_events is not None:
                _gemm_events[1].record(stream)
        else:
            if _gemm_events is not None:
                _gemm_events[0].record(stream)
            if use_fused_gemm:
                _run_compiled(
                    fused_base, y, x, w, scale_x, scale_w, masked_m,
                    cfg.max_m, cfg.inter_dim, stream)
            else:
                _run_compiled(
                    raw_base, tmp, x, w, scale_x, scale_w, masked_m,
                    cfg.max_m, 2 * cfg.inter_dim, stream)
            if _gemm_events is not None:
                _gemm_events[1].record(stream)
        if use_fused_gemm:
            return y
        if _debug_tmp_out is not None:
            # Holding a detached reference is enough to keep the per-call tmp
            # buffer alive for diagnostics; no clone needed.
            _debug_tmp_out.append(tmp.detach())
        if _skip_epilogue:
            return tmp
        _run_compiled(finalize_act, y, tmp, masked_m, stream)
        return y

    return launch


@functools.lru_cache(maxsize=128)
def compile_moe_grouped_gemm2_a8w4_masked(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    max_m: int,
    tile_m: int = 16,
    tile_n: int = 64,
    tile_k: int = 128,
    m_warp: int = 1,
    n_warp: int = 2,
    out_dtype: str = "f16",
    num_buffers: int = 2,
    waves_per_eu: int | None = None,
    use_tdm_store: bool = True,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    split_k: int = 1,
    cluster_m: int = 1,
    cluster_n: int = 1,
    use_scale_opsel: bool = False,
    expert_sched_mode: bool = True,
    grouped_persistent_m: bool = True,
    persistent_workers: int | None = None,
    data_format: str = "a8w4",
):
    cfg = _GroupedA8W4Config(
        model_dim=int(model_dim), inter_dim=int(inter_dim), experts=int(experts),
        max_m=int(max_m), tile_m=int(tile_m), tile_n=int(tile_n), tile_k=int(tile_k),
        m_warp=int(m_warp), n_warp=int(n_warp), num_buffers=int(num_buffers),
        waves_per_eu=waves_per_eu, out_dtype=str(out_dtype), use_tdm_store=bool(use_tdm_store),
        inst_prefetch=bool(inst_prefetch), wave_specialized_tdm=bool(wave_specialized_tdm),
        split_k=int(split_k),
        cluster_m=int(cluster_m), cluster_n=int(cluster_n), use_scale_opsel=bool(use_scale_opsel),
        expert_sched_mode=bool(expert_sched_mode), grouped_persistent_m=bool(grouped_persistent_m),
        persistent_workers=persistent_workers, data_format=str(data_format),
    )
    _validate_common(cfg)
    base = _compile_base_a8w4_gemm(K=cfg.inter_dim, N=cfg.model_dim, cfg=cfg)

    def launch(y, x, w, scale_x, scale_w, masked_m, max_m_arg, model_dim_arg,
               inter_dim_arg, experts_arg, *, stream=None, _gemm_events=None,
               _m_tile_prefix=None, _m_tile_map=None):
        """If `_gemm_events=(start, end)` is given, record around the GEMM
        kernel launch only -- excludes prefix-sum prep work."""
        if int(max_m_arg) != cfg.max_m or int(model_dim_arg) != cfg.model_dim or int(inter_dim_arg) != cfg.inter_dim or int(experts_arg) != cfg.experts:
            raise ValueError("runtime dimensions must match compile-time grouped A8W4 stage2 config")
        _check_stage2_args(y, x, w, scale_x, scale_w, masked_m, cfg)
        if stream is None:
            stream = torch.cuda.current_stream()
        if cfg.split_k > 1:
            y.zero_()
        if cfg.grouped_persistent_m:
            m_tile_prefix = _m_tile_prefix
            if m_tile_prefix is None:
                m_tile_prefix = _make_m_tile_prefix(masked_m, cfg)
            m_tile_map = _m_tile_map
            if m_tile_map is None:
                m_tile_map = _make_m_tile_map(masked_m, cfg)
            if _gemm_events is not None:
                _gemm_events[0].record(stream)
            _run_compiled(
                base, y, x, w, scale_x, scale_w, masked_m, m_tile_prefix, m_tile_map,
                cfg.max_m, cfg.model_dim, stream)
            if _gemm_events is not None:
                _gemm_events[1].record(stream)
        else:
            if _gemm_events is not None:
                _gemm_events[0].record(stream)
            _run_compiled(
                base, y, x, w, scale_x, scale_w, masked_m,
                cfg.max_m, cfg.model_dim, stream)
            if _gemm_events is not None:
                _gemm_events[1].record(stream)
        return y

    return launch


def compile_moe_grouped_gemm1_mxfp4_masked(**kwargs):
    return compile_moe_grouped_gemm1_a8w4_masked(data_format="fp4", **kwargs)


def compile_moe_grouped_gemm2_mxfp4_masked(**kwargs):
    return compile_moe_grouped_gemm2_a8w4_masked(data_format="fp4", **kwargs)


__all__ = [
    "compile_moe_grouped_gemm1_a8w4_masked",
    "compile_moe_grouped_gemm2_a8w4_masked",
    "compile_moe_grouped_gemm1_mxfp4_masked",
    "compile_moe_grouped_gemm2_mxfp4_masked",
]
