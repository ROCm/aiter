from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr

from aiter.ops.flydsl.moe_common import GateMode


def make_mixed_moe_gemm1_launcher(
    *,
    moe_gemm1,
    cache_tag,
    allocator_pong,
    allocator_ping,
    gate_only: bool,
    mock_gate_only: bool,
    gate_up_interleave: bool,
    inter_dim: int,
    inter_dim_pad: int,
    tile_k: int,
    tile_n: int,
    total_threads: int,
    persist_m: int,
    k_batch: int,
):
    @flyc.jit
    def launch_mixed_moe_gemm1(
        arg_out: fx.Pointer,
        arg_x: fx.Pointer,
        arg_w: fx.Pointer,
        arg_scale_x: fx.Pointer,
        arg_scale_w: fx.Pointer,
        arg_sorted_token_ids: fx.Pointer,
        arg_expert_ids: fx.Pointer,
        arg_sorted_weights: fx.Pointer,
        arg_max_token_ids: fx.Pointer,
        arg_bias: fx.Pointer,
        arg_out_scale_sorted: fx.Pointer,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()

        inter_dim_pad_total = fx.Index(2 * inter_dim_pad)
        tile2_pad = 0
        if const_expr(not gate_only):
            tile_k_stage2 = tile_k // 2
            tile2_pad = (
                tile_k_stage2 - (inter_dim - inter_dim_pad) % tile_k_stage2
            ) % tile_k_stage2

        inter_in = fx.Index(i32_inter_in.ir_value())
        tile_n_index = fx.Index(tile_n)
        if const_expr(mock_gate_only or gate_up_interleave):
            gx = (
                inter_in - inter_dim_pad_total + tile2_pad + tile_n_index - 1
            ) // tile_n_index
        else:
            gx = (
                (inter_in - inter_dim_pad_total + tile2_pad + 2 * tile_n_index - 1)
                // tile_n_index
                // 2
            )

        c_pm_l = fx.Index(persist_m)
        gy = (fx.Index(i32_size_expert_ids_in.ir_value()) + c_pm_l - 1) // c_pm_l

        moe_gemm1(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_max_token_ids,
            arg_bias,
            arg_out_scale_sorted,
            i32_tokens_in,
            i32_inter_in,
            i32_k_in,
            i32_size_expert_ids_in,
        ).launch(grid=(gx, gy, k_batch), block=(total_threads, 1, 1), stream=stream)

    return launch_mixed_moe_gemm1


def make_mixed_moe_gemm2_launcher(
    *,
    moe_gemm2,
    cache_tag,
    allocator_pong,
    allocator_ping,
    persistent: bool,
    cu_num: int,
    model_dim_pad: int,
    tile_n: int,
    persist_m: int,
):
    @flyc.jit
    def launch_mixed_moe_gemm2(
        arg_out: fx.Pointer,
        arg_x: fx.Pointer,
        arg_w: fx.Pointer,
        arg_scale_x: fx.Pointer,
        arg_scale_w: fx.Pointer,
        arg_sorted_token_ids: fx.Pointer,
        arg_expert_ids: fx.Pointer,
        arg_sorted_weights: fx.Pointer,
        arg_num_valid_ids: fx.Pointer,
        arg_bias: fx.Pointer,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()

        n_in = fx.Index(i32_n_in.ir_value())
        tile_n_idx = fx.Index(tile_n)
        model_dim_pad_idx = fx.Index(model_dim_pad)
        gx = (n_in - model_dim_pad_idx + tile_n_idx - 1) // tile_n_idx
        if const_expr(persistent):
            gy = fx.Index(cu_num)
        else:
            c_pm_l = fx.Index(persist_m)
            gy = (fx.Index(i32_size_expert_ids_in.ir_value()) + c_pm_l - 1) // c_pm_l

        moe_gemm2(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_num_valid_ids,
            arg_bias,
            i32_tokens_in,
            i32_n_in,
            i32_k_in,
            i32_size_expert_ids_in,
        ).launch(
            grid=(gx, gy, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_mixed_moe_gemm2


@functools.lru_cache(maxsize=None)
def compile_mixed_moe_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "f16",
    act: str = "silu",
    use_cshuffle_epilog: bool | None = None,
    enable_bias: bool = False,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    persist_m: int = 1,
    use_async_copy: bool = False,
    waves_per_eu: int = 4,
    k_batch: int = 1,
    b_nt: int = 0,
    gate_mode: GateMode = GateMode.SEPARATED,
    a_scale_one: bool = False,
    xcd_swizzle: int = 0,
    swiglu_limit: float = 0.0,
):
    from .mixed_moe_gemm_2stage import build_mixed_moe_gemm1_kernel

    return build_mixed_moe_gemm1_kernel(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=doweight_stage1,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        act=act,
        use_cshuffle_epilog=use_cshuffle_epilog,
        enable_bias=enable_bias,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        persist_m=persist_m,
        use_async_copy=use_async_copy,
        waves_per_eu=waves_per_eu,
        k_batch=k_batch,
        b_nt=b_nt,
        gate_mode=gate_mode,
        a_scale_one=a_scale_one,
        xcd_swizzle=xcd_swizzle,
        swiglu_limit=swiglu_limit,
    )


@functools.lru_cache(maxsize=None)
def compile_mixed_moe_gemm2(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    accumulate: bool = True,
    enable_bias: bool = False,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    persist_m: int = 4,
    sort_block_m: int = 0,
    b_nt: int = 2,
    xcd_swizzle: int = 0,
):
    from .mixed_moe_gemm_2stage import build_mixed_moe_gemm2_kernel

    return build_mixed_moe_gemm2_kernel(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=doweight_stage2,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        use_cshuffle_epilog=use_cshuffle_epilog,
        accumulate=accumulate,
        enable_bias=enable_bias,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        persist_m=persist_m,
        sort_block_m=sort_block_m,
        b_nt=b_nt,
        xcd_swizzle=xcd_swizzle,
    )
