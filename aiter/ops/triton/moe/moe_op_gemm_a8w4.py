# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs.py

import itertools
import torch
import triton
from aiter.ops.triton.moe.moe_routing.routing import RoutingData
from aiter.ops.triton._triton_kernels.moe.moe_op_gemm_a8w4 import (
    _moe_gemm_a8w4 as _moe_gemm_a8w4_triton,
    _reduce_grouped,
)
from aiter.ops.triton._gluon_kernels.moe.moe_op_gemm_a8w4 import (
    _moe_gemm_a8w4 as _moe_gemm_a8w4_gluon,
)
from aiter.ops.triton.moe.reduce import reduce_grouped
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.device_info import get_num_sms


def can_overflow_int32(tensor: torch.Tensor):
    max_int32 = (1 << 31) - 1
    offset = 0
    for i in range(tensor.ndim):
        offset += (tensor.shape[i] - 1) * tensor.stride(i)
    return offset > max_int32


def should_upcast_indices(*args):
    return any(tensor is not None and can_overflow_int32(tensor) for tensor in args)


def allocate_output(
    M,
    N,
    out_dtype,
    reduction_n_matmul,
    reduction_n_reduction,
    routing_data,
    gather_indx,
    scatter_indx,
    block_m,
    split_k,
    device,
):
    if routing_data.n_expts_act == 1 or scatter_indx is None:
        y_rows = M
    else:
        y_rows = (
            scatter_indx.shape[0] // routing_data.n_expts_act
        )  # compressed number of rows
    matmul_shape = (split_k, M, N // reduction_n_matmul)
    final_shape = (y_rows, N // reduction_n_matmul // reduction_n_reduction)
    matmul_output = torch.empty(matmul_shape, device=device, dtype=out_dtype)
    if scatter_indx is not None or split_k > 1:
        final_output = torch.empty(final_shape, device=device, dtype=out_dtype)
    else:
        final_output = None
    return matmul_output, final_output


def get_kernel_config_triton(m, n, k, routing_data):
    block_m = routing_data.block_m
    group_m = 4
    num_xcds = 8
    xcd_swizzle = num_xcds
    w_cache_modifier = ".cg" if block_m <= 32 else None
    arch = get_arch()
    num_stages = 1 if arch == "gfx950" else 2
    split_k = 1
    block_k = 256

    if block_m == 16:
        block_n = 128
        num_warps = 4

        grid_m = routing_data.n_blocks(m, block_m)
        grid_n = triton.cdiv(n, block_n)
        grid = grid_m * grid_n * split_k
        while block_n >= 64 and grid < get_num_sms():
            block_n = block_n // 2
            grid_m = routing_data.n_blocks(m, block_m)
            grid_n = triton.cdiv(n, block_n)
            grid = grid_m * grid_n * split_k

    elif block_m == 32:
        if n <= 1024:
            block_n = 128
            num_warps = 4
        elif n <= 4096:
            block_n = 256
            num_warps = 8
        else:
            block_n = 512
            num_warps = 8

    else:
        block_n = 512
        num_warps = 8

    ret = {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "group_m": group_m,
        "xcd_swizzle": xcd_swizzle,
        "w_cache_modifier": w_cache_modifier,
        "split_k": split_k,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    }
    return ret


def get_kernel_config_gluon(m, n, k, routing_data):
    block_m = routing_data.block_m
    num_xcds = 1
    w_cache_modifier = ".cg" if block_m <= 32 else None
    num_stages = 2
    split_k = 1
    block_k = 512
    num_buffers = 3

    if block_m == 16:
        block_n = 128
        num_warps = 4

    elif block_m == 32:
        if n <= 1024:
            block_n = 128
            num_warps = 4
        elif n <= 4096:
            block_n = 256
            num_warps = 4
        else:
            block_n = 256
            num_warps = 4

    else:
        block_n = 512
        num_warps = 4

    ret = {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "xcd_swizzle": num_xcds,
        "split_k": split_k,
        "w_cache_modifier": w_cache_modifier,
        "waves_per_eu": 0,
        "num_buffers": num_buffers,
    }
    return ret


def swizzle_scales_gfx950(data):
    NON_K_PRESHUFFLE_BLOCK_SIZE = 32
    block_shape = data.shape
    SCALE_K = block_shape[-2]
    N = block_shape[-1]
    data = data.transpose(-1, -2)
    data = data.view(-1, N // NON_K_PRESHUFFLE_BLOCK_SIZE, 2, 16, SCALE_K // 8, 2, 4, 1)
    data = data.permute(0, 1, 4, 6, 3, 5, 2, 7).contiguous()
    E = block_shape[0]
    data = data.reshape(E, N // 32, SCALE_K * 32)
    return data.transpose(-1, -2)


def swizzle_scales_gfx1250(data):
    E, K_SCALE, N = data.shape
    preshuffle_factor = 128
    num_chunk_n = N // preshuffle_factor
    SCALE_KWIDTH = 4 if K_SCALE >= 4 else K_SCALE
    num_chunk_k = K_SCALE // SCALE_KWIDTH

    data = data.transpose(-1, -2)
    data = data.view(
        E, num_chunk_n, 4, preshuffle_factor // 4, num_chunk_k, SCALE_KWIDTH
    )
    data = data.permute(0, 1, 4, 3, 2, 5).contiguous()
    data = data.view(E, N // preshuffle_factor, K_SCALE * preshuffle_factor)
    data = data.transpose(-1, -2)

    return data


# -----------------------------------------------------------------------------
# Triton Implementation
# -----------------------------------------------------------------------------


def moe_gemm_a8w4(
    x,
    w,
    x_scales,
    w_scales,
    x_static_scale=None,
    quant_static_scale=None,
    bias=None,
    routing_data: RoutingData | None = None,
    gather_indx=None,
    scatter_indx=None,
    gammas=None,
    swizzle_mx_scale=None,
    out_dtype=torch.bfloat16,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    add_residual=True,
    unpadded_N=None,
    unpadded_K=None,
):
    """
    Y[:, :] = 0.
    for e in num_experts:
        Y[idxs_y_m(e), :] += matmul(X[idxs_x_m(e), :], W[e, :, :])
    """
    use_gluon = get_arch() == "gfx1250"
    assert w.stride(-2) == 1, "`w` must be column-major when it has data-type mxfp"
    x_has_mx = x_scales is not None
    if x_has_mx:
        assert x.stride(-1) == 1, "'x' must be row-major when it has data-type mxfp"
    if x_has_mx:
        stride_x_mx_m = x_scales.stride(0)
        stride_x_mx_k = x_scales.stride(1)
    else:
        stride_x_mx_m = 0
        stride_x_mx_k = 0
    # determine shapes
    num_tokens = x.shape[-2]
    M = num_tokens if gather_indx is None else gather_indx.shape[0]
    K, N = x.shape[-1], w.shape[-1]
    block_m = routing_data.block_m
    if unpadded_N and block_m == 16:
        N = unpadded_N
    if unpadded_K and block_m == 16:
        K = unpadded_K
    if use_gluon:
        w = w.transpose(1, 2)
        w_scales = w_scales.transpose(1, 2)
    # compute optimization flags
    if use_gluon:
        config = get_kernel_config_gluon(M, N, K, routing_data)
    else:
        config = get_kernel_config_triton(M, N, K, routing_data)
    if apply_swiglu and config["split_k"] > 1:
        apply_swiglu_matmul = False
        reduction_n_matmul = 1
        apply_swiglu_reduction = True
        reduction_n_reduction = 2
    elif apply_swiglu:
        apply_swiglu_matmul = True
        reduction_n_matmul = 2
        apply_swiglu_reduction = False
        reduction_n_reduction = 1
    else:
        apply_swiglu_matmul = False
        reduction_n_matmul = 1
        apply_swiglu_reduction = False
        reduction_n_reduction = 1
    # allocate output memory
    y, y_final = allocate_output(
        M,
        N,
        out_dtype,
        reduction_n_matmul,
        reduction_n_reduction,
        routing_data,
        gather_indx,
        scatter_indx,
        config["block_m"],
        config["split_k"],
        x.device,
    )
    stride_bias = None if bias is None else bias.stride(0)
    # moe metadata
    expt_data = routing_data.expt_data
    expt_hist = None if expt_data is None else expt_data.hist
    expt_hist_sum = None if expt_data is None else expt_data.token_offs_pad[-1]
    expt_token_offs_raw = None if expt_data is None else expt_data.token_offs_raw
    expt_block_pid_map = None if expt_data is None else expt_data.block_pid_map
    # pid grid
    grid_m = routing_data.n_blocks(M, config["block_m"])
    grid_n = triton.cdiv(N, config["block_n"])
    grid = grid_m * grid_n * config["split_k"]
    # launch kernel
    if use_gluon:
        _moe_gemm_a8w4_gluon[(grid,)](
            y,
            # y.stride(0),
            y.stride(1),
            y.stride(2),
            x,
            x.stride(0),
            x.stride(1),
            x_scales,
            stride_x_mx_m,
            stride_x_mx_k,
            w,
            w.stride(0),
            w.stride(1),
            w.stride(2),
            w_scales,
            w_scales.stride(0),
            w_scales.stride(1),
            w_scales.stride(2),
            x_static_scale,
            quant_static_scale,
            bias,
            stride_bias,
            gammas,
            num_tokens,
            N,
            K,
            gather_indx,
            expt_hist,
            expt_token_offs_raw,
            expt_hist_sum,
            expt_block_pid_map,
            grid_m,
            grid_n,
            apply_swiglu_matmul,
            alpha,
            limit,
            reduction_n_matmul,
            routing_data.n_expts_act,
            config["block_m"],
            config["block_n"],
            config["block_k"],
            XCD_SWIZZLE=config["xcd_swizzle"],
            NUM_BUFFERS=config["num_buffers"] if config["num_buffers"] is not None else config["num_stages"],
            SWIZZLE_MX_SCALE=swizzle_mx_scale,
            EVEN_K=K % config["block_k"] == 0,
            MASK_K_LIMIT=K % config["block_k"],
            W_CACHE_MODIFIER=config["w_cache_modifier"],
            num_warps=config["num_warps"],
            UPCAST_INDICES=should_upcast_indices(x, w, y),
            waves_per_eu=config["waves_per_eu"],
        )
    else:
        _moe_gemm_a8w4_triton[(grid,)](
            y,
            y.stride(0),
            y.stride(1),
            y.stride(2),
            x,
            x.stride(0),
            x.stride(1),
            x_scales,
            stride_x_mx_m,
            stride_x_mx_k,
            w,
            w.stride(0),
            w.stride(1),
            w.stride(2),
            w_scales,
            w_scales.stride(0),
            w_scales.stride(1),
            w_scales.stride(2),
            x_static_scale,
            quant_static_scale,
            bias,
            stride_bias,
            gammas,
            N,
            K,
            gather_indx,
            expt_hist,
            expt_token_offs_raw,
            expt_hist_sum,
            expt_block_pid_map,
            grid_m,
            grid_n,
            apply_swiglu_matmul,
            alpha,
            limit,
            reduction_n_matmul,
            routing_data.n_expts_act,
            config["block_m"],
            config["block_n"],
            config["block_k"],
            config["group_m"],
            XCD_SWIZZLE=config["xcd_swizzle"],
            SWIZZLE_MX_SCALE=swizzle_mx_scale,
            SPLIT_K=config["split_k"],
            EVEN_K=K % config["block_k"] == 0,
            MASK_K_LIMIT=K % config["block_k"],
            W_CACHE_MODIFIER=config["w_cache_modifier"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
            UPCAST_INDICES=should_upcast_indices(x, w, y),
            waves_per_eu=config["waves_per_eu"],
            matrix_instr_nonkdim=config["matrix_instr_nonkdim"],
            kpack=config["kpack"],
        )

    # Build grouped reduction inputs in a uniform way
    group_indx = (
        None
        if scatter_indx is None
        else scatter_indx.view(-1, routing_data.n_expts_act)
    )
    y_final = reduce_grouped(
        y,
        group_indx,
        y_final,
        apply_swiglu_reduction,
        alpha,
        limit,
        reduction_n_reduction,
        out_dtype=out_dtype,
        add_residual=add_residual,
    )
    return y_final


# -----------------------------------------------------------------------------
# Reference Implementation
# -----------------------------------------------------------------------------


def swiglu_torch(a, alpha, limit, add_residual=True):
    a_gelu = a[..., ::2]
    if limit is not None:
        a_gelu = a_gelu.clamp(max=limit)
    a_linear = a[..., 1::2]
    if limit is not None:
        a_linear = a_linear.clamp(min=-limit, max=limit)

    out_gelu = a_gelu * torch.sigmoid(alpha * a_gelu)
    if add_residual:
        out = out_gelu * (a_linear + 1)
    else:
        out = out_gelu * a_linear
    return out


def moe_gemm_torch(
    x,
    w,
    bias,
    routing_data: RoutingData = None,
    gather_indx=None,
    scatter_indx=None,
    gammas=None,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    add_residual=True,
):
    assert x.dtype.itemsize > 1
    assert w.dtype.itemsize > 1
    if bias is not None and bias.ndim == 1:
        bias = bias.view(1, *bias.shape)
    if w.ndim == 2:
        w = w.view(1, *w.shape)
    n_expts_act = routing_data.n_expts_act
    # memory offsets
    if routing_data.n_expts_tot > 1:
        sizes = routing_data.expt_hist
        off = torch.zeros(sizes.shape[0] + 1, dtype=torch.int32)
        off[1:] = torch.cumsum(sizes, 0)
        offs = list(itertools.pairwise(off))
    else:
        offs = [[0, x.shape[0]] for _ in range(w.shape[0])]
    # compute
    n_rows = x.shape[0] if gather_indx is None else gather_indx.shape[0]
    n_cols = w.shape[-1] // 2 if apply_swiglu else w.shape[-1]
    y = torch.zeros((n_rows, n_cols), device=x.device, dtype=x.dtype)
    for i, (lo, hi) in enumerate(offs):
        if gather_indx is None:
            idx = torch.arange(lo, hi, device=x.device)
        else:
            idx = gather_indx[lo:hi] // n_expts_act
        out = torch.matmul(x[idx, :].float(), w[i].float())
        if bias is not None:
            out += bias[i, :]
        if apply_swiglu:
            out = swiglu_torch(out, alpha, limit, add_residual)
        if gammas is not None:
            out *= gammas[lo:hi, None]
        y[lo:hi, :] = out
    if scatter_indx is None:
        return y
    # accumulate output from all experts
    n_rows = y.shape[0] // n_expts_act
    out = torch.zeros((n_rows, y.shape[-1]), dtype=torch.float32, device=x.device)
    src_idx = scatter_indx.view(-1, n_expts_act)
    for i in range(n_rows):
        out[i, :] = y[src_idx[i], :].float().sum(0)

    return out


# -----------------------------------------------------------------------------
# Main Function for Testing
# -----------------------------------------------------------------------------


def main():
    import argparse
    from aiter.ops.triton.moe.moe_routing.routing import routing
    from aiter.ops.triton.moe.quant_moe import (
        downcast_to_static_fp8,
        downcast_to_mxfp,
        upcast_from_mxfp,
    )

    parser = argparse.ArgumentParser(description="Run MoE GEMM A8W4 test")
    parser.add_argument("--M", type=int, default=32)
    # parser.add_argument("--N", type=int, default=1024)
    # parser.add_argument("--K", type=int, default=1024)
    # parser.add_argument("--N", type=int, default=8192)
    # parser.add_argument("--K", type=int, default=2880)
    parser.add_argument("--N", type=int, default=5780)
    parser.add_argument("--K", type=int, default=2880)
    # parser.add_argument("--N", type=int, default=6144)
    # parser.add_argument("--K", type=int, default=3072)
    parser.add_argument("--E", type=int, default=1, help="Total experts")
    parser.add_argument("--n_expts_act", type=int, default=1, help="Active experts per token")
    parser.add_argument(
        "--do_gather", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--do_scatter", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--has_y_gammas", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--apply_swiglu", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fused_quant", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--hbm_swizzling", action=argparse.BooleanOptionalAction, default=False,
        help="Enable HBM scale swizzling (default: False).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is not available")

    arch = get_arch()
    assert arch in ("gfx950", "gfx1250"), (
        f"a8w4 kernel requires gfx950 or gfx1250, got {arch}"
    )

    if args.hbm_swizzling:
        if arch == "gfx950" and (args.N % 32 != 0 or args.K % (32 * 8) != 0):
            raise ValueError(
                f"Shape {args.M}x{args.N}x{args.K} not supported for scale swizzling on gfx950"
            )
        if arch == "gfx1250" and (args.N % 128 != 0 or args.K % (32 * 4) != 0):
            raise ValueError(
                f"Shape {args.M}x{args.N}x{args.K} not supported for scale swizzling on gfx1250"
            )

    print("Testing MoE GEMM A8W4 kernel")
    print(
        f"  M={args.M}, K={args.K}, N={args.N}, E={args.E}, "
        f"n_expts_act={args.n_expts_act}"
    )
    print(
        f"  Flags: gather={args.do_gather}, scatter={args.do_scatter}, "
        f"swiglu={args.apply_swiglu}, fused_quant={args.fused_quant}, "
        f"gammas={args.has_y_gammas}, hbm_swizzling={args.hbm_swizzling}"
    )
    print(f"  Device: {device}, Architecture: {arch}")

    logits = torch.randn((args.M, args.E), dtype=torch.float16, device=device)
    routing_data, gather_idx, scatter_idx = routing(logits, args.n_expts_act)

    config = get_kernel_config_gluon(args.M, args.N, args.K, routing_data)
    print(
        f"  Config: block_m={config['block_m']}, block_n={config['block_n']}, "
        f"block_k={config['block_k']}, num_warps={config['num_warps']}, "
        f"num_buffers={config['num_buffers']}"
    )
    routing_data.gate_scal = None
    gather_idx = gather_idx if args.do_gather else None
    scatter_idx = scatter_idx if args.do_scatter else None

    in_m = args.M * (args.n_expts_act if gather_idx is None else 1)

    x_bf16 = torch.randn((in_m, args.K), dtype=torch.bfloat16, device=device) / 10
    w_bf16 = torch.randn((args.E, args.K, args.N), dtype=torch.bfloat16, device=device) / 10
    bias = torch.randn((args.E, args.N), dtype=torch.float32, device=device)
    gammas = (
        2 ** torch.randint(
            -5, 0, (args.M * args.n_expts_act,), device=device, dtype=torch.float32
        )
        if args.has_y_gammas
        else None
    )

    w_tri, w_scale_tri = downcast_to_mxfp(w_bf16, torch.uint8, axis=1)
    w_ref = upcast_from_mxfp(w_tri, w_scale_tri, torch.bfloat16, axis=1)

    swizzle_mx_scale = None
    if args.hbm_swizzling:
        if arch == "gfx1250":
            swizzle_mx_scale = "GFX1250_SCALE"
            w_scale_tri = swizzle_scales_gfx1250(w_scale_tri)
        else:
            swizzle_mx_scale = "CDNA4_SCALE"
            w_scale_tri = swizzle_scales_gfx950(w_scale_tri)

    x_mx_scales = None
    x_static_scale = x_bf16.abs().max().float() / 448.0
    x_tri = downcast_to_static_fp8(x_bf16, x_static_scale)
    x_ref = x_bf16.clone()

    ref_y = moe_gemm_torch(
        x_ref, w_ref, bias.clone(), routing_data, gather_idx, scatter_idx,
        gammas, args.apply_swiglu,
    )

    # block_k = config['block_k']
    # block_n = config['block_n']
    # K_padded = 3072
    # N_padded = 6144

    # xK = x_tri.shape[1]
    # x_pad = torch.zeros((in_m, K_padded), dtype=x_tri.dtype, device=device)
    # x_pad[:, :xK] = x_tri
    # x_tri = x_pad[:, :xK]

    # wK, wN = w_tri.shape[1], w_tri.shape[2]
    # w_pad = torch.zeros((args.E, N_padded, K_padded // 2), dtype=w_tri.dtype, device=device)
    # w_pad = w_pad.transpose(1, 2)
    # w_pad[:, :wK, :wN] = w_tri
    # w_tri = w_pad[:, :wK, :wN]

    # sK, sN = w_scale_tri.shape[1], w_scale_tri.shape[2]
    # ws_pad = torch.zeros((args.E, N_padded, sK), dtype=w_scale_tri.dtype, device=device)
    # ws_pad = ws_pad.transpose(1, 2)
    # ws_pad[:, :sK, :sN] = w_scale_tri
    # w_scale_tri = ws_pad[:, :sK, :sN]

    # bias_pad = torch.zeros((args.E, N_padded), dtype=bias.dtype, device=device)
    # bias_pad[:, :args.N] = bias
    # bias = bias_pad[:, :args.N]

    # print(f"  Stride padding: K {args.K}->{K_padded}, N {args.N}->{N_padded}")
    # print(f"  x_tri       shape={tuple(x_tri.shape)}  stride={x_tri.stride()}")
    # print(f"  w_tri       shape={tuple(w_tri.shape)}  stride={w_tri.stride()}")
    # print(f"  w_scale_tri shape={tuple(w_scale_tri.shape)}  stride={w_scale_tri.stride()}")
    # print(f"  bias        shape={tuple(bias.shape)}  stride={bias.stride()}")

    quant_static_scale = None
    out_dtype = torch.bfloat16
    if args.fused_quant:
        quant_static_scale = ref_y.abs().max().float() / 448.0
        out_dtype = torch.float8_e4m3fn

    tri_y = moe_gemm_a8w4(
        x_tri,
        w_tri,
        x_mx_scales,
        w_scale_tri,
        x_static_scale,
        quant_static_scale,
        bias,
        routing_data,
        gather_idx,
        scatter_idx,
        gammas,
        swizzle_mx_scale,
        out_dtype,
        args.apply_swiglu,
        # unpadded_N=args.N,
        # unpadded_K=args.K,
    )
    if args.fused_quant:
        tri_y = (tri_y.float() * quant_static_scale).to(ref_y.dtype)

    ref_f = ref_y.to(torch.float32).detach()
    tri_f = tri_y.to(torch.float32).detach()
    eps = 1.0e-30
    multiplier = 1.0 / (torch.max(torch.abs(ref_f)) + eps)
    refn = ref_f * multiplier
    trin = tri_f * multiplier
    ref_rms = torch.sqrt(torch.square(refn).mean()) + eps
    rel_err = torch.abs(refn - trin) / torch.maximum(ref_rms, torch.abs(refn))
    max_err = torch.max(rel_err).item()
    rms_err = torch.sqrt(torch.square(rel_err).mean()).item()

    maxtol, rmstol = 4e-1, 4e-2
    print(f"maximum relative error = {max_err} (threshold = {maxtol})")
    print(f"RMS relative error = {rms_err} (threshold = {rmstol})")
    if max_err > maxtol or rms_err > rmstol:
        raise AssertionError("Wrapper test failed against reference")
    print("Test completed successfully")
    return 0


if __name__ == "__main__":
    main()
