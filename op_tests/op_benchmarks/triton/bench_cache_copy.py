import torch
import triton
import triton.language as tl
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
import sys
import aiter.ops.triton.utils._triton.arch_info as arch_info
import argparse
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)
import math

DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)
WARP_SIZE = 32 if IS_DEVICE_ARCH_GFX12 else 64
WAPR_SIZE_LOG2 = int(math.log2(WARP_SIZE))


def make_kv_cache_shuffled_layout(
    BLOCK_SIZE_N_SHFL,
    BLOCK_SIZE_INNER_DIM_SHFL,
    fastest_dim_num_warps,
    total_num_warps,
    dtype=torch.bfloat16,
):
    num_warps_log2 = int(math.log2(fastest_dim_num_warps))
    BLOCK_SIZE_N_SHFL_log2 = int(math.log2(BLOCK_SIZE_N_SHFL))
    BLOCK_SIZE_INNER_DIM_SHFL_log2 = int(math.log2(BLOCK_SIZE_INNER_DIM_SHFL))
    # TODO: support e4m3_dtype and mxfp4x2
    # assert dtype in [torch.bfloat16, e4m3_dtype, torch.uint8], f"Unsupported dtype: {dtype} for making linear layout for shuffled weights"
    assert dtype in [
        torch.bfloat16
    ], f"Unsupported dtype: {dtype} for making linear layout for shuffled weights"
    if dtype == torch.bfloat16:
        # (8 elements per thread for BF16)
        coalesced_size_log2 = 3
    # elif dtype == e4m3_dtype:
    #     # (16 elements per thread for e4m3_dtype)
    #     coalesced_size_log2 = 4
    # else:
    #     # (16*2 elements per thread for mxfp4x2)
    #     coalesced_size_log2 = 4
    assert (
        BLOCK_SIZE_INNER_DIM_SHFL_log2 > coalesced_size_log2 + WAPR_SIZE_LOG2
    ), "BLOCK_SIZE_INNER_DIM_SHFL_log2 must be greater than coalesced_size_log2 + WAPR_SIZE_LOG2, please increase block_size to at least 64"
    reg_bases = (
        [[0, 1 << v] for v in range(coalesced_size_log2)]
        + [
            [0, 1 << v]
            for v in range(
                coalesced_size_log2 + WAPR_SIZE_LOG2, BLOCK_SIZE_INNER_DIM_SHFL_log2
            )
        ]
        + [[1 << v, 0] for v in range(num_warps_log2, BLOCK_SIZE_N_SHFL_log2)]
    )
    lane_bases = [
        [0, 1 << v]
        for v in range(coalesced_size_log2, coalesced_size_log2 + WAPR_SIZE_LOG2)
    ]
    if num_warps_log2 > 0:
        warp_bases = [[1 << v, 0] for v in range(0, num_warps_log2)]
    elif total_num_warps == 1:
        warp_bases = []
    else:
        warp_bases = [[0, 0]]

    layout = gl.constexpr(
        gl.DistributedLinearLayout(
            reg_bases=reg_bases,
            lane_bases=lane_bases,
            warp_bases=warp_bases,
            block_bases=[],
            shape=[BLOCK_SIZE_N_SHFL, BLOCK_SIZE_INNER_DIM_SHFL],
        )
    )
    return layout


@gluon.jit
def simple_tdm_kernel(
    key_cache_ptr,
    y_ptr,
    num_blocks,
    NUM_KV_HEADS: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    key_cache_stride_1: gl.constexpr,
    y_stride_m: gl.constexpr,
    y_stride_d: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    num_warps: gl.constexpr,
    waves_per_eu: gl.constexpr,
    use_tdm: gl.constexpr,
):
    kv_head_idx = gl.program_id(0)

    K_SHARED_LAYOUT: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[1, 0]
    )
    K_DOT_LAYOUT: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=WMMA_LAYOUT, k_width=8
    )

    k_desc = None
    if use_tdm:
        STORE_LAYOUT: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[4, 8],
            warps_per_cta=[num_warps, 1],
            order=[1, 0],
        )
        k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=key_cache_ptr,
            shape=(
                num_blocks * NUM_KV_HEADS,
                BLOCK_SIZE * HEAD_SIZE,
            ),
            strides=(key_cache_stride_1, 1),
            block_shape=(gl.constexpr(1), BLOCK_SIZE * HEAD_SIZE),
            layout=K_SHARED_LAYOUT,
        )
        smem = gl.allocate_shared_memory(
            k_desc.dtype,
            shape=[2] + k_desc.block_shape,
            layout=k_desc.layout,
        )
    else:
        STORE_LAYOUT: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[8, 8],
            warps_per_cta=[num_warps, 1],
            order=[1, 0],
        )
        k_desc = key_cache_ptr
        smem = gl.allocate_shared_memory(
            key_cache_ptr.type.element_ty,
            shape=[2] + [BLOCK_SIZE // 16, HEAD_SIZE * 16],
            layout=K_SHARED_LAYOUT,
        )

    buffer_id: gl.int32 = 0
    acc = gl.zeros((HEAD_SIZE, BLOCK_SIZE), dtype=gl.float32, layout=K_DOT_LAYOUT)

    if use_tdm:
        gl.amd.gfx1250.tdm.async_load(
            k_desc,
            [(0 * NUM_KV_HEADS + kv_head_idx).to(gl.int32), 0],
            smem.index(buffer_id),
        )
        K_LOAD_LAYOUT = None
        offsets = None
    else:
        # K_LOAD_LAYOUT: gl.constexpr = gl.BlockedLayout(
        #     size_per_thread=[1, 8],
        #     threads_per_warp=[1, 64],
        #     warps_per_cta=[1, num_warps],
        #     order=[0],
        # )
        K_LOAD_LAYOUT: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [0, 512], [1, 0], [2, 0]],
            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256]],
            warp_bases=[],
            block_bases=[],
            shape=[BLOCK_SIZE // 16, HEAD_SIZE * 16],
        )
        offs_k_t = gl.arange(
            0, BLOCK_SIZE // 16, layout=gl.SliceLayout(1, K_LOAD_LAYOUT)
        )
        offs_k_d = gl.arange(0, HEAD_SIZE * 16, layout=gl.SliceLayout(0, K_LOAD_LAYOUT))

        offsets = (
            kv_head_idx * key_cache_stride_1
            + offs_k_t[:, None] * (HEAD_SIZE * 16)
            + offs_k_d[None, :]
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=smem.index(buffer_id),
            ptr=k_desc,
            offsets=offsets.to(gl.int32),
            cache_modifier=".cg",
        )
        gl.amd.cdna4.async_copy.commit_group()

    buffer_id = 1 - buffer_id

    for block_idx in range(1, num_blocks):
        if use_tdm:
            gl.amd.gfx1250.tdm.async_load(
                k_desc,
                [(block_idx * NUM_KV_HEADS + kv_head_idx).to(gl.int32), 0],
                smem.index(buffer_id),
            )
            gl.amd.gfx1250.tdm.async_wait(1)
        else:
            k_desc += key_cache_stride_1 * NUM_KV_HEADS
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                dest=smem.index(buffer_id),
                ptr=k_desc,
                offsets=offsets.to(gl.int32),
                cache_modifier=".cg",
            )
            gl.amd.cdna4.async_copy.commit_group()
            gl.amd.cdna4.async_copy.wait_group(1)

        next_buffer_id = 1 - buffer_id

        smem_load = (
            smem.index(next_buffer_id)
            .reshape(
                (
                    1,
                    BLOCK_SIZE // 16,
                    HEAD_SIZE // 16,
                    2,
                    16,
                    8,
                )
            )
            .permute((0, 1, 4, 2, 3, 5))
            .reshape((BLOCK_SIZE, HEAD_SIZE))
            .permute((1, 0))
        )
        if use_tdm:
            X = smem_load.load(layout=K_DOT_LAYOUT)
        else:
            # X = smem_load.load(layout=K_DOT_LAYOUT)
            X = gl.amd.cdna4.async_copy.load_shared_relaxed(
                smem_load, layout=K_DOT_LAYOUT
            )

        acc = acc + X.to(gl.float32)

        buffer_id = next_buffer_id

    buffer_id = 1 - buffer_id
    if use_tdm:
        gl.amd.gfx1250.tdm.async_wait(0)
    else:
        gl.amd.cdna4.async_copy.wait_group(0)

    smem_load = (
        smem.index(buffer_id)
        .reshape(
            (
                1,
                BLOCK_SIZE // 16,
                HEAD_SIZE // 16,
                2,
                16,
                8,
            )
        )
        .permute((0, 1, 4, 2, 3, 5))
        .reshape((BLOCK_SIZE, HEAD_SIZE))
        .permute((1, 0))
    )
    if use_tdm:
        X = smem_load.load(layout=K_DOT_LAYOUT)
    else:
        # X = smem_load.load(layout=K_DOT_LAYOUT)
        X = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_load, layout=K_DOT_LAYOUT)

    acc = acc + X.to(gl.float32)

    acc = gl.convert_layout(acc, layout=STORE_LAYOUT)
    offs_d = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(1, STORE_LAYOUT))
    offs_b = gl.arange(0, BLOCK_SIZE, layout=gl.SliceLayout(0, STORE_LAYOUT))
    gl.amd.cdna4.buffer_store(
        stored_value=acc.to(y_ptr.type.element_ty),
        ptr=y_ptr,
        offsets=kv_head_idx * y_stride_m
        + offs_d[:, None] * y_stride_d
        + offs_b[None, :],
    )


def benchmark(args):
    num_blocks = args.num_blocks
    block_size = args.block_size
    num_kv_heads = args.num_kv_heads
    head_size = args.head_size
    num_warps = args.num_warps
    waves_per_eu = args.waves_per_eu
    use_tdm = IS_DEVICE_ARCH_GFX12
    # assert IS_DEVICE_ARCH_GFX12, "Gluon Cache Copy only supports gfx1250"
    if IS_DEVICE_ARCH_GFX12:
        assert (
            num_warps == 1 and head_size == 64 and block_size == 64
        ), "Gluon Cache Copy only supports gfx1250 with 1 warp, 64 head size, and 64 block size"
    configs = []
    x_names = [
        "num_blocks",
        "block_size",
        "num_kv_heads",
        "head_size",
        "num_warps",
        "waves_per_eu",
    ]
    x_vals_list = [
        (
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            num_warps,
            waves_per_eu,
        )
    ]
    if args.metric == "time":
        unit = "ms"
    elif args.metric == "bandwidth":
        unit = "TB/s"
    else:
        raise ValueError("Unknown metric: " + args.metric)

    line_vals = [args.metric]
    line_names = ["TDM " if use_tdm else "ASYNC_COPY " + args.metric]
    configs.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_names,
            plot_name=get_caller_name_no_ext(),
            styles=[("red", "-"), ("green", "-")],
            ylabel=unit,
            args={},
        )
    )

    @triton.testing.perf_report(configs)
    def bench_cache_copy(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        num_warps: int,
        waves_per_eu: int,
        provider,
    ):
        warmup = 25
        rep = 100

        key_cache = torch.randn(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        key_cache_shuffled = key_cache.view(
            -1, block_size, num_kv_heads, head_size
        ).permute(0, 2, 1, 3)
        key_cache_shuffled = key_cache_shuffled.view(
            -1,
            num_kv_heads,
            block_size // 16,
            16,
            head_size // 16,
            2,  # there are 2 groups of threads, t0 ~ t15 and t16 ~ t31
            8,
        )
        key_cache_shuffled = key_cache_shuffled.permute(
            0, 1, 2, 4, 5, 3, 6
        ).contiguous()
        key_cache_shuffled = key_cache_shuffled.view(
            -1, num_kv_heads, block_size // 16, head_size * 16
        )
        if num_warps == 1:
            warp_bases = []
        elif num_warps == 2:
            warp_bases = [[1, 0]]
        elif num_warps == 4:
            warp_bases = [[1, 0], [2, 0]]
        if IS_DEVICE_ARCH_GFX12:
            WMMA_LAYOUT = gl.constexpr(
                gl.amd.AMDWMMALayout(
                    version=3,
                    transposed=True,
                    warp_bases=warp_bases,
                    reg_bases=[],
                    instr_shape=[16, 16, 32],
                )
            )
        else:
            WMMA_LAYOUT = gl.constexpr(
                gl.amd.AMDMFMALayout(
                    version=4,
                    instr_shape=[16, 16, 32],
                    transposed=True,
                    warps_per_cta=[num_warps, 1],
                )
            )

        y = torch.empty(
            num_kv_heads, head_size, block_size, dtype=torch.bfloat16, device="cuda"
        )

        def fn():
            simple_tdm_kernel[(num_kv_heads,)](
                key_cache_ptr=key_cache_shuffled,
                y_ptr=y,
                num_blocks=num_blocks,
                NUM_KV_HEADS=num_kv_heads,
                BLOCK_SIZE=block_size,
                HEAD_SIZE=head_size,
                key_cache_stride_1=key_cache_shuffled.stride(1),
                y_stride_m=y.stride(0),
                y_stride_d=y.stride(1),
                WMMA_LAYOUT=WMMA_LAYOUT,
                num_warps=num_warps,
                waves_per_eu=waves_per_eu,
                use_tdm=use_tdm,
            )
            # try:
            #     ref = key_cache.sum(dim=0).permute(1, 2, 0)
            #     torch.testing.assert_close(y, ref)
            #     print(f"Passed")
            # except Exception as e:
            #     print(f"Failed")
            #     raise e

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        mem = (
            (num_blocks + 1)
            * num_kv_heads
            * head_size
            * block_size
            * torch.bfloat16.itemsize
            * 1e-12
        )
        if "time" in provider:
            return ms
        else:  # TB/s
            return mem / ms * 1e3

    bench_cache_copy.run(
        save_path="." if args.o else None, print_data=True, show_plots=False
    )


def run_bench(args):
    torch.manual_seed(0)
    torch.set_default_device(args.device)
    benchmark(args)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark Unified Attention",
        allow_abbrev=False,
    )
    parser.add_argument("--num_blocks", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--num_kv_heads", type=int, default=2048)
    parser.add_argument("--head_size", type=int, default=64)
    parser.add_argument("--num_warps", type=int, default=1)
    parser.add_argument("--waves_per_eu", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "-metric",
        nargs="?",
        const="bandwidth",
        choices=["time", "bandwidth"],
        default="bandwidth",
        help="Metrics for the kernel benchmark.",
    )
    parser.add_argument(
        "-o",
        action="store_true",
        default=False,
        help="Write performance results to CSV file",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_bench(args)


if __name__ == "__main__":
    main()
