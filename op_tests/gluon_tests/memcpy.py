"""
Tensor Layouts
It is ported from 02-layouts.py in gluon tutorial os triton project.
"""

import pytest
import torch
import triton
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton import language as tl

def _enabled(label):
    from sys import argv
    return len(argv) > 1 and label in argv[1].split(",")


@triton.jit
def memcpy_2d_kernel_triton_base(in_ptr, out_ptr,  #
                     xnumel, ynumel, xstride_in, ystride_in, xstride_out, ystride_out,  #
                     layout: gl.constexpr, XBLOCK: gl.constexpr, YBLOCK: gl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK
    indices_x = start_x + tl.arange(0, XBLOCK)
    indices_y = start_y + tl.arange(0, YBLOCK)

    in_offsets = xstride_in * indices_x[:, None] + ystride_in * indices_y[None, :]
    out_offsets = xstride_out * indices_x[:, None] + ystride_out * indices_y[None, :]
    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)

    value = tl.load(in_ptr + in_offsets, mask=mask)
    tl.store(out_ptr + out_offsets, value, mask=mask)


@gluon.jit
def memcpy_2d_kernel_gluon_base(in_ptr, out_ptr,  #
                     xnumel, ynumel, xstride_in, ystride_in, xstride_out, ystride_out,  #
                     layout: gl.constexpr, XBLOCK: gl.constexpr, YBLOCK: gl.constexpr):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)

    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK

    indices_x = start_x + gl.arange(0, XBLOCK, layout=gl.SliceLayout(dim=1, parent=layout))
    indices_y = start_y + gl.arange(0, YBLOCK, layout=gl.SliceLayout(dim=0, parent=layout))

    in_offsets = xstride_in * indices_x[:, None] + ystride_in * indices_y[None, :]
    out_offsets = xstride_out * indices_x[:, None] + ystride_out * indices_y[None, :]

    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)

    value = gl.load(in_ptr + in_offsets, mask=mask)
    gl.store(out_ptr + out_offsets, value, mask=mask)


# global memory -> shared memory -> vreg -> global memroy
# not tested due to https://github.com/ROCm/triton-internal/issues/1123
@gluon.jit
def memcpy_2d_kernel_v1(in_ptr, out_ptr,  #
                     xnumel, ynumel, xstride_in, ystride_in, xstride_out, ystride_out,  #
                     layout: gl.constexpr, XBLOCK: gl.constexpr, YBLOCK: gl.constexpr):
    shared: gl.constexpr =  gl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=4, order=[1, 0])

    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)

    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK

    blocked: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[8, 8], warps_per_cta=[4, 2],
                                                order=[1, 0])
    indices_x = start_x + gl.arange(0, XBLOCK, layout=gl.SliceLayout(dim=1, parent=blocked))
    indices_y = start_y + gl.arange(0, YBLOCK, layout=gl.SliceLayout(dim=0, parent=blocked))

    in_offsets = xstride_in * indices_x[:, None] + ystride_in * indices_y[None, :]
    out_offsets = xstride_out * indices_x[:, None] + ystride_out * indices_y[None, :]

    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)

    smem_a = gl.allocate_shared_memory(in_ptr.dtype.element_ty, [XBLOCK, YBLOCK], shared)
    other = tl.cast(0.0, in_ptr.dtype.element_ty)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_a, in_ptr, in_offsets, other=other, mask=mask)
    a = smem_a.load(blocked);

    gl.amd.cdna4.buffer_store(stored_value=a, ptr=out_ptr, offsets=out_offsets, mask=mask)


# global memory -> vreg -> global memory
@gluon.jit
def memcpy_2d_kernel_v0(in_ptr, out_ptr,  #
                     xnumel, ynumel, xstride_in, ystride_in, xstride_out, ystride_out,  #
                     layout: gl.constexpr, XBLOCK: gl.constexpr, YBLOCK: gl.constexpr):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)

    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK

    indices_x = start_x + gl.arange(0, XBLOCK, layout=gl.SliceLayout(dim=1, parent=layout))
    indices_y = start_y + gl.arange(0, YBLOCK, layout=gl.SliceLayout(dim=0, parent=layout))

    in_offsets = xstride_in * indices_x[:, None] + ystride_in * indices_y[None, :]
    out_offsets = xstride_out * indices_x[:, None] + ystride_out * indices_y[None, :]

    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)

    # global memory -> vreg -> global memory
    other = tl.cast(0.0, in_ptr.dtype.element_ty)

    # cache modidier is none??
    a = gl.amd.cdna4.buffer_load(ptr=in_ptr, offsets=in_offsets, other=other, mask=mask)
    gl.amd.cdna4.buffer_store(stored_value=a, ptr=out_ptr, offsets=out_offsets, mask=mask)


memcpy_2d_kernels = {
    'triton_base': memcpy_2d_kernel_triton_base,
    'gluon_base': memcpy_2d_kernel_gluon_base,
    'v0': memcpy_2d_kernel_v0, # buffer_load and buffer_store
    'v1': memcpy_2d_kernel_v1 # buffer_load_to_shared
}


def memcpy_2d_impl(input, output, kernel, XBLOCK, YBLOCK, layout, num_warps):
    xnumel, ynumel = input.shape
    grid = (triton.cdiv(xnumel, XBLOCK), triton.cdiv(ynumel, YBLOCK))

    compiled_kernel = kernel[grid](  #
        input, output, xnumel, ynumel,  #
        *input.stride(), *output.stride(),  #
        layout, XBLOCK, YBLOCK, num_warps=num_warps)
    return compiled_kernel


def bench_memcpy_2d(impl, kernel, transposed=False):
    xnumel = 8 * 1024
    ynumel = 16 * 1024
    input = torch.randn((xnumel, ynumel), device="cuda")
    output = torch.empty_like(input)
    input = input.T if transposed else input
    output = output.T if transposed else output

    compiled_kernel = impl(input, output, kernel)
    fn = lambda: impl(input, output, kernel)
    ms = triton.testing.do_bench(fn)

    torch.testing.assert_close(input, output, atol=0, rtol=0)
    return compiled_kernel, get_throughput(input, ms)


if __name__ == "__main__" and _enabled("memcpy_2d_layout"):
    print("Benchmarking 2D memcpy")
    print("======================")
    base_layout = gl.BlockedLayout([1, 4], [1, 64], [1, 4], [1, 0])
    blocked_1 = gl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[64, 1], warps_per_cta=[1, 4], order=[1, 0])
    blocked_2 = gl.BlockedLayout(size_per_thread=[4, 1], threads_per_warp=[64, 1], warps_per_cta=[1, 4], order=[1, 0])
    blocked_1x4 = gl.BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[2, 32], warps_per_cta=[1, 4], order=[1, 0])
    blocked_4x1 = gl.BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[2, 32], warps_per_cta=[4, 1], order=[1, 0])

    layouts = {
        'triton_base': [base_layout], # for trition kernel, layout is not needed, just to keep kernel signatures the same
        'gluon_base': [base_layout, blocked_1x4, blocked_4x1],
        'v0': [blocked_1x4, blocked_4x1]
    }

    num_warps = 4
    for XBLOCK, YBLOCK in [[1, 1024], [64, 64], [32, 32], [32, 64]]:
        print(f"XBLOCK={XBLOCK} YBLOCK={YBLOCK}")
        for kernel_name in ['triton_base', 'gluon_base', 'v0']:
            print(f"\tkernel_name={kernel_name}")
            for layout in layouts[kernel_name]:
                if kernel_name != 'triton_base':
                    print(f"\tlayout={layout}")
                kernel = memcpy_2d_kernels[kernel_name]
                impl = partial(memcpy_2d_impl, XBLOCK=XBLOCK, YBLOCK=YBLOCK, layout=layout, num_warps=num_warps)
                _, throughput = bench_memcpy_2d(impl, kernel)
                print(f"\tThroughput: {throughput:.3f} TB/s")
            print()
        print()

"""
With manual picked data layout, the throughput of amd.cdna4.buffer_load/store > gl.load/store > tl.load/store
                                        XBLOCK=1 YBLOCK=1024           XBLOCK=64 YBLOCK=64         XBLOCK=32 YBLOCK=32      XBLOCK=32 YBLOCK=64
triton:                                     4.178 TB/s                      3.136 TB/s                 2.484 TB/s                2.717 TB/s
gluon with gl.load/store                    4.181 TB/s                      3.155 TB/s                 2.490 TB/s                2.732 TB/s
gluon with gl.amd.buffer_load/store         4.237 TB/s                      3.477 TB/s                 3.084 TB/s                2.885 TB/s


Raw Data
Benchmarking 2D memcpy
======================
XBLOCK=1 YBLOCK=1024
        kernel_name=triton_base
        Throughput: 4.178 TB/s

        kernel_name=gluon_base
        layout=BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[1, 64], warps_per_cta=[1, 4], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
        Throughput: 4.181 TB/s
        layout=BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[2, 32], warps_per_cta=[1, 4], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
        Throughput: 4.178 TB/s
        layout=BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[2, 32], warps_per_cta=[4, 1], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
        Throughput: 4.179 TB/s

        kernel_name=v0
        layout=BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[2, 32], warps_per_cta=[1, 4], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
        Throughput: 4.237 TB/s
        layout=BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[2, 32], warps_per_cta=[4, 1], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
        Throughput: 2.828 TB/s


XBLOCK=64 YBLOCK=64
        kernel_name=triton_base
        Throughput: 3.136 TB/s

        kernel_name=gluon_base
        layout=BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[1, 64], warps_per_cta=[1, 4], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
        Throughput: 3.137 TB/s
        layout=BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[2, 32], warps_per_cta=[1, 4], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
        Throughput: 3.150 TB/s
        layout=BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[2, 32], warps_per_cta=[4, 1], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
        Throughput: 3.155 TB/s

        kernel_name=v0
        layout=BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[2, 32], warps_per_cta=[1, 4], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
        Throughput: 2.665 TB/s
        layout=BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[2, 32], warps_per_cta=[4, 1], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
        Throughput: 3.477 TB/s


XBLOCK=32 YBLOCK=32
        kernel_name=triton_base
        Throughput: 2.484 TB/s

        kernel_name=gluon_base
        layout=BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[1, 64], warps_per_cta=[1, 4], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
        Throughput: 2.480 TB/s
        layout=BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[2, 32], warps_per_cta=[1, 4], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
        Throughput: 2.471 TB/s
        layout=BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[2, 32], warps_per_cta=[4, 1], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
        Throughput: 2.490 TB/s

        kernel_name=v0
        layout=BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[2, 32], warps_per_cta=[1, 4], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
        Throughput: 1.589 TB/s
        layout=BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[2, 32], warps_per_cta=[4, 1], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
        Throughput: 3.084 TB/s


XBLOCK=32 YBLOCK=64
        kernel_name=triton_base
        Throughput: 2.717 TB/s

        kernel_name=gluon_base
        layout=BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[1, 64], warps_per_cta=[1, 4], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
        Throughput: 2.723 TB/s
        layout=BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[2, 32], warps_per_cta=[1, 4], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
        Throughput: 2.732 TB/s
        layout=BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[2, 32], warps_per_cta=[4, 1], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
        Throughput: 2.727 TB/s

        kernel_name=v0
        layout=BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[2, 32], warps_per_cta=[1, 4], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
        Throughput: 2.787 TB/s
        layout=BlockedLayout(size_per_thread=[1, 4], threads_per_warp=[2, 32], warps_per_cta=[4, 1], order=[1, 0], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])
        Throughput: 2.885 TB/s
"""
