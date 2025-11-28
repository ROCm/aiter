
import functools
import json
import triton
import triton.language as tl
import torch
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton.pid_preprocessing import remap_xcd
from aiter import dtypes

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async_copy
from aiter.test_common import checkAllclose

@gluon.jit
def test_async_load(
    input,
    output,
    in_stride,
    out_stride,
    BLOCK_H: gl.constexpr,
    BLOCK_C: gl.constexpr,
    ):

    pid = gl.program_id(0)

    # blocked_ld_in: gl.constexpr = gl.BlockedLayout(
    #     size_per_thread=[2, 8],  # 64 * 512
    #     threads_per_warp=[8, 8],
    #     warps_per_cta=[1, 4],
    #     order=[1, 0],
    # )
    blocked_ld_in: gl.constexpr = gl.BlockedLayout([1, 8], [2, 32], [4, 1], [1, 0])
    blocked_ld_out: gl.constexpr = gl.BlockedLayout([4, 8], [2, 32], [4, 1], [1, 0])

    cur_head = gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, blocked_ld_in)
    )

    offs_k_c = gl.arange(
        0, BLOCK_C, layout=gl.SliceLayout(0, blocked_ld_in)
    )

    shared_in: gl.constexpr = gl.SwizzledSharedLayout(
        8, 1, 1, [1, 0]
    )
    smem_in = gl.allocate_shared_memory(
        input.type.element_ty, [BLOCK_H * 4, BLOCK_C], layout=shared_in
    )
    smem_in0 = smem_in.slice(0,           BLOCK_H)
    smem_in1 = smem_in.slice(BLOCK_H,     BLOCK_H)
    smem_in2 = smem_in.slice(BLOCK_H * 2, BLOCK_H)
    smem_in3 = smem_in.slice(BLOCK_H * 3, BLOCK_H)

    offs = cur_head[:, None] * BLOCK_C + offs_k_c[None, :]

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_in0,
        input,
        offsets=offs,
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_in1,
        input,
        offsets=offs + BLOCK_H * BLOCK_C,
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_in2,
        input,
        offsets=offs + BLOCK_H * BLOCK_C * 2,
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_in3,
        input,
        offsets=offs + BLOCK_H * BLOCK_C * 3,
    )

    cur_o = smem_in.load(layout=blocked_ld_out)
    # cdna4_async_copy.async_wait(0)
    # cur_o = cdna4_async_copy.load_shared_relaxed(smem_in, blocked_ld_in)
    cur_head_out = gl.arange(
        0, BLOCK_H * 4, layout=gl.SliceLayout(1, blocked_ld_out)
    )
    offs_k_c_out = gl.arange(
        0, BLOCK_C, layout=gl.SliceLayout(0, blocked_ld_out)
    )

    offs_out = cur_head_out[:, None] * BLOCK_C + offs_k_c_out[None, :]

    gl.amd.cdna3.buffer_store(
        stored_value=cur_o,
        ptr=output,
        offsets=offs_out,
    )


@gluon.jit
def test_async_load_mfma(
    A_ptr,
    K_Buffer,
    output,
    in_stride,
    out_stride,
    BLOCK_H: gl.constexpr,
    BLOCK_C: gl.constexpr,
    ):

    pid = gl.program_id(0)

    # blocked_ld_in: gl.constexpr = gl.BlockedLayout(
    #     size_per_thread=[2, 8],  # 64 * 512
    #     threads_per_warp=[8, 8],
    #     warps_per_cta=[1, 4],
    #     order=[1, 0],
    # )
    mfma_layout_qk: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[1, 4]
    )
    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout_qk, k_width=16
    )
    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout_qk, k_width=16
    )

    blocked_ld_in: gl.constexpr = gl.BlockedLayout([1, 8], [2, 32], [4, 1], [1, 0])
    blocked_ld_out: gl.constexpr = gl.BlockedLayout([4, 8], [2, 32], [4, 1], [1, 0])
    blocked_a_mk: gl.constexpr = gl.BlockedLayout([4, 8], [2, 32], [4, 1], [1, 0])
    cur_head_a = gl.arange(
        0, BLOCK_H * 4, layout=gl.SliceLayout(1, blocked_ld_out)
    )
    offs_k_c_a = gl.arange(
        0, BLOCK_C, layout=gl.SliceLayout(0, blocked_ld_out)
    )

    shared_q: gl.constexpr = gl.SwizzledSharedLayout(
        8, 1, 1, [1, 0]
    )
    smem_a = gl.allocate_shared_memory(
        A_ptr.type.element_ty, [BLOCK_H * 4, BLOCK_C], layout=shared_q
    )

    offs_a = cur_head_a[:, None] * BLOCK_C + offs_k_c_a[None, :]
    a = gl.amd.cdna3.buffer_load(
        ptr=A_ptr,
        offsets=offs_a,
    )
    smem_a.store(a)
    cur_a = smem_a.load(layout=dot_q_layout)

    cur_head = gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, blocked_ld_in)
    )

    offs_k_c = gl.arange(
        0, BLOCK_C, layout=gl.SliceLayout(0, blocked_ld_in)
    )

    shared_in: gl.constexpr = gl.SwizzledSharedLayout(
        8, 1, 1, [1, 0]
    )
    shared_k_cal: gl.constexpr = gl.SwizzledSharedLayout(
        8, 1, 1, [0, 1]
    )
    smem_kv1 = gl.allocate_shared_memory(
        K_Buffer.type.element_ty, [BLOCK_H * 4, BLOCK_C], layout=shared_in
    )
    smem_kv10 = smem_kv1.slice(0,           BLOCK_H)
    smem_kv11 = smem_kv1.slice(BLOCK_H,     BLOCK_H)
    smem_kv12 = smem_kv1.slice(BLOCK_H * 2, BLOCK_H)
    smem_kv13 = smem_kv1.slice(BLOCK_H * 3, BLOCK_H)

    offs = cur_head[:, None] * BLOCK_C + offs_k_c[None, :]

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv10,
        K_Buffer,
        offsets=offs,
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv11,
        K_Buffer,
        offsets=offs + BLOCK_H * BLOCK_C,
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv12,
        K_Buffer,
        offsets=offs + BLOCK_H * BLOCK_C * 2,
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv13,
        K_Buffer,
        offsets=offs + BLOCK_H * BLOCK_C * 3,
    )

    # cdna4_async_copy.async_wait(0)
    # cur_o = cdna4_async_copy.load_shared_relaxed(smem_kv1, blocked_ld_in)
    out = gl.zeros(
        (BLOCK_H * 4, BLOCK_H * 4), dtype=gl.float32, layout=mfma_layout_qk,
    )

    smem_kv1 = smem_kv1._reinterpret(
        K_Buffer.type.element_ty, [256, BLOCK_H * 4], layout=shared_k_cal)

    cur_k1 = smem_kv1.load(layout=dot_k_layout)
    out = gl.amd.cdna3.mfma(cur_a, cur_k1, out)


    cur_head_out1 = gl.arange(
        0, BLOCK_H * 4, layout=gl.SliceLayout(1, mfma_layout_qk)
    )
    cur_head_out2 = gl.arange(
        0, BLOCK_H * 4, layout=gl.SliceLayout(0, mfma_layout_qk)
    )

    offs_out = cur_head_out1[:, None] * BLOCK_H * 4 + cur_head_out2[None, :]

    gl.amd.cdna3.buffer_store(
        stored_value=out.to(output.type.element_ty),
        ptr=output,
        offsets=offs_out,
    )

def test():
    TILE_H = 32
    TILE_C = 256
    BLOCK_H = 8
    BLOCK_C = 256



    A = torch.randn([TILE_H, TILE_C], dtype=torch.bfloat16, device="cuda")
    B = torch.randn([TILE_H, TILE_C], dtype=torch.bfloat16, device="cuda")
    C = A @ B.T

    grid = (1, 1, 1)
    # output = torch.empty_like(B)
    # test_async_load[grid](
    #     B,
    #     output,
    #     BLOCK_C,
    #     BLOCK_C,
    #     BLOCK_H,
    #     BLOCK_C,
    # )
    # checkAllclose(input, output)
    output = torch.empty_like(C)
    test_async_load_mfma[grid](
        A,
        B,
        output,
        BLOCK_C,
        BLOCK_C,
        BLOCK_H,
        BLOCK_C,
    )
    checkAllclose(C, output)


if __name__ == "__main__":
    test()

