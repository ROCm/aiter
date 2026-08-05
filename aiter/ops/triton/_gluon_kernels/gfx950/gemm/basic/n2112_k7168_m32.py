# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 OpenAI

"""M=32 FP8 GEMM producer with fixed split-K and FP32 rounding order."""

import torch
import triton.experimental.gluon.language as gl
from triton.experimental import gluon

M = 32
N = 2112
K = 7168
GROUP_K = 128
NUM_KSPLIT = 7


@gluon.jit
def _m32_fp8_splitk(x_ptr, w_ptr, out_ptr, x_scale_ptr, w_scale_ptr):
    pid = gl.program_id(0)
    pid_k = pid // 66
    pid_n = pid % 66

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=False,
        warps_per_cta=[2, 1],
        tiles_per_warp=[1, 2],
    )
    a_dot: gl.constexpr = gl.DotOperandLayout(0, mfma_layout, 16)
    b_dot: gl.constexpr = gl.DotOperandLayout(1, mfma_layout, 16)

    a_copy: gl.constexpr = gl.BlockedLayout([1, 16], [8, 8], [2, 1], [1, 0])
    b_copy: gl.constexpr = gl.BlockedLayout([16, 1], [8, 8], [1, 2], [0, 1])

    a_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[1024, 32]], [32, 128], [1, 0]
    )
    b_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[1024, 32]], [128, 32], [0, 1]
    )
    a_pages = gl.allocate_shared_memory(
        x_ptr.type.element_ty, [2, 32, 128], layout=a_shared
    )
    b_pages = gl.allocate_shared_memory(
        w_ptr.type.element_ty, [2, 128, 32], layout=b_shared
    )

    am = gl.arange(0, 32, layout=gl.SliceLayout(1, a_copy))
    ak = gl.arange(0, 128, layout=gl.SliceLayout(0, a_copy))
    bn = pid_n * 32 + gl.arange(0, 32, layout=gl.SliceLayout(0, b_copy))
    bk = gl.arange(0, 128, layout=gl.SliceLayout(1, b_copy))

    a_off = am[:, None] * 7168 + ak[None, :] + pid_k * 1024
    b_off = bn[None, :] * 7168 + bk[:, None] + pid_k * 1024

    sm = gl.arange(0, 32, layout=gl.SliceLayout(1, mfma_layout))
    sn = pid_n * 32 + gl.arange(0, 32, layout=gl.SliceLayout(0, mfma_layout))
    sx_off = (sm * 56 + pid_k * 8).to(gl.int32)
    sw_off = ((sn // 128) * 56 + pid_k * 8).to(gl.int32)

    zero = gl.zeros((32, 32), gl.float32, layout=mfma_layout)
    acc = zero

    wp1 = gl.amd.cdna4.buffer_load(w_scale_ptr, sw_off + 1)
    wp3 = gl.amd.cdna4.buffer_load(w_scale_ptr, sw_off + 3)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(b_pages.index(0), w_ptr, b_off)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(a_pages.index(0), x_ptr, a_off)
    gl.amd.cdna4.async_copy.commit_group()
    a_off += 128
    b_off += 128
    gl.amd.cdna4.async_copy.buffer_load_to_shared(b_pages.index(1), w_ptr, b_off)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(a_pages.index(1), x_ptr, a_off)
    gl.amd.cdna4.async_copy.commit_group()
    xp4 = gl.amd.cdna4.buffer_load(x_scale_ptr, sx_off + 4)
    xp7 = gl.amd.cdna4.buffer_load(x_scale_ptr, sx_off + 7)
    gl.amd.cdna4.async_copy.wait_group(1)
    a_frag = a_pages.index(0).load(a_dot)
    b_frag = b_pages.index(0).load(b_dot)
    a_off += 128
    b_off += 128
    gl.amd.cdna4.async_copy.buffer_load_to_shared(b_pages.index(0), w_ptr, b_off)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(a_pages.index(0), x_ptr, a_off)
    gl.amd.cdna4.async_copy.commit_group()
    tile = gl.amd.cdna4.mfma_scaled(a_frag, None, "e4m3", b_frag, None, "e4m3", zero)
    xs = gl.amd.cdna4.buffer_load(x_scale_ptr, sx_off)
    ws = gl.amd.cdna4.buffer_load(w_scale_ptr, sw_off)
    acc = gl.fma((tile * xs[:, None]), ws[None, :], acc)
    sx_off += 1
    sw_off += 1
    gl.amd.cdna4.async_copy.wait_group(1)
    a_frag = a_pages.index(1).load(a_dot)
    b_frag = b_pages.index(1).load(b_dot)
    a_off += 128
    b_off += 128
    gl.amd.cdna4.async_copy.buffer_load_to_shared(b_pages.index(1), w_ptr, b_off)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(a_pages.index(1), x_ptr, a_off)
    gl.amd.cdna4.async_copy.commit_group()
    tile = gl.amd.cdna4.mfma_scaled(a_frag, None, "e4m3", b_frag, None, "e4m3", zero)
    xs = gl.amd.cdna4.buffer_load(x_scale_ptr, sx_off)
    ws = wp1
    acc = gl.fma((tile * xs[:, None]), ws[None, :], acc)
    sx_off += 1
    sw_off += 1
    gl.amd.cdna4.async_copy.wait_group(1)
    a_frag = a_pages.index(0).load(a_dot)
    b_frag = b_pages.index(0).load(b_dot)
    a_off += 128
    b_off += 128
    gl.amd.cdna4.async_copy.buffer_load_to_shared(b_pages.index(0), w_ptr, b_off)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(a_pages.index(0), x_ptr, a_off)
    gl.amd.cdna4.async_copy.commit_group()
    tile = gl.amd.cdna4.mfma_scaled(a_frag, None, "e4m3", b_frag, None, "e4m3", zero)
    xs = gl.amd.cdna4.buffer_load(x_scale_ptr, sx_off)
    ws = gl.amd.cdna4.buffer_load(w_scale_ptr, sw_off)
    acc = gl.fma((tile * xs[:, None]), ws[None, :], acc)
    sx_off += 1
    sw_off += 1
    gl.amd.cdna4.async_copy.wait_group(1)
    a_frag = a_pages.index(1).load(a_dot)
    b_frag = b_pages.index(1).load(b_dot)
    a_off += 128
    b_off += 128
    gl.amd.cdna4.async_copy.buffer_load_to_shared(b_pages.index(1), w_ptr, b_off)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(a_pages.index(1), x_ptr, a_off)
    gl.amd.cdna4.async_copy.commit_group()
    tile = gl.amd.cdna4.mfma_scaled(a_frag, None, "e4m3", b_frag, None, "e4m3", zero)
    xs = gl.amd.cdna4.buffer_load(x_scale_ptr, sx_off)
    ws = wp3
    acc = gl.fma((tile * xs[:, None]), ws[None, :], acc)
    sx_off += 1
    sw_off += 1
    gl.amd.cdna4.async_copy.wait_group(1)
    a_frag = a_pages.index(0).load(a_dot)
    b_frag = b_pages.index(0).load(b_dot)
    a_off += 128
    b_off += 128
    gl.amd.cdna4.async_copy.buffer_load_to_shared(b_pages.index(0), w_ptr, b_off)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(a_pages.index(0), x_ptr, a_off)
    gl.amd.cdna4.async_copy.commit_group()
    tile = gl.amd.cdna4.mfma_scaled(a_frag, None, "e4m3", b_frag, None, "e4m3", zero)
    ws = gl.amd.cdna4.buffer_load(w_scale_ptr, sw_off)
    xs = xp4
    acc = gl.fma((tile * xs[:, None]), ws[None, :], acc)
    sx_off += 1
    sw_off += 1
    gl.amd.cdna4.async_copy.wait_group(1)
    a_frag = a_pages.index(1).load(a_dot)
    b_frag = b_pages.index(1).load(b_dot)
    a_off += 128
    b_off += 128
    gl.amd.cdna4.async_copy.buffer_load_to_shared(b_pages.index(1), w_ptr, b_off)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(a_pages.index(1), x_ptr, a_off)
    gl.amd.cdna4.async_copy.commit_group()
    tile = gl.amd.cdna4.mfma_scaled(a_frag, None, "e4m3", b_frag, None, "e4m3", zero)
    ws = gl.amd.cdna4.buffer_load(w_scale_ptr, sw_off)
    xs = gl.amd.cdna4.buffer_load(x_scale_ptr, sx_off)
    acc = gl.fma((tile * xs[:, None]), ws[None, :], acc)
    sx_off += 1
    sw_off += 1
    gl.amd.cdna4.async_copy.wait_group(1)
    a_frag = a_pages.index(0).load(a_dot)
    b_frag = b_pages.index(0).load(b_dot)
    tile = gl.amd.cdna4.mfma_scaled(a_frag, None, "e4m3", b_frag, None, "e4m3", zero)
    ws = gl.amd.cdna4.buffer_load(w_scale_ptr, sw_off)
    xs = gl.amd.cdna4.buffer_load(x_scale_ptr, sx_off)
    acc = gl.fma((tile * xs[:, None]), ws[None, :], acc)
    sx_off += 1
    sw_off += 1
    gl.amd.cdna4.async_copy.wait_group(0)
    a_frag = a_pages.index(1).load(a_dot)
    b_frag = b_pages.index(1).load(b_dot)
    tile = gl.amd.cdna4.mfma_scaled(a_frag, None, "e4m3", b_frag, None, "e4m3", zero)
    xs = xp7
    ws = gl.amd.cdna4.buffer_load(w_scale_ptr, sw_off)
    acc = gl.fma((tile * xs[:, None]), ws[None, :], acc)
    sx_off += 1
    sw_off += 1

    out_off = pid_k * 67584 + sm[:, None] * 2112 + sn[None, :]
    gl.amd.cdna4.buffer_store(
        acc, out_ptr, out_off, cache=".wt", mask=sn[None, :] < 2112
    )


def fp8_blockscale_q_a_kv_a_m32_producer(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    """Compute the seven FP32 K-partials consumed by the fixed reducer."""

    _m32_fp8_splitk[(462,)](
        x,
        w,
        workspace,
        x_scale,
        w_scale,
        num_warps=2,
        num_stages=3,
        waves_per_eu=1,
    )
    return workspace


def block_scaled_mm_n2112_k7168_m32(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Return the exact BF16 q_a+kv_a result for the registered shape."""
    from .splitk_reduce import reduce_n2112_k7168_m32

    assert out.shape == (32, 2112)
    assert out.dtype == torch.bfloat16
    assert out.device == x.device and out.is_contiguous()
    workspace = torch.empty((7, 32, 2112), dtype=torch.float32, device=x.device)
    fp8_blockscale_q_a_kv_a_m32_producer(
        x,
        w,
        x_scale,
        w_scale,
        workspace,
    )
    reduce_n2112_k7168_m32(workspace, out)
    return out


__all__ = ["block_scaled_mm_n2112_k7168_m32"]
