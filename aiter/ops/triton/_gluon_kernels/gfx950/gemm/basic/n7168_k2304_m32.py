# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 OpenAI

"""Exact M=32, N=7168, K=2304 FP8 GEMM specialization for MI355X."""

from __future__ import annotations

import torch
from triton.experimental import gluon as Gluon
from triton.experimental.gluon import language as gl


@Gluon.jit
def _keep_running_sum_2304(x):
    # Preserve the fp32 accumulation boundary through LLVM scheduling.
    return gl.inline_asm_elementwise(
        "", "=v,0", [x], dtype=gl.float32, is_pure=False, pack=1
    )


@Gluon.jit
def _sl6(p):
    # Load the tile's six uniform B scales into SGPRs.
    return gl.inline_asm_elementwise(
        "s_load_dword $0, $6, 0\n s_load_dword $1, $6, 4\n s_load_dword $2, $6, 8\n s_load_dword $3, $6, 12\n s_load_dword $4, $6, 16\n s_load_dword $5, $6, 20",
        "=s,=s,=s,=s,=s,=s,s",
        [p],
        dtype=(gl.float32, gl.float32, gl.float32, gl.float32, gl.float32, gl.float32),
        is_pure=False,
        pack=1,
    )


@Gluon.jit
def _g_gemm_2304_async(a_ptr, b_ptr, as_ptr, bs_ptr, out_ptr):
    mf: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[2, 2]
    )
    da: gl.constexpr = gl.DotOperandLayout(0, mf, 16)
    db: gl.constexpr = gl.DotOperandLayout(1, mf, 16)
    copya: gl.constexpr = gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])
    copyb: gl.constexpr = gl.BlockedLayout([16, 1], [8, 8], [1, 4], [0, 1])
    sha: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [1, 0])
    shb: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [0, 1])
    raw = gl.program_id(0)
    pid_n = raw
    am = gl.arange(0, 32, layout=gl.SliceLayout(1, copya))
    ak = gl.arange(0, 128, layout=gl.SliceLayout(0, copya))
    bk = gl.arange(0, 128, layout=gl.SliceLayout(1, copyb))
    bn = pid_n * 32 + gl.arange(0, 32, layout=gl.SliceLayout(0, copyb))
    ao = am[:, None] * 2304 + ak[None, :]
    bo = bk[:, None] + bn[None, :] * 2304
    a0 = gl.allocate_shared_memory(gl.float8e4nv, [32, 128], sha)
    a1 = gl.allocate_shared_memory(gl.float8e4nv, [32, 128], sha)
    a2 = gl.allocate_shared_memory(gl.float8e4nv, [32, 128], sha)
    a3 = gl.allocate_shared_memory(gl.float8e4nv, [32, 128], sha)
    a4 = gl.allocate_shared_memory(gl.float8e4nv, [32, 128], sha)
    a5 = gl.allocate_shared_memory(gl.float8e4nv, [32, 128], sha)
    b0 = gl.allocate_shared_memory(gl.float8e4nv, [128, 32], shb)
    b1 = gl.allocate_shared_memory(gl.float8e4nv, [128, 32], shb)
    b2 = gl.allocate_shared_memory(gl.float8e4nv, [128, 32], shb)
    b3 = gl.allocate_shared_memory(gl.float8e4nv, [128, 32], shb)
    b4 = gl.allocate_shared_memory(gl.float8e4nv, [128, 32], shb)
    b5 = gl.allocate_shared_memory(gl.float8e4nv, [128, 32], shb)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b0, b_ptr + 0, bo, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a0, a_ptr + 0, ao, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b1, b_ptr + 128, bo, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a1, a_ptr + 128, ao, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b2, b_ptr + 256, bo, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a2, a_ptr + 256, ao, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b3, b_ptr + 384, bo, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a3, a_ptr + 384, ao, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b4, b_ptr + 512, bo, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a4, a_ptr + 512, ao, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b5, b_ptr + 640, bo, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a5, a_ptr + 640, ao, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.commit_group()
    rm = gl.arange(0, 32, layout=gl.SliceLayout(1, mf))
    scg = (pid_n >> 2) * 18
    acc = gl.zeros((32, 32), dtype=gl.float32, layout=mf)
    z = gl.zeros((32, 32), dtype=gl.float32, layout=mf)
    s0x = gl.load(as_ptr + rm * 18 + 0)
    s1x = gl.load(as_ptr + rm * 18 + 1)
    s2x = gl.load(as_ptr + rm * 18 + 2)
    s3x = gl.load(as_ptr + rm * 18 + 3)
    s4x = gl.load(as_ptr + rm * 18 + 4)
    s5x = gl.load(as_ptr + rm * 18 + 5)
    t0x, t1x, t2x, t3x, t4x, t5x = _sl6(bs_ptr + scg + 0)
    gl.amd.cdna4.async_copy.wait_group(1)
    # wait_group is wave-local; publish the cooperative LDS page CTA-wide.
    gl.barrier()
    aq0x = gl.amd.cdna4.async_copy.load_shared_relaxed(a0, layout=da)
    bq0x = gl.amd.cdna4.async_copy.load_shared_relaxed(b0, layout=db)
    q0x = gl.amd.cdna4.mfma_scaled(aq0x, None, "e4m3", bq0x, None, "e4m3", z)
    aq1x = gl.amd.cdna4.async_copy.load_shared_relaxed(a1, layout=da)
    bq1x = gl.amd.cdna4.async_copy.load_shared_relaxed(b1, layout=db)
    q1x = gl.amd.cdna4.mfma_scaled(aq1x, None, "e4m3", bq1x, None, "e4m3", z)
    q1x = _keep_running_sum_2304(q1x)
    aq2x = gl.amd.cdna4.async_copy.load_shared_relaxed(a2, layout=da)
    bq2x = gl.amd.cdna4.async_copy.load_shared_relaxed(b2, layout=db)
    q2x = gl.amd.cdna4.mfma_scaled(aq2x, None, "e4m3", bq2x, None, "e4m3", z)
    gl.barrier()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b0, b_ptr + 768, bo, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a0, a_ptr + 768, ao, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b1, b_ptr + 896, bo, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a1, a_ptr + 896, ao, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b2, b_ptr + 1024, bo, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a2, a_ptr + 1024, ao, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.wait_group(1)
    gl.barrier()
    aq3x = gl.amd.cdna4.async_copy.load_shared_relaxed(a3, layout=da)
    bq3x = gl.amd.cdna4.async_copy.load_shared_relaxed(b3, layout=db)
    q3x = gl.amd.cdna4.mfma_scaled(aq3x, None, "e4m3", bq3x, None, "e4m3", z)
    aq4x = gl.amd.cdna4.async_copy.load_shared_relaxed(a4, layout=da)
    bq4x = gl.amd.cdna4.async_copy.load_shared_relaxed(b4, layout=db)
    q4x = gl.amd.cdna4.mfma_scaled(aq4x, None, "e4m3", bq4x, None, "e4m3", z)
    aq5x = gl.amd.cdna4.async_copy.load_shared_relaxed(a5, layout=da)
    bq5x = gl.amd.cdna4.async_copy.load_shared_relaxed(b5, layout=db)
    q5x = gl.amd.cdna4.mfma_scaled(aq5x, None, "e4m3", bq5x, None, "e4m3", z)
    gl.barrier()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b3, b_ptr + 1152, bo, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a3, a_ptr + 1152, ao, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b4, b_ptr + 1280, bo, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a4, a_ptr + 1280, ao, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b5, b_ptr + 1408, bo, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a5, a_ptr + 1408, ao, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.commit_group()
    acc = gl.fma(q0x * s0x[:, None], t0x, acc)
    acc = _keep_running_sum_2304(acc)
    acc = gl.fma(q1x * s1x[:, None], t1x, acc)
    acc = _keep_running_sum_2304(acc)
    acc = gl.fma(q2x * s2x[:, None], t2x, acc)
    acc = _keep_running_sum_2304(acc)
    acc = gl.fma(q3x * s3x[:, None], t3x, acc)
    acc = _keep_running_sum_2304(acc)
    acc = gl.fma(q4x * s4x[:, None], t4x, acc)
    acc = _keep_running_sum_2304(acc)
    acc = gl.fma(q5x * s5x[:, None], t5x, acc)
    acc = _keep_running_sum_2304(acc)
    s0x = gl.load(as_ptr + rm * 18 + 6)
    s1x = gl.load(as_ptr + rm * 18 + 7)
    s2x = gl.load(as_ptr + rm * 18 + 8)
    s3x = gl.load(as_ptr + rm * 18 + 9)
    s4x = gl.load(as_ptr + rm * 18 + 10)
    s5x = gl.load(as_ptr + rm * 18 + 11)
    t0x, t1x, t2x, t3x, t4x, t5x = _sl6(bs_ptr + scg + 6)
    gl.amd.cdna4.async_copy.wait_group(1)
    gl.barrier()
    aq0x = gl.amd.cdna4.async_copy.load_shared_relaxed(a0, layout=da)
    bq0x = gl.amd.cdna4.async_copy.load_shared_relaxed(b0, layout=db)
    q0x = gl.amd.cdna4.mfma_scaled(aq0x, None, "e4m3", bq0x, None, "e4m3", z)
    aq1x = gl.amd.cdna4.async_copy.load_shared_relaxed(a1, layout=da)
    bq1x = gl.amd.cdna4.async_copy.load_shared_relaxed(b1, layout=db)
    q1x = gl.amd.cdna4.mfma_scaled(aq1x, None, "e4m3", bq1x, None, "e4m3", z)
    aq2x = gl.amd.cdna4.async_copy.load_shared_relaxed(a2, layout=da)
    bq2x = gl.amd.cdna4.async_copy.load_shared_relaxed(b2, layout=db)
    q2x = gl.amd.cdna4.mfma_scaled(aq2x, None, "e4m3", bq2x, None, "e4m3", z)
    gl.barrier()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b0, b_ptr + 1536, bo, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a0, a_ptr + 1536, ao, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b1, b_ptr + 1664, bo, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a1, a_ptr + 1664, ao, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b2, b_ptr + 1792, bo, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a2, a_ptr + 1792, ao, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.wait_group(1)
    gl.barrier()
    aq3x = gl.amd.cdna4.async_copy.load_shared_relaxed(a3, layout=da)
    bq3x = gl.amd.cdna4.async_copy.load_shared_relaxed(b3, layout=db)
    q3x = gl.amd.cdna4.mfma_scaled(aq3x, None, "e4m3", bq3x, None, "e4m3", z)
    aq4x = gl.amd.cdna4.async_copy.load_shared_relaxed(a4, layout=da)
    bq4x = gl.amd.cdna4.async_copy.load_shared_relaxed(b4, layout=db)
    q4x = gl.amd.cdna4.mfma_scaled(aq4x, None, "e4m3", bq4x, None, "e4m3", z)
    aq5x = gl.amd.cdna4.async_copy.load_shared_relaxed(a5, layout=da)
    bq5x = gl.amd.cdna4.async_copy.load_shared_relaxed(b5, layout=db)
    q5x = gl.amd.cdna4.mfma_scaled(aq5x, None, "e4m3", bq5x, None, "e4m3", z)
    gl.barrier()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b3, b_ptr + 1920, bo, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a3, a_ptr + 1920, ao, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b4, b_ptr + 2048, bo, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a4, a_ptr + 2048, ao, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b5, b_ptr + 2176, bo, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a5, a_ptr + 2176, ao, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.commit_group()
    acc = gl.fma(q0x * s0x[:, None], t0x, acc)
    acc = _keep_running_sum_2304(acc)
    acc = gl.fma(q1x * s1x[:, None], t1x, acc)
    acc = _keep_running_sum_2304(acc)
    acc = gl.fma(q2x * s2x[:, None], t2x, acc)
    acc = _keep_running_sum_2304(acc)
    acc = gl.fma(q3x * s3x[:, None], t3x, acc)
    acc = _keep_running_sum_2304(acc)
    acc = gl.fma(q4x * s4x[:, None], t4x, acc)
    acc = _keep_running_sum_2304(acc)
    acc = gl.fma(q5x * s5x[:, None], t5x, acc)
    acc = _keep_running_sum_2304(acc)
    s0x = gl.load(as_ptr + rm * 18 + 12)
    s1x = gl.load(as_ptr + rm * 18 + 13)
    s2x = gl.load(as_ptr + rm * 18 + 14)
    s3x = gl.load(as_ptr + rm * 18 + 15)
    s4x = gl.load(as_ptr + rm * 18 + 16)
    s5x = gl.load(as_ptr + rm * 18 + 17)
    t0x, t1x, t2x, t3x, t4x, t5x = _sl6(bs_ptr + scg + 12)
    gl.amd.cdna4.async_copy.wait_group(1)
    gl.barrier()
    aq0x = gl.amd.cdna4.async_copy.load_shared_relaxed(a0, layout=da)
    bq0x = gl.amd.cdna4.async_copy.load_shared_relaxed(b0, layout=db)
    q0x = gl.amd.cdna4.mfma_scaled(aq0x, None, "e4m3", bq0x, None, "e4m3", z)
    aq1x = gl.amd.cdna4.async_copy.load_shared_relaxed(a1, layout=da)
    bq1x = gl.amd.cdna4.async_copy.load_shared_relaxed(b1, layout=db)
    q1x = gl.amd.cdna4.mfma_scaled(aq1x, None, "e4m3", bq1x, None, "e4m3", z)
    aq2x = gl.amd.cdna4.async_copy.load_shared_relaxed(a2, layout=da)
    bq2x = gl.amd.cdna4.async_copy.load_shared_relaxed(b2, layout=db)
    q2x = gl.amd.cdna4.mfma_scaled(aq2x, None, "e4m3", bq2x, None, "e4m3", z)
    gl.amd.cdna4.async_copy.wait_group(0)
    gl.barrier()
    aq3x = gl.amd.cdna4.async_copy.load_shared_relaxed(a3, layout=da)
    bq3x = gl.amd.cdna4.async_copy.load_shared_relaxed(b3, layout=db)
    q3x = gl.amd.cdna4.mfma_scaled(aq3x, None, "e4m3", bq3x, None, "e4m3", z)
    aq4x = gl.amd.cdna4.async_copy.load_shared_relaxed(a4, layout=da)
    bq4x = gl.amd.cdna4.async_copy.load_shared_relaxed(b4, layout=db)
    q4x = gl.amd.cdna4.mfma_scaled(aq4x, None, "e4m3", bq4x, None, "e4m3", z)
    aq5x = gl.amd.cdna4.async_copy.load_shared_relaxed(a5, layout=da)
    bq5x = gl.amd.cdna4.async_copy.load_shared_relaxed(b5, layout=db)
    q5x = gl.amd.cdna4.mfma_scaled(aq5x, None, "e4m3", bq5x, None, "e4m3", z)
    acc = gl.fma(q0x * s0x[:, None], t0x, acc)
    acc = _keep_running_sum_2304(acc)
    acc = gl.fma(q1x * s1x[:, None], t1x, acc)
    acc = _keep_running_sum_2304(acc)
    acc = gl.fma(q2x * s2x[:, None], t2x, acc)
    acc = _keep_running_sum_2304(acc)
    acc = gl.fma(q3x * s3x[:, None], t3x, acc)
    acc = _keep_running_sum_2304(acc)
    acc = gl.fma(q4x * s4x[:, None], t4x, acc)
    acc = _keep_running_sum_2304(acc)
    acc = gl.fma(q5x * s5x[:, None], t5x, acc)
    acc = _keep_running_sum_2304(acc)
    cn = pid_n * 32 + gl.arange(0, 32, layout=gl.SliceLayout(0, mf))
    gl.store(
        out_ptr + rm[:, None] * 7168 + cn[None, :],
        acc.to(out_ptr.dtype.element_ty),
        cache_modifier=".cg",
    )


def block_scaled_mm_n7168_k2304_m32(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run the exact M=32, N=7168, K=2304 specialization."""
    assert a.shape == (32, 2304) and b.shape == (7168, 2304)
    assert a_scale.shape == (32, 18) and b_scale.shape == (56, 18)
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn
    assert a_scale.dtype == torch.float32 and b_scale.dtype == torch.float32
    assert all(x.is_contiguous() for x in (a, b, a_scale, b_scale))
    assert out.shape == (32, 7168)
    assert out.dtype == torch.bfloat16
    assert out.device == a.device and out.is_contiguous()
    _g_gemm_2304_async[(224,)](
        a, b, a_scale, b_scale, out, num_warps=4, num_stages=3, waves_per_eu=5
    )
    return out


__all__ = ["block_scaled_mm_n7168_k2304_m32"]
