#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Accuracy tests for the FlyDSL-alignment extensions of
``aiter.fused_qk_norm_rope_group_quant_cache``:

  * Q fp8 group quant (item 1)
  * optional ``q_weight`` (item 2)
  * Q ``quant_group_size`` ? {32, 64, 128} (item 5)
  * Q ``scale_dtype`` ? {"e8m0", "fp32"} (item 6)

The K-side path (paged KV cache, e8m0 inline scale) is unchanged and is
already covered by ``test_fused_qk_norm_rope_group_quant_cache.py``.

Usage:
    python op_tests/test_fused_qk_norm_rope_group_quant_cache_qquant.py
"""

import itertools
import random

import torch

import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose

_FP8 = dtypes.fp8
_FP8_MAX = float(torch.finfo(_FP8).max)
_DEVICE = "cuda"


def _build_cos_sin(max_pos, rope_dim, dtype):
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, rope_dim, 2, device=_DEVICE).float() / rope_dim)
    )
    pos_range = torch.arange(max_pos, device=_DEVICE).float()
    freqs = torch.einsum("i,j->ij", pos_range, inv_freq)
    return freqs.cos().to(dtype).contiguous(), freqs.sin().to(dtype).contiguous()


def _torch_ref(
    q,
    kv,
    k_weight,
    cos_cache,
    sin_cache,
    positions,
    eps,
    *,
    is_neox,
    is_nope_first,
    q_weight=None,
    quant_group_size=64,
    scale_dtype="e8m0",
):
    """Torch reference for the FULL kernel path, including the new Q-quant
    extensions. Returns ``(q_out_fp8, q_scale, k_pe_rotated, k_nope_quant)``.
    """
    T, H, D = q.shape
    RD = cos_cache.shape[-1] * 2
    nope_dim = D - RD
    NG = D // quant_group_size

    # Compute the rms-normed + (optional) weighted, post-rope Q value, in f32.
    q_f = q.float()
    q_rms = torch.rsqrt(q_f.pow(2).mean(-1, keepdim=True) + eps)
    q_n = q_f * q_rms
    if q_weight is not None:
        q_n = q_n * q_weight.float()

    if is_nope_first:
        q_nope = q_n[..., :nope_dim]
        q_pe_in = q_n[..., nope_dim:]
    else:
        q_pe_in = q_n[..., :RD]
        q_nope = q_n[..., RD:]

    # GPT-J vs NeoX RoPE on q_pe_in
    cos = cos_cache[positions].float()  # [T, RD/2]
    sin = sin_cache[positions].float()
    if is_neox:
        # NeoX: x_{i} = x[i] (i in [0, RD/2)), y_i = x[i + RD/2]
        x = q_pe_in[..., : RD // 2]
        y = q_pe_in[..., RD // 2 :]
        c = cos.view(T, 1, RD // 2)
        s = sin.view(T, 1, RD // 2)
        x_new = x * c - y * s
        y_new = x * s + y * c
        q_pe_rot = torch.cat([x_new, y_new], dim=-1)
    else:
        x = q_pe_in[..., 0::2]
        y = q_pe_in[..., 1::2]
        c = cos.view(T, 1, RD // 2)
        s = sin.view(T, 1, RD // 2)
        new_x = x * c - y * s
        new_y = x * s + y * c
        q_pe_rot = torch.stack([new_x, new_y], dim=-1).reshape(T, H, RD)

    if is_nope_first:
        q_full_f = torch.cat([q_nope, q_pe_rot], dim=-1)
    else:
        q_full_f = torch.cat([q_pe_rot, q_nope], dim=-1)

    # Group quant. Kernel uses POST-rope amax per group (no SQRT2 padding needed).
    q_grouped = q_full_f.reshape(T, H, NG, quant_group_size)
    am = q_grouped.abs().amax(-1).clamp_min(1e-12)
    group_scale = am / _FP8_MAX  # [T, H, NG]

    if scale_dtype == "e8m0":
        # Mirror kernel: e8m0 round-up encoding
        u32 = group_scale.float().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        exp = (u32 >> 23) & 0xFF
        has_mantissa = (u32 & 0x7FFFFF) != 0
        exp = exp + has_mantissa.to(torch.int64)
        e8m0_byte = exp.to(torch.uint8)
        # Reconstruct float scale = 2^(exp - 127); kernel uses 1.0 / __builtin_bit_cast(float, exp << 23)
        e8m0_u32 = exp.to(torch.int32) << 23
        scale_recon = e8m0_u32.view(torch.float32)
        inv_scale = 1.0 / scale_recon
        # broadcast across the group
        inv_scale_full = (
            inv_scale.unsqueeze(-1)
            .expand(*scale_recon.shape, quant_group_size)
            .reshape(T, H, D)
        )
        q_fp8 = (q_full_f * inv_scale_full).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)
        return q_fp8, e8m0_byte, group_scale, inv_scale
    elif scale_dtype == "fp32":
        # In the kernel, even fp32 scale uses the e8m0-rounded-up exponent for inv_scale
        # (because Q_SCALE_FP32 branch only changes what gets STORED, not the multiplier).
        # So store the reconstructed e8m0 float scale (matches kernel) -- not raw am/FP8_MAX.
        u32 = group_scale.float().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        exp = (u32 >> 23) & 0xFF
        has_mantissa = (u32 & 0x7FFFFF) != 0
        exp = exp + has_mantissa.to(torch.int64)
        e8m0_u32 = exp.to(torch.int32) << 23
        scale_recon = e8m0_u32.view(torch.float32)
        inv_scale = 1.0 / scale_recon
        inv_scale_full = (
            inv_scale.unsqueeze(-1)
            .expand(*scale_recon.shape, quant_group_size)
            .reshape(T, H, D)
        )
        q_fp8 = (q_full_f * inv_scale_full).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)
        return q_fp8, scale_recon, group_scale, inv_scale
    else:
        raise ValueError(scale_dtype)


def _one_test(
    T,
    H,
    *,
    is_neox=False,
    is_nope_first=True,
    q_weighted=False,
    quant_group_size=64,
    scale_dtype="e8m0",
    num_kv_heads=1,
):
    torch.manual_seed(0)
    random.seed(0)
    D = 512
    RD = 64
    block_size = 1
    num_blocks = max(1, T + 4)
    eps = 1e-6

    cos, sin = _build_cos_sin(max(T, 64) + 4, RD, torch.bfloat16)
    pos = torch.randint(0, cos.shape[0] - 1, (T,), dtype=torch.int64, device=_DEVICE)
    slot_mapping = torch.tensor(
        random.sample(range(num_blocks * block_size), T),
        dtype=torch.long,
        device=_DEVICE,
    )

    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=_DEVICE) * 0.1
    kv = torch.randn(T, num_kv_heads, D, dtype=torch.bfloat16, device=_DEVICE) * 0.1
    k_weight = torch.ones(D, dtype=torch.bfloat16, device=_DEVICE)
    q_weight = (
        (torch.randn(D, dtype=torch.bfloat16, device=_DEVICE).abs() + 0.5)
        if q_weighted
        else None
    )

    kv_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, D, dtype=torch.bfloat16, device=_DEVICE
    )
    NG = D // quant_group_size
    q_out_fp8 = torch.empty((T, H, D), dtype=_FP8, device=_DEVICE)
    q_scale_dtype = torch.float32 if scale_dtype == "fp32" else torch.uint8
    q_scale = torch.zeros((T, H, NG), dtype=q_scale_dtype, device=_DEVICE)
    k_pe_out = torch.empty(T, num_kv_heads, RD, dtype=torch.bfloat16, device=_DEVICE)

    aiter.fused_qk_norm_rope_group_quant_cache(
        q,
        kv,
        k_pe_out,
        k_weight,
        kv_cache,
        q_out_fp8,
        slot_mapping,
        pos,
        cos,
        sin,
        eps,
        is_neox,
        is_nope_first,
        q_weight=q_weight,
        q_scale=q_scale,
        quant_group_size=quant_group_size,
        scale_dtype=scale_dtype,
    )

    # Compute torch reference and compare
    ref_q_fp8, ref_q_scale, ref_group_scale, ref_inv_scale = _torch_ref(
        q,
        kv,
        k_weight,
        cos,
        sin,
        pos,
        eps,
        is_neox=is_neox,
        is_nope_first=is_nope_first,
        q_weight=q_weight,
        quant_group_size=quant_group_size,
        scale_dtype=scale_dtype,
    )

    # Dequant both to fp32 (using kernel's e8m0-derived inv_scale, broadcast over group).
    if scale_dtype == "e8m0":
        ker_scale_f = (q_scale.to(torch.int32) << 23).view(torch.float32)
        ref_scale_f = (ref_q_scale.to(torch.int32) << 23).view(torch.float32)
    else:
        ker_scale_f = q_scale.float()
        ref_scale_f = ref_q_scale.float()

    G = quant_group_size
    ker_scale_full = ker_scale_f.unsqueeze(-1).expand(T, H, NG, G).reshape(T, H, D)
    ref_scale_full = ref_scale_f.unsqueeze(-1).expand(T, H, NG, G).reshape(T, H, D)
    ker_deq = q_out_fp8.float() * ker_scale_full
    ref_deq = ref_q_fp8.float() * ref_scale_full

    name = (
        f"T={T} H={H} neox={is_neox} nope_first={is_nope_first} qw={q_weighted} "
        f"G={quant_group_size} scale={scale_dtype}"
    )
    err_scale = checkAllclose(
        ker_scale_f,
        ref_scale_f,
        atol=0.0,
        rtol=0.0,
        msg=f"[{name}] q_scale bit-identical",
    )
    err_deq = checkAllclose(
        ker_deq,
        ref_deq,
        atol=0.05,
        rtol=0.02,
        msg=f"[{name}] q_out dequant",
    )
    return err_scale, err_deq


def main():
    configs = list(
        itertools.product(
            [4, 32, 256],  # T
            [16, 128],  # H
            [False, True],  # q_weighted
            [32, 64, 128],  # quant_group_size
            ["e8m0", "fp32"],  # scale_dtype
        )
    )
    print(f"Running {len(configs)} configs (is_neox=False, is_nope_first=True)...")
    n_pass, n_fail = 0, 0
    for T, H, qw, g, sd in configs:
        try:
            err_s, err_d = _one_test(
                T,
                H,
                q_weighted=qw,
                quant_group_size=g,
                scale_dtype=sd,
                is_neox=False,
                is_nope_first=True,
            )
            n_pass += 1
        except Exception as e:
            print(f"FAIL: T={T} H={H} qw={qw} G={g} sd={sd}: {e}")
            n_fail += 1
    print(f"\n=== Summary: {n_pass} passed, {n_fail} failed ===")

    # Smoke test of is_neox=True and !is_nope_first
    print("\nSmoke tests of alternate RoPE/layout flags...")
    _one_test(32, 16, is_neox=True, is_nope_first=True)
    _one_test(32, 16, is_neox=False, is_nope_first=False)
    _one_test(32, 16, is_neox=True, is_nope_first=False)
    print("Smoke tests done.")


if __name__ == "__main__":
    main()
