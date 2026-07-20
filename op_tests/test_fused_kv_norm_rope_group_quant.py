#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter op-test for ``fused_kv_norm_rope_group_quant`` (DeepSeek-V4-Pro KV-only path A).

Validates the fused KV RMSNorm + GPT-J RoPE + 1xG e8m0 FP8 group-quant kernel
against a pure-torch reference modeled on ``Attention.forward`` from
DeepSeek-V4-Pro ``model.py`` (lines 682-686)::

    kv = self.wkv(x)                                # [B*S, head_dim] bf16
    kv = self.kv_norm(kv)                           # RMSNorm
    apply_rotary_emb(kv[..., -rd:], freqs_cis)      # GPT-J RoPE on PE tail
    act_quant(kv[..., :-rd], 64, ..., inplace=True) # FP8 1xG e8m0 on NoPE only

V4-Pro layout: NoPE fp8 (1x64 e8m0 group scale), PE bf16 (NOT quantized).
Outputs are scattered into a PAGED KV cache via slot_mapping (caches shaped
[num_blocks, page_size, NK, entry]); the test uses a random permutation of
distinct slots and gathers rows back for the accuracy check. NK (num_kv_heads)
is 1 (MLA latent KV), matching the kernel's MQA hard-coding.

NOTE: the wrapper currently issues a dummy bf16 Q wave alongside the K wave
(see ``aiter/ops/fused_qk_norm_rope_cache_quant.py::fused_kv_norm_rope_group_quant``),
so reported us / GB-s include that overhead until a true K-only kernel lands.

By default sweeps every (head_dim, rot_dim, group_size) shape the K-only kernel
supports (KV_KERNEL_SUPPORTED_SHAPES, mirrors KV_K_ONLY_DISPATCH_TABLE on the
C++ side); pass --D/--RD/--G to pin a single shape.

Usage::

    python op_tests/test_fused_kv_norm_rope_group_quant.py
    python op_tests/test_fused_kv_norm_rope_group_quant.py -T 64 256 1024
    python op_tests/test_fused_kv_norm_rope_group_quant.py --D 192 --RD 64
    python op_tests/test_fused_kv_norm_rope_group_quant.py --neox
"""

import argparse
import random

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility.fp4_utils import f32_to_mx_e8m0_scale
from aiter.utility.mx_types import MxDtypeInt, MxScaleRoundModeInt

torch.set_default_device("cuda")

_FP8 = dtypes.fp8
_DEV = "cuda"
# MI355X HBM3e peak. Used only for the "%peak" perf column.
_PEAK_BW_GBPS = 8000.0


# ============================================================================
# Reference (pure torch) -- mirrors Attention.forward L682-L686 in model.py.
# ============================================================================


def _cos_sin(max_pos, rope_dim, dtype):
    """Build a RoPE cos/sin table: [max_pos, rope_dim/2]."""
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, rope_dim, 2, device=_DEV).float() / rope_dim)
    )
    freqs = torch.einsum(
        "i,j->ij", torch.arange(max_pos, device=_DEV).float(), inv_freq
    )
    return freqs.cos().to(dtype).contiguous(), freqs.sin().to(dtype).contiguous()


def _apply_gptj_rope(pe, cos, sin, pos, *, is_neox):
    """Rotate the PE part [.., rope_dim].

    is_neox: half-split pairing (x=[:rd/2], y=[rd/2:]); else GPT-J adjacent
    pairing (x=even, y=odd). Matches the kernel's RoPE convention; equivalent
    to ``apply_rotary_emb`` in V4-Pro model.py for the appropriate freqs_cis
    layout.
    """
    T, n_heads = pe.shape[0], pe.shape[1]
    rope_dim = pe.shape[-1]
    c = cos[pos].float().view(T, 1, rope_dim // 2)
    s = sin[pos].float().view(T, 1, rope_dim // 2)
    if is_neox:
        x, y = pe[..., : rope_dim // 2], pe[..., rope_dim // 2 :]
        return torch.cat([x * c - y * s, x * s + y * c], -1)
    x, y = pe[..., 0::2], pe[..., 1::2]
    return torch.stack([x * c - y * s, x * s + y * c], -1).reshape(T, n_heads, rope_dim)


def _kv_only_ref(kv, kv_weight, cos, sin, pos, eps, *, is_neox, group_size):
    """Pure-torch KV-only reference for one stream:
        RMSNorm(kv * kv_weight) -> split nope/pe -> RoPE(pe) ->
        nope 1xG e8m0 fp8 quant, pe stays bf16.

    Returns ``(nope_fp8, scale_e8m0[T,NK,n_groups] uint8, pe_bf16)``.
    """
    T, n_heads, head_dim = kv.shape
    rope_dim = cos.shape[-1] * 2
    nope_dim = head_dim - rope_dim
    n_groups = nope_dim // group_size

    normed = kv.float() * torch.rsqrt(kv.float().pow(2).mean(-1, keepdim=True) + eps)
    normed = normed * kv_weight.float()
    nope, pe = normed[..., :nope_dim], normed[..., nope_dim:]
    pe_rotated = _apply_gptj_rope(pe, cos, sin, pos, is_neox=is_neox)

    # nope: per-group amax -> e8m0 scale (MX RoundUp, FP8 E4M3) -> fp8.
    amax = (
        nope.reshape(T, n_heads, n_groups, group_size).abs().amax(-1).clamp_min(1e-12)
    )
    scale_e8m0 = f32_to_mx_e8m0_scale(
        amax, mode=MxScaleRoundModeInt.RoundUp, dtype=MxDtypeInt.FP8_E4M3
    ).view(torch.uint8)
    inv_scale = 1.0 / (scale_e8m0.to(torch.int32) << 23).view(torch.float32)
    nope_fp8 = (
        nope
        * inv_scale.unsqueeze(-1)
        .expand(T, n_heads, n_groups, group_size)
        .reshape(T, n_heads, nope_dim)
    ).to(_FP8)
    return nope_fp8, scale_e8m0, pe_rotated.to(torch.bfloat16)


# ============================================================================
# Main test
# ============================================================================


@benchmark()
def test_fused_kv_norm_rope_group_quant(T, D, RD, *, is_neox, G):
    """One config: T tokens, head_dim=D, rope_dim=RD, group_size=G, NK=1."""
    NK = 1  # V4-Pro MLA: exactly one latent KV "head"
    torch.manual_seed(0)
    random.seed(0)
    nope = D - RD
    eps = 1e-6
    n_groups = nope // G  # e.g. 7 for D=512, RD=64, G=64
    entry = D  # nope_scale_buff = head_dim bytes (nope + 2*n_groups + pad)

    cos, sin = _cos_sin(max(T, 64) + 4, RD, torch.bfloat16)
    pos = torch.randint(0, cos.shape[0] - 1, (T,), dtype=torch.int64, device=_DEV)
    kv = (torch.randn(T, NK, D, device=_DEV) * 0.1).bfloat16()
    kv_weight = (torch.randn(D, device=_DEV).abs() + 0.5).bfloat16()

    # Reference
    ref_nope, ref_scale, ref_pe = _kv_only_ref(
        kv, kv_weight, cos, sin, pos, eps, is_neox=is_neox, group_size=G
    )

    # Paged KV cache + slot_mapping. Use a random permutation of distinct slots
    # (no collisions) so the test exercises a non-identity paged scatter. Caches
    # are [num_blocks, page_size, entry] (MQA: no num_kv_heads dim); the kernel
    # writes each token to slot_mapping[token] = block*page_size + offset.
    # Zero-init so unwritten slots AND each entry's trailing pad read back as zero
    # (asm-reader contract).
    page_size = 1
    num_blocks = (T + page_size - 1) // page_size + 1  # a little slack
    num_slots = num_blocks * page_size
    slot_mapping = torch.randperm(num_slots, device=_DEV)[:T].to(torch.int64)
    k_nope_scale_buff = torch.zeros(
        num_blocks, page_size, entry, dtype=_FP8, device=_DEV
    )
    k_rope_buff = torch.zeros(
        num_blocks, page_size, RD, dtype=torch.bfloat16, device=_DEV
    )
    (k_nope_scale_buff, k_rope_buff), us = run_perftest(
        aiter.fused_kv_norm_rope_group_quant,
        kv,
        kv_weight,
        pos,
        slot_mapping,
        cos,
        sin,
        eps,
        k_nope_scale_buff,
        k_rope_buff,
        is_neox=is_neox,
        quant_group_size=G,
        scale_dtype="e8m0",
    )

    # Gather per-token rows back from the paged cache via slot_mapping, then add
    # back the unit NK axis the token-indexed accuracy check expects ([T, NK, ...]).
    nope_scale_buff = k_nope_scale_buff.view(num_slots, entry)[slot_mapping].unsqueeze(
        1
    )
    rope_buff = k_rope_buff.view(num_slots, RD)[slot_mapping].unsqueeze(1)

    # --- Accuracy ---
    # K nope fp8 from nope_scale_buff[..., 0:nope]; e8m0 scale @[nope:nope+2*n_groups),
    # written as each tile-scale duplicated x2 (s0,s0,s1,s1,...). Take the first of
    # each pair for dequant; verify BOTH halves equal the reference scale.
    k_nope_got = nope_scale_buff[..., :nope]
    k_scale_pairs = (
        nope_scale_buff.view(torch.uint8)[..., nope : nope + 2 * n_groups]
        .contiguous()
        .reshape(T, NK, n_groups, 2)
    )
    got_k_scale_f32 = (k_scale_pairs[..., 0].to(torch.int32) << 23).view(torch.float32)
    ref_k_scale_f32 = (ref_scale.to(torch.int32) << 23).view(torch.float32)
    k_deq = k_nope_got.float() * got_k_scale_f32.unsqueeze(-1).expand(
        T, NK, n_groups, G
    ).reshape(T, NK, nope)
    ref_k_deq = ref_nope.float() * ref_k_scale_f32.unsqueeze(-1).expand(
        T, NK, n_groups, G
    ).reshape(T, NK, nope)
    err_k = checkAllclose(k_deq, ref_k_deq, atol=0.05, rtol=0.02, msg="K-nope fp8")
    checkAllclose(
        k_scale_pairs[..., 0].float(),
        ref_scale.float(),
        atol=0.0,
        rtol=0.0,
        msg="K scale e8m0",
    )
    checkAllclose(
        k_scale_pairs[..., 1].float(),
        ref_scale.float(),
        atol=0.0,
        rtol=0.0,
        msg="K scale e8m0 dup",
    )

    # K pe bf16 (NOT quantized) from the separate rope_buff
    err_kpe = checkAllclose(
        rope_buff.float(), ref_pe.float(), atol=0.01, rtol=0.01, msg="K-pe bf16"
    )

    # --- Bandwidth (effective): read kv + kv_weight, write K (nope+scale+rope) ---
    # The wrapper still pays for a dummy bf16 Q wave alongside (see module
    # docstring); that read+write is included in the kernel time but excluded
    # from this "useful" byte count, so reported %peak is conservative.
    bytes_in = T * NK * D * 2 + D * 2
    bytes_out = T * NK * (nope + 2 * n_groups) + T * NK * RD * 2
    gbps = (bytes_in + bytes_out) / (us * 1e-6) / 1e9

    return {
        "D": D,
        "RD": RD,
        "G": G,
        "hip_us": round(us, 3),
        "GB/s": round(gbps, 0),
        "%peak": round(gbps / _PEAK_BW_GBPS * 100, 1),
        "err_k": err_k,
        "err_kpe": err_kpe,
    }


# ============================================================================
# argparse + matrix sweep
# ============================================================================

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="aiter test for fused_kv_norm_rope_group_quant (V4-Pro KV-only path A).",
)
parser.add_argument(
    "-T",
    "--T",
    type=int,
    nargs="*",
    default=[4, 16, 64, 256, 1024, 4096, 16384],
    help="token-count sweep. e.g. -T 4 64 1024",
)
parser.add_argument(
    "--D",
    type=int,
    default=None,
    help="head_dim override (single shape). Default: sweep all supported shapes.",
)
parser.add_argument(
    "--RD", type=int, default=None, help="rope_head_dim override (used with --D)."
)
parser.add_argument(
    "--G", type=int, default=None, help="group_size override (used with --D)."
)
parser.add_argument(
    "--neox", action="store_true", help="also sweep is_neox=True (default: GPT-J only)."
)
args = parser.parse_args()

neox_modes = [False, True] if args.neox else [False]

# (head_dim, rot_dim, group_size) shapes the K-only kernel supports today; mirrors
# KV_K_ONLY_DISPATCH_TABLE in csrc/kernels/fused_qk_norm_rope_cache_quant.cu.
KV_KERNEL_SUPPORTED_SHAPES = (
    (512, 64, 64),  # DeepSeek V4-Pro (default)
    (192, 64, 64),  # DeepSeek V2 / V3 MLA
    (384, 128, 64),  # head_dim=384, rope=128 (Qwen-style)
)

if args.D is not None:
    shapes = [(args.D, args.RD or 64, args.G or 64)]
else:
    shapes = list(KV_KERNEL_SUPPORTED_SHAPES)

rows = []
for neox in neox_modes:
    for D, RD, G in shapes:
        for T in args.T:
            rows.append(
                test_fused_kv_norm_rope_group_quant(
                    T,
                    D,
                    RD,
                    is_neox=neox,
                    G=G,
                )
            )

df = pd.DataFrame(rows)
aiter.logger.info(
    "fused_kv_norm_rope_group_quant (V4-Pro KV-only) summary (markdown):\n%s",
    df.to_markdown(index=False),
)
