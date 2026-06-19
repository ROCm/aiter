# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FlyDSL MX-FP4 a16w4 fused-MoE correctness test (Track A).

a16w4 = bf16 activation x MX-FP4 (E2M1, per-1x32 E8M0 block scale) weight.

On gfx942 (CDNA3) there is no scaled MX-FP4 MFMA, so the FlyDSL path
dequantizes the FP4 weight to bf16 in-kernel (``dequant_fp4_to_bf16``) and runs
the legacy bf16 MFMA hot loop -- the ``moe_gemm_2stage`` ``in_dtype="fp4_bf16"``
kernels.  This test exercises those (re-synced) vendored kernels directly and
validates both stages against a torch reference that decodes the E2M1 codes via
the canonical LUT and applies the per-32 power-of-two block scale.

Usage:
    python op_tests/test_flydsl_moe_a16w4_gfx942.py
    pytest op_tests/test_flydsl_moe_a16w4_gfx942.py
"""

import sys

import pytest
import torch

import flydsl.compiler as flyc
from aiter.fused_moe import moe_sorting
from aiter.ops.flydsl.kernels.moe_gemm_2stage import (
    compile_moe_gemm1,
    compile_moe_gemm2,
)
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
from aiter.ops.shuffle import shuffle_weight, shuffle_scale_for_int4

# Canonical MX-FP4 (E2M1) code -> magnitude LUT (matches fp4_utils.mxfp4_to_f32).
_E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _is_gfx942() -> bool:
    if not torch.cuda.is_available():
        return False
    return "gfx94" in torch.cuda.get_device_properties(0).gcnArchName.lower()


def _pack_shuffled_int8_to_packed_int4_no_perm(x_shuf_i8: torch.Tensor) -> torch.Tensor:
    """Pack a preshuffled int8 tensor into packed 4-bit bytes (2 codes/byte).

    Each contiguous 8-value block [v0..v7] -> 4 bytes:
      b0=(v4<<4)|v0, b1=(v5<<4)|v1, b2=(v6<<4)|v2, b3=(v7<<4)|v3.
    Matches the in-kernel unpack order (low nibble of byte i, high nibble of i).
    """
    flat = x_shuf_i8.contiguous().view(-1).to(torch.int16)
    assert flat.numel() % 8 == 0
    u = (flat & 0xF).to(torch.uint8).view(-1, 8)
    out = torch.empty((u.shape[0], 4), device=u.device, dtype=torch.uint8)
    out[:, 0] = u[:, 0] | (u[:, 4] << 4)
    out[:, 1] = u[:, 1] | (u[:, 5] << 4)
    out[:, 2] = u[:, 2] | (u[:, 6] << 4)
    out[:, 3] = u[:, 3] | (u[:, 7] << 4)
    return out.view(-1).to(torch.int8)


def _decode_w(codes_i8: torch.Tensor, scale_groups: torch.Tensor, group_size: int) -> torch.Tensor:
    """Decode E2M1 weight codes [E, N, K] and apply per-32 block scale.

    codes_i8: int8 codes 0..15, shape [E, N, K].
    scale_groups: pow2 f32, shape [E, K//group_size, N].
    Returns decoded (scaled) weights as f32 [E, N, K].
    """
    lut = torch.tensor(_E2M1_LUT, device=codes_i8.device, dtype=torch.float32)
    w_dec = lut[codes_i8.long()]  # [E, N, K]
    E, N, K = w_dec.shape
    # scale_groups[e, g, n] -> expand over the 32-wide K block, transpose to [E, N, K].
    sc = scale_groups.permute(0, 2, 1).reshape(E, N, K // group_size)
    sc = sc.repeat_interleave(group_size, dim=2)  # [E, N, K]
    return w_dec * sc


def _check(ref: torch.Tensor, out: torch.Tensor, label: str,
           rel_l2_max: float = 0.06):
    """Relative-L2 gate -- robust to MX-FP4's wide dynamic range (the elementwise
    error of a single fp4 weight is large, but the aggregate direction is tight)."""
    ref = ref.float()
    out = out.float()
    max_delta = (ref - out).abs().max().item()
    rel_l2 = ((out - ref).norm() / ref.norm().clamp_min(1e-30)).item()
    passed = rel_l2 <= rel_l2_max
    print(f"[{label}] rel_L2={rel_l2:.4f} (<= {rel_l2_max})  max_delta={max_delta:.3f} "
          f"-> {'PASS' if passed else 'FAIL'}")
    print(f"    ref : {ref.reshape(-1)[:6].tolist()}")
    print(f"    out : {out.reshape(-1)[:6].tolist()}")
    return passed


def _gen_common(tokens, model_dim, inter_dim, experts, topk, tile_m, seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dev = torch.device("cuda")
    s = 0.2
    x = (torch.randn((tokens, model_dim), device=dev, dtype=torch.float32) * s).to(torch.bfloat16)
    score = torch.rand((tokens, experts), device=dev, dtype=torch.float32)
    topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, experts, model_dim, torch.bfloat16, tile_m
    )
    return dict(
        dev=dev, x=x, topk_ids=topk_ids, topk_weights=topk_weights,
        sorted_ids=sorted_ids, sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids, num_valid_ids=num_valid_ids,
        blocks=int(sorted_expert_ids.shape[0]),
    )


def _prep_w4(codes_i8: torch.Tensor):
    """Preshuffle + pack E2M1 codes [E, N, K] for the kernel (int4-style layout)."""
    E, N, K = codes_i8.shape
    shuf = shuffle_weight(codes_i8.view(torch.int8), layout=(16, 16))
    packed = _pack_shuffled_int8_to_packed_int4_no_perm(shuf).contiguous()
    return packed


def _gen_w1(dev, experts, inter_dim, model_dim, group_size):
    """Random MX-FP4 stage1 weight: E2M1 codes + per-32 pow2 (E8M0) group scale."""
    N1 = 2 * inter_dim
    codes = torch.randint(0, 16, (experts, N1, model_dim), device=dev, dtype=torch.int8)
    exps = torch.randint(-2, 3, (experts, model_dim // group_size, N1), device=dev)
    scale_groups = torch.pow(2.0, exps.to(torch.float32))
    kernel = _prep_w4(codes)
    scale_1d = shuffle_scale_for_int4(scale_groups, group_size=group_size).view(-1).contiguous()
    return codes, scale_groups, kernel, scale_1d


def _gen_w2(dev, experts, inter_dim, model_dim, group_size):
    codes = torch.randint(0, 16, (experts, model_dim, inter_dim), device=dev, dtype=torch.int8)
    exps = torch.randint(-2, 3, (experts, inter_dim // group_size, model_dim), device=dev)
    scale_groups = torch.pow(2.0, exps.to(torch.float32))
    kernel = _prep_w4(codes)
    scale_1d = shuffle_scale_for_int4(scale_groups, group_size=group_size).view(-1).contiguous()
    return codes, scale_groups, kernel, scale_1d


def _ref_stage1(g, w1_codes, scale_w1_groups, inter_dim, group_size):
    dev = g["dev"]
    w1_scaled = _decode_w(w1_codes, scale_w1_groups, group_size).to(torch.bfloat16)
    x = g["x"]
    tokens, topk = x.shape[0], g["topk_ids"].shape[1]
    ref = torch.empty((tokens, topk, inter_dim), device=dev, dtype=torch.float32)
    for t in range(tokens):
        for j in range(topk):
            e = int(g["topk_ids"][t, j])
            acc = x[t].float() @ w1_scaled[e].float().t()  # [2*inter]
            gate, up = acc[:inter_dim], acc[inter_dim:]
            ref[t, j] = torch.nn.functional.silu(gate) * up
    return ref


def _ref_stage2(g, a2, w2_codes, scale_w2_groups, model_dim, group_size):
    dev = g["dev"]
    w2_scaled = _decode_w(w2_codes, scale_w2_groups, group_size).to(torch.bfloat16)
    tokens, topk = a2.shape[0], a2.shape[1]
    # doweight off / unit weights (topk=1): kernel sums slot down-projections.
    ref = torch.zeros((tokens, model_dim), device=dev, dtype=torch.float32)
    for t in range(tokens):
        for j in range(topk):
            e = int(g["topk_ids"][t, j])
            ref[t] += a2[t, j].float() @ w2_scaled[e].float().t()
    return ref


def _run_stage1(tokens, model_dim, inter_dim, experts, topk, tile_m, tile_n, tile_k, group_size=32):
    """Direct compile_moe_gemm1 call (validates the vendored kernel in isolation)."""
    g = _gen_common(tokens, model_dim, inter_dim, experts, topk, tile_m)
    dev = g["dev"]
    w1_codes, scale_w1_groups, w1_kernel, scale_w1_1d = _gen_w1(
        dev, experts, inter_dim, model_dim, group_size
    )

    out = torch.empty((tokens, topk, inter_dim), device=dev, dtype=torch.bfloat16)
    empty_a_scale = torch.empty(0, device=dev, dtype=torch.float32)

    exe = compile_moe_gemm1(
        model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, doweight_stage1=False,
        in_dtype="fp4_bf16", group_size=group_size, out_dtype="bf16",
        use_cshuffle_epilog=False, scale_is_bf16=False,
    )
    # Pass tensors directly (memref-typed) -- matches the moe_gemm launcher
    # signature: out, a, w, a_scale, w_scale, sorted_ids, sorted_eids,
    # sorted_weights, num_valid_ids, token_num, n_in(=inter_dim), k_in(=model_dim),
    # blocks, stream.
    args = (
        out, g["x"], w1_kernel, empty_a_scale, scale_w1_1d,
        g["sorted_ids"], g["sorted_expert_ids"], g["sorted_weights"].view(-1),
        g["num_valid_ids"], tokens, inter_dim, model_dim, g["blocks"],
        torch.cuda.current_stream(),
    )
    compiled = flyc.compile(exe, *args)
    compiled(*args)
    torch.cuda.synchronize()

    ref = _ref_stage1(g, w1_codes, scale_w1_groups, inter_dim, group_size)
    return g, out, ref, w1_codes, scale_w1_groups


def _run_stage2(g, a2, tokens, model_dim, inter_dim, experts, topk,
                tile_m, tile_n, tile_k, group_size=32):
    dev = g["dev"]
    w2_codes, scale_w2_groups, w2_kernel, scale_w2_1d = _gen_w2(
        dev, experts, inter_dim, model_dim, group_size
    )

    target = torch.zeros((tokens, model_dim), device=dev, dtype=torch.bfloat16)
    empty_a_scale = torch.empty(0, device=dev, dtype=torch.float32)

    exe = compile_moe_gemm2(
        model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, doweight_stage2=True,
        in_dtype="fp4_bf16", group_size=group_size, out_dtype="bf16",
        accumulate=True, scale_is_bf16=False,
    )
    args = (
        target, a2.reshape(tokens * topk, inter_dim), w2_kernel, empty_a_scale, scale_w2_1d,
        g["sorted_ids"], g["sorted_expert_ids"], g["sorted_weights"].view(-1),
        g["num_valid_ids"], tokens, model_dim, inter_dim, g["blocks"],
        torch.cuda.current_stream(),
    )
    # flyc.compile JIT-compiles *and runs* once; with accumulate=True that is one
    # atomic-add pass into the zeroed buffer.  Re-zero before the second (cached)
    # launch so the accumulator holds exactly one pass.
    compiled = flyc.compile(exe, *args)
    target.zero_()
    compiled(*args)
    torch.cuda.synchronize()

    ref = _ref_stage2(g, a2, w2_codes, scale_w2_groups, model_dim, group_size)
    return target, ref


def _run_stage1_bridge(tokens, model_dim, inter_dim, experts, topk,
                       tile_m, tile_n, tile_k, group_size=32):
    """Public flydsl_moe_stage1 API (validates the gfx942 a16w4 arg-fork)."""
    g = _gen_common(tokens, model_dim, inter_dim, experts, topk, tile_m)
    dev = g["dev"]
    w1_codes, scale_w1_groups, w1_kernel, scale_w1_1d = _gen_w1(
        dev, experts, inter_dim, model_dim, group_size
    )
    w1 = w1_kernel.view(experts, 2 * inter_dim, model_dim // 2)

    out = flydsl_moe_stage1(
        a=g["x"], w1=w1,
        sorted_token_ids=g["sorted_ids"], sorted_expert_ids=g["sorted_expert_ids"],
        num_valid_ids=g["num_valid_ids"], topk=topk,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        a_dtype="bf16", b_dtype="fp4", out_dtype="bf16", act="silu",
        w1_scale=scale_w1_1d, a1_scale=None, sorted_weights=None,
    )
    torch.cuda.synchronize()
    ref = _ref_stage1(g, w1_codes, scale_w1_groups, inter_dim, group_size)
    return g, out, ref


def _run_stage2_bridge(g, a2, tokens, model_dim, inter_dim, experts, topk,
                       tile_m, tile_n, tile_k, group_size=32):
    dev = g["dev"]
    w2_codes, scale_w2_groups, w2_kernel, scale_w2_1d = _gen_w2(
        dev, experts, inter_dim, model_dim, group_size
    )
    w2 = w2_kernel.view(experts, model_dim, inter_dim // 2)
    # atomic mode accumulates -> caller must supply a zeroed output buffer.
    out = torch.zeros((tokens, model_dim), device=dev, dtype=torch.bfloat16)

    out = flydsl_moe_stage2(
        inter_states=a2, w2=w2,
        sorted_token_ids=g["sorted_ids"], sorted_expert_ids=g["sorted_expert_ids"],
        num_valid_ids=g["num_valid_ids"], out=out, topk=topk,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        a_dtype="bf16", b_dtype="fp4", out_dtype="bf16", mode="atomic",
        w2_scale=scale_w2_1d, a2_scale=None, sorted_weights=None,
    )
    torch.cuda.synchronize()
    ref = _ref_stage2(g, a2, w2_codes, scale_w2_groups, model_dim, group_size)
    return out, ref


# Default small gfx942-feasible shape (LDS-safe tiles).  topk=1 keeps the
# stage2 expert reduction unambiguous (softmax over k=1 => unit weight), so the
# torch reference needs no slot re-weighting.
_SHAPE = dict(tokens=64, model_dim=256, inter_dim=128, experts=4, topk=1,
              tile_m=16, tile_n=64, tile_k=128)


@pytest.mark.skipif(not _is_gfx942(), reason="gfx942-only decode path")
def test_a16w4_stage1_gfx942():
    g, out, ref, _, _ = _run_stage1(**_SHAPE)
    assert _check(ref, out, "stage1_a16w4")


@pytest.mark.skipif(not _is_gfx942(), reason="gfx942-only decode path")
def test_a16w4_stage2_gfx942():
    g, out1, _, _, _ = _run_stage1(**_SHAPE)
    a2 = out1.to(torch.bfloat16)
    target, ref = _run_stage2(g, a2, **_SHAPE)
    assert _check(ref, target, "stage2_a16w4")


@pytest.mark.skipif(not _is_gfx942(), reason="gfx942-only decode path")
def test_a16w4_stage1_bridge_gfx942():
    g, out, ref = _run_stage1_bridge(**_SHAPE)
    assert _check(ref, out.reshape(ref.shape), "stage1_a16w4_bridge")


@pytest.mark.skipif(not _is_gfx942(), reason="gfx942-only decode path")
def test_a16w4_stage2_bridge_gfx942():
    g, out1, ref1 = _run_stage1_bridge(**_SHAPE)
    a2 = out1.reshape(ref1.shape).to(torch.bfloat16)
    out, ref = _run_stage2_bridge(g, a2, **_SHAPE)
    assert _check(ref, out, "stage2_a16w4_bridge")


def _main():
    if not _is_gfx942():
        print("[SKIP] not gfx942")
        return 0
    g, out, ref, _, _ = _run_stage1(**_SHAPE)
    ok1 = _check(ref, out, "stage1_a16w4")
    a2 = out.to(torch.bfloat16)
    target, ref2 = _run_stage2(g, a2, **_SHAPE)
    ok2 = _check(ref2, target, "stage2_a16w4")
    # Public bridge API (exercises the gfx942 a16w4 arg-fork).
    gb, outb, refb = _run_stage1_bridge(**_SHAPE)
    ok3 = _check(refb, outb.reshape(refb.shape), "stage1_a16w4_bridge")
    a2b = outb.reshape(refb.shape).to(torch.bfloat16)
    out2b, ref2b = _run_stage2_bridge(gb, a2b, **_SHAPE)
    ok4 = _check(ref2b, out2b, "stage2_a16w4_bridge")
    return 0 if (ok1 and ok2 and ok3 and ok4) else 1


if __name__ == "__main__":
    sys.exit(_main())
