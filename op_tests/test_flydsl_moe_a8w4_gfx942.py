# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FlyDSL MX-FP4 a8w4 fused-MoE correctness test (Track B / B1).

a8w4 = MX-FP8 activation (E4M3FNUZ, per-1x32 E8M0 scale) x MX-FP4 (E2M1,
per-1x32 E8M0 scale) weight.

On gfx942 (CDNA3) there is no scaled MX-FP4 MFMA, so the FlyDSL path dequantizes
BOTH operands to bf16 in-kernel -- fp8 activation via ``dequant_fp8_to_bf16``
(folding the per-32 E8M0 A-scale) and FP4 weight via ``dequant_fp4_to_bf16``
(folding the per-32 E8M0 weight scale) -- then runs the legacy bf16 MFMA hot
loop (the ``moe_gemm_2stage`` ``in_dtype="a8w4_bf16"`` kernels, B1 of the
ticket). This test exercises those (vendored) kernels directly and through the
public bridge, validating both stages against a torch reference that decodes the
fp8 codes and the E2M1 codes and applies the per-32 power-of-two block scales.

Usage:
    python op_tests/test_flydsl_moe_a8w4_gfx942.py
    python op_tests/test_flydsl_moe_a8w4_gfx942.py --bench
    pytest op_tests/test_flydsl_moe_a8w4_gfx942.py
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

# gfx942 fp8 activation format is E4M3FNUZ.
_FP8 = torch.float8_e4m3fnuz

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
    """Decode E2M1 weight codes [E, N, K] and apply per-32 block scale -> f32 [E, N, K]."""
    lut = torch.tensor(_E2M1_LUT, device=codes_i8.device, dtype=torch.float32)
    w_dec = lut[codes_i8.long()]
    E, N, K = w_dec.shape
    sc = scale_groups.permute(0, 2, 1).reshape(E, N, K // group_size)
    sc = sc.repeat_interleave(group_size, dim=2)
    return w_dec * sc


def _decode_a(a_fp8: torch.Tensor, a_scale_groups: torch.Tensor, group_size: int) -> torch.Tensor:
    """Decode fp8 activation [M, K] + per-32 pow2 scale [M, K//gs] -> f32 [M, K]."""
    a = a_fp8.to(torch.float32)
    sc = a_scale_groups.repeat_interleave(group_size, dim=1)
    return a * sc


def _check(ref: torch.Tensor, out: torch.Tensor, label: str, rel_l2_max: float = 0.08):
    """Relative-L2 gate -- robust to MX dynamic range (single-element fp4/fp8 error
    is large, but the aggregate direction is tight)."""
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


def _gen_common(tokens, model_dim, inter_dim, experts, topk, tile_m, group_size, seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dev = torch.device("cuda")
    # fp8 activation + per-1x32 E8M0 (pow2) A-scale, row-major [tokens, K//gs].
    a_f32 = (torch.randn((tokens, model_dim), device=dev, dtype=torch.float32) * 0.5)
    a_fp8 = a_f32.to(_FP8)
    a_exps = torch.randint(-2, 3, (tokens, model_dim // group_size), device=dev)
    a_scale_groups = torch.pow(2.0, a_exps.to(torch.float32))
    a_scale_1d = a_scale_groups.contiguous().view(-1)

    score = torch.rand((tokens, experts), device=dev, dtype=torch.float32)
    topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, experts, model_dim, torch.bfloat16, tile_m
    )
    return dict(
        dev=dev, a_fp8=a_fp8, a_scale_groups=a_scale_groups, a_scale_1d=a_scale_1d,
        topk_ids=topk_ids, topk_weights=topk_weights,
        sorted_ids=sorted_ids, sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids, num_valid_ids=num_valid_ids,
        blocks=int(sorted_expert_ids.shape[0]),
    )


def _prep_w4(codes_i8: torch.Tensor):
    """Preshuffle + pack E2M1 codes [E, N, K] for the kernel (int4-style layout)."""
    shuf = shuffle_weight(codes_i8.view(torch.int8), layout=(16, 16))
    return _pack_shuffled_int8_to_packed_int4_no_perm(shuf).contiguous()


def _gen_w1(dev, experts, inter_dim, model_dim, group_size):
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
    a_dec = _decode_a(g["a_fp8"], g["a_scale_groups"], group_size)
    w1_scaled = _decode_w(w1_codes, scale_w1_groups, group_size)
    tokens, topk = a_dec.shape[0], g["topk_ids"].shape[1]
    ref = torch.empty((tokens, topk, inter_dim), device=dev, dtype=torch.float32)
    for t in range(tokens):
        for j in range(topk):
            e = int(g["topk_ids"][t, j])
            acc = a_dec[t] @ w1_scaled[e].t()
            gate, up = acc[:inter_dim], acc[inter_dim:]
            ref[t, j] = torch.nn.functional.silu(gate) * up
    return ref


def _ref_stage2(g, a2_fp8, a2_scale_groups, w2_codes, scale_w2_groups, model_dim, group_size):
    dev = g["dev"]
    w2_scaled = _decode_w(w2_codes, scale_w2_groups, group_size)
    tokens, topk = a2_fp8.shape[0], a2_fp8.shape[1]
    ref = torch.zeros((tokens, model_dim), device=dev, dtype=torch.float32)
    for t in range(tokens):
        for j in range(topk):
            e = int(g["topk_ids"][t, j])
            a_dec = _decode_a(a2_fp8[t, j].unsqueeze(0),
                              a2_scale_groups[t * topk + j].unsqueeze(0), group_size)[0]
            ref[t] += a_dec @ w2_scaled[e].t()
    return ref


def _quant_a2(out1_f32, group_size):
    """Quantize stage1 output [tokens, topk, inter_dim] f32 -> fp8 codes + per-32
    pow2 (E8M0) scale [tokens*topk, inter_dim//gs] (the a8w4 stage2 layout)."""
    tokens, topk, inter_dim = out1_f32.shape
    flat = out1_f32.reshape(tokens * topk, inter_dim)
    blocks = flat.reshape(tokens * topk, inter_dim // group_size, group_size)
    amax = blocks.abs().amax(dim=-1).clamp_min(1e-12)
    # E8M0 pow2 scale so the block max maps below the fp8 E4M3 max (~240).
    # ceil(log2(amax/224)) guarantees amax/scale <= 224 < 240 (no overflow->inf).
    exp = torch.ceil(torch.log2(amax / 224.0))
    scale = torch.pow(2.0, exp)  # [tokens*topk, inter_dim//gs]
    q = (blocks / scale.unsqueeze(-1)).reshape(tokens * topk, inter_dim).to(_FP8)
    return q.reshape(tokens, topk, inter_dim), scale


def _run_stage1(tokens, model_dim, inter_dim, experts, topk, tile_m, tile_n, tile_k, group_size=32):
    g = _gen_common(tokens, model_dim, inter_dim, experts, topk, tile_m, group_size)
    dev = g["dev"]
    w1_codes, scale_w1_groups, w1_kernel, scale_w1_1d = _gen_w1(
        dev, experts, inter_dim, model_dim, group_size
    )
    out = torch.empty((tokens, topk, inter_dim), device=dev, dtype=torch.bfloat16)

    exe = compile_moe_gemm1(
        model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, doweight_stage1=False,
        in_dtype="a8w4_bf16", group_size=group_size, out_dtype="bf16",
        use_cshuffle_epilog=False, scale_is_bf16=False,
    )
    args = (
        out, g["a_fp8"], w1_kernel, g["a_scale_1d"], scale_w1_1d,
        g["sorted_ids"], g["sorted_expert_ids"], g["sorted_weights"].view(-1),
        g["num_valid_ids"], tokens, inter_dim, model_dim, g["blocks"],
        torch.cuda.current_stream(),
    )
    compiled = flyc.compile(exe, *args)
    compiled(*args)
    torch.cuda.synchronize()

    ref = _ref_stage1(g, w1_codes, scale_w1_groups, inter_dim, group_size)
    return g, out, ref


def _run_stage2(g, a2_fp8, a2_scale_1d, tokens, model_dim, inter_dim, experts, topk,
                tile_m, tile_n, tile_k, group_size=32):
    dev = g["dev"]
    w2_codes, scale_w2_groups, w2_kernel, scale_w2_1d = _gen_w2(
        dev, experts, inter_dim, model_dim, group_size
    )
    target = torch.zeros((tokens, model_dim), device=dev, dtype=torch.bfloat16)

    exe = compile_moe_gemm2(
        model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, doweight_stage2=True,
        in_dtype="a8w4_bf16", group_size=group_size, out_dtype="bf16",
        accumulate=True, scale_is_bf16=False,
    )
    args = (
        target, a2_fp8.reshape(tokens * topk, inter_dim), w2_kernel, a2_scale_1d, scale_w2_1d,
        g["sorted_ids"], g["sorted_expert_ids"], g["sorted_weights"].view(-1),
        g["num_valid_ids"], tokens, model_dim, inter_dim, g["blocks"],
        torch.cuda.current_stream(),
    )
    compiled = flyc.compile(exe, *args)
    target.zero_()
    compiled(*args)
    torch.cuda.synchronize()

    ref = _ref_stage2(g, a2_fp8, a2_scale_groups_from_1d(a2_scale_1d, tokens * topk, inter_dim, group_size),
                      w2_codes, scale_w2_groups, model_dim, group_size)
    return target, ref


def a2_scale_groups_from_1d(a2_scale_1d, rows, inter_dim, group_size):
    return a2_scale_1d.reshape(rows, inter_dim // group_size)


def _run_stage1_bridge(tokens, model_dim, inter_dim, experts, topk,
                       tile_m, tile_n, tile_k, group_size=32):
    g = _gen_common(tokens, model_dim, inter_dim, experts, topk, tile_m, group_size)
    dev = g["dev"]
    w1_codes, scale_w1_groups, w1_kernel, scale_w1_1d = _gen_w1(
        dev, experts, inter_dim, model_dim, group_size
    )
    w1 = w1_kernel.view(experts, 2 * inter_dim, model_dim // 2)

    out = flydsl_moe_stage1(
        a=g["a_fp8"], w1=w1,
        sorted_token_ids=g["sorted_ids"], sorted_expert_ids=g["sorted_expert_ids"],
        num_valid_ids=g["num_valid_ids"], topk=topk,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        a_dtype="fp8", b_dtype="fp4", out_dtype="bf16", act="silu",
        w1_scale=scale_w1_1d, a1_scale=g["a_scale_1d"], sorted_weights=None,
    )
    torch.cuda.synchronize()
    ref = _ref_stage1(g, w1_codes, scale_w1_groups, inter_dim, group_size)
    return g, out, ref


def _run_stage2_bridge(g, a2_fp8, a2_scale_1d, tokens, model_dim, inter_dim, experts, topk,
                       tile_m, tile_n, tile_k, group_size=32):
    dev = g["dev"]
    w2_codes, scale_w2_groups, w2_kernel, scale_w2_1d = _gen_w2(
        dev, experts, inter_dim, model_dim, group_size
    )
    w2 = w2_kernel.view(experts, model_dim, inter_dim // 2)
    out = torch.zeros((tokens, model_dim), device=dev, dtype=torch.bfloat16)

    out = flydsl_moe_stage2(
        inter_states=a2_fp8, w2=w2,
        sorted_token_ids=g["sorted_ids"], sorted_expert_ids=g["sorted_expert_ids"],
        num_valid_ids=g["num_valid_ids"], out=out, topk=topk,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        a_dtype="fp8", b_dtype="fp4", out_dtype="bf16", mode="atomic",
        w2_scale=scale_w2_1d, a2_scale=a2_scale_1d, sorted_weights=None,
    )
    torch.cuda.synchronize()
    ref = _ref_stage2(g, a2_fp8, a2_scale_groups_from_1d(a2_scale_1d, tokens * topk, inter_dim, group_size),
                      w2_codes, scale_w2_groups, model_dim, group_size)
    return out, ref


# Default small gfx942-feasible shape (LDS-safe tiles). topk=1 keeps the stage2
# expert reduction unambiguous (softmax over k=1 => unit weight).
_SHAPE = dict(tokens=64, model_dim=256, inter_dim=128, experts=4, topk=1,
              tile_m=16, tile_n=64, tile_k=128)


@pytest.mark.skipif(not _is_gfx942(), reason="gfx942-only decode path")
def test_a8w4_stage1_gfx942():
    g, out, ref = _run_stage1(**_SHAPE)
    assert _check(ref, out, "stage1_a8w4")


@pytest.mark.skipif(not _is_gfx942(), reason="gfx942-only decode path")
def test_a8w4_stage2_gfx942():
    g, out1, ref1 = _run_stage1(**_SHAPE)
    a2_fp8, a2_scale = _quant_a2(out1.float(), 32)
    target, ref = _run_stage2(g, a2_fp8, a2_scale.view(-1).contiguous(), **_SHAPE)
    assert _check(ref, target, "stage2_a8w4")


@pytest.mark.skipif(not _is_gfx942(), reason="gfx942-only decode path")
def test_a8w4_stage1_bridge_gfx942():
    g, out, ref = _run_stage1_bridge(**_SHAPE)
    assert _check(ref, out.reshape(ref.shape), "stage1_a8w4_bridge")


@pytest.mark.skipif(not _is_gfx942(), reason="gfx942-only decode path")
def test_a8w4_stage2_bridge_gfx942():
    g, out1, ref1 = _run_stage1_bridge(**_SHAPE)
    a2_fp8, a2_scale = _quant_a2(out1.reshape(ref1.shape).float(), 32)
    out, ref = _run_stage2_bridge(g, a2_fp8, a2_scale.view(-1).contiguous(), **_SHAPE)
    assert _check(ref, out, "stage2_a8w4_bridge")


def _main():
    if not _is_gfx942():
        print("[SKIP] not gfx942")
        return 0
    g, out, ref = _run_stage1(**_SHAPE)
    ok1 = _check(ref, out, "stage1_a8w4")
    a2_fp8, a2_scale = _quant_a2(out.float(), 32)
    target, ref2 = _run_stage2(g, a2_fp8, a2_scale.view(-1).contiguous(), **_SHAPE)
    ok2 = _check(ref2, target, "stage2_a8w4")
    gb, outb, refb = _run_stage1_bridge(**_SHAPE)
    ok3 = _check(refb, outb.reshape(refb.shape), "stage1_a8w4_bridge")
    a2b_fp8, a2b_scale = _quant_a2(outb.reshape(refb.shape).float(), 32)
    out2b, ref2b = _run_stage2_bridge(gb, a2b_fp8, a2b_scale.view(-1).contiguous(), **_SHAPE)
    ok4 = _check(ref2b, out2b, "stage2_a8w4_bridge")
    return 0 if (ok1 and ok2 and ok3 and ok4) else 1


def _bench(shape, num_iters=50, num_warmup=10):
    """Time the public-bridge a8w4 stage1/stage2 decode kernels and report TFLOPS."""
    from aiter.test_common import run_perftest

    tokens, model_dim, inter_dim = shape["tokens"], shape["model_dim"], shape["inter_dim"]
    experts, topk, gsize = shape["experts"], shape["topk"], 32
    tile_m, tile_n, tile_k = shape["tile_m"], shape["tile_n"], shape["tile_k"]

    g = _gen_common(tokens, model_dim, inter_dim, experts, topk, tile_m, gsize)
    dev = g["dev"]
    _, _, w1_kernel, scale_w1_1d = _gen_w1(dev, experts, inter_dim, model_dim, gsize)
    _, _, w2_kernel, scale_w2_1d = _gen_w2(dev, experts, inter_dim, model_dim, gsize)
    w1 = w1_kernel.view(experts, 2 * inter_dim, model_dim // 2)
    w2 = w2_kernel.view(experts, model_dim, inter_dim // 2)

    def _s1():
        return flydsl_moe_stage1(
            a=g["a_fp8"], w1=w1,
            sorted_token_ids=g["sorted_ids"], sorted_expert_ids=g["sorted_expert_ids"],
            num_valid_ids=g["num_valid_ids"], topk=topk,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            a_dtype="fp8", b_dtype="fp4", out_dtype="bf16", act="silu",
            w1_scale=scale_w1_1d, a1_scale=g["a_scale_1d"], sorted_weights=None,
        )

    out1, us1 = run_perftest(_s1, num_iters=num_iters, num_warmup=num_warmup)
    a2_fp8, a2_scale = _quant_a2(out1.reshape(tokens, topk, inter_dim).float(), gsize)
    a2_scale_1d = a2_scale.view(-1).contiguous()

    def _s2():
        out = torch.zeros((tokens, model_dim), device=dev, dtype=torch.bfloat16)
        return flydsl_moe_stage2(
            inter_states=a2_fp8, w2=w2,
            sorted_token_ids=g["sorted_ids"], sorted_expert_ids=g["sorted_expert_ids"],
            num_valid_ids=g["num_valid_ids"], out=out, topk=topk,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            a_dtype="fp8", b_dtype="fp4", out_dtype="bf16", mode="atomic",
            w2_scale=scale_w2_1d, a2_scale=a2_scale_1d, sorted_weights=None,
        )

    _, us2 = run_perftest(_s2, num_iters=num_iters, num_warmup=num_warmup)

    f1 = 2 * tokens * topk * (2 * inter_dim) * model_dim
    f2 = 2 * tokens * topk * inter_dim * model_dim
    print(
        f"\na8w4 decode (gfx942)  tokens={tokens} model_dim={model_dim} "
        f"inter_dim={inter_dim} E={experts} topk={topk} "
        f"tile={tile_m}x{tile_n}x{tile_k}"
    )
    print(f"[stage1 gate+up] {us1:9.2f} us   {f1 / us1 / 1e6:8.1f} TFLOPS")
    print(f"[stage2 down   ] {us2:9.2f} us   {f2 / us2 / 1e6:8.1f} TFLOPS")
    return 0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", action="store_true",
                   help="report kernel time + TFLOPS instead of correctness")
    p.add_argument("-t", "--tokens", type=int, default=4096)
    p.add_argument("--model-dim", type=int, default=4096)
    p.add_argument("--inter-dim", type=int, default=1536)
    p.add_argument("-e", "--experts", type=int, default=32)
    p.add_argument("--tile-m", type=int, default=64)
    p.add_argument("--tile-n", type=int, default=128)
    p.add_argument("--tile-k", type=int, default=128)
    args = p.parse_args()

    if args.bench:
        if not _is_gfx942():
            print("[SKIP] not gfx942")
            sys.exit(0)
        sys.exit(_bench(dict(
            tokens=args.tokens, model_dim=args.model_dim, inter_dim=args.inter_dim,
            experts=args.experts, topk=1,
            tile_m=args.tile_m, tile_n=args.tile_n, tile_k=args.tile_k,
        )))
    sys.exit(_main())
