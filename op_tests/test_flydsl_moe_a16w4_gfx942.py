# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FlyDSL a16w4 fused-MoE correctness test (Track A).

Supported weight dtypes for ``--bench``:

* ``fp4_bf16`` (default) -- bf16 activation x MX-FP4 (E2M1, per-1x32 E8M0 block
  scale) weight.  On gfx942 there is no scaled MX-FP4 MFMA, so the FlyDSL path
  dequantizes FP4 to bf16 in-kernel and runs the ``in_dtype="fp4_bf16"`` kernels.
* ``int4_bf16`` -- bf16 activation x packed int4 weight with per-32 bf16 groupwise
  scale (``in_dtype="int4_bf16"``, ``scale_is_bf16=True``).

Pytest coverage remains fp4-only.  Bench runs one dtype per invocation (no
side-by-side compare).

Usage:
    python op_tests/test_flydsl_moe_a16w4_gfx942.py
    python op_tests/test_flydsl_moe_a16w4_gfx942.py --bench --dtype fp4_bf16
    python op_tests/test_flydsl_moe_a16w4_gfx942.py --bench --dtype int4_bf16
    python op_tests/test_flydsl_moe_a16w4_gfx942.py --bench --dtype int4_bf16 --no-verify
    pytest op_tests/test_flydsl_moe_a16w4_gfx942.py
"""

import sys

import pytest
import torch

import flydsl.compiler as flyc
from aiter import pertoken_quant
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

_BENCH_DTYPES = ("fp4_bf16", "int4_bf16")


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


def _bridge_dtypes(dtype: str) -> dict:
    """Map bench dtype label to flydsl_moe_stage* a_dtype/b_dtype kwargs."""
    if dtype == "fp4_bf16":
        return dict(a_dtype="bf16", b_dtype="fp4")
    if dtype == "int4_bf16":
        return dict(a_dtype="bf16", b_dtype="int4")
    raise ValueError(f"unsupported bench dtype {dtype!r}; expected one of {_BENCH_DTYPES}")


def _decode_int4_w(w_int: torch.Tensor, scale_groups: torch.Tensor, group_size: int) -> torch.Tensor:
    """Decode packed-int4 int8 codes [E, N, K] with per-group bf16 scale.

    scale_groups: [E, K//group_size, N] (Opt 0 layout, matches shuffle_scale_for_int4).
    Returns decoded weights as f32 [E, N, K].
    """
    w = w_int.to(torch.float32)
    E, N, K = w.shape
    sc = scale_groups.permute(0, 2, 1).reshape(E, N, K // group_size)
    sc = sc.repeat_interleave(group_size, dim=2)
    return w * sc.float()


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
           rel_l2_max: float = 0.06, *, quiet_on_pass: bool = True):
    """Relative-L2 gate -- robust to MX-FP4's wide dynamic range (the elementwise
    error of a single fp4 weight is large, but the aggregate direction is tight).

    When ``quiet_on_pass`` is True (default), success prints only ``[label] PASS``.
    Failures always emit rel_L2, shapes, and sample ref/out values for debugging.
    """
    ref = ref.float()
    out = out.float()
    max_delta = (ref - out).abs().max().item()
    rel_l2 = ((out - ref).norm() / ref.norm().clamp_min(1e-30)).item()
    passed = rel_l2 <= rel_l2_max
    if passed:
        if quiet_on_pass:
            print(f"[{label}] PASS")
        else:
            print(f"[{label}] rel_L2={rel_l2:.4f} (<= {rel_l2_max})  max_delta={max_delta:.3f} -> PASS")
            print(f"    ref : {ref.reshape(-1)[:6].tolist()}")
            print(f"    out : {out.reshape(-1)[:6].tolist()}")
    else:
        print(f"[{label}] FAIL  rel_L2={rel_l2:.4f} (<= {rel_l2_max})  max_delta={max_delta:.3f}")
        print(f"    shapes: ref={tuple(ref.shape)} out={tuple(out.shape)}")
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


def _gen_w1(dev, experts, inter_dim, model_dim, group_size, dtype: str = "fp4_bf16"):
    """Random stage1 weight for fp4_bf16 (MX-FP4) or int4_bf16 (packed int4)."""
    N1 = 2 * inter_dim
    if dtype == "fp4_bf16":
        codes = torch.randint(0, 16, (experts, N1, model_dim), device=dev, dtype=torch.int8)
        exps = torch.randint(-2, 3, (experts, model_dim // group_size, N1), device=dev)
        # bf16 E8M0 scale (lossless: power-of-two fits bf16's 8 exp bits) -- halves
        # weight-scale HBM traffic vs f32, matching the int4 path's scale_is_bf16.
        scale_groups = torch.pow(2.0, exps.to(torch.float32)).to(torch.bfloat16)
    elif dtype == "int4_bf16":
        w_fp32 = torch.randn((experts, N1, model_dim), device=dev, dtype=torch.float32) * 0.05
        codes, _ = pertoken_quant(w_fp32, quant_dtype=torch.int8, dtypeMax=7)
        scale_groups = (
            torch.rand(experts, model_dim // group_size, N1, device=dev, dtype=torch.float32) * 0.05 + 0.005
        ).to(torch.bfloat16)
    else:
        raise ValueError(f"unsupported weight dtype {dtype!r}")
    kernel = _prep_w4(codes)
    scale_1d = shuffle_scale_for_int4(scale_groups, group_size=group_size).view(-1).contiguous()
    return codes, scale_groups, kernel, scale_1d


def _gen_w2(dev, experts, inter_dim, model_dim, group_size, dtype: str = "fp4_bf16"):
    if dtype == "fp4_bf16":
        codes = torch.randint(0, 16, (experts, model_dim, inter_dim), device=dev, dtype=torch.int8)
        exps = torch.randint(-2, 3, (experts, inter_dim // group_size, model_dim), device=dev)
        # bf16 E8M0 scale (lossless: power-of-two fits bf16's 8 exp bits) -- halves
        # weight-scale HBM traffic vs f32, matching the int4 path's scale_is_bf16.
        scale_groups = torch.pow(2.0, exps.to(torch.float32)).to(torch.bfloat16)
    elif dtype == "int4_bf16":
        w_fp32 = torch.randn((experts, model_dim, inter_dim), device=dev, dtype=torch.float32) * 0.05
        codes, _ = pertoken_quant(w_fp32, quant_dtype=torch.int8, dtypeMax=7)
        scale_groups = (
            torch.rand(experts, inter_dim // group_size, model_dim, device=dev, dtype=torch.float32) * 0.05 + 0.005
        ).to(torch.bfloat16)
    else:
        raise ValueError(f"unsupported weight dtype {dtype!r}")
    kernel = _prep_w4(codes)
    scale_1d = shuffle_scale_for_int4(scale_groups, group_size=group_size).view(-1).contiguous()
    return codes, scale_groups, kernel, scale_1d


def _decode_stage_w(w_codes, scale_groups, group_size, dtype: str) -> torch.Tensor:
    if dtype == "fp4_bf16":
        return _decode_w(w_codes, scale_groups, group_size)
    if dtype == "int4_bf16":
        return _decode_int4_w(w_codes, scale_groups, group_size)
    raise ValueError(f"unsupported weight dtype {dtype!r}")


def _ref_stage1(g, w1_codes, scale_w1_groups, inter_dim, group_size, dtype: str = "fp4_bf16"):
    dev = g["dev"]
    w1_scaled = _decode_stage_w(w1_codes, scale_w1_groups, group_size, dtype).to(torch.bfloat16)
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


def _ref_stage2(g, a2, w2_codes, scale_w2_groups, model_dim, group_size, dtype: str = "fp4_bf16"):
    dev = g["dev"]
    w2_scaled = _decode_stage_w(w2_codes, scale_w2_groups, group_size, dtype).to(torch.bfloat16)
    tokens, topk = a2.shape[0], a2.shape[1]
    ref = torch.zeros((tokens, model_dim), device=dev, dtype=torch.float32)
    for t in range(tokens):
        for j in range(topk):
            e = int(g["topk_ids"][t, j])
            w = float(g["topk_weights"][t, j])
            ref[t] += w * (a2[t, j].float() @ w2_scaled[e].float().t())
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
        use_cshuffle_epilog=False, scale_is_bf16=True,
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
        accumulate=True, scale_is_bf16=True,
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


def _bench(shape, dtype: str = "fp4_bf16", num_iters=50, num_warmup=10, verify=True):
    """Time the public-bridge stage1/stage2 kernels and report TFLOPS.

    When ``verify`` is True (default), runs one correctness pass at the bench
    shape via ``_check`` against the torch ref for stage1 and stage2 *before*
    timing (same gate as the pytest tests and the scratch MoE benches). On
    failure, timing is skipped. Pass ``verify=False`` (``--no-verify``) to
    benchmark only.

    ``dtype`` selects the weight path: ``fp4_bf16`` (MX-FP4 a16w4) or
    ``int4_bf16`` (packed int4 + bf16 groupwise scale). One dtype per run.

    Weights/scales are built once so run_perftest times the kernel launch, not
    the input preparation. Stage2 re-zeros its output each call (atomic mode),
    so repeated timing iterations do not over-accumulate.
    """
    from aiter.test_common import run_perftest

    if dtype not in _BENCH_DTYPES:
        raise ValueError(f"unsupported bench dtype {dtype!r}; expected one of {_BENCH_DTYPES}")
    bridge = _bridge_dtypes(dtype)

    tokens, model_dim, inter_dim = shape["tokens"], shape["model_dim"], shape["inter_dim"]
    experts, topk, gsize = shape["experts"], shape["topk"], 32
    tile_m, tile_n, tile_k = shape["tile_m"], shape["tile_n"], shape["tile_k"]

    g = _gen_common(tokens, model_dim, inter_dim, experts, topk, tile_m)
    dev = g["dev"]
    w1_codes, scale_w1_groups, w1_kernel, scale_w1_1d = _gen_w1(
        dev, experts, inter_dim, model_dim, gsize, dtype=dtype
    )
    w2_codes, scale_w2_groups, w2_kernel, scale_w2_1d = _gen_w2(
        dev, experts, inter_dim, model_dim, gsize, dtype=dtype
    )
    w1 = w1_kernel.view(experts, 2 * inter_dim, model_dim // 2)
    w2 = w2_kernel.view(experts, model_dim, inter_dim // 2)
    # Match fused_moe (doweight_stage1=False): stage2 applies sorted top-k weights.
    stage2_sorted_weights = g["sorted_weights"]

    if verify:
        out1 = flydsl_moe_stage1(
            a=g["x"], w1=w1,
            sorted_token_ids=g["sorted_ids"], sorted_expert_ids=g["sorted_expert_ids"],
            num_valid_ids=g["num_valid_ids"], topk=topk,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            out_dtype="bf16", act="silu",
            w1_scale=scale_w1_1d, a1_scale=None, sorted_weights=None,
            **bridge,
        )
        torch.cuda.synchronize()
        ref1 = _ref_stage1(g, w1_codes, scale_w1_groups, inter_dim, gsize, dtype=dtype)
        if not _check(ref1, out1.reshape(ref1.shape), f"bench_stage1_{dtype}_bridge"):
            print("[bench] stage1 CORRECTNESS FAILED -- not timing")
            return 1
        a2 = out1.reshape(tokens, topk, inter_dim).to(torch.bfloat16)
        out2 = torch.zeros((tokens, model_dim), device=dev, dtype=torch.bfloat16)
        out2 = flydsl_moe_stage2(
            inter_states=a2, w2=w2,
            sorted_token_ids=g["sorted_ids"], sorted_expert_ids=g["sorted_expert_ids"],
            num_valid_ids=g["num_valid_ids"], out=out2, topk=topk,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            out_dtype="bf16", mode="atomic",
            w2_scale=scale_w2_1d, a2_scale=None, sorted_weights=stage2_sorted_weights,
            **bridge,
        )
        torch.cuda.synchronize()
        ref2 = _ref_stage2(g, a2, w2_codes, scale_w2_groups, model_dim, gsize, dtype=dtype)
        if not _check(ref2, out2, f"bench_stage2_{dtype}_bridge"):
            print("[bench] stage2 CORRECTNESS FAILED -- not timing")
            return 1

    def _s1():
        return flydsl_moe_stage1(
            a=g["x"], w1=w1,
            sorted_token_ids=g["sorted_ids"], sorted_expert_ids=g["sorted_expert_ids"],
            num_valid_ids=g["num_valid_ids"], topk=topk,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            out_dtype="bf16", act="silu",
            w1_scale=scale_w1_1d, a1_scale=None, sorted_weights=None,
            **bridge,
        )

    out1, us1 = run_perftest(_s1, num_iters=num_iters, num_warmup=num_warmup)
    a2 = out1.reshape(tokens, topk, inter_dim).to(torch.bfloat16)

    def _s2():
        out = torch.zeros((tokens, model_dim), device=dev, dtype=torch.bfloat16)
        return flydsl_moe_stage2(
            inter_states=a2, w2=w2,
            sorted_token_ids=g["sorted_ids"], sorted_expert_ids=g["sorted_expert_ids"],
            num_valid_ids=g["num_valid_ids"], out=out, topk=topk,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            out_dtype="bf16", mode="atomic",
            w2_scale=scale_w2_1d, a2_scale=None, sorted_weights=stage2_sorted_weights,
            **bridge,
        )

    _, us2 = run_perftest(_s2, num_iters=num_iters, num_warmup=num_warmup)

    # 2 FLOP/MAC. stage1 = gate+up (2*inter_dim cols); stage2 = down (model_dim cols).
    f1 = 2 * tokens * topk * (2 * inter_dim) * model_dim
    f2 = 2 * tokens * topk * inter_dim * model_dim
    print(
        f"\na16w4 {dtype} (gfx942)  tokens={tokens} model_dim={model_dim} "
        f"inter_dim={inter_dim} E={experts} topk={topk} "
        f"tile={tile_m}x{tile_n}x{tile_k}"
    )
    print(f"[stage1 gate+up] {us1:9.2f} us   {f1 / us1 / 1e6:8.1f} TFLOPS")
    print(f"[stage2 down   ] {us2:9.2f} us   {f2 / us2 / 1e6:8.1f} TFLOPS")
    print(f"[stage1+2 total] {us1 + us2:9.2f} us  (kernels only; excludes moe_sorting)")
    return 0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", action="store_true",
                   help="report kernel time + TFLOPS instead of correctness")
    p.add_argument("--dtype", choices=_BENCH_DTYPES, default="fp4_bf16",
                   help="weight dtype for --bench: fp4_bf16 (MX-FP4) or int4_bf16 (packed int4)")
    p.add_argument("--no-verify", action="store_true",
                   help="(bench only) skip torch-ref correctness check before timing")
    p.add_argument("-t", "--tokens", type=int, default=4096)
    p.add_argument("--model-dim", type=int, default=4096)
    p.add_argument("--inter-dim", type=int, default=1536)
    p.add_argument("-e", "--experts", type=int, default=32)
    p.add_argument("-k", "--topk", type=int, default=1,
                   help="experts per token (use 8 to match test_moe_2stage -k 8)")
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
            experts=args.experts, topk=args.topk,
            tile_m=args.tile_m, tile_n=args.tile_n, tile_k=args.tile_k,
        ), dtype=args.dtype, verify=not args.no_verify))
    sys.exit(_main())
