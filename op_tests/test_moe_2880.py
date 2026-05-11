#!/usr/bin/env python3
"""Test flatmm MOE kernel with intermediate_size_per_partition=2880.

Exercises the cktile flatmm path via a16w4 (per_1x32 blockscale, Swiglu).
Weights are padded to the next multiple of 256 before quantization/preshuffle,
matching how production weight loaders handle non-aligned intermediate sizes.
"""
import sys
import torch
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose, run_perftest
from aiter.fused_moe import (
    fused_topk, fused_moe,
    torch_moe_stage1, torch_moe_stage2,
)
from aiter.ops.shuffle import shuffle_weight_a16w4, shuffle_scale_a16w4

torch.int4 = getattr(torch, "int4", torch.uint32)
torch.set_default_device("cuda")

# preshuffle + scale shuffle require inter_dim to be a multiple of 256
# (weight: K_packed % 64 == 0 -> inter_dim%128; scale: (K/32)%8==0 -> inter_dim%256)
_INTER_ALIGN = 256


def test_2880(token, model_dim=4096, real_inter_dim=2880, E=8, topk=2):
    dtype = dtypes.bf16
    qType = aiter.QuantType.per_1x32
    WQDType = dtypes.fp4x2
    actType = aiter.ActivationType.Swiglu

    # pad inter_dim so preshuffle and kernel are happy
    inter_pad = (-real_inter_dim) % _INTER_ALIGN      # 192 for 2880
    inter_dim = real_inter_dim + inter_pad             # 3072

    print(f"\n{'='*70}")
    print(f"  token={token}  model_dim={model_dim}")
    print(f"  real_inter_dim={real_inter_dim}  padded_inter_dim={inter_dim}  "
          f"intermediate_pad={inter_pad}")
    print(f"  E={E}  topk={topk}  a16w4 per_1x32 Swiglu gate+up")
    print(f"{'='*70}")

    torch_quant = aiter.get_torch_quant(qType)
    inp = torch.randn((token, model_dim), dtype=dtype)

    # build weights at the PADDED size, zero the padding
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype)
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype)
    if inter_pad:
        w1[:, inter_dim - inter_pad : inter_dim, :] = 0   # gate pad
        w1[:, -inter_pad:, :] = 0                          # up pad
        w2[:, :, -inter_pad:] = 0                          # w2 K pad

    exp_bias1 = torch.clamp(torch.randn((E, inter_dim * 2), dtype=dtype), -1.0, 1.0)
    exp_bias2 = torch.clamp(torch.randn((E, model_dim), dtype=dtype), -1.0, 1.0)
    score = torch.randn((token, E), dtype=dtype)

    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    # quantize
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=WQDType)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=WQDType)
    w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
    w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

    a1_qt = inp.to(dtypes.bf16)
    a1_scale = None
    exp_bias1_aiter = exp_bias1.to(dtypes.fp32)
    exp_bias2_aiter = exp_bias2.to(dtypes.fp32)

    # preshuffle
    w1_qt_aiter = shuffle_weight_a16w4(w1_qt.clone(), 16, True)
    w1_scale_aiter = shuffle_scale_a16w4(w1_scale, E, True)
    w2_qt_aiter = shuffle_weight_a16w4(w2_qt.clone(), 16, False)
    w2_scale_aiter = shuffle_scale_a16w4(w2_scale, E, False)

    # torch reference
    out1_ref = torch_moe_stage1(
        a1_qt, w1_qt, w2_qt, topk_weights, topk_ids,
        dtype=dtype, activation=actType, quant_type=qType,
        a1_scale=a1_scale, w1_scale=w1_scale,
        w1_bias=exp_bias1, doweight=False,
    )
    a2_qt = out1_ref.view(token, topk, -1)
    out2_ref = torch_moe_stage2(
        a2_qt, w1_qt, w2_qt, topk_weights, topk_ids,
        dtype=dtype, quant_type=qType,
        w2_scale=w2_scale, a2_scale=None,
        w2_bias=exp_bias2, doweight=True,
    )

    # CK-tile kernel
    try:
        out2_ck, us = run_perftest(
            fused_moe, inp,
            w1_qt_aiter, w2_qt_aiter, topk_weights, topk_ids,
            w1_scale=w1_scale_aiter, w2_scale=w2_scale_aiter,
            quant_type=qType, activation=actType,
            doweight_stage1=False,
            intermediate_pad=inter_pad, hidden_pad=0,
            bias1=exp_bias1_aiter, bias2=exp_bias2_aiter,
            num_iters=3, num_warmup=1,
        )
    except Exception as e:
        print(f"  FAIL (exception): {e}")
        import traceback; traceback.print_exc()
        return False

    def cosine_sim(a, b):
        a, b = a.double().flatten(), b.double().flatten()
        return (a @ b) / (a.norm() * b.norm() + 1e-12)

    sim = cosine_sim(out2_ref, out2_ck).item()
    checkAllclose(
        out2_ref, out2_ck,
        msg=f"  inter_dim={real_inter_dim} padded={inter_dim} ({us:.1f} us)",
    )
    print(f"  cosine similarity: {sim:.6f}")
    ok = sim > 0.99
    print(f"  {'PASS' if ok else 'FAIL'} (cosine > 0.99 ? {ok})")
    return ok


if __name__ == "__main__":
    passed = failed = 0
    for t in [1, 32]:
        if test_2880(t):
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*70}")
    print(f"  SUMMARY: {passed} passed, {failed} failed")
    print(f"{'='*70}")
    sys.exit(0 if failed == 0 else 1)
