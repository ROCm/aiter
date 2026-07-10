# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
gpt-oss-120b prefill attention reproducer (CK ``mha_varlen_fwd``).

Purpose
-------
The gpt-oss-120b CI accuracy failure was traced to the *prefill attention*
output diverging at layer 0 between two CK commits.  From the serving log
(``atom_server.log``) the ONLY CK attention op gpt-oss builds/calls is
``mha_varlen_fwd`` (variants ``..._bf16_nlogits_nbias_{mask,nmask}_nlse_
ndropout_nskip_nqscale``).  This test drives that exact op with the real
gpt-oss-120b attention configuration and validates it against a torch
reference, so it fails on the bad CK commit and passes on the good one.

Faithful gpt-oss-120b attention config (HF ``GptOssConfig`` / openai gpt-oss):
    num_attention_heads = 64
    num_key_value_heads = 8      (GQA ratio 8)
    head_dim            = 64
    sliding_window      = 128     (layer 0 is a *sliding* layer)
    attention sink      = learned per-head scalar (virtual softmax logit)
    dtype               = bf16, no logits soft-cap, no fp8

We call ``aiter.mha_varlen_fwd`` directly (not ``flash_attn_varlen_func``)
so the CK path is exercised unconditionally -- the public router would try
FlyDSL first, which only *declines* for this config at runtime.

Run standalone on each CK commit:
    python3 op_tests/test_mha_varlen_gptoss.py
or via pytest:
    python3 -m pytest op_tests/test_mha_varlen_gptoss.py -v
"""

import math

import pytest
import torch

import aiter
from aiter import dtypes

# ---------------------------------------------------------------------------
# gpt-oss-120b attention hyper-parameters
# ---------------------------------------------------------------------------
NUM_Q_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 64
SLIDING_WINDOW = 128
DTYPE = torch.bfloat16


def _ref_attention_with_sink(q, k, v, window_left, sink_ptr):
    """Torch ground-truth for one sequence.

    q: [sq, Hq, D]   k/v: [sk, Hkv, D] (GQA)   sink_ptr: [Hq] or None.

    Semantics (match CK ``mha_varlen_fwd`` sink):
      * causal + optional sliding window (``window_left`` >= 0)
      * ``sink_ptr[h]`` is a virtual key column with a fixed logit that
        steals softmax mass but contributes no value.
    """
    sq, hq, d = q.shape
    sk, hkv, _ = k.shape
    scale = 1.0 / math.sqrt(d)

    # expand GQA kv -> q heads
    rep = hq // hkv
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)

    attn = scale * torch.einsum("qhd,khd->hqk", q.float(), k.float())

    i_q = torch.arange(sq, device=q.device).unsqueeze(1)
    i_k = torch.arange(sk, device=q.device).unsqueeze(0)
    abs_q = sk - sq + i_q  # bottom-right aligned causal
    k_end = abs_q
    if window_left < 0:
        k_start = torch.zeros_like(abs_q)
    else:
        k_start = torch.clamp(abs_q - window_left, min=0)
    valid = (i_k >= k_start) & (i_k <= k_end)
    attn.masked_fill_(~valid.unsqueeze(0), float("-inf"))

    if sink_ptr is not None:
        virt = sink_ptr.float().view(hq, 1, 1).expand(-1, sq, 1)
        attn_ext = torch.cat([attn, virt], dim=-1)
        p = torch.softmax(attn_ext, dim=-1)[:, :, :sk]
    else:
        p = torch.softmax(attn, dim=-1)

    out = torch.einsum("hqk,khd->qhd", p, v.float())
    return out.to(q.dtype)


def _run(layer_type, seqlens=(512, 640, 384, 500), seed=0):
    """Run CK mha_varlen_fwd + torch ref for a given layer type.

    layer_type: "sliding" (window=128) or "full" (global causal).
    Returns dict of error metrics.
    """
    torch.manual_seed(seed)
    device = "cuda"
    window_left = SLIDING_WINDOW if layer_type == "sliding" else -1

    seqlens = list(seqlens)
    total = sum(seqlens)
    cu = torch.tensor(
        [0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=dtypes.i32, device=device
    )
    max_s = max(seqlens)

    q = torch.randn(total, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    k = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    v = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    # learned per-head attention sink (real-valued logit)
    sink_ptr = torch.randn(NUM_Q_HEADS, device=device, dtype=torch.float32)

    out = aiter.mha_varlen_fwd(
        q,
        k,
        v,
        cu,
        cu,
        max_s,
        max_s,
        0,  # min_seqlen_q
        0.0,  # dropout_p
        1.0 / math.sqrt(HEAD_DIM),
        0.0,  # logits_soft_cap
        False,  # zero_tensors
        True,  # is_causal
        window_left,  # window_size_left
        0,  # window_size_right
        0,  # sink_size (streaming sink region; gpt-oss uses sink_ptr)
        False,  # return_softmax_lse
        False,  # return_dropout_randval
        sink_ptr=sink_ptr,
    )
    out = out[0] if isinstance(out, (tuple, list)) else out

    # torch reference, per sequence
    ref = torch.empty_like(out)
    off = 0
    for s in seqlens:
        ref[off : off + s] = _ref_attention_with_sink(
            q[off : off + s],
            k[off : off + s],
            v[off : off + s],
            window_left,
            sink_ptr,
        )
        off += s

    of, rf = out.float(), ref.float()
    abs_err = (of - rf).abs()
    l2_rel = (of - rf).norm().item() / max(rf.norm().item(), 1e-9)
    return {
        "layer_type": layer_type,
        "nan": int(torch.isnan(of).sum().item()),
        "inf": int(torch.isinf(of).sum().item()),
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "l2_rel": l2_rel,
    }


def _assert_ok(m):
    assert m["nan"] == 0, f"{m['layer_type']}: {m['nan']} NaNs in CK output"
    assert m["inf"] == 0, f"{m['layer_type']}: {m['inf']} Infs in CK output"
    # bf16 attention tolerance
    assert m["l2_rel"] < 0.02, (
        f"{m['layer_type']}: l2_rel={m['l2_rel']:.4f} exceeds 2% "
        f"(max_abs_err={m['max_abs_err']:.4f}) -- CK mha_varlen_fwd diverges "
        f"from torch reference"
    )


@pytest.mark.parametrize("layer_type", ["sliding", "full"])
def test_mha_varlen_fwd_gptoss(layer_type):
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")
    _assert_ok(_run(layer_type))


if __name__ == "__main__":
    print("gpt-oss-120b prefill attention (CK mha_varlen_fwd) vs torch ref")
    print(
        f"  cfg: Hq={NUM_Q_HEADS} Hkv={NUM_KV_HEADS} d={HEAD_DIM} "
        f"win={SLIDING_WINDOW} dtype=bf16 sink=per-head\n"
    )
    rows = []
    failed = False
    for lt in ("sliding", "full"):
        m = _run(lt)
        rows.append(m)
        try:
            _assert_ok(m)
            status = "PASS"
        except AssertionError as e:
            status = "FAIL"
            failed = True
            print(f"  [{status}] {e}")
        print(
            f"  {lt:8s} | nan={m['nan']:>3d} inf={m['inf']:>3d} "
            f"max_abs={m['max_abs_err']:.4f} l2_rel={m['l2_rel']:.4f} "
            f"-> {status}"
        )
    print(
        "\n"
        + (
            "RESULT: FAIL (attention reproduces gpt-oss divergence)"
            if failed
            else "RESULT: PASS (attention matches reference)"
        )
    )
    raise SystemExit(1 if failed else 0)
