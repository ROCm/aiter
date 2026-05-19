#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Independent correctness check for the contiguous (THD) UA-CK kernel.

Compares the CK kernel run in `kv_contiguous=True` mode against
`aiter.test_mha_common.attention_ref` — a hand-rolled fp32 attention
reference that the rest of aiter's attention test suite already trusts.
Small / moderate shapes only; reference cost is O(sq * sk * hq * d)
memory and time.
"""

from __future__ import annotations

import math
import sys

import torch

from aiter.test_mha_common import attention_ref


def make_paged_view(k: torch.Tensor, v: torch.Tensor, page_size: int):
    """Reshape contiguous K/V (sk, hk, d) into the paged 4-D signature
    the CK kernel binding expects. The bytes are unchanged — this is a
    pure metadata reinterpretation. The kernel sees the same flat
    [sk, hk, d] memory whether kv_contiguous is True or False; only the
    address-computation path inside refresh_*_offsets differs."""
    sk, hk, d = k.shape
    assert sk % page_size == 0, f"sk={sk} not divisible by page_size={page_size}"
    n_pages = sk // page_size
    k4 = k.view(n_pages, page_size, hk, d).contiguous()
    v4 = v.view(n_pages, page_size, hk, d).contiguous()
    # `block_tables` is ignored when kv_contiguous=True. We still pass a
    # 1-D int32 tensor so the binding's `.data_ptr<int32_t>()` is happy.
    block_tables = torch.zeros((1, n_pages), dtype=torch.int32, device=k.device)
    return k4, v4, block_tables


def run_contig(out, q, k4, v4, block_tables, seq_lens, cu_q,
               scale, mask_type):
    from aiter.ops.unified_attention import unified_attention_fwd
    unified_attention_fwd(
        out, q, k4, v4,
        block_tables, seq_lens, cu_q,
        mask_type=mask_type, scale_s=scale,
        scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
        allow_splitkv=False, kv_contiguous=True,
    )


def stage(name, *, sq, sk, hq, hk, d, dtype, causal, page_size=32, seed=42,
          tol=None):
    # bf16 ULP at scale 1 is 1/128 = 7.8e-3, fp16 ULP is 9.8e-4, fp32 is
    # essentially zero. GQA broadcasts an N-element reduction across the
    # group axis, so per-row error scales by ~sqrt(group_size). Default
    # tolerance follows that.
    if tol is None:
        group_size = max(1, hq // hk)
        # Per-row error scales by group_size (reduction depth) and dtype ULP.
        # bf16 ULP at output magnitude ~1 is 1/128 = 7.8e-3; allow ~2 ULPs
        # for GQA-8 reductions.
        if dtype == torch.bfloat16:
            tol = 1e-2 * max(1, group_size // 4)
        elif dtype == torch.float16:
            tol = 2e-3 * max(1, group_size // 4)
        else:
            tol = 5e-2
    print(f"\n========== {name} ==========")
    print(f"sq={sq} sk={sk} hq={hq} hk={hk} d={d} dtype={dtype} causal={causal}")
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    q = torch.randn(sq, hq, d, dtype=dtype, device=device)
    k = torch.randn(sk, hk, d, dtype=dtype, device=device)
    v = torch.randn(sk, hk, d, dtype=dtype, device=device)
    scale = 1.0 / math.sqrt(d)
    mask_type = 2 if causal else 0

    k4, v4, block_tables = make_paged_view(k, v, page_size)
    seq_lens = torch.tensor([sk], dtype=torch.int32, device=device)
    cu_q     = torch.tensor([0, sq], dtype=torch.int32, device=device)
    out = torch.zeros(sq, hq, d, dtype=dtype, device=device)
    run_contig(out, q, k4, v4, block_tables, seq_lens, cu_q,
               scale, mask_type)

    # attention_ref wants (batch, seq, heads, d) and broadcasts GQA internally.
    ref_out, _, _ = attention_ref(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        dropout_p=0.0, dropout_mask=None, causal=causal, upcast=True,
    )
    ref_out = ref_out.squeeze(0)
    torch.cuda.synchronize()

    diff = (out.float() - ref_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    n_bad = (diff > tol).sum().item()
    n_total = diff.numel()

    print(f"  contig vs attention_ref: max={max_diff:.3e}  mean={mean_diff:.3e}")
    print(f"  elements > {tol:g}: {n_bad}/{n_total} ({100*n_bad/n_total:.4f}%)")
    ok = max_diff < tol
    print(f"  {'PASS' if ok else 'FAIL'}  (tol={tol})")
    return ok


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(
        "Only prefill-routed shapes are tested. Decode-tier shapes "
        "(MHA sq<=128, GQA-8 sq<=16) don't have a contiguous-mode "
        "instance compiled; the dispatcher falls back to the paged "
        "kernel and reads `block_tables`, so kv_contiguous=True is "
        "meaningless on those shapes today.\n"
        "For no-mask we additionally require sq to be a multiple of "
        "kBlockM (=256 for prefill_d128, =256 for prefill_d64) — the "
        "kernel's *paged* no-mask path also fails the reference at "
        "partial-Q-block sq's, so that is not contig-specific."
    )
    results = []
    bf16 = torch.bfloat16
    fp16 = torch.float16

    # --- MHA d=128, prefill_d128 (kBlockM=256) ----------------------------
    # no-mask: sq must be a multiple of 256
    for sq in (256, 512, 1024, 2048):
        results.append(stage(
            f"MHA  sq=sk={sq:<4d} d=128 bf16 no-mask",
            sq=sq, sk=sq, hq=5, hk=5, d=128,
            dtype=bf16, causal=False,
        ))
    # causal: any prefill sq
    for sq in (256, 384, 512, 1024):
        results.append(stage(
            f"MHA  sq=sk={sq:<4d} d=128 bf16 causal",
            sq=sq, sk=sq, hq=5, hk=5, d=128,
            dtype=bf16, causal=True,
        ))

    # --- MHA d=64, prefill_d64 (kBlockM=256, kBlockN=64) ------------------
    for sq in (256, 512, 1024):
        results.append(stage(
            f"MHA  sq=sk={sq:<4d} d=64  bf16 no-mask",
            sq=sq, sk=sq, hq=5, hk=5, d=64,
            dtype=bf16, causal=False,
        ))
    for sq in (256, 512, 1024):
        results.append(stage(
            f"MHA  sq=sk={sq:<4d} d=64  bf16 causal",
            sq=sq, sk=sq, hq=5, hk=5, d=64,
            dtype=bf16, causal=True,
        ))

    # --- GQA-8 d=128, prefill_d128 ---------------------------------------
    for sq in (256, 512, 1024):
        results.append(stage(
            f"GQA8 sq=sk={sq:<4d} d=128 bf16 no-mask",
            sq=sq, sk=sq, hq=64, hk=8, d=128,
            dtype=bf16, causal=False,
        ))
    for sq in (256, 512, 1024):
        results.append(stage(
            f"GQA8 sq=sk={sq:<4d} d=128 bf16 causal",
            sq=sq, sk=sq, hq=64, hk=8, d=128,
            dtype=bf16, causal=True,
        ))

    # --- fp16 spot check (separate kernel instance) ----------------------
    results.append(stage(
        "MHA  sq=sk=512  d=128 fp16 no-mask",
        sq=512, sk=512, hq=5, hk=5, d=128,
        dtype=fp16, causal=False,
    ))
    results.append(stage(
        "MHA  sq=sk=512  d=128 fp16 causal",
        sq=512, sk=512, hq=5, hk=5, d=128,
        dtype=fp16, causal=True,
    ))

    print(f"\n=== {sum(results)} / {len(results)} stages passed ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
