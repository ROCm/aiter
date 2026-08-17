#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Stage-1 validation for the decode-specialized fp8 kernel (SILOTIGER-877).

Milestone 1a = linear 4D paged cache; 1b = 5D shuffled cache. Both are driven
through the SAME compute core (only the global->LDS staging differs), and both
are validated against a per-sequence fp32 decode reference built from the
dequantized fp8 QKV -- err / cosine / bad-row count (bad-row count is the sharp
signal).

Input modes, in order:
  1. all-1s     -- layout check (does NOT catch a wrong-axis softmax).
  2. distinct   -- per-(m_q, n_kv) distinct logits + per-token V; catches a
                   wrong-axis (lane%16 vs M-axis) softmax reduction.
  3. random     -- realistic.

Run from the worktree root:
    cd ~/projects/aiter/flydsl-unified-attention
    rm -rf ~/.flydsl/cache
    ENABLE_CK=0 GPU_ARCHS=gfx950 PYTHONPATH=$PWD python3 -u scripts/test_stage1_decode.py
"""

import os
import shutil
import sys

sys.path.insert(0, os.getcwd())

import torch

from aiter.ops.flydsl.kernels.flash_attn_fp8_decode_gfx950 import (
    build_flash_attn_fp8_decode_module,
)

DEV = "cuda"
PAGE = 64
H, HKV, D = 64, 4, 128
VEC = 16
OUT_DT = {"bf16": torch.bfloat16, "f16": torch.float16}
NUM_WAVES = int(os.environ.get("DECODE_NUM_WAVES", "4"))


def clear_cache():
    p = os.path.expanduser("~/.flydsl/cache")
    if os.path.isdir(p):
        shutil.rmtree(p)


def q8(x):
    s = x.abs().amax().clamp(min=1e-4) / 448.0
    return (x / s).to(torch.float8_e4m3fn), s.reshape(1).float().to(DEV)


def shuffle_k_to_vectorized(kpool, vec=VEC):
    nb = kpool.shape[0]
    x = kpool.permute(0, 2, 3, 1).contiguous()  # [nb,HKV,D,PAGE]
    x = x.view(nb, HKV, D // vec, vec, PAGE)
    x = x.permute(0, 1, 2, 4, 3).contiguous()  # [nb,HKV,D//vec,PAGE,vec]
    return x


def shuffle_v_to_vectorized(vpool, vec=VEC):
    nb = vpool.shape[0]
    x = vpool.permute(0, 2, 3, 1).contiguous()  # [nb,HKV,D,PAGE]
    x = x.view(nb, HKV, D, PAGE // vec, vec)
    x = x.permute(0, 1, 3, 2, 4).contiguous()  # [nb,HKV,PAGE//vec,D,vec]
    return x


def make_inputs(mode, seqs, seed):
    """Ragged decode inputs. Returns qf/kf/vf (fp8) + scales, cu_kv, and the
    per-seq kv lengths. Q is [b,1,H,D]; K/V are packed [total_kv, HKV, D]."""
    b = len(seqs)
    total_kv = sum(seqs)
    g = torch.Generator(device=DEV).manual_seed(seed)
    if mode == "ones":
        q = torch.ones(b, 1, H, D, device=DEV, dtype=torch.float32)
        k = torch.ones(total_kv, HKV, D, device=DEV, dtype=torch.float32)
        v = torch.ones(total_kv, HKV, D, device=DEV, dtype=torch.float32)
    elif mode == "distinct":
        # Per-(m_q, n_kv) distinct logits: q depends on q-head, k on token; V
        # depends on token so a wrong-axis reduction cannot coincide.
        q = torch.zeros(b, 1, H, D, device=DEV, dtype=torch.float32)
        for qh in range(H):
            q[:, 0, qh, :] = 0.20 + 0.02 * qh
        k = torch.zeros(total_kv, HKV, D, device=DEV, dtype=torch.float32)
        v = torch.zeros(total_kv, HKV, D, device=DEV, dtype=torch.float32)
        tok = torch.arange(total_kv, device=DEV, dtype=torch.float32)
        for hh in range(HKV):
            k[:, hh, :] = (0.10 + 0.05 * ((tok % 37) / 37.0)).unsqueeze(1)
            for dd in range(D):
                v[:, hh, dd] = 0.05 + 0.90 * ((tok + dd) % 53) / 53.0
    else:  # random
        q = torch.randn(b, 1, H, D, generator=g, device=DEV, dtype=torch.float32)
        k = torch.randn(total_kv, HKV, D, generator=g, device=DEV, dtype=torch.float32)
        v = torch.randn(total_kv, HKV, D, generator=g, device=DEV, dtype=torch.float32)
    qf, qs = q8(q)
    kf, ks = q8(k)
    vf, vs = q8(v)
    cu = torch.zeros(b + 1, device=DEV, dtype=torch.int32)
    cu[1:] = torch.tensor(seqs, device=DEV).cumsum(0)
    return qf, qs, kf, ks, vf, vs, cu


def build_pool(kf, vf, seqs, cu, seed):
    """Scatter packed ragged KV into a scrambled linear page pool + block table.
    Per-seq page pool, block_table [b, max_pages]."""
    b = len(seqs)
    npages_per = [(ln + PAGE - 1) // PAGE for ln in seqs]
    stride = max(npages_per)
    tot_pages = sum(npages_per)
    pg = torch.Generator().manual_seed(seed + 101)
    flat = torch.randperm(tot_pages, generator=pg).tolist()
    kpool = torch.zeros(tot_pages, PAGE, HKV, D, device=DEV, dtype=kf.dtype)
    vpool = torch.zeros_like(kpool)
    bt = torch.zeros(b, stride, device=DEV, dtype=torch.int32)
    c = 0
    for bi, ln in enumerate(seqs):
        lo = int(cu[bi])
        for t in range(npages_per[bi]):
            pid = flat[c]
            c += 1
            s0 = lo + t * PAGE
            s1 = min(lo + (t + 1) * PAGE, lo + ln)
            npart = s1 - s0
            kpool[pid, :npart] = kf[s0:s1]
            vpool[pid, :npart] = vf[s0:s1]
            bt[bi, t] = pid
    return kpool, vpool, bt, stride


def reference(qf, qs, kf, ks, vf, vs, cu, seqs):
    b = len(seqs)
    out = torch.empty(b, 1, H, D, device=DEV, dtype=torch.float32)
    for i in range(b):
        lo, hi = int(cu[i]), int(cu[i + 1])
        qi = qf[i].float() * qs  # [1,H,D]
        ki = kf[lo:hi].float() * ks  # [Lk,HKV,D]
        vi = vf[lo:hi].float() * vs
        rep = H // HKV
        kt = ki.repeat_interleave(rep, 1).transpose(0, 1)  # [H,Lk,D]
        vt = vi.repeat_interleave(rep, 1).transpose(0, 1)
        att = qi.transpose(0, 1) @ kt.transpose(-1, -2) / (D**0.5)  # [H,1,Lk]
        out[i] = (att.softmax(-1) @ vt).transpose(0, 1)
    return out


def _launch(mode, seqs, layout, out_dt, seed):
    """Build inputs, run the kernel, return (output, reference, inputs)."""
    qf, qs, kf, ks, vf, vs, cu = make_inputs(mode, seqs, seed)
    kpool, vpool, bt, stride = build_pool(kf, vf, seqs, cu, seed)
    if layout == "vectorized":
        k_arg = shuffle_k_to_vectorized(kpool)
        v_arg = shuffle_v_to_vectorized(vpool)
    else:
        k_arg, v_arg = kpool, vpool

    b = len(seqs)
    mod = build_flash_attn_fp8_decode_module(
        num_heads=H,
        head_dim=D,
        num_kv_heads=HKV,
        causal=True,
        out_dtype_str=out_dt,
        varlen=True,
        paged=True,
        kv_cache_layout=layout,
        num_waves=NUM_WAVES,
    )
    o = torch.zeros(b, 1, H, D, device=DEV, dtype=OUT_DT[out_dt])
    cu_q = torch.arange(b + 1, device=DEV, dtype=torch.int32)
    mod(
        qf.contiguous().view(-1),
        k_arg.contiguous().view(-1),
        v_arg.contiguous().view(-1),
        o.view(-1),
        b,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu,
        block_table=bt.contiguous().view(-1),
        block_table_stride=stride,
        q_descale=qs,
        k_descale=ks,
        v_descale=vs,
    )
    torch.cuda.synchronize()
    return o, reference(qf, qs, kf, ks, vf, vs, cu, seqs)


def run_one(mode, seqs, layout, out_dt, seed=7):
    o, ref = _launch(mode, seqs, layout, out_dt, seed)
    of = o.float()
    err = (of - ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        of.flatten(), ref.flatten(), dim=0
    ).item()
    bad = int(((of - ref).abs().amax(dim=(2, 3)) > 1e-1).sum().item())
    n_rows = len(seqs) * H
    tag = f"{mode:8s} {layout:10s} {out_dt:4s}"
    ok = (bad == 0) and (cos > 0.99)
    print(
        f"[{'PASS' if ok else 'FAIL'}] {tag}  err={err:.4g}  cos={cos:.6f}  "
        f"bad={bad}/{n_rows}"
    )
    return ok


def check_isolation(seqs, seed=7):
    """1a->1b isolation invariant: the linear and 5D-shuffled caches carry the
    same logical KV and the compute core is untouched between them, so the
    kernel OUTPUT must be BIT-IDENTICAL (not merely close). This hardens the
    invariant against a compute-core change slipping into the 1b staging path."""
    o_lin, _ = _launch("random", seqs, "linear", "bf16", seed)
    o_vec, _ = _launch("random", seqs, "vectorized", "bf16", seed)
    same = torch.equal(o_lin, o_vec)
    print(
        f"[{'PASS' if same else 'FAIL'}] isolation: torch.equal(linear, vectorized) = {same}"
    )
    assert (
        same
    ), "1a->1b isolation broken: linear vs vectorized output not bit-identical"
    return same


def main():
    clear_cache()
    # ragged: short (<1 tile), single-page, odd, long -> per-seq segment math
    small = [300, 64, 1000, 4096]
    ok = True
    print("=== Stage 1 decode kernel validation (ragged varlen, causal) ===")
    for layout in ("linear", "vectorized"):
        for out_dt in ("bf16", "f16"):
            for mode in ("ones", "distinct", "random"):
                ok &= run_one(mode, small, layout, out_dt)
    print("\n=== scale-up: b=8, ctx~4096 (random) ===")
    big = [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096]
    for layout in ("linear", "vectorized"):
        ok &= run_one("random", big, layout, "bf16")

    print("\n=== 1a->1b isolation (bit-identity) ===")
    ok &= check_isolation(small)

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
