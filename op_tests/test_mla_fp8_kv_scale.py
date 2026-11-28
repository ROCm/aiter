# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 MLA decode must apply q_scale and kv_scale (ROCm/aiter#7393).

With FP8 Q and KV, the scores are scaled by q_scale * kv_scale and the output by
kv_scale, since V is read from the same FP8 KV cache. Most MLA tests use unit
scales, which cannot detect a missing multiply. This test compares persistent
and non-persistent decode against an FP32 reference on the dequantized inputs,
for kv_scale = 1 and kv_scale != 1. It exits nonzero on any failure.

    python op_tests/test_mla_fp8_kv_scale.py
"""

import itertools
import sys

import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx

D_QK, D_V = 576, 512
SM_SCALE = 1.0 / D_QK**0.5
NUM_KV_SPLITS = 16


def make_case(nhead, q_len, batch, seq_len, seed):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(batch * q_len, nhead, D_QK, generator=g) * 5).to(dtypes.fp8)
    kv = (torch.randn(batch * seq_len, 1, 1, D_QK, generator=g) * 5).to(dtypes.fp8)
    to_dev = {"dtype": torch.int32, "device": "cuda"}
    return {
        "q": q.cuda(),
        "kv": kv.cuda(),
        "qo_indptr": torch.arange(0, (batch + 1) * q_len, q_len, **to_dev),
        "kv_indptr": torch.arange(0, (batch + 1) * seq_len, seq_len, **to_dev),
        "kv_indices": torch.randperm(batch * seq_len, generator=g).to(**to_dev),
        "kv_last": torch.ones(batch, **to_dev),
    }


def reference(c, nhead, q_len, batch, seq_len, q_scale, kv_scale):
    """Causal FP32 attention on the dequantized inputs; query t sees seq_len - q_len + t + 1 keys."""
    q = c["q"].float().view(batch, q_len, nhead, D_QK) * q_scale
    kv = c["kv"].float().view(-1, D_QK)[c["kv_indices"].long()] * kv_scale
    kv = kv.view(batch, seq_len, D_QK)
    s = torch.einsum("bthd,bkd->bhtk", q, kv) * SM_SCALE
    t = torch.arange(q_len, device="cuda").view(q_len, 1)
    k = torch.arange(seq_len, device="cuda").view(1, seq_len)
    s = s.masked_fill(k > seq_len - q_len + t, float("-inf"))
    o = torch.einsum("bhtk,bkd->bthd", s.softmax(-1), kv[..., :D_V])
    return o.reshape(batch * q_len, nhead, D_V)


def persistent_metadata(c, nhead, q_len, batch):
    info = aiter.get_mla_metadata_info_v1(
        batch,
        q_len,
        nhead,
        dtypes.fp8,
        dtypes.fp8,
        is_sparse=False,
        fast_mode=True,
        num_kv_splits=NUM_KV_SPLITS,
        intra_batch_mode=False,
    )
    names = (
        "work_meta_data",
        "work_indptr",
        "work_info_set",
        "reduce_indptr",
        "reduce_final_map",
        "reduce_partial_map",
    )
    md = {
        n: torch.empty(sz, dtype=dt, device="cuda") for n, (sz, dt) in zip(names, info)
    }
    aiter.get_mla_metadata_v1(
        c["qo_indptr"],
        c["kv_indptr"],
        c["kv_last"],
        nhead,
        1,
        True,
        md["work_meta_data"],
        md["work_info_set"],
        md["work_indptr"],
        md["reduce_indptr"],
        md["reduce_final_map"],
        md["reduce_partial_map"],
        page_size=1,
        kv_granularity=32 if nhead == 64 else 16,
        max_seqlen_qo=q_len,
        uni_seqlen_qo=q_len,
        fast_mode=True,
        max_split_per_batch=NUM_KV_SPLITS,
        intra_batch_mode=False,
        dtype_q_nope=dtypes.fp8,
        dtype_kv_nope=dtypes.fp8,
    )
    return md


def run(c, nhead, q_len, batch, q_scale, kv_scale, persistent):
    o = torch.full(
        (batch * q_len, nhead, D_V), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    md = persistent_metadata(c, nhead, q_len, batch) if persistent else {}
    aiter.mla.mla_decode_fwd(
        c["q"],
        c["kv"],
        o,
        c["qo_indptr"],
        c["kv_indptr"],
        c["kv_indices"],
        c["kv_last"],
        q_len,
        page_size=1,
        nhead_kv=1,
        sm_scale=SM_SCALE,
        num_kv_splits=NUM_KV_SPLITS,
        q_scale=torch.full([1], q_scale, dtype=torch.float32, device="cuda"),
        kv_scale=torch.full([1], kv_scale, dtype=torch.float32, device="cuda"),
        **md,
    )
    torch.cuda.synchronize()
    return o.float()


def main():
    batch, seq_len = 2, 2048
    # (nhead, q_len, persistent); gfx942 routes 8-head FP8 decode to a separate kernel.
    configs = [(16, q, p) for q in (1, 2, 3, 4) for p in (False, True)]
    if get_gfx() == "gfx942":
        configs.append((8, 1, False))
    failures = []
    for (nhead, q_len, persistent), kv_scale in itertools.product(configs, (1.0, 0.05)):
        q_scale = 0.01
        c = make_case(nhead, q_len, batch, seq_len, seed=7393)
        ref = reference(c, nhead, q_len, batch, seq_len, q_scale, kv_scale)
        out = run(c, nhead, q_len, batch, q_scale, kv_scale, persistent)
        ratio = float((out * ref).sum() / (ref * ref).sum())
        err = float((out - ref).abs().max() / ref.abs().max())
        # FP8 rounding gives max_rel_err up to ~0.04 on MI325X; a missing scale gives ~1/kv_scale.
        ok = abs(ratio - 1) < 0.02 and err < 0.1
        print(
            f"{nhead=:3d} {q_len=} {kv_scale=:.2f} {persistent=!s:5}: "
            f"out/ref={ratio:.4f} max_rel_err={err:.4f} {'PASS' if ok else 'FAIL'}",
            flush=True,
        )
        if not ok:
            failures.append((nhead, q_len, kv_scale, persistent, ratio))
    if failures:
        print(f"{len(failures)} failing case(s): {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
