# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter import dtypes
from aiter.fused_moe import topk_softmax_sorting, moe_sorting

BLOCK_SIZE = 32


def ref_chain(gating, num_experts, topk, model_dim, moebuf_dtype, need_renorm):
    M = gating.shape[0]
    device = gating.device
    topk_weights = torch.empty(M, topk, dtype=dtypes.fp32, device=device)
    topk_ids = torch.empty(M, topk, dtype=dtypes.i32, device=device)
    tei = torch.empty(M, topk, dtype=dtypes.i32, device=device)
    aiter.topk_softmax(topk_weights, topk_ids, tei, gating, need_renorm)
    return moe_sorting(
        topk_ids,
        topk_weights,
        num_experts,
        model_dim,
        moebuf_dtype,
        block_size=BLOCK_SIZE,
    )


def run(M, E, K, renorm, dtype):
    torch.manual_seed(0)
    model_dim = 256
    gating = torch.randn(M, E, dtype=dtype, device="cuda")

    r_ids, r_w, r_eid, r_nv, r_buf = ref_chain(
        gating, E, K, model_dim, dtypes.bf16, renorm
    )
    f_ids, f_w, f_eid, f_nv, f_buf = topk_softmax_sorting(
        gating, K, E, model_dim, dtypes.bf16, need_renorm=renorm, block_size=BLOCK_SIZE
    )

    nv_ok = torch.equal(r_nv, f_nv)
    n = int(r_nv[0].item())
    ids_ok = torch.equal(r_ids[:n], f_ids[:n])
    eid_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    eid_ok = torch.equal(r_eid[:eid_blocks], f_eid[:eid_blocks])
    w_ok = torch.allclose(r_w[:n], f_w[:n], atol=1e-5, rtol=1e-4)
    buf_ok = torch.count_nonzero(f_buf).item() == 0

    tag = f"M={M:>3} E={E:>3} K={K} renorm={int(renorm)} {str(dtype):>14}"
    status = "PASS" if (nv_ok and ids_ok and eid_ok and w_ok and buf_ok) else "FAIL"
    print(
        f"[{status}] {tag} | nv={nv_ok} ids={ids_ok} eid={eid_ok} w={w_ok} "
        f"buf={buf_ok} (valid={n})"
    )
    if status == "FAIL":
        # surface first mismatch for debugging
        if not ids_ok:
            diff = (r_ids[:n] != f_ids[:n]).nonzero().flatten()[:8]
            print("   id diff idx:", diff.tolist())
            print("   ref:", r_ids[:n][diff].tolist())
            print("   fus:", f_ids[:n][diff].tolist())
        if not w_ok:
            d = (r_w[:n] - f_w[:n]).abs()
            print("   max w diff:", d.max().item())
    return status == "PASS"


if __name__ == "__main__":
    ok = True
    for dtype in [dtypes.fp32, dtypes.bf16, dtypes.fp16]:
        for renorm in [True, False]:
            for (M, E, K) in [
                (1, 128, 4),
                (8, 128, 4),
                (32, 128, 4),
                (64, 128, 4),
                (1, 256, 8),
                (16, 256, 8),
                (32, 256, 8),
                (60, 256, 8),
                (7, 128, 4),
                (13, 256, 8),
            ]:
                ok &= run(M, E, K, renorm, dtype)
    print("\nALL PASS" if ok else "\nSOME FAILED")
