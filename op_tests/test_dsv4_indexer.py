# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Correctness test for DSV4 indexer kernels.

import importlib
import os
import sys

import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load_module(
    "aiter.ops.triton._triton_kernels.attention.dsv4_indexer",
    os.path.join(
        _REPO,
        "aiter",
        "ops",
        "triton",
        "_triton_kernels",
        "attention",
        "dsv4_indexer.py",
    ),
)
_wrapper = _load_module(
    "aiter.ops.triton.attention.dsv4_indexer",
    os.path.join(_REPO, "aiter", "ops", "triton", "attention", "dsv4_indexer.py"),
)

indexer_fwd = _wrapper.indexer_fwd
indexer_bwd = _wrapper.indexer_bwd
dsv4_indexer = _wrapper.dsv4_indexer

torch.set_default_device("cuda")


def ref_indexer_fwd(q, k, w, compress_ratio):
    """PyTorch reference: q=[S,H,Hd], k=[P,Hd], w=[S,H] → scores=[S,P]."""
    S, _H, _Hd = q.shape
    P = k.shape[0]

    dot = torch.einsum("shd,pd->shp", q.float(), k.float())
    relu = F.relu(dot)
    scores = (relu * w.float().unsqueeze(-1)).sum(dim=1)

    s_idx = torch.arange(S, device=q.device)[:, None]
    p_idx = torch.arange(P, device=q.device)[None, :]
    allowed = (p_idx + 1) * compress_ratio - 1 <= s_idx
    scores = torch.where(allowed, scores, float("-inf"))
    return scores


TEST_CONFIGS = [
    # (S, H, Hd, P) — compress_ratio = S // P
    # Basic shapes
    (64, 4, 32, 16),
    (128, 8, 64, 32),
    (256, 8, 128, 64),
    (512, 8, 128, 128),
    (1024, 8, 128, 256),
    # H=16 (from Miles test suite)
    (128, 16, 128, 32),
    (256, 16, 128, 64),
    (512, 16, 128, 128),
    # H=64 V4 real config
    (256, 64, 128, 64),
    (512, 64, 128, 128),
    (1024, 64, 128, 256),
    (2048, 64, 128, 512),
    # C128 layer (compress_ratio=128)
    (2048, 8, 128, 16),
    (1024, 16, 128, 8),
    # Small edge case
    (16, 8, 128, 4),
    # V4-Flash production shape
    (4096, 8, 128, 1024),
]


def test_fwd():
    print("FWD Tests:")
    for S, H, Hd, P in TEST_CONFIGS:
        compress_ratio = S // P
        q = torch.randn(S, H, Hd, dtype=torch.bfloat16)
        k = torch.randn(P, Hd, dtype=torch.bfloat16)
        w = torch.randn(S, H, dtype=torch.float32) * 0.1

        ref = ref_indexer_fwd(q, k, w, compress_ratio)
        out = indexer_fwd(q, k, w, compress_ratio)

        finite_mask = torch.isfinite(ref) & torch.isfinite(out)
        if finite_mask.sum() == 0:
            cos = 1.0
        else:
            cos = F.cosine_similarity(
                ref[finite_mask].unsqueeze(0),
                out[finite_mask].unsqueeze(0),
            ).item()

        inf_match = (torch.isneginf(ref) == torch.isneginf(out)).all().item()

        status = "PASS" if cos > 0.999 and inf_match else "FAIL"
        print(
            f"  S={S} H={H} Hd={Hd} P={P} cr={compress_ratio} ... cos={cos:.6f} inf_ok={inf_match} {status}"
        )
        assert status == "PASS", f"FWD failed: cos={cos}, inf_ok={inf_match}"


def test_bwd():
    print("BWD Tests:")
    for S, H, Hd, P in TEST_CONFIGS:
        compress_ratio = S // P
        q = torch.randn(S, H, Hd, dtype=torch.bfloat16)
        k = torch.randn(P, Hd, dtype=torch.bfloat16)
        w = torch.randn(S, H, dtype=torch.float32) * 0.1

        ref_scores = ref_indexer_fwd(q, k, w, compress_ratio)
        d_scores = torch.randn_like(ref_scores)
        d_scores = torch.where(torch.isfinite(ref_scores), d_scores, 0.0)

        q_f = q.float().requires_grad_(True)
        k_f = k.float().requires_grad_(True)
        w_f = w.float().requires_grad_(True)

        dot_ref = torch.einsum("shd,pd->shp", q_f, k_f)
        relu_ref = F.relu(dot_ref)
        scores_ref = (relu_ref * w_f.unsqueeze(-1)).sum(dim=1)

        s_idx = torch.arange(S, device=q.device)[:, None]
        p_idx = torch.arange(P, device=q.device)[None, :]
        allowed = (p_idx + 1) * compress_ratio - 1 <= s_idx
        scores_ref = torch.where(
            allowed, scores_ref, torch.tensor(0.0, device=q.device)
        )

        (scores_ref * d_scores).sum().backward()
        ref_dq = q_f.grad
        ref_dk = k_f.grad
        ref_dw = w_f.grad

        dq, dk, dw = indexer_bwd(q, k, w, d_scores, compress_ratio)

        dq_cos = F.cosine_similarity(ref_dq.reshape(1, -1), dq.reshape(1, -1)).item()
        dk_cos = F.cosine_similarity(ref_dk.reshape(1, -1), dk.reshape(1, -1)).item()
        dw_cos = F.cosine_similarity(ref_dw.reshape(1, -1), dw.reshape(1, -1)).item()

        status = (
            "PASS" if dq_cos > 0.999 and dk_cos > 0.999 and dw_cos > 0.999 else "FAIL"
        )
        print(
            f"  S={S} H={H} Hd={Hd} P={P} cr={compress_ratio} ... dq={dq_cos:.6f} dk={dk_cos:.6f} dw={dw_cos:.6f} {status}"
        )
        assert status == "PASS", f"BWD failed: dq={dq_cos}, dk={dk_cos}, dw={dw_cos}"


def test_autograd():
    print("AUTOGRAD Tests:")
    for S, H, Hd, P in TEST_CONFIGS:
        compress_ratio = S // P
        q = torch.randn(S, H, Hd, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(P, Hd, dtype=torch.bfloat16, requires_grad=True)
        w = (torch.randn(S, H, dtype=torch.float32) * 0.1).requires_grad_(True)

        scores, _indices = dsv4_indexer(q, k, w, compress_ratio, topk=min(512, P))
        valid = scores != float("-inf")
        loss = scores[valid].sum()
        loss.backward()

        ok = q.grad is not None and k.grad is not None and w.grad is not None
        status = "PASS" if ok else "FAIL"
        print(
            f"  S={S} H={H} Hd={Hd} P={P} cr={compress_ratio} ... grads_exist={ok} {status}"
        )
        assert status == "PASS", "AUTOGRAD failed"


if __name__ == "__main__":
    print("=" * 60)
    print("DSV4 Indexer Kernel — Correctness Tests")
    print("=" * 60)
    test_fwd()
    test_bwd()
    test_autograd()
    print()
    print("ALL TESTS PASSED")
