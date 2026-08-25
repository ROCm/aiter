# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Correctness test for DSV4 sparse-MLA training kernels.

import importlib
import os
import sys

import torch

# Avoid importing aiter.__init__ which loads C extensions.
# Instead, directly import the pure-Python/Triton modules.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_kernel_mod = _load_module(
    "aiter.ops.triton._triton_kernels.attention.sparse_mla_dsv4_train",
    os.path.join(
        _REPO,
        "aiter",
        "ops",
        "triton",
        "_triton_kernels",
        "attention",
        "sparse_mla_dsv4_train.py",
    ),
)
_wrapper_mod = _load_module(
    "aiter.ops.triton.attention.sparse_mla_dsv4_train",
    os.path.join(
        _REPO, "aiter", "ops", "triton", "attention", "sparse_mla_dsv4_train.py"
    ),
)

sparse_mla_fwd = _wrapper_mod.sparse_mla_fwd
sparse_mla_bwd = _wrapper_mod.sparse_mla_bwd
sparse_mla_dsv4_train = _wrapper_mod.sparse_mla_dsv4_train

torch.set_default_device("cuda")

# ── PyTorch reference ────────────────────────────────────────────────


def ref_sparse_mla_fwd(q, kv, attn_sink, indices, scale):
    """Pure-PyTorch forward: q=[N,H,D], kv=[N_kv,D], indices=[N,topk]."""
    N, H, D = q.shape
    indices.shape[1]

    # Gather KV: [N, topk, D]
    idx_clamped = indices.clamp(min=0).long()
    kv_gathered = kv[idx_clamped]  # [N, topk, D]

    # Scores: [N, H, topk]
    scores = torch.einsum("nhd,ntd->nht", q.float(), kv_gathered.float()) * scale

    # Mask invalid indices
    valid = (indices >= 0).unsqueeze(1).expand_as(scores)  # [N, H, topk]
    scores = scores.masked_fill(~valid, float("-inf"))

    if attn_sink is not None:
        # Add virtual column for sink: logit = attn_sink[h]
        sink_col = attn_sink.unsqueeze(0).unsqueeze(-1).expand(N, H, 1)  # [N,H,1]
        scores = torch.cat([scores, sink_col], dim=-1)  # [N, H, topk+1]
        # Values for sink column are zero
        zero_col = torch.zeros(N, 1, D, device=q.device, dtype=kv_gathered.dtype)
        kv_gathered = torch.cat([kv_gathered, zero_col], dim=1)  # [N, topk+1, D]

    # Softmax
    m = scores.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores - m)
    l = exp_scores.sum(dim=-1, keepdim=True)
    p = exp_scores / l.clamp(min=1e-30)

    # Output: [N, H, D]
    out = torch.einsum("nht,ntd->nhd", p.to(kv_gathered.dtype), kv_gathered)

    # LSE = log(l) + m, shape [N, H]
    lse = (torch.log(l.clamp(min=1e-30)) + m).squeeze(-1)

    return out.to(q.dtype), lse.float()


def cos_sim(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return (a @ b) / (a.norm() * b.norm() + 1e-30)


def max_abs_err(a, b):
    return (a.float() - b.float()).abs().max().item()


# ── Test runner ──────────────────────────────────────────────────────


def test_fwd(N, H, D, N_kv, topk, has_sink=True):
    print(
        f"  FWD  N={N} H={H} D={D} N_kv={N_kv} topk={topk} sink={has_sink} ...", end=" "
    )

    q = torch.randn(N, H, D, dtype=torch.bfloat16)
    kv = torch.randn(N_kv, D, dtype=torch.bfloat16)
    indices = torch.randint(0, N_kv, (N, topk), dtype=torch.int32)
    attn_sink = torch.randn(H, dtype=torch.float32) * 0.1 if has_sink else None
    scale = 1.0 / (D**0.5)

    out, lse = sparse_mla_fwd(q, kv, attn_sink, indices, scale)
    ref_out, ref_lse = ref_sparse_mla_fwd(q, kv, attn_sink, indices, scale)

    cs_out = cos_sim(out, ref_out)
    cs_lse = cos_sim(lse, ref_lse)
    print(f"out_cos={cs_out:.6f}  lse_cos={cs_lse:.6f}", end="")

    ok = cs_out > 0.999 and cs_lse > 0.999
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_bwd(N, H, D, N_kv, topk, has_sink=True):
    print(
        f"  BWD  N={N} H={H} D={D} N_kv={N_kv} topk={topk} sink={has_sink} ...", end=" "
    )

    q = torch.randn(N, H, D, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(N_kv, D, dtype=torch.bfloat16, requires_grad=True)
    indices = torch.randint(0, N_kv, (N, topk), dtype=torch.int32)
    attn_sink = (
        (torch.randn(H, dtype=torch.float32) * 0.1).requires_grad_(True)
        if has_sink
        else None
    )
    scale = 1.0 / (D**0.5)

    # Reference forward + backward
    ref_out, _ref_lse = ref_sparse_mla_fwd(q, kv, attn_sink, indices, scale)
    do = torch.randn_like(ref_out)
    ref_out.backward(do)
    ref_dq = q.grad.clone()
    ref_dkv = kv.grad.clone()
    ref_d_sink = attn_sink.grad.clone() if has_sink else None

    q.grad = None
    kv.grad = None
    if has_sink:
        attn_sink.grad = None

    # Triton forward + backward
    out, lse = sparse_mla_fwd(
        q.detach(),
        kv.detach(),
        attn_sink.detach() if has_sink else None,
        indices,
        scale,
    )
    dq, dkv, d_sink = sparse_mla_bwd(
        q.detach(),
        kv.detach(),
        out,
        do,
        indices,
        lse,
        attn_sink.detach() if has_sink else None,
        scale,
    )

    cs_dq = cos_sim(dq, ref_dq)
    cs_dkv = cos_sim(dkv, ref_dkv)
    msg = f"dq_cos={cs_dq:.6f}  dkv_cos={cs_dkv:.6f}"

    ok = cs_dq > 0.99 and cs_dkv > 0.99
    if has_sink:
        cs_sink = cos_sim(d_sink, ref_d_sink)
        msg += f"  dsink_cos={cs_sink:.6f}"
        ok = ok and cs_sink > 0.99

    print(f"  {msg}", end="")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_autograd(N, H, D, N_kv, topk, has_sink=True):
    print(
        f"  AUTOGRAD  N={N} H={H} D={D} N_kv={N_kv} topk={topk} sink={has_sink} ...",
        end=" ",
    )

    q = torch.randn(N, H, D, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(N_kv, D, dtype=torch.bfloat16, requires_grad=True)
    indices = torch.randint(0, N_kv, (N, topk), dtype=torch.int32)
    attn_sink = (
        (torch.randn(H, dtype=torch.float32) * 0.1).requires_grad_(True)
        if has_sink
        else None
    )
    scale = 1.0 / (D**0.5)

    out = sparse_mla_dsv4_train(q, kv, attn_sink, indices, scale)
    do = torch.randn_like(out)
    out.backward(do)

    has_grads = q.grad is not None and kv.grad is not None
    if has_sink:
        has_grads = has_grads and attn_sink.grad is not None

    ok = has_grads
    print(f"grads_exist={has_grads}  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("=" * 60)
    print("DSV4 Sparse MLA Training Kernel — Correctness Tests")
    print("=" * 60)

    all_pass = True

    # Small shapes for quick testing
    configs = [
        (32, 8, 64, 64, 32),
        (64, 16, 128, 128, 64),
        (128, 16, 256, 256, 128),
    ]

    # Miles sparse MLA shapes (batch=1 → unbatched N,H,D,N_kv,topk)
    miles_configs = [
        (128, 8, 512, 160, 64),
        (256, 8, 512, 320, 128),
        (256, 16, 512, 320, 128),
        (512, 8, 512, 640, 256),
        (512, 16, 512, 640, 128),
        (256, 64, 512, 320, 128),
        (512, 64, 512, 640, 256),
        (1024, 64, 512, 1280, 512),
        (256, 8, 512, 320, 256),
        (256, 8, 512, 320, 64),
    ]
    configs.extend(miles_configs)

    # V4-Flash production shape (if enough memory)
    try:
        torch.empty(512 * 64 * 512, device="cuda", dtype=torch.bfloat16)
        configs.append((512, 64, 512, 640, 640))
    except RuntimeError:
        print("  (skipping V4-Flash shape due to memory)")

    for N, H, D, N_kv, topk in configs:
        for has_sink in [True, False]:
            all_pass &= test_fwd(N, H, D, N_kv, topk, has_sink)
            all_pass &= test_bwd(N, H, D, N_kv, topk, has_sink)
            all_pass &= test_autograd(N, H, D, N_kv, topk, has_sink)

    print()
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
