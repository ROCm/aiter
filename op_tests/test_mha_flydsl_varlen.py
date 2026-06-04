"""Unit test for FlyDSL MHA varlen kernel on gfx1250.

Tests with THD packed layout and variable-length sequences via cu_seqlens.
Supports both causal and non-causal modes.
Uses the production path (flash_attn_varlen_func -> flash_attn_varlen_flydsl).

Usage:
    bash run_mha_flydsl_varlen.sh
"""

import os
import sys
import math
import torch

_KERNEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "aiter", "ops", "flydsl", "kernels", "mha_1250")
_KERNEL_DIR = os.path.abspath(_KERNEL_DIR)

_REPO = os.path.join(_KERNEL_DIR, "FlyDSL")
_BUILD_PKGS = os.path.join(_REPO, "build-fly", "python_packages")
os.environ.setdefault("FLYDSL_ROOT", _REPO)

_AITER_ROOT = os.path.join(os.path.dirname(__file__), "..")
_AITER_ROOT = os.path.abspath(_AITER_ROOT)
for p in [_BUILD_PKGS, os.path.join(_REPO, "python"), _AITER_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["ARCH"] = "gfx1250"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
os.environ["ENABLE_CK"] = "0"

from aiter.ops.mha import flash_attn_varlen_func

HEAD_DIM_QK = 192
HEAD_DIM_V = 128


def _ref_mha_varlen(q, k, v, cu_q, cu_k, scale, causal=False):
    """PyTorch reference for varlen THD layout, per-batch."""
    B = len(cu_q) - 1
    H = q.shape[1]
    outs = []
    for b in range(B):
        sq = cu_q[b+1] - cu_q[b]
        sk = cu_k[b+1] - cu_k[b]
        qb = q[cu_q[b]:cu_q[b+1]].float()
        kb = k[cu_k[b]:cu_k[b+1]].float()
        vb = v[cu_k[b]:cu_k[b+1]].float()
        qk = torch.bmm(qb.permute(1,0,2), kb.permute(1,2,0)) * scale
        if causal:
            mask = torch.triu(torch.ones(sq, sk, device=qk.device, dtype=torch.bool), diagonal=1)
            qk = qk.masked_fill(mask.unsqueeze(0), float('-inf'))
        p = torch.softmax(qk, dim=-1)
        ob = torch.bmm(p, vb.permute(1,0,2))
        outs.append(ob.permute(1,0,2))
    return torch.cat(outs, dim=0)


def _checkAllclose(a, b, rtol=1e-2, atol=1e-2, msg=""):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)
    if isClose.all():
        print(f"{msg}\033[32mpassed~\033[0m")
        return 0.0
    mask = ~isClose
    num = mask.sum()
    pct = (num / a.numel()).item()
    delta = (a[mask] - b[mask]).abs()
    color = "\033[31m" if pct > 0.05 else "\033[33m"
    print(f"{msg}{color}{'failed' if pct > 0.05 else 'warning'}!\033[0m  "
          f"max={delta.max():.6f}  {pct:.1%} ({num}/{a.numel()})")
    return pct


def run_varlen_test(cu_q_list, cu_k_list, H=1, causal=False):
    device = torch.device("cuda")
    torch.manual_seed(42)

    cu_q, cu_k = cu_q_list, cu_k_list
    B = len(cu_q) - 1
    total_q, total_k = cu_q[-1], cu_k[-1]
    max_sq = max(cu_q[i+1] - cu_q[i] for i in range(B))
    max_sk = max(cu_k[i+1] - cu_k[i] for i in range(B))

    q = torch.randn(total_q, H, HEAD_DIM_QK, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_k, H, HEAD_DIM_QK, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_k, H, HEAD_DIM_V, dtype=torch.bfloat16, device=device)

    scale = 1.0 / math.sqrt(HEAD_DIM_QK)

    cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor(cu_k, dtype=torch.int32, device=device)

    o = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_sq, max_sk,
        softmax_scale=scale,
        causal=causal,
    )
    torch.cuda.synchronize()

    ref = _ref_mha_varlen(q, k, v, cu_q, cu_k, scale, causal=causal)

    seqs = [cu_q[i+1] - cu_q[i] for i in range(B)]
    tag = f"B={B} H={H} seqs={seqs} causal={causal}"
    err = _checkAllclose(o.cpu().float(), ref.cpu().float(),
                         rtol=1e-2, atol=1e-2, msg=f"  [{tag}] ")

    if err > 0.0 and B > 1:
        o_f = o.cpu().float()
        r_f = ref.cpu().float()
        for b in range(B):
            sq = cu_q[b+1] - cu_q[b]
            ob = o_f[cu_q[b]:cu_q[b+1]]
            rb = r_f[cu_q[b]:cu_q[b+1]]
            isC = torch.isclose(ob, rb, rtol=1e-2, atol=1e-2)
            bad = (~isC).sum().item()
            if bad > 0:
                delta = (ob[~isC] - rb[~isC]).abs()
                bad_idx = torch.nonzero(~isC)
                print(f"    batch {b} (sq={sq}): {bad} bad, max_err={delta.max():.6f}, "
                      f"first_tok={bad_idx[0,0].item()}, last_tok={bad_idx[-1,0].item()}")

    if err > 0.0 and B > 1:
        o_f = o.cpu().float()
        r_f = ref.cpu().float()
        for b in range(B):
            sq = cu_q[b+1] - cu_q[b]
            ob = o_f[cu_q[b]:cu_q[b+1]]
            rb = r_f[cu_q[b]:cu_q[b+1]]
            isC = torch.isclose(ob, rb, rtol=1e-2, atol=1e-2)
            bad = (~isC).sum().item()
            if bad > 0:
                delta = (ob[~isC] - rb[~isC]).abs()
                bad_idx = torch.nonzero(~isC)
                toks = bad_idx[:, 0].unique()
                print(f"    batch {b} (sq={sq}): {bad} bad, max_err={delta.max():.6f}, "
                      f"tok_range=[{toks.min().item()}..{toks.max().item()}], "
                      f"n_bad_toks={len(toks)}")

    return err < 0.05


if __name__ == "__main__":
    print("=" * 60)
    print("FlyDSL MHA Varlen Unit Tests")
    print("=" * 60)

    tests = [
        # ([0, 128], [0, 128], 1, False),
        ([0, 128], [0, 128], 1, True),
        # ([0, 184], [0, 184], 1, False),
        ([0, 184], [0, 184], 128, True),
        # ([0, 341], [0, 341], 1, False),
        ([0, 341], [0, 341], 128, True),
        ([0, 5], [0, 5], 128, True),
        ([0, 481, 581, 982], [0, 481, 581, 982], 128, True),
    ]

    n_pass = 0
    for cu_q, cu_k, H, causal in tests:
        seqs = [cu_q[i+1] - cu_q[i] for i in range(len(cu_q)-1)]
        try:
            ok = run_varlen_test(cu_q, cu_k, H=H, causal=causal)
            if ok:
                n_pass += 1
        except Exception as e:
            print(f"  [seqs={seqs} causal={causal}] ERROR: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"{n_pass}/{len(tests)} passed")
    print(f"{'='*60}")
