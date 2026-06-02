"""Accuracy test: FMHA kernel via flash_attn_varlen_flydsl production path.

THD varlen layout: q(total_q, H, D_qk), k(total_k, H, D_qk), v(total_k, H, D_v).
Causal mask always on.

Usage:
    bash run_test.sh
"""

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "FlyDSL")
_BUILD_PKGS = os.path.join(_REPO, "build-fly", "python_packages")
_AITER_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", ".."))
for p in [_BUILD_PKGS, os.path.join(_REPO, "python"), _AITER_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("FLYDSL_ROOT", _REPO)
os.environ["ARCH"] = "gfx1250"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"

import torch
from aiter.ops.flydsl.mha_flydsl import flash_attn_varlen_flydsl

HEAD_QK = 192
HEAD_V = 128


def _ref_mha_varlen(q, k, v, cu_q, cu_k, scale, causal=True):
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
        print(f"{msg}\033[32mpassed\033[0m")
        return 0.0
    mask = ~isClose
    num = mask.sum()
    pct = (num / a.numel()).item()
    delta = (a[mask] - b[mask]).abs()
    color = "\033[31m" if pct > 0.05 else "\033[33m"
    print(f"{msg}{color}{'FAIL' if pct > 0.05 else 'warning'}\033[0m  "
          f"max_err={delta.max():.6f}  {pct:.1%} ({num}/{a.numel()})")
    return pct


def run_varlen_test(cu_q_list, cu_k_list, H=1, causal=True):
    device = torch.device("cuda")
    torch.manual_seed(42)

    cu_q, cu_k = cu_q_list, cu_k_list
    B = len(cu_q) - 1
    total_q, total_k = cu_q[-1], cu_k[-1]
    max_sq = max(cu_q[i+1] - cu_q[i] for i in range(B))
    max_sk = max(cu_k[i+1] - cu_k[i] for i in range(B))

    q = torch.randn(total_q, H, HEAD_QK, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_k, H, HEAD_QK, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_k, H, HEAD_V, dtype=torch.bfloat16, device=device)

    scale = 1.0 / math.sqrt(HEAD_QK)
    cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor(cu_k, dtype=torch.int32, device=device)

    o = flash_attn_varlen_flydsl(
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
    o_cpu = o.cpu().float()
    ref_cpu = ref.cpu().float()

    nan_mask = torch.isnan(o_cpu)
    if nan_mask.any():
        nan_count = nan_mask.sum().item()
        print(f"  [{tag}] WARNING: {nan_count} NaN values in output!")
        for b in range(B):
            sq = cu_q[b+1] - cu_q[b]
            blk = o_cpu[cu_q[b]:cu_q[b+1]]
            bn = torch.isnan(blk).sum().item()
            if bn > 0:
                nan_rows = torch.isnan(blk).any(dim=-1).any(dim=-1).nonzero().squeeze(-1)
                print(f"    batch {b} (sq={sq}): {bn} NaN, rows: {nan_rows[:10].tolist()}")

    err = _checkAllclose(o_cpu, ref_cpu, rtol=1e-2, atol=1e-2, msg=f"  [{tag}] ")
    return err < 0.05


if __name__ == "__main__":
    print("=" * 60)
    print("FMHA Accuracy Tests (THD varlen, causal, production path)")
    print("=" * 60)

    tests = [
        ([0, 128], [0, 128], 1, True),
        ([0, 256], [0, 256], 1, True),
        ([0, 300], [0, 300], 1, True),
        ([0, 384], [0, 384], 1, True),
        ([0, 128, 256], [0, 128, 256], 1, True),
        ([0, 128, 300], [0, 128, 300], 1, True),
        ([0, 481, 581, 982], [0, 481, 581, 982], 1, True),
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
