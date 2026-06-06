"""Unit test for FlyDSL MHA varlen kernel on gfx1250.

Tests with THD packed layout and variable-length sequences
via cu_seqlens. Covers causal, non-causal, sq!=sk,
seqlen_k==0, mixed zero/nonzero batches, and return_lse.

Usage:
    bash run_mha_flydsl_varlen.sh
"""

import os
import sys
import math

# Environment setup must precede aiter imports
os.environ["ARCH"] = "gfx1250"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
os.environ["ENABLE_CK"] = "0"
os.environ.setdefault(
    "FLYDSL_ROOT",
    os.path.join(
        os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..",
            "aiter", "ops", "flydsl", "kernels",
            "mha_1250",
        )),
        "FlyDSL",
    ),
)
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), ".."),
))
sys.path.insert(0, os.path.join(
    os.environ["FLYDSL_ROOT"], "python",
))
sys.path.insert(0, os.path.join(
    os.environ["FLYDSL_ROOT"],
    "build-fly", "python_packages",
))

import torch

import aiter
from aiter.ops.mha import flash_attn_varlen_func

if aiter.get_gfx() != "gfx1250":
    print(
        "Skipping: test requires gfx1250 "
        f"(current: {aiter.get_gfx()})"
    )
    sys.exit(0)

HEAD_DIM_QK = 192
HEAD_DIM_V = 128


def _ref_mha_varlen(q, k, v, cu_q, cu_k, scale, causal=False, return_lse=False):
    """PyTorch reference for varlen THD layout, per-batch."""
    B = len(cu_q) - 1
    outs = []
    lses = []
    for b in range(B):
        sq = cu_q[b+1] - cu_q[b]
        sk = cu_k[b+1] - cu_k[b]
        qb = q[cu_q[b]:cu_q[b+1]].float()
        kb = k[cu_k[b]:cu_k[b+1]].float()
        vb = v[cu_k[b]:cu_k[b+1]].float()
        qk = torch.bmm(qb.permute(1,0,2), kb.permute(1,2,0)) * scale
        if causal:
            mask = torch.triu(
                torch.ones(
                    sq, sk,
                    device=qk.device,
                    dtype=torch.bool,
                ),
                diagonal=sk - sq + 1,
            )
            qk = qk.masked_fill(mask.unsqueeze(0), float('-inf'))
        if return_lse:
            lse_b = torch.logsumexp(qk, dim=-1)
            lses.append(lse_b)
        p = torch.softmax(qk, dim=-1)
        ob = torch.bmm(p, vb.permute(1,0,2))
        outs.append(ob.permute(1,0,2))
    if return_lse:
        return torch.cat(outs, dim=0), lses
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


def run_varlen_test(cu_q_list, cu_k_list, H=1, causal=False, return_lse=False):
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

    result = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_sq, max_sk,
        softmax_scale=scale,
        causal=causal,
        return_lse=return_lse,
    )
    torch.cuda.synchronize()

    if return_lse:
        o, lse = result
    else:
        o = result

    seqs = [cu_q[i+1] - cu_q[i] for i in range(B)]
    tag = f"B={B} H={H} seqs={seqs} causal={causal} lse={return_lse}"

    ref_result = _ref_mha_varlen(
        q, k, v, cu_q, cu_k, scale,
        causal=causal, return_lse=return_lse,
    )
    if return_lse:
        ref, ref_lses = ref_result
    else:
        ref = ref_result

    err = _checkAllclose(o.cpu().float(), ref.cpu().float(),
                         rtol=1e-2, atol=1e-2, msg=f"  [{tag}] out: ")

    if return_lse:
        lse_f = lse.cpu().float()
        for b in range(B):
            sq = cu_q[b+1] - cu_q[b]
            ref_lse_b = ref_lses[b]
            lse_b = lse_f[cu_q[b]:cu_q[b+1]].permute(1, 0)
            lse_err = _checkAllclose(lse_b, ref_lse_b.cpu(),
                                     rtol=1e-2, atol=1e-2,
                                     msg=f"  [{tag}] lse batch {b} (sq={sq}): ")
            err = max(err, lse_err)

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

    # (cu_q, cu_k, H)
    # All combos of causal x return_lse are generated below.
    base_shapes = [
        # --- basic sq == sk ---
        ([0, 128], [0, 128], 1),
        ([0, 184], [0, 184], 128),
        ([0, 341], [0, 341], 128),
        ([0, 5], [0, 5], 128),
        # --- multi-batch ---
        ([0, 481, 581, 982], [0, 481, 581, 982], 128),
        # --- sq != sk ---
        ([0, 128], [0, 512], 1),
        ([0, 128], [0, 256], 1),
        ([0, 128, 256], [0, 512, 1024], 1),
        ([0, 128], [0, 512], 2),
        ([0, 128, 256], [0, 256, 512], 2),
        # --- sq << sk (decode-like) ---
        ([0, 72], [0, 600], 1),
        ([0, 72], [0, 600], 2),
        ([0, 1], [0, 512], 1),
        ([0, 1], [0, 512], 2),
        ([0, 16], [0, 1024], 2),
        ([0, 72, 144], [0, 600, 1200], 2),
        ([0, 1, 129], [0, 512, 1536], 2),
        ([0, 72, 73], [0, 600, 856], 4),
        # --- noncausal various sq/sk ---
        ([0, 128], [0, 256], 1),
        ([0, 128, 384], [0, 128, 384], 1),
        ([0, 128, 384], [0, 256, 640], 2),
        ([0, 300], [0, 300], 2),
        ([0, 128, 256], [0, 256, 512], 4),
        # --- cu_q != cu_k (chunked prefill) ---
        ([0, 693, 1385, 1846], [0, 693, 1385, 2086], 128),
        # --- seqlen_k == 0 (output must be all zeros) ---
        ([0, 128], [0, 0], 1),
        ([0, 256], [0, 0], 2),
        ([0, 128, 256], [0, 0, 0], 1),
        ([0, 300], [0, 0], 4),
        # --- mixed seqlen_k == 0 (some batches zero) ---
        ([0, 128, 256], [0, 0, 128], 1),
        ([0, 128, 384], [0, 128, 0], 2),
        ([0, 128, 256, 384], [0, 0, 0, 128], 1),
    ]
    tests = []
    for cu_q, cu_k, H in base_shapes:
        for causal in [False, True]:
            for return_lse in [False, True]:
                tests.append((cu_q, cu_k, H, causal, return_lse))

    n_pass = 0
    for cu_q, cu_k, H, causal, return_lse in tests:
        seqs = [cu_q[i+1] - cu_q[i] for i in range(len(cu_q)-1)]
        try:
            ok = run_varlen_test(cu_q, cu_k, H=H, causal=causal, return_lse=return_lse)
            if ok:
                n_pass += 1
        except Exception as e:
            print(f"  [seqs={seqs} causal={causal} lse={return_lse}] ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"{n_pass}/{len(tests)} passed")
    print(f"{'='*60}")
