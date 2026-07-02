"""Unit test for FlyDSL MHA varlen kernel on gfx1250.

Tests with THD packed layout and variable-length sequences
via cu_seqlens. Covers causal, non-causal, sq!=sk,
seqlen_k==0, mixed zero/nonzero batches, and return_lse.

Usage:
    bash run_mha_flydsl_varlen.sh
"""

import argparse
import math
import sys

import pandas as pd
import torch

import aiter
from aiter.ops.mha import flash_attn_varlen_func
from aiter.test_common import checkAllclose
from aiter.utility import dtypes

if aiter.get_gfx() != "gfx1250":
    print("Skipping: test requires gfx1250 " f"(current: {aiter.get_gfx()})")
    sys.exit(0)

HEAD_DIM_QK = 192
HEAD_DIM_V = 128


def _time_fn(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    latencies = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        start_event.record()
        fn()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    return sum(latencies) / len(latencies)


def _ref_mha_varlen(q, k, v, cu_q, cu_k, scale, causal=False, return_lse=False):
    """PyTorch reference for varlen THD layout, per-batch."""
    B = len(cu_q) - 1
    outs = []
    lses = []
    for b in range(B):
        sq = cu_q[b + 1] - cu_q[b]
        sk = cu_k[b + 1] - cu_k[b]
        qb = q[cu_q[b] : cu_q[b + 1]].float()
        kb = k[cu_k[b] : cu_k[b + 1]].float()
        vb = v[cu_k[b] : cu_k[b + 1]].float()
        qk = torch.bmm(qb.permute(1, 0, 2), kb.permute(1, 2, 0)) * scale
        if causal:
            mask = torch.triu(
                torch.ones(
                    sq,
                    sk,
                    device=qk.device,
                    dtype=torch.bool,
                ),
                diagonal=sk - sq + 1,
            )
            qk = qk.masked_fill(mask.unsqueeze(0), float("-inf"))
        if return_lse:
            lse_b = torch.logsumexp(qk, dim=-1)
            lses.append(lse_b)
        p = torch.softmax(qk, dim=-1)
        p = torch.nan_to_num(p, nan=0.0)  # all-masked rows: softmax(-inf)=NaN → 0
        ob = torch.bmm(p, vb.permute(1, 0, 2))
        outs.append(ob.permute(1, 0, 2))
    if return_lse:
        return torch.cat(outs, dim=0), lses
    return torch.cat(outs, dim=0)


def run_varlen_test(
    cu_q_list, cu_k_list, H=1, causal=False, return_lse=False, warmup=1, repeat=5
):
    device = torch.device("cuda")
    torch.manual_seed(42)

    cu_q, cu_k = cu_q_list, cu_k_list
    B = len(cu_q) - 1
    total_q, total_k = cu_q[-1], cu_k[-1]

    max_sq = max(cu_q[i + 1] - cu_q[i] for i in range(B))
    max_sk = max(cu_k[i + 1] - cu_k[i] for i in range(B))

    q = torch.randn(total_q, H, HEAD_DIM_QK, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_k, H, HEAD_DIM_QK, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_k, H, HEAD_DIM_V, dtype=torch.bfloat16, device=device)

    scale = 1.0 / math.sqrt(HEAD_DIM_QK)

    cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor(cu_k, dtype=torch.int32, device=device)

    def _run():
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_sq,
            max_sk,
            softmax_scale=scale,
            causal=causal,
            return_lse=return_lse,
        )

    avg_ms = _time_fn(_run, warmup, repeat)
    result = _run()

    if return_lse:
        o, lse = result
    else:
        o = result

    fwd_flop = _fwd_flops_varlen(cu_q, cu_k, H, HEAD_DIM_QK, HEAD_DIM_V, causal)
    fwd_tflops = _tflops(fwd_flop, avg_ms)
    avg_us = avg_ms * 1000

    seqs = [cu_q[i + 1] - cu_q[i] for i in range(B)]
    tag = f"B={B} H={H} seqs={seqs} causal={causal} lse={return_lse}"
    print(f"  [{tag}] avg: {avg_ms:.3f}ms ({avg_us:.1f} us)  {fwd_tflops:.1f} TFLOPS")

    ref_result = _ref_mha_varlen(
        q,
        k,
        v,
        cu_q,
        cu_k,
        scale,
        causal=causal,
        return_lse=return_lse,
    )
    if return_lse:
        ref, ref_lses = ref_result
    else:
        ref = ref_result

    err = checkAllclose(
        o.cpu().float(), ref.cpu().float(), rtol=1e-2, atol=1e-2, msg=f"  [{tag}] out: "
    )

    if return_lse:
        lse_f = lse.cpu().float()
        for b in range(B):
            sq = cu_q[b + 1] - cu_q[b]
            ref_lse_b = ref_lses[b]
            lse_b = lse_f[cu_q[b] : cu_q[b + 1]].permute(1, 0)
            lse_err = checkAllclose(
                lse_b,
                ref_lse_b.cpu(),
                rtol=1e-2,
                atol=1e-2,
                msg=f"  [{tag}] lse batch {b} (sq={sq}): ",
            )
            err = max(err, lse_err)

    if err > 0.0 and B > 1:
        o_f = o.cpu().float()
        r_f = ref.cpu().float()
        for b in range(B):
            sq = cu_q[b + 1] - cu_q[b]
            ob = o_f[cu_q[b] : cu_q[b + 1]]
            rb = r_f[cu_q[b] : cu_q[b + 1]]
            isC = torch.isclose(ob, rb, rtol=1e-2, atol=1e-2)
            bad = (~isC).sum().item()
            if bad > 0:
                delta = (ob[~isC] - rb[~isC]).abs()
                bad_idx = torch.nonzero(~isC)
                toks = bad_idx[:, 0].unique()
                print(
                    f"    batch {b} (sq={sq}): {bad} bad, max_err={delta.max():.6f}, "
                    f"tok_range=[{toks.min().item()}..{toks.max().item()}], "
                    f"n_bad_toks={len(toks)}"
                )

    passed = err < 0.05
    ret = {
        "B": B,
        "H": H,
        "seqs_q": [cu_q[i + 1] - cu_q[i] for i in range(B)],
        "seqs_k": [cu_k[i + 1] - cu_k[i] for i in range(B)],
        "causal": causal,
        "lse": return_lse,
        "avg_us": round(avg_us, 2),
        "tflops": round(fwd_tflops, 2),
        "pass": passed,
    }
    return passed, ret


def _fwd_flops_varlen(cu_q, cu_k, H, d_qk, d_v, causal):
    """FLOPs for varlen forward: sum per-batch QK^T + PV, causal halves each batch."""
    flop = 0
    B = len(cu_q) - 1
    for b in range(B):
        sq = cu_q[b + 1] - cu_q[b]
        sk = cu_k[b + 1] - cu_k[b]
        f = H * (2 * sq * sk * d_qk + 2 * sq * sk * d_v)
        if causal:
            f //= 2
        flop += f
    return flop


def _tflops(flop, ms):
    if ms <= 0:
        return float("inf")
    return flop / ms / 1e9


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL MHA varlen unit test & benchmark (gfx1250, D_qk=192, D_v=128, bf16)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=None,
        help="Batch size. When set, runs a single shape instead of the full suite.\ne.g.: -b 2",
    )
    parser.add_argument(
        "-nh",
        "--nheads",
        type=int,
        default=None,
        help="Number of attention heads.\ne.g.: -nh 2",
    )
    parser.add_argument(
        "-sq",
        "--seqlen_q",
        type=int,
        default=None,
        help="Sequence length of query (uniform across batches).\ne.g.: -sq 124",
    )
    parser.add_argument(
        "-sk",
        "--seqlen_k",
        type=int,
        default=None,
        help="Sequence length of key (uniform across batches).\ne.g.: -sk 712",
    )
    parser.add_argument(
        "-d_qk_v",
        type=dtypes.str2tuple,
        nargs="+",
        default=[(192, 128)],
        help="Dimension of query/key and value. Currently only 192,128 is supported.\n"
        "e.g.: -d_qk_v 192,128",
    )
    parser.add_argument(
        "-c",
        "--causal",
        type=str,
        default=None,
        help="Causal mode: true/false. Default runs both.\ne.g.: -c true",
    )
    parser.add_argument(
        "-l",
        "--return_lse",
        type=str,
        default=None,
        help="Return LSE: true/false. Default runs both.\ne.g.: -l false",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup iterations for benchmark (default 2).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Repeat iterations for benchmark (default 5).",
    )
    parser.add_argument(
        "--varlen",
        action="store_true",
        help="Randomize per-batch sq/sk (requires -b -sq -sk).\n"
        "sq_i ~ [1, SQ], sk_i ~ [1, SK], with sq_i <= sk_i guaranteed.",
    )
    parser.add_argument(
        "--cmp-triton",
        action="store_true",
        help="Also time Triton for each case and print speedup.",
    )
    args = parser.parse_args()

    for d_qk_v in args.d_qk_v:
        assert d_qk_v == (
            192,
            128,
        ), f"Currently only D_qk=192, D_v=128 is supported, got {d_qk_v}"

    def _parse_bool(s):
        if s is None:
            return None
        return s.lower() in ("true", "1", "yes")

    causal_filter = _parse_bool(args.causal)
    lse_filter = _parse_bool(args.return_lse)
    single_shape = all(
        x is not None
        for x in [args.batch_size, args.nheads, args.seqlen_q, args.seqlen_k]
    )

    import random

    def _build_varlen_cu(B, max_seq):
        """Build random cu_seqlens where each batch has seq_i ~ [1, max_seq]."""
        seqs = [random.randint(1, max_seq) for _ in range(B)]
        cu = [0]
        for s in seqs:
            cu.append(cu[-1] + s)
        return cu, seqs

    # Build cu_seqlens once for single_shape mode; reused by both sections.
    if single_shape:
        B, H, SQ, SK = args.batch_size, args.nheads, args.seqlen_q, args.seqlen_k
        if args.varlen:
            random.seed(42)
            cu_k, sk_list = _build_varlen_cu(B, SK)
            cu_q = [0]
            for i in range(B):
                sq_i = random.randint(1, min(SQ, sk_list[i]))
                cu_q.append(cu_q[-1] + sq_i)
            print(f"  [varlen] cu_q={cu_q} cu_k={cu_k}")
        else:
            cu_q = [i * SQ for i in range(B + 1)]
            cu_k = [i * SK for i in range(B + 1)]

    # =====================================================================
    # Run all cases: correctness + timing in one pass
    # =====================================================================
    print("=" * 60)
    print("FlyDSL MHA Varlen Tests")
    print("=" * 60)

    if single_shape:
        base_shapes = [(cu_q, cu_k, H)]
    else:
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
            ([0, 128, 256, 384], [0, 0, 0, 128], 1),
            # --- larger shapes (converted from bench_shapes) ---
            ([0, 512], [0, 512], 128),
            ([0, 1024], [0, 1024], 128),
            ([0, 256, 512, 768, 1024], [0, 256, 512, 768, 1024], 128),
            ([0, 128], [0, 2048], 128),
            ([0, 1], [0, 512], 128),
        ]

    causal_list = [causal_filter] if causal_filter is not None else [False, True]
    lse_list = [lse_filter] if lse_filter is not None else [False, True]

    tests = []
    for cu_q, cu_k, H in base_shapes:
        for causal in causal_list:
            for return_lse in lse_list:
                tests.append((cu_q, cu_k, H, causal, return_lse))

    if args.cmp_triton:
        from aiter.ops.triton.attention.mha import (
            flash_attn_varlen_func as triton_varlen_func,
        )

    n_pass = 0
    collected = []
    for cu_q, cu_k, H, causal, return_lse in tests:
        seqs = [cu_q[i + 1] - cu_q[i] for i in range(len(cu_q) - 1)]
        try:
            ok, ret = run_varlen_test(
                cu_q,
                cu_k,
                H=H,
                causal=causal,
                return_lse=return_lse,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            if args.cmp_triton:
                device = torch.device("cuda")
                total_q, total_k = cu_q[-1], cu_k[-1]
                max_sq = max(cu_q[i + 1] - cu_q[i] for i in range(len(cu_q) - 1))
                max_sk = max(cu_k[i + 1] - cu_k[i] for i in range(len(cu_k) - 1))
                scale = 1.0 / math.sqrt(HEAD_DIM_QK)
                torch.manual_seed(42)
                q = torch.randn(
                    total_q, H, HEAD_DIM_QK, dtype=torch.bfloat16, device=device
                )
                k = torch.randn(
                    total_k, H, HEAD_DIM_QK, dtype=torch.bfloat16, device=device
                )
                v = torch.randn(
                    total_k, H, HEAD_DIM_V, dtype=torch.bfloat16, device=device
                )
                cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=device)
                cu_k_t = torch.tensor(cu_k, dtype=torch.int32, device=device)
                tri_ms = _time_fn(
                    lambda: triton_varlen_func(
                        q=q,
                        k=k,
                        v=v,
                        cu_seqlens_q=cu_q_t,
                        cu_seqlens_k=cu_k_t,
                        max_seqlen_q=max_sq,
                        max_seqlen_k=max_sk,
                        softmax_scale=scale,
                        causal=causal,
                    ),
                    args.warmup,
                    args.repeat,
                )
                fwd_flop = _fwd_flops_varlen(
                    cu_q, cu_k, H, HEAD_DIM_QK, HEAD_DIM_V, causal
                )
                ret["triton_us"] = round(tri_ms * 1000, 2)
                ret["triton_tflops"] = round(_tflops(fwd_flop, tri_ms), 2)
                ret["speedup"] = round(tri_ms / ret["avg_us"] * 1000, 2)
            collected.append(ret)
            if ok:
                n_pass += 1
        except Exception as e:
            print(f"  [seqs={seqs} causal={causal} lse={return_lse}] ERROR: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"{n_pass}/{len(tests)} passed")
    print(f"{'='*60}")
    if collected:
        df = pd.DataFrame(collected)
        aiter.logger.info(f"flydsl_mha_varlen summary:\n{df.to_string(index=False)}")
