"""Unit test for FlyDSL MHA kernels on gfx1250.

Two layouts: THD packed [total_tokens, H, D] with variable-length sequences
via cu_seqlens, and batched BSHD [B, S, H, D] with a uniform seq_len. Covers
causal, non-causal, sq!=sk, seqlen_k==0, mixed zero/nonzero batches, GQA, and
return_lse. Both suites run by default; --varlen or --batch runs just one.

Usage:
    python op_tests/test_mha_flydsl.py
"""

import argparse
import math
import sys

import pandas as pd
import torch

import aiter
from aiter.jit.core import is_experimental_enabled
from aiter.ops.mha import flash_attn_func, flash_attn_varlen_func
from aiter.test_common import checkAllclose
from aiter.utility import dtypes

if aiter.get_gfx() != "gfx1250":
    print("Skipping: test requires gfx1250 " f"(current: {aiter.get_gfx()})")
    sys.exit(0)

HEAD_DIM_QK = 192
HEAD_DIM_V = 128

# (d_qk, d_v) pairs each layout's FlyDSL kernel can serve. Extend as the
# kernels grow; anything else is rejected up front rather than silently
# falling through to CK/Triton and testing nothing.
SUPPORTED_D_QK_V = {
    "thd": [(192, 128), (128, 128)],
    "bshd": [(128, 128)],
}


def _check_d_qk_v(layout, d_qk_v):
    assert d_qk_v in SUPPORTED_D_QK_V[layout], (
        f"layout={layout} supports d_qk,d_v in {SUPPORTED_D_QK_V[layout]}, "
        f"got {d_qk_v}"
    )


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


def _expand_kv(t, gqa, dim=-2):
    """Broadcast nheads_k KV heads up to nheads_q for the reference."""
    return t if gqa == 1 else t.repeat_interleave(gqa, dim=dim)


def _causal_mask(sq, sk, device):
    """Bottom-right aligned: query s attends kv [0, s + (sk - sq)]."""
    return torch.triu(
        torch.ones(sq, sk, device=device, dtype=torch.bool), diagonal=sk - sq + 1
    )


def _ref_mha_varlen(q, k, v, cu_q, cu_k, scale, causal=False, return_lse=False):
    """PyTorch reference for varlen THD layout, per-batch."""
    B = len(cu_q) - 1
    gqa = q.shape[1] // k.shape[1]  # q head h reads kv head h // gqa
    outs = []
    lses = []
    for b in range(B):
        sq = cu_q[b + 1] - cu_q[b]
        sk = cu_k[b + 1] - cu_k[b]
        qb = q[cu_q[b] : cu_q[b + 1]].float()
        kb = _expand_kv(k[cu_k[b] : cu_k[b + 1]].float(), gqa)
        vb = _expand_kv(v[cu_k[b] : cu_k[b + 1]].float(), gqa)
        qk = torch.bmm(qb.permute(1, 0, 2), kb.permute(1, 2, 0)) * scale
        if causal:
            mask = _causal_mask(sq, sk, qk.device)
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
    cu_q_list,
    cu_k_list,
    H_q=1,
    H_kv=None,
    d_qk=HEAD_DIM_QK,
    d_v=HEAD_DIM_V,
    causal=False,
    return_lse=False,
    warmup=1,
    repeat=5,
):
    device = torch.device("cuda")
    torch.manual_seed(42)

    _check_d_qk_v("thd", (d_qk, d_v))
    H_kv = H_q if H_kv is None else H_kv
    assert H_q % H_kv == 0, f"nheads_q={H_q} must be a multiple of nheads_kv={H_kv}"

    cu_q, cu_k = cu_q_list, cu_k_list
    B = len(cu_q) - 1
    total_q, total_k = cu_q[-1], cu_k[-1]

    max_sq = max(cu_q[i + 1] - cu_q[i] for i in range(B))
    max_sk = max(cu_k[i + 1] - cu_k[i] for i in range(B))

    q = torch.randn(total_q, H_q, d_qk, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_k, H_kv, d_qk, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_k, H_kv, d_v, dtype=torch.bfloat16, device=device)

    scale = 1.0 / math.sqrt(d_qk)

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

    fwd_flop = _fwd_flops_varlen(cu_q, cu_k, H_q, d_qk, d_v, causal)
    fwd_tflops = _tflops(fwd_flop, avg_ms)
    avg_us = avg_ms * 1000

    seqs = [cu_q[i + 1] - cu_q[i] for i in range(B)]
    tag = (
        f"thd B={B} H={H_q}/{H_kv} d={d_qk}/{d_v} seqs={seqs} "
        f"causal={causal} lse={return_lse}"
    )
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
        "layout": "thd",
        "B": B,
        "H_q": H_q,
        "H_kv": H_kv,
        "d_qk": d_qk,
        "d_v": d_v,
        "seqs_q": [cu_q[i + 1] - cu_q[i] for i in range(B)],
        "seqs_k": [cu_k[i + 1] - cu_k[i] for i in range(B)],
        "causal": causal,
        "lse": return_lse,
        "avg_us": round(avg_us, 2),
        "tflops": round(fwd_tflops, 2),
        "pass": passed,
    }
    return passed, ret


def _ref_mha_batch(q, k, v, scale, causal=False, return_lse=False):
    """PyTorch reference for batched BSHD layout, per-batch.

    Loops over the batch so peak scratch stays at one [H_q, sq, sk] score
    matrix instead of [B, H_q, sq, sk].
    """
    B, sq, _, _ = q.shape
    sk = k.shape[1]
    gqa = q.shape[2] // k.shape[2]
    outs = []
    lses = []
    for b in range(B):
        qb = q[b].float().permute(1, 0, 2)  # [H_q, sq, d_qk]
        kb = _expand_kv(k[b].float(), gqa).permute(1, 0, 2)
        vb = _expand_kv(v[b].float(), gqa).permute(1, 0, 2)
        qk = torch.bmm(qb, kb.transpose(1, 2)) * scale
        if causal:
            mask = _causal_mask(sq, sk, qk.device)
            qk = qk.masked_fill(mask.unsqueeze(0), float("-inf"))
        if return_lse:
            lses.append(torch.logsumexp(qk, dim=-1))  # [H_q, sq]
        p = torch.softmax(qk, dim=-1)
        p = torch.nan_to_num(p, nan=0.0)
        outs.append(torch.bmm(p, vb).permute(1, 0, 2))  # [sq, H_q, d_v]
    out = torch.stack(outs, dim=0)
    if return_lse:
        return out, torch.stack(lses, dim=0)  # [B, H_q, sq]
    return out


def run_batch_test(
    B,
    sq,
    sk,
    H_q=1,
    H_kv=None,
    d_qk=128,
    d_v=128,
    causal=False,
    return_lse=False,
    warmup=1,
    repeat=5,
):
    """Batched BSHD [B, S, H, D] — uniform seq_len, no cu_seqlens."""
    device = torch.device("cuda")
    torch.manual_seed(42)

    _check_d_qk_v("bshd", (d_qk, d_v))
    H_kv = H_q if H_kv is None else H_kv
    assert H_q % H_kv == 0, f"nheads_q={H_q} must be a multiple of nheads_kv={H_kv}"

    q = torch.randn(B, sq, H_q, d_qk, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, sk, H_kv, d_qk, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, sk, H_kv, d_v, dtype=torch.bfloat16, device=device)

    scale = 1.0 / math.sqrt(d_qk)

    def _run():
        return flash_attn_func(
            q,
            k,
            v,
            softmax_scale=scale,
            causal=causal,
            return_lse=return_lse,
        )

    tag = (
        f"bshd B={B} H={H_q}/{H_kv} d={d_qk}/{d_v} sq={sq} sk={sk} "
        f"causal={causal} lse={return_lse}"
    )

    avg_ms = _time_fn(_run, warmup, repeat)
    result = _run()

    if return_lse:
        o, lse = result
    else:
        o = result

    fwd_flop = _fwd_flops_batch(B, sq, sk, H_q, d_qk, d_v, causal)
    fwd_tflops = _tflops(fwd_flop, avg_ms)
    avg_us = avg_ms * 1000

    print(f"  [{tag}] avg: {avg_ms:.3f}ms ({avg_us:.1f} us)  {fwd_tflops:.1f} TFLOPS")

    ref_result = _ref_mha_batch(q, k, v, scale, causal=causal, return_lse=return_lse)
    if return_lse:
        ref, ref_lse = ref_result
    else:
        ref = ref_result

    assert tuple(o.shape) == (B, sq, H_q, d_v), f"[{tag}] bad out {tuple(o.shape)}"

    err = checkAllclose(
        o.cpu().float(), ref.cpu().float(), rtol=1e-2, atol=1e-2, msg=f"  [{tag}] out: "
    )

    if return_lse:
        # Kernel LSE is [B, nheads_q, sq]; the reference matches that layout.
        lse_err = checkAllclose(
            lse.cpu().float(),
            ref_lse.cpu().float(),
            rtol=1e-2,
            atol=1e-2,
            msg=f"  [{tag}] lse: ",
        )
        err = max(err, lse_err)

    passed = err < 0.05
    ret = {
        "layout": "bshd",
        "B": B,
        "H_q": H_q,
        "H_kv": H_kv,
        "d_qk": d_qk,
        "d_v": d_v,
        "seqs_q": [sq] * B,
        "seqs_k": [sk] * B,
        "causal": causal,
        "lse": return_lse,
        "avg_us": round(avg_us, 2),
        "tflops": round(fwd_tflops, 2),
        "pass": passed,
    }
    return passed, ret


def run_route_m16x8_test():
    """Routing check: with AITER_ENABLE_EXPERIMENTAL=1, a D_qk=D_v=128 varlen
    call must dispatch to the experimental m16x8 kernel (both causal and
    non-causal). No correctness check — the kernel body is an empty scaffold and
    writes no output yet; we only assert the dispatch and the output shape.

    TEMPORARY: remove this together with the --route-m16x8 flag once the m16x8
    kernel body is implemented and covered by the normal correctness suite.
    """
    from aiter.jit.core import is_experimental_enabled
    import aiter.ops.flydsl.fmha_kernels as fk

    if not is_experimental_enabled():
        print(
            "  [route-m16x8] SKIP: run with AITER_ENABLE_EXPERIMENTAL=1 to route "
            "the 128/128 path to the m16x8 kernel"
        )
        return True

    device = torch.device("cuda")
    torch.manual_seed(42)
    B, H, S, D = 2, 8, 128, 128
    total = B * S
    q = torch.randn(total, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(total, H, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(total, H, D, dtype=torch.bfloat16, device=device)
    cu = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(D)

    # Spy on the kernel-file entry that flydsl_flash_attn_varlen_func dispatches to.
    calls = {"n": 0}
    orig = fk.flash_attn_varlen_m16x8

    def _spy(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    fk.flash_attn_varlen_m16x8 = _spy
    ok = True
    try:
        for causal in (False, True):
            calls["n"] = 0
            out = flash_attn_varlen_func(
                q, k, v, cu, cu, S, S, softmax_scale=scale, causal=causal
            )
            torch.cuda.synchronize()
            routed = calls["n"] == 1
            shape_ok = tuple(out.shape) == (total, H, D)
            status = "OK" if (routed and shape_ok) else "FAIL"
            print(
                f"  [route-m16x8] causal={causal}: dispatched={calls['n']} "
                f"out={tuple(out.shape)} -> {status}"
            )
            ok = ok and routed and shape_ok
    finally:
        fk.flash_attn_varlen_m16x8 = orig

    print(f"  [route-m16x8] {'PASS' if ok else 'FAIL'}")
    return ok


def _fwd_flops_varlen(cu_q, cu_k, H_q, d_qk, d_v, causal):
    """FLOPs for varlen forward: sum per-batch QK^T + PV, causal halves each batch."""
    flop = 0
    B = len(cu_q) - 1
    for b in range(B):
        sq = cu_q[b + 1] - cu_q[b]
        sk = cu_k[b + 1] - cu_k[b]
        f = H_q * (2 * sq * sk * d_qk + 2 * sq * sk * d_v)
        if causal:
            f //= 2
        flop += f
    return flop


def _fwd_flops_batch(B, sq, sk, H_q, d_qk, d_v, causal):
    """FLOPs for batched forward: QK^T + PV over B uniform-length sequences."""
    f = B * H_q * (2 * sq * sk * d_qk + 2 * sq * sk * d_v)
    return f // 2 if causal else f


def _tflops(flop, ms):
    if ms <= 0:
        return float("inf")
    return flop / ms / 1e9


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL MHA unit test & benchmark (gfx1250, bf16).\n"
        "Runs both the varlen (thd) and batched (bshd) suites by default.",
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
        help="Number of query heads.\ne.g.: -nh 2",
    )
    parser.add_argument(
        "-nhkv",
        "--nheads_kv",
        type=int,
        default=None,
        help="Number of key/value heads (GQA). Must divide --nheads.\n"
        "Defaults to --nheads (plain MHA).\ne.g.: -nhkv 2",
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
        help="Dimension of query/key and value for the varlen (thd) suite.\n"
        f"Supported: {SUPPORTED_D_QK_V['thd']}.\ne.g.: -d_qk_v 192,128",
    )
    parser.add_argument(
        "-bd_qk_v",
        "--batch_d_qk_v",
        type=dtypes.str2tuple,
        nargs="+",
        default=[(128, 128)],
        help="Dimension of query/key and value for the batched (bshd) suite.\n"
        f"Supported: {SUPPORTED_D_QK_V['bshd']}.\ne.g.: -bd_qk_v 128,128",
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
        help="Run only the varlen (thd) suite. Default runs thd and bshd.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run only the batched (bshd) suite. Default runs thd and bshd.",
    )
    parser.add_argument(
        "--rand-seqlens",
        action="store_true",
        help="Randomize per-batch sq/sk for the thd suite (requires -b -sq -sk).\n"
        "sq_i ~ [1, SQ], sk_i ~ [1, SK], with sq_i <= sk_i guaranteed.",
    )
    parser.add_argument(
        "--cmp-triton",
        action="store_true",
        help="Also time Triton for each case and print speedup.",
    )
    # TEMPORARY: routing-only check for the empty m16x8 scaffold.
    # TODO(m16x8): remove --route-m16x8 (and run_route_m16x8_test) once the
    # kernel body is implemented and the normal correctness suite covers 128/128.
    parser.add_argument(
        "--route-m16x8",
        action="store_true",
        help="Routing check only: verify a D_qk=D_v=128 varlen call dispatches to\n"
        "the experimental m16x8 kernel. Requires AITER_ENABLE_EXPERIMENTAL=1.\n"
        "No correctness check (kernel body is an empty scaffold).",
    )
    args = parser.parse_args()

    if args.route_m16x8:
        print("=" * 60)
        print("FlyDSL MHA m16x8 routing check")
        print("=" * 60)
        sys.exit(0 if run_route_m16x8_test() else 1)

    # Neither selector -> run both suites.
    run_thd = args.varlen or not args.batch
    run_bshd = args.batch or not args.varlen

    # Drop cases whose kernel is behind the experimental gate: without it the
    # public wrappers fall through to CK and would report a pass for a FlyDSL
    # kernel that never ran.
    if not is_experimental_enabled():
        if run_bshd:
            print("  SKIP bshd suite: set AITER_ENABLE_EXPERIMENTAL=1")
            run_bshd = False
        if run_thd:
            gated = [d for d in args.d_qk_v if d == (128, 128)]
            if gated:
                print("  SKIP thd d_qk_v=128,128: set AITER_ENABLE_EXPERIMENTAL=1")
                args.d_qk_v = [d for d in args.d_qk_v if d not in gated]
                run_thd = bool(args.d_qk_v)

    for d_qk_v in args.d_qk_v:
        _check_d_qk_v("thd", d_qk_v)
    for d_qk_v in args.batch_d_qk_v:
        _check_d_qk_v("bshd", d_qk_v)

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
        H_KV = args.nheads_kv if args.nheads_kv is not None else H
        assert H % H_KV == 0, f"--nheads {H} must be a multiple of --nheads_kv {H_KV}"
        if args.rand_seqlens:
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
    print("FlyDSL MHA Tests")
    print("=" * 60)

    # thd shapes: (cu_seqlens_q, cu_seqlens_k, nheads_q[, nheads_kv]).
    # nheads_kv defaults to nheads_q (plain MHA).
    if single_shape:
        base_shapes = [(cu_q, cu_k, H, H_KV)]
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
            # --- GQA (nheads_q > nheads_kv) ---
            ([0, 128], [0, 128], 2, 1),
            ([0, 256], [0, 256], 8, 1),  # MQA
            ([0, 341], [0, 341], 8, 2),  # non-tile-multiple seq
            ([0, 512], [0, 512], 16, 4),
            ([0, 128], [0, 512], 6, 3),  # non-power-of-two nheads
            ([0, 128, 384], [0, 256, 640], 8, 2),  # multi-batch, sq != sk
            ([0, 72], [0, 600], 32, 4),  # decode-like
            ([0, 481, 581, 982], [0, 481, 581, 982], 32, 8),
            ([0, 128, 256], [0, 0, 128], 8, 2),  # mixed seqlen_k == 0
            ([0, 1024], [0, 1024], 32, 4),
            # --- larger shapes (converted from bench_shapes) ---
            ([0, 512], [0, 512], 128),
            ([0, 1024], [0, 1024], 128),
            ([0, 256, 512, 768, 1024], [0, 256, 512, 768, 1024], 128),
            ([0, 128], [0, 2048], 128),
            ([0, 1], [0, 512], 128),
        ]

    # bshd shapes: (batch, seqlen_q, seqlen_k, nheads_q[, nheads_kv]).
    if single_shape:
        batch_shapes = [(B, SQ, SK, H, H_KV)]
    else:
        batch_shapes = [
            # --- basic sq == sk ---
            (1, 128, 128, 1),
            (1, 128, 128, 8),
            (2, 256, 256, 8),
            (4, 512, 512, 8),
            # --- non-tile-multiple / tiny seq ---
            (1, 5, 5, 8),
            (1, 341, 341, 4),
            (2, 184, 184, 4),
            # --- sq != sk ---
            (1, 128, 512, 1),
            (2, 128, 512, 4),
            (2, 300, 1024, 2),
            # --- sq << sk (decode-like) ---
            (1, 1, 512, 8),
            (2, 16, 1024, 4),
            # --- GQA (nheads_q > nheads_kv) ---
            (1, 128, 128, 2, 1),
            (2, 256, 256, 8, 1),  # MQA
            (1, 341, 341, 8, 2),  # non-tile-multiple seq
            (2, 512, 512, 16, 4),
            (1, 128, 512, 6, 3),  # non-power-of-two nheads, sq != sk
            (1, 16, 1024, 32, 4),  # decode-like
            (2, 1024, 1024, 32, 8),
            # --- larger shapes ---
            (1, 2048, 2048, 8),
            (1, 4096, 4096, 4),
        ]

    causal_list = [causal_filter] if causal_filter is not None else [False, True]
    lse_list = [lse_filter] if lse_filter is not None else [False, True]

    tests = []
    if run_thd:
        for d_qk, d_v in args.d_qk_v:
            for shape in base_shapes:
                cu_q, cu_k, H = shape[:3]
                H_kv = shape[3] if len(shape) > 3 else H
                for causal in causal_list:
                    for return_lse in lse_list:
                        tests.append(
                            ("thd", cu_q, cu_k, H, H_kv, d_qk, d_v, causal, return_lse)
                        )
    if run_bshd:
        for d_qk, d_v in args.batch_d_qk_v:
            for shape in batch_shapes:
                bs, sq_i, sk_i, H = shape[:4]
                H_kv = shape[4] if len(shape) > 4 else H
                for causal in causal_list:
                    for return_lse in lse_list:
                        tests.append(
                            (
                                "bshd",
                                bs,
                                sq_i,
                                sk_i,
                                H,
                                H_kv,
                                d_qk,
                                d_v,
                                causal,
                                return_lse,
                            )
                        )

    if args.cmp_triton:
        from aiter.ops.triton.attention.mha import (
            flash_attn_varlen_func as triton_varlen_func,
        )

    n_pass = 0
    collected = []
    for case in tests:
        causal, return_lse = case[-2], case[-1]
        try:
            if case[0] == "thd":
                _, cu_q, cu_k, H, H_kv, d_qk, d_v, _, _ = case
                ok, ret = run_varlen_test(
                    cu_q,
                    cu_k,
                    H_q=H,
                    H_kv=H_kv,
                    d_qk=d_qk,
                    d_v=d_v,
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
                    scale = 1.0 / math.sqrt(d_qk)
                    torch.manual_seed(42)
                    q = torch.randn(
                        total_q, H, d_qk, dtype=torch.bfloat16, device=device
                    )
                    k = torch.randn(
                        total_k, H_kv, d_qk, dtype=torch.bfloat16, device=device
                    )
                    v = torch.randn(
                        total_k, H_kv, d_v, dtype=torch.bfloat16, device=device
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
                    fwd_flop = _fwd_flops_varlen(cu_q, cu_k, H, d_qk, d_v, causal)
                    ret["triton_us"] = round(tri_ms * 1000, 2)
                    ret["triton_tflops"] = round(_tflops(fwd_flop, tri_ms), 2)
                    ret["speedup"] = round(tri_ms / ret["avg_us"] * 1000, 2)
            else:
                _, bs, sq_i, sk_i, H, H_kv, d_qk, d_v, _, _ = case
                ok, ret = run_batch_test(
                    bs,
                    sq_i,
                    sk_i,
                    H_q=H,
                    H_kv=H_kv,
                    d_qk=d_qk,
                    d_v=d_v,
                    causal=causal,
                    return_lse=return_lse,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
            collected.append(ret)
            if ok:
                n_pass += 1
        except Exception as e:
            print(f"  [{case[:-2]} causal={causal} lse={return_lse}] ERROR: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"{n_pass}/{len(tests)} passed")
    print(f"{'='*60}")
    if collected:
        df = pd.DataFrame(collected)
        aiter.logger.info(f"flydsl_mha summary:\n{df.to_string(index=False)}")
