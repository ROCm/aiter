"""Unit test for FlyDSL MHA batch-mode kernel on gfx1250.

Tests with BSHD layout [B, S, H, D] and fixed-length sequences.
Covers causal, non-causal, sq!=sk (cross-attention),
sq<<sk (decode-like), GQA, and return_lse.

The batch-mode wrapper reshapes BSHD -> THD and calls the
existing varlen kernel with uniform cu_seqlens.

Usage:
    python op_tests/test_mha_flydsl_batch.py
    python op_tests/test_mha_flydsl_batch.py -b 2 -nh 128 -sq 256 -sk 256
    python op_tests/test_mha_flydsl_batch.py -c true -l false
"""

import argparse
import math
import sys

import pandas as pd
import torch

import aiter
from aiter.ops.mha import flash_attn_func
from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_batch_func
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


def _ref_mha_batch(q, k, v, scale, causal=False, return_lse=False):
    """PyTorch reference for BSHD layout.

    q: [B, S_q, H, D_qk]
    k: [B, S_k, H, D_qk]
    v: [B, S_k, H, D_v]
    """
    B, S_q, H, D_qk = q.shape
    S_k = k.shape[1]

    # [B, S, H, D] -> [B, H, S, D]
    qf = q.float().permute(0, 2, 1, 3)
    kf = k.float().permute(0, 2, 1, 3)
    vf = v.float().permute(0, 2, 1, 3)

    # QK^T: [B, H, S_q, S_k]
    qk = torch.matmul(qf, kf.transpose(-2, -1)) * scale

    if causal:
        # Bottom-right aligned causal mask
        mask = torch.triu(
            torch.ones(S_q, S_k, device=qk.device, dtype=torch.bool),
            diagonal=S_k - S_q + 1,
        )
        qk = qk.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    if return_lse:
        lse = torch.logsumexp(qk, dim=-1)  # [B, H, S_q]

    p = torch.softmax(qk, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)

    # PV: [B, H, S_q, D_v] -> [B, S_q, H, D_v]
    out = torch.matmul(p, vf).permute(0, 2, 1, 3)

    if return_lse:
        return out, lse
    return out


def _fwd_flops(B, S_q, S_k, H, d_qk, d_v, causal):
    """FLOPs for batch forward: QK^T + PV, causal halves."""
    f = B * H * (2 * S_q * S_k * d_qk + 2 * S_q * S_k * d_v)
    if causal:
        f //= 2
    return f


def _tflops(flop, ms):
    if ms <= 0:
        return float("inf")
    return flop / ms / 1e9


def run_batch_test(
    B, S_q, S_k, H, causal=False, return_lse=False, warmup=1, repeat=5
):
    device = torch.device("cuda")
    torch.manual_seed(42)

    q = torch.randn(B, S_q, H, HEAD_DIM_QK, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, S_k, H, HEAD_DIM_QK, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, S_k, H, HEAD_DIM_V, dtype=torch.bfloat16, device=device)

    scale = 1.0 / math.sqrt(HEAD_DIM_QK)

    def _run():
        return flydsl_flash_attn_batch_func(
            q,
            k,
            v,
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

    fwd_flop = _fwd_flops(B, S_q, S_k, H, HEAD_DIM_QK, HEAD_DIM_V, causal)
    fwd_tflops = _tflops(fwd_flop, avg_ms)
    avg_us = avg_ms * 1000

    tag = f"B={B} H={H} Sq={S_q} Sk={S_k} causal={causal} lse={return_lse}"
    print(f"  [{tag}] avg: {avg_ms:.3f}ms ({avg_us:.1f} us)  {fwd_tflops:.1f} TFLOPS")

    ref_result = _ref_mha_batch(
        q, k, v, scale, causal=causal, return_lse=return_lse,
    )
    if return_lse:
        ref, ref_lse = ref_result
    else:
        ref = ref_result

    err = checkAllclose(
        o.cpu().float(), ref.cpu().float(), rtol=1e-2, atol=1e-2, msg=f"  [{tag}] out: "
    )

    if return_lse:
        lse_err = checkAllclose(
            lse.cpu().float(),
            ref_lse.cpu().float(),
            rtol=1e-2,
            atol=1e-2,
            msg=f"  [{tag}] lse: ",
        )
        err = max(err, lse_err)

    if err > 0.0 and B > 1:
        o_f = o.cpu().float()
        r_f = ref.cpu().float()
        for b in range(B):
            ob = o_f[b]
            rb = r_f[b]
            isC = torch.isclose(ob, rb, rtol=1e-2, atol=1e-2)
            bad = (~isC).sum().item()
            if bad > 0:
                delta = (ob[~isC] - rb[~isC]).abs()
                print(
                    f"    batch {b}: {bad} bad, max_err={delta.max():.6f}"
                )

    passed = err < 0.05
    ret = {
        "B": B,
        "H": H,
        "S_q": S_q,
        "S_k": S_k,
        "causal": causal,
        "lse": return_lse,
        "avg_us": round(avg_us, 2),
        "tflops": round(fwd_tflops, 2),
        "pass": passed,
    }
    return passed, ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL MHA batch-mode unit test & benchmark\n"
        "(gfx1250, D_qk=192, D_v=128, bf16, BSHD layout)",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=None,
        help="Batch size. When set with -nh/-sq/-sk, runs a single shape.\ne.g.: -b 2",
    )
    parser.add_argument(
        "-nh", "--nheads", type=int, default=None,
        help="Number of attention heads.\ne.g.: -nh 128",
    )
    parser.add_argument(
        "-sq", "--seqlen_q", type=int, default=None,
        help="Sequence length of query.\ne.g.: -sq 256",
    )
    parser.add_argument(
        "-sk", "--seqlen_k", type=int, default=None,
        help="Sequence length of key.\ne.g.: -sk 512",
    )
    parser.add_argument(
        "-c", "--causal", type=str, default=None,
        help="Causal mode: true/false. Default runs both.\ne.g.: -c true",
    )
    parser.add_argument(
        "-l", "--return_lse", type=str, default=None,
        help="Return LSE: true/false. Default runs both.\ne.g.: -l false",
    )
    parser.add_argument(
        "--warmup", type=int, default=2,
        help="Warmup iterations for benchmark (default 2).",
    )
    parser.add_argument(
        "--repeat", type=int, default=5,
        help="Repeat iterations for benchmark (default 5).",
    )
    parser.add_argument(
        "--cmp-triton", action="store_true",
        help="Also time Triton for each case and print speedup.",
    )
    args = parser.parse_args()

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

    # =====================================================================
    # Run all cases: correctness + timing in one pass
    # =====================================================================
    print("=" * 60)
    print("FlyDSL MHA Batch-Mode Tests (BSHD)")
    print("=" * 60)

    if single_shape:
        base_shapes = [
            (args.batch_size, args.seqlen_q, args.seqlen_k, args.nheads),
        ]
    else:
        base_shapes = [
            # --- basic sq == sk ---
            (1, 128, 128, 1),
            (1, 184, 184, 128),
            (1, 256, 256, 128),
            (1, 341, 341, 128),
            # --- multi-batch ---
            (2, 256, 256, 128),
            (4, 128, 128, 128),
            # --- sq != sk (cross-attention) ---
            (1, 128, 512, 1),
            (1, 128, 256, 1),
            (2, 128, 512, 2),
            (2, 256, 512, 2),
            # --- sq << sk (decode-like) ---
            (1, 1, 512, 1),
            (1, 1, 512, 2),
            (2, 1, 512, 128),
            (1, 16, 1024, 2),
            (4, 72, 600, 2),
            # --- larger shapes ---
            (1, 512, 512, 128),
            (1, 1024, 1024, 128),
            (2, 512, 512, 128),
            (1, 128, 2048, 128),
        ]

    causal_list = [causal_filter] if causal_filter is not None else [False, True]
    lse_list = [lse_filter] if lse_filter is not None else [False, True]

    tests = []
    for B, S_q, S_k, H in base_shapes:
        for causal in causal_list:
            for return_lse in lse_list:
                tests.append((B, S_q, S_k, H, causal, return_lse))

    if args.cmp_triton:
        from aiter.ops.triton.attention.mha import (
            flash_attn_func as triton_flash_attn_func,
        )

    n_pass = 0
    collected = []
    for B, S_q, S_k, H, causal, return_lse in tests:
        try:
            ok, ret = run_batch_test(
                B, S_q, S_k, H,
                causal=causal,
                return_lse=return_lse,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            if args.cmp_triton:
                device = torch.device("cuda")
                scale = 1.0 / math.sqrt(HEAD_DIM_QK)
                torch.manual_seed(42)
                q = torch.randn(B, S_q, H, HEAD_DIM_QK, dtype=torch.bfloat16, device=device)
                k = torch.randn(B, S_k, H, HEAD_DIM_QK, dtype=torch.bfloat16, device=device)
                v = torch.randn(B, S_k, H, HEAD_DIM_V, dtype=torch.bfloat16, device=device)
                tri_ms = _time_fn(
                    lambda: triton_flash_attn_func(
                        q=q, k=k, v=v,
                        softmax_scale=scale,
                        causal=causal,
                    ),
                    args.warmup,
                    args.repeat,
                )
                fwd_flop = _fwd_flops(B, S_q, S_k, H, HEAD_DIM_QK, HEAD_DIM_V, causal)
                ret["triton_us"] = round(tri_ms * 1000, 2)
                ret["triton_tflops"] = round(_tflops(fwd_flop, tri_ms), 2)
                ret["speedup"] = round(tri_ms / ret["avg_us"] * 1000, 2)
            collected.append(ret)
            if ok:
                n_pass += 1
        except Exception as e:
            print(f"  [B={B} Sq={S_q} Sk={S_k} H={H} causal={causal} lse={return_lse}] ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"{n_pass}/{len(tests)} passed")
    print(f"{'='*60}")
    if collected:
        df = pd.DataFrame(collected)
        aiter.logger.info(f"flydsl_mha_batch summary:\n{df.to_string(index=False)}")
