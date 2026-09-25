#!/usr/bin/env python3
"""Benchmark block skipping in unified_attention's 2D Triton kernel.

Every row carries an `elide` column that reports the number of tiles elided.
That column comes from the KERNEL'S OWN counter: the kernel that actually ran,
in its own arithmetic, over every tile it visited and indicates achieved sparsity.

Output is a table per shape and, with --csv, a machine-readable file whose
`chain_dot` column records TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF -- see
run_ab.sh, which runs this twice and diffs the two arms.
"""
import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get("AITER_SRC", "/workspace/aiter"))

from aiter.ops.triton.attention.unified_attention import unified_attention  # noqa: E402

WARMUP, REPEAT = 5, 20

# 1e-9 skips nothing, so it isolates the fixed cost of the skip check: the
# overhead floor. The rest bracket the usable range. Thresholds that elide
# heavily are a ceiling probe, not an operating point.
DEFAULT_THRESHOLDS = [1e-9, 0.02878, 0.1, 0.3, 1.0, 1.3, 2.0]


def make_synthetic(seqlen, nq, nkv, head_dim, block_size, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    num_blocks = (seqlen + block_size - 1) // block_size
    q = torch.randn(seqlen, nq, head_dim, dtype=torch.bfloat16, device=dev)
    k = torch.randn(num_blocks, block_size, nkv, head_dim, dtype=torch.bfloat16, device=dev)
    v = torch.randn(num_blocks, block_size, nkv, head_dim, dtype=torch.bfloat16, device=dev)
    return _pack(q, k, v, seqlen, head_dim, num_blocks)


def _pack(q, k, v, seqlen, head_dim, num_blocks):
    dev = q.device
    return dict(
        q=q, k=k, v=v, out=torch.empty_like(q),
        cu_seqlens_q=torch.tensor([0, seqlen], dtype=torch.int32, device=dev),
        max_seqlen_q=seqlen,
        seqused_k=torch.tensor([seqlen], dtype=torch.int32, device=dev),
        max_seqlen_k=seqlen,
        softmax_scale=head_dim**-0.5, causal=True, window_size=(-1, -1),
        block_table=torch.arange(num_blocks, dtype=torch.int32,
                                 device=dev).unsqueeze(0),
        softcap=0, q_descale=None, k_descale=None, v_descale=None,
    )


def unified_attention_out(case, threshold, skip_counter=None):
    """One un-timed launch, returning a copy of the output."""
    unified_attention(backend="triton", block_skip_threshold=threshold,
                      skip_counter=skip_counter, **case)
    return case["out"].clone()


def elide_from_kernel(case, threshold):
    """Achieved elision straight from the kernel's own counter.

    Computed by the kernel that actually ran, in its own arithmetic, over every
    tile it visited. Costs one extra launch and two atomics per program.
    """
    if threshold <= 0:
        return 0.0
    buf = torch.zeros(2, dtype=torch.int32, device=case["q"].device)
    unified_attention(backend="triton", block_skip_threshold=threshold,
                      skip_counter=buf, **case)
    torch.cuda.synchronize()
    seen, elided = (int(x) for x in buf.cpu())
    return elided / seen if seen else float("nan")


def timed(case, threshold):
    def go():
        unified_attention(backend="triton", block_skip_threshold=threshold, **case)
    for _ in range(WARMUP):
        go()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(REPEAT):
        go()
    torch.cuda.synchronize()
    return (time.time() - t0) / REPEAT * 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", default="32x8,64x4",
                    help="comma-separated NQxNKV query/KV head counts")
    ap.add_argument("--seqlens", default="8192,16384")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--thresholds", default="")
    ap.add_argument("--skip-elision", action="store_true",
                    help="omit the elision column entirely")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: GPU required")
        return 1
    import triton

    chain_dot = os.environ.get("TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF", "0")
    sched = os.environ.get("AITER_UA_BLASST_SCHED", "1")
    thresholds = ([float(x) for x in a.thresholds.split(",")]
                  if a.thresholds else list(DEFAULT_THRESHOLDS))

    print("=" * 74)
    print("BLASST block skipping, unified_attention 2D")
    print("=" * 74)
    print(f"  device             {torch.cuda.get_device_name(0)}")
    print(f"  torch / triton     {torch.__version__} / {triton.__version__}")
    print(f"  chain-dot patch    TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF={chain_dot}")
    print(f"  scheduling         AITER_UA_BLASST_SCHED={sched}")
    print(f"  inputs             random")
    print("=" * 74)

    cases = []
    for shape in a.shapes.split(","):
        nq, nkv = (int(x) for x in shape.split("x"))
        for seqlen in (int(x) for x in a.seqlens.split(",")):
            cases.append((
                f"{nq}q/{nkv}kv seqlen={seqlen}",
                make_synthetic(seqlen, nq, nkv, a.head_dim, a.block_size),
                nq, nkv, seqlen,
            ))

    rows = []
    for label, case, nq, nkv, seqlen in cases:
        dense = timed(case, 0.0)
        # Keep the dense output to measure how far each threshold moves the
        # answer as skipping is approximate by construction.
        dense_out = unified_attention_out(case, 0.0)
        print(f"\n  {label}     dense {dense:8.3f} ms")
        print(f"  {'lambda':>10} {'ms':>9} {'speedup':>9} {'elide':>8} "
              f"{'rel_err':>11}")
        for thr in thresholds:
            ms = timed(case, thr)
            el = None if a.skip_elision else elide_from_kernel(case, thr)
            out = unified_attention_out(case, thr)
            err = ((out.float() - dense_out.float()).abs().mean()
                   / dense_out.float().abs().mean().clamp_min(1e-6)).item()
            finite = bool(torch.isfinite(out).all())
            print(f"  {thr:>10} {ms:9.3f} {dense / ms:8.4f}x "
                  f"{'' if el is None else f'{100 * el:7.1f}%'} {err:11.6g}"
                  f"{'' if finite else '  NON-FINITE'}")
            rows.append(dict(
                case=label, shape=f"{nq}x{nkv}", seqlen=seqlen, threshold=thr,
                dense_ms=round(dense, 4), ms=round(ms, 4),
                speedup=round(dense / ms, 4),
                elide=("" if el is None else round(el, 4)),
                rel_err=err, finite=finite,
                chain_dot=chain_dot, sched=sched,
            ))
        del case, dense_out
        torch.cuda.empty_cache()

    if a.csv and rows:
        os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
