# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Verify the one-pass FlyDSL MoE gather-reduce epilogue matches the reference.

The reference (``ref_moe_gather_reduce``) is the per-expert ``index_add_``
scatter loop from ``fused_moe.py``.  The optimized version
(``flydsl_moe_gather_reduce``) is a single gather-reduce kernel pass.

Usage:
    python op_tests/test_moe_gather_reduce.py
    python op_tests/test_moe_gather_reduce.py -t 1 -t 128
    python op_tests/test_moe_gather_reduce.py --dtype bf16
"""

import argparse
import importlib.util
import os
import sys
import types

# Allow `import aiter` on boxes with triton<3.6 (gluon's hard requirement) so
# aiter.test_common.run_perftest is importable for --perf. Harmless otherwise.
os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")

import torch

torch.set_default_device("cuda")

WARP_TILE_M = 16  # tile_m // m_warp in the grouped a8w4/a4w4 path


# ---------------------------------------------------------------------------
# Reference scatter epilogue (mirrors fused_moe.py:1005-1013): the per-expert
# ``index_add_`` loop the optimized FlyDSL kernel must reproduce.
# ---------------------------------------------------------------------------
def ref_moe_gather_reduce(
    grouped_out,    # (E, max_m, model_dim)
    route_tokens,   # (E, max_m) long  -- dest token per grouped row
    route_weights,  # (E, max_m)       -- route weight per grouped row
    counts,         # (E,)             -- valid rows per expert
    token_num,
    doweight_stage1,
):
    E, max_m, model_dim = grouped_out.shape
    moe_out = torch.zeros(
        (token_num, model_dim), dtype=grouped_out.dtype, device=grouped_out.device
    )
    for e in range(E):
        n = int(counts[e].item())
        if n == 0:
            continue
        vals = grouped_out[e, :n]
        if not doweight_stage1:
            vals = vals * route_weights[e, :n].to(vals.dtype).view(-1, 1)
        moe_out.index_add_(0, route_tokens[e, :n], vals)
    return moe_out


def _import_flydsl_gather_reduce():
    """Return (flydsl_moe_gather_reduce, build_gather_reduce_src_rows).

    Prefer the normal package import. If the full ``aiter`` package can't be
    imported in this environment (e.g. a FLIR build mismatch in unrelated
    modules), fall back to loading just the two files we need by path, stubbing
    the parent packages so the heavy package ``__init__`` chain is skipped.
    """
    try:
        from aiter.ops.flydsl.moe_kernels import (
            flydsl_moe_gather_reduce,
            build_gather_reduce_src_rows,
        )

        return flydsl_moe_gather_reduce, build_gather_reduce_src_rows
    except Exception:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base = os.path.join(root, "aiter", "ops", "flydsl")

        def _load(path, name):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        for n in ("aiter", "aiter.ops", "aiter.ops.flydsl", "aiter.ops.flydsl.kernels"):
            sys.modules.setdefault(n, types.ModuleType(n))
        kern = _load(
            os.path.join(base, "kernels", "moe_gather_reduce.py"),
            "aiter.ops.flydsl.kernels.moe_gather_reduce",
        )
        sys.modules["aiter.ops.flydsl.kernels.moe_gather_reduce"] = kern
        mk = _load(os.path.join(base, "moe_kernels.py"), "aiter.ops.flydsl.moe_kernels")
        return mk.flydsl_moe_gather_reduce, mk.build_gather_reduce_src_rows


flydsl_moe_gather_reduce, build_gather_reduce_src_rows = _import_flydsl_gather_reduce()


def _build_grouped_layout(token_num, topk, E_total, ep, model_dim, dtype, seed):
    """Recreate fused_moe's grouped routing for one rank, parallelism degree ep.

    The grouped a8w4/a4w4 path requires ``topk_ids`` to be local expert ids in
    ``[0, E_local)`` with exactly ``topk`` per token (validated in fused_moe). We
    model that: experts are sharded ``E_local = E_total // ep`` and each token
    routes to ``topk`` distinct *local* experts. route_tokens/route_weights are
    derived for the scatter reference; topk_ids/topk_weight feed the kernel.

    Returns grouped_out, topk_ids, topk_weight, route_tokens, route_weights,
    counts, max_m, E_local.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    E_local = max(1, E_total // ep)
    assert topk <= E_local, f"need topk<=E_local, got topk={topk} E_local={E_local}"

    # Each token routes to topk distinct LOCAL experts (the real-path contract).
    topk_ids = torch.stack(
        [torch.randperm(E_local, generator=gen, device="cuda")[:topk]
         for _ in range(token_num)]
    ).to(torch.long)
    topk_weight = torch.rand(
        (token_num, topk), generator=gen, device="cuda", dtype=torch.float32
    )
    topk_weight = (topk_weight / topk_weight.sum(-1, keepdim=True)).to(dtype)

    # route_tokens/route_weights for the scatter reference (token-major order).
    flat_e = topk_ids.reshape(-1)
    flat_t = torch.arange(token_num * topk, device="cuda") // topk
    flat_w = topk_weight.reshape(-1).to(dtype)
    counts = torch.bincount(flat_e, minlength=E_local)
    max_m = int(counts.max().item()) if counts.numel() else 0
    max_m = max(WARP_TILE_M, ((max_m + WARP_TILE_M - 1) // WARP_TILE_M) * WARP_TILE_M)

    route_tokens = torch.zeros((E_local, max_m), dtype=torch.long, device="cuda")
    route_weights = torch.zeros((E_local, max_m), dtype=dtype, device="cuda")
    for e in range(E_local):
        mask = flat_e == e
        n = int(counts[e].item())
        if n == 0:
            continue
        route_tokens[e, :n] = flat_t[mask]
        route_weights[e, :n] = flat_w[mask]

    # Random grouped output; rows beyond counts[e] are deliberately garbage so
    # the test confirms neither path reads them.
    grouped_out = torch.randn(
        (E_local, max_m, model_dim), generator=gen, device="cuda", dtype=torch.float32
    ).to(dtype)
    return (grouped_out, topk_ids, topk_weight, route_tokens, route_weights,
            counts, max_m, E_local)


def _run_one(token_num, topk, E_total, ep, model_dim, dtype, doweight_stage1,
             seed=0, name=""):
    (grouped_out, topk_ids, topk_weight, route_tokens, route_weights,
     counts, max_m, E_local) = _build_grouped_layout(
        token_num, topk, E_total, ep, model_dim, dtype, seed
    )

    # reference: scatter (index_add_) using route_tokens/route_weights
    ref = ref_moe_gather_reduce(
        grouped_out, route_tokens, route_weights, counts, token_num, doweight_stage1
    )
    # optimized: precompute the gather map (argsort-free), then thin launch
    src_rows = build_gather_reduce_src_rows(topk_ids, max_m, E_local)
    gather_w = (
        torch.ones((token_num, topk), dtype=torch.float32, device="cuda")
        if doweight_stage1
        else topk_weight.to(torch.float32)
    )
    opt = flydsl_moe_gather_reduce(grouped_out, src_rows, gather_w)
    torch.cuda.synchronize()

    ref_f = ref.float()
    opt_f = opt.float()
    abs_err = (ref_f - opt_f).abs()
    denom = ref_f.abs().amax().clamp_min(1e-6)
    max_abs = abs_err.amax().item()
    rel_l2 = (abs_err.norm() / ref_f.norm().clamp_min(1e-6)).item()

    # ref accumulates in low precision (index_add_ in bf16/f16); the kernel
    # accumulates in f32. The relative-L2 gap reflects that accumulation
    # difference and is the meaningful gate; max_abs gets a generous guard
    # (a single bf16 element rounding can be a few % of the tensor's peak).
    if dtype == torch.bfloat16:
        atol, rtol_l2 = 0.15 * denom.item(), 1.5e-2
    else:
        atol, rtol_l2 = 0.05 * denom.item(), 4e-3
    ok = (max_abs <= atol) and (rel_l2 <= rtol_l2)
    tag = "PASS" if ok else "FAIL"
    print(
        f"[{tag}] {name:<12} tp={ep} tok={token_num:<5} topk={topk} "
        f"E={E_local:<3}(/{E_total}) dim={model_dim:<5} "
        f"{str(dtype).replace('torch.',''):<8} dw1={int(doweight_stage1)} "
        f"max_abs={max_abs:.3e} rel_l2={rel_l2:.3e} (max_m={max_m})"
    )
    return ok


# Real-world MoE shapes: (name, hidden/model_dim, E_total, topk). hidden is the
# stage2 output width the epilogue reduces over; experts are sharded by TP/EP.
REAL_MODELS = [
    ("gptoss-20b", 2880, 32, 4),
    ("gptoss-120b", 2880, 128, 4),
    ("deepseek-v3", 7168, 256, 8),
]


def _run_perf(args):
    """Profile the gather-reduce epilogue (map build + launch) via run_perftest."""
    os.environ["AITER_LOG_MORE"] = "1"  # makes run_perftest log the per-op table
    from aiter.test_common import run_perftest

    token_num = args.tokens[0] if args.tokens else 1
    dtype = torch.bfloat16
    (grouped_out, topk_ids, topk_weight, _rt, _rw, counts, max_m, E_local) = (
        _build_grouped_layout(token_num, args.topk, args.E, 1, args.model_dim, dtype, 0)
    )
    out = torch.empty((token_num, args.model_dim), dtype=dtype, device="cuda")
    gather_w = topk_weight.to(torch.float32)

    # Profile build (argsort-free src_rows) + thin launch -- the full epilogue.
    def _epilogue():
        src_rows = build_gather_reduce_src_rows(topk_ids, max_m, E_local)
        return flydsl_moe_gather_reduce(grouped_out, src_rows, gather_w, out=out)

    print(
        f"perf: tok={token_num} E={args.E} topk={args.topk} dim={args.model_dim} "
        f"max_m={max_m} dtype=bf16 -> profiling build_src_rows + gather_reduce"
    )
    _, avg_us = run_perftest(
        _epilogue,
        num_iters=args.iters,
        num_warmup=10,
        num_rotate_args=1,
    )
    print(
        f"\n[run_perftest] avg device time: {avg_us:.1f} us/iter "
        f"(per-op breakdown / hot loop logged above via AITER_LOG_MORE=1)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tokens", type=int, action="append", default=None,
                    help="token counts for the real-world sweep (decode=1, prefill=N)")
    ap.add_argument("--tp", type=int, action="append", default=None,
                    help="tensor/expert-parallel degrees (default: 1 and 8)")
    ap.add_argument("--dtype", choices=["bf16", "f16", "both"], default="both")
    ap.add_argument("--skip-generic", action="store_true",
                    help="run only the real-world model shapes")
    # --perf: profile flydsl_moe_gather_reduce with test_common.run_perftest and
    # dump the per-op breakdown (the host-side hot loop) via AITER_LOG_MORE.
    ap.add_argument("--perf", action="store_true",
                    help="profile the gather-reduce hot loop instead of correctness")
    ap.add_argument("--E", type=int, default=32, help="[--perf] experts")
    ap.add_argument("--topk", type=int, default=8, help="[--perf] topk")
    ap.add_argument("--model-dim", type=int, default=4096, help="[--perf] model dim")
    ap.add_argument("--iters", type=int, default=50, help="[--perf] run_perftest iters")
    args = ap.parse_args()

    if args.perf:
        _run_perf(args)
        return

    tokens = args.tokens if args.tokens else [1, 256]
    tps = args.tp if args.tp else [1, 8]
    dtypes = (
        [torch.bfloat16, torch.float16]
        if args.dtype == "both"
        else [torch.bfloat16 if args.dtype == "bf16" else torch.float16]
    )

    # (token_num, topk, E_total, ep, model_dim, dtype, dw1, name)
    configs = []

    # Generic correctness sweep (small shapes, edge cases), single rank.
    if not args.skip_generic:
        for tn in [1, 7, 128]:
            for (E, topk) in [(8, 1), (8, 2), (32, 8)]:
                for model_dim in [512, 4096]:
                    for dt in dtypes:
                        for dw1 in (False, True):
                            configs.append((tn, topk, E, 1, model_dim, dt, dw1, "generic"))

    # Real-world model shapes at TP1 and TP8 (experts sharded by TP), decode and
    # a prefill batch. bf16 is the deployment dtype for these models.
    for (name, hidden, E_total, topk) in REAL_MODELS:
        for tp in tps:
            for tn in tokens:
                for dw1 in (False, True):
                    configs.append(
                        (tn, topk, E_total, tp, hidden, torch.bfloat16, dw1, name)
                    )

    all_ok = True
    for i, (tn, topk, E_total, ep, md, dt, dw1, name) in enumerate(configs):
        if topk > max(1, E_total // ep):
            print(f"[SKIP] {name} tp={ep}: topk={topk} > E_local={E_total // ep}")
            continue
        all_ok &= _run_one(tn, topk, E_total, ep, md, dt, dw1, seed=i, name=name)

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
