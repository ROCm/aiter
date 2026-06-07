# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Verify the one-pass FlyDSL MoE route-gather (scatter-copy) matches the ref.

The reference (``ref_moe_scatter_copy_token``) is the per-expert copy loop from
``fused_moe.py`` that gathers each token's payload/scale into the grouped
per-expert layout. The optimized version (``flydsl_moe_scatter_copy_token``)
does the row copies in a single FlyDSL kernel pass. The op is a pure copy, so
outputs must be *byte-exact*.

Usage:
    python op_tests/test_moe_scatter_copy_token.py
    python op_tests/test_moe_scatter_copy_token.py -t 1 -t 256
    python op_tests/test_moe_scatter_copy_token.py --tp 1 --tp 8
    python op_tests/test_moe_scatter_copy_token.py --skip-generic
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
# Reference route-gather (mirrors fused_moe.py): the per-expert copy loop the
# optimized FlyDSL kernel must reproduce, byte for byte.
# ---------------------------------------------------------------------------
def ref_moe_scatter_copy_token(
    a1_payload,  # (token_num, Wp) uint8
    a1_scale_token_u8,  # (token_num, Ws) uint8 or None
    flat_experts,  # (token_num*topk,) long
    flat_tokens,  # (token_num*topk,) long
    flat_weights,  # (token_num*topk,) dtype
    counts,  # (E,)
    E,
    max_m,
):
    device = a1_payload.device
    Wp = a1_payload.shape[1]
    grouped_a1 = torch.zeros((E, max_m, Wp), dtype=torch.uint8, device=device)
    route_tokens = torch.zeros((E, max_m), dtype=torch.long, device=device)
    route_weights = torch.zeros((E, max_m), dtype=flat_weights.dtype, device=device)
    a1_scale_raw = None
    if a1_scale_token_u8 is not None:
        Ws = a1_scale_token_u8.shape[1]
        a1_scale_raw = torch.zeros((E, max_m, Ws), dtype=torch.uint8, device=device)

    for e in range(E):
        mask = flat_experts == e
        n = int(counts[e].item())
        if n == 0:
            continue
        toks = flat_tokens[mask]
        grouped_a1[e, :n].copy_(a1_payload[toks])
        if a1_scale_token_u8 is not None:
            a1_scale_raw[e, :n].copy_(a1_scale_token_u8[toks])
        route_tokens[e, :n].copy_(toks)
        route_weights[e, :n].copy_(flat_weights[mask])
    return grouped_a1, a1_scale_raw, route_tokens, route_weights


def _import_flydsl():
    """Return (flydsl_moe_scatter_copy_token, build_topids_to_rows).

    Prefer the normal package import. If the full ``aiter`` package can't be
    imported in this environment (e.g. a FLIR build mismatch in unrelated
    modules), fall back to loading just the two files we need by path, stubbing
    the parent packages so the heavy package ``__init__`` chain is skipped.
    """
    try:
        from aiter.ops.flydsl.moe_kernels import (
            flydsl_moe_scatter_copy_token,
            build_topids_to_rows,
        )

        return flydsl_moe_scatter_copy_token, build_topids_to_rows
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
            os.path.join(base, "kernels", "moe_scatter_copy_token.py"),
            "aiter.ops.flydsl.kernels.moe_scatter_copy_token",
        )
        sys.modules["aiter.ops.flydsl.kernels.moe_scatter_copy_token"] = kern
        mk = _load(os.path.join(base, "moe_kernels.py"), "aiter.ops.flydsl.moe_kernels")
        return mk.flydsl_moe_scatter_copy_token, mk.build_topids_to_rows


flydsl_moe_scatter_copy_token, build_topids_to_rows = _import_flydsl()


def _rows_to_tokens_from(topids_to_rows, token_num, topk, E, max_m):
    """Inverse of topids_to_rows: grouped row -> source token (-1 for padding rows).
    (In fused_moe this is produced by build_route_maps; here we derive it from the
    deterministic topids_to_rows for byte-exact testing.)"""
    flat_tokens = torch.arange(token_num * topk, device="cuda") // topk
    rows_to_tokens = torch.full((E * max_m,), -1, dtype=torch.int32, device="cuda")
    rows_to_tokens[topids_to_rows.reshape(-1).long()] = flat_tokens.to(torch.int32)
    return rows_to_tokens


def _build_routing(token_num, topk, E_total, ep, model_dim, fp4, dtype, seed):
    """All-local routing (the real-path contract: topk_ids in [0, E_local),
    exactly topk per token), parallelism degree ``ep`` -> ``E_local``.

    Returns a1_payload, a1_scale, topk_ids, topk_weight, flat_experts,
    flat_tokens, flat_weights, counts, E_local, max_m. The flat_* / counts are
    for the scatter reference; topk_ids/topk_weight feed the kernel.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    E_local = max(1, E_total // ep)
    assert topk <= E_local, f"need topk<=E_local, got topk={topk} E_local={E_local}"

    topk_ids = torch.stack(
        [
            torch.randperm(E_local, generator=gen, device="cuda")[:topk]
            for _ in range(token_num)
        ]
    ).to(torch.long)
    topk_weight = torch.rand(
        (token_num, topk), generator=gen, device="cuda", dtype=torch.float32
    )
    topk_weight = (topk_weight / topk_weight.sum(-1, keepdim=True)).to(dtype)

    flat_experts = topk_ids.reshape(-1)
    flat_tokens = torch.arange(token_num * topk, device="cuda") // topk
    flat_weights = topk_weight.reshape(-1).to(dtype)
    counts = torch.bincount(flat_experts, minlength=E_local)
    max_m = int(counts.max().item()) if counts.numel() else 0
    max_m = max(WARP_TILE_M, ((max_m + WARP_TILE_M - 1) // WARP_TILE_M) * WARP_TILE_M)

    Wp = (model_dim // 2) if fp4 else model_dim  # a4w4 packs 2 fp4 per byte
    Ws = model_dim // 32  # per-32 e8m0 scale, 1 byte each
    a1_payload = torch.randint(
        0, 256, (token_num, Wp), dtype=torch.uint8, device="cuda"
    )
    a1_scale = torch.randint(0, 256, (token_num, Ws), dtype=torch.uint8, device="cuda")
    return (
        a1_payload,
        a1_scale,
        topk_ids,
        topk_weight,
        flat_experts,
        flat_tokens,
        flat_weights,
        counts,
        E_local,
        max_m,
    )


def _run_one(token_num, topk, E_total, ep, model_dim, fp4, dtype, seed=0, name=""):
    (
        a1_payload,
        a1_scale,
        topk_ids,
        topk_weight,
        flat_experts,
        flat_tokens,
        flat_weights,
        counts,
        E_local,
        max_m,
    ) = _build_routing(token_num, topk, E_total, ep, model_dim, fp4, dtype, seed)

    # reference: per-expert copy loop using flat routing
    ref = ref_moe_scatter_copy_token(
        a1_payload,
        a1_scale,
        flat_experts,
        flat_tokens,
        flat_weights,
        counts,
        E_local,
        max_m,
    )
    # optimized: scatter-copy is a pure copy driven by rows_to_tokens (the inverse
    # of the deterministic topids_to_rows). It produces only grouped_a1 +
    # a1_scale_raw (route_tokens/route_weights are not its job anymore).
    topids_to_rows = build_topids_to_rows(topk_ids, max_m, E_local)
    rows_to_tokens = _rows_to_tokens_from(
        topids_to_rows, token_num, topk, E_local, max_m
    )
    opt = flydsl_moe_scatter_copy_token(
        a1_payload,
        a1_scale,
        rows_to_tokens,
        E_local,
        max_m,
    )
    torch.cuda.synchronize()

    names = ("grouped_a1", "a1_scale_raw")
    ok = True
    bad = ""
    for nm, r, o in zip(names, ref[:2], opt):
        if r is None and o is None:
            continue
        if not torch.equal(r, o):
            ok = False
            bad = nm
            break

    Wp = a1_payload.shape[1]
    Ws = a1_scale.shape[1]
    tag = "PASS" if ok else f"FAIL({bad})"
    print(
        f"[{tag}] {name:<12} tp={ep} tok={token_num:<5} topk={topk} "
        f"E={E_local:<3}(/{E_total}) dim={model_dim:<5} fp4={int(fp4)} "
        f"Wp={Wp:<4} Ws={Ws:<3} (max_m={max_m})"
    )
    return ok


# Real-world MoE shapes: (name, hidden/model_dim, E_total, topk).
REAL_MODELS = [
    ("gptoss-20b", 2880, 32, 4),
    ("gptoss-120b", 2880, 128, 4),
    ("deepseek-v3", 7168, 256, 8),
]


def _run_perf(args):
    """Profile the scatter-copy epilogue (map build + launch) via run_perftest."""
    os.environ["AITER_LOG_MORE"] = "1"  # makes run_perftest log the per-op table
    from aiter.test_common import run_perftest

    token_num = args.tokens[0] if args.tokens else 1
    dtype = torch.bfloat16
    (
        a1_payload,
        a1_scale,
        topk_ids,
        topk_weight,
        _fe,
        _ft,
        _fw,
        counts,
        E_local,
        max_m,
    ) = _build_routing(
        token_num, args.topk, args.E, 1, args.model_dim, args.fp4, dtype, 0
    )
    Wp, Ws = a1_payload.shape[1], a1_scale.shape[1]
    # Pre-allocated output buffers (as fused_moe passes them): kernel writes only
    # valid rows, so a1_scale_raw padding stays 127.
    grouped_a1 = torch.zeros((E_local, max_m, Wp), dtype=torch.uint8, device="cuda")
    a1_scale_raw = torch.full(
        (E_local, max_m, Ws), 127, dtype=torch.uint8, device="cuda"
    )

    topids_to_rows = build_topids_to_rows(topk_ids, max_m, E_local)
    rows_to_tokens = _rows_to_tokens_from(
        topids_to_rows, token_num, args.topk, E_local, max_m
    )

    # Profile the scatter-copy launch (maps precomputed, as in fused_moe).
    def bench():
        return flydsl_moe_scatter_copy_token(
            a1_payload,
            a1_scale,
            rows_to_tokens,
            E_local,
            max_m,
            grouped_a1=grouped_a1,
            a1_scale_raw=a1_scale_raw,
        )

    print(
        f"perf: tok={token_num} E={args.E} topk={args.topk} dim={args.model_dim} "
        f"fp4={int(args.fp4)} Wp={Wp} Ws={Ws} max_m={max_m} "
        f"-> profiling build_topids_to_rows + scatter_copy"
    )
    _, avg_us = run_perftest(
        bench,
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
    ap.add_argument("-t", "--tokens", type=int, action="append", default=None)
    ap.add_argument("--tp", type=int, action="append", default=None)
    ap.add_argument("--skip-generic", action="store_true")
    # --perf: profile flydsl_moe_scatter_copy_token with run_perftest and dump
    # the per-op breakdown (host-side hot loop) via AITER_LOG_MORE.
    ap.add_argument(
        "--perf",
        action="store_true",
        help="profile the scatter-copy hot loop instead of correctness",
    )
    ap.add_argument("--E", type=int, default=32, help="[--perf] experts")
    ap.add_argument("--topk", type=int, default=8, help="[--perf] topk")
    ap.add_argument("--model-dim", type=int, default=4096, help="[--perf] model dim")
    ap.add_argument(
        "--fp4", action="store_true", help="[--perf] a4w4 payload (Wp=dim/2)"
    )
    ap.add_argument("--iters", type=int, default=50, help="[--perf] run_perftest iters")
    args = ap.parse_args()

    if args.perf:
        _run_perf(args)
        return

    tokens = args.tokens if args.tokens else [1, 256]
    tps = args.tp if args.tp else [1, 8]
    dtype = torch.bfloat16  # route_weights dtype; payload/scale are uint8

    # (token_num, topk, E_total, ep, model_dim, fp4, name)
    configs = []

    # Generic sweep: small shapes, both aligned (dword) and unaligned (byte)
    # scale widths, a8w4 + a4w4 payloads, single rank.
    if not args.skip_generic:
        for tn in [1, 7, 128]:
            for E, topk in [(8, 1), (8, 2), (32, 8)]:
                # 512 -> Ws=16 (dword); 2880 -> Ws=90 (byte tail)
                for model_dim in [512, 2880]:
                    for fp4 in (False, True):
                        configs.append((tn, topk, E, 1, model_dim, fp4, "generic"))

    # Real-world shapes at TP1 and TP8, decode + prefill batch, a8w4 + a4w4.
    for name, hidden, E_total, topk in REAL_MODELS:
        for tp in tps:
            for tn in tokens:
                for fp4 in (False, True):
                    configs.append((tn, topk, E_total, tp, hidden, fp4, name))

    all_ok = True
    for i, (tn, topk, E_total, ep, md, fp4, name) in enumerate(configs):
        if topk > max(1, E_total // ep):
            print(f"[SKIP] {name} tp={ep}: topk={topk} > E_local={E_total // ep}")
            continue
        all_ok &= _run_one(tn, topk, E_total, ep, md, fp4, dtype, seed=i, name=name)

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
