#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Sweep BLOCK_N (KV tile rows) for the FlyDSL GLM/DSv3.2 sparse MLA prefill.

For each candidate BLOCK_N this:
  1. tries to compile the GLM kernel (block_n is a compile-time constant),
  2. gates infeasible tiles with the builder's NotImplementedError,
  3. correctness-checks vs the prod ``mla_decode_fwd`` (cosine),
  4. times the fused FlyDSL kernel (median ms).

On gfx942 the V2 software-transpose KV layout hardwires 4 rows/sub, so only
BLOCK_N=32 is feasible: BLOCK_N=16 needs a KvManagerV2 redesign and BLOCK_N=64
busts the 64 KB LDS budget at head_dim=576 (it needs gfx950 HW V-transpose).
This script makes that result reproducible.

Run:
    cd /home/AMD/samremes/dev/aiter
    python op_tests/flydsl_tests/sweep_glm_block_n.py --T 4096
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_AITER = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _AITER not in sys.path:
    sys.path.insert(0, _AITER)

from op_tests.flydsl_tests.bench_sparse_mla_prefill import (  # noqa: E402
    GLM_TOPK,
    build_glm_decode_state,
    build_glm_inputs,
    _cosine,
    _median_ms,
    _run_mla_decode_once,
)


def _bench_flydsl(glm, block_n, warmup, iters):
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    kv_scale = torch.tensor([glm.kv_scale], dtype=torch.float32, device=glm.q.device)

    def run_once():
        flydsl_sparse_mla_prefill(
            glm.q, glm.cache, glm.indices, glm.indptr, glm.out,
            block_table=glm.block_table, block_size=glm.block_size,
            packed=True, scale_mode="per_tensor", kv_scale=kv_scale, block_n=block_n,
        )
        return {}

    run_once()  # compile
    med, _ = _median_ms(run_once, warmup, iters)
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--topk", type=int, default=GLM_TOPK)
    ap.add_argument("--num-tokens", type=int, default=65536)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--block-n", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    glm = build_glm_inputs(args.T, args.topk, args.num_tokens, args.block_size, args.seed)

    # prod baseline + correctness reference
    prod_ms = None
    state = None
    try:
        state = build_glm_decode_state(glm, args.topk, q_scale=1.0)

        def prod_once():
            _run_mla_decode_once(glm, state)
            return {}

        prod_once()
        prod_ms, _ = _median_ms(prod_once, args.warmup, args.iters)
    except Exception as exc:  # noqa: BLE001
        print(f"prod baseline unavailable: {exc}")

    print(f"\nGLM block_n sweep  T={args.T} topk={args.topk} num_tokens={args.num_tokens}")
    if prod_ms is not None:
        print(f"prod aiter_mla_decode_fwd: {prod_ms:.3f} ms (q_quant+decode+reduce)")
    print(f"{'block_n':>8} {'status':>10} {'cosine':>9} {'flydsl_ms':>10} {'vs_prod':>8}")

    for bn in args.block_n:
        try:
            ms = _bench_flydsl(glm, bn, args.warmup, args.iters)
        except NotImplementedError as exc:
            print(f"{bn:>8} {'infeasible':>10} {'-':>9} {'-':>10} {'-':>8}   {str(exc)[:70]}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"{bn:>8} {'error':>10} {'-':>9} {'-':>10} {'-':>8}   {str(exc)[:70]}")
            continue
        cos = float("nan")
        if state is not None:
            fly_out = torch.empty_like(glm.out)
            from aiter.ops.flydsl import flydsl_sparse_mla_prefill

            flydsl_sparse_mla_prefill(
                glm.q, glm.cache, glm.indices, glm.indptr, fly_out,
                block_table=glm.block_table, block_size=glm.block_size,
                packed=True, scale_mode="per_tensor",
                kv_scale=torch.tensor([glm.kv_scale], dtype=torch.float32, device=glm.q.device),
                block_n=bn,
            )
            _run_mla_decode_once(glm, state)
            cos = _cosine(fly_out, state.out_asm)
        ratio = f"{prod_ms / ms:.2f}x" if prod_ms else "-"
        print(f"{bn:>8} {'ok':>10} {cos:>9.4f} {ms:>10.3f} {ratio:>8}")


if __name__ == "__main__":
    main()
