# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multi-GPU determinism of the MiniMax-M3 fused sparse-layer decode op.

Four spawned ranks build ``MiniMaxM3MonoDecode`` over random layers of the real
shapes (MXFP4 experts as random bytes with sane E8M0 scales) and a synthetic
paged KV / index cache in the engine layout (``block_pages`` 16), then run the
same decode step repeatedly. Every run must be bit-identical:

* across repeats: the layers' cross-CTA and cross-GPU hand-offs are ordered;
* with the step's own cache slots refilled with alternating junk before each
  run: the kernel writes those slots before it reads them, so a run that reads
  them back stale changes the output;
* with the last row a CUDA-graph pad row (slot -1, seq_len 0, zero block-table
  row) whose input changes between runs: a pad row must not reach real rows.

Runs on gfx950 with 256 compute units and at least four GPUs; skips otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool, freeze_support, set_start_method

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx_runtime

set_start_method("spawn", force=True)

TP = 4
SEQ_LENS = (1800, 1500, 1600, 1300, 1700, 1400, 1900, 1200,
            2100, 1650, 1550, 1750, 1350, 1450, 1850, 2000)  # fmt: skip


def _build(rank, n_layers, tokens, device):
    from aiter.ops.flydsl.kernels.minimax_m3_mono.config import (
        HEAD_DIM,
        HIDDEN,
        INTER,
        N_ROUTED,
        O_K,
        ONE_INDEX_HEAD,
        ROTARY_DIM,
        SPARSE_BLOCK,
    )
    from aiter.ops.flydsl.minimax_m3_mono import MonoLayerWeights

    g = torch.Generator(device=device).manual_seed(1234 + rank)
    fp8 = torch.float8_e4m3fn
    experts = N_ROUTED + 1

    def rnd(*shape, scale=0.02):
        return (torch.randn(*shape, generator=g, device=device) * scale).to(
            torch.bfloat16
        )

    def fp8w(n, k):
        return (torch.randn(n, k, generator=g, device=device) * 0.5).to(fp8)

    def u8(*shape, lo=0, hi=256):
        return torch.randint(
            lo, hi, shape, generator=g, device=device, dtype=torch.uint8
        )

    seqs = list(SEQ_LENS[:tokens])
    per_req = max(seqs) // SPARSE_BLOCK + 2
    n_blocks = tokens * per_req + 1
    block_table = torch.zeros(tokens, per_req, dtype=torch.int32, device=device)
    for i in range(tokens):
        block_table[i] = torch.arange(i * per_req + 1, (i + 1) * per_req + 1)
    # per block: 8 K pages then 8 V pages of 16 tokens (2 KB each)
    kv = u8(n_blocks + 1, 2, SPARSE_BLOCK, HEAD_DIM, lo=0x10, hi=0x50)
    k_cache = kv.view(-1, 1, 2048).view(fp8)
    v_cache = (
        torch.cat(
            [kv.view(-1)[16384:], torch.zeros(16384, dtype=torch.uint8, device=device)]
        )
        .view(-1, 1, 2048)
        .view(fp8)
    )
    cos_sin = rnd(1 << 16, ROTARY_DIM, scale=0.5)
    one = torch.ones(1, device=device)
    layers = []
    for _ in range(n_layers):
        layers.append(
            MonoLayerWeights(
                input_norm=rnd(HIDDEN, scale=0.1),
                w_qkv=fp8w(ONE_INDEX_HEAD.rows, HIDDEN),
                s_qkv=torch.rand(ONE_INDEX_HEAD.rows, 1, generator=g, device=device)
                * 0.01,
                q_norm=rnd(HEAD_DIM, scale=0.1),
                k_norm=rnd(HEAD_DIM, scale=0.1),
                index_q_norm=rnd(HEAD_DIM, scale=0.1),
                index_k_norm=rnd(HEAD_DIM, scale=0.1),
                cos_sin=cos_sin,
                w_o=fp8w(HIDDEN, O_K),
                s_o=torch.rand(HIDDEN, 1, generator=g, device=device) * 0.01,
                post_norm=rnd(HIDDEN, scale=0.1),
                gate=torch.randn(N_ROUTED, HIDDEN, generator=g, device=device) * 0.02,
                gate_bias=torch.randn(N_ROUTED, generator=g, device=device) * 0.01,
                w13=u8(experts, 2 * INTER, HIDDEN // 2),
                s13=u8(experts, 2 * INTER, HIDDEN // 32, lo=118, hi=124),
                w2=u8(experts, HIDDEN, INTER // 2),
                s2=u8(experts, HIDDEN, INTER // 32, lo=118, hi=124),
                k_cache=k_cache,
                v_cache=v_cache,
                k_scale=one,
                v_scale=one,
                index_cache=u8(
                    n_blocks + 1, SPARSE_BLOCK, HEAD_DIM, lo=0x10, hi=0x50
                ).view(fp8),
            )
        )
    seq_lens = torch.tensor(seqs, dtype=torch.int32, device=device)
    slots = torch.tensor(
        [
            int(block_table[i, (s - 1) // SPARSE_BLOCK]) * SPARSE_BLOCK
            + (s - 1) % SPARSE_BLOCK
            for i, s in enumerate(seqs)
        ],
        dtype=torch.int64,
        device=device,
    )
    step = {
        "hidden": rnd(tokens, HIDDEN, scale=1.0),
        "residual": rnd(tokens, HIDDEN, scale=1.0),
        "positions": (seq_lens - 1).to(torch.int64),
        "slot_mapping": slots,
        "block_table": block_table,
        "seq_lens": seq_lens,
    }
    return layers, kv, step, rnd


def _junk_own_slots(kv, layers, slots, value):
    """Fill the K, V and index-cache entries of ``slots`` with ``value``."""
    from aiter.ops.flydsl.kernels.minimax_m3_mono.config import (
        HEAD_DIM,
        SPARSE_BLOCK,
    )

    flat = kv.view(-1)
    dims = torch.arange(HEAD_DIM, device=kv.device)
    for s in slots.tolist():
        if s < 0:
            continue
        blk, off = divmod(s, SPARSE_BLOCK)
        page, t = divmod(off, 16)
        k_page = ((blk * 2) * 8 + page) * 2048
        v_page = ((blk * 2 + 1) * 8 + page) * 2048
        flat[k_page + (dims // 16) * 256 + t * 16 + dims % 16] = value
        flat[v_page + dims * 16 + t] = value
        for w in layers:
            w.index_cache.view(torch.uint8).view(-1, HEAD_DIM)[s] = value


def _run_rank(rank, init_method, n_layers, tokens, iters, pad):
    import torch.distributed as dist

    from aiter.ops.flydsl.kernels.minimax_m3_mono.config import HEAD_DIM, HIDDEN
    from aiter.ops.flydsl.minimax_m3_mono import MiniMaxM3MonoDecode

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo", init_method=init_method, world_size=TP, rank=rank
    )
    layers, kv, step, rnd = _build(rank, n_layers, tokens, device)
    if pad:
        step["slot_mapping"][-1] = -1
        step["seq_lens"][-1] = 0
        step["block_table"][-1].zero_()
    op = MiniMaxM3MonoDecode(
        dist.group.WORLD, rank, TP, device,
        sm_scale=HEAD_DIM**-0.5, eps=1e-6, route_scale=2.0, shared_weight=1.0,
        swiglu_limit=7.0, init_blocks=0, local_blocks=1,
        scalar_kv_scale=True, gate_fp32=True,
    )  # fmt: skip
    op.register_layers([(3 + i, w) for i, w in enumerate(layers)], block_pages=16)
    kv0 = kv.clone()
    ic0 = [w.index_cache.view(torch.uint8).clone() for w in layers]
    live = tokens - 1 if pad else tokens
    failures = []
    ref = None
    try:
        for it in range(iters):
            kv.copy_(kv0)
            for w, c in zip(layers, ic0):
                w.index_cache.view(torch.uint8).copy_(c)
            _junk_own_slots(kv, layers, step["slot_mapping"], 0x7F * (it % 2))
            hidden = step["hidden"].clone()
            residual = step["residual"].clone()
            if pad:
                hidden[-1] = rnd(HIDDEN, scale=1.0) * (it % 2)
                residual[-1] = rnd(HIDDEN, scale=1.0) * (it % 2)
            h, r, _ = op.run(
                hidden,
                residual,
                step["positions"],
                step["slot_mapping"],
                step["block_table"],
                step["seq_lens"],
            )
            out = torch.cat([h, r], -1)[:live].float()
            torch.cuda.synchronize()
            if not torch.isfinite(out).all():
                failures.append((it, "non-finite output"))
            elif ref is None:
                ref = out.clone()
            elif not torch.equal(out, ref):
                rows = ((out - ref).abs().amax(-1) > 0).nonzero().flatten().tolist()
                failures.append((it, f"rows {rows} differ"))
    finally:
        op.close()
        dist.destroy_process_group()
    return rank, failures


def run(n_layers, tokens, iters, pad):
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(TP) as pool:
        results = pool.starmap(
            _run_rank,
            [(r, init_method, n_layers, tokens, iters, pad) for r in range(TP)],
        )
    bad = [(rank, f) for rank, f in results if f]
    tag = f"layers={n_layers} tokens={tokens} iters={iters} pad={pad}"
    if bad:
        for rank, f in bad:
            print(f"FAIL {tag} rank {rank}: {len(f)} runs differ, first {f[0]}")
        return False
    print(f"PASS {tag}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    try:
        arch = get_gfx_runtime()
    except (KeyError, RuntimeError):
        arch = None
    if arch != "gfx950" or torch.cuda.device_count() < TP:
        print(
            f"SKIP: needs {TP} gfx950 GPUs (arch={arch}, gpus={torch.cuda.device_count()})"
        )
        return
    if torch.cuda.get_device_properties(0).multi_processor_count != 256:
        print("SKIP: needs 256 compute units per GPU")
        return
    ok = True
    for tokens in args.tokens:
        ok &= run(args.layers, tokens, args.iters, pad=False)
        if tokens > 1:
            ok &= run(args.layers, tokens, args.iters, pad=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    freeze_support()
    main()
