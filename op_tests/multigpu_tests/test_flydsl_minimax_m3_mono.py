# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multi-GPU checks of the MiniMax-M3 fused sparse-layer decode op.

Four spawned ranks build ``MiniMaxM3MonoDecode`` over random layers of the real
shapes (MXFP4 experts as random bytes with sane E8M0 scales), each layer with
its own synthetic paged KV and index cache, in either cache layout:

* ``engine``: a block's K pages then its V pages (``block_pages`` 16), one fixed
  K / V scale (vLLM);
* ``planes``: separate K and V planes (``block_pages`` 8), per-token K / V
  scales (ATOM).

Determinism: the same decode step runs repeatedly and must be bit-identical

* across repeats, with a random rank's launch delayed each run: the cross-CTA
  and cross-GPU hand-offs are ordered whatever the ranks' skew;
* with the step's own cache slots (K, V, their scales, the index cache) refilled
  with alternating junk before each run: the kernel writes those slots before it
  reads them, so a consumer that reads one before its store lands changes the
  output;
* with the last row a CUDA-graph pad row (slot -1, seq_len 0, zero block-table
  row) whose input changes between runs: a pad row must not reach real rows.

Placement: the same step with its blocks moved to ids whose bytes lie just below
and past 2^31 and 2^32 in the KV and index caches gives the same output.

Guards: on every rank, before any launch, ``run`` raises when a cache was
reallocated without ``register_layers()`` (the old one still alive) and when CUDA
graph capture reaches an uncompiled step size; after re-registering it runs.

Runs on gfx950 with 256 compute units and at least four GPUs; skips otherwise.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import os
import random
import sys
import warnings
from multiprocessing import Pool, TimeoutError, freeze_support, set_start_method

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
LAYOUTS = ("engine", "planes")
BLOCK_PAGES = {"engine": 16, "planes": 8}
PAGE_BYTES = 16 * 128  # a page-16 of one fp8 KV head
INDEX_BLOCK = 128 * 128  # one block of fp8 index keys
DELAY_CYCLES = 200_000
CASE_TIMEOUT_S = 600


def _high_blocks(count, layout):
    """``count`` distinct block ids around the blocks whose bytes start at 2^31 and
    2^32 in the KV cache and in the index cache, nearest first."""
    kv_block = BLOCK_PAGES[layout] * PAGE_BYTES  # a block of the K (or K | V) tensor
    edges = [lim // b for lim in (1 << 31, 1 << 32) for b in (kv_block, INDEX_BLOCK)]
    ids = []
    for k in itertools.count():
        for e in edges:
            for b in (e - 1 - k, e + k):
                if b not in ids:
                    ids.append(b)
                if len(ids) == count:
                    return ids


def _build(rank, n_layers, tokens, device, layout, high=False, shared=False):
    """Random layers, their caches (a dict of uint8 / fp32 tensors a layer) and
    one decode step. ``high``: the step's blocks sit past 2^31 and 2^32 bytes of
    the caches. ``shared``: one cache set serves every layer."""
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
    n_used = tokens * per_req
    ids = _high_blocks(n_used, layout) if high else list(range(1, n_used + 1))
    n_blocks = max(ids) + 1
    block_table = torch.tensor(ids, dtype=torch.int32, device=device).view(
        tokens, per_req
    )
    used = block_table.view(-1).long()
    pages = SPARSE_BLOCK // 16
    cos_sin = rnd(1 << 16, ROTARY_DIM, scale=0.5)
    one = torch.ones(1, device=device)

    def filled(values):
        """Zeros, a row per block, with the used blocks' rows set to ``values``:
        the same values wherever the blocks are placed."""
        t = values.new_zeros(n_blocks, *values.shape[1:])
        t[used] = values
        return t

    def fp8_bytes(*shape):
        return filled(u8(n_used, *shape, lo=0x10, hi=0x50))

    def layer_caches():
        c = {"index": fp8_bytes(SPARSE_BLOCK, HEAD_DIM)}
        if layout == "engine":
            kv = fp8_bytes(2, pages, PAGE_BYTES)
            c["k"] = kv.view(-1, PAGE_BYTES)
            # V pages sit 8 pages into a block: one page id reaches K and V
            c["v"] = kv.view(-1)[pages * PAGE_BYTES :].view(-1, PAGE_BYTES)
            c["ks"] = c["vs"] = one
        else:
            for side in ("k", "v"):
                c[side] = fp8_bytes(pages, PAGE_BYTES).view(-1, PAGE_BYTES)
            for side in ("ks", "vs"):
                scales = torch.rand(n_used, pages * 16, generator=g, device=device)
                c[side] = filled(scales * 0.02 + 0.01).view(-1)
        return c

    if shared:
        caches = [layer_caches()] * n_layers
    else:
        caches = [layer_caches() for _ in range(n_layers)]
    layers = []
    for c in caches:
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
                k_cache=c["k"].view(fp8),
                v_cache=c["v"].view(fp8),
                k_scale=c["ks"],
                v_scale=c["vs"],
                index_cache=c["index"].view(fp8),
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
    return layers, caches, step, rnd


def _junk_own_slots(caches, slots, layout, value):
    """Fill the K, V, K / V scale and index-cache entries of ``slots``."""
    from aiter.ops.flydsl.kernels.minimax_m3_mono.config import HEAD_DIM

    pages = BLOCK_PAGES[layout]
    for c in caches:
        dims = torch.arange(HEAD_DIM, device=c["k"].device)
        for s in slots.tolist():
            if s < 0:
                continue
            page = (s // 128) * pages + s % 128 // 16
            t = s % 16
            c["k"][page, (dims // 16) * 256 + t * 16 + dims % 16] = value
            c["v"][page, dims * 16 + t] = value
            if layout == "planes":
                c["ks"][page * 16 + t] = float(value + 1)
                c["vs"][page * 16 + t] = float(value + 1)
            c["index"].view(-1, HEAD_DIM)[s] = value


def _init_rank(rank, init_method):
    import torch.distributed as dist

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo", init_method=init_method, world_size=TP, rank=rank
    )
    return device


def _make_op(rank, device, layout):
    import torch.distributed as dist

    from aiter.ops.flydsl.kernels.minimax_m3_mono.config import HEAD_DIM
    from aiter.ops.flydsl.minimax_m3_mono import MiniMaxM3MonoDecode

    return MiniMaxM3MonoDecode(
        dist.group.WORLD, rank, TP, device,
        sm_scale=HEAD_DIM**-0.5, eps=1e-6, route_scale=2.0, shared_weight=1.0,
        swiglu_limit=7.0, init_blocks=0, local_blocks=1,
        scalar_kv_scale=layout == "engine", gate_fp32=True,
    )  # fmt: skip


def _register(op, layers, layout):
    op.register_layers(
        [(3 + i, w) for i, w in enumerate(layers)], block_pages=BLOCK_PAGES[layout]
    )


def _run(op, step, layers, hidden=None, residual=None):
    return op.run(
        step["hidden"] if hidden is None else hidden,
        step["residual"] if residual is None else residual,
        step["positions"],
        step["slot_mapping"],
        step["block_table"],
        step["seq_lens"],
        [w.caches for w in layers],
    )


def _determinism_rank(rank, init_method, layout, n_layers, tokens, iters, pad):
    import torch.distributed as dist

    from aiter.ops.flydsl.kernels.minimax_m3_mono.config import HIDDEN

    device = _init_rank(rank, init_method)
    layers, caches, step, rnd = _build(rank, n_layers, tokens, device, layout)
    if pad:
        step["slot_mapping"][-1] = -1
        step["seq_lens"][-1] = 0
        step["block_table"][-1].zero_()
    op = _make_op(rank, device, layout)
    _register(op, layers, layout)
    tensors = list({id(t): t for c in caches for t in c.values()}.values())
    initial = [t.clone() for t in tensors]
    live = tokens - 1 if pad else tokens
    delayed = random.Random(2 * tokens + pad)
    failures = []
    ref = None
    try:
        for it in range(iters):
            for t, t0 in zip(tensors, initial):
                t.copy_(t0)
            _junk_own_slots(caches, step["slot_mapping"], layout, 0x7F * (it % 2))
            hidden = step["hidden"].clone()
            residual = step["residual"].clone()
            if pad:
                hidden[-1] = rnd(HIDDEN, scale=1.0) * (it % 2)
                residual[-1] = rnd(HIDDEN, scale=1.0) * (it % 2)
            torch.cuda.synchronize()
            dist.barrier()
            if delayed.randrange(TP) == rank:
                torch.cuda._sleep(DELAY_CYCLES)
            h, r, _ = _run(op, step, layers, hidden, residual)
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


def _expect_raise(failures, what, exc, match, fn):
    try:
        fn()
        failures.append(f"{what} did not raise")
    except exc as e:
        if match not in str(e):
            raise


def _placement_rank(rank, init_method, layout, n_layers, tokens):
    import torch.distributed as dist

    device = _init_rank(rank, init_method)
    outs = []
    try:
        for high in (False, True):
            layers, caches, step, _ = _build(
                rank, n_layers, tokens, device, layout, high, shared=True
            )
            op = _make_op(rank, device, layout)
            try:
                _register(op, layers, layout)
                h, r, _ = _run(op, step, layers)
                outs.append(torch.cat([h, r], -1).float().cpu())
            finally:
                op.close()
            del layers, caches, step
            torch.cuda.empty_cache()
    finally:
        dist.destroy_process_group()
    if not torch.isfinite(outs[0]).all():
        return rank, ["non-finite output"]
    if not torch.equal(outs[0], outs[1]):
        rows = ((outs[0] - outs[1]).abs().amax(-1) > 0).nonzero().flatten().tolist()
        return rank, [f"rows {rows} differ at high block ids"]
    return rank, []


def _guards_rank(rank, init_method):
    import torch.distributed as dist

    device = _init_rank(rank, init_method)
    layers, _, step, _ = _build(rank, 2, 2, device, "engine")
    op = _make_op(rank, device, "engine")
    failures = []
    try:
        _register(op, layers, "engine")
        _run(op, step, layers)
        torch.cuda.synchronize()

        one = {name: t[:1] for name, t in step.items()}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        def capture_uncompiled():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "The CUDA Graph is empty")
                with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
                    _run(op, one, layers)

        _expect_raise(
            failures, "capture of an uncompiled step size", RuntimeError,
            "not compiled", capture_uncompiled,
        )  # fmt: skip
        torch.cuda.synchronize()

        # The engine reallocates a cache and misses register_layers(); the old
        # cache stays alive, so only the pointer comparison can catch it.
        realloc = [
            dataclasses.replace(w, index_cache=w.index_cache.clone()) for w in layers
        ]
        _expect_raise(
            failures, "run after a missed register_layers()", RuntimeError,
            "not the registered tensor", lambda: _run(op, step, realloc),
        )  # fmt: skip
        _register(op, realloc, "engine")
        _run(op, step, realloc)
        torch.cuda.synchronize()
        dist.barrier()
    finally:
        op.close()
        dist.destroy_process_group()
    return rank, failures


def _spawn(fn, *args):
    """``fn`` on every rank; a rank that hangs (a deadlocked launch) fails the
    case after CASE_TIMEOUT_S instead of stalling the run."""
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(TP) as pool:
        job = pool.starmap_async(fn, [(r, init_method, *args) for r in range(TP)])
        try:
            return job.get(CASE_TIMEOUT_S)
        except TimeoutError:
            pool.terminate()
            return [(r, [f"no result within {CASE_TIMEOUT_S} s"]) for r in range(TP)]


def _report(tag, results):
    bad = [(rank, f) for rank, f in results if f]
    for rank, f in bad:
        print(f"FAIL {tag} rank {rank}: {len(f)} failures, first {f[0]}")
    if not bad:
        print(f"PASS {tag}")
    return not bad


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layers", type=int, default=57)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=LAYOUTS)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["determinism", "placement", "guards"],
        default=["determinism", "placement", "guards"],
    )
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
    if "determinism" in args.cases:
        for layout in args.layouts:
            for tokens in args.tokens:
                for pad in (False, True) if tokens > 1 else (False,):
                    ok &= _report(
                        f"determinism layout={layout} layers={args.layers} "
                        f"tokens={tokens} iters={args.iters} pad={pad}",
                        _spawn(
                            _determinism_rank, layout, args.layers, tokens,
                            args.iters, pad,
                        ),
                    )  # fmt: skip
    if "placement" in args.cases:
        for layout in args.layouts:
            for tokens in args.tokens:
                ok &= _report(
                    f"placement layout={layout} tokens={tokens}",
                    _spawn(_placement_rank, layout, 2, tokens),
                )
    if "guards" in args.cases:
        ok &= _report("guards", _spawn(_guards_rank))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    freeze_support()
    main()
