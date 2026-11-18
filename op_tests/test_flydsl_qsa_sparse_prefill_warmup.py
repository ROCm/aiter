# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""qsa_sparse_prefill_warmup: after it, serving a paged cache compiles no kernel.

Every mode runs in a fresh process with an empty FlyDSL cache directory (AITER keeps its
compiled launchers in module dictionaries, FlyDSL on disk), on one paged K|V cache in
a production-like geometry: 1568-token pages 19,992 rows apart, one KV head, the K and V halves
of one 512-element record per token (the ``wide`` cache spans past 4 GiB, which the
direct kernels compile in). A call "compiles" when it adds a .pkl file to the cache
directory (FlyDSL writes one per compiled kernel) or a key to the kernels' ``_COMPILED``
tables (AITER's guard of every compile).

    warm       warmup, then a sweep of layouts: no sweep call compiles (the warmup did)
    cold       the sweep alone: its first call compiles (the counter works)

gfx950 only: the cache (K|V records, strided rows) is read in place there only,
and the layouts below are those of gfx950's 1536-row union gate; other architectures
skip.

Usage:
    python op_tests/test_flydsl_qsa_sparse_prefill_warmup.py [-g 12 6 3]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

PAGE, PITCH, DIM = 1568, 19992, 256
SCALE = DIM**-0.5
TOPK = 512
# Each layout class of the warmup has a call here (one full prefill takes the wide tile
# at 6 and 3 query heads per KV head).
SWEEP = (
    ((768,), (0,)),
    ((1024,), (0,)),
    ((3072,), (0,)),
    ((8192,), (0,)),
    ((8192,), (132096,)),
    ((4096,), (32768,)),
    ((1000, 3096), (0, 0)),
    ((1000, 2500, 700), (0, 16384, 3000)),
)


def selection(query_lens, prefix_lens, device):
    """The op's [rows, 4 TOPK + 3] indices (its docstring's layout), written here
    independently of the warmup's: row p of a request (v = prefix + p + 1 tokens) holds its
    last min(v // 4, TOPK) complete four-token blocks, then its v % 4 tail tokens, then -1.
    """
    rows = []
    for n, prefix in zip(query_lens, prefix_lens):
        v = prefix + torch.arange(1, n + 1, device=device)
        blocks = (v // 4).clamp(max=TOPK)
        column = torch.arange(4 * TOPK + 3, device=device)[None, :]
        body = column < 4 * blocks[:, None]
        tail = ~body & (column < 4 * blocks[:, None] + (v % 4)[:, None])
        token = torch.where(
            body,
            4 * (v // 4 - blocks)[:, None] + column,
            4 * (v // 4)[:, None] + column - 4 * blocks[:, None],
        )
        rows.append(torch.where(body | tail, token, -1))
    return torch.cat(rows).to(torch.int32)


def cache_files(root):
    return sum(f.endswith(".pkl") for _, _, files in os.walk(root) for f in files)


def child(mode, group, wide, cache_dir):
    """One mode in this process: print its record as JSON."""
    from aiter.ops import qsa
    from aiter.ops.flydsl.kernels.qsa import (
        sparse_prefill_direct,
        sparse_prefill_prepare,
        sparse_prefill_union,
    )

    device = torch.device("cuda")
    pages = 220 if wide else 100
    record = torch.zeros(pages, PITCH, 2, 1, DIM, dtype=torch.bfloat16, device=device)
    pool = qsa.qsa_paged_kv_pool(record[:, :PAGE, 0], record[:, :PAGE, 1])
    assert pool is not None, "the cache has no pool"
    span = ((pool.k.shape[0] - 1) * 2 * DIM + DIM) * 2
    assert (span > 2**32 - 2**20) == wide, f"pool span {span} is not {wide=}"
    heads = group
    kernels = (sparse_prefill_prepare, sparse_prefill_union, sparse_prefill_direct)
    generator = torch.Generator(device="cpu").manual_seed(7)

    def compiled():
        return sum(len(k._COMPILED) for k in kernels), cache_files(cache_dir)

    def table(query_lens, prefix_lens, random):
        need = [-(-(n + p) // PAGE) for n, p in zip(query_lens, prefix_lens)]
        if not random:
            return torch.zeros(len(need), max(need), dtype=torch.int32, device=device)
        chosen = torch.randperm(pages, generator=generator)[: sum(need)]
        result = torch.zeros(len(need), max(need), dtype=torch.int32)
        start = 0
        for r, count in enumerate(need):
            result[r, :count] = chosen[start : start + count]
            start += count
        return result.to(device)

    def call(q, indices, query_lens, prefix_lens, random):
        slots = qsa.qsa_paged_kv_slots(
            query_lens,
            prefix_lens,
            table(query_lens, prefix_lens, random),
            pool.page_size,
            pool.rows_per_page,
        )
        kwargs = {
            "query_lens": query_lens,
            "prefix_lens": prefix_lens,
            "softmax_scale": SCALE,
            "out": torch.empty_like(q),
            "kv_slots": slots,
        }
        if qsa.qsa_sparse_prefill_problem(q, pool.k, pool.v, indices, **kwargs):
            return False
        qsa.qsa_sparse_prefill(q, pool.k, pool.v, indices, **kwargs)
        return True

    result = {"mode": mode, "group": group, "wide": wide}
    if mode != "cold":
        warmed = qsa.qsa_sparse_prefill_warmup(
            lambda q, indices, lens, prefix: call(q, indices, lens, prefix, False),
            num_q_heads=heads,
            num_kv_heads=1,
            selection_width=4 * TOPK + 3,
            device=device,
        )
        result["warmed"] = warmed
        result["after_warmup"] = compiled()
    calls = []
    for query_lens, prefix_lens in SWEEP:
        q = torch.randn(
            sum(query_lens), heads, DIM, dtype=torch.bfloat16, device=device
        )
        indices = selection(query_lens, prefix_lens, device)
        before = compiled()
        assert call(q, indices, query_lens, prefix_lens, True), (
            query_lens,
            prefix_lens,
        )
        torch.cuda.synchronize()
        after = compiled()
        calls.append(
            {
                "layout": [query_lens, prefix_lens],
                "new_keys": after[0] - before[0],
                "new_pkl": after[1] - before[1],
            }
        )
    result["calls"] = calls
    print("RECORD " + json.dumps(result), flush=True)


def run_child(mode, group, wide):
    with tempfile.TemporaryDirectory(prefix="qsa-warmup-") as cache_dir:
        env = {**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": cache_dir, "TQDM_DISABLE": "1"}
        args = [sys.executable, __file__, "--child", mode, str(group), str(int(wide))]
        done = subprocess.run(
            args + [cache_dir], env=env, capture_output=True, text=True, check=False
        )
    lines = [l for l in done.stdout.splitlines() if l.startswith("RECORD ")]
    assert done.returncode == 0 and lines, f"{mode} g{group}: {done.stderr[-2000:]}"
    return json.loads(lines[-1][len("RECORD ") :])


def compiles(record):
    return {
        (tuple(c["layout"][0]), tuple(c["layout"][1])): (c["new_keys"], c["new_pkl"])
        for c in record["calls"]
    }


def check(group, wide):
    warm = run_child("warm", group, wide)
    layouts = 2 + (group in (6, 3))
    assert len(warm["warmed"]) == layouts, f"g{group}: warmed {warm['warmed']}"
    assert warm["after_warmup"][0] > 0 and warm["after_warmup"][1] > 0, warm
    busy = {k: v for k, v in compiles(warm).items() if v != (0, 0)}
    assert not busy, f"g{group} wide={wide}: served calls compiled after warmup: {busy}"
    print(
        f"g{group} wide={wide}: warmup {warm['after_warmup']}, 0 compiles in {len(SWEEP)} calls"
    )
    return warm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--group", type=int, nargs="*", default=[12, 6, 3])
    parser.add_argument("--child", nargs=4, metavar=("MODE", "GROUP", "WIDE", "DIR"))
    args = parser.parse_args()
    if args.child:
        mode, group, wide, cache_dir = args.child
        return child(mode, int(group), bool(int(wide)), cache_dir)
    arch = get_gfx_runtime()
    if arch != "gfx950":
        print(f"qsa_sparse_prefill_warmup test needs gfx950, got {arch}; skipping")
        return
    for group in args.group:
        check(group, False)
    if 6 in args.group:
        check(6, True)
        cold = compiles(run_child("cold", 6, False))
        assert cold[SWEEP[0]] != (
            0,
            0,
        ), f"the control's first call compiled nothing: {cold}"
        print(f"g6 control: first call compiled {cold[SWEEP[0]]}")


if __name__ == "__main__":
    main()
