# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Focused B16/Nq2/H{32,64}/D128/KVB64 test for the gfx950 indexer mapping."""

import argparse

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_fp8_paged_mqa_logits
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from aiter.test_common import run_perftest
from op_tests.flydsl_tests.test_flydsl_fp8_paged_mqa_logits import (
    _build_inputs,
    _kernel_inputs,
    calc_diff,
    ref_fp8_paged_mqa_logits,
)

torch.set_default_device("cuda")

BATCH = 16
NEXT_N = 2
HEADS = (32, 64)
DEFAULT_HEADS = 64
HEAD_DIM = 128
KV_LEN = 32768
KV_BLOCK_SIZE = 64
WIDE_MAX_MODEL_LEN = 1 << 20


def _inputs(heads, next_n=NEXT_N, kv_len=KV_LEN, batch=BATCH):
    inp = _build_inputs(
        batch,
        next_n,
        heads,
        HEAD_DIM,
        kv_len,
        get_fp8_e4m3_dtype(),
        block_size=KV_BLOCK_SIZE,
    )
    kv_cache, out = _kernel_inputs(inp, batch, next_n, HEAD_DIM, True, KV_BLOCK_SIZE)
    return inp, kv_cache, out


def _launch(inp, kv_cache, out):
    return flydsl_fp8_paged_mqa_logits(
        inp.q_fp8,
        kv_cache,
        inp.weights,
        out,
        inp.context_lens,
        inp.block_tables,
        inp.max_model_len,
        Preshuffle=True,
        KVBlockSize=KV_BLOCK_SIZE,
    )


def _check(inp, kv_cache, out, tag):
    with torch.inference_mode():
        ref = ref_fp8_paged_mqa_logits(
            inp.q,
            inp.kv_cache_fp8,
            inp.weights,
            inp.context_lens,
            inp.block_tables,
            inp.max_model_len,
            inp.fp8_dtype,
            block_size=KV_BLOCK_SIZE,
        )
        got = _launch(inp, kv_cache, out)

    ref_mask = ref == float("-inf")
    got_mask = got == float("-inf")
    assert torch.equal(got_mask, ref_mask), f"{tag}: causal/padding -inf mask mismatch"
    diff = calc_diff(got.masked_fill(got_mask, 0), ref.masked_fill(ref_mask, 0))
    assert diff < 1e-3, f"{tag} calc_diff={diff}"
    print(f"correctness {tag}: pass calc_diff={float(diff):.3e}")


def test_gfx950_indexer_mapping(heads=DEFAULT_HEADS):
    assert get_gfx() == "gfx950"
    inp, kv_cache, out = _inputs(heads)
    _check(inp, kv_cache, out, f"H={heads}")


def test_gfx950_nq1_and_short_pages(heads=32):
    assert get_gfx() == "gfx950"
    inp, kv_cache, out = _inputs(heads, next_n=1)
    _check(inp, kv_cache, out, f"H={heads} Nq=1")
    for pages in (2, 3, 4):
        inp, kv_cache, out = _inputs(heads, kv_len=pages * KV_BLOCK_SIZE, batch=2)
        _check(inp, kv_cache, out, f"H={heads} pages={pages}")


@pytest.mark.skipif(get_gfx() != "gfx950", reason="requires gfx950")
def test_gfx950_wide_output_no_tail_drop():
    """Row 512 crosses the signed i32 byte-offset boundary at this stride."""
    batch = 513
    inp, kv_cache, _ = _inputs(32, next_n=1, kv_len=KV_BLOCK_SIZE, batch=batch)
    compact = torch.full(
        (batch, inp.max_model_len), float("-inf"), dtype=torch.float32
    )
    wide = torch.full(
        (batch, WIDE_MAX_MODEL_LEN), float("-inf"), dtype=torch.float32
    )

    with torch.inference_mode():
        flydsl_fp8_paged_mqa_logits(
            inp.q_fp8,
            kv_cache,
            inp.weights,
            compact,
            inp.context_lens,
            inp.block_tables,
            inp.max_model_len,
            Preshuffle=True,
            KVBlockSize=KV_BLOCK_SIZE,
        )
        flydsl_fp8_paged_mqa_logits(
            inp.q_fp8,
            kv_cache,
            inp.weights,
            wide,
            inp.context_lens,
            inp.block_tables,
            WIDE_MAX_MODEL_LEN,
            Preshuffle=True,
            KVBlockSize=KV_BLOCK_SIZE,
        )
    torch.cuda.synchronize()

    valid_cols = KV_BLOCK_SIZE
    assert torch.equal(wide[:, :valid_cols], compact[:, :valid_cols])
    assert torch.equal(wide[512:, :valid_cols], compact[512:, :valid_cols])


def _benchmark(heads):
    from aiter.ops.triton.attention.pa_mqa_logits import (
        deepgemm_fp8_paged_mqa_logits,
        enable_gluon_pa_mqa_logits,
        triton_version,
    )

    inp, kv_cache, out_new = _inputs(heads)
    out_triton = torch.full_like(out_new, float("-inf"))

    def new_kernel():
        return _launch(inp, kv_cache, out_new)

    def triton_kernel():
        return deepgemm_fp8_paged_mqa_logits(
            inp.q_fp8,
            kv_cache,
            inp.weights,
            out_triton,
            inp.context_lens,
            inp.block_tables,
            inp.max_model_len,
            ChunkK=256,
            Preshuffle=True,
            KVBlockSize=KV_BLOCK_SIZE,
            WavePerEU=2,
        )

    backend = "gluon" if enable_gluon_pa_mqa_logits else "triton-jit"
    print(f"H={heads} triton {triton_version} backend={backend}")
    with torch.inference_mode():
        _, new_us = run_perftest(new_kernel, num_iters=50, num_warmup=8)
        _, triton_us = run_perftest(triton_kernel, num_iters=50, num_warmup=8)
    # WaveScope benchmarkPattern matches the first `time: {us} us` line (H=64).
    if heads == DEFAULT_HEADS:
        print(f"time: {new_us:.3f} us")
    else:
        print(f"time H={heads}: {new_us:.3f} us")
    print(f"{backend}: {triton_us:.3f} us")
    print(f"vs {backend}: {triton_us / new_us:.3f}x")


def _profile(heads):
    inp, kv_cache, out = _inputs(heads)
    with torch.inference_mode():
        _launch(inp, kv_cache, out)
    torch.cuda.synchronize()
    print(f"profile launch H={heads}: pass")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--time", action="store_true")
    parser.add_argument(
        "--heads",
        type=int,
        choices=HEADS,
        default=None,
        help="Restrict to one head count (default: both; --time still prints H=64 first)",
    )
    args = parser.parse_args()
    heads = (args.heads,) if args.heads is not None else HEADS
    if args.time:
        # H=64 first so WaveScope still sees `time: {us} us` as the production line.
        timed = tuple(h for h in (DEFAULT_HEADS,) + heads if h in heads)
        seen = set()
        ordered = []
        for h in timed:
            if h not in seen:
                ordered.append(h)
                seen.add(h)
        for h in ordered:
            _benchmark(h)
    elif args.profile:
        for h in heads:
            _profile(h)
    else:
        for h in heads:
            test_gfx950_indexer_mapping(h)
            test_gfx950_nq1_and_short_pages(h)


if __name__ == "__main__":
    main()
