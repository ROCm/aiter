# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Focused B16/Nq2/H64/D128/KVB64 test for the gfx950 indexer mapping."""

import argparse

import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import (
    flydsl_fp8_paged_mqa_logits,
    flydsl_fp8_paged_mqa_logits_gfx950,
)
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
HEADS = 64
HEAD_DIM = 128
KV_LEN = 32768
KV_BLOCK_SIZE = 64


def _inputs():
    inp = _build_inputs(
        BATCH,
        NEXT_N,
        HEADS,
        HEAD_DIM,
        KV_LEN,
        get_fp8_e4m3_dtype(),
        block_size=KV_BLOCK_SIZE,
    )
    kv_cache, out = _kernel_inputs(
        inp, BATCH, NEXT_N, HEAD_DIM, True, KV_BLOCK_SIZE
    )
    return inp, kv_cache, out


def _launch(inp, kv_cache, out):
    return flydsl_fp8_paged_mqa_logits_gfx950(
        inp.q_fp8,
        kv_cache,
        inp.weights,
        out,
        inp.context_lens,
        inp.block_tables,
        inp.max_model_len,
        KVBlockSize=KV_BLOCK_SIZE,
    )


def test_gfx950_indexer_mapping():
    assert get_gfx() == "gfx950"
    inp, kv_cache, out = _inputs()
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
    assert torch.equal(got_mask, ref_mask), "causal/padding -inf mask mismatch"
    diff = calc_diff(got.masked_fill(got_mask, 0), ref.masked_fill(ref_mask, 0))
    assert diff < 1e-3, f"calc_diff={diff}"
    print(f"correctness: pass calc_diff={float(diff):.3e}")


def _benchmark():
    from aiter.ops.triton.attention.pa_mqa_logits import (
        deepgemm_fp8_paged_mqa_logits,
        enable_gluon_pa_mqa_logits,
        triton_version,
    )

    inp, kv_cache, out_new = _inputs()
    out_old = torch.full_like(out_new, float("-inf"))
    out_triton = torch.full_like(out_new, float("-inf"))

    def new_kernel():
        return _launch(inp, kv_cache, out_new)

    def old_kernel():
        return flydsl_fp8_paged_mqa_logits(
            inp.q_fp8,
            kv_cache,
            inp.weights,
            out_old,
            inp.context_lens,
            inp.block_tables,
            inp.max_model_len,
            Preshuffle=True,
            KVBlockSize=KV_BLOCK_SIZE,
        )

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
    print(f"triton {triton_version} backend={backend}")
    with torch.inference_mode():
        _, new_us = run_perftest(new_kernel, num_iters=50, num_warmup=8)
        _, old_us = run_perftest(old_kernel, num_iters=50, num_warmup=8)
        _, triton_us = run_perftest(triton_kernel, num_iters=50, num_warmup=8)
    print(f"time: {new_us:.3f} us")
    print(f"flydsl baseline: {old_us:.3f} us")
    print(f"{backend}: {triton_us:.3f} us")
    print(f"vs flydsl: {old_us / new_us:.3f}x")
    print(f"vs {backend}: {triton_us / new_us:.3f}x")


def _profile():
    inp, kv_cache, out = _inputs()
    with torch.inference_mode():
        _launch(inp, kv_cache, out)
    torch.cuda.synchronize()
    print("profile launch: pass")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()
    if args.profile:
        _profile()
    elif args.time:
        _benchmark()
    else:
        test_gfx950_indexer_mapping()


if __name__ == "__main__":
    main()
