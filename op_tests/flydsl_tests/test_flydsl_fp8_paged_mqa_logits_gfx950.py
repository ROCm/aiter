# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Focused long-context tests for the gfx950 padded ragged LDS indexer."""

import argparse

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_fp8_paged_mqa_logits
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from aiter.test_common import run_perftest
from op_tests.flydsl_tests.ragged_nn import (
    MAX_NN,
    ref_padded_ragged,
    sample_next_n_lens,
)
from op_tests.flydsl_tests.test_flydsl_fp8_paged_mqa_logits import (
    _build_inputs,
    _kernel_inputs,
    calc_diff,
)

torch.set_default_device("cuda")

BATCH = 16
HEADS = 32
HEAD_DIM = 128
KV_LEN = 32768
KV_BLOCK_SIZE = 64
WIDE_MAX_MODEL_LEN = 1 << 20


def _inputs(kv_len=KV_LEN, batch=BATCH, seed=1079):
    torch.manual_seed(seed)
    inp = _build_inputs(
        batch,
        MAX_NN,
        HEADS,
        HEAD_DIM,
        kv_len,
        get_fp8_e4m3_dtype(),
        block_size=KV_BLOCK_SIZE,
    )
    kv_cache, out = _kernel_inputs(inp, batch, MAX_NN, HEAD_DIM, True, KV_BLOCK_SIZE)
    next_n_lens = sample_next_n_lens(batch, MAX_NN, seed=seed).cuda()
    return inp, kv_cache, out, next_n_lens


def _launch(inp, kv_cache, out, next_n_lens, *, max_model_len=None, split_kv=None):
    return flydsl_fp8_paged_mqa_logits(
        inp.q_fp8,
        kv_cache,
        inp.weights,
        out,
        inp.context_lens,
        inp.block_tables,
        inp.max_model_len if max_model_len is None else max_model_len,
        next_n_lens=next_n_lens,
        Preshuffle=True,
        KVBlockSize=KV_BLOCK_SIZE,
        SplitKV=split_kv,
    )


def _check(inp, kv_cache, out, next_n_lens, tag):
    with torch.inference_mode():
        ref = ref_padded_ragged(
            inp.q,
            inp.kv_cache_fp8,
            inp.weights,
            inp.context_lens,
            inp.block_tables,
            next_n_lens,
            inp.max_model_len,
            inp.fp8_dtype,
            max_nn=MAX_NN,
            block_size=KV_BLOCK_SIZE,
        )
        got = _launch(inp, kv_cache, out, next_n_lens)

    ref_mask = ref == float("-inf")
    got_mask = got == float("-inf")
    assert torch.equal(got_mask, ref_mask), f"{tag}: causal/padding -inf mask mismatch"
    diff = calc_diff(got.masked_fill(got_mask, 0), ref.masked_fill(ref_mask, 0))
    assert diff < 1e-3, f"{tag} calc_diff={diff}"


@pytest.mark.skipif(get_gfx() != "gfx950", reason="requires gfx950")
def test_gfx950_ragged_long_context():
    inp, kv_cache, out, next_n_lens = _inputs()
    _check(inp, kv_cache, out, next_n_lens, "ragged long context")


@pytest.mark.skipif(get_gfx() != "gfx950", reason="requires gfx950")
@pytest.mark.parametrize("pages", [2, 3, 4, 7])
def test_gfx950_ragged_short_pages(pages):
    inp, kv_cache, out, next_n_lens = _inputs(
        kv_len=pages * KV_BLOCK_SIZE, batch=2, seed=pages
    )
    _check(inp, kv_cache, out, next_n_lens, f"pages={pages}")


@pytest.mark.skipif(get_gfx() != "gfx950", reason="requires gfx950")
def test_gfx950_wide_output_no_tail_drop():
    """Padded output row 512 crosses the signed i32 byte-offset boundary."""
    batch = 65
    inp, kv_cache, compact, _ = _inputs(kv_len=KV_BLOCK_SIZE, batch=batch, seed=5)
    next_n_lens = torch.ones(batch, dtype=torch.int32)
    wide = torch.full(
        (batch * MAX_NN, WIDE_MAX_MODEL_LEN),
        float("-inf"),
        dtype=torch.float32,
    )

    with torch.inference_mode():
        _launch(inp, kv_cache, compact, next_n_lens, split_kv=1)
        _launch(
            inp,
            kv_cache,
            wide,
            next_n_lens,
            max_model_len=WIDE_MAX_MODEL_LEN,
            split_kv=1,
        )
    torch.cuda.synchronize()

    valid_cols = KV_BLOCK_SIZE
    assert torch.equal(wide[:, :valid_cols], compact[:, :valid_cols])
    assert torch.equal(wide[512:, :valid_cols], compact[512:, :valid_cols])


def _benchmark():
    inp, kv_cache, out, next_n_lens = _inputs()

    def kernel():
        return _launch(inp, kv_cache, out, next_n_lens)

    with torch.inference_mode():
        _, us = run_perftest(kernel, num_iters=50, num_warmup=8)
    print(f"time: {us:.3f} us")


def _profile():
    inp, kv_cache, out, next_n_lens = _inputs()
    with torch.inference_mode():
        _launch(inp, kv_cache, out, next_n_lens)
    torch.cuda.synchronize()
    print("profile launch: pass")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()
    if args.time:
        _benchmark()
    elif args.profile:
        _profile()
    else:
        test_gfx950_ragged_long_context()
        for pages in (2, 3, 4, 7):
            test_gfx950_ragged_short_pages(pages)
        test_gfx950_wide_output_no_tail_drop()


if __name__ == "__main__":
    main()
