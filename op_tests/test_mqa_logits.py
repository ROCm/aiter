# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``aiter.mqa_logits`` (tuned dispatch) vs each backend vs torch.

Inputs follow the DSA indexer prefill call: Q ``[seq_len, 32, 128]`` FP8,
contiguous KV ``[seq_len_kv, 128]`` FP8 with per-key FP32 scales, and causal
windows for one sequence whose last ``seq_len`` tokens are the queries.
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys

import pandas as pd
import torch

import aiter
import aiter.mqa_logits as mql
from aiter import dtypes
from aiter.jit.utils.chip_info import get_cu_num, get_gfx, get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
HEADS = 32
HEAD_DIM = 128
_SHIPPED_ROW_CHILD = "AITER_TEST_MQA_LOGITS_SHIPPED_ROW"


def build_inputs(seq_len, seq_len_kv, heads=HEADS, head_dim=HEAD_DIM, seed=0):
    torch.manual_seed(seed)
    q = torch.randn((seq_len, heads, head_dim), dtype=torch.bfloat16)
    kv = torch.randn((seq_len_kv, head_dim), dtype=torch.bfloat16)
    fp8_max = torch.finfo(dtypes.fp8).max
    scale = kv.abs().float().amax(dim=-1).clamp(1e-4) / fp8_max
    kv_fp8 = (kv.float() / scale[:, None]).to(dtypes.fp8)
    q_fp8 = q.to(dtypes.fp8)
    rows = torch.arange(seq_len, dtype=torch.int32)
    return {
        "Q": q_fp8,
        "KV": kv_fp8,
        "kv_scales": scale.contiguous(),
        "weights": torch.randn((seq_len, heads), dtype=torch.float32),
        "cu_starts": torch.zeros((seq_len,), dtype=torch.int32),
        "cu_ends": rows + (seq_len_kv - seq_len + 1),
        "q_ref": q_fp8.float(),
        "kv_ref": kv_fp8.float() * scale[:, None],
    }


def run_torch(inp):
    """``[seq_len, seq_len_kv]`` logits, -inf outside each row's window."""
    q, kv, w = inp["q_ref"], inp["kv_ref"], inp["weights"]
    seq_len, heads, _ = q.shape
    seq_len_kv = kv.shape[0]
    rows_per_step = max(1, 2**28 // (heads * seq_len_kv))
    cols = torch.arange(seq_len_kv)
    out = torch.empty((seq_len, seq_len_kv), dtype=torch.float32)
    for s in range(0, seq_len, rows_per_step):
        e = min(seq_len, s + rows_per_step)
        score = torch.einsum("mhd,nd->mhn", q[s:e], kv).relu()
        logits = (score * w[s:e, :, None]).sum(dim=1)
        inside = (cols[None] >= inp["cu_starts"][s:e, None]) & (
            cols[None] < inp["cu_ends"][s:e, None]
        )
        out[s:e] = logits.masked_fill(~inside, float("-inf"))
    return out


def calc_diff(x, y):
    x, y = x.double(), y.double()
    return float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())


def launch_args(inp):
    return (
        inp["Q"],
        inp["KV"],
        inp["kv_scales"],
        inp["weights"],
        inp["cu_starts"],
        inp["cu_ends"],
    )


@benchmark()
def test_mqa_logits(seq_len, seq_len_kv):
    inp = build_inputs(seq_len, seq_len_kv)
    ref = run_torch(inp)
    args = launch_args(inp)
    picked = mql.get_mqa_logits_config(seq_len, seq_len_kv, HEADS, HEAD_DIM)
    candidates = {
        "mqa_logits": (mql.mqa_logits, args),
        "triton": (mql.run_mqa_logits, (mql.DEFAULT_CONFIG, *args)),
    }
    flydsl_auto = {"backend": "flydsl", "variant": ""}
    if mql._flydsl_kernels() is not None and mql.flydsl_supports(
        inp["Q"], inp["KV"], mql._flydsl_kernels().DEFAULT_VARIANT
    ):
        candidates["flydsl"] = (mql.run_mqa_logits, (flydsl_auto, *args))

    flops = 2 * HEADS * HEAD_DIM * seq_len * seq_len_kv
    ret = {
        "gfx": get_gfx_runtime(),
        "picked": f"{picked['backend']} {picked['variant']}",
    }
    ref_inf = torch.isneginf(ref)
    for name, (fn, fn_args) in candidates.items():
        with torch.inference_mode():
            out, us = run_perftest(fn, *fn_args)
        got = out[:, :seq_len_kv]
        assert torch.equal(torch.isneginf(got), ref_inf), f"{name}: -inf mask mismatch"
        got, want = got.masked_fill(ref_inf, 0), ref.masked_fill(ref_inf, 0)
        diff = calc_diff(want, got)
        assert diff < 1e-3, f"{name}: calc_diff={diff}"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} err"] = checkAllclose(
            want, got, rtol=1e-2, atol=5.0, msg=f"{name}: mqa_logits", printLog=False
        )
    return ret


def _shipped_flydsl_row():
    mql.reload_tuned_table()
    rows = mql._load_tuned_table().get(
        (get_gfx_runtime(), get_cu_num(), HEADS, HEAD_DIM), {}
    )
    flydsl = sorted(k for k, cfg in rows.items() if cfg["backend"] == "flydsl")
    if not flydsl:
        return None
    seq_len_kv, seq_len = flydsl[0]
    return seq_len, seq_len_kv, rows[(seq_len_kv, seq_len)]


def check_lookup_uses_live_gpu():
    row = _shipped_flydsl_row()
    if row is None:
        aiter.logger.warning(
            "mqa_logits shipped-row check skipped on %s cu=%s",
            get_gfx_runtime(),
            get_cu_num(),
        )
        return
    # chip_info is imported under two names, so GPU_ARCHS only applies in a new process.
    other = "gfx942" if get_gfx_runtime() != "gfx942" else "gfx950"
    env = {**os.environ, "GPU_ARCHS": other, _SHIPPED_ROW_CHILD: "1"}
    subprocess.run([sys.executable, os.path.abspath(__file__)], env=env, check=True)


def _lookup_shipped_row():
    seq_len, seq_len_kv, expected = _shipped_flydsl_row()
    got = mql.get_mqa_logits_config(seq_len, seq_len_kv, HEADS, HEAD_DIM)
    assert (
        got == expected
    ), f"GPU_ARCHS={os.environ['GPU_ARCHS']} looked up {got}, shipped row {expected}"


def _dispatched_config(seq_len, seq_len_kv):
    original = mql.run_mqa_logits
    launched = {}

    def spy(config, *args, **kwargs):
        launched["config"] = dict(config)
        return args[0]

    mql.run_mqa_logits = spy
    try:
        mql.mqa_logits(
            torch.empty((seq_len, HEADS, HEAD_DIM), dtype=dtypes.fp8),
            torch.empty((seq_len_kv, HEAD_DIM), dtype=dtypes.fp8),
            torch.empty((seq_len_kv,), dtype=torch.float32),
            torch.empty((seq_len, HEADS), dtype=torch.float32),
            torch.zeros((seq_len,), dtype=torch.int32),
            torch.ones((seq_len,), dtype=torch.int32),
        )
    finally:
        mql.run_mqa_logits = original
    return launched["config"]


def check_dispatches_shipped_row():
    row = _shipped_flydsl_row()
    if row is None:
        return
    seq_len, seq_len_kv, expected = row
    got = _dispatched_config(seq_len, seq_len_kv)
    assert got == expected, f"dispatched {got}, shipped row {expected}"


def check_flydsl_fallback_uses_triton():
    original_cfg = mql.get_mqa_logits_config
    original_supports = mql.flydsl_supports
    mql.get_mqa_logits_config = lambda *a, **k: {"backend": "flydsl", "variant": "x"}
    mql.flydsl_supports = lambda *a, **k: False
    try:
        assert _dispatched_config(16, 64) == dict(mql.DEFAULT_CONFIG)
    finally:
        mql.get_mqa_logits_config = original_cfg
        mql.flydsl_supports = original_supports


def main():
    if os.environ.get(_SHIPPED_ROW_CHILD):
        assert get_gfx() != get_gfx_runtime()
        _lookup_shipped_row()
        return
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "mqa_logits unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    check_lookup_uses_live_gpu()
    check_dispatches_shipped_row()
    check_flydsl_fallback_uses_triton()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-q",
        "--seq-len",
        type=int,
        nargs="*",
        default=[1024, 4096],
        help="Query rows per call.\n    e.g.: -q 2048",
    )
    parser.add_argument(
        "-k",
        "--seq-len-kv",
        type=int,
        nargs="*",
        default=[8192, 32768],
        help="Keys per call.\n    e.g.: -k 60000",
    )
    args = parser.parse_args()

    rows = [
        test_mqa_logits(seq_len, seq_len_kv)
        for seq_len, seq_len_kv in itertools.product(args.seq_len, args.seq_len_kv)
        if seq_len <= seq_len_kv
    ]
    df = pd.DataFrame(rows)
    aiter.logger.info("mqa_logits summary (markdown):\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()
