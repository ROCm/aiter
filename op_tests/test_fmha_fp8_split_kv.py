# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Split-KV of the gfx1250 FP8 varlen ASM forward (aiter/ops/fmha_fp8_split_kv.py): a split call must give the
unsplit call's output and LSE.

The ASM kernel's LSE is the log of the sum of the FP8-rounded probabilities, so a part with its own running max
rounds a little differently: up to ~5e-3 (natural log) is expected and the tolerance is 2e-2.  Run with
``python -m pytest op_tests/test_fmha_fp8_split_kv.py``; ``python op_tests/test_fmha_fp8_split_kv.py`` also prints
the times."""

import os

import pytest
import torch

from aiter.ops import mha
from aiter.ops.fmha_fp8_split_kv import (
    can_split,
    fmha_fwd_fp8_varlen_asm_split,
    split_factor,
)

pytestmark = pytest.mark.skipif(
    not getattr(mha, "_IS_GFX1250_ASM_DEVICE", False),
    reason="needs a gfx1250 GPU with the FP8 varlen ASM kernel",
)

H, DQK, DV = 12, 192, 128
FP8 = torch.float8_e4m3fn


def _inputs(batch, sq, sk, seed=0):
    torch.manual_seed(seed)
    lq = [max(1, sq - 37 * i) for i in range(batch)]
    lk = [max(0, sk - 1501 * i) for i in range(batch)]
    dev = "cuda"
    cu_q = torch.tensor(
        [0] + torch.tensor(lq).cumsum(0).tolist(), device=dev, dtype=torch.int32
    )
    cu_k = torch.tensor(
        [0] + torch.tensor(lk).cumsum(0).tolist(), device=dev, dtype=torch.int32
    )
    q = (torch.randn(sum(lq), H, DQK, device=dev) * 2).to(FP8)
    k = (torch.randn(max(1, sum(lk)), H, DQK, device=dev) * 2).to(FP8)
    v = (torch.randn(max(1, sum(lk)), H, DV, device=dev) * 2).to(FP8)
    s = torch.tensor([0.02], device=dev)
    return q, k, v, s, cu_q, cu_k, max(lq), max(max(lk), 1)


def _plain(q, k, v, s, cu_q, cu_k, mq, mk):
    old = os.environ.get("AITER_FMHA_FP8_SPLIT_KV")
    os.environ["AITER_FMHA_FP8_SPLIT_KV"] = "0"
    try:
        return mha.fmha_fwd_fp8_varlen_asm(
            q,
            k,
            v,
            s,
            s,
            s,
            cu_q,
            cu_k,
            mq,
            mk,
            softmax_scale=DQK**-0.5,
            is_causal=False,
            return_lse=True,
        )
    finally:
        if old is None:
            os.environ.pop("AITER_FMHA_FP8_SPLIT_KV")
        else:
            os.environ["AITER_FMHA_FP8_SPLIT_KV"] = old


@pytest.mark.parametrize(
    "batch,sq,sk",
    [
        (1, 1024, 16384),
        (1, 401, 12288),
        (1, 128, 16384),
        (3, 600, 16384),
        (2, 1800, 9000),
        (4, 64, 2000),
    ],
)
@pytest.mark.parametrize("parts", [2, 4, 8])
def test_split_matches_plain(batch, sq, sk, parts):
    q, k, v, s, cu_q, cu_k, mq, mk = _inputs(batch, sq, sk)
    o0, l0 = _plain(q, k, v, s, cu_q, cu_k, mq, mk)
    assert can_split(q)
    o1, l1 = fmha_fwd_fp8_varlen_asm_split(
        mha._fmha_fwd_fp8_varlen_asm,
        q,
        k,
        v,
        s,
        s,
        s,
        cu_q,
        cu_k,
        mq,
        mk,
        DQK**-0.5,
        parts,
    )
    torch.testing.assert_close(o1.float(), o0.float(), atol=1e-2, rtol=2e-2)
    fin = torch.isfinite(l0)
    assert torch.equal(fin, torch.isfinite(l1))
    torch.testing.assert_close(l1[fin], l0[fin], atol=2e-2, rtol=1e-3)


def test_empty_sequence_part():
    """A sequence with fewer keys than parts gets empty parts: zero weight in the merge."""
    q, k, v, s, cu_q, cu_k, mq, mk = _inputs(2, 256, 3)
    o0, _ = _plain(q, k, v, s, cu_q, cu_k, mq, mk)
    o1, _ = fmha_fwd_fp8_varlen_asm_split(
        mha._fmha_fwd_fp8_varlen_asm, q, k, v, s, s, s, cu_q, cu_k, mq, mk, DQK**-0.5, 8
    )
    torch.testing.assert_close(o1.float(), o0.float(), atol=1e-2, rtol=2e-2)


def test_dispatch_and_causal_untouched():
    q, k, v, s, cu_q, cu_k, mq, mk = _inputs(1, 1024, 16384)
    assert split_factor(1, H, mq, mk, False, q.device) > 1
    assert split_factor(1, H, mq, mk, True, q.device) == 1
    o_auto, _ = mha.fmha_fwd_fp8_varlen_asm(
        q,
        k,
        v,
        s,
        s,
        s,
        cu_q,
        cu_k,
        mq,
        mk,
        softmax_scale=DQK**-0.5,
        is_causal=False,
        return_lse=True,
    )
    o0, _ = _plain(q, k, v, s, cu_q, cu_k, mq, mk)
    torch.testing.assert_close(o_auto.float(), o0.float(), atol=1e-2, rtol=2e-2)


def _bench(batch, sq, sk):
    """(parts, plain us, dispatched us) for one case, by CUDA events over back-to-back calls."""

    def _time(fn, n=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        st.record()
        for _ in range(n):
            fn()
        en.record()
        torch.cuda.synchronize()
        return st.elapsed_time(en) / n * 1e3

    args = _inputs(batch, sq, sk)
    q, k, v, s, cu_q, cu_k, mq, mk = args
    parts = split_factor(batch, H, mq, mk, False, q.device)
    t0 = _time(lambda: _plain(*args))
    t1 = _time(
        lambda: mha.fmha_fwd_fp8_varlen_asm(
            q,
            k,
            v,
            s,
            s,
            s,
            cu_q,
            cu_k,
            mq,
            mk,
            softmax_scale=DQK**-0.5,
            is_causal=False,
            return_lse=True,
        )
    )
    return parts, t0, t1


if __name__ == "__main__":
    print("batch  sq    sk     parts  plain us  split us  speedup")
    for batch, sq, sk in [
        (1, 1024, 16384),
        (1, 512, 16384),
        (1, 256, 16384),
        (1, 925, 12288),
        (2, 1024, 16384),
        (3, 600, 16384),
        (1, 1024, 4096),
        (1, 4096, 16384),
    ]:
        parts, t0, t1 = _bench(batch, sq, sk)
        print(
            f"{batch:5d} {sq:5d} {sk:6d} {parts:6d} {t0:9.1f} {t1:9.1f} {t0 / t1:8.2f}x"
        )
