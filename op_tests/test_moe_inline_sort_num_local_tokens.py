# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""fused_moe with ``num_local_tokens`` on the tuned MXFP4 inline-sort rows.

A caller that pads its rows (a CUDA-graph decode bucket over fewer real
requests) routes the rows past its real count to expert -1 and passes the
count on device as ``num_local_tokens``. The tuned inline-sort rows must stay
selected (Qwen3.8-Flash-Next TP1 in ``qwen3_8_flash_fp4_tuned_fmoe.csv``), the
leading rows must match a call in which every row is real, and nothing in the
pad rows (NaN activations, -1 routes) may reach them.
"""

import sys

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter import fused_moe as fused_moe_module
from aiter.fused_moe import fused_moe
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight

torch.set_default_device("cuda")

# Qwen3.8-Flash-Next TP1; the last expert is the fused shared expert.
E, MODEL_DIM, INTER_DIM, TOPK = 513, 2560, 640, 11
SHARED = E - 1
# (rows, real counts). The tuned rows are BM16 inline-sort up to 512 tokens;
# from 63 rows on the aux sort is Opus instead of the fused BM16 sort.
CASES = (
    (1, (1,)),
    (4, (1, 3)),
    (16, (5, 16)),
    (44, (1, 43)),
    (128, (64, 127)),
    (256, (200,)),
)
GRAPH_ROWS = (16, 128)
# Same kernels: valid rows differ from the all-real call only by the order of
# stage 2's bf16 atomic adds.
SAME_PATH_REL_ERR = 1e-2
# The legacy path quantizes the activations to MXFP4 before stage 1, the tuned
# inline-sort kernels read them in f16.
LEGACY_REL_ERR = 0.1


class _Weights:
    def __init__(self):
        g = torch.Generator(device="cuda").manual_seed(0)

        def fp4(*shape):
            data = torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)
            return data.view(dtypes.fp4x2)

        def e8m0(rows, cols):
            data = torch.randint(118, 122, (rows, cols), dtype=torch.uint8, generator=g)
            return data.view(dtypes.fp8_e8m0)

        w1 = fp4(E, 2 * INTER_DIM, MODEL_DIM // 2)
        w2 = fp4(E, MODEL_DIM, INTER_DIM // 2)
        self.w1 = moe_shuffle_weight(w1, experts_cnt=E, gate_up=True)
        self.w2 = moe_shuffle_weight(w2, experts_cnt=E, gate_up=False)
        self.w1.is_shuffled = True
        self.w2.is_shuffled = True
        self.w1_scale = moe_shuffle_scale(
            e8m0(E * 2 * INTER_DIM, MODEL_DIM // 32), E, gate_up=True
        )
        self.w2_scale = moe_shuffle_scale(
            e8m0(E * MODEL_DIM, INTER_DIM // 32), E, gate_up=False
        )


def _rows(tokens, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand((tokens, SHARED), generator=g)
    routed = scores.topk(TOPK - 1, dim=1).indices.to(torch.int32)
    shared = torch.full((tokens, 1), SHARED, dtype=torch.int32)
    topk_ids = torch.cat((routed, shared), dim=1).contiguous()
    topk_weights = torch.cat(
        (torch.rand((tokens, TOPK - 1), generator=g), torch.ones((tokens, 1))), dim=1
    ).contiguous()
    hidden = torch.randn((tokens, MODEL_DIM), generator=g).to(torch.bfloat16)
    return hidden, topk_ids, topk_weights


def _pad(rows, valid):
    hidden, topk_ids, topk_weights = (t.clone() for t in rows)
    hidden[valid:] = float("nan")
    topk_ids[valid:] = -1
    topk_weights[valid:] = float("nan")
    return hidden, topk_ids, topk_weights


def _moe(weights, hidden, topk_ids, topk_weights, count=None):
    return fused_moe(
        hidden,
        weights.w1,
        weights.w2,
        topk_weights,
        topk_ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        w1_scale=weights.w1_scale,
        w2_scale=weights.w2_scale,
        num_local_tokens=count,
    )


class _InlineSortVerdicts:
    """Record (or override) what the inline-sort capability check answers."""

    def __init__(self):
        self.verdicts = []
        self.force = None
        self._check = fused_moe_module._mxfp4_inline_sort_unsupported

    def __call__(self, *args, **kwargs):
        verdict = self.force or self._check(*args, **kwargs)
        self.verdicts.append(verdict)
        return verdict

    def take(self):
        verdicts, self.verdicts = self.verdicts, []
        return verdicts


def _close(out, ref, msg, max_rel_err=SAME_PATH_REL_ERR):
    out, ref = out.float(), ref.float()
    rel_err = ((out - ref).norm() / ref.norm()).item()
    aiter.logger.info(f"{msg}: relative L2 error {rel_err:.2e}")
    assert rel_err <= max_rel_err, f"{msg}: relative L2 error {rel_err:.2e}"


def _check_rows(weights, spy, tokens, valid):
    rows = _rows(tokens, seed=tokens)
    count = torch.tensor([valid], dtype=torch.int32)
    padded = _pad(rows, valid)

    out = _moe(weights, *padded, count)
    assert spy.take() == [""], f"M={tokens}: tuned inline-sort row not used"
    assert torch.isfinite(out[:valid]).all(), f"M={tokens} valid={valid}: non-finite"

    every_row_real = _moe(weights, *rows)
    spy.take()
    _close(out[:valid], every_row_real[:valid], f"M={tokens} valid={valid} vs all")

    spy.force = "forced legacy path for the reference"
    legacy = _moe(weights, *padded, count)
    spy.force = None
    spy.take()
    _close(
        out[:valid],
        legacy[:valid],
        f"M={tokens} valid={valid} vs legacy",
        max_rel_err=LEGACY_REL_ERR,
    )


def _check_graph_replay(weights, spy, tokens):
    """A captured call follows the count it reads on every replay."""
    hidden, topk_ids, topk_weights = _pad(_rows(tokens, seed=tokens), tokens)
    count = torch.tensor([tokens], dtype=torch.int32)
    _moe(weights, hidden, topk_ids, topk_weights, count)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _moe(weights, hidden, topk_ids, topk_weights, count)
    for valid in (tokens, tokens // 2 + 1, 1):
        step = _pad(_rows(tokens, seed=1000 + valid), valid)
        for dst, src in zip((hidden, topk_ids, topk_weights), step):
            dst.copy_(src)
        count.fill_(valid)
        graph.replay()
        eager = _moe(weights, *step, count)
        torch.cuda.synchronize()
        _close(out[:valid], eager[:valid], f"graph M={tokens} valid={valid}")
    spy.take()


def main():
    if not torch.cuda.is_available() or get_gfx_runtime() != "gfx950":
        print("skip: the MXFP4 inline-sort rows are gfx950-only")
        return 0
    if get_cu_num() != 256:
        print(f"skip: no tuned Qwen3.8 rows for cu_num={get_cu_num()}")
        return 0
    spy = _InlineSortVerdicts()
    fused_moe_module._mxfp4_inline_sort_unsupported = spy
    weights = _Weights()
    for tokens, counts in CASES:
        for valid in counts:
            _check_rows(weights, spy, tokens, valid)
    for tokens in GRAPH_ROWS:
        _check_graph_replay(weights, spy, tokens)
    aiter.logger.info("inline-sort num_local_tokens: all cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
