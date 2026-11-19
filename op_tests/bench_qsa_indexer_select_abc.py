# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Interleaved session for arms A, B, C, and the fused indexer.

One gpuq job. Live indexer pages are filled with random bf16 before any
arm runs, because an uninitialized pool makes top-k time depend on garbage.
Arms alternate inside each repeat. Sparse GQA is not launched.

The stage probe is not used. It replaces ``paged._run_compiled`` for every
caller while it is open, so a probe around this loop adds events inside the
fused window and inside arm A's own stage split. Arm A is one event pair
around ``qsa_indexer_block_ids`` plus expand: ``T_score`` is that block-ids
call and ``T_topk`` is 0. The vLLM arms stay the sum of their score, top-k,
and expand kernels. Fused ``T_step`` is the whole call.

Launch from the repo root, in flydsl_venv, through gpuq, after the three
arms' kernels are already compiled::

    gpuq --num-gpus=1 /workspaces/aiter/flydsl_venv/bin/python -u \\
        op_tests/bench_qsa_indexer_select_abc.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_qsa_indexer_select as bench
from qsa_phase0_arm_vllm62589 import ARM_LABEL as ARM_C_LABEL
from qsa_phase0_arm_vllm62589 import Arm as ArmC
from qsa_phase0_arm_vllm65188 import ARM_LABEL as ARM_B_LABEL
from qsa_phase0_arm_vllm65188 import Arm as ArmB
from qsa_phase0_arm_vllm65188 import num_decode_rows

from aiter.ops.flydsl import qsa_paged as paged
from aiter.ops.qsa import qsa_expand_block_indices

ARM_A_LABEL = "aiter#5996"
ARM_F_LABEL = "fused"


def _live_pages(cases) -> int:
    pages = 0
    for case in cases:
        n = 0
        for req in case.requests:
            cols = req.context // bench.COMPRESS
            n += (cols + bench.PAGE - 1) // bench.PAGE
        pages = max(pages, n)
    return pages


def _us(pairs) -> float:
    return sum(start.elapsed_time(end) for start, end in pairs) * 1000.0


def _arm_a(q, index_k, table, token_to_req, positions, context, block_ids, token_ids):
    """Whole block-ids call plus expand. No stage probe."""
    start = torch.cuda.Event(enable_timing=True)
    mid = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    paged.qsa_indexer_block_ids(
        q,
        index_k,
        table,
        token_to_req,
        positions,
        context,
        out=block_ids,
        heads=(bench.HEADS,),
    )
    mid.record()
    qsa_expand_block_indices(
        block_ids,
        positions,
        context,
        token_to_req,
        bench.COMPRESS,
        bench.TOKEN_TOPK,
        token_ids,
    )
    end.record()
    torch.cuda.synchronize()
    return (
        start.elapsed_time(mid) * 1000.0,
        0.0,
        mid.elapsed_time(end) * 1000.0,
    )


def _arm_fused(
    q,
    index_k,
    table,
    token_to_req,
    positions,
    context,
    block_ids,
    token_ids,
    scratch,
    max_seq_len,
):
    """Fused score, merge, and expand. The whole call is ``T_step``."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    paged._write_fused_indexer_block_ids(
        q,
        index_k,
        table,
        token_to_req,
        positions,
        context,
        block_ids,
        token_ids,
        paged._SCORE_SCALE,
        scratch,
        max_seq_len,
    )
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0, 0.0, 0.0


def _arm_vllm(arm, q, table, token_to_req, positions, context, token_ids, decode_rows):
    probe = {"score": [], "topk": [], "expand": []}
    arm.step(q, table, token_to_req, positions, context, token_ids, decode_rows, probe)
    torch.cuda.synchronize()
    return _us(probe["score"]), _us(probe["topk"]), _us(probe["expand"])


def main() -> None:
    bench._print_device()
    bench._assert_idle()
    cases = bench._cases()
    index_k = bench._wide_indexer_cache()
    live = _live_pages(cases)
    torch.manual_seed(0)
    index_k[:live].normal_()
    torch.cuda.synchronize()
    print(
        f"indexer pool pages={index_k.shape[0]} span_bytes={paged.span_bytes(index_k)} "
        f"filled_pages={live}",
        flush=True,
    )
    arm_b = ArmB(index_k, bench.MAX_MODEL_LEN)
    arm_c = ArmC(index_k, bench.MAX_MODEL_LEN)
    flush = torch.empty(bench.FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    order = (ARM_A_LABEL, ARM_B_LABEL, ARM_C_LABEL, ARM_F_LABEL)
    rows = []
    for case in cases:
        built = bench._build(case, index_k)
        (
            q,
            _,
            table,
            token_to_req,
            positions,
            context,
            _,
            _,
            block_ids,
            token_ids,
        ) = built
        decode_rows = num_decode_rows(case)
        max_seq_len = max(req.context for req in case.requests)
        arm_b.set_max_seq_len(context, max_seq_len)
        scratch = torch.empty(
            paged.qsa_indexer_select_workspace_elems(
                q.shape[0], max_seq_len, bench.PAGE
            ),
            dtype=torch.float32,
            device=q.device,
        )
        ids = {
            ARM_A_LABEL: (
                block_ids,
                token_ids.clone(),
            ),
            ARM_B_LABEL: (None, token_ids.clone()),
            ARM_C_LABEL: (None, token_ids.clone()),
            ARM_F_LABEL: (block_ids.clone(), token_ids.clone()),
        }

        def once(
            label,
            cold,
            ids=ids,
            q=q,
            table=table,
            token_to_req=token_to_req,
            positions=positions,
            context=context,
            decode_rows=decode_rows,
            scratch=scratch,
            max_seq_len=max_seq_len,
        ):
            if cold:
                bench._flush(flush)
            block, tokens = ids[label]
            if label == ARM_A_LABEL:
                return _arm_a(
                    q,
                    index_k,
                    table,
                    token_to_req,
                    positions,
                    context,
                    block,
                    tokens,
                )
            if label == ARM_F_LABEL:
                return _arm_fused(
                    q,
                    index_k,
                    table,
                    token_to_req,
                    positions,
                    context,
                    block,
                    tokens,
                    scratch,
                    max_seq_len,
                )
            arm = arm_b if label == ARM_B_LABEL else arm_c
            return _arm_vllm(
                arm, q, table, token_to_req, positions, context, tokens, decode_rows
            )

        for repeat, cold in (("warm-1", False), ("warm-2", False), ("cold", True)):
            print(f"{case.name}: {repeat}", flush=True)
            for _ in range(bench.WARMUP):
                for label in order:
                    once(label, cold)
            samples = {label: [] for label in order}
            for _ in range(bench.ITERS):
                for label in order:
                    samples[label].append(once(label, cold))
            for label in order:
                med = [
                    statistics.median(sample[i] for sample in samples[label])
                    for i in range(3)
                ]
                step = sum(med)
                got = {
                    "case": case.name,
                    "arm": label,
                    "repeat": repeat,
                    "score": med[0],
                    "topk": med[1],
                    "expand": med[2],
                    "step": step,
                    "share": (med[1] + med[2]) / step if step else float("nan"),
                }
                print(
                    f"{case.name} {label} {repeat}: T_score {got['score']:.1f} "
                    f"T_topk {got['topk']:.1f} T_expand {got['expand']:.1f} "
                    f"T_step {got['step']:.1f} share {got['share']:.3f}",
                    flush=True,
                )
                rows.append(got)
        del built
        torch.cuda.empty_cache()
    print(
        "arm A T_score is the block-ids call and T_topk is 0; "
        "vLLM T_step is the sum of its kernels; fused T_step is the whole call",
        flush=True,
    )
    print(
        "case | arm | repeat | T_score | T_topk | T_expand | T_step | top-k+expand share",
        flush=True,
    )
    for got in rows:
        print(
            f"{got['case']} | {got['arm']} | {got['repeat']} | {got['score']:.1f} | "
            f"{got['topk']:.1f} | {got['expand']:.1f} | {got['step']:.1f} | "
            f"{got['share']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
