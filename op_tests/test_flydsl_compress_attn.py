# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for flydsl ``fused_compress_attn`` kernels (V4-Pro / V4-Flash).

Three shape configs cover both models (they share compressor geometry; the
model_dim difference 7168 vs 4096 is invisible to the kernel):

    CSA Main    : D=512, RD=64, ratio=4,   overlap=True,  BF16
    CSA Indexer : D=128, RD=64, ratio=4,   overlap=True,  FP8 + ue8m0 + preshuffle
    HCA Main    : D=512, RD=64, ratio=128, overlap=False, BF16

Each shape is swept across batch sizes {1,2,4,8,16,32,65,128,256,512} and
speculative-decode step counts {0,3} (MTP3). The HCA Main shape is
additionally cross-checked against the 2-kernel split launcher.
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.ops.flydsl.kernels.fused_compress_attn import flydsl_fused_compress_attn
from aiter.ops.flydsl.kernels.fused_compress_attn_hca import flydsl_hca_compress_attn
from aiter.ops.torch_ref.fused_compress_attn import (
    fused_compress_attn as fused_compress_attn_reference,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

# (label, head_dim, rope_head_dim, ratio, overlap, quant, use_ue8m0, preshuffle)
SHAPES = [
    ("csa_main", 512, 64, 4, True, False, False, False),
    ("csa_indexer", 128, 64, 4, True, True, True, True),
    ("hca_main", 512, 64, 128, False, False, False, False),
]
K_PER_BLOCK = (
    64  # paged-cache compressed-slot block size (multiple of 16 for preshuffle)
)
RMS_EPS = 1e-6
SEED = 2026
PREFILL_CONTEXT_LEN = 256  # per-seq context length for prefill mode
SENTINEL_PAD = 16  # extra plan rows with position=-1 (kernel must skip them)
CG_TOKEN_PAD = 32  # extra kv_in/score_in tail rows filled with NaN (CUDAGraph
# bucket-padding: kernel must not read past the last valid ragged_id, else
# NaN propagates and the test fails loudly)


def _shape_by_label(label):
    for s in SHAPES:
        if s[0] == label:
            return s
    raise KeyError(label)


def _build_inputs(shape, bs, mtp, mode):
    """Build a synthetic plan + tensors for one (shape, bs, mtp, mode) case.

    ``mode`` ∈ {"decode", "prefill"}:

    * decode (CG decode-step path):
        - num_per_seq = max(1, ceil((mtp+1)/ratio)) boundaries — each
          boundary needs ``ratio`` new tokens, so MTP3 with ratio=4
          generates at most 1/seq, with ratio=128 generates at most
          1/seq (sparsely). Matches production CG worst-case.
        - boundary s in seq b: position = (s+1)*ratio - 1, comp slot ci = s
        - ragged_id = (K_pool-1) + b*num_per_seq + s  (offset so input-phase
          rows always have valid in_row even with window_len=0)
        - window_len = K_pool//2 for s==0 (exercise state-cache phase),
                       0 otherwise (exercise pure input phase)

    * prefill (eager fwd over context_len tokens per seq, no MTP):
        - num_per_seq = PREFILL_CONTEXT_LEN // ratio boundaries
        - boundary s_in_seq in seq b: position = (s_in_seq+1)*ratio - 1
        - ragged_id = b * PREFILL_CONTEXT_LEN + position  (one row per input
          token in the ragged stream)
        - window_len = max(0, K_pool - 1 - ragged_id)  → natural state-cache
          reads only for the first 1-2 boundaries of seq 0 (overlap shapes);
          all other boundaries pure input phase

    Both modes append SENTINEL_PAD trailing rows with position=-1 so the
    kernel's plan-capacity > num_compress padding-bail path is exercised.
    """
    label, D, RD, ratio, overlap, quant, ue8m0, preshuffle = shape
    dim_full = (2 if overlap else 1) * D
    K_pool = (2 if overlap else 1) * ratio

    if mode == "decode":
        # Each boundary needs ratio new tokens; in one decode fwd a seq
        # generates at most ceil((mtp+1)/ratio) boundaries. For all our
        # (mtp ∈ {0,3}, ratio ∈ {4,128}) this collapses to 1.
        num_per_seq = max(1, -(-(mtp + 1) // ratio))
        num_compress = bs * num_per_seq
        extra_pad = K_pool - 1
        num_valid_tokens = num_compress + extra_pad
        state_size = K_pool + mtp  # spec-decode ring size convention
    elif mode == "prefill":
        num_per_seq = PREFILL_CONTEXT_LEN // ratio
        num_compress = bs * num_per_seq
        num_valid_tokens = bs * PREFILL_CONTEXT_LEN
        state_size = K_pool  # prefill: state size = pool window (no spec)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    plan_capacity = num_compress + SENTINEL_PAD
    num_q_tokens = num_valid_tokens + CG_TOKEN_PAD  # CUDAGraph bucket padding

    g = torch.Generator(device="cuda").manual_seed(SEED + bs * 1000 + mtp)

    kv_in = torch.randn(num_q_tokens, dim_full, dtype=torch.bfloat16, generator=g) * 0.1
    score_in = (
        torch.randn(num_q_tokens, dim_full, dtype=torch.bfloat16, generator=g) * 0.5
    )
    # Poison the CG-pad tail: kernel must never read these rows (they live
    # past the last valid ragged_id). NaN propagates through softmax/RMSNorm,
    # so any accidental read shows up as NaN in the cache scatter.
    kv_in[num_valid_tokens:] = float("nan")
    score_in[num_valid_tokens:] = float("nan")
    kv_state = (
        torch.randn(bs, state_size, dim_full, dtype=torch.float32, generator=g) * 0.1
    )
    score_state = (
        torch.randn(bs, state_size, dim_full, dtype=torch.float32, generator=g) * 0.5
    )
    state_slot_mapping = torch.arange(bs, dtype=torch.int32)
    ape = torch.randn(ratio, dim_full, dtype=torch.float32, generator=g) * 0.1
    rms_weight = (
        torch.rand(D, dtype=torch.float32, generator=g) * 0.5 + 0.5
    )  # in [0.5, 1.0]

    # cos / sin caches: cover all comp_pos values we'll use.
    max_comp_pos = num_per_seq * ratio
    max_pos = max(max_comp_pos + 1, 64)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, RD, 2, dtype=torch.float32) / RD))
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
    cos_cache = freqs.cos().to(torch.bfloat16).contiguous()
    sin_cache = freqs.sin().to(torch.bfloat16).contiguous()

    # Plan: [num_compress, 4] valid rows + [SENTINEL_PAD, 4] position=-1 rows.
    # Kernel must bail-skip the sentinel rows without touching kv_in or kv_cache.
    plan = torch.full((plan_capacity, 4), -1, dtype=torch.int32)
    for b in range(bs):
        for s in range(num_per_seq):
            pid = b * num_per_seq + s
            if mode == "decode":
                ragged_id = extra_pad + pid
                window_len = K_pool // 2 if s == 0 else 0
            else:  # prefill
                position_in_seq = (s + 1) * ratio - 1
                ragged_id = b * PREFILL_CONTEXT_LEN + position_in_seq
                # window_len uses position_in_seq (not ragged_id) so the
                # K-loop's input-phase reads stay within seq b's input
                # range [b*seq_len, (b+1)*seq_len). Earlier source positions
                # for the first few boundaries of every seq fall back to
                # state-cache reads (or padding when s < 0).
                window_len = max(0, K_pool - 1 - position_in_seq)
            plan[pid, 0] = ragged_id
            plan[pid, 1] = b
            plan[pid, 2] = (s + 1) * ratio - 1  # position → comp slot = s
            plan[pid, 3] = window_len

    # paged cache: one block per seq is enough (num_per_seq ≤ K_PER_BLOCK).
    blocks_per_seq = (num_per_seq + K_PER_BLOCK - 1) // K_PER_BLOCK
    total_blocks = bs * blocks_per_seq + 4
    if quant:
        kv_cache = torch.zeros(total_blocks, K_PER_BLOCK, D, dtype=dtypes.fp8)
        cache_scale = torch.zeros(total_blocks, K_PER_BLOCK, dtype=torch.float32)
    else:
        kv_cache = torch.zeros(total_blocks, K_PER_BLOCK, D, dtype=torch.bfloat16)
        cache_scale = None
    block_tables = torch.zeros(bs, blocks_per_seq, dtype=torch.int32)
    for b in range(bs):
        for j in range(blocks_per_seq):
            block_tables[b, j] = b * blocks_per_seq + j

    return dict(
        kv_in=kv_in,
        score_in=score_in,
        kv_state=kv_state,
        score_state=score_state,
        state_slot_mapping=state_slot_mapping,
        ape=ape,
        rms_weight=rms_weight,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        plan_gpu=plan,
        kv_cache=kv_cache,
        cache_scale=cache_scale,
        block_tables=block_tables,
        k_per_block=K_PER_BLOCK,
        head_dim=D,
        rope_head_dim=RD,
        ratio=ratio,
        overlap=overlap,
        quant=quant,
        use_ue8m0=ue8m0,
        preshuffle=preshuffle,
        rms_eps=RMS_EPS,
    )


def _run_kernel(inp, *, use_2kernel):
    """Run kernel into ``inp['kv_cache']`` / ``inp['cache_scale']`` in place."""
    common = dict(
        kv_in=inp["kv_in"],
        score_in=inp["score_in"],
        kv_state=inp["kv_state"],
        score_state=inp["score_state"],
        state_slot_mapping=inp["state_slot_mapping"],
        plan_gpu=inp["plan_gpu"],
        ape=inp["ape"],
        rms_weight=inp["rms_weight"],
        rms_eps=inp["rms_eps"],
        cos_cache=inp["cos_cache"],
        sin_cache=inp["sin_cache"],
        kv_cache=inp["kv_cache"],
        block_tables=inp["block_tables"],
        k_per_block=inp["k_per_block"],
        ratio=inp["ratio"],
        head_dim=inp["head_dim"],
        rope_head_dim=inp["rope_head_dim"],
    )
    if use_2kernel:
        flydsl_hca_compress_attn(**common)
    else:
        flydsl_fused_compress_attn(
            **common,
            overlap=inp["overlap"],
            quant=inp["quant"],
            cache_scale=inp["cache_scale"],
            use_ue8m0=inp["use_ue8m0"],
            preshuffle=inp["preshuffle"],
        )


@benchmark()
def test_flydsl_compress_attn(shape_label, bs, mtp, mode, path):
    """One case. ``mode`` ∈ {'decode','prefill'}, ``path`` ∈ {'single','2kernel'}."""
    shape = _shape_by_label(shape_label)
    _, D, RD, ratio, overlap, quant, ue8m0, preshuffle = shape
    use_2kernel = path == "2kernel"
    inp = _build_inputs(shape, bs, mtp, mode)

    # Two cache clones — kernel writes to ``inp``, reference to ``ref_inp``.
    ref_inp = dict(inp)
    ref_inp["kv_cache"] = inp["kv_cache"].clone()
    ref_inp["cache_scale"] = (
        inp["cache_scale"].clone() if inp["cache_scale"] is not None else None
    )

    _, us_kernel = run_perftest(_run_kernel, inp, use_2kernel=use_2kernel)

    fused_compress_attn_reference(
        kv_in=ref_inp["kv_in"],
        score_in=ref_inp["score_in"],
        kv_state=ref_inp["kv_state"],
        score_state=ref_inp["score_state"],
        plan_gpu=ref_inp["plan_gpu"],
        state_slot_mapping=ref_inp["state_slot_mapping"],
        ape=ref_inp["ape"],
        rms_weight=ref_inp["rms_weight"],
        rms_eps=ref_inp["rms_eps"],
        cos_cache=ref_inp["cos_cache"],
        sin_cache=ref_inp["sin_cache"],
        kv_cache=ref_inp["kv_cache"],
        block_tables=ref_inp["block_tables"],
        k_per_block=ref_inp["k_per_block"],
        overlap=overlap,
        ratio=ratio,
        head_dim=D,
        rope_head_dim=RD,
        quant=quant,
        cache_scale=ref_inp["cache_scale"],
        use_ue8m0=ue8m0,
        preshuffle=preshuffle,
    )

    msg = f"{shape_label}/{mode}/{path} bs={bs} mtp={mtp}"
    if quant:
        err = checkAllclose(
            inp["kv_cache"].to(dtypes.fp32),
            ref_inp["kv_cache"].to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            tol_err_ratio=0.05,
            msg=f"{msg} kv_cache(fp8)",
        )
        # cache_scale: bit-exact. Reference mirrors the kernel's exact fp32
        # ops (am_safe * inv_fp8_max constant + ue8m0 ceil-pow2), so the
        # scale per row must match to the bit.
        checkAllclose(
            inp["cache_scale"],
            ref_inp["cache_scale"],
            rtol=0,
            atol=0,
            tol_err_ratio=0.0,
            msg=f"{msg} cache_scale(bit-exact)",
        )
    else:
        # BF16 kv_cache: rare rounding-boundary flips at ≤1 ulp because online
        # softmax (kernel) and torch.softmax (reference) sum in different
        # orders. Prefill processes 10-100× more boundaries per case than
        # decode → more chances to land on a rounding boundary, so the
        # element-mismatch ratio scales. Tolerate ≤2%; bound max delta at
        # 2 ulp of bf16 ≈ 2e-2 at unit magnitude.
        err = checkAllclose(
            inp["kv_cache"].to(dtypes.fp32),
            ref_inp["kv_cache"].to(dtypes.fp32),
            rtol=1e-2,
            atol=2e-2,
            tol_err_ratio=0.02,
            msg=f"{msg} kv_cache(bf16)",
        )
    return {"us_kernel": us_kernel, "err_pct": err}


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-s",
    "--shapes",
    type=str,
    nargs="*",
    default=[s[0] for s in SHAPES],
    choices=[s[0] for s in SHAPES],
    help="""Shape labels to run.
    e.g.: -s csa_main hca_main""",
)
parser.add_argument(
    "-b",
    "--bs",
    type=int,
    nargs="*",
    default=[1, 2, 4, 8, 16, 32, 65, 128, 256, 512],
    help="""Batch sizes.
    e.g.: -b 1 32 512""",
)
parser.add_argument(
    "-m",
    "--mtp",
    type=int,
    nargs="*",
    default=[0, 3],
    help="""Speculative-decode step counts (decode mode: num_per_seq = mtp+1).
    Ignored in prefill mode.
    e.g.: -m 0 3""",
)
parser.add_argument(
    "--prefill-bs",
    type=int,
    nargs="*",
    default=[1, 4, 32],
    help="""Batch sizes for prefill mode (prefill ragged tokens = bs*context_len
    grows fast; trimmed list by default).
    e.g.: --prefill-bs 1 4 32""",
)
parser.add_argument(
    "--modes",
    type=str,
    nargs="*",
    default=["decode", "prefill"],
    choices=["decode", "prefill"],
    help="""Which modes to sweep.""",
)

args = parser.parse_args()

df = []
for mode in args.modes:
    bs_list = args.bs if mode == "decode" else args.prefill_bs
    mtp_list = args.mtp if mode == "decode" else [0]  # prefill: mtp irrelevant
    # Sweep order (slowest → fastest changing): shape → mtp → bs.
    # Within each shape, all bs cases for mtp=0 print first, then mtp=3,
    # which makes the perf-vs-bs trend easy to read off the summary table.
    for shape_label, mtp, bs in itertools.product(args.shapes, mtp_list, bs_list):
        df.append(test_flydsl_compress_attn(shape_label, bs, mtp, mode, "single"))
        if shape_label == "hca_main":
            df.append(test_flydsl_compress_attn(shape_label, bs, mtp, mode, "2kernel"))
df = pd.DataFrame(df)
aiter.logger.info(
    "flydsl_compress_attn summary (markdown):\n%s", df.to_markdown(index=False)
)
