# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for gfx1250 (wave32) ``fused_compress_attn`` kernels.

Directly imports the gfx1250 kernels (bypassing the dispatch routing) so that
correctness can be verified independently. Tests the same three shape configs
as the wave64 test, but with preshuffle forced False (gfx1250 uses linear
FP8 layout).

    CSA Main    : D=512, RD=64, ratio=4,   overlap=True,  BF16
    CSA Indexer : D=128, RD=64, ratio=4,   overlap=True,  FP8 + ue8m0, no preshuffle
    HCA Main    : D=512, RD=64, ratio=128, overlap=False, BF16

Each shape swept across batch sizes and speculative-decode step counts.
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.ops.flydsl.kernels.fused_compress_attn_gfx1250 import (
    flydsl_fused_compress_attn_gfx1250,
)
from aiter.ops.flydsl.kernels.fused_compress_attn_hca_gfx1250 import (
    flydsl_hca_compress_attn_gfx1250,
)
from aiter.ops.torch_ref.fused_compress_attn import (
    fused_compress_attn as fused_compress_attn_reference,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

# (label, head_dim, rope_head_dim, ratio, overlap, quant, use_ue8m0, preshuffle)
# gfx1250: preshuffle forced False for all shapes
SHAPES = [
    ("csa_main", 512, 64, 4, True, False, False, False),
    ("csa_indexer", 128, 64, 4, True, True, True, False),
    ("hca_main", 512, 64, 128, False, False, False, False),
]
K_PER_BLOCK = 64
RMS_EPS = 1e-6
SEED = 2026
PREFILL_CONTEXT_LEN = 256
SENTINEL_PAD = 16
CG_TOKEN_PAD = 32


def _shape_by_label(label):
    for s in SHAPES:
        if s[0] == label:
            return s
    raise KeyError(label)


def _build_inputs(shape, bs, mtp, mode):
    """Build synthetic plan + tensors for one (shape, bs, mtp, mode) case."""
    label, D, RD, ratio, overlap, quant, ue8m0, preshuffle = shape
    dim_full = (2 if overlap else 1) * D
    K_pool = (2 if overlap else 1) * ratio

    if mode == "decode":
        num_per_seq = max(1, -(-(mtp + 1) // ratio))
        num_compress = bs * num_per_seq
        extra_pad = K_pool - 1
        num_valid_tokens = num_compress + extra_pad
        state_size = K_pool + mtp
    elif mode == "prefill":
        num_per_seq = PREFILL_CONTEXT_LEN // ratio
        num_compress = bs * num_per_seq
        num_valid_tokens = bs * PREFILL_CONTEXT_LEN
        state_size = K_pool
    else:
        raise ValueError(f"unknown mode {mode!r}")
    plan_capacity = num_compress + SENTINEL_PAD
    num_q_tokens = num_valid_tokens + CG_TOKEN_PAD

    g = torch.Generator(device="cuda").manual_seed(SEED + bs * 1000 + mtp)

    kv_in = torch.randn(num_q_tokens, dim_full, dtype=torch.bfloat16, generator=g) * 0.1
    score_in = (
        torch.randn(num_q_tokens, dim_full, dtype=torch.bfloat16, generator=g) * 0.5
    )
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
    rms_weight = torch.rand(D, dtype=torch.float32, generator=g) * 0.5 + 0.5

    max_comp_pos = num_per_seq * ratio
    max_pos = max(max_comp_pos + 1, 64)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, RD, 2, dtype=torch.float32) / RD))
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
    cos_cache = freqs.cos().to(torch.bfloat16).contiguous()
    sin_cache = freqs.sin().to(torch.bfloat16).contiguous()

    plan = torch.full((plan_capacity, 4), -1, dtype=torch.int32)
    for b in range(bs):
        for s in range(num_per_seq):
            pid = b * num_per_seq + s
            if mode == "decode":
                ragged_id = extra_pad + pid
                window_len = K_pool // 2 if s == 0 else 0
            else:
                position_in_seq = (s + 1) * ratio - 1
                ragged_id = b * PREFILL_CONTEXT_LEN + position_in_seq
                window_len = max(0, K_pool - 1 - position_in_seq)
            plan[pid, 0] = ragged_id
            plan[pid, 1] = b
            plan[pid, 2] = (s + 1) * ratio - 1
            plan[pid, 3] = window_len

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
    """Run gfx1250 kernel into ``inp['kv_cache']`` / ``inp['cache_scale']``."""
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
        flydsl_hca_compress_attn_gfx1250(**common)
    else:
        flydsl_fused_compress_attn_gfx1250(
            **common,
            overlap=inp["overlap"],
            quant=inp["quant"],
            cache_scale=inp["cache_scale"],
            use_ue8m0=inp["use_ue8m0"],
            preshuffle=inp["preshuffle"],
        )


@benchmark()
def test_flydsl_compress_attn_gfx1250(shape_label, bs, mtp, mode, path):
    """One case. ``mode`` in {'decode','prefill'}, ``path`` in {'single','2kernel'}."""
    shape = _shape_by_label(shape_label)
    _, D, RD, ratio, overlap, quant, ue8m0, preshuffle = shape
    use_2kernel = path == "2kernel"
    inp = _build_inputs(shape, bs, mtp, mode)

    ref_inp = dict(inp)
    ref_inp["kv_cache"] = inp["kv_cache"].clone()
    ref_inp["cache_scale"] = (
        inp["cache_scale"].clone() if inp["cache_scale"] is not None else None
    )

    _, us_kernel = run_perftest(_run_kernel, inp, use_2kernel=use_2kernel)

    # Reference uses preshuffle=False for gfx1250 comparison
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
        preshuffle=False,  # gfx1250: always linear layout
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
        checkAllclose(
            inp["cache_scale"],
            ref_inp["cache_scale"],
            rtol=0,
            atol=0,
            tol_err_ratio=0.0,
            msg=f"{msg} cache_scale(bit-exact)",
        )
    else:
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
    description="gfx1250 (wave32) compress_attn correctness + perf test",
)
parser.add_argument(
    "-s",
    "--shapes",
    type=str,
    nargs="*",
    default=[s[0] for s in SHAPES],
    choices=[s[0] for s in SHAPES],
    help="Shape labels to run. e.g.: -s csa_main hca_main",
)
parser.add_argument(
    "-b",
    "--bs",
    type=int,
    nargs="*",
    default=[1, 2, 4, 8, 16, 32, 128, 512],
    help="Batch sizes. e.g.: -b 1 32 512",
)
parser.add_argument(
    "-m",
    "--mtp",
    type=int,
    nargs="*",
    default=[0, 3],
    help="Speculative-decode step counts. e.g.: -m 0 3",
)
parser.add_argument(
    "--prefill-bs",
    type=int,
    nargs="*",
    default=[1, 4, 32],
    help="Batch sizes for prefill mode.",
)
parser.add_argument(
    "--modes",
    type=str,
    nargs="*",
    default=["decode", "prefill"],
    choices=["decode", "prefill"],
    help="Which modes to sweep.",
)

args = parser.parse_args()

df = []
for mode in args.modes:
    bs_list = args.bs if mode == "decode" else args.prefill_bs
    mtp_list = args.mtp if mode == "decode" else [0]
    for shape_label, mtp, bs in itertools.product(args.shapes, mtp_list, bs_list):
        df.append(
            test_flydsl_compress_attn_gfx1250(shape_label, bs, mtp, mode, "single")
        )
        if shape_label == "hca_main":
            df.append(
                test_flydsl_compress_attn_gfx1250(shape_label, bs, mtp, mode, "2kernel")
            )
df = pd.DataFrame(df)
aiter.logger.info(
    "flydsl_compress_attn_gfx1250 summary (markdown):\n%s",
    df.to_markdown(index=False),
)
