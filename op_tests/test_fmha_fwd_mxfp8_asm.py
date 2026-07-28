# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Correctness + performance test for the dedicated gfx1250 MXFP8 ASM FMHA
# forward path (aiter.fmha_fwd_mxfp8_asm).  Follows the aiter op_test standard:
# a @benchmark sweep fn whose call args are the table columns, candidates timed
# with run_perftest + checked with checkAllclose against a torch reference, and
# one markdown summary table emitted from main().

import argparse
import itertools

import aiter
import numpy as np
import pandas as pd
import torch
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.test_mha_common import attention_ref

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx1250"]  # MXFP8 ASM kernel is only shipped for gfx1250

BLOCK_SIZE = 32
SUB_Q = 256
SUB_K = 128

# E8M0 byte for scale 2^0 = 1.0 (no scaling); used to pad align/shuffle regions.
E8M0_ONE = 0x7F
# FP8 (e4m3) representable max; pattern-2 inverse-scaled values are clamped to it.
FP8_MAX = float(torch.finfo(dtypes.fp8).max)

# --- init_pattern controls how the fp8 q/k/v data and E8M0 scales are built ---
#   0: data and scale both random -- logical ~ N(0,1), E8M0 scale = 2^uniform-int,
#      stored fp8 = logical / scale (inverse-scale, then cast to fp8).
#   1: data = 0.25 (fixed), scale = 1.0 (0x7f)  -- data-independent peak-perf run.
#   2: the company init method (poc fmha_fwd_mxfp8.cpp):
#        - logical data ~ N(0, 1)
#        - E8M0 scale  = 2^(Binomial(2N+1, 0.5) - (N+1)), N=10
#        - stored fp8  = logical / scale  (inverse-scale, then cast to fp8)
# Patterns 0 and 2 differ only in the scale distribution (uniform vs Binomial);
# both inverse-scale so the dequant stays well-conditioned.  In every pattern the
# kernel and CPU reference both reconstruct `deq = stored_fp8 * 2^(scale_exp)`,
# so the two stay bit-consistent.
DATA_CONST = 0.25  # init_pattern=1 fixed q/k/v value
BINOMIAL_N = 10  # E8M0 scale ~ 2^(Binomial(2N+1,0.5) - (N+1)), matches poc host
# init_pattern=0 scale exponent ~ uniform int in [-R, R].  Because pattern 0 now
# inverse-scales (logical is fixed to N(0,1)), the dequant stays well-conditioned
# for any range, so R can span a wide scale spread without hurting accuracy.
PATTERN0_EXP_RANGE = 2


def _exp_to_e8m0_bytes(e):
    """int exponent tensor -> E8M0 byte array (uint8), value = 2^(byte-127)."""
    return (e.to(torch.int32) + 127).clamp(0, 254).to(torch.uint8).cpu().numpy()


def _gen_scale_exp(shape, init_pattern):
    """E8M0 scale exponent tensor (real scale = 2^exp) for the given shape.

    Kept in a valid, well-conditioned range so the dequantized q/k/v never
    overflow: pattern 0 uses a small uniform exponent, pattern 2 uses the poc
    Binomial recipe (torch native), pattern 1 is exponent 0 (scale = 1.0).
    """
    if init_pattern == 1:
        return torch.zeros(shape)
    if init_pattern == 2:
        n = BINOMIAL_N
        binom = torch.distributions.Binomial(2 * n + 1, torch.tensor(0.5))
        return binom.sample(shape) - (n + 1)
    r = PATTERN0_EXP_RANGE  # init_pattern == 0
    return torch.randint(-r, r + 1, shape).float()


def _gen_stored_fp8(shape, scale, init_pattern, block_dim):
    """Stored fp8 data for a block layout.  `block_dim` is the axis index that
    is block-quantized (scale broadcasts over the BLOCK_SIZE within it).

    Patterns 0 and 2 both inverse-scale: the dequantized logical data is fixed
    to N(0,1) and the stored fp8 is `logical / scale`, so the dequant
    (`stored * scale`) stays well-conditioned regardless of the random scale.
    Without inverse-scaling, an independent random per-block scale blows up the
    intra-row dynamic range and the fp8 matmul error exceeds atol.
    """
    if init_pattern == 1:
        return torch.full(shape, DATA_CONST, dtype=torch.bfloat16).to(dtypes.fp8)
    # patterns 0 and 2: logical ~ N(0,1), stored = clamp(logical / scale) -> fp8.
    logical = torch.randn(shape, dtype=torch.float32)
    lb = logical.unflatten(block_dim, (-1, BLOCK_SIZE))
    stored = torch.clamp(lb / scale.unsqueeze(block_dim + 1), -FP8_MAX, FP8_MAX)
    return stored.reshape(shape).to(dtypes.fp8)


def _gen_qkv(shape, init_pattern, block_dim):
    """Generate (stored_fp8, dense E8M0 scale bytes, dequant bf16) for one of
    q/k/v.  `block_dim`: 3 for Q/K (block along head_dim), 2 for V (block along
    seq).  Scale is per block; deq = stored_fp8 * 2^scale_exp."""
    scale_shape = list(shape)
    scale_shape[block_dim] //= BLOCK_SIZE
    scale_exp = _gen_scale_exp(tuple(scale_shape), init_pattern)
    scale = torch.exp2(scale_exp)  # float, per block

    stored = _gen_stored_fp8(shape, scale, init_pattern, block_dim)
    sb = stored.float().unflatten(block_dim, (-1, BLOCK_SIZE))
    # Exact fp32 dequant (fp8 * 2^exp is exact): this is the ground-truth the
    # kernel reconstructs from the same fp8 bytes + scale.  Keep it fp32 (do not
    # round to bf16) so the reference does not add error the kernel never had.
    deq = (sb * scale.unsqueeze(block_dim + 1)).reshape(shape)
    return stored, _exp_to_e8m0_bytes(scale_exp), deq


# ---------------------------------------------------------------------------
# GPU scale-buffer layout: per-batch tile-aligned + shuffled, matching the poc
# host (fmha_fwd_mxfp8.cpp: fmha_scale_shuffle_seq / fmha_v_scale_shuffle_hdim).
# ---------------------------------------------------------------------------

# Q/K seq shuffle within each 64-row tile: dst row i takes src row _SEQ_PERM[i].
_SEQ_TILE = 64
_SEQ_PERM = np.array(
    [
        (i % 32) // 2 + (i // 32) * 32 + (16 if (i % 32) % 2 else 0)
        for i in range(_SEQ_TILE)
    ],
    dtype=np.int64,
)


def _e8m0_tensor_from_bytes(flat_u8):
    """Reinterpret a flat uint8 numpy array as a float8_e8m0fnu cuda tensor."""
    e8 = torch.empty(flat_u8.shape, dtype=torch.float8_e8m0fnu, device="cuda")
    e8.view(torch.uint8).copy_(torch.from_numpy(flat_u8))
    return e8


def build_gpu_scale_qk(
    dense_bytes, batch, head_num, seq_len, head_dim, sub, extra_tiles=0
):
    """Q/K GPU scale: pad seq to `sub`-aligned then interleave-shuffle each
    64-row tile.  dense_bytes: [batch, head_num, seq_len, head_dim//BLOCK_SIZE].

    `extra_tiles` appends N `sub`-sized tiles of 0x7f (scale=1.0) to the very
    end of the flat buffer (NOT per head, which would corrupt the per-head seq
    stride the kernel indexes by).  This guards the K-scale over-read (2 tiles)
    that create_mxfp8_scale_buffer(extra_tiles=2) also guards in the all-1.0
    path -- it keeps the last head's over-read in-bounds.
    """
    scale_cols = head_dim // BLOCK_SIZE
    aligned_seq = align_to_tile(seq_len, sub)
    gpu = np.full((batch, head_num, aligned_seq, scale_cols), E8M0_ONE, dtype=np.uint8)
    gpu[:, :, :seq_len, :] = dense_bytes
    n_tiles = aligned_seq // _SEQ_TILE
    gpu = gpu.reshape(batch * head_num, n_tiles, _SEQ_TILE, scale_cols)
    gpu = gpu[:, :, _SEQ_PERM, :]
    flat = np.ascontiguousarray(gpu).reshape(-1)
    if extra_tiles:
        pad = np.full(extra_tiles * sub * scale_cols, E8M0_ONE, dtype=np.uint8)
        flat = np.concatenate([flat, pad])
    return _e8m0_tensor_from_bytes(flat)


def build_gpu_scale_v(dense_bytes, batch, kv_head, seq_len, v_head_dim, sub_k):
    """V GPU scale: pad seq-rows to `sub_k`-aligned then regroup head_dim into
    64-wide units per `sub_k//BLOCK_SIZE`-row tile.  dense_bytes:
    [batch, kv_head, align(seq,BLOCK_SIZE)//BLOCK_SIZE, v_head_dim]."""
    seq_rows_cpu = align_to_tile(seq_len, BLOCK_SIZE) // BLOCK_SIZE
    seq_rows_gpu = align_to_tile(seq_len, sub_k) // BLOCK_SIZE
    gpu = np.full((batch, kv_head, seq_rows_gpu, v_head_dim), E8M0_ONE, dtype=np.uint8)
    gpu[:, :, :seq_rows_cpu, :] = dense_bytes

    unit_hdim = 64
    num_units = v_head_dim // unit_hdim
    if num_units > 1:
        tile_seq_rows = sub_k // BLOCK_SIZE
        n_tiles = seq_rows_gpu // tile_seq_rows
        gpu = gpu.reshape(batch * kv_head, n_tiles, tile_seq_rows, num_units, unit_hdim)
        gpu = gpu.transpose(
            0, 1, 3, 2, 4
        )  # -> [.., num_units, tile_seq_rows, unit_hdim]
        gpu = np.ascontiguousarray(gpu)
    return _e8m0_tensor_from_bytes(gpu.reshape(-1))


def align_to_tile(original, tile_size):
    return (original + tile_size - 1) // tile_size * tile_size


def make_inputs(batch, nheads, nheads_k, seqlen_q, seqlen_k, d, init_pattern=0):
    """Build fp8 q/k/v as BSHD-shaped views over BHSD memory + e8m0 scales.

    Reproduces the real call layout: the MXFP8 kernel consumes bshd-shaped
    tensors backed by bhsd memory (head stride > seq stride), so q/k/v are
    transposed views of contiguous [b, h, s, d] tensors (not fresh contiguous
    [b, s, h, d] tensors).

    `init_pattern` (see the module-level note) selects the data/scale init.
    The GPU scale buffers are always laid out exactly as the kernel expects
    (per-batch tile-aligned + shuffled, mirroring the poc host).

    Returns q/k/v bshd views, the three GPU scale buffers, and the dequantized
    bf16 q/k/v (bhsd) used to build the CPU reference.
    """
    torch.random.manual_seed(0)
    d_v = d

    # block along head_dim (dim 3) for Q/K, along seq (dim 2) for V.
    q_fp8, q_sb, q_deq = _gen_qkv((batch, nheads, seqlen_q, d), init_pattern, 3)
    k_fp8, k_sb, k_deq = _gen_qkv((batch, nheads_k, seqlen_k, d), init_pattern, 3)
    v_fp8, v_sb, v_deq = _gen_qkv((batch, nheads_k, seqlen_k, d_v), init_pattern, 2)

    q_scale = build_gpu_scale_qk(q_sb, batch, nheads, seqlen_q, d, SUB_Q)
    # K-scale kernel over-read (2 tiles) workaround.
    k_scale = build_gpu_scale_qk(
        k_sb, batch, nheads_k, seqlen_k, d, SUB_K, extra_tiles=2
    )
    v_scale = build_gpu_scale_v(v_sb, batch, nheads_k, seqlen_k, d_v, SUB_K)

    q_in = q_fp8.transpose(1, 2)
    k_in = k_fp8.transpose(1, 2)
    v_in = v_fp8.transpose(1, 2)

    return q_in, k_in, v_in, q_scale, k_scale, v_scale, q_deq, k_deq, v_deq


def run_torch(q_deq, k_deq, v_deq, causal):
    """Reference only: bf16 attention over the dequantized q/k/v (fp32 math
    inside attention_ref).  q/k/v_deq are bhsd; not timed, not in the table."""
    q_ref = q_deq.transpose(1, 2)
    k_ref = k_deq.transpose(1, 2)
    v_ref = v_deq.transpose(1, 2)
    out_ref, _, _ = attention_ref(q_ref, k_ref, v_ref, causal=causal, upcast=True)
    return out_ref


@benchmark()
def test_fmha_fwd_mxfp8(batch, nheads, nheads_k, seqlen, d, causal, init_pattern=0):
    (
        q_in,
        k_in,
        v_in,
        q_scale,
        k_scale,
        v_scale,
        q_deq,
        k_deq,
        v_deq,
    ) = make_inputs(
        batch, nheads, nheads_k, seqlen, seqlen, d, init_pattern=init_pattern
    )
    d_v = d

    ref = run_torch(q_deq, k_deq, v_deq, causal)

    candidates = {
        "asm": lambda: aiter.fmha_fwd_mxfp8_asm(
            q_in,
            k_in,
            v_in,
            q_scale,
            k_scale,
            v_scale,
            is_causal=causal,
            return_lse=True,
        )[0],
    }

    # forward attention FLOPs: QK^T (d) + P*V (d_v) per (q,k) pair, 2 per MAC.
    # causal only computes the lower triangle -> ~half the (q,k) pairs.
    n_pairs = seqlen * (seqlen + 1) // 2 if causal else seqlen * seqlen
    flops = 2 * batch * nheads * n_pairs * (d + d_v)
    # element traffic (bytes): fp8 q/k/v in + bf16 out.
    nbytes = (
        batch * nheads * seqlen * d  # q   (fp8, 1B)
        + batch * nheads_k * seqlen * d  # k   (fp8, 1B)
        + batch * nheads_k * seqlen * d_v  # v   (fp8, 1B)
        + batch * nheads * seqlen * d_v * 2  # out (bf16, 2B)
    )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        # fp8 MHA tolerance: the fp8 matmul has an inherent per-element error
        # (worst ~0.056 across all swept shapes).  atol=0.0625 (2^-4) / rtol=0.02
        # keeps every element within tolerance (clean pass, no warning); the
        # kernel is deterministic (bit-exact across launches) so this is stable.
        # Still far tighter than aiter's fp8 attention_ref_with_tol floors of
        # atol=0.5/rtol=0.1.
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=2e-2,
            atol=6.25e-2,
            msg=f"{name}: fmha_fwd_mxfp8",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("fmha_fwd_mxfp8 unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MXFP8 ASM FMHA forward test (gfx1250)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[3],
        help="batch sizes to sweep",
    )
    parser.add_argument(
        "-hk",
        "--hqk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(5, 5), (8, 2)],
        help="(head_num_q, head_num_kv) pairs, e.g. 5,5 8,2",
    )
    parser.add_argument(
        "-s",
        "--seqlen",
        type=int,
        nargs="*",
        default=[256, 384, 512, 1024, 8192],
        help="sequence lengths to sweep (seqlen_q == seqlen_k)",
    )
    parser.add_argument(
        "-d",
        "--head_dim",
        type=int,
        nargs="*",
        default=[128],
        help="head dims to sweep",
    )
    parser.add_argument(
        "-c",
        "--causal",
        type=dtypes.str2bool,
        nargs="*",
        default=[False],
        help="causal flags to sweep (requires seqlen_q == seqlen_k)",
    )
    parser.add_argument(
        "-ip",
        "--init_pattern",
        type=int,
        nargs="*",
        default=[0],
        help=(
            "init patterns to sweep: 0=random fp8 data + random E8M0 scale, "
            "1=data 0.25 + scale 1.0 (0x7f), 2=N(0,1) data + Binomial E8M0 scale "
            "(poc method)"
        ),
    )
    args = parser.parse_args()

    df = []
    for (
        (nheads, nheads_k),
        batch,
        seqlen,
        d,
        causal,
        init_pattern,
    ) in itertools.product(
        args.hqk,
        args.batch,
        args.seqlen,
        args.head_dim,
        args.causal,
        args.init_pattern,
    ):
        df.append(
            test_fmha_fwd_mxfp8(
                batch, nheads, nheads_k, seqlen, d, causal, init_pattern=init_pattern
            )
        )
    df = pd.DataFrame(df)
    aiter.logger.info("fmha_fwd_mxfp8 summary:\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()
