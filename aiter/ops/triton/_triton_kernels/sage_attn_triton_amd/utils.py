import csv
import math
import torch
import os
import random
import functools
import triton
import triton.language as tl
import numpy as np
from typing import Literal, Optional, Union, Tuple
from enum import Enum

# -------------------------------
# Gloabl Variables
# -------------------------------
AUTOTUNE = os.environ.get("SAGE_ATTENTION_TRITON_AMD_AUTOTUNE", "0").lower() in (
    "1",
    "true",
    "yes",
)
DEBUG = os.environ.get("SAGE_ATTENTION_TRITON_AMD_DEBUG", "0").lower() in (
    "1",
    "true",
    "yes",
)
if AUTOTUNE or DEBUG:
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
USE_TRITON_ROCM = os.getenv("SAGE_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
USE_TRITON_INTERPRET = os.environ.get("TRITON_INTERPRET", "0").lower() in (
    "1",
    "true",
    "yes",
)
DEBUG_TRITON = (
    os.environ.get("DEBUG_TRITON", "0").lower() in ("1", "true", "yes")
    and USE_TRITON_INTERPRET
)
DEBUG_TRITON_DETAIL = (
    os.environ.get("DEBUG_TRITON_DETAIL", "0").lower() in ("1", "true", "yes")
    and USE_TRITON_INTERPRET
)
if USE_TRITON_ROCM:  # TODO remove this
    random.seed(42)
BWD_MODE: Literal["fused", "fused_atomic", "split"] = "fused"
USE_EXP2 = True
PHILOX_SEED = 0x1BF58
PHILOX_OFFSET = 0x1D4B49
SHAPE_EXPECTATIONS: Literal["exact", "rounded"] = "exact"
FP8_AUTO_DESCALE = False


# -------------------------------
# Input Helper
# -------------------------------
def map_dims(shape, indices):
    return [shape[i] for i in indices]


def random_seqlens_composition(SEQ_LEN, BATCH):
    # generate a random composition of N into Z positive parts.
    idx = torch.randperm(SEQ_LEN - 1)[: BATCH - 1] + 1
    idx, _ = torch.sort(idx)
    breakpoints = torch.cat(
        [
            torch.tensor([0], dtype=torch.long),
            idx,
            torch.tensor([SEQ_LEN], dtype=torch.long),
        ]
    )
    seqlens = (breakpoints[1:] - breakpoints[:-1]).to(torch.int32)
    return seqlens


def generate_varlen_tensor(
    total_seqlen: int,
    num_heads: int,
    head_size: int,
    batch_size: Optional[int] = None,
    equal_seqlens: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    mode: Literal["random", "ones", "incremental", "identity"] = "random",
):
    if DEBUG:
        print("total_seqlen", total_seqlen)
        print("num_heads", num_heads)
        print("head_size", head_size)

    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        og_fp8_dtype = dtype
        dtype = torch.float32

    # get valid batch_size
    if batch_size is None:
        valid_batch_sizes = [
            bs for bs in [1, 2, 4, 8, 16, 32, 64] if bs <= total_seqlen
        ]
        batch_size = random.choice(valid_batch_sizes)

    # get seqlens
    if equal_seqlens:
        seqlens = torch.full(
            (batch_size,), total_seqlen // batch_size, dtype=torch.int32, device=device
        )
        seqlens[-1] += total_seqlen % batch_size
    else:
        seqlens = random_seqlens_composition(total_seqlen, batch_size).to(device=device)

    # create cumulative sequence lengths
    cu_seqlens = (
        torch.cat(
            [torch.tensor([0], dtype=torch.int32, device=device), seqlens.cumsum(dim=0)]
        )
        .to(torch.int32)
        .to(device=device)
    )
    max_seqlen = torch.max(seqlens).to(torch.int32).item()

    # create varlen tensor based on mode
    if mode == "incremental":
        x = torch.zeros(total_seqlen, num_heads, head_size, dtype=dtype, device=device)
        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            length = end - start

            x[start:end, :, :] = (
                torch.arange(length, dtype=dtype, device=device)
                .view(length, 1, 1)
                .expand(length, num_heads, head_size)
            )
    elif mode == "identity":
        x = torch.zeros(total_seqlen, num_heads, head_size, dtype=dtype, device=device)
        # for each batch, create identity pattern within that batch's sequence
        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            length = end - start

            # create identity pattern for positions within this batch
            for pos in range(min(length, head_size)):
                x[start + pos, :, pos] = 1.0
    elif mode == "random":
        x = torch.randn(
            (total_seqlen, num_heads, head_size), dtype=dtype, device=device
        )
    elif mode == "ones":
        x = torch.ones((total_seqlen, num_heads, head_size), dtype=dtype, device=device)
    else:
        raise ValueError(f"Unkown mode {mode}")

    if is_fp8_dtype:
        # cast to fp8
        x, descale_x = cast_to_fp8(
            x, og_fp8_dtype, "thd", cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )
        x.requires_grad_()
        return x, cu_seqlens, max_seqlen, descale_x
    else:
        x.requires_grad_()
        return x, cu_seqlens, max_seqlen


def generate_bshd_tensor(
    BATCH,
    SEQ_LEN,
    NUM_HEADS,
    D_HEAD,
    dtype: torch.dtype = torch.float16,
    device="cuda",
    mode: Literal["random", "ones", "incremental", "identity"] = "random",
):
    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        og_fp8_dtype = dtype
        dtype = torch.float32

    # gen tensor based on mode
    tensor_shape = (BATCH, SEQ_LEN, NUM_HEADS, D_HEAD)
    if mode == "incremental":
        x = (
            torch.arange(SEQ_LEN, dtype=dtype, device=device)
            .view(1, SEQ_LEN, 1, 1)
            .expand(*tensor_shape)
            .contiguous()
        )
    elif mode == "identity":
        x = torch.zeros(tensor_shape, dtype=dtype, device=device)
        # create identity pattern: position i has value 1 at dimension i
        for i in range(min(SEQ_LEN, D_HEAD)):
            x[:, i, :, i] = 1.0
    elif mode == "random":
        x = torch.randn(tensor_shape, dtype=dtype, device=device)
    elif mode == "ones":
        x = torch.ones(tensor_shape, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unkown mode {mode}")

    if is_fp8_dtype:
        # cast to fp8
        x, descale_x = cast_to_fp8(x, og_fp8_dtype, "bshd")
        x.requires_grad_()
        return x, descale_x
    else:
        x.requires_grad_()
        return x


def generate_bhsd_tensor(
    BATCH,
    NUM_HEADS,
    SEQ_LEN,
    D_HEAD,
    dtype: torch.dtype = torch.float16,
    device="cuda",
    mode: Literal["random", "ones", "incremental", "identity"] = "random",
):
    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        dtype = torch.float32

    # gen tensor based on mode
    tensor_shape = (BATCH, NUM_HEADS, SEQ_LEN, D_HEAD)
    if mode == "incremental":
        x = (
            torch.arange(SEQ_LEN, dtype=dtype, device=device)
            .view(1, 1, SEQ_LEN, 1)
            .expand(*tensor_shape)
            .contiguous()
        )
    elif mode == "identity":
        x = torch.zeros(tensor_shape, dtype=dtype, device=device)
        # create identity pattern: position i has value 1 at dimension i
        for i in range(min(SEQ_LEN, D_HEAD)):
            x[:, :, i, i] = 1.0
    elif mode == "random":
        x = torch.randn(tensor_shape, dtype=dtype, device=device)
    elif mode == "ones":
        x = torch.ones(tensor_shape, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unkown mode {mode}")

    if is_fp8_dtype:
        raise ValueError("fp8 not supported for bhsd yet")
    else:
        x.requires_grad_()
        return x


def generate_bshd_qkv_packed(
    BATCH,
    SEQ_LEN,
    NUM_HEADS,
    D_HEAD,
    dtype: torch.dtype = torch.float16,
    device="cuda",
    DEBUG_INPUT=False,
):
    """Generate QKV packed tensor with shape (BATCH, SEQ_LEN, 3, NUM_HEADS, D_HEAD)"""
    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        dtype = torch.float32

    # gen tensor
    tensor_shape = (BATCH, SEQ_LEN, 3, NUM_HEADS, D_HEAD)
    if DEBUG_INPUT:
        x = (
            torch.arange(SEQ_LEN, dtype=dtype, device=device)
            .view(1, SEQ_LEN, 1, 1, 1)
            .expand(*tensor_shape)
            .contiguous()
        )
    else:
        x = torch.randn(tensor_shape, dtype=dtype, device=device)

    if is_fp8_dtype:
        # cast to fp8 - need to handle the packed dimension
        raise NotImplementedError("FP8 not supported for QKV packing yet")
    else:
        x.requires_grad_()
        return x


def generate_bshd_kv_packed(
    BATCH,
    SEQ_LEN,
    NUM_HEADS,
    D_HEAD,
    dtype: torch.dtype = torch.float16,
    device="cuda",
    DEBUG_INPUT=False,
):
    """Generate KV packed tensor with shape (BATCH, SEQ_LEN, 2, NUM_HEADS, D_HEAD)"""
    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        dtype = torch.float32

    # gen tensor
    tensor_shape = (BATCH, SEQ_LEN, 2, NUM_HEADS, D_HEAD)
    if DEBUG_INPUT:
        x = (
            torch.arange(SEQ_LEN, dtype=dtype, device=device)
            .view(1, SEQ_LEN, 1, 1, 1)
            .expand(*tensor_shape)
            .contiguous()
        )
    else:
        x = torch.randn(tensor_shape, dtype=dtype, device=device)

    if is_fp8_dtype:
        # cast to fp8 - need to handle the packed dimension
        raise NotImplementedError("FP8 not supported for KV packing yet")
    else:
        x.requires_grad_()
        return x


def generate_bhsd_qkv_packed(
    BATCH,
    NUM_HEADS,
    SEQ_LEN,
    D_HEAD,
    dtype: torch.dtype = torch.float16,
    device="cuda",
    DEBUG_INPUT=False,
):
    """Generate QKV packed tensor with shape (BATCH, 3, NUM_HEADS, SEQ_LEN, D_HEAD)"""
    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        dtype = torch.float32

    # gen tensor
    tensor_shape = (BATCH, 3, NUM_HEADS, SEQ_LEN, D_HEAD)
    if DEBUG_INPUT:
        x = (
            torch.arange(SEQ_LEN, dtype=dtype, device=device)
            .view(1, 1, 1, SEQ_LEN, 1)
            .expand(*tensor_shape)
            .contiguous()
        )
    else:
        x = torch.randn(tensor_shape, dtype=dtype, device=device)

    if is_fp8_dtype:
        # cast to fp8 - need to handle the packed dimension
        raise NotImplementedError("FP8 not supported for QKV packing yet")
    else:
        x.requires_grad_()
        return x


def generate_bhsd_kv_packed(
    BATCH,
    NUM_HEADS,
    SEQ_LEN,
    D_HEAD,
    dtype: torch.dtype = torch.float16,
    device="cuda",
    DEBUG_INPUT=False,
):
    """Generate KV packed tensor with shape (BATCH, 2, NUM_HEADS, SEQ_LEN, D_HEAD)"""
    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        dtype = torch.float32

    # gen tensor
    tensor_shape = (BATCH, 2, NUM_HEADS, SEQ_LEN, D_HEAD)
    if DEBUG_INPUT:
        x = (
            torch.arange(SEQ_LEN, dtype=dtype, device=device)
            .view(1, 1, 1, SEQ_LEN, 1)
            .expand(*tensor_shape)
            .contiguous()
        )
    else:
        x = torch.randn(tensor_shape, dtype=dtype, device=device)

    if is_fp8_dtype:
        # cast to fp8 - need to handle the packed dimension
        raise NotImplementedError("FP8 not supported for KV packing yet")
    else:
        x.requires_grad_()
        return x


def generate_varlen_qkv_packed(
    total_seqlen: int,
    num_heads: int,
    head_size: int,
    batch_size: Optional[int] = None,
    equal_seqlens: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    DEBUG_INPUT: bool = False,
):
    """Generate varlen QKV packed tensor with shape (total_seqlen, 3, num_heads, head_size)"""
    if DEBUG:
        print("generate_varlen_qkv_packed")
        print("total_seqlen", total_seqlen)
        print("num_heads", num_heads)
        print("head_size", head_size)

    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        dtype = torch.float32

    # get valid batch_size
    if batch_size is None:
        valid_batch_sizes = [
            bs for bs in [1, 2, 4, 8, 16, 32, 64] if bs <= total_seqlen
        ]
        batch_size = random.choice(valid_batch_sizes)

    # get seqlens
    if equal_seqlens:
        seqlens = torch.full(
            (batch_size,), total_seqlen // batch_size, dtype=torch.int32, device=device
        )
        seqlens[-1] += total_seqlen % batch_size
    else:
        seqlens = random_seqlens_composition(total_seqlen, batch_size).to(device=device)

    # create cumulative sequence lengths
    cu_seqlens = (
        torch.cat(
            [torch.tensor([0], dtype=torch.int32, device=device), seqlens.cumsum(dim=0)]
        )
        .to(torch.int32)
        .to(device=device)
    )
    max_seqlen = torch.max(seqlens).to(torch.int32).item()

    # create varlen qkv packed tensor
    if DEBUG_INPUT:
        x = torch.zeros(
            total_seqlen, 3, num_heads, head_size, dtype=dtype, device=device
        )
        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            length = end - start

            x[start:end, :, :, :] = (
                torch.arange(length, dtype=dtype, device=device)
                .view(length, 1, 1, 1)
                .expand(length, 3, num_heads, head_size)
            )
    else:
        x = torch.randn(
            (total_seqlen, 3, num_heads, head_size), dtype=dtype, device=device
        )

    if is_fp8_dtype:
        # cast to fp8 - need to handle the packed dimension
        raise NotImplementedError("FP8 not supported for QKV packing yet")
    else:
        x.requires_grad_()
        return x, cu_seqlens, max_seqlen


def generate_varlen_kv_packed(
    total_seqlen: int,
    num_heads: int,
    head_size: int,
    batch_size: Optional[int] = None,
    equal_seqlens: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    DEBUG_INPUT: bool = False,
):
    """Generate varlen KV packed tensor with shape (total_seqlen, 2, num_heads, head_size)"""
    if DEBUG:
        print("generate_varlen_kv_packed")
        print("total_seqlen", total_seqlen)
        print("num_heads", num_heads)
        print("head_size", head_size)

    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        dtype = torch.float32

    # get valid batch_size
    if batch_size is None:
        valid_batch_sizes = [
            bs for bs in [1, 2, 4, 8, 16, 32, 64] if bs <= total_seqlen
        ]
        batch_size = random.choice(valid_batch_sizes)

    # get seqlens
    if equal_seqlens:
        seqlens = torch.full(
            (batch_size,), total_seqlen // batch_size, dtype=torch.int32, device=device
        )
        seqlens[-1] += total_seqlen % batch_size
    else:
        seqlens = random_seqlens_composition(total_seqlen, batch_size).to(device=device)

    # create cumulative sequence lengths
    cu_seqlens = (
        torch.cat(
            [torch.tensor([0], dtype=torch.int32, device=device), seqlens.cumsum(dim=0)]
        )
        .to(torch.int32)
        .to(device=device)
    )
    max_seqlen = torch.max(seqlens).to(torch.int32).item()

    # create varlen kv packed tensor
    if DEBUG_INPUT:
        x = torch.zeros(
            total_seqlen, 2, num_heads, head_size, dtype=dtype, device=device
        )
        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            length = end - start

            x[start:end, :, :, :] = (
                torch.arange(length, dtype=dtype, device=device)
                .view(length, 1, 1, 1)
                .expand(length, 2, num_heads, head_size)
            )
    else:
        x = torch.randn(
            (total_seqlen, 2, num_heads, head_size), dtype=dtype, device=device
        )

    if is_fp8_dtype:
        # cast to fp8 - need to handle the packed dimension
        raise NotImplementedError("FP8 not supported for KV packing yet")
    else:
        x.requires_grad_()
        return x, cu_seqlens, max_seqlen


# -------------------------------
# Alibi
# -------------------------------
@triton.jit
def compute_alibi_block(
    alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n, transpose=False
):
    # when seqlen_k and seqlen_q are different we want the diagonal to stick to the bottom right of the attention matrix
    # for casual mask we want something like this where (1 is kept and 0 is masked)
    # seqlen_q = 2 and seqlen_k = 5
    #   1 1 1 1 0
    #   1 1 1 1 1
    # seqlen_q = 5 and seqlen_k = 2
    #        0 0
    #        0 0
    #        0 0
    #        1 0
    #        1 1
    # for alibi the diagonal is 0 indicating no penalty for attending to that spot and increasing penalty for attending further from the diagonal
    # e.g. alibi_slope = 1, seqlen_q = 2, seqlen_k = 5, offs_m = [0, 1, 2, 3], offs_n = [0, 1, 2, 3, 4], transpose = False
    # 1. offs_m[:,None] = [[0],
    #                       [1],
    # 2. offs_m[:,None] + seqlen_k = [[5],
    #                                  [6],
    # 3. offs_m[:,None] + seqlen_k - seqlen_q = [[3],
    #                                             [4],
    # 4. offs_m[:,None] + seqlen_k - seqlen_q - offs_n[None,:] = [[3], - [[0, 1, 2, 3, 4]] =  [[ 3, 2, 1, 0,-1],
    #                                                            [4],                           [ 4, 3, 2, 1, 0]]
    # 5. -1 * alibi_slope * tl.abs(relative_pos_block) = [[ -3, -2, -1, 0,-1],
    #                                                     [ -4, -3, -2, -1, 0]],
    relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :]
    alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
    if transpose:
        return alibi_block.T
    else:
        return alibi_block


# -------------------------------
# FP8
# -------------------------------
def is_dtype_fp8(dtype) -> bool:
    supported = {
        torch.float8_e4m3fnuz,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    }
    if dtype not in supported:
        return False
    return True


_RECOMMENDED_FP8_REPLACEMENTS = {
    "gfx942": {
        torch.float8_e4m3fn: torch.float8_e4m3fnuz,
        torch.float8_e5m2: torch.float8_e5m2fnuz,
    },
}


def get_recommended_fp8_dtype(x):
    dtype = x.dtype if isinstance(x, torch.Tensor) else x
    if not is_dtype_fp8(dtype):
        return dtype
    arch = get_arch()
    return _RECOMMENDED_FP8_REPLACEMENTS.get(arch, {}).get(dtype, dtype)


def is_fp8(x) -> bool:
    """Return whether tensor(s) use FP8.

    Accepts either a single tensor or a list/tuple of tensors.

    Rules:
      * Single tensor: return True if FP8 (after arch validation), else False.
      * Multiple tensors:
          - If all tensors are FP8 -> return True.
          - If none are FP8 -> return False.
          - If a mix of FP8 and non-FP8 -> raise ValueError.

    Empty list/tuple returns False.
    """

    def _is_fp8_single(t: torch.Tensor) -> bool:
        if is_dtype_fp8(t.dtype):
            arch = get_arch()
            if arch not in ("gfx942", "gfx950"):
                raise RuntimeError(
                    f"{arch} is not in the list of supported architectures for FP8"
                )
            return True
        return False

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return False
        flags = [_is_fp8_single(t) for t in x]
        if all(flags):
            return True
        if not any(flags):
            return False
        raise ValueError(
            "Mixed FP8 and non-FP8 tensors provided; either all or none must be FP8."
        )
    else:
        return _is_fp8_single(x)


@triton.jit
def compute_fp8_scaling_factors(x, fp8_max: tl.constexpr):
    # compute fp8 scaling and descaling factor for a block
    x_amax = tl.max(tl.abs(x))  # NOTE: abs deals with negative values
    x_amax = tl.where(x_amax <= 1e-9, 1e-9, x_amax)
    scale_x = fp8_max / x_amax
    descale_x = x_amax / fp8_max
    return scale_x, descale_x


@triton.jit
def _cast_varlen_to_fp8_kernel_2d(
    X,
    X_fp8,
    Descale,
    cu_seqlens,
    H,
    MAX_SEQLEN,
    stride_batch,
    stride_seq,
    stride_head,
    stride_dim,
    stride_out_batch,
    stride_out_seq,
    stride_out_head,
    stride_out_dim,
    stride_desc_batch,
    stride_desc_head,
    FP8_CLAMP_VAL,
    FP8_MAX,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # Process one (batch, head) pair per kernel
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)

    # Get sequence bounds for this batch
    if IS_VARLEN:
        seq_start = tl.load(cu_seqlens + b_id)
        seq_end = tl.load(cu_seqlens + b_id + 1)
        seqlen = seq_end - seq_start
    else:
        seq_start = 0
        seqlen = MAX_SEQLEN

    # initialize max value tracker
    x_max_val = 0.0

    # STEP 1: Find max absolute value across the entire sequence
    num_of_blocks = tl.cdiv(seqlen, BLOCK_SIZE)
    for blk_idx in range(0, num_of_blocks):
        # print("blk_idx:", blk_idx)
        # offsets
        offs_seq = blk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_dim = tl.arange(0, HEAD_DIM)

        # Create mask for valid elements
        mask_seq = offs_seq[:, None] < seqlen
        if ACTUAL_HEAD_DIM != HEAD_DIM:
            mask_dim = offs_dim[None, :] < ACTUAL_HEAD_DIM
            mask_seq = mask_seq & mask_dim

        # Load block
        adj_x = (
            b_id * stride_batch
            + h_id * stride_head
            + seq_start * stride_seq
            + offs_seq[:, None] * stride_seq
            + offs_dim[None, :] * stride_dim
        )
        x_block = tl.load(X + adj_x, mask=mask_seq, other=0.0)
        # print("x_block:", x_block)

        # Find max absolute value in this block
        block_max = tl.max(tl.abs(x_block))
        # print("block_max:", block_max)

        # Update overall max
        x_max_val = tl.maximum(x_max_val, block_max)
        # print("x_max_val:", x_max_val)

    # clamp to avoid division by zero issues
    x_max_val = tl.maximum(x_max_val, FP8_CLAMP_VAL)

    # compute scale and descale factors for the entire sequence
    scale = FP8_MAX / x_max_val
    descale = x_max_val / FP8_MAX

    # store descale factor for this (batch, head) pair
    desc_ptr = Descale + b_id * stride_desc_batch + h_id  # * stride_desc_head
    tl.store(desc_ptr, descale)

    # STEP 2: Apply scaling to the entire sequence and convert to FP8
    for blk_idx in range(0, num_of_blocks):
        # offsets
        offs_seq = blk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_dim = tl.arange(0, HEAD_DIM)

        # Create mask for valid elements
        mask_seq = offs_seq[:, None] < seqlen
        if ACTUAL_HEAD_DIM != HEAD_DIM:
            mask_dim = offs_dim[None, :] < ACTUAL_HEAD_DIM
            mask_seq = mask_seq & mask_dim

        # Load block - Using the fixed addressing
        addr = (
            b_id * stride_batch
            + h_id * stride_head
            + seq_start * stride_seq
            + offs_seq[:, None] * stride_seq
            + offs_dim[None, :] * stride_dim
        )
        x_block = tl.load(X + addr, mask=mask_seq, other=0.0)

        # Apply scale and convert to FP8
        x_fp8_block = (x_block * scale).to(X_fp8.type.element_ty)

        # Store results
        addr_out = (
            b_id * stride_out_batch
            + h_id * stride_out_head
            + seq_start * stride_out_seq
            + offs_seq[:, None] * stride_out_seq
            + offs_dim[None, :] * stride_out_dim
        )
        tl.store(X_fp8 + addr_out, x_fp8_block, mask=mask_seq)


def cast_to_fp8(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    layout: Literal["bshd", "thd"],
    clamp_val: float = 1e-9,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Note: removed unreachable debug prints that were guarded by `if False:`.

    # check types are valid
    assert x.dtype in {
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    } and is_dtype_fp8(fp8_dtype), f"Cannot cast {x.dtype} to {fp8_dtype}"

    # extract dimensions
    batch, max_seqlen_final, num_heads, head_dim = get_shape_from_layout(
        x, layout, cu_seqlens, max_seqlen
    )
    is_varlen = layout == "thd"
    fp8_max = torch.finfo(fp8_dtype).max
    if False:
        print("batch:", batch)
        print("max_seqlen_final:", max_seqlen_final)
        print("num_heads:", num_heads)
        print("head_dim:", head_dim)

    # get closest power of 2 for head_dim
    padded_head_dim = 1 << (head_dim - 1).bit_length()
    padded_head_dim = max(padded_head_dim, 32)

    # kernel params
    x_fp8 = torch.zeros_like(x, dtype=fp8_dtype)
    descale_factors = torch.zeros(
        (batch, num_heads), device=x.device, dtype=torch.float32
    )
    BLOCK_SIZE = 128

    # calculate strides
    stride_batch, stride_head, stride_seq, stride_dim = get_stride_from_layout(
        x, layout
    )
    stride_out_batch, stride_out_head, stride_out_seq, stride_out_dim = (
        get_stride_from_layout(x_fp8, layout)
    )
    stride_desc_batch, stride_desc_head = descale_factors.stride()

    if False:
        print("stride_batch", stride_batch)
        print("stride_head", stride_head)
        print("stride_seq", stride_seq)
        print("stride_dim", stride_dim)
        print("stride_out_batch", stride_out_batch)
        print("stride_out_head", stride_out_head)
        print("stride_out_seq", stride_out_seq)
        print("stride_out_dim", stride_out_dim)
        print("stride_desc_batch", stride_desc_batch)
        print("stride_desc_head", stride_desc_head)

    grid = (batch, num_heads)
    _cast_varlen_to_fp8_kernel_2d[grid](
        x,
        x_fp8,
        descale_factors,
        cu_seqlens,
        num_heads,
        max_seqlen_final,
        stride_batch,
        stride_seq,
        stride_head,
        stride_dim,
        stride_out_batch,
        stride_out_seq,
        stride_out_head,
        stride_out_dim,
        stride_desc_batch,
        stride_desc_head,
        clamp_val,
        fp8_max,
        BLOCK_SIZE=BLOCK_SIZE,
        HEAD_DIM=padded_head_dim,
        ACTUAL_HEAD_DIM=head_dim,
        IS_VARLEN=is_varlen,
    )

    if False:
        print("x_fp8:", x_fp8, x_fp8.shape)
        print("descale_factors:", descale_factors, descale_factors.shape)
    return x_fp8, descale_factors


# -------------------------------
# Misc
# -------------------------------
def get_shape_from_layout(
    x: torch.Tensor,
    layout: Literal["bshd", "bhsd", "thd"],
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> tuple[int, int, int, int]:
    if layout == "bhsd":
        batch, num_heads, max_seqlen_final, head_dim = x.shape
    elif layout == "bshd":
        batch, max_seqlen_final, num_heads, head_dim = x.shape
    elif layout == "thd":
        total_seqlen, num_heads, head_dim = x.shape
        if cu_seqlens is None:
            raise ValueError("cu_seqlens must be provided for varlen (thd) layout")
        if max_seqlen is None:
            raise ValueError("max_seqlen must be provided for varlen (thd) layout")

        batch, max_seqlen_final, num_heads, head_dim = (
            len(cu_seqlens) - 1,
            max_seqlen,
            num_heads,
            head_dim,
        )
    else:
        assert False, "Got unsupported layout."

    return batch, max_seqlen_final, num_heads, head_dim


def get_shapes_from_layout(
    q,
    k,
    layout,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
):
    batch_q, seqlen_q, nheads_q, head_size_q = get_shape_from_layout(
        q, layout, cu_seqlens_q, max_seqlen_q
    )
    batch_k, seqlen_k, nheads_k, head_size_k = get_shape_from_layout(
        k, layout, cu_seqlens_k, max_seqlen_k
    )

    # assert
    assert batch_q == batch_k
    assert head_size_q == head_size_k

    return batch_q, nheads_q, nheads_k, head_size_q, seqlen_q, seqlen_k


def get_stride_from_layout(x: torch.Tensor, layout: Literal["bshd", "bhsd", "thd"]):
    if layout == "thd":
        strides = (0, x.stride(1), x.stride(0), x.stride(2))
    elif layout == "bhsd":
        strides = (x.stride(0), x.stride(1), x.stride(2), x.stride(3))
    elif layout == "bshd":
        strides = (x.stride(0), x.stride(2), x.stride(1), x.stride(3))
    else:
        assert False, "Got unsupported layout."
    return strides


def get_shape_and_strides_from_layout(
    x: torch.Tensor,
    layout: Literal["bshd", "bhsd", "thd"],
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    return get_shape_from_layout(
        x, layout, cu_seqlens, max_seqlen
    ), get_stride_from_layout(x, layout)


def get_strides_from_layout(q, k, v, o, layout):
    q_strides = get_stride_from_layout(q, layout)
    k_strides = get_stride_from_layout(k, layout)
    v_strides = get_stride_from_layout(v, layout)
    o_strides = get_stride_from_layout(o, layout)
    return q_strides, k_strides, v_strides, o_strides


def get_padded_headsize(size):
    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (size - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 16)
    return padded_d_model


def compute_alibi_tensor_ref(alibi_slopes, seqlen_q, seqlen_k):
    q_idx = torch.arange(seqlen_q, dtype=torch.int32, device="cuda").unsqueeze(
        -1
    )  # (N_CTX_Q, 1)
    k_idx = torch.arange(seqlen_k, dtype=torch.int32, device="cuda").unsqueeze(
        0
    )  # (1, N_CTX_K)
    relative_pos = torch.abs(q_idx + seqlen_k - seqlen_q - k_idx)  # (N_CTX_Q, N_CTX_K)
    return (
        -1 * alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos
    )  # (Z, H, N_CTX_Q, N_CTX_K)


def round_multiple(x, m):
    return (x + m - 1) // m * m


def save_tensor_to_csv(tensor, filename, decimal_places=2):
    """
    save a 2d tensor to csv file

    args:
        tensor: torch tensor of shape [rows, cols]
        filename: output csv filename
        decimal_places: number of decimal places (default: 2)
    """
    # ensure tensor is 2d
    if tensor.ndim != 2:
        raise ValueError(f"tensor must be 2d, got shape {tensor.shape}")

    # ensure filename ends with .csv
    if not filename.endswith(".csv"):
        filename = filename + ".csv"

    # save to csv using numpy
    np.savetxt(
        filename,
        tensor.detach().cpu().numpy(),
        delimiter=",",
        fmt=f"%.{decimal_places}f",
    )


# -------------------------------
# Dropouts
# -------------------------------
def create_dropout_mask(dropout_p, shape, seed):
    device = "cuda"
    rand_vals = torch.rand(
        shape,
        generator=torch.Generator(device=device).manual_seed(seed),
        device=device,
        dtype=torch.float32,
    )
    return rand_vals > dropout_p


def create_dropout_mask_varlen(
    dropout_p, batch, nheads_q, cu_seqlens_q, cu_seqlens_k, philox_seed
):
    device = "cuda"
    qlens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    klens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    max_qlen = qlens.max()
    max_klen = klens.max()
    dropout_mask = torch.zeros((batch, nheads_q, max_qlen, max_klen), device=device)
    for b in range(batch):
        qlen = qlens[b]
        klen = klens[b]
        rand_vals = torch.rand(
            (nheads_q, qlen, klen),
            generator=torch.Generator(device=device).manual_seed(philox_seed),
            device=device,
            dtype=torch.float32,
        )
        submask = rand_vals > dropout_p
        dropout_mask[b, :, :qlen, :klen] = submask

    return dropout_mask


def write_dropout_mask(x, tensor_name="tensor"):
    batch, head, seqlen_m, seqlen_n = x.shape
    x = x.tolist()

    with open(f"{tensor_name}.csv", "w") as f:
        writer = csv.writer(f)
        for b in range(batch):
            for h in range(head):
                dropout_mask = x[b][h]
                if True:
                    BLOCK_M = 64
                    BLOCK_N = 64

                    # Calculate number of blocks in each dimension
                    m_blocks = math.ceil(seqlen_m / BLOCK_M)
                    n_blocks = math.ceil(seqlen_n / BLOCK_N)

                    # Process each block
                    for m_block in range(m_blocks):
                        # Calculate row range for current block
                        row_start = m_block * BLOCK_M
                        row_end = min(row_start + BLOCK_M, seqlen_m)

                        for n_block in range(n_blocks):
                            # Calculate column range for current block
                            col_start = n_block * BLOCK_N
                            col_end = min(col_start + BLOCK_N, seqlen_n)

                            # Extract and write the current block
                            for row_idx in range(row_start, row_end):
                                row_data = dropout_mask[row_idx][col_start:col_end]
                                writer.writerow(row_data)
                else:
                    writer.writerows(dropout_mask)


# -------------------------------
# Rotary
# -------------------------------
@triton.jit
def _rotary_kernel(
    OUT,
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,
    seqlen,
    nheads,
    seqlen_ro,
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    ROTARY_DIM: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    BLOCK_K: tl.constexpr = triton.next_power_of_2(ROTARY_DIM)
    ROTARY_DIM_HALF = ROTARY_DIM // 2
    pid_head = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch
        OUT = OUT + pid_batch * stride_out_batch
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen
        OUT = OUT + start_idx * stride_out_seqlen

    if pid_m * BLOCK_M >= seqlen:
        return

    rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)

    rk_half = tl.arange(0, BLOCK_K // 2)
    COS = COS + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
    SIN = SIN + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
    mask_cs = (rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < ROTARY_DIM_HALF)
    cos = tl.load(COS, mask=mask_cs, other=1.0).to(tl.float32)
    sin = tl.load(SIN, mask=mask_cs, other=0.0).to(tl.float32)
    if CONJUGATE:
        sin = -sin

    if not INTERLEAVED:
        X = X + (
            rh[:, None, None] * stride_x_nheads
            + rm[None, :, None] * stride_x_seqlen
            + rk_half[None, None, :] * stride_x_headdim
        )
        OUT = OUT + (
            rh[:, None, None] * stride_out_nheads
            + rm[None, :, None] * stride_out_seqlen
            + rk_half[None, None, :] * stride_out_headdim
        )
        mask = (
            (rh[:, None, None] < nheads)
            & (rm[None, :, None] < seqlen)
            & (rk_half[None, None, :] < ROTARY_DIM_HALF)
        )
        x0 = tl.load(X, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(X + ROTARY_DIM_HALF * stride_x_headdim, mask=mask, other=0.0).to(
            tl.float32
        )
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        tl.store(OUT, o0, mask=mask)
        tl.store(OUT + ROTARY_DIM_HALF * stride_out_headdim, o1, mask=mask)
    else:
        rk = tl.arange(0, BLOCK_K)
        X = X + (
            rh[:, None, None] * stride_x_nheads
            + rm[None, :, None] * stride_x_seqlen
            + rk[None, None, :] * stride_x_headdim
        )
        OUT = OUT + (
            rh[:, None, None] * stride_out_nheads
            + rm[None, :, None] * stride_out_seqlen
            + rk[None, None, :] * stride_out_headdim
        )
        mask = (
            (rh[:, None, None] < nheads)
            & (rm[None, :, None] < seqlen)
            & (rk[None, None, :] < ROTARY_DIM)
        )
        x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
        x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, BLOCK_K // 2, 2]))
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        o = tl.reshape(tl.join(o0, o1), [BLOCK_H, BLOCK_M, BLOCK_K])
        tl.store(OUT, o, mask=mask)


def _apply_rotary_kernel(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert (
            max_seqlen is not None
        ), "If cu_seqlens is passed, max_seqlen must also be provided"
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim_half = cos.shape
    assert sin.shape == cos.shape
    rotary_dim = 2 * rotary_dim_half
    assert rotary_dim <= headdim
    assert headdim <= 256
    assert seqlen_ro >= seqlen

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in (torch.int32, torch.int64)
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    out = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        out[..., rotary_dim:].copy_(x[..., rotary_dim:])

    # Block heuristics
    BLOCK_M = 8 if rotary_dim <= 128 else 4
    grid = (
        triton.cdiv(nheads, 2),
        triton.cdiv(seqlen, BLOCK_M),
        batch,
    )

    with torch.cuda.device(x.device.index):
        torch.library.wrap_triton(_rotary_kernel)[grid](
            out,
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen_offsets,
            seqlen,
            nheads,
            seqlen_ro,
            out.stride(0) if not is_varlen else 0,
            out.stride(-3),
            out.stride(-2),
            out.stride(-1),
            x.stride(0) if not is_varlen else 0,
            x.stride(-3),
            x.stride(-2),
            x.stride(-1),
            rotary_dim,
            isinstance(seqlen_offsets, torch.Tensor),
            is_varlen,
            interleaved,
            conjugate,
            BLOCK_M=BLOCK_M,
            BLOCK_H=2,
        )
    return out


class _ApplyRotary(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        interleaved: bool,
        inplace: bool,
        seqlen_offsets: Union[int, torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        max_seqlen: Optional[int],
    ):
        out = _apply_rotary_kernel(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
            conjugate=False,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cu_seqlens)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        dx = _apply_rotary_kernel(
            do,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=ctx.inplace,
            conjugate=True,
        )
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    inplace: bool = False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> torch.Tensor:
    """Public API: apply rotary embeddings to tensor x.

    Args:
        x: (B, S, H, D) if `cu_seqlens` is None else (total_S, H, D).
        cos, sin: (S_rotary, rotary_dim/2)
        interleaved: GPT-J style if True.
        inplace: modify x in place (saves memory if rotary_dim == D).
        seqlen_offsets: int or (B,) tensor of starting offsets per sequence (KV cache decode).
        cu_seqlens: (B+1,) tensor enabling varlen mode.
        max_seqlen: required when `cu_seqlens` is provided.
    """
    # FP8 path: upcast to bfloat16 (preferred) or float16 for rotary math to avoid excessive error
    original_dtype = x.dtype
    is_fp8_input = original_dtype == getattr(torch, "float8_e4m3fn", None)
    if is_fp8_input:
        # Choose bf16 if available in cos.dtype path; otherwise fallback to float16
        target_dtype = (
            torch.bfloat16
            if cos.dtype == torch.bfloat16 or torch.cuda.is_bf16_supported()
            else torch.float16
        )
        # Upcast x, cos, sin for computation (without modifying originals in-place)
        x_up = x.to(target_dtype)
        cos_up = cos.to(target_dtype) if cos.dtype != target_dtype else cos
        sin_up = sin.to(target_dtype) if sin.dtype != target_dtype else sin
        out_up = _ApplyRotary.apply(
            x_up,
            cos_up,
            sin_up,
            interleaved,
            False,
            seqlen_offsets,
            cu_seqlens,
            max_seqlen,
        )
        # Cast result back to original fp8 dtype
        if inplace:
            x.copy_(out_up.to(original_dtype))
            return x
        return out_up.to(original_dtype)
    else:
        return _ApplyRotary.apply(
            x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
        )


def apply_rotary(
    q: torch.Tensor,
    k_new: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    causal: bool,
    local: bool,
    interleaved: bool = False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """High-level rotary application used by AMD prefill & decode paths.

    Policy (matches test reference & legacy semantics):
      - If causal OR local attention ⇒ apply rotary directly on (B, S, H, D).
      - Else (non-causal global) ⇒ flatten heads into sequence: (B, 1, S*H, D),
        apply rotary once, then unflatten back.
      - k_new (incremental KV slice) is always rotated directly when provided.

    Args:
        q: (B, S, H, D)
        k_new: Optional (B, S_k, H_k, D)
        cos, sin: rotary caches (S_rotary, rotary_dim/2)
        causal: causal attention flag
        local: sliding-window / local attention flag (pre-computed outside)
        interleaved: GPT-J style rotary layout
        seqlen_offsets: int or (B,) tensor of per-sequence start offsets
    Returns:
        (q_rot, k_new_rot)
    """
    assert q.ndim == 4, f"Expected q shape (B,S,H,D), got {q.shape}"
    B, S, H, D = q.shape
    use_flatten = (not causal) and (not local)

    if use_flatten:
        # Flatten (S,H) -> (S*H) with an added singleton dim to preserve expected 4D shape.
        q_flat = q.reshape(B, S * H, D).unsqueeze(1)  # (B, 1, S*H, D)
        q_flat = apply_rotary_emb(
            q_flat,
            cos,
            sin,
            interleaved=interleaved,
            seqlen_offsets=seqlen_offsets,
        )
        # Restore shape back to (B, S, H, D)
        q = q_flat.view(B, 1, S * H, D).reshape(B, S, H, D)
    else:
        q = apply_rotary_emb(
            q,
            cos,
            sin,
            interleaved=interleaved,
            seqlen_offsets=seqlen_offsets,
        )

    if k_new is not None:
        k_new = apply_rotary_emb(
            k_new,
            cos,
            sin,
            interleaved=interleaved,
            seqlen_offsets=seqlen_offsets,
        )
    return q, k_new


# -------------------------------
# Runtime info
# -------------------------------
@functools.cache
def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@functools.cache
def get_arch():
    return triton.runtime.driver.active.get_current_target().arch


@functools.cache
def get_cu_count():
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


@functools.cache
def is_cdna():
    return is_hip() and get_arch() in (
        "gfx908",
        "gfx90a",
        "gfx940",
        "gfx941",
        "gfx942",
        "gfx950",
    )


@functools.cache
def is_rdna():
    return is_hip() and get_arch() in (
        "gfx1030",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1200",
        "gfx1201",
    )

@triton.jit
def _get_max_quant_val(dtype: tl.constexpr):
    if dtype == tl.uint8:
        return 6.0
    elif dtype == tl.float8e5:
        return 57344.0
    elif dtype == tl.float8e4nv:
        return 448.0
    else:
        tl.static_assert(False, f"Invalid {dtype=}")


@triton.jit
def _compute_mx_quant_and_scale(
    src_tensor,
    valid_src_mask,
    mx_tensor_dtype: tl.constexpr,
    DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr = 0,
):
    is_fp8: tl.constexpr = (
        mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    )
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // 32

    # Explicit cast to fp32 since most ops are not supported on bfloat16. We avoid needless conversions to and from bf16
    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(
        valid_src_mask, abs_tensor, -1.0
    )  # Don't consider padding tensors in scale computation
    abs_tensor = tl.reshape(
        abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32]
    )
    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)
    dequant_scale = max_val / _get_max_quant_val(mx_tensor_dtype)
    if DEQUANT_SCALE_ROUNDING_MODE == 0:
        # DequantScaleRoundingMode.ROUND_UP
        # compute 2 ** ceil(log2(dequant_scale))
        # Adding 0x007FFFFF adds exponent by 1 unless mantissa is all zeros
        # A corner case: exponent is 0xFF that will overflow but that's already
        # NaN so assume we don't care.
        dequant_scale_exponent = (
            dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF
        ) & 0x7F800000
    else:
        # DequantScaleRoundingMode.ROUND_DOWN
        # compute 2 ** floor(log2(dequant_scale))
        assert DEQUANT_SCALE_ROUNDING_MODE == 1
        dequant_scale_exponent = dequant_scale.to(tl.uint32, bitcast=True) & 0x7F800000
    dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
    quant_scale = tl.where(dequant_scale_rounded == 0, 0, 1.0 / dequant_scale_rounded)

    f32_tensor = tl.reshape(
        f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32]
    )
    quant_tensor = f32_tensor * quant_scale

    # Reshape the tensors after scaling
    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    # Set the invalid portions of the tensor to 0. This will ensure that any padding tensors are 0 in the mx format.
    quant_tensor = tl.where(valid_src_mask, quant_tensor, 0)
    dequant_scale_exponent = dequant_scale_exponent.reshape(
        [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE]
    )

    # First, we simply extract the exponent part of the scales and store the result
    # dequant_scale = dequant_scale_exponent
    dequant_scale_exponent = (dequant_scale_exponent >> 23).to(tl.uint8)
    # Now we must convert the tensors to the mx format.
    if is_fp8:
        out_tensor = quant_tensor.to(mx_tensor_dtype)
    else:
        quant_tensor = quant_tensor.to(tl.uint32, bitcast=True)
        signs = quant_tensor & 0x80000000
        exponents = (quant_tensor >> 23) & 0xFF
        mantissas = quant_tensor & 0x7FFFFF

        # 0.25 <= x < 0.75 maps to 0.5, a denormal number
        E8_BIAS = 127
        E2_BIAS = 1
        # Move implicit bit 1 at the beginning to mantissa for denormals
        adjusted_exponents = tl.core.sub(
            E8_BIAS, exponents + 1, sanitize_overflow=False
        )
        mantissas = tl.where(
            exponents < E8_BIAS,
            (0x400000 | (mantissas >> 1)) >> adjusted_exponents,
            mantissas,
        )

        # For normal numbers, we change the bias from 127 to 1, and for subnormals, we keep exponent as 0.
        exponents = tl.maximum(exponents, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

        # Combine sign, exponent, and mantissa, while saturating
        # rounding nearest with tie breaking up by adding +1 to one bit right of the LSB, then shift right
        e2m1_tmp = tl.minimum((((exponents << 2) | (mantissas >> 21)) + 1) >> 1, 0x7)
        e2m1_value = ((signs >> 28) | e2m1_tmp).to(tl.uint8)

        e2m1_value = tl.reshape(
            e2m1_value, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2]
        )
        evens, odds = tl.split(e2m1_value)
        out_tensor = evens | (odds << 4)


    fp32_dequant_scale = (1.0/quant_scale).reshape(
        [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE]
    )

    return out_tensor, dequant_scale_exponent, fp32_dequant_scale


@triton.jit
def _downcast_to_mxfp(
    mx_tensor_ptr,
    stride_mxt_outer,
    stride_mxt_quant: tl.constexpr,
    mx_scale_ptr,
    fp32_scale_ptr,
    stride_mx_scale_outer,
    stride_mx_scale_quant,
    stride_fp32_scale_outer,
    stride_fp32_scale_quant,
    src_ptr,
    stride_src_outer,
    stride_src_quant,
    outer_dim,
    quant_dim,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr,
):

    tl.static_assert(
        stride_mxt_quant == 1, f"Output stride, {stride_mxt_quant=} must be 1."
    )
    tl.static_assert(
        BLOCK_SIZE_QUANT_DIM % 32 == 0,
        f"{BLOCK_SIZE_QUANT_DIM=} must be a multiple of 32",
    )

    # uint8 signifies two fp4 e2m1 values packed into a single byte
    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    tl.static_assert(
        mx_tensor_dtype == tl.uint8
        or (mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5),
        f"Invalid {mx_tensor_dtype=}. Must be uint8 or float8.",
    )

    src_dtype: tl.constexpr = src_ptr.dtype.element_ty
    tl.static_assert(
        mx_scale_ptr.dtype.element_ty == tl.uint8,
        f"{mx_scale_ptr.dtype.element_ty=} must be uint8",
    )
    tl.static_assert(
        (src_dtype == tl.bfloat16) or (src_dtype == tl.float16),
        f"{src_dtype=} must be bfloat16 or float16",
    )
    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    K_DIVISOR: tl.constexpr = 2 if is_fp4 else 1
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // 32
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    start_src_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_mx_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    src_ptr += start_src_quant * stride_src_quant + start_out * stride_src_outer
    mx_scale_ptr += (
        start_mx_scale_quant * stride_mx_scale_quant + start_out * stride_mx_scale_outer
    )
    fp32_scale_ptr += (
        start_mx_scale_quant * stride_fp32_scale_quant
        + start_out * stride_fp32_scale_outer
    )


    mx_tensor_ptr += start_mx_quant * stride_mxt_quant + start_out * stride_mxt_outer

    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_mxt_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_scale_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)

    mask_src_quant = start_src_quant + offs_src_quant < quant_dim
    mask_n = start_out + offs_outer < outer_dim
    full_mask_src = mask_src_quant & mask_n

    mask_mxt_quant = start_mx_quant + offs_mxt_quant < tl.cdiv(quant_dim, K_DIVISOR)
    full_mask_mxt = mask_mxt_quant & mask_n

    scale_mask_k = start_mx_scale_quant + offs_scale_quant < tl.cdiv(quant_dim, 32)
    full_scale_mask = scale_mask_k & mask_n

    src_tensor_offsets = (
        offs_src_quant * stride_src_quant + offs_outer * stride_src_outer
    )
    mx_scale_offsets = (
        offs_scale_quant * stride_mx_scale_quant + offs_outer * stride_mx_scale_outer
    )
    fp32_scale_offsets = (
        offs_scale_quant * stride_fp32_scale_quant + offs_outer * stride_fp32_scale_outer
    )

    mx_tensor_offsets = (
        offs_mxt_quant * stride_mxt_quant + offs_outer * stride_mxt_outer
    )
    src_tensor = tl.load(src_ptr + src_tensor_offsets, mask=full_mask_src)

    out_tensor, scale_tensor, scale_tensor_fp32 = _compute_mx_quant_and_scale(
        src_tensor, full_mask_src, mx_tensor_dtype, DEQUANT_SCALE_ROUNDING_MODE
    )

    tl.store(fp32_scale_ptr + fp32_scale_offsets, scale_tensor_fp32, mask=full_scale_mask)

    tl.store(mx_scale_ptr + mx_scale_offsets, scale_tensor, mask=full_scale_mask)
    tl.store(mx_tensor_ptr + mx_tensor_offsets, out_tensor, mask=full_mask_mxt)

class DequantScaleRoundingMode(Enum):
    ROUND_UP = 0
    ROUND_DOWN = 1

def downcast_to_mxfp(
    src_tensor: torch.Tensor,
    out_quant_type: torch.dtype,
    axis: int,
    DEQUANT_SCALE_ROUNDING_MODE: DequantScaleRoundingMode = DequantScaleRoundingMode.ROUND_UP,
):
    """
    Convert the src weights to mx format. The src weight is quantized along the axis dimension.

    If weight_quant_type is torch.uint8, we output mxfp4 where two e2m1 values are packed into a single byte.
    Note that this means the k_dim of the tensor will be half of the logical k_dim.

    If weight_quant_type is torch.float8_e4m3fn or torch.float8_e5m2, we output mxfp8 with the float8s are stored
    in their respective formats.
    """
    ndim = src_tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim
    # downcast
    src_tensor = src_tensor.transpose(axis, src_tensor.ndim - 1)
    is_fp4 = out_quant_type == torch.uint8
    is_fp8 = out_quant_type in (torch.float8_e4m3fn, torch.float8_e5m2)
    assert is_fp4 or is_fp8
    divisor = 2 if is_fp4 else 1
    L = src_tensor.shape[-1]
    if is_fp4:
        assert L % 2 == 0, f"axis dim must be divisible by 2 for e2m1. Got {L}"
    out_shape = src_tensor.shape[:-1] + (L // divisor,)
    out_scale_shape = src_tensor.shape[:-1] + (triton.cdiv(L, 32),)

    out_quant_tensor = src_tensor.new_empty(out_shape, dtype=out_quant_type)
    out_scale = src_tensor.new_empty(out_scale_shape, dtype=torch.uint8)
    
 
    out_scale_fp32 = src_tensor.new_empty(out_scale_shape, dtype=torch.float32)


    kernel_src_tensor = src_tensor.reshape(-1, src_tensor.shape[-1])
    kernel_quant_tensor = out_quant_tensor.view(-1, out_quant_tensor.shape[-1])
    kernel_scale = out_scale.view(-1, out_scale.shape[-1])
    
   
    kernel_scale_fp32 = out_scale_fp32.view(-1, out_scale_fp32.shape[-1])


    BLOCK_OUT_DIM = 128
    BLOCK_QUANT_DIM = 32
    grid_out = triton.cdiv(kernel_src_tensor.shape[0], BLOCK_OUT_DIM)
    grid_quant = triton.cdiv(kernel_src_tensor.shape[1], BLOCK_QUANT_DIM)

    _downcast_to_mxfp[(grid_out, grid_quant)](
        kernel_quant_tensor,
        *kernel_quant_tensor.stride(),
        kernel_scale,
        kernel_scale_fp32,
        *kernel_scale.stride(),
        *kernel_scale_fp32.stride(),
        kernel_src_tensor,
        *kernel_src_tensor.stride(),
        *kernel_src_tensor.shape,
        BLOCK_OUT_DIM,
        BLOCK_QUANT_DIM,
        DEQUANT_SCALE_ROUNDING_MODE.value,
        num_warps=8,
    )

    out_quant_tensor = out_quant_tensor.transpose(axis, src_tensor.ndim - 1)
    out_scale = out_scale.transpose(axis, src_tensor.ndim - 1)
    out_scale_fp32 = out_scale_fp32.transpose(axis, src_tensor.ndim - 1)

    return out_quant_tensor, out_scale, out_scale_fp32


def unpack_fp4_to_fp32(uint8_tensor):
    """
    Unpack uint8 tensor containing packed e2m1 fp4 values into fp32.
    Each uint8 contains two 4-bit e2m1 values (2-bit exponent, 1-bit mantissa).

    Args:
        uint8_tensor: torch.Tensor with dtype uint8, shape [..., D]

    Returns:
        torch.Tensor with dtype float32, shape [..., 2*D]
    """
    # Move to CPU for processing
    uint8_np = uint8_tensor.detach().cpu().numpy()
    original_shape = uint8_np.shape

    # Flatten to process
    uint8_flat = uint8_np.flatten()

    # Extract two 4-bit values from each uint8
    # Lower 4 bits
    low_nibble = uint8_flat & 0x0F
    # Upper 4 bits
    high_nibble = (uint8_flat >> 4) & 0x0F

    # Interleave them to maintain order
    fp4_values = np.empty(len(uint8_flat) * 2, dtype=np.uint8)
    fp4_values[0::2] = low_nibble
    fp4_values[1::2] = high_nibble

    # Convert e2m1 fp4 to fp32
    # e2m1 format: [sign:1][exp:2][mantissa:1]
    sign = ((fp4_values >> 3) & 0x1).astype(np.float32)
    exp = ((fp4_values >> 1) & 0x3).astype(np.int32)
    mantissa = (fp4_values & 0x1).astype(np.float32)

    # Convert to float
    # For e2m1: value = (-1)^sign * 2^(exp-1) * (1 + mantissa * 0.5)
    # The mantissa bit represents 0.5, so mantissa=0 → 1.0, mantissa=1 → 1.5
    # Special cases: exp=0 means subnormal or zero
    fp32_values = np.zeros_like(sign, dtype=np.float32)

    # Normal numbers (exp != 0)
    normal_mask = exp != 0
    fp32_values[normal_mask] = (1 - 2 * sign[normal_mask]) * np.power(2.0, exp[normal_mask] - 1) * (1 + mantissa[normal_mask] * 0.5)

    # Subnormal numbers (exp == 0, mantissa != 0)
    subnormal_mask = (exp == 0) & (mantissa != 0)
    fp32_values[subnormal_mask] = (1 - 2 * sign[subnormal_mask]) * np.power(2.0, -1) * (mantissa[subnormal_mask] * 0.5)

    # Reshape to [..., 2*D]
    new_shape = list(original_shape)
    new_shape[-1] = new_shape[-1] * 2
    fp32_values = fp32_values.reshape(new_shape)

    return torch.from_numpy(fp32_values).to(uint8_tensor.device)


def accuracy_analysis(q_fp4, k_fp4, q_scale_e8m0, k_scale_e8m0, q_scale_fp32, k_scale_fp32, q_bf16, k_bf16):

    # Unpack FP4 tensors (two fp4 values packed into one fp8) to FP32
    q_fp4_unpacked = unpack_fp4_to_fp32(q_fp4)
    k_fp4_unpacked = unpack_fp4_to_fp32(k_fp4)

    # Convert e8m0 scales (stored as uint8) to fp32 scale values
    q_scale_fp32_expanded = torch.pow(2.0, q_scale_e8m0.to(torch.float32) - 127.0)
    k_scale_fp32_expanded = torch.pow(2.0, k_scale_e8m0.to(torch.float32) - 127.0)

    ref = torch.einsum("bhsd,bhtd->bhst", q_bf16.to(torch.float32), k_bf16.to(torch.float32))
    print("q_fp4_unpacked.max()", q_fp4_unpacked.max().item())
    print("k_fp4_unpacked.max()", k_fp4_unpacked.max().item())

    print("q_bf16 max", q_bf16.max().item())
    print("(q_fp4_unpacked * q_scale_fp32_expanded.repeat_interleave(32,dim=-1)) max", (q_fp4_unpacked * q_scale_fp32_expanded.repeat_interleave(32,dim=-1)).max().item())
    print("(q_fp4_unpacked * q_scale_fp32.repeat_interleave(32,dim=-1)) max", (q_fp4_unpacked * q_scale_fp32.repeat_interleave(32,dim=-1)).max().item())
    q_error = torch.mean(torch.abs(q_bf16.to(torch.float32) - (q_fp4_unpacked * q_scale_fp32_expanded.repeat_interleave(32,dim=-1))))
    print(f"Q e8m0 descale error: {q_error.item()}")
    q_error_fp32_scale = torch.mean(torch.abs(q_bf16.to(torch.float32) - (q_fp4_unpacked * q_scale_fp32.repeat_interleave(32,dim=-1))))
    print(f"Q FP32 descale error: {q_error_fp32_scale.item()}")


    # Need to do the matmul in the quantized groups
    group_size = q_fp4_unpacked.shape[-1] // q_scale_e8m0.shape[-1]
    num_groups = q_scale_e8m0.shape[-1]
    
    quantized = torch.zeros_like(ref)
    
    for group_idx in range(num_groups):
        start_idx = group_idx * group_size
        end_idx = (group_idx + 1) * group_size
        # Extract group slices
        q_group = q_fp4_unpacked[:, :, :, start_idx:end_idx]
        k_group = k_fp4_unpacked[:, :, :, start_idx:end_idx]
        q_scale_group = q_scale_fp32_expanded[:, :, :, group_idx:group_idx+1]
        k_scale_group = k_scale_fp32_expanded[:, :, :, group_idx:group_idx+1]
        group_descale = (q_scale_group * k_scale_group.permute(0, 1, 3, 2))

        # Compute grouped matmul with broadcasting
        group_result = torch.einsum(
            "bhsg,bhtg->bhst",
            q_group.to(torch.float32),
            k_group.to(torch.float32),
        )
        group_result = group_result * group_descale
        quantized += group_result
    
    # print("Sage V2 Quantization Debug Info:")
    print(f"Q*K (e8m0 descale) error: {torch.mean(torch.abs(ref - quantized)).item()}")

    quantized_fp32_scale = torch.zeros_like(ref)
    
    for group_idx in range(num_groups):
        start_idx = group_idx * group_size
        end_idx = (group_idx + 1) * group_size
        # Extract group slices
        q_group = q_fp4_unpacked[:, :, :, start_idx:end_idx]
        k_group = k_fp4_unpacked[:, :, :, start_idx:end_idx]
        q_scale_group = q_scale_fp32[:, :, :, group_idx:group_idx+1]
        k_scale_group = k_scale_fp32[:, :, :, group_idx:group_idx+1]
        group_descale = (q_scale_group * k_scale_group.permute(0, 1, 3, 2))
        # Compute grouped matmul with broadcasting
        group_result = torch.einsum(
            "bhsg,bhtg->bhst",
            q_group.to(torch.float32),
            k_group.to(torch.float32),
        )
        group_result = group_result * group_descale
        quantized_fp32_scale += group_result

    print(f"Q*K (FP32 descale) error: {torch.mean(torch.abs(ref - quantized_fp32_scale)).item()}")
