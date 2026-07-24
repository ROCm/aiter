# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pure-Python launcher for the gfx1250 v4 MLA decode ``.co`` kernel.

This is the Python peer of ``csrc/py_itfs_cu/asm_mla_v4.cu`` for **gfx1250
only**. It reproduces that dispatcher's gfx1250 "preload" path (the compact
120-byte DIRECT_PARAM kernarg ABI) without the C++/ctypes-FFI host bridge:
the shipped ``.co`` is loaded and launched directly from Python through the
generic :mod:`aiter.ops.asm.asm_utils` helpers.

Scope is intentionally narrow — it is a thin wrapper, exactly like the .cu:
  * resolve the kernel binary via the shipped ``hsa/gfx1250/mla_v4`` registry
    (``mla_v4_asm.csv``), using the same lookup keys as the C++ dispatcher;
  * pack the 120-byte preload kernarg;
  * compute the same launch geometry;
  * launch.

It does NOT handle gfx950 (that arch uses the legacy 21-slot kernarg and keeps
going through the C++ dispatcher ``aiter.mla_decode_v4_asm``) and it does NOT
touch v3. The public entry :func:`mla_decode_v4_asm_gfx1250` mirrors the
signature of ``aiter.mla_decode_v4_asm`` so callers can swap between the two.
"""

import ctypes
import math
import os

import torch

from aiter.jit.core import get_asm_dir
from aiter.ops.asm.asm_utils import (
    dtype_str,
    get_function,
    get_warp_size,
    launch_co,
    load_asm_cfg_csv,
)

# kV4DimNope + kV4DimRope = 448 + 64 = 512. The kernel hardcodes 1/sqrt(512)
# as its softmax pre-scale, independent of head_size (mirror asm_mla_v4.cu).
_KV4_DIM_NOPE = 448
_KV4_DIM_ROPE = 64

_MLA_V4_SUBDIR = "mla_v4"
_MLA_V4_CSV = "mla_v4_asm.csv"


class MlaV4KernelArgsPreload(ctypes.Structure):
    """120-byte compact preload kernarg (gfx1250 DIRECT_PARAM=1 ABI).

    Byte-for-byte identical to ``MlaV4KernelArgsPreload`` in asm_mla_v4.cu
    (the ``#if EN_MLA_V4_KERNARG_PRELOAD`` struct). Offsets are annotated to keep
    the two definitions in lock-step.
    """

    _pack_ = 1
    _fields_ = [
        ("ptr_R", ctypes.c_void_p),  # 0x00 splitData (logits) FP32 (rw)
        ("ptr_Q", ctypes.c_void_p),  # 0x08 Q packed FP8 + e8m0 scale
        ("ptr_KV", ctypes.c_void_p),  # 0x10 KV packed FP8
        ("ptr_LTP", ctypes.c_void_p),  # 0x18 kv_indptr
        ("ptr_LTL", ctypes.c_void_p),  # 0x20 kv_last_page_lens
        ("ptr_QTP", ctypes.c_void_p),  # 0x28 qo_indptr
        ("ptr_QROPE", ctypes.c_void_p),  # 0x30 Q rope BF16
        ("ptr_KVROPE", ctypes.c_void_p),  # 0x38 KV rope BF16
        ("scalar_f", ctypes.c_float),  # 0x40 1/sqrt(512)
        ("s_gqa_ratio", ctypes.c_uint32),  # 0x44 gqa_ratio * max_seqlen_q (MQA)
        ("s_kv_split", ctypes.c_uint32),  # 0x48 num_kv_splits == passes
        ("s_total_kv", ctypes.c_uint32),  # 0x4C kv_seq_lens * num_seqs
        ("out_16_nosplit", ctypes.c_uint32),  # 0x50 0=fp32 split, 1=bf16 nosplit
        ("ptr_LSE", ctypes.c_void_p),  # 0x54 splitLse (attn_lse) FP32 (rw)
        ("ptr_LTD", ctypes.c_void_p),  # 0x5C kv_page_indices
        ("ptr_valid_split", ctypes.c_void_p),  # 0x64 [num_seqs] i32 scratch (rw)
        ("s_use_valid_split", ctypes.c_uint32),  # 0x6C gates valid_split write
        ("ptr_sink", ctypes.c_void_p),  # 0x70 [num_heads] FP32 sink logit
    ]


assert ctypes.sizeof(MlaV4KernelArgsPreload) == 120, ctypes.sizeof(
    MlaV4KernelArgsPreload
)


def _mla_v4_csv_path() -> str:
    """Path to the shipped gfx1250 v4 kernel registry (``mla_v4_asm.csv``)."""
    return os.path.join(get_asm_dir(), _MLA_V4_SUBDIR, _MLA_V4_CSV)


def _get_heuristic_kernel(q_type, kv_type, gqa, ps, prefill, causal, qseqlen, lse):
    """Return the CSV row matching the 8 lookup keys, or raise (mirror
    asm_mla_v4.cu::get_heuristic_kernel_mla_v4). The registry is parsed once
    (process-cached) by :func:`aiter.ops.asm.asm_utils.load_asm_cfg_csv`."""
    for cfg in load_asm_cfg_csv(_mla_v4_csv_path()):
        if cfg["qType"] != q_type or cfg["kvType"] != kv_type:
            continue
        if cfg["Gqa"] != gqa or cfg["ps"] != ps or cfg["prefill"] != prefill:
            continue
        if cfg["causal"] != causal or cfg["qSeqLen"] != qseqlen:
            continue
        if cfg["lse"] != lse:
            continue
        return cfg
    raise RuntimeError(
        f"mla_decode_v4_asm_gfx1250: no shipped variant for q_type:{q_type} "
        f"kv_type:{kv_type} gqa:{gqa} ps:{ps} qSeqLen:{qseqlen} prefill:{prefill} "
        f"causal:{causal} lse:{lse} arch:gfx1250"
    )


def mla_decode_v4_asm_gfx1250(
    Q,
    qrope,
    KV,
    kvrope,
    qo_indptr,
    kv_indptr,
    kv_page_indices,
    split_indptr,
    sink,
    max_seqlen_q,
    softmax_scale,
    out_16_nosplit,
    num_kv_splits,
    splitData,
    splitLse,
    output,
    valid_split_count=None,
    use_valid_split_count_reduce=0,
    kv_last_page_lens=None,
    stream=None,
):
    """gfx1250 v4 nm decode stage1 launch — Python peer of
    ``aiter.mla_decode_v4_asm`` (asm_mla_v4.cu) restricted to the gfx1250
    preload path. Same call signature; ``softmax_scale`` and ``split_indptr``
    are accepted for parity but unused on this ABI (the preload kernarg carries
    neither: the kernel hardcodes 1/sqrt(512) and derives splits from
    s_kv_split)."""
    del softmax_scale  # kernel hardcodes 1/sqrt(512)
    del split_indptr  # not part of the compact preload kernarg

    # ---- contract checks (mirror the AITER_CHECKs in the .cu) --------------
    if sink is None or sink.data_ptr() == 0:
        raise ValueError("mla_decode_v4_asm_gfx1250: `sink` must not be NULL")
    if not (Q.is_contiguous() and KV.is_contiguous()):
        raise ValueError(
            "mla_decode_v4_asm_gfx1250: only support Q/KV.is_contiguous() for now"
        )
    if not (qrope.is_contiguous() and kvrope.is_contiguous()):
        raise ValueError(
            "mla_decode_v4_asm_gfx1250: only support qrope/kvrope.is_contiguous()"
        )

    num_seqs = qo_indptr.shape[0] - 1
    num_heads = Q.size(1)
    num_kv_heads = KV.size(2)
    gqa_ratio = num_heads // num_kv_heads
    page_size = KV.size(1)
    dim_qk_packed = KV.size(3)
    q_type = dtype_str(Q)
    kv_type = dtype_str(KV)
    scalar_f = 1.0 / math.sqrt(float(_KV4_DIM_NOPE + _KV4_DIM_ROPE))
    ps = prefill = causal = lse_flag = 0

    if num_kv_heads != 1:
        raise ValueError(
            "mla_decode_v4_asm_gfx1250: only support num_kv_heads==1 for now"
        )
    if Q.size(2) != dim_qk_packed:
        raise ValueError(
            "mla_decode_v4_asm_gfx1250: Q head_size must equal KV head_size "
            "(= dim_qk_packed)"
        )

    # ---- Kernel selection: pure CSV table lookup (no computed heuristic) ---
    cfg = _get_heuristic_kernel(
        q_type, kv_type, gqa_ratio, ps, prefill, causal, max_seqlen_q, lse_flag
    )
    sub_Q = int(cfg["sub_Q"])
    co_path = os.path.join(get_asm_dir(), _MLA_V4_SUBDIR, cfg["co_name"])
    func = get_function(co_path, cfg["knl_name"])

    # ---- pack the 120-byte preload kernarg ---------------------------------
    args = MlaV4KernelArgsPreload()
    args.ptr_R = splitData.data_ptr()
    args.ptr_Q = Q.data_ptr()
    args.ptr_KV = KV.data_ptr()
    args.ptr_LTP = kv_indptr.data_ptr()
    args.ptr_LTL = (
        kv_last_page_lens.data_ptr() if kv_last_page_lens is not None else None
    )
    args.ptr_QTP = qo_indptr.data_ptr()
    args.ptr_QROPE = qrope.data_ptr()
    args.ptr_KVROPE = kvrope.data_ptr()
    args.scalar_f = scalar_f
    args.s_gqa_ratio = gqa_ratio * max_seqlen_q
    args.s_kv_split = int(num_kv_splits)
    args.s_total_kv = KV.size(0) * page_size
    args.out_16_nosplit = int(out_16_nosplit)
    args.ptr_LSE = splitLse.data_ptr()
    args.ptr_LTD = kv_page_indices.data_ptr()
    if use_valid_split_count_reduce != 0:
        if valid_split_count is None or valid_split_count.data_ptr() == 0:
            raise ValueError(
                "mla_decode_v4_asm_gfx1250: gfx1250 requires valid_split_count "
                "scratch tensor when use_valid_split_count_reduce!=0"
            )
    if valid_split_count is not None and valid_split_count.data_ptr() != 0:
        if valid_split_count.dtype != torch.int32:
            raise ValueError(
                "mla_decode_v4_asm_gfx1250: valid_split_count must be int32"
            )
        if valid_split_count.size(0) < num_seqs:
            raise ValueError(
                "mla_decode_v4_asm_gfx1250: valid_split_count must have at least "
                "num_seqs entries"
            )
        args.ptr_valid_split = valid_split_count.data_ptr()
    else:
        args.ptr_valid_split = None
    args.s_use_valid_split = 1 if use_valid_split_count_reduce != 0 else 0
    args.ptr_sink = sink.data_ptr()

    # ---- launch geometry (mirror asm_mla_v4.cu) ----------------------------
    #   gdx = ceil(gqa*max_seqlen_q / sub_Q), gdy = num_seqs, gdz = num_kv_splits
    #   block = 4 * warp_size
    block_dim = 4 * get_warp_size()
    q_seq_lens_internal = gqa_ratio * max_seqlen_q
    gdx = (q_seq_lens_internal + sub_Q - 1) // sub_Q
    gdy = num_seqs
    gdz = int(num_kv_splits)

    launch_co(func, (gdx, gdy, gdz), (block_dim, 1, 1), args, stream=stream)
