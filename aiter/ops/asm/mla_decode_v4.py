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
operation of ``aiter.mla_decode_v4_asm``, but uses a buffer-first argument
order so the eager launcher can be registered directly as a custom op.

It also hosts the fused split-KV variant (``mla_v4_fused_asm.csv``,
:func:`mla_decode_v4_fused_asm_gfx1250`): stage1 and the cross-split LSE
merge run in one cluster launch, replacing the stage1 + triton stage2 pair for
``2 <= num_kv_splits <= MLA_V4_FUSED_MAX_SPLITS``.
"""

import ctypes
import functools
import math
import os

import torch

from aiter.jit.core import AITER_ASM_DIR
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.asm.asm_utils import (
    dtype_str,
    get_function,
    get_warp_size,
    launch_co,
    launch_co_cluster,
    load_asm_cfg_csv,
    register_asm_custom_op,
)

# kV4DimNope + kV4DimRope = 448 + 64 = 512. The kernel hardcodes 1/sqrt(512)
# as its softmax pre-scale, independent of head_size (mirror asm_mla_v4.cu).
_KV4_DIM_NOPE = 448
_KV4_DIM_ROPE = 64
_KV4_HEAD_DIM = _KV4_DIM_NOPE + _KV4_DIM_ROPE

_MLA_V4_SUBDIR = "mla_v4"
_MLA_V4_CSV = "mla_v4_asm.csv"
_MLA_V4_FUSED_CSV = "mla_v4_fused_asm.csv"
# The fused kernel merges splits inside one hardware cluster of num_kv_splits
# workgroups; 16 is the largest cluster it supports.
MLA_V4_FUSED_MAX_SPLITS = 16


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


def _mla_v4_csv_path(csv_name: str = _MLA_V4_CSV) -> str:
    """Path to a shipped gfx1250 v4 kernel registry (default ``mla_v4_asm.csv``)."""
    return os.path.join(AITER_ASM_DIR, get_gfx_runtime(), _MLA_V4_SUBDIR, csv_name)


def _find_kernel_cfg(csv_name, q_type, kv_type, gqa, ps, prefill, causal, qseqlen, lse):
    """Return the ``csv_name`` row matching the 8 lookup keys, or None."""
    csv_path = _mla_v4_csv_path(csv_name)
    if not os.path.isfile(csv_path):
        return None
    for cfg in load_asm_cfg_csv(csv_path):
        if cfg["qType"] != q_type or cfg["kvType"] != kv_type:
            continue
        if cfg["Gqa"] != gqa or cfg["ps"] != ps or cfg["prefill"] != prefill:
            continue
        if cfg["causal"] != causal or cfg["qSeqLen"] != qseqlen:
            continue
        if cfg["lse"] != lse:
            continue
        return cfg
    return None


def _get_heuristic_kernel(q_type, kv_type, gqa, ps, prefill, causal, qseqlen, lse):
    """Return the CSV row matching the 8 lookup keys, or raise (mirror
    asm_mla_v4.cu::get_heuristic_kernel_mla_v4). The registry is parsed once
    (process-cached) by :func:`aiter.ops.asm.asm_utils.load_asm_cfg_csv`."""
    cfg = _find_kernel_cfg(
        _MLA_V4_CSV, q_type, kv_type, gqa, ps, prefill, causal, qseqlen, lse
    )
    if cfg is not None:
        return cfg
    raise RuntimeError(
        f"mla_decode_v4_asm_gfx1250: no shipped variant for q_type:{q_type} "
        f"kv_type:{kv_type} gqa:{gqa} ps:{ps} qSeqLen:{qseqlen} prefill:{prefill} "
        f"causal:{causal} lse:{lse} arch:gfx1250"
    )


def mla_decode_v4_asm_gfx1250_eager(
    Q: torch.Tensor,
    qrope: torch.Tensor,
    KV: torch.Tensor,
    kvrope: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    split_indptr: torch.Tensor,
    sink: torch.Tensor,
    splitData: torch.Tensor,
    splitLse: torch.Tensor,
    output: torch.Tensor,
    valid_split_count: torch.Tensor | None,
    max_seqlen_q: int,
    softmax_scale: float,
    out_16_nosplit: int,
    num_kv_splits: int,
    use_valid_split_count_reduce: int = 0,
    kv_last_page_lens: torch.Tensor | None = None,
) -> None:
    """gfx1250 v4 nm decode stage1 launch (eager, pure Python + ctypes) — Python
    peer of ``aiter.mla_decode_v4_asm`` (asm_mla_v4.cu) restricted to the gfx1250
    preload path. It uses a buffer-first signature for direct custom-op
    registration; ``softmax_scale`` and ``split_indptr`` are accepted for
    semantic parity but unused on this ABI (the preload kernarg carries neither:
    the kernel hardcodes 1/sqrt(512) and derives splits from s_kv_split).
    ``out_16_nosplit`` is derived from ``num_kv_splits`` to match the canonical
    dispatcher.

    This is the raw launcher: lowest host overhead, but opaque to TorchDynamo.
    Prefer the :func:`mla_decode_v4_asm_gfx1250` dispatcher, which routes to the
    ``torch.compile``-safe custom op while tracing and here otherwise."""
    runtime_gfx = get_gfx_runtime()
    if runtime_gfx != "gfx1250":
        raise RuntimeError(
            "mla_decode_v4_asm_gfx1250 is only supported on gfx1250, "
            f"got {runtime_gfx}"
        )
    del softmax_scale  # kernel hardcodes 1/sqrt(512)
    del split_indptr  # not part of the compact preload kernarg

    # ---- contract checks (mirror the AITER_CHECKs in the .cu) --------------
    out_16_nosplit = 1 if int(num_kv_splits) == 1 else 0
    if out_16_nosplit != 0 and splitData.data_ptr() != output.data_ptr():
        raise ValueError(
            "mla_decode_v4_asm_gfx1250: when out_16_nosplit!=0, the kernel "
            "writes through splitData (ptr_R); splitData must alias output"
        )
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
    co_path = os.path.join(AITER_ASM_DIR, runtime_gfx, _MLA_V4_SUBDIR, cfg["co_name"])
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
    if use_valid_split_count_reduce != 0 and (
        valid_split_count is None or valid_split_count.data_ptr() == 0
    ):
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

    launch_co(func, (gdx, gdy, gdz), (block_dim, 1, 1), args)


# Mutated buffers precede SymInt scalars so this implementation can be
# registered directly without a schema adapter. This order is required by
# torch's auto-functionalization under ``torch.compile(fullgraph=True)``.
mla_decode_v4_asm_gfx1250 = register_asm_custom_op(
    "mla_decode_v4_asm_gfx1250",
    mla_decode_v4_asm_gfx1250_eager,
    mutates_args=["splitData", "splitLse", "output", "valid_split_count"],
)


# ---------------------------------------------------------------------------
# Fused split-KV decode (mla_v4_fused_asm.csv).
#
# One launch does stage1 + the cross-split LSE merge: grid.x = nsplit * gdx,
# and the nsplit workgroups of one (seq, q-tile) form a hardware cluster along
# x. Split 0 of each cluster reduces the partials of the others (read back from
# the fp32 scratch) and writes the final bf16 result. Kernarg ABI is the same
# 120-byte MlaV4KernelArgsPreload, with:
#   ptr_R           = fp32 partial scratch [total_q, nsplit, (gqa+1)*dv]
#   ptr_LSE         = fp32 partial lse     [total_q, nsplit, num_heads, 1]
#   ptr_valid_split = final bf16 output    [total_q, num_heads, dv]
#   out_16_nosplit = 0, s_use_valid_split = 0
# The .co carries no .cluster_dims metadata, so the cluster size comes from the
# launch attribute and one code object serves every split count.
# ---------------------------------------------------------------------------
def mla_v4_fused_slot_f32(num_heads: int, v_head_dim: int) -> int:
    """fp32 elements per (token, split) slot of the fused partial scratch: the
    kernel pads each slot by one extra fixed-width head row."""
    if v_head_dim != _KV4_HEAD_DIM:
        raise ValueError(
            "mla_decode_v4_fused_asm_gfx1250: fused QH32 requires "
            f"v_head_dim={_KV4_HEAD_DIM}, got {v_head_dim}"
        )
    return (num_heads + 1) * _KV4_HEAD_DIM


@functools.cache
def _fused_co_for(q_type, kv_type, gqa, qseqlen):
    cfg = _find_kernel_cfg(_MLA_V4_FUSED_CSV, q_type, kv_type, gqa, 0, 0, 0, qseqlen, 0)
    if cfg is None:
        return None
    co_path = os.path.join(
        AITER_ASM_DIR, get_gfx_runtime(), _MLA_V4_SUBDIR, cfg["co_name"]
    )
    if not os.path.isfile(co_path):
        return None
    return cfg, co_path


def get_mla_v4_fused_kernel(Q, KV, max_seqlen_q, num_kv_splits):
    """Return ``(cfg, co_path)`` of the fused kernel for this call, or None when
    the fused path does not apply (not gfx1250, split count outside
    ``[2, MLA_V4_FUSED_MAX_SPLITS]``, no shipped variant / .co, or disabled
    via ``AITER_MLA_V4_FUSED=0``). Callers fall back to stage1 + stage2 on
    None."""
    if get_gfx_runtime() != "gfx1250":
        return None
    nsplit = int(num_kv_splits)
    if not (2 <= nsplit <= MLA_V4_FUSED_MAX_SPLITS):
        return None
    if os.environ.get("AITER_MLA_V4_FUSED", "1") == "0":
        return None
    gqa = Q.size(1) // KV.size(2)
    return _fused_co_for(dtype_str(Q), dtype_str(KV), gqa, int(max_seqlen_q))


def mla_decode_v4_fused_asm_gfx1250_eager(
    Q: torch.Tensor,
    qrope: torch.Tensor,
    KV: torch.Tensor,
    kvrope: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    sink: torch.Tensor,
    splitData: torch.Tensor,
    splitLse: torch.Tensor,
    output: torch.Tensor,
    max_seqlen_q: int,
    num_kv_splits: int,
    kv_last_page_lens: torch.Tensor | None = None,
) -> None:
    """gfx1250 fused v4 decode launch (stage1 + split merge, eager ctypes).

    ``splitData`` / ``splitLse`` are write-only scratch (no init needed; empty
    splits are never read back). The final bf16 result lands in ``output``."""
    nsplit = int(num_kv_splits)
    found = get_mla_v4_fused_kernel(Q, KV, max_seqlen_q, nsplit)
    if found is None:
        raise RuntimeError(
            f"mla_decode_v4_fused_asm_gfx1250: no fused variant for "
            f"q_type:{dtype_str(Q)} kv_type:{dtype_str(KV)} "
            f"gqa:{Q.size(1) // KV.size(2)} qSeqLen:{max_seqlen_q} "
            f"num_kv_splits:{nsplit} (supported 2..{MLA_V4_FUSED_MAX_SPLITS})"
        )
    cfg, co_path = found

    if sink is None or sink.data_ptr() == 0:
        raise ValueError("mla_decode_v4_fused_asm_gfx1250: `sink` must not be NULL")
    if not (Q.is_contiguous() and KV.is_contiguous()):
        raise ValueError(
            "mla_decode_v4_fused_asm_gfx1250: only support Q/KV.is_contiguous()"
        )
    if not (qrope.is_contiguous() and kvrope.is_contiguous()):
        raise ValueError(
            "mla_decode_v4_fused_asm_gfx1250: only support "
            "qrope/kvrope.is_contiguous()"
        )
    if KV.size(2) != 1:
        raise ValueError(
            "mla_decode_v4_fused_asm_gfx1250: only support num_kv_heads==1"
        )
    if Q.size(2) != KV.size(3):
        raise ValueError(
            "mla_decode_v4_fused_asm_gfx1250: Q head_size must equal KV "
            "head_size (= dim_qk_packed)"
        )

    num_seqs = qo_indptr.shape[0] - 1
    num_heads = Q.size(1)
    v_head_dim = output.size(-1)
    total_q = num_seqs * max_seqlen_q
    if output.dtype != torch.bfloat16 or not output.is_contiguous():
        raise ValueError(
            "mla_decode_v4_fused_asm_gfx1250: `output` must be contiguous bf16"
        )
    if output.numel() < total_q * num_heads * v_head_dim:
        raise ValueError(
            "mla_decode_v4_fused_asm_gfx1250: `output` smaller than "
            "[total_q, num_heads, v_head_dim]"
        )
    need_data = total_q * nsplit * mla_v4_fused_slot_f32(num_heads, v_head_dim)
    if (
        splitData.dtype != torch.float32
        or not splitData.is_contiguous()
        or splitData.numel() < need_data
    ):
        raise ValueError(
            f"mla_decode_v4_fused_asm_gfx1250: `splitData` must be contiguous "
            f"fp32 with >= {need_data} elements "
            f"([total_q, nsplit, (num_heads+1)*v_head_dim]), got "
            f"{splitData.dtype} numel={splitData.numel()}"
        )
    need_lse = total_q * nsplit * num_heads
    if (
        splitLse.dtype != torch.float32
        or not splitLse.is_contiguous()
        or splitLse.numel() < need_lse
    ):
        raise ValueError(
            f"mla_decode_v4_fused_asm_gfx1250: `splitLse` must be contiguous "
            f"fp32 with >= {need_lse} elements"
        )

    func = get_function(co_path, cfg["knl_name"])
    sub_Q = int(cfg["sub_Q"])

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
    args.scalar_f = 1.0 / math.sqrt(float(_KV4_DIM_NOPE + _KV4_DIM_ROPE))
    args.s_gqa_ratio = num_heads * max_seqlen_q
    args.s_kv_split = nsplit
    args.s_total_kv = KV.size(0) * KV.size(1)
    args.out_16_nosplit = 0
    args.ptr_LSE = splitLse.data_ptr()
    args.ptr_LTD = kv_page_indices.data_ptr()
    args.ptr_valid_split = output.data_ptr()
    args.s_use_valid_split = 0
    args.ptr_sink = sink.data_ptr()

    gdx = (num_heads * max_seqlen_q + sub_Q - 1) // sub_Q
    launch_co_cluster(
        func,
        (nsplit * gdx, num_seqs, 1),
        (4 * get_warp_size(), 1, 1),
        args,
        cluster_dim=(nsplit, 1, 1),
    )


mla_decode_v4_fused_asm_gfx1250 = register_asm_custom_op(
    "mla_decode_v4_fused_asm_gfx1250",
    mla_decode_v4_fused_asm_gfx1250_eager,
    mutates_args=["splitData", "splitLse", "output"],
)
