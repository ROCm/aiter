# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pure-Python launcher for the v4 MLA decode ``.co`` kernels.

The shipped ``.co`` is loaded and launched directly from Python through the
generic :mod:`aiter.ops.asm.asm_utils` helpers; there is no C++ host bridge.
Each entry is a thin wrapper:
  * resolve the kernel binary via the shipped ``hsa/<arch>/mla_v4`` registry
    (``mla_v4_asm.csv``);
  * pack the kernarg (gfx1250: compact 120-byte preload ABI; gfx950: legacy
    21 x 16-byte slot ABI);
  * compute the launch geometry;
  * launch.

Entries (all registered as ``torch.compile``-safe aiter custom ops):
  * :func:`mla_decode_v4_asm` -- non-persistent split-KV stage1 (gfx950,
    gfx1250);
  * :func:`mla_decode_v4_ps_asm` -- persistent decode with in-kernel split
    plan and merge (gfx950, ``ps=1`` registry row);
  * :func:`mla_decode_v4_fused_asm_gfx1250` -- fused split-KV variant
    (``mla_v4_fused_asm.csv``): stage1 and the cross-split LSE merge run in one
    cluster launch, replacing the stage1 + triton stage2 pair for
    ``2 <= num_kv_splits <= MLA_V4_FUSED_MAX_SPLITS``.

Setting ``AITER_V4_NM_DUMP_KERNARG=1`` prints each :func:`mla_decode_v4_asm`
kernarg packet and launch geometry to stderr.
"""

import ctypes
import functools
import math
import os
import sys

import torch

from aiter.jit.core import AITER_ASM_DIR
from aiter.ops.asm.asm_utils import (
    current_raw_stream,
    device_guard,
    dtype_str,
    get_function,
    get_gfx_from_device,
    get_warp_size,
    launch_co,
    launch_co_cluster,
    load_asm_cfg_csv,
    register_asm_custom_op,
)

# kV4DimNope + kV4DimRope = 448 + 64 = 512. The kernel hardcodes 1/sqrt(512)
# as its softmax pre-scale, independent of head_size.
_KV4_DIM_NOPE = 448
_KV4_DIM_ROPE = 64
_KV4_HEAD_DIM = _KV4_DIM_NOPE + _KV4_DIM_ROPE
_SCALAR_F = 1.0 / math.sqrt(float(_KV4_HEAD_DIM))

_MLA_V4_SUBDIR = "mla_v4"
_MLA_V4_CSV = "mla_v4_asm.csv"
_MLA_V4_FUSED_CSV = "mla_v4_fused_asm.csv"
# The fused kernel merges splits inside one hardware cluster of num_kv_splits
# workgroups; 16 is the largest cluster it supports.
MLA_V4_FUSED_MAX_SPLITS = 16


class MlaV4KernelArgsPreload(ctypes.Structure):
    """120-byte compact preload kernarg (gfx1250 DIRECT_PARAM=1 ABI)."""

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

_P2 = ctypes.c_uint32 * 2  # pads an 8-byte field to a 16-byte slot
_P3 = ctypes.c_uint32 * 3  # pads a 4-byte field to a 16-byte slot


class MlaV4KernelArgsLegacy(ctypes.Structure):
    """336-byte legacy kernarg (21 x 16-byte slots, gfx950 ABI)."""

    _pack_ = 1
    _fields_ = [
        ("ptr_R", ctypes.c_void_p),  # 0:  splitData (logits) FP32 (rw)
        ("_p_r", _P2),
        ("ptr_LSE", ctypes.c_void_p),  # 1:  splitLse (attn_lse) FP32 (rw)
        ("_p_lse", _P2),
        ("ptr_Q", ctypes.c_void_p),  # 2:  Q packed FP8 + e8m0 scale
        ("_p_q", _P2),
        ("ptr_KV", ctypes.c_void_p),  # 3:  KV packed FP8
        ("_p_kv", _P2),
        ("ptr_LTP", ctypes.c_void_p),  # 4:  kv_indptr
        ("_p_ltp", _P2),
        ("ptr_LTD", ctypes.c_void_p),  # 5:  kv_page_indices
        ("_p_ltd", _P2),
        ("ptr_LTL", ctypes.c_void_p),  # 6:  kv_last_page_lens
        ("_p_ltl", _P2),
        ("scalar_f", ctypes.c_float),  # 7:  1/sqrt(512)
        ("_p_sc", _P3),
        ("s_gqa_ratio", ctypes.c_uint32),  # 8:  raw gqa_ratio
        ("_p_gr", _P3),
        ("s_kv_split", ctypes.c_uint32),  # 9:  num_kv_splits == passes
        ("_p_ps", _P3),
        ("s_total_kv", ctypes.c_uint32),  # 10: not read by the kernel, 0
        ("_p_tk", _P3),
        ("s_stride_page", ctypes.c_uint32),  # 11: not read by the kernel, 0
        ("_p_sp", _P3),
        ("s_log2_page", ctypes.c_uint32),  # 12: log2(page_size)
        ("_p_lp", _P3),
        ("ptr_QTP", ctypes.c_void_p),  # 13: qo_indptr
        ("_p_qtp", _P2),
        ("ptr_STP", ctypes.c_void_p),  # 14: split_indptr
        ("_p_stp", _P2),
        ("out_16_nosplit", ctypes.c_uint32),  # 15: 0=fp32 split, 1=bf16 nosplit
        ("_p_o16", _P3),
        ("ptr_QROPE", ctypes.c_void_p),  # 16: Q rope BF16
        ("_p_qrope", _P2),
        ("ptr_KVROPE", ctypes.c_void_p),  # 17: KV rope BF16
        ("_p_kvrope", _P2),
        ("ptr_sink", ctypes.c_void_p),  # 18: [num_heads] FP32 sink logit
        ("_p_sink", _P2),
        ("ptr_valid_split", ctypes.c_void_p),  # 19: [num_seqs] i32 scratch (rw)
        ("_p_vs", _P2),
        ("s_use_valid_split", ctypes.c_uint32),  # 20: gates valid_split write
        ("_p_uvs", _P3),
    ]


assert ctypes.sizeof(MlaV4KernelArgsLegacy) == 21 * 16, ctypes.sizeof(
    MlaV4KernelArgsLegacy
)


class MlaV4PsKernelArgs(ctypes.Structure):
    """336-byte kernarg of the persistent kernel. Keeps the 21 x 16-byte slot
    grid of :class:`MlaV4KernelArgsLegacy`, but several slots carry a
    different value and some upper slot halves carry an extra field."""

    _pack_ = 1
    _fields_ = [
        ("ptr_O_acc", ctypes.c_void_p),  # 0x00 workspace o_acc [2P,128,512] FP32
        ("ptr_out", ctypes.c_void_p),  # 0x08 out [N,128,512] BF16
        ("ptr_L_acc", ctypes.c_void_p),  # 0x10 workspace lse_acc [2P,128] FP32
        ("ptr_lse", ctypes.c_void_p),  # 0x18 final LSE [N,128] FP32 or NULL
        ("ptr_Q", ctypes.c_void_p),  # 0x20 Q packed FP8 + e8m0 scale
        ("_p_q", _P2),
        ("ptr_KV", ctypes.c_void_p),  # 0x30 KV packed FP8
        ("_p_kv", _P2),
        ("ptr_LTP", ctypes.c_void_p),  # 0x40 kv_indptr
        ("_p_ltp", _P2),
        ("ptr_LTD", ctypes.c_void_p),  # 0x50 kv_page_indices
        ("_p_ltd", _P2),
        ("ptr_desc", ctypes.c_void_p),  # 0x60 workspace desc [P,8] int32 (rw)
        ("ptr_dbg", ctypes.c_void_p),  # 0x68 unused, NULL
        ("scalar_f", ctypes.c_float),  # 0x70 1/sqrt(512)
        ("_p_sc", _P3),
        ("s_gqa_ratio", ctypes.c_uint32),  # 0x80 128 (heads)
        ("_p_gr", _P3),
        ("s_kv_split", ctypes.c_uint32),  # 0x90 1
        ("_p_ks", ctypes.c_uint32),  # 0x94
        ("ptr_cnt", ctypes.c_void_p),  # 0x98 workspace cnt int32 (rw)
        ("s_num_p", ctypes.c_uint32),  # 0xa0 number of partitions P
        ("s_num_q", ctypes.c_uint32),  # 0xa4 number of query rows N
        ("_p_nq", _P2),  # 0xa8
        ("s_plan", ctypes.c_uint32),  # 0xb0 planner config F | (MT << 8)
        ("_p_pl", ctypes.c_uint32),  # 0xb4
        ("ptr_unused", ctypes.c_void_p),  # 0xb8 not read, NULL
        ("s_zero", ctypes.c_uint32),  # 0xc0 0
        ("_p_z", _P3),
        ("ptr_QTP", ctypes.c_void_p),  # 0xd0 workspace arange int32
        ("_p_qtp", _P2),
        ("ptr_STP", ctypes.c_void_p),  # 0xe0 the same arange
        ("_p_stp", _P2),
        ("out_16_nosplit", ctypes.c_uint32),  # 0xf0 1
        ("_p_o16", _P3),
        ("ptr_QROPE", ctypes.c_void_p),  # 0x100 Q rope BF16
        ("_p_qrope", _P2),
        ("ptr_KVROPE", ctypes.c_void_p),  # 0x110 KV rope BF16
        ("_p_kvrope", _P2),
        ("ptr_sink", ctypes.c_void_p),  # 0x120 [128] FP32 sink logit
        ("_p_sink", _P2),
        ("ptr_zero", ctypes.c_void_p),  # 0x130 NULL
        ("_p_pz", _P2),
        ("s_zero2", ctypes.c_uint32),  # 0x140 0
        ("_p_z2", _P3),
    ]


assert ctypes.sizeof(MlaV4PsKernelArgs) == 21 * 16, ctypes.sizeof(MlaV4PsKernelArgs)
assert (
    MlaV4PsKernelArgs.ptr_cnt.offset == 0x98
    and MlaV4PsKernelArgs.s_num_q.offset == 0xA4
    and MlaV4PsKernelArgs.ptr_unused.offset == 0xB8
    and MlaV4PsKernelArgs.ptr_sink.offset == 0x120
)


def _mla_v4_csv_path(arch: str, csv_name: str = _MLA_V4_CSV) -> str:
    """Path to a shipped gfx1250 v4 kernel registry (default ``mla_v4_asm.csv``)."""
    return os.path.join(AITER_ASM_DIR, arch, _MLA_V4_SUBDIR, csv_name)


def _find_kernel_cfg(
    arch, csv_name, q_type, kv_type, gqa, ps, prefill, causal, qseqlen, lse
):
    """Return the ``csv_name`` row matching the 8 lookup keys, or None."""
    csv_path = _mla_v4_csv_path(arch, csv_name)
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


def _get_heuristic_kernel(
    arch, q_type, kv_type, gqa, ps, prefill, causal, qseqlen, lse
):
    """Return the ``mla_v4_asm.csv`` row matching the 8 lookup keys, or raise.
    The registry is parsed once (process-cached) by
    :func:`aiter.ops.asm.asm_utils.load_asm_cfg_csv`."""
    cfg = _find_kernel_cfg(
        arch, _MLA_V4_CSV, q_type, kv_type, gqa, ps, prefill, causal, qseqlen, lse
    )
    if cfg is not None:
        return cfg
    raise RuntimeError(
        f"mla_decode_v4_asm: no shipped variant for q_type:{q_type} "
        f"kv_type:{kv_type} gqa:{gqa} ps:{ps} qSeqLen:{qseqlen} prefill:{prefill} "
        f"causal:{causal} lse:{lse} arch:{arch}"
    )


@functools.cache
def _device_info(device_index):
    """(arch, warp_size) of a device; launch-path helper, not for traced code."""
    return get_gfx_from_device(device_index), get_warp_size(device_index)


@functools.cache
def _select_nm_kernel(arch, q_type, kv_type, gqa_ratio, max_seqlen_q):
    """Pick the non-persistent decode row, its per-WG Q tile ``sub_Q`` and the
    ``.co`` path.

    Serviceable fp8 (gqa, max_seqlen_q) pairs: gqa 16 with msq 1/2/4, gqa
    32/64/128 with msq 1. gfx950 ships one 64-row-tile kernel registered as
    (Gqa=64, qSeqLen=1) that serves all of them, so those pairs are looked up
    under that key there. gfx1250 registers every variant explicitly and
    carries ``sub_Q`` in its registry; elsewhere ``sub_Q = min(64, gqa*msq)``.
    """
    supported = (
        q_type == "fp8"
        and kv_type == "fp8"
        and (
            (gqa_ratio == 16 and max_seqlen_q in (1, 2, 4))
            or (gqa_ratio in (32, 64, 128) and max_seqlen_q == 1)
        )
    )
    csv_gqa, csv_qseqlen = gqa_ratio, max_seqlen_q
    if arch != "gfx1250" and supported:
        csv_gqa, csv_qseqlen = 64, 1
    cfg = _get_heuristic_kernel(arch, q_type, kv_type, csv_gqa, 0, 0, 0, csv_qseqlen, 0)
    if "sub_Q" in cfg:
        sub_Q = int(cfg["sub_Q"])
    else:
        sub_Q = min(64, gqa_ratio * max_seqlen_q) if supported else 64
    co_path = os.path.join(AITER_ASM_DIR, arch, _MLA_V4_SUBDIR, cfg["co_name"])
    return cfg, sub_Q, co_path


def _dump_kernarg(args, grid, block):
    """``AITER_V4_NM_DUMP_KERNARG=1``: hexdump the kernarg packet (16 bytes per
    row) and the launch geometry to stderr."""
    if not os.environ.get("AITER_V4_NM_DUMP_KERNARG", "").startswith("1"):
        return
    raw = bytes(args)
    rows = [
        " ".join(f"{b:02x}" for b in raw[i : i + 16]) for i in range(0, len(raw), 16)
    ]
    sys.stderr.write(
        f"[aiter kernarg {len(raw)}B]\n"
        + "\n".join(rows)
        + f"\n[aiter grid ({grid[0]},{grid[1]},{grid[2]}) block ({block[0]},1,1)]\n"
    )
    sys.stderr.flush()


def mla_decode_v4_asm_eager(
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
    """v4 nm decode stage1 launch (eager), gfx950 and gfx1250.

    Uses a buffer-first signature for direct custom-op registration.
    ``softmax_scale`` is accepted for API parity but unused (the kernel
    hardcodes 1/sqrt(512)); ``split_indptr`` is only consumed by the gfx950
    kernarg (gfx1250 derives splits from ``num_kv_splits``); and
    ``out_16_nosplit`` is derived from ``num_kv_splits``: with one split the
    kernel writes the final packed-BF16 result through ``splitData``.

    This is the raw launcher: lowest host overhead, but opaque to TorchDynamo.
    Prefer :func:`mla_decode_v4_asm`, the ``torch.compile``-safe custom op."""
    del softmax_scale  # kernel hardcodes 1/sqrt(512)
    dev = Q.device.index
    runtime_gfx, warp_size = _device_info(dev)
    preload = runtime_gfx == "gfx1250"

    out_16_nosplit = 1 if int(num_kv_splits) == 1 else 0
    if sink is None or sink.data_ptr() == 0:
        raise ValueError("mla_decode_v4_asm: `sink` must not be NULL")
    if not (Q.is_contiguous() and KV.is_contiguous()):
        raise ValueError("mla_decode_v4_asm: only support Q/KV.is_contiguous() for now")
    if not (qrope.is_contiguous() and kvrope.is_contiguous()):
        raise ValueError("mla_decode_v4_asm: only support qrope/kvrope.is_contiguous()")

    num_seqs = qo_indptr.shape[0] - 1
    num_heads = Q.size(1)
    num_kv_heads = KV.size(2)
    gqa_ratio = num_heads // num_kv_heads
    page_size = KV.size(1)
    dim_qk_packed = KV.size(3)
    q_type = dtype_str(Q)
    kv_type = dtype_str(KV)

    if num_kv_heads != 1:
        raise ValueError("mla_decode_v4_asm: only support num_kv_heads==1 for now")
    if page_size <= 0 or page_size & (page_size - 1):
        raise ValueError("mla_decode_v4_asm: KV page_size must be a power of two")
    if preload and page_size != 1:
        raise ValueError("mla_decode_v4_asm: only support KV page_size==1")
    if Q.size(2) != _KV4_HEAD_DIM or dim_qk_packed != _KV4_HEAD_DIM:
        raise ValueError(
            f"mla_decode_v4_asm: Q and KV head_size must be {_KV4_HEAD_DIM}, got "
            f"{Q.size(2)} / {dim_qk_packed}"
        )
    if qrope.size(-1) != _KV4_DIM_ROPE or kvrope.size(-1) != _KV4_DIM_ROPE:
        raise ValueError(
            f"mla_decode_v4_asm: qrope / kvrope last dim must be {_KV4_DIM_ROPE}"
        )
    if out_16_nosplit and (
        splitData.numel() * splitData.element_size()
        < output.numel() * output.element_size()
    ):
        raise ValueError(
            "mla_decode_v4_asm: with num_kv_splits==1 the kernel writes the "
            "BF16 result through splitData, which is smaller than output"
        )
    if not preload and split_indptr is None:
        raise ValueError(f"mla_decode_v4_asm: {runtime_gfx} requires split_indptr")

    cfg, sub_Q, co_path = _select_nm_kernel(
        runtime_gfx, q_type, kv_type, gqa_ratio, int(max_seqlen_q)
    )

    args = MlaV4KernelArgsPreload() if preload else MlaV4KernelArgsLegacy()
    args.ptr_R = splitData.data_ptr()
    args.ptr_LSE = splitLse.data_ptr()
    args.ptr_Q = Q.data_ptr()
    args.ptr_KV = KV.data_ptr()
    args.ptr_LTP = kv_indptr.data_ptr()
    args.ptr_LTD = kv_page_indices.data_ptr()
    args.ptr_LTL = (
        kv_last_page_lens.data_ptr() if kv_last_page_lens is not None else None
    )
    args.ptr_QTP = qo_indptr.data_ptr()
    args.ptr_QROPE = qrope.data_ptr()
    args.ptr_KVROPE = kvrope.data_ptr()
    args.ptr_sink = sink.data_ptr()
    args.scalar_f = _SCALAR_F
    args.s_kv_split = int(num_kv_splits)
    args.out_16_nosplit = out_16_nosplit
    if preload:
        # The gfx1250 kernel takes the flattened MQA row count and a real
        # s_total_kv.
        args.s_gqa_ratio = gqa_ratio * max_seqlen_q
        args.s_total_kv = KV.size(0) * page_size
    else:
        args.s_gqa_ratio = gqa_ratio
        args.s_log2_page = page_size.bit_length() - 1
        args.ptr_STP = split_indptr.data_ptr()

    if (
        preload
        and use_valid_split_count_reduce != 0
        and (valid_split_count is None or valid_split_count.data_ptr() == 0)
    ):
        raise ValueError(
            "mla_decode_v4_asm: gfx1250 requires valid_split_count "
            "scratch tensor when use_valid_split_count_reduce!=0"
        )
    if valid_split_count is not None and valid_split_count.data_ptr() != 0:
        if valid_split_count.dtype != torch.int32:
            raise ValueError("mla_decode_v4_asm: valid_split_count must be int32")
        if valid_split_count.size(0) < num_seqs:
            raise ValueError(
                "mla_decode_v4_asm: valid_split_count must have at least "
                "num_seqs entries"
            )
        args.ptr_valid_split = valid_split_count.data_ptr()
    else:
        args.ptr_valid_split = None
    args.s_use_valid_split = 1 if use_valid_split_count_reduce != 0 else 0

    # gdx = ceil(gqa*max_seqlen_q / sub_Q), gdy = num_seqs, gdz = num_kv_splits
    grid = (
        (gqa_ratio * max_seqlen_q + sub_Q - 1) // sub_Q,
        num_seqs,
        int(num_kv_splits),
    )
    block = (4 * warp_size, 1, 1)
    _dump_kernarg(args, grid, block)
    with device_guard(dev):
        func = get_function(co_path, cfg["knl_name"], dev)
        launch_co(func, grid, block, args, stream=current_raw_stream(dev))


# Mutated buffers precede SymInt scalars so this implementation can be
# registered directly without a schema adapter. This order is required by
# torch's auto-functionalization under ``torch.compile(fullgraph=True)``.
mla_decode_v4_asm = register_asm_custom_op(
    "mla_decode_v4_asm",
    mla_decode_v4_asm_eager,
    mutates_args=["splitData", "splitLse", "output", "valid_split_count"],
)


# ---------------------------------------------------------------------------
# Persistent decode (mla_v4_asm.csv row ps=1, gfx950).
#
# One launch per call: grid (2 * P, 1, 1), block 256. The kernel plans the KV
# split over P partitions in-kernel from kv_indptr, runs the attention and
# merges the split partials itself (the last arriving partition of a row
# combines it), so there is no host split plan and no stage-2 merge. Same
# packed layouts as mla_decode_v4_asm with gqa=128, max_seqlen_q=1 and
# page_size=1; rows with K=0 are left unwritten.
# ---------------------------------------------------------------------------
_PS_HEADS = 128
_PS_MAX_PARTS = 1024
_PS_MAX_ROWS = 32768  # byte offsets of out / q wrap above this
_PS_ARANGE = 65537
# cnt: [0, 2*65536) row counters, [2*65536, +16*512) reserved and
# [.., +4*1024) group counters.
_PS_CNT_INTS = 2 * 65536 + 16 * 512 + 4 * 1024
_PS_PLAN_CFG = 6 | (1 << 8)


@functools.cache
def _select_ps_kernel(arch):
    cfg = _get_heuristic_kernel(arch, "fp8", "fp8", 64, 1, 0, 0, 1, 1)
    return cfg, os.path.join(AITER_ASM_DIR, arch, _MLA_V4_SUBDIR, cfg["co_name"])


def _ps_check_buf(t, dtype, name, device):
    if t is None or t.data_ptr() == 0:
        raise RuntimeError(f"mla_decode_v4_ps_asm: `{name}` is NULL")
    if t.dtype != dtype:
        raise RuntimeError(
            f"mla_decode_v4_ps_asm: `{name}` has dtype {t.dtype}, expected {dtype}"
        )
    if not t.is_contiguous():
        raise RuntimeError(f"mla_decode_v4_ps_asm: `{name}` must be contiguous")
    if t.device != device:
        raise RuntimeError(f"mla_decode_v4_ps_asm: `{name}` is on another device")


def mla_decode_v4_ps_asm_eager(
    Q: torch.Tensor,
    qrope: torch.Tensor,
    KV: torch.Tensor,
    kvrope: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    sink: torch.Tensor,
    o_acc: torch.Tensor,
    lse_acc: torch.Tensor,
    desc: torch.Tensor,
    cnt: torch.Tensor,
    arange: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor | None = None,
) -> None:
    """Persistent v4 nm decode launch (eager).

    ``o_acc`` / ``lse_acc`` / ``desc`` / ``cnt`` / ``arange`` are the workspace
    from ``aiter.mla.get_mla_v4_nm_ps_workspace``; ``lse`` is optional (None =
    final LSE not written)."""
    if Q is None:
        raise RuntimeError("mla_decode_v4_ps_asm: `Q` is NULL")
    dev = Q.device
    fp8, bf16, f32, i32 = (
        torch.float8_e4m3fn,
        torch.bfloat16,
        torch.float32,
        torch.int32,
    )
    for t, dt, name in (
        (Q, fp8, "Q"),
        (qrope, bf16, "qrope"),
        (KV, fp8, "KV"),
        (kvrope, bf16, "kvrope"),
        (kv_indptr, i32, "kv_indptr"),
        (kv_page_indices, i32, "kv_page_indices"),
        (sink, f32, "sink"),
        (o_acc, f32, "o_acc"),
        (lse_acc, f32, "lse_acc"),
        (desc, i32, "desc"),
        (cnt, i32, "cnt"),
        (arange, i32, "arange"),
        (output, bf16, "output"),
    ):
        _ps_check_buf(t, dt, name, dev)
    has_lse = lse is not None and lse.data_ptr() != 0
    if has_lse:
        _ps_check_buf(lse, f32, "lse", dev)

    if Q.dim() != 3 or Q.size(1) != _PS_HEADS or Q.size(2) != _KV4_HEAD_DIM:
        raise RuntimeError("mla_decode_v4_ps_asm: Q must be [N, 128, 512]")
    num_q = Q.size(0)
    if num_q > _PS_MAX_ROWS:
        raise RuntimeError(f"mla_decode_v4_ps_asm: N={num_q} exceeds {_PS_MAX_ROWS}")
    if qrope.numel() != num_q * _PS_HEADS * _KV4_DIM_ROPE:
        raise RuntimeError("mla_decode_v4_ps_asm: qrope must be [N, 128, 64]")
    if output.dim() != 3 or tuple(output.shape) != (num_q, _PS_HEADS, _KV4_HEAD_DIM):
        raise RuntimeError("mla_decode_v4_ps_asm: output must be [N, 128, 512]")
    if has_lse and lse.numel() != num_q * _PS_HEADS:
        raise RuntimeError("mla_decode_v4_ps_asm: lse must be [N, 128]")
    if (
        KV.size(-1) != _KV4_HEAD_DIM
        or kvrope.size(-1) != _KV4_DIM_ROPE
        or KV.numel() // _KV4_HEAD_DIM != kvrope.numel() // _KV4_DIM_ROPE
    ):
        raise RuntimeError(
            "mla_decode_v4_ps_asm: KV / kvrope must be row-dense [rows, 512] / "
            "[rows, 64] with equal rows"
        )
    if kv_indptr.numel() < num_q + 1:
        raise RuntimeError("mla_decode_v4_ps_asm: kv_indptr needs at least N+1 entries")
    if sink.numel() != _PS_HEADS:
        raise RuntimeError("mla_decode_v4_ps_asm: sink must have 128 entries")

    if desc.dim() != 2 or desc.size(1) != 8:
        raise RuntimeError("mla_decode_v4_ps_asm: desc must be [P, 8]")
    num_p = desc.size(0)
    if not 1 <= num_p <= _PS_MAX_PARTS:
        raise RuntimeError(
            f"mla_decode_v4_ps_asm: num_partitions={num_p} must be in "
            f"[1, {_PS_MAX_PARTS}]"
        )
    if o_acc.numel() != 2 * num_p * _PS_HEADS * _KV4_HEAD_DIM:
        raise RuntimeError("mla_decode_v4_ps_asm: o_acc must be [2P, 128, 512]")
    if lse_acc.numel() != 2 * num_p * _PS_HEADS:
        raise RuntimeError("mla_decode_v4_ps_asm: lse_acc must be [2P, 128]")
    if cnt.numel() < _PS_CNT_INTS:
        raise RuntimeError(f"mla_decode_v4_ps_asm: cnt needs {_PS_CNT_INTS} entries")
    if arange.numel() < _PS_ARANGE:
        raise RuntimeError(f"mla_decode_v4_ps_asm: arange needs {_PS_ARANGE} entries")
    if num_q == 0:
        return

    cfg, co_path = _select_ps_kernel(_device_info(dev.index)[0])

    args = MlaV4PsKernelArgs()
    args.ptr_O_acc = o_acc.data_ptr()
    args.ptr_out = output.data_ptr()
    args.ptr_L_acc = lse_acc.data_ptr()
    args.ptr_lse = lse.data_ptr() if has_lse else None
    args.ptr_Q = Q.data_ptr()
    args.ptr_KV = KV.data_ptr()
    args.ptr_LTP = kv_indptr.data_ptr()
    args.ptr_LTD = kv_page_indices.data_ptr()
    args.ptr_desc = desc.data_ptr()
    args.scalar_f = _SCALAR_F
    args.s_gqa_ratio = _PS_HEADS
    args.s_kv_split = 1
    args.ptr_cnt = cnt.data_ptr()
    args.s_num_p = num_p
    args.s_num_q = num_q
    args.s_plan = _PS_PLAN_CFG
    args.ptr_QTP = arange.data_ptr()
    args.ptr_STP = arange.data_ptr()
    args.out_16_nosplit = 1
    args.ptr_QROPE = qrope.data_ptr()
    args.ptr_KVROPE = kvrope.data_ptr()
    args.ptr_sink = sink.data_ptr()

    with device_guard(dev.index):
        func = get_function(co_path, cfg["knl_name"], dev.index)
        launch_co(
            func,
            (2 * num_p, 1, 1),
            (256, 1, 1),
            args,
            stream=current_raw_stream(dev.index),
        )


mla_decode_v4_ps_asm = register_asm_custom_op(
    "mla_decode_v4_ps_asm",
    mla_decode_v4_ps_asm_eager,
    mutates_args=["o_acc", "lse_acc", "desc", "cnt", "output", "lse"],
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
def _fused_co_for(arch, q_type, kv_type, gqa, qseqlen):
    cfg = _find_kernel_cfg(
        arch, _MLA_V4_FUSED_CSV, q_type, kv_type, gqa, 0, 0, 0, qseqlen, 0
    )
    if cfg is None:
        return None
    co_path = os.path.join(AITER_ASM_DIR, arch, _MLA_V4_SUBDIR, cfg["co_name"])
    if not os.path.isfile(co_path):
        return None
    return cfg, co_path


def is_mla_v4_fused_eligible(Q, KV, max_seqlen_q, num_kv_splits):
    """Compile-safe static dispatch predicate for the shipped fused variant.

    Registry and filesystem validation intentionally happen inside the custom
    op's eager implementation, outside the Dynamo-traced region.
    """
    return _fused_eligible(
        get_gfx_from_device(Q.device), Q, KV, max_seqlen_q, num_kv_splits
    )


def _fused_eligible(arch, Q, KV, max_seqlen_q, num_kv_splits):
    if arch != "gfx1250":
        return False
    nsplit = int(num_kv_splits)
    if not (2 <= nsplit <= MLA_V4_FUSED_MAX_SPLITS):
        return False
    if os.environ.get("AITER_MLA_V4_FUSED", "1") == "0":
        return False
    gqa = Q.size(1) // KV.size(2)
    return (
        Q.dtype == torch.float8_e4m3fn
        and KV.dtype == torch.float8_e4m3fn
        and gqa == 32
        and int(max_seqlen_q) == 1
    )


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
    dev = Q.device.index
    runtime_gfx, warp_size = _device_info(dev)
    found = None
    if _fused_eligible(runtime_gfx, Q, KV, max_seqlen_q, nsplit):
        found = _fused_co_for(
            runtime_gfx,
            dtype_str(Q),
            dtype_str(KV),
            Q.size(1) // KV.size(2),
            int(max_seqlen_q),
        )
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
    if KV.size(1) != 1:
        raise ValueError(
            "mla_decode_v4_fused_asm_gfx1250: only support KV page_size==1"
        )
    if Q.size(2) != _KV4_HEAD_DIM or KV.size(3) != _KV4_HEAD_DIM:
        raise ValueError(
            f"mla_decode_v4_fused_asm_gfx1250: Q and KV head_size must be "
            f"{_KV4_HEAD_DIM}, got {Q.size(2)} / {KV.size(3)}"
        )
    if qrope.size(-1) != _KV4_DIM_ROPE or kvrope.size(-1) != _KV4_DIM_ROPE:
        raise ValueError(
            f"mla_decode_v4_fused_asm_gfx1250: qrope / kvrope last dim must be "
            f"{_KV4_DIM_ROPE}"
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
    args.scalar_f = _SCALAR_F
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
    with device_guard(dev):
        func = get_function(co_path, cfg["knl_name"], dev)
        launch_co_cluster(
            func,
            (nsplit * gdx, num_seqs, 1),
            (4 * warp_size, 1, 1),
            args,
            cluster_dim=(nsplit, 1, 1),
            stream=current_raw_stream(dev),
        )


mla_decode_v4_fused_asm_gfx1250 = register_asm_custom_op(
    "mla_decode_v4_fused_asm_gfx1250",
    mla_decode_v4_fused_asm_gfx1250_eager,
    mutates_args=["splitData", "splitLse", "output"],
)
