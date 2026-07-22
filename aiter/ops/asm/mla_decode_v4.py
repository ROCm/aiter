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
    register_asm_custom_op,
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
    max_seqlen_q: int,
    softmax_scale: float,
    out_16_nosplit: int,
    num_kv_splits: int,
    splitData: torch.Tensor,
    splitLse: torch.Tensor,
    output: torch.Tensor,
    valid_split_count: torch.Tensor | None = None,
    use_valid_split_count_reduce: int = 0,
    kv_last_page_lens: torch.Tensor | None = None,
):
    """gfx1250 v4 nm decode stage1 launch (eager, pure Python + ctypes) — Python
    peer of ``aiter.mla_decode_v4_asm`` (asm_mla_v4.cu) restricted to the gfx1250
    preload path. Same call signature; ``softmax_scale`` and ``split_indptr``
    are accepted for parity but unused on this ABI (the preload kernarg carries
    neither: the kernel hardcodes 1/sqrt(512) and derives splits from
    s_kv_split).

    This is the raw launcher: lowest host overhead, but opaque to TorchDynamo.
    Prefer the :func:`mla_decode_v4_asm_gfx1250` dispatcher, which routes to the
    ``torch.compile``-safe custom op while tracing and here otherwise."""
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


# ---------------------------------------------------------------------------
# torch.compile support.
#
# The launcher above is pure Python + ctypes, so it graph-breaks under
# `torch.compile(fullgraph=True)`. Exposing it as an aiter custom op via the
# generic `register_asm_custom_op` helper (asm_utils) makes Dynamo treat the
# launch as one opaque graph node. This does NOT change the eager dispatch in
# aiter/mla.py; callers opt in via
# `torch.ops.aiter.mla_decode_v4_asm_gfx1250(...)`. Only the schema-clean,
# type-annotated adapter below is op-specific — the registration + no-op fake
# are shared.
# ---------------------------------------------------------------------------
# TODO(mla-pyco): remove this adapter once ``mla_decode_v4_asm_gfx1250_eager``
# can be registered as a custom op directly. ``_eager`` is now fully type
# annotated, so ``torch.library.infer_schema`` can derive a schema from it.
# The ONLY remaining reason this adapter still exists is ARG ORDER (verified
# empirically): ``_eager`` uses the C-ABI parity order (SymInt scalars BEFORE
# the mutated buffers), and registering that order makes ``fullgraph=True`` fail
# ("Attempted to call function marked as skipped"), because a SymInt ahead of
# the mutated tensors breaks torch's auto-functionalization arg boxing. The
# buffer-first order below is what makes fullgraph pass (12/12); this adapter
# just reorders to buffer-first and forwards to ``_eager``.
#
# To drop it in the future, move the mutated buffers (``splitData`` /
# ``splitLse`` / ``output`` / ``valid_split_count``) ahead of the SymInt scalars
# in ``_eager`` itself, then pass ``_eager`` straight to
# ``register_asm_custom_op`` and update the ``mla_decode_v4_asm_gfx1250``
# dispatcher to the new arg order. The cost is that ``_eager`` stops mirroring
# ``aiter.mla_decode_v4_asm``'s C-ABI signature. Alternatively, revisit if a
# newer torch fixes the SymInt-before-mutated-tensor auto-functionalization
# ordering constraint, which would remove the need entirely.
def _mla_decode_v4_asm_gfx1250_op(
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
    use_valid_split_count_reduce: int,
    kv_last_page_lens: torch.Tensor | None = None,
) -> None:
    """Schema-clean, `torch.compile`-safe entry point for the gfx1250 v4 nm
    launch. Thin adapter: reorders args to the positional signature of
    :func:`mla_decode_v4_asm_gfx1250` (which carries keyword defaults torch
    schema inference cannot represent) and forwards to it.

    NOTE: the mutated tensors (`splitData` / `splitLse` / `output` /
    `valid_split_count`) are deliberately placed BEFORE the SymInt scalars here.
    Putting a value-0 SymInt (e.g. out_16_nosplit=0) ahead of the mutated tensors
    trips torch's auto-functionalization arg boxing under
    `torch.compile(fullgraph=True)` (it mis-types a later SymInt as a Tensor), so
    this buffer-first order is load-bearing, not cosmetic."""
    mla_decode_v4_asm_gfx1250_eager(
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
        valid_split_count,
        use_valid_split_count_reduce,
        kv_last_page_lens,
    )


# Pure in-place op (writes splitData/splitLse/output/valid_split_count), so the
# default no-op fake in register_asm_custom_op suffices — no fake_impl needed.
mla_decode_v4_asm_gfx1250_compiled = register_asm_custom_op(
    "mla_decode_v4_asm_gfx1250",
    _mla_decode_v4_asm_gfx1250_op,
    mutates_args=["splitData", "splitLse", "output", "valid_split_count"],
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
):
    """gfx1250 v4 nm decode stage1 launch — always via the registered custom op.

    Signature-compatible drop-in for the raw launcher, so callers (aiter/mla.py)
    need no change. It ALWAYS routes through the custom op
    ``torch.ops.aiter.mla_decode_v4_asm_gfx1250``: in eager it dispatches
    straight to the ctypes launcher, and under ``torch.compile`` /
    ``torch.export`` it becomes ONE opaque, ``fullgraph=True``-safe graph node.
    This is the idiomatic "just wrap it as a custom op" pattern — one code path
    for eager + traced, no ``is_compiling()`` branch to keep in sync. The launch
    always runs on torch's current stream (as compiled/traced graphs require).

    The op reorders the mutated buffers (``splitData`` / ``splitLse`` /
    ``output`` / ``valid_split_count``) ahead of the SymInt scalars, as required
    by torch's auto-functionalization (see ``_mla_decode_v4_asm_gfx1250_op``).
    ``valid_split_count`` is an optional mutated tensor (``Tensor(a!)?``), so
    ``None`` is accepted (null scratch ptr) in eager and under compile alike —
    no separate fallback path is needed.
    """
    torch.ops.aiter.mla_decode_v4_asm_gfx1250(
        Q,
        qrope,
        KV,
        kvrope,
        qo_indptr,
        kv_indptr,
        kv_page_indices,
        split_indptr,
        sink,
        splitData,
        splitLse,
        output,
        valid_split_count,
        max_seqlen_q,
        softmax_scale,
        int(out_16_nosplit),
        int(num_kv_splits),
        int(use_valid_split_count_reduce),
        kv_last_page_lens,
    )
