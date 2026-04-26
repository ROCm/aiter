# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Derived from ROCm/FlyDSL kernels/moe_blockscale_2stage.py (Apache-2.0).
# Upstream ref: ROCm/FlyDSL commit 8bee73f (verified to contain the source).
"""FlyDSL Blockscale MOE kernel management (per_1x128 FP8/FP8, g1u1).

Mirrors aiter/ops/flydsl/moe_kernels.py for the BLOCKSCALE variant used by
DSV4 and other models with ``weight_block_size: [128, 128]``
(``QuantType.per_1x128``).

Kernel name pattern:
    flydsl_moe{stage}_afp8_wfp8_{out}_blockscale_t{M}x{N}x{K}[_{mode}]

Source of truth: ROCm/FlyDSL ``kernels/moe_blockscale_2stage.py`` —
``compile_moe_blockscale_gemm1`` and ``compile_moe_blockscale_gemm2``.

Sprint 4 (issue sunway513/atom#37 W4.5 accuracy half) ports the kernel
implementation here so aiter's ``fused_moe`` dispatcher can route DSV4's
per_1x128 quant scheme through FlyDSL instead of CK MoE (which has an
ABI mismatch with this checkpoint, producing gibberish outputs).
"""

from __future__ import annotations

import re
from typing import Dict, Optional

import functools
import logging
import os
from contextlib import contextmanager

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext  # noqa: F401
from flydsl.expr import arith
from flydsl.expr import buffer_ops, gpu, rocdl, vector  # noqa: F401
from flydsl.expr import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr  # noqa: F401

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf, memref  # noqa: F401
from flydsl._mlir.dialects import math as math_dialect  # noqa: F401
from flydsl.expr.typing import T
from flydsl.expr.arith import ArithValue  # noqa: F401

from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    crd2idx,
    lds_load_pack_k32,  # noqa: F401  # re-exported for parity with upstream
    lds_store_4b_xor16,
    lds_store_8b_xor16,
    lds_store_16b_xor16,
    make_preshuffle_b_layout,
    load_b_pack_k32,
    tile_chunk_coord_i32,
    swizzle_xor16,
)
from aiter.ops.flydsl.kernels.mfma_epilogues import (
    c_shuffle_epilog,
    default_epilog,
    mfma_epilog,
)

# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------

_KERNEL_PARAMS: Dict[str, Dict] = {}

# Kernel-name suffix parser: optional waves_per_eu (_w{N}), b_nt (_bnt{N}),
# xcd_swizzle (_xcd{N}). All optional — defaults handled in lookup.
_SUFFIX_RE = re.compile(
    r"(?:_w(?P<wpe>\d+))?(?:_bnt(?P<bnt>\d+))?(?:_xcd(?P<xcd>\d+))?$"
)


def flydsl_blockscale_kernel_name(
    stage: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    *,
    waves_per_eu: Optional[int] = None,
    b_nt: Optional[int] = None,
    xcd_swizzle: Optional[int] = None,
    mode: str = "",
) -> str:
    """Construct kernel name.

    Pattern: ``flydsl_moe{stage}_afp8_wfp8_bf16_blockscale_t{M}x{N}x{K}[_{mode}][_w{WPE}][_bnt{B}][_xcd{X}]``.
    """
    name = f"flydsl_moe{stage}_afp8_wfp8_bf16_blockscale_t{tile_m}x{tile_n}x{tile_k}"
    if mode:
        name += f"_{mode}"
    if waves_per_eu is not None and waves_per_eu != 1:
        name += f"_w{waves_per_eu}"
    if b_nt is not None and b_nt != 0:
        name += f"_bnt{b_nt}"
    if xcd_swizzle is not None and xcd_swizzle > 0:
        name += f"_xcd{xcd_swizzle}"
    return name


def get_flydsl_blockscale_stage1_kernels() -> Dict[str, Dict]:
    """Generate FlyDSL blockscale stage1 kernel registry (FP8/FP8, bf16 out)."""
    kernels: Dict[str, Dict] = {}
    # Tile space: tile_m for token block size; tile_n for inter_dim block;
    # tile_k for K-stride. ScaleBlockK is fixed to 128 (per_1x128 quant).
    for tm in (32, 64, 128):
        for tn in (128, 256):
            # tile_k must be a multiple of ScaleBlockK=128 to align with the
            # block-scale stride. Allow tile_k=128 (single block) and
            # tile_k=256 (two blocks).
            for tk in (128, 256):
                for wpe in (1, 2, 3, 4):
                    for bnt in (0, 2):
                        name = flydsl_blockscale_kernel_name(
                            1, tm, tn, tk, waves_per_eu=wpe, b_nt=bnt
                        )
                        kernels[name] = {
                            "stage": 1,
                            "tile_m": tm,
                            "tile_n": tn,
                            "tile_k": tk,
                            "waves_per_eu": wpe,
                            "b_nt": bnt,
                            "scale_block_k": 128,
                            "a_dtype": "fp8",
                            "b_dtype": "fp8",
                            "out_dtype": "bf16",
                        }
    _KERNEL_PARAMS.update(kernels)
    return kernels


def get_flydsl_blockscale_stage2_kernels() -> Dict[str, Dict]:
    """Generate FlyDSL blockscale stage2 kernel registry (FP8/FP8, bf16 out).

    Stage2 uses ``_atomic`` mode by default (atomic add into output buffer).
    Optional ``_persist`` mode for persistent-CTA execution.
    """
    kernels: Dict[str, Dict] = {}
    for tm in (32, 64, 128):
        for tn in (128, 256):
            for tk in (128, 256):
                for mode in ("atomic", "atomic_persist"):
                    for wpe in (1, 2, 3, 4):
                        for bnt in (0, 2):
                            name = flydsl_blockscale_kernel_name(
                                2,
                                tm,
                                tn,
                                tk,
                                waves_per_eu=wpe,
                                b_nt=bnt,
                                mode=mode,
                            )
                            kernels[name] = {
                                "stage": 2,
                                "tile_m": tm,
                                "tile_n": tn,
                                "tile_k": tk,
                                "waves_per_eu": wpe,
                                "b_nt": bnt,
                                "scale_block_k": 128,
                                "a_dtype": "fp8",
                                "b_dtype": "fp8",
                                "out_dtype": "bf16",
                                "mode": mode,
                            }
    _KERNEL_PARAMS.update(kernels)
    return kernels


def get_flydsl_blockscale_kernel_params(name: str) -> Optional[Dict]:
    """Lookup kernel params by name. Auto-populates registry on first call."""
    if not _KERNEL_PARAMS:
        get_flydsl_blockscale_stage1_kernels()
        get_flydsl_blockscale_stage2_kernels()
    return _KERNEL_PARAMS.get(name)


@contextmanager
def _if_then(if_op):
    """Compat helper for SCF IfOp then-region across old/new Python APIs."""
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


@contextmanager
def _if_else(if_op):
    """Compat helper for SCF IfOp else-region across old/new Python APIs."""
    if getattr(if_op, "else_block", None) is None:
        raise RuntimeError("IfOp has no else block")
    with ir.InsertionPoint(if_op.else_block):
        try:
            yield if_op.else_block
        finally:
            blk = if_op.else_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


@functools.lru_cache(maxsize=1024)
def compile_moe_blockscale_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    scale_block_k: int = 128,
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    waves_per_eu: int | None = None,
):
    """Compile stage1 kernel (`moe_gemm1`) and return the compiled executable.

    in_dtype:
      - "fp8": X/W are fp8
      - "fp16": X/W are fp16
      - "int8": X/W are int8 (X is [tokens, K])
      - "int8smooth": X/W are int8, but X is pre-expanded to [tokens*topk, K] with per-(token,slot)
        quant scales (used to emulate MoE smoothquant behavior where each (token,slot)->expert route can
        have a distinct input scaling before quantization).
      - "int4": W4A8 path: X is int8, W is packed int4 (2 values per byte) unpacked to int8 in-kernel
    """

    gpu_arch = get_hip_arch()
    _is_gfx950 = str(gpu_arch).startswith("gfx95")
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    in_dtype = "fp8"  # blockscale is FP8-only
    is_f16 = in_dtype == "fp16"
    elem_bytes = 2 if is_f16 else 1
    if out_dtype not in ("f16", "bf16"):
        raise ValueError(f"out_dtype must be 'f16' or 'bf16', got {out_dtype!r}")
    # NOTE: don't materialize MLIR types outside an active MLIR Context.
    out_mlir = lambda: (
        (lambda ty: ty() if callable(ty) else ty)(
            T.f16 if out_dtype == "f16" else T.bf16
        )
    )
    tile_k_bytes = int(tile_k) * int(elem_bytes)
    # K64-byte micro-step: always 64 bytes per `ku`. For fp16 this is 32 elements.
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={elem_bytes})"
        )
    is_int4 = in_dtype == "int4"
    # INT4 here means W4A8: X is int8, W is packed int4 and unpacked to int8 in-kernel.
    is_int8 = (in_dtype == "int8") or is_int4
    x_is_token_slot = in_dtype == "int8smooth"
    # "int8smooth" still uses int8 MFMA, but X/scale_x are provided per (token,slot).
    is_int8 = is_int8 or x_is_token_slot

    # Blockscale compile-time constants (K=model_dim for stage1)
    if model_dim % scale_block_k != 0:
        raise ValueError(
            f"model_dim ({model_dim}) must be divisible by scale_block_k ({scale_block_k})"
        )
    if (2 * inter_dim) % 128 != 0:
        raise ValueError(
            f"2*inter_dim ({2*inter_dim}) must be divisible by 128 (ScaleBlockN)"
        )
    sb_per_tile_s1 = tile_k // scale_block_k  # scale blocks per tile (in K dim)
    ku_per_sb_s1 = scale_block_k // 64  # K64-steps per scale block = 2
    nblk_k_w1 = model_dim // scale_block_k  # K-blocks in W1 (=scale_k)
    nblk_n_w1 = (2 * inter_dim) // 128  # N-blocks in W1 (ScaleBlockN=128)

    mfma_i32_k32 = None
    if is_int8:
        mfma_i32_k32 = getattr(rocdl, "mfma_i32_16x16x32i8", None) or getattr(
            rocdl, "mfma_i32_16x16x32_i8", None
        )
        if mfma_i32_k32 is None:
            raise AttributeError(
                "INT8 K32 MFMA op not found: expected `rocdl.mfma_i32_16x16x32i8` "
                "(or `rocdl.mfma_i32_16x16x32_i8`)."
            )

    DYN = ir.ShapedType.get_dynamic_size()
    size_out = DYN
    size_x = DYN
    # W is packed int4 for W4A8: 2 values per byte.
    size_w = (
        (experts * (2 * inter_dim) * model_dim) // 2
        if is_int4
        else (experts * (2 * inter_dim) * model_dim)
    )
    size_sorted = DYN
    size_expert_ids = DYN

    total_threads = 256
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={elem_bytes}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads
    # Keep MoE stage1 X gmem->LDS pipeline consistent with the optimized GEMM kernel:
    # split into <=16B pieces and use `fly.copy(load-only)` for buffer_load_dwordx4.
    # (Compute the split lens inside the kernel so the code matches GEMM structure.)

    # LDS128 mode (same idea as test_preshuffle_gemm.py):
    # - LDS stride == tile_k (no extra padding) + XOR16 swizzle
    # - Use ds_{read,write}_b128 (16B) and extract 8B halves for MFMA steps
    _ck_lds128 = os.environ.get("FLYDSL_CK_LDS128", "1") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    pad_k = 0 if _ck_lds128 else 8
    lds_stride = tile_k + pad_k
    if use_cshuffle_epilog is None:
        use_cshuffle_epilog = os.environ.get("FLYDSL_MOE_STAGE1_CSHUFFLE", "1") in (
            "1",
            "true",
            "True",
            "YES",
            "yes",
        )
    use_cshuffle_epilog = bool(use_cshuffle_epilog)
    if out_dtype != "f16" and use_cshuffle_epilog:
        raise ValueError(
            "stage1 cshuffle epilog currently supports only f16 output (out_dtype='f16')"
        )

    epilog_tag = "cshuffle" if use_cshuffle_epilog else "direct"
    # IMPORTANT: module name participates in FlyDSL's compile cache key.
    # Keep an explicit ABI tag so signature changes can't accidentally reuse an old binary.
    _wpe_tag = f"_wpe{waves_per_eu}" if waves_per_eu is not None else ""
    module_name = (
        f"mfma_moe1_bs_{in_dtype}_{out_dtype}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}{_wpe_tag}"
        f"_abi8"  # scf.for main loop (reduced ISA size)
    ).replace("-", "_")

    # ── LDS sizing (pure Python; no MLIR Context needed) ─────────────────────
    _use_cshuffle_epilog = bool(use_cshuffle_epilog)
    lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(elem_bytes)
    lds_out_bytes = 2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    lds_total_bytes = max(lds_x_bytes, lds_out_bytes)
    lds_total_elems = lds_total_bytes if elem_bytes == 1 else (lds_total_bytes // 2)

    lds_alloc_bytes = int(lds_total_elems) * int(elem_bytes)
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_alloc_bytes

    @flyc.kernel(name=module_name)
    def moe_blockscale_gemm1(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_max_token_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
    ):
        tokens_in = arith.index_cast(T.index, i32_tokens_in)
        inter_in = arith.index_cast(T.index, i32_inter_in)
        k_in = arith.index_cast(T.index, i32_k_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        tokens_i32_v = i32_tokens_in
        k_i32_v = i32_k_in
        x_elem = T.f16 if is_f16 else (T.i8 if is_int8 else T.f8)
        # For int4, weights are stored as packed bytes (i8) and unpacked to i8 packs.
        w_elem = T.f16 if is_f16 else (T.i8 if is_int8 else T.f8)
        vec16_elems = 16 if elem_bytes == 1 else 8
        vec8_elems = 8 if elem_bytes == 1 else 4
        vec4_elems = 4 if elem_bytes == 1 else 2
        vec8_x = T.vec(vec8_elems, x_elem)
        vec16_x = T.vec(vec16_elems, x_elem)

        def silu(x):
            # device fast path:
            #   emu = exp(-x)  ~= exp2(log2e * (-x))  -> v_exp_f32
            #   sig = rcp(1 + emu)                   -> v_rcp_f32
            #   y = x * sig
            #
            # Using llvm.amdgcn intrinsics prevents lowering to the div_scale/div_fixup
            # sequences that introduce extra compares/cndmasks.
            t = x * (-1.4426950408889634)  # -log2(e)
            emu = rocdl.exp2(T.f32, t)
            den = 1.0 + emu
            sig = rocdl.rcp(T.f32, den)
            return x * sig

        acc_init = (
            arith.constant_vector(0, T.i32x4)
            if is_int8
            else arith.constant_vector(0.0, T.f32x4)
        )

        # Layouts
        layout_x = fx.make_layout((tokens_i32_v, k_i32_v), stride=(k_i32_v, 1))

        # B preshuffle layout: match GEMM test helper exactly.
        c_n_total = arith.index(experts * (2 * inter_dim))
        kpack_bytes = 8 if is_int4 else 16
        b_layout = make_preshuffle_b_layout(
            arith,
            c_n=c_n_total,
            c_k=k_in,
            kpack_bytes=kpack_bytes,
            elem_bytes=elem_bytes,
        )
        layout_b = b_layout.layout_b
        c_k0 = (k_in * arith.index(int(elem_bytes))) // fx.Index(64)

        shape_lds = fx.make_shape(tile_m, tile_k)
        stride_lds = fx.make_stride(lds_stride, 1)
        layout_lds = fx.make_layout(shape_lds, stride_lds)

        tx = gpu.thread_id("x")
        # Align with Aiter launch mapping (NSwizzle==false):
        # - blockIdx.x -> N dimension (tile along inter_dim)
        # - blockIdx.y -> expert-block id / M dimension (tile along sorted M)
        by = gpu.block_id("x")  # tile along inter_dim
        bx = gpu.block_id("y")  # tile along sorted M

        # Block validity: compute as early as possible so invalid blocks skip all buffer-resource
        # setup, LDS pointer math, and gmem prefetch work.
        bx_m = bx * fx.Index(tile_m)
        maxids_rsrc = buffer_ops.create_buffer_resource(
            arg_max_token_ids, max_size=False, num_records_bytes=fx.Index(4)
        )
        max_token_id_i32 = buffer_ops.buffer_load(
            maxids_rsrc, fx.Index(0), vec_width=1, dtype=T.i32
        )
        bx_m_i32 = arith.index_cast(T.i32, bx_m)
        blk_valid = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, max_token_id_i32)
        # Common constants/atoms (hoisted): keep IR small like GEMM.
        # XOR16 swizzle parameter (in bytes; constant, power-of-two in our configs).
        k_blocks16 = arith.index(tile_k_bytes // 16)
        layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

        # Everything below is gated by `blk_valid` to avoid doing buffer-resource setup and
        # gmem work for padding blocks.
        _if_blk = scf.IfOp(blk_valid)
        with _if_then(_if_blk):
            base_ptr = allocator.get_base()
            lds_x_ptr = SmemPtr(
                base_ptr,
                lds_alloc_offset,
                (T.f16 if is_f16 else (T.i8 if is_int8 else T.f8)),
                shape=(lds_total_elems,),
            )
            lds_x = lds_x_ptr.get()
            # Alias LDS bytes as fp16 for optional CShuffle epilogue.
            lds_out = (
                SmemPtr(
                    base_ptr, lds_x_ptr.byte_offset, T.f16, shape=(tile_m * tile_n,)
                ).get()
                if _use_cshuffle_epilog
                else None
            )

            # Buffer resources: for dynamic memrefs, provide `num_records_bytes` explicitly so
            # hardware OOB behavior is stable (otherwise it falls back to a large max size).
            c_topk = fx.Index(topk)

            # X: [tokens, k] bytes = tokens*k*elem_bytes
            x_rows = tokens_in * (c_topk if x_is_token_slot else fx.Index(1))
            x_nbytes_idx = x_rows * k_in * arith.index(int(elem_bytes))
            x_rsrc = buffer_ops.create_buffer_resource(
                arg_x,
                max_size=False,
                num_records_bytes=arith.index_cast(T.i64, x_nbytes_idx),
            )

            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)

            # OUT: [tokens, topk, inter] f16/bf16 -> bytes = tokens*topk*inter*out_elem_bytes
            out_elem_bytes = 2  # f16/bf16
            out_nbytes_idx = tokens_in * c_topk * inter_in * fx.Index(out_elem_bytes)
            out_rsrc = buffer_ops.create_buffer_resource(
                arg_out,
                max_size=False,
                num_records_bytes=arith.index_cast(T.i64, out_nbytes_idx),
            )

            # fp16 path ignores scales completely (implicit scale=1.0).
            if is_f16:
                sx_rsrc = None
                sw_rsrc = None
            else:
                # scale_x: [nblk_k_w1, tokens] f32 transposed -> total = nblk_k_w1 * tokens
                sx_nbytes_idx = arith.index(nblk_k_w1) * tokens_in * fx.Index(4)
                sx_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_x,
                    max_size=False,
                    num_records_bytes=arith.index_cast(T.i64, sx_nbytes_idx),
                )
                sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)

            sorted_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_token_ids, max_size=False
            )
            sorted_w_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_weights, max_size=False
            )

            # expert ids: [blocks] i32 -> bytes = size_expert_ids_in*4
            expert_rsrc = buffer_ops.create_buffer_resource(
                arg_expert_ids,
                max_size=False,
                num_records_bytes=arith.index_cast(
                    T.i64, size_expert_ids_in * fx.Index(4)
                ),
            )

            # Expert id for this M tile (keep address math in `index`)
            expert_i32 = buffer_ops.buffer_load(
                expert_rsrc, bx, vec_width=1, dtype=T.i32
            )
            expert_idx = arith.index_cast(T.index, expert_i32)
            inter2_idx = arith.index(2 * inter_dim)
            expert_off_idx = expert_idx * inter2_idx  # index

            # ---- X gmem->reg prefetch (match preshuffle GEMM mapping) ----
            # Prefer 16B buffer-load (dwordx4). If the per-thread byte count isn't divisible by
            # 16, fall back to 8B (dwordx2) or 4B (dword) loads. For fp16 we require 16B.
            if is_f16:
                if bytes_per_thread_x % 16 != 0:
                    raise ValueError(
                        f"[fp16] bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 16"
                    )
                x_load_bytes = 16
            else:
                if bytes_per_thread_x % 16 == 0:
                    x_load_bytes = 16
                elif bytes_per_thread_x % 8 == 0:
                    x_load_bytes = 8
                elif bytes_per_thread_x % 4 == 0:
                    x_load_bytes = 4
                else:
                    raise ValueError(
                        f"bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 4 to use the dword-indexed load mapping."
                    )
            num_x_loads = bytes_per_thread_x // x_load_bytes
            chunk_i32 = x_load_bytes // 4  # dwords per chunk (1/2/4)

            c_k_div4 = (k_in * arith.index(int(elem_bytes))) // fx.Index(4)
            c_k_div4_i32 = arith.index_cast(T.i32, c_k_div4)
            layout_x_div4 = fx.make_layout(
                (tokens_i32_v, c_k_div4_i32), stride=(c_k_div4_i32, 1)
            )
            tile_k_dwords = (int(tile_k) * int(elem_bytes)) // 4
            layout_x_tile_div4 = fx.make_layout(
                (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
            )
            c_chunk_i32 = fx.Index(chunk_i32)
            tx_i32_base = tx * c_chunk_i32
            mask24 = fx.Int32(0xFFFFFF)
            # Keep i32 constants available for epilogue index math.
            tokens_i32 = arith.index_cast(T.i32, tokens_in)
            topk_i32 = fx.Int32(topk)

            def x_tile_chunk_coord_i32(i: int):
                return tile_chunk_coord_i32(
                    arith,
                    tx_i32_base=tx_i32_base,
                    i=i,
                    total_threads=total_threads,
                    layout_tile_div4=layout_x_tile_div4,
                    chunk_i32=chunk_i32,
                )

            # decode token once (per thread's M-slice) and build a base row offset.
            x_row_base_div4 = []
            x_col_local_i32 = []
            x_row_local = []
            for i in range_constexpr(num_x_loads):
                row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                x_row_local.append(row_local)
                x_col_local_i32.append(col_local_i32)

                sorted_row_i = bx_m + row_local
                # NOTE: rows beyond `num_valid_ids` can contain garbage (within the allocated
                # buffer). That's OK as long as we never use an out-of-range token id to index X.
                fused_i = buffer_ops.buffer_load(
                    sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32
                )
                t_raw = fused_i & mask24
                # NOTE: aiter moe_sorting uses sentinel token_id == tokens for padding.
                # Do NOT rely on buffer OOB semantics for X loads; explicitly mask to a safe row.
                t_valid_i32 = arith.cmpi(arith.CmpIPredicate.ult, t_raw, tokens_i32)
                if x_is_token_slot:
                    s_raw = fused_i >> 24
                    # X is indexed by token-slot in **slot-major** order:
                    #   row_ts = slot * tokens + token
                    # This matches CK's moe_smoothquant output layout.
                    row_ts_i32 = s_raw * tokens_i32 + t_raw
                    row_ts_idx = arith.index_cast(T.index, row_ts_i32)
                    # Apply bounds check to token-slot index
                    row_ts_safe = t_valid_i32.select(row_ts_idx, fx.Index(0))
                    x_row_base_div4.append(row_ts_safe * c_k_div4)
                else:
                    t_idx = arith.index_cast(T.index, t_raw)
                    t_safe = t_valid_i32.select(t_idx, fx.Index(0))
                    x_row_base_div4.append(t_safe * c_k_div4)

            vec1_i32 = T.vec(1, T.i32)
            vec2_i32 = T.vec(2, T.i32)
            vec4_x = T.vec(4, x_elem)

            def load_x(idx_i32):
                """Load `x_load_bytes` bytes from X (gmem) into regs.

                For 16B, keep the fast dwordx4 path. For 8B/4B, use byte offsets.
                """
                if x_load_bytes == 16:
                    idx_elem = idx_i32 if elem_bytes == 1 else (idx_i32 * fx.Index(2))
                    return buffer_copy_gmem16_dwordx4(
                        buffer_ops,
                        vector,
                        elem_type=x_elem,
                        idx_i32=idx_elem,
                        rsrc=x_rsrc,
                        vec_elems=vec16_elems,
                        elem_bytes=elem_bytes,
                    )
                if x_load_bytes == 8:
                    return buffer_ops.buffer_load(
                        x_rsrc, idx_i32, vec_width=2, dtype=T.i32
                    )
                return buffer_ops.buffer_load(x_rsrc, idx_i32, vec_width=1, dtype=T.i32)

            def load_x_tile(base_k):
                """Prefetch the per-thread X tile portion (gmem -> regs) for a given K base (in elements)."""
                base_k_div4 = (base_k * arith.index(int(elem_bytes))) // fx.Index(4)
                parts = []
                for i in range_constexpr(num_x_loads):
                    idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                    x_vec = load_x(idx_i32)
                    if x_load_bytes == 16:
                        parts.append(vector.bitcast(T.i32x4, x_vec))
                    elif x_load_bytes == 8:
                        parts.append(x_vec)
                    else:
                        parts.append(x_vec)
                return parts

            # tx -> wave/lane (GEMM-style decomposition).
            coord_wl = fx.idx2crd(tx, layout_tx_wave_lane)
            wave_id = fx.get(coord_wl, 0)
            lane_id = fx.get(coord_wl, 1)
            coord_l16 = fx.idx2crd(lane_id, layout_lane16)
            lane_div_16 = fx.get(coord_l16, 0)
            lane_mod_16 = fx.get(coord_l16, 1)

            # Match GEMM naming/pattern: row in LDS is lane_mod_16, and col base is lane_div_16*16.
            row_a_lds = lane_mod_16
            a_kpack_elems = 16 // elem_bytes
            col_offset_base = lane_div_16 * arith.index(int(a_kpack_elems))
            col_offset_base_bytes = (
                col_offset_base
                if elem_bytes == 1
                else (col_offset_base * arith.index(int(elem_bytes)))
            )

            # Dynamic N tiling within block (same as existing kernels)
            by_n = by * fx.Index(tile_n)
            num_waves = 4
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16
            c_n_per_wave = fx.Index(n_per_wave)
            wave_mod_4 = wave_id % fx.Index(4)
            n_tile_base = wave_mod_4 * c_n_per_wave

            # Precompute n_blk/n_intra for gate and up rows (GEMM-style: idx2crd/get)
            n_intra_gate = []
            n_blk_gate = []
            n_intra_up = []
            n_blk_up = []
            col_g_list = []
            inter_idx = fx.Index(inter_dim)
            # layout for (row -> (blk,intra)) where intra is 0..15
            c_n0 = c_n_total // fx.Index(16)
            c_n0_i32 = arith.index_cast(T.i32, c_n0)
            layout_n_blk_intra = fx.make_layout((c_n0_i32, 16), stride=(16, 1))
            for ni in range_constexpr(num_acc_n):
                offset = arith.index(ni * 16)
                col_g = by_n + n_tile_base
                col_g = col_g + offset
                col_g = col_g + lane_mod_16
                col_g_list.append(col_g)

                row_gate = expert_off_idx + col_g
                row_up = row_gate + inter_idx

                coord_gate = fx.idx2crd(row_gate, layout_n_blk_intra)
                n_blk_gate.append(fx.get(coord_gate, 0))
                n_intra_gate.append(fx.get(coord_gate, 1))

                coord_up = fx.idx2crd(row_up, layout_n_blk_intra)
                n_blk_up.append(fx.get(coord_up, 0))
                n_intra_up.append(fx.get(coord_up, 1))

            m_repeat = tile_m // 16
            k_unroll = tile_k_bytes // 64  # K64-byte micro-step (2x MFMA)

            # --- B Load Logic (K64) - shared layout with preshuffle GEMM ---
            def load_b_pack(base_k, ki_step, ni, blk_list, intra_list):
                return load_b_pack_k32(
                    buffer_ops,
                    arith,
                    vector,
                    arg_b=arg_w,
                    b_rsrc=w_rsrc,
                    layout_b=layout_b,
                    base_k=base_k,
                    ki_step=ki_step,
                    n_blk=blk_list[ni],
                    n_intra=intra_list[ni],
                    lane_div_16=lane_div_16,  # 0..3
                    elem_type=w_elem,
                    kpack_bytes=kpack_bytes,
                    elem_bytes=elem_bytes,
                    unpack_int4=is_int4,
                )

            def load_b_tile(base_k, blk_list, intra_list):
                """Prefetch the entire per-thread B tile (gmem -> regs) for a given K base.

                Returns a list of length `k_unroll`, where each entry is a tuple:
                  (packs_half0[ni], packs_half1[ni])  for the K64 micro-step.
                """
                b_tile = []
                for ku in range_constexpr(k_unroll):
                    packs0 = []
                    packs1 = []
                    for ni in range_constexpr(num_acc_n):
                        ki0 = (ku * 2) + 0
                        ki1 = (ku * 2) + 1
                        b0 = load_b_pack(base_k, ki0, ni, blk_list, intra_list)
                        b1 = load_b_pack(base_k, ki1, ni, blk_list, intra_list)
                        packs0.append(b0)
                        packs1.append(b1)
                    b_tile.append((packs0, packs1))
                return b_tile

            acc_gate = [arith.constant_vector(0.0, T.f32x4)] * (num_acc_n * m_repeat)
            acc_up = [arith.constant_vector(0.0, T.f32x4)] * (num_acc_n * m_repeat)

            # ---- Pipeline helpers: store X tile to LDS with ping-pong base ----
            def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                for i in range_constexpr(num_x_loads):
                    row_local = x_row_local[i]
                    col_local_i32 = x_col_local_i32[i]
                    if x_load_bytes == 16:
                        lds_store_16b_xor16(
                            arith,
                            vector,
                            lds_memref=lds_x,
                            vec16_ty=vec16_x,
                            layout_lds=layout_lds,
                            row_local=row_local,
                            col_local_i32=col_local_i32,
                            tx_c4=fx.Index(4),
                            k_blocks16=k_blocks16,
                            lds_base=lds_base,
                            vec_part_i32x4=vec_x_in_parts[i],
                            elem_bytes=elem_bytes,
                        )
                    elif x_load_bytes == 8:
                        lds_store_8b_xor16(
                            arith,
                            vector,
                            lds_memref=lds_x,
                            vec8_ty=vec8_x,
                            layout_lds=layout_lds,
                            row_local=row_local,
                            col_local_i32=col_local_i32,
                            tx_c4=fx.Index(4),
                            k_blocks16=k_blocks16,
                            lds_base=lds_base,
                            vec_part_i32x2=vec_x_in_parts[i],
                        )
                    else:
                        lds_store_4b_xor16(
                            arith,
                            vector,
                            lds_memref=lds_x,
                            vec4_ty=vec4_x,
                            layout_lds=layout_lds,
                            row_local=row_local,
                            col_local_i32=col_local_i32,
                            tx_c4=fx.Index(4),
                            k_blocks16=k_blocks16,
                            lds_base=lds_base,
                            vec_part_i32x1=vec_x_in_parts[i],
                        )

            # --- A LDS load helper for K64 (load 16B once, extract 2x i64 halves) ---
            def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_base):
                col_base_swz_bytes = swizzle_xor16(
                    curr_row_a_lds, col_base_bytes, k_blocks16
                )
                col_base_swz = (
                    col_base_swz_bytes
                    if elem_bytes == 1
                    else (col_base_swz_bytes // arith.index(int(elem_bytes)))
                )
                idx_a16 = crd2idx((curr_row_a_lds, col_base_swz), layout_lds)
                idx_a16 = idx_a16 + lds_base
                loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                return a0, a1

            # --- Blockscale pre-decode and helpers ---
            c_scale_block_k = fx.Index(scale_block_k)
            c_128 = fx.Index(128)
            c_nblk_k_w1 = fx.Index(nblk_k_w1)
            row_off_base = lane_div_16 * fx.Index(4)

            # Pre-decode sorted token IDs as i32 (constant across all K-tiles).
            # OOB buffer loads return 0, so no validity masking needed for scale values.
            _pre_t_safe_i32 = []
            for _mi in range_constexpr(m_repeat):
                _mi_safe = []
                for _ii in range_constexpr(4):
                    _row_in_tile = arith.index(_mi * 16) + row_off_base + fx.Index(_ii)
                    _sorted_row = bx_m + _row_in_tile
                    _fused_pre = buffer_ops.buffer_load(
                        sorted_rsrc, _sorted_row, vec_width=1, dtype=T.i32
                    )
                    _t_id_pre = _fused_pre & mask24
                    _t_valid_pre = arith.cmpi(
                        arith.CmpIPredicate.ult, _t_id_pre, tokens_i32
                    )
                    _t_safe_pre = _t_valid_pre.select(_t_id_pre, fx.Int32(0))
                    _mi_safe.append(_t_safe_pre)
                _pre_t_safe_i32.append(_mi_safe)

            # Pre-compute N-block indices for scale_w (constant per CTA)
            _pre_n_block_gate = []
            _pre_n_block_up = []
            for _ni in range_constexpr(num_acc_n):
                _col_base_ni_pre = by_n + n_tile_base + arith.index(_ni * 16)
                _pre_n_block_gate.append((expert_off_idx + _col_base_ni_pre) // c_128)
                _pre_n_block_up.append(
                    (expert_off_idx + inter_idx + _col_base_ni_pre) // c_128
                )

            def load_scales_s1(k_base):
                all_combined = []
                for sb in range_constexpr(sb_per_tile_s1):
                    kb = k_base // c_scale_block_k + fx.Index(sb)
                    sa_base_offset = kb * tokens_in

                    s_a_vecs = []
                    sa_base_i32 = arith.index_cast(T.i32, sa_base_offset)
                    for mi in range_constexpr(m_repeat):
                        s_a_row = []
                        for ii in range_constexpr(4):
                            t_safe_i32 = _pre_t_safe_i32[mi][ii]
                            sa_idx_i32 = sa_base_i32 + t_safe_i32
                            sa_idx = arith.index_cast(T.index, sa_idx_i32)
                            s_a_val = buffer_ops.buffer_load(
                                sx_rsrc, sa_idx, vec_width=1, dtype=T.f32
                            )
                            s_a_row.append(s_a_val)
                        s_a_vecs.append(s_a_row)

                    _sw_shared_n = n_per_wave <= 128
                    s_w_gate_vals = []
                    s_w_up_vals = []
                    for ni in range_constexpr(num_acc_n):
                        if ni == 0 or not _sw_shared_n:
                            sw_gate_idx = _pre_n_block_gate[ni] * c_nblk_k_w1 + kb
                            s_w_gate = buffer_ops.buffer_load(
                                sw_rsrc, sw_gate_idx, vec_width=1, dtype=T.f32
                            )
                            sw_up_idx = _pre_n_block_up[ni] * c_nblk_k_w1 + kb
                            s_w_up = buffer_ops.buffer_load(
                                sw_rsrc, sw_up_idx, vec_width=1, dtype=T.f32
                            )
                        s_w_gate_vals.append(s_w_gate)
                        s_w_up_vals.append(s_w_up)

                    s_a_vec4_list = []
                    for mi in range_constexpr(m_repeat):
                        s_a_vec4_list.append(
                            vector.from_elements(T.f32x4, s_a_vecs[mi])
                        )
                    all_combined.append((s_a_vec4_list, s_w_gate_vals, s_w_up_vals))
                return all_combined

            def compute_tile_bs_s1(
                acc_gate_in,
                acc_up_in,
                b_gate_tile_in,
                b_up_tile_in,
                lds_base,
                pre_scales,
                *,
                a0_prefetch=None,
            ):
                current_gate = list(acc_gate_in)
                current_up = list(acc_up_in)
                mfma_res_ty = T.f32x4

                if _is_gfx950:

                    def _pack128(x0, x1, x2, x3):
                        v4 = vector.from_elements(T.vec(4, T.i64), [x0, x1, x2, x3])
                        return vector.bitcast(T.vec(8, T.i32), v4)

                    for sb in range_constexpr(sb_per_tile_s1):
                        s_a_vec4_list, s_w_gate_vals, s_w_up_vals = pre_scales[sb]
                        ku0 = sb * ku_per_sb_s1
                        ku1 = ku0 + 1
                        bg0_p0, bg0_p1 = b_gate_tile_in[ku0]
                        bg1_p0, bg1_p1 = b_gate_tile_in[ku1]
                        bu0_p0, bu0_p1 = b_up_tile_in[ku0]
                        bu1_p0, bu1_p1 = b_up_tile_in[ku1]
                        col0 = col_offset_base_bytes + arith.index(ku0 * 64)
                        col1 = col_offset_base_bytes + arith.index(ku1 * 64)
                        for mi in range_constexpr(m_repeat):
                            curr_row = row_a_lds + arith.index(mi * 16)
                            if a0_prefetch is not None and sb == 0 and mi == 0:
                                a0, a1 = a0_prefetch
                            else:
                                a0, a1 = lds_load_packs_k64(curr_row, col0, lds_base)
                            a2, a3 = lds_load_packs_k64(curr_row, col1, lds_base)
                            a128 = _pack128(a0, a1, a2, a3)
                            s_a_v4 = s_a_vec4_list[mi]
                            pending_gate_up = None
                            for ni in range_constexpr(num_acc_n):
                                acc_idx = mi * num_acc_n + ni
                                bg128 = _pack128(
                                    bg0_p0[ni], bg0_p1[ni], bg1_p0[ni], bg1_p1[ni]
                                )
                                bu128 = _pack128(
                                    bu0_p0[ni], bu0_p1[ni], bu1_p0[ni], bu1_p1[ni]
                                )
                                blk_g = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                    mfma_res_ty,
                                    [
                                        a128,
                                        bg128,
                                        acc_init,
                                        0,
                                        0,
                                        0,
                                        0x7F7F7F7F,
                                        0,
                                        0x7F7F7F7F,
                                    ],
                                )
                                blk_u = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                    mfma_res_ty,
                                    [
                                        a128,
                                        bu128,
                                        acc_init,
                                        0,
                                        0,
                                        0,
                                        0x7F7F7F7F,
                                        0,
                                        0x7F7F7F7F,
                                    ],
                                )
                                rocdl.sched_barrier(0)
                                if pending_gate_up is not None:
                                    prev_acc_idx, prev_blk_g, prev_blk_u, prev_ni = (
                                        pending_gate_up
                                    )
                                    s_wg_bc = vector.broadcast(
                                        T.f32x4, s_w_gate_vals[prev_ni]
                                    )
                                    s_wu_bc = vector.broadcast(
                                        T.f32x4, s_w_up_vals[prev_ni]
                                    )
                                    scale_g = ArithValue(s_a_v4) * ArithValue(s_wg_bc)
                                    scale_u = ArithValue(s_a_v4) * ArithValue(s_wu_bc)
                                    current_gate[prev_acc_idx] = math_dialect.fma(
                                        prev_blk_g, scale_g, current_gate[prev_acc_idx]
                                    )
                                    current_up[prev_acc_idx] = math_dialect.fma(
                                        prev_blk_u, scale_u, current_up[prev_acc_idx]
                                    )
                                pending_gate_up = (acc_idx, blk_g, blk_u, ni)
                            if pending_gate_up is not None:
                                prev_acc_idx, prev_blk_g, prev_blk_u, prev_ni = (
                                    pending_gate_up
                                )
                                s_wg_bc = vector.broadcast(
                                    T.f32x4, s_w_gate_vals[prev_ni]
                                )
                                s_wu_bc = vector.broadcast(
                                    T.f32x4, s_w_up_vals[prev_ni]
                                )
                                scale_g = ArithValue(s_a_v4) * ArithValue(s_wg_bc)
                                scale_u = ArithValue(s_a_v4) * ArithValue(s_wu_bc)
                                current_gate[prev_acc_idx] = math_dialect.fma(
                                    prev_blk_g, scale_g, current_gate[prev_acc_idx]
                                )
                                current_up[prev_acc_idx] = math_dialect.fma(
                                    prev_blk_u, scale_u, current_up[prev_acc_idx]
                                )
                else:
                    mfma_fn = (
                        mfma_i32_k32
                        if is_int8
                        else (
                            rocdl.mfma_f32_16x16x16f16
                            if is_f16
                            else rocdl.mfma_f32_16x16x32_fp8_fp8
                        )
                    )

                    def _i64_to_v4f16(x_i64):
                        v1 = vector.from_elements(T.vec(1, T.i64), [x_i64])
                        return vector.bitcast(T.f16x4, v1)

                    def mfma_k64(acc_in, a0, a1, b0, b1):
                        if is_f16:
                            a0v = _i64_to_v4f16(a0)
                            a1v = _i64_to_v4f16(a1)
                            b0v = _i64_to_v4f16(b0)
                            b1v = _i64_to_v4f16(b1)
                            acc_mid = mfma_fn(mfma_res_ty, [a0v, b0v, acc_in, 0, 0, 0])
                            return mfma_fn(mfma_res_ty, [a1v, b1v, acc_mid, 0, 0, 0])
                        acc_mid = mfma_fn(mfma_res_ty, [a0, b0, acc_in, 0, 0, 0])
                        return mfma_fn(mfma_res_ty, [a1, b1, acc_mid, 0, 0, 0])

                    for sb in range_constexpr(sb_per_tile_s1):
                        s_a_vec4_list, s_w_gate_vals, s_w_up_vals = pre_scales[sb]
                        for mi in range_constexpr(m_repeat):
                            s_a_v4 = s_a_vec4_list[mi]
                            for ni in range_constexpr(num_acc_n):
                                acc_idx = mi * num_acc_n + ni
                                blk_g = acc_init
                                blk_u = acc_init
                                for ku_local in range_constexpr(ku_per_sb_s1):
                                    ku = sb * ku_per_sb_s1 + ku_local
                                    b_gate_packs0, b_gate_packs1 = b_gate_tile_in[ku]
                                    b_up_packs0, b_up_packs1 = b_up_tile_in[ku]
                                    ki64 = arith.index(ku * 64)
                                    col_base = col_offset_base_bytes + ki64
                                    if (
                                        (a0_prefetch is not None)
                                        and (sb == 0)
                                        and (ku_local == 0)
                                        and (mi == 0)
                                    ):
                                        a0, a1 = a0_prefetch
                                    else:
                                        a0, a1 = lds_load_packs_k64(
                                            row_a_lds + arith.index(mi * 16),
                                            col_base,
                                            lds_base,
                                        )
                                    blk_g = mfma_k64(
                                        blk_g,
                                        a0,
                                        a1,
                                        b_gate_packs0[ni],
                                        b_gate_packs1[ni],
                                    )
                                    blk_u = mfma_k64(
                                        blk_u, a0, a1, b_up_packs0[ni], b_up_packs1[ni]
                                    )
                                s_wg_bc = vector.broadcast(T.f32x4, s_w_gate_vals[ni])
                                s_wu_bc = vector.broadcast(T.f32x4, s_w_up_vals[ni])
                                scale_g = ArithValue(s_a_v4) * ArithValue(s_wg_bc)
                                scale_u = ArithValue(s_a_v4) * ArithValue(s_wu_bc)
                                current_gate[acc_idx] = math_dialect.fma(
                                    blk_g, scale_g, current_gate[acc_idx]
                                )
                                current_up[acc_idx] = math_dialect.fma(
                                    blk_u, scale_u, current_up[acc_idx]
                                )
                return current_gate, current_up

            def compute_tile(
                acc_gate_in,
                acc_up_in,
                b_gate_tile_in,
                b_up_tile_in,
                lds_base,
                *,
                prefetch_epilogue: bool = False,
                a0_prefetch=None,
            ):
                gate_list = list(acc_gate_in)
                up_list = list(acc_up_in)
                mfma_res_ty = T.i32x4 if is_int8 else T.f32x4
                mfma_fn = (
                    mfma_i32_k32
                    if is_int8
                    else (
                        rocdl.mfma_f32_16x16x16f16
                        if is_f16
                        else rocdl.mfma_f32_16x16x32_fp8_fp8
                    )
                )

                # Optional: prefetch epilogue scales while we are about to run the last MFMA tile,
                # matching the preshuffle GEMM pattern of overlapping scale loads with MFMA.
                epilogue_pf = None
                if prefetch_epilogue:
                    expert_off_pf = expert_off_idx
                    sw_gate_pf = []
                    sw_up_pf = []
                    for ni in range_constexpr(num_acc_n):
                        col_g = col_g_list[ni]
                        row_gate_idx = expert_off_pf + col_g
                        row_up_idx = row_gate_idx + inter_idx
                        sw_gate_pf.append(
                            fx.Float32(1.0)
                            if is_f16
                            else buffer_ops.buffer_load(
                                sw_rsrc, row_gate_idx, vec_width=1, dtype=T.f32
                            )
                        )
                        sw_up_pf.append(
                            fx.Float32(1.0)
                            if is_f16
                            else buffer_ops.buffer_load(
                                sw_rsrc, row_up_idx, vec_width=1, dtype=T.f32
                            )
                        )
                    epilogue_pf = (sw_gate_pf, sw_up_pf)

                def _i64_to_v4f16(x_i64):
                    v1 = vector.from_elements(T.vec(1, T.i64), [x_i64])
                    return vector.bitcast(T.f16x4, v1)

                def mfma_k64(acc_in, a0, a1, b0, b1):
                    if is_f16:
                        a0v = _i64_to_v4f16(a0)
                        a1v = _i64_to_v4f16(a1)
                        b0v = _i64_to_v4f16(b0)
                        b1v = _i64_to_v4f16(b1)
                        acc_mid = mfma_fn(mfma_res_ty, [a0v, b0v, acc_in, 0, 0, 0])
                        return mfma_fn(mfma_res_ty, [a1v, b1v, acc_mid, 0, 0, 0])
                    acc_mid = mfma_fn(mfma_res_ty, [a0, b0, acc_in, 0, 0, 0])
                    return mfma_fn(mfma_res_ty, [a1, b1, acc_mid, 0, 0, 0])

                for ku in range_constexpr(k_unroll):
                    b_gate_packs0, b_gate_packs1 = b_gate_tile_in[ku]
                    b_up_packs0, b_up_packs1 = b_up_tile_in[ku]
                    ki64 = arith.index(ku * 64)
                    col_base = col_offset_base_bytes + ki64

                    for mi in range_constexpr(m_repeat):
                        mi_val = arith.index(mi * 16)
                        curr_row_a_lds = row_a_lds + mi_val

                        if (a0_prefetch is not None) and (ku == 0) and (mi == 0):
                            a0, a1 = a0_prefetch
                        else:
                            a0, a1 = lds_load_packs_k64(
                                curr_row_a_lds, col_base, lds_base
                            )

                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            gate_list[acc_idx] = mfma_k64(
                                gate_list[acc_idx],
                                a0,
                                a1,
                                b_gate_packs0[ni],
                                b_gate_packs1[ni],
                            )
                            up_list[acc_idx] = mfma_k64(
                                up_list[acc_idx],
                                a0,
                                a1,
                                b_up_packs0[ni],
                                b_up_packs1[ni],
                            )
                return gate_list, up_list, epilogue_pf

            # ── scf.for loop helpers (acc-only loop state, CK-style) ──────
            n_accs_half = m_repeat * num_acc_n

            # ---------------- 2-stage pipeline (ping-pong LDS + B tile prefetch) ----------------
            lds_tile_elems = arith.index(tile_m * lds_stride)
            lds_base_cur = fx.Index(0)
            lds_base_nxt = lds_tile_elems

            rocdl.sched_barrier(0)

            def hot_loop_scheduler():
                mfma_per_ku = m_repeat * num_acc_n * 2 * 2
                total_mfma = k_unroll * mfma_per_ku
                rocdl.sched_group_barrier(rocdl.mask_dsrd, ku_per_sb_s1 * m_repeat, 0)
                rocdl.sched_group_barrier(rocdl.mask_mfma, total_mfma, 1)
                rocdl.sched_group_barrier(rocdl.mask_vmem_rd, num_x_loads, 2)
                rocdl.sched_group_barrier(rocdl.mask_dswr, num_x_loads, 3)
                rocdl.sched_barrier(0)

            def do_one_stage(
                acc_gate_in, acc_up_in, k_compute, k_next, lds_compute, lds_store
            ):
                """One pipeline stage: load next tile data, compute current tile, store X to LDS."""
                scale_fn = load_scales_s1
                pre_scales = scale_fn(k_compute)
                x_regs_next = load_x_tile(k_next)
                b_gate_cur = load_b_tile(k_compute, n_blk_gate, n_intra_gate)
                b_up_cur = load_b_tile(k_compute, n_blk_up, n_intra_up)

                ag, au = compute_tile_bs_s1(
                    acc_gate_in,
                    acc_up_in,
                    b_gate_cur,
                    b_up_cur,
                    lds_compute,
                    pre_scales,
                )
                store_x_tile_to_lds(x_regs_next, lds_store)
                hot_loop_scheduler()
                gpu.barrier()
                return ag, au

            # Prologue: prefetch tile0 X into LDS, sync.
            k0 = fx.Index(0)
            x_regs0 = load_x_tile(k0)
            store_x_tile_to_lds(x_regs0, lds_base_cur)
            gpu.barrier()

            lds_base_pong = lds_base_cur
            lds_base_ping = lds_base_nxt

            c2_tile_k = arith.index(tile_k * 2)
            c_tile_k = arith.index(tile_k)
            total_tiles = int(model_dim) // int(tile_k)
            pair_iters = max((total_tiles - 2) // 2, 0)
            c_k_main = pair_iters * tile_k * 2

            init_state = list(acc_gate) + list(acc_up)

            for k_iv, inner in range(0, c_k_main, tile_k * 2, init=init_state):
                n = n_accs_half
                acc_gate_in = list(inner[:n])
                acc_up_in = list(inner[n : 2 * n])

                next_k1 = k_iv + c_tile_k

                acc_gate_s0, acc_up_s0 = do_one_stage(
                    acc_gate_in, acc_up_in, k_iv, next_k1, lds_base_pong, lds_base_ping
                )

                next_k2 = k_iv + c2_tile_k

                acc_gate_s1, acc_up_s1 = do_one_stage(
                    acc_gate_s0,
                    acc_up_s0,
                    next_k1,
                    next_k2,
                    lds_base_ping,
                    lds_base_pong,
                )

                results = yield list(acc_gate_s1) + list(acc_up_s1)

            n = n_accs_half
            acc_gate = list(results[:n])
            acc_up = list(results[n : 2 * n])

            # Tail: use fresh scale decode (no dependency on prologue _pre_t_safe_idx)
            k_tail0 = k_in - c2_tile_k
            k_tail1 = k_in - c_tile_k

            acc_gate, acc_up = do_one_stage(
                acc_gate, acc_up, k_tail0, k_tail1, lds_base_pong, lds_base_ping
            )

            pre_scales_tail1 = load_scales_s1(k_tail1)
            b_gate_last = load_b_tile(k_tail1, n_blk_gate, n_intra_gate)
            b_up_last = load_b_tile(k_tail1, n_blk_up, n_intra_up)
            acc_gate, acc_up = compute_tile_bs_s1(
                acc_gate,
                acc_up,
                b_gate_last,
                b_up_last,
                lds_base_ping,
                pre_scales_tail1,
            )

            # Store epilogue to out[t, slot, inter]
            expert_off = expert_off_idx
            bx_m0 = bx_m
            tokens_i32_v = tokens_i32
            topk_i32_v = topk_i32
            inter_i32_v = fx.Int32(inter_dim)
            mask24_i32 = fx.Int32(0xFFFFFF)

            # Blockscale: dequant already done in compute_tile_bs_s1, no sw/sx needed here.

            # Epilogue hoists to keep IR + Python build time small:
            col_i32_list = []
            for ni in range_constexpr(num_acc_n):
                col_i32_list.append(arith.index_cast(T.i32, col_g_list[ni]))

            lane_div_16_mul4 = lane_div_16 * fx.Index(4)
            inter_i32_local = inter_i32_v

            if _use_cshuffle_epilog:
                if lds_out is None:
                    raise RuntimeError(
                        "CShuffle epilogue enabled but lds_out is not allocated/aliased."
                    )

                def write_row_to_lds(
                    *,
                    mi: int,
                    ii: int,
                    row_in_tile,
                    row,
                    row_base_lds,
                    col_base_local,
                    num_acc_n: int,
                    lds_out,
                ):
                    # Blockscale: dequant already done in compute_tile_bs_s1.
                    # Just apply silu + optional sorted weight.
                    if doweight_stage1:
                        tw = buffer_ops.buffer_load(
                            sorted_w_rsrc, row, vec_width=1, dtype=T.f32
                        )

                    for ni in range_constexpr(num_acc_n):
                        col_local = col_base_local + (ni * 16)

                        acc_idx = mi * num_acc_n + ni
                        vg = vector.extract(
                            acc_gate[acc_idx], static_position=[ii], dynamic_position=[]
                        )
                        vu = vector.extract(
                            acc_up[acc_idx], static_position=[ii], dynamic_position=[]
                        )

                        y = silu(vg) * vu
                        if doweight_stage1:
                            y = y * tw
                        y16 = arith.trunc_f(T.f16, y)

                        lds_idx = row_base_lds + col_local
                        v1 = vector.from_elements(T.vec(1, T.f16), [y16])
                        vector.store(v1, lds_out, [lds_idx], alignment=2)

                def precompute_row(*, row_local, row):
                    fused2 = buffer_ops.buffer_load(
                        sorted_rsrc, row, vec_width=1, dtype=T.i32
                    )
                    t2 = fused2 & mask24_i32
                    s2 = fused2 >> 24
                    return (t2 * topk_i32_v + s2) * inter_i32_local

                def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                    # Guard against sentinel token ids (t == tokens) produced by aiter moe_sorting padding.
                    # OOB buffer stores are not guaranteed to be safe on all paths, so predicate explicitly.
                    fused2 = buffer_ops.buffer_load(
                        sorted_rsrc, row, vec_width=1, dtype=T.i32
                    )
                    t2 = fused2 & mask24_i32
                    t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                    _if_valid = scf.IfOp(t_valid)
                    with _if_then(_if_valid):
                        idx0 = row_ctx
                        col_i32 = arith.index_cast(T.i32, col_g0)
                        idx_out = idx0 + col_i32
                        # Vectorized fp16 store (EVec=4).
                        buffer_ops.buffer_store(frag, out_rsrc, idx_out)

                mfma_epilog(
                    use_cshuffle=True,
                    arith=arith,
                    vector=vector,
                    gpu=gpu,
                    scf=scf,
                    range_constexpr=range_constexpr,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    e_vec=4,
                    m_repeat=m_repeat,
                    num_acc_n=num_acc_n,
                    tx=tx,
                    lane_div_16=lane_div_16,
                    lane_mod_16=lane_mod_16,
                    bx_m=bx_m,
                    by_n=by_n,
                    n_tile_base=n_tile_base,
                    lds_out=lds_out,
                    write_row_to_lds=write_row_to_lds,
                    precompute_row=precompute_row,
                    store_pair=store_pair,
                )
                return

            def _stage1_store_row(*, mi: int, ii: int, row_in_tile, row):
                # Blockscale: dequant already done in compute_tile_bs_s1.
                fused2 = buffer_ops.buffer_load(
                    sorted_rsrc, row, vec_width=1, dtype=T.i32
                )
                t2 = fused2 & mask24_i32
                s2 = fused2 >> 24
                t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)

                # out linear index base = ((t*topk + s)*inter_dim) (invariant across ni)
                idx0 = (t2 * topk_i32_v + s2) * inter_i32_local

                # Sorted weight aligned with `row` (matches aiter moe_sorting output).
                if doweight_stage1:
                    tw = buffer_ops.buffer_load(
                        sorted_w_rsrc, row, vec_width=1, dtype=T.f32
                    )

                _if_valid = scf.IfOp(t_valid)
                with _if_then(_if_valid):
                    for ni in range_constexpr(num_acc_n):
                        col_i32 = col_i32_list[ni]

                        acc_idx = mi * num_acc_n + ni
                        vg = vector.extract(
                            acc_gate[acc_idx], static_position=[ii], dynamic_position=[]
                        )
                        vu = vector.extract(
                            acc_up[acc_idx], static_position=[ii], dynamic_position=[]
                        )

                        y = silu(vg) * vu
                        if doweight_stage1:
                            y = y * tw
                        y = arith.trunc_f(out_mlir(), y)
                        idx_out0 = idx0 + col_i32
                        buffer_ops.buffer_store(y, out_rsrc, idx_out0)

            mfma_epilog(
                use_cshuffle=False,
                arith=arith,
                range_constexpr=range_constexpr,
                m_repeat=m_repeat,
                lane_div_16=lane_div_16,
                bx_m=bx_m,
                body_row=_stage1_store_row,
            )

    # ── Host launcher (flyc.jit + .launch) ────────────────────────────────
    @flyc.jit
    def launch_moe_blockscale_gemm1(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_max_token_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        inter_in = arith.index_cast(T.index, i32_inter_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        gx = inter_in // fx.Index(tile_n)
        gy = size_expert_ids_in

        launcher1 = moe_blockscale_gemm1(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_max_token_ids,
            i32_tokens_in,
            i32_inter_in,
            i32_k_in,
            i32_size_expert_ids_in,
        )
        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            T.i32, _wpe
                        )
        launcher1.launch(grid=(gx, gy, 1), block=(256, 1, 1), stream=stream)

    return launch_moe_blockscale_gemm1


@functools.lru_cache(maxsize=1024)
def compile_moe_blockscale_gemm2(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    scale_block_k: int = 128,
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    # Optional experiment: write per-(token,slot) output (no atomics) into an output shaped
    # [tokens*topk, model_dim] (or [tokens, topk, model_dim] flattened), then reduce over topk outside.
    # This can reduce atomic contention for small tokens at the cost of extra bandwidth / reduction.
    accumulate: bool = True,
    waves_per_eu: int | None = None,
):
    """Compile stage2 kernel (`moe_gemm2`) and return the compiled executable.

    in_dtype:
      - "fp8": A2/W are fp8
      - "fp16": A2/W are fp16
      - "int8": A2/W are int8
      - "int4": W4A8 path: A2 is int8, W is packed int4 unpacked to int8 in-kernel

    Stage2 output supports:
      - out_dtype="f16": fp16 half2 atomics (fast, can overflow to +/-inf for bf16 workloads)
      - out_dtype="f32": fp32 scalar atomics (slower, but avoids fp16 atomic overflow)

    `use_cshuffle_epilog` controls whether we use the LDS CShuffle epilogue before
    global atomics (recommended for performance).
    """
    gpu_arch = get_hip_arch()
    _is_gfx950 = str(gpu_arch).startswith("gfx95")
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    in_dtype = "fp8"  # blockscale is FP8-only
    is_f16 = in_dtype == "fp16"
    elem_bytes = 2 if is_f16 else 1
    out_s = str(out_dtype).strip().lower()
    if out_s not in ("f16", "fp16", "half", "bf16", "bfloat16", "f32", "fp32", "float"):
        raise ValueError(
            f"out_dtype must be 'f16', 'bf16', or 'f32', got {out_dtype!r}"
        )
    out_is_f32 = out_s in ("f32", "fp32", "float")
    out_is_bf16 = out_s in ("bf16", "bfloat16")
    if (not bool(accumulate)) and out_is_f32:
        raise ValueError(
            "compile_moe_blockscale_gemm2(accumulate=False) only supports out_dtype in {'f16','bf16'}"
        )
    is_int4 = in_dtype == "int4"
    # INT4 here means W4A8: A2 is int8, W is packed int4 and unpacked to int8 in-kernel.
    is_int8 = (in_dtype in ("int8", "int8smooth")) or is_int4

    # Blockscale compile-time constants (K=inter_dim for stage2)
    if inter_dim % scale_block_k != 0:
        raise ValueError(
            f"inter_dim ({inter_dim}) must be divisible by scale_block_k ({scale_block_k})"
        )
    if model_dim % 128 != 0:
        raise ValueError(
            f"model_dim ({model_dim}) must be divisible by 128 (ScaleBlockN)"
        )
    sb_per_tile_s2 = tile_k // scale_block_k  # scale blocks per tile (in K dim)
    ku_per_sb_s2 = scale_block_k // 64  # K64-steps per scale block = 2
    nblk_k_w2 = inter_dim // scale_block_k  # K-blocks in W2 (=scale_k)
    nblk_n_w2 = model_dim // 128  # N-blocks in W2 (ScaleBlockN=128)

    mfma_i32_k32 = None
    if is_int8:
        mfma_i32_k32 = getattr(rocdl, "mfma_i32_16x16x32i8", None) or getattr(
            rocdl, "mfma_i32_16x16x32_i8", None
        )
        if mfma_i32_k32 is None:
            raise AttributeError(
                "INT8 K32 MFMA op not found: expected `rocdl.mfma_i32_16x16x32i8` "
                "(or `rocdl.mfma_i32_16x16x32_i8`)."
            )

    DYN = ir.ShapedType.get_dynamic_size()
    size_out = DYN
    size_x = DYN
    size_sorted = DYN
    size_expert_ids_shape = DYN
    size_scale_x = DYN
    # W is packed int4 for W4A8: 2 values per byte.
    size_w = (
        (experts * model_dim * inter_dim) // 2
        if is_int4
        else (experts * model_dim * inter_dim)
    )

    total_threads = 256
    tile_k_bytes = int(tile_k) * int(elem_bytes)
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={elem_bytes})"
        )
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={elem_bytes}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads

    _ck_lds128 = os.environ.get("FLYDSL_CK_LDS128", "1") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    pad_k = 0 if _ck_lds128 else 8
    lds_stride = tile_k + pad_k
    # gfx950+ has buffer_atomic_pk_add_bf16 → bf16 can use buffer atomics (same as f16).
    # gfx942 only has global_atomic_pk_add_bf16 → must use global atomics with raw pointer.
    _has_buffer_atomic_bf16 = str(gpu_arch).startswith(("gfx95", "gfx12"))
    _needs_global_atomic_bf16 = out_is_bf16 and not _has_buffer_atomic_bf16
    if out_is_bf16:
        if not (
            gpu_arch.startswith("gfx942")
            or gpu_arch.startswith("gfx950")
            or gpu_arch.startswith("gfx12")
        ):
            raise ValueError(
                f"out_dtype='bf16' requires bf16 global atomics (gfx942/gfx950/gfx12), got arch={gpu_arch!r}"
            )

    if out_is_f32:
        # Match origin/dev_a16w4: f32 output uses scalar atomics and does NOT use the CShuffle epilogue.
        _use_cshuffle_epilog = (
            False if use_cshuffle_epilog is None else bool(use_cshuffle_epilog)
        )
        if _use_cshuffle_epilog:
            raise ValueError(
                "out_dtype='f32' does not support CShuffle epilogue (set use_cshuffle_epilog=False)."
            )
    else:
        if use_cshuffle_epilog is None:
            _use_cshuffle_epilog = os.environ.get(
                "FLYDSL_MOE_STAGE2_CSHUFFLE", "1"
            ) in (
                "1",
                "true",
                "True",
                "YES",
                "yes",
            )
        else:
            _use_cshuffle_epilog = bool(use_cshuffle_epilog)
        if not _use_cshuffle_epilog:
            raise ValueError(
                "stage2 f16 output currently requires CShuffle epilogue (FLYDSL_MOE_STAGE2_CSHUFFLE=1)."
            )

    # NOTE: Keep this as a callable so we don't require an MLIR Context at Python-time.
    def out_elem():
        ty = T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)
        return ty() if callable(ty) else ty

    epilog_tag = "cshuffle"
    # IMPORTANT: include tiling in the module name to avoid accidentally reusing a compiled
    # binary for a different (tile_m, tile_n, tile_k) configuration.
    # See stage1 note: include ABI tag to prevent binary reuse across signature changes.
    # IMPORTANT: module name participates in FlyDSL's compile cache key.
    # Dynamic-shape variant: safe to reuse across (tokens/sorted_size/size_expert_ids) at runtime.
    # Keep a distinct ABI tag so the compile cache never mixes with historical signatures.
    _wpe_tag2 = f"_wpe{waves_per_eu}" if waves_per_eu is not None else ""
    module_name = (
        f"mfma_moe2_{in_dtype}_{out_s}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}{_wpe_tag2}"
        f"_abi6"  # scale prefetch before VMEM tile loads
    ).replace("-", "_")

    # ── LDS sizing (pure Python; no MLIR Context needed) ─────────────────────
    lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(elem_bytes)
    lds_out_bytes = 2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    lds_total_bytes = max(lds_x_bytes, lds_out_bytes)
    lds_total_elems = lds_total_bytes if elem_bytes == 1 else (lds_total_bytes // 2)

    lds_alloc_bytes = int(lds_total_elems) * int(elem_bytes)
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_alloc_bytes

    _cshuffle_nlane = 32
    if bool(accumulate):
        _e_vec = 2
    else:
        _e_vec = 8 if int(tile_n) % (_cshuffle_nlane * 8) == 0 else 2
        _cshuffle_stride = _cshuffle_nlane * _e_vec
        if int(tile_n) % _cshuffle_stride != 0:
            raise ValueError(
                f"tile_n={tile_n} must be divisible by {_cshuffle_stride} when accumulate=False"
            )

    if True:

        @flyc.kernel(name=module_name)
        def moe_blockscale_gemm2(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_sorted_token_ids: fx.Tensor,
            arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor,
            arg_num_valid_ids: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_n_in: fx.Int32,
            i32_k_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
        ):
            tokens_in = arith.index_cast(T.index, i32_tokens_in)
            n_in = arith.index_cast(T.index, i32_n_in)
            k_in = arith.index_cast(T.index, i32_k_in)
            size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
            tokens_i32_v = i32_tokens_in
            k_i32_v = i32_k_in
            x_elem = T.f16 if is_f16 else (T.i8 if is_int8 else T.f8)
            # For int4, weights are stored as packed bytes (i8) and unpacked to i8 packs.
            w_elem = T.f16 if is_f16 else (T.i8 if is_int8 else T.f8)
            vec16_elems = 16 if elem_bytes == 1 else 8
            vec8_elems = 8 if elem_bytes == 1 else 4
            vec4_elems = 4 if elem_bytes == 1 else 2
            vec8_x = T.vec(vec8_elems, x_elem)
            vec16_x = T.vec(vec16_elems, x_elem)

            acc_init = (
                arith.constant_vector(0, T.i32x4)
                if is_int8
                else arith.constant_vector(0.0, T.f32x4)
            )

            # A2 layout (flatten token-slot -> M).
            topk_idx = fx.Index(topk)
            m_in = tokens_in * topk_idx
            m_i32_v = arith.index_cast(T.i32, m_in)
            layout_x = fx.make_layout((m_i32_v, k_i32_v), stride=(k_i32_v, 1))

            # B preshuffle layout: [experts*model_dim, inter_dim]
            c_n_total = arith.index(experts * model_dim)
            kpack_bytes = 8 if is_int4 else 16
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=k_in,
                kpack_bytes=kpack_bytes,
                elem_bytes=elem_bytes,
            )
            layout_b = b_layout.layout_b
            c_k0 = (k_in * arith.index(int(elem_bytes))) // fx.Index(64)

            shape_lds = fx.make_shape(tile_m, tile_k)
            stride_lds = fx.make_stride(lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            # Align with Aiter launch mapping:
            # - blockIdx.x -> N dimension (tile along model_dim)
            # - blockIdx.y -> expert-block id / M dimension (tile along sorted M)
            by = gpu.block_id("x")  # tile along model_dim
            bx = gpu.block_id("y")  # tile along sorted M

            # XOR16 swizzle parameter (in bytes; constant, power-of-two in our configs).
            k_blocks16 = arith.index(tile_k_bytes // 16)
            layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
            layout_lin_rowcol = fx.make_layout((tile_m, tile_k), stride=(tile_k, 1))

            base_ptr = allocator.get_base()
            lds_x_ptr = SmemPtr(
                base_ptr,
                lds_alloc_offset,
                (T.f16 if is_f16 else (T.i8 if is_int8 else T.f8)),
                shape=(lds_total_elems,),
            )
            lds_x = lds_x_ptr.get()
            # Alias the same underlying LDS bytes as f16/bf16 for epilogue shuffle.
            lds_out = (
                SmemPtr(
                    base_ptr,
                    lds_x_ptr.byte_offset,
                    (T.bf16 if out_is_bf16 else T.f16),
                    shape=(tile_m * tile_n,),
                ).get()
                if _use_cshuffle_epilog
                else None
            )

            # Buffer resources.
            # For dynamic memrefs, `max_size=False` cannot infer the logical size from the memref *type*,
            # so we should pass `num_records_bytes` explicitly for stable hardware OOB behavior.
            c_topk = fx.Index(topk)

            # X(A2): [tokens*topk, inter_dim] bytes = tokens*topk*k*elem_bytes
            x_nbytes_idx = (tokens_in * c_topk) * k_in * arith.index(int(elem_bytes))
            x_rsrc = buffer_ops.create_buffer_resource(
                arg_x,
                max_size=False,
                num_records_bytes=arith.index_cast(T.i64, x_nbytes_idx),
            )

            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)

            # OUT: [tokens, model_dim] -> clamp to descriptor max (i32 bytes) to avoid overflow on huge tokens.
            out_elem_bytes = 4 if out_is_f32 else 2
            out_nbytes_idx = tokens_in * n_in * fx.Index(out_elem_bytes)
            if not bool(accumulate):
                out_nbytes_idx = (
                    tokens_in * fx.Index(topk) * n_in * fx.Index(out_elem_bytes)
                )
            out_rsrc = buffer_ops.create_buffer_resource(
                arg_out,
                max_size=False,
                num_records_bytes=arith.index_cast(T.i64, out_nbytes_idx),
            )
            # fp16 path ignores scales completely (implicit scale=1.0).
            if is_f16:
                sx_rsrc = None
                sw_rsrc = None
            else:
                # scale_x (A2 scale): [nblk_k_w2, tokens*topk] f32 transposed -> total = nblk_k_w2 * tokens * topk
                sx_nbytes_idx = (
                    arith.index(nblk_k_w2) * (tokens_in * c_topk) * fx.Index(4)
                )
                sx_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_x,
                    max_size=False,
                    num_records_bytes=arith.index_cast(T.i64, sx_nbytes_idx),
                )
                # scale_w: [experts*model_dim] f32 (static shape in practice)
                sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)

            # sorted_token_ids / sorted_weights: [blocks*tile_m] (CK-style padded length)
            sorted_nbytes_idx = size_expert_ids_in * fx.Index(tile_m) * fx.Index(4)
            sorted_nbytes_i64 = arith.index_cast(T.i64, sorted_nbytes_idx)
            sorted_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_token_ids,
                max_size=False,
                num_records_bytes=sorted_nbytes_i64,
            )
            sorted_w_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_weights, max_size=False, num_records_bytes=sorted_nbytes_i64
            )

            # expert ids: [blocks] i32 -> bytes = size_expert_ids_in*4
            eid_nbytes_idx = size_expert_ids_in * fx.Index(4)
            expert_rsrc = buffer_ops.create_buffer_resource(
                arg_expert_ids,
                max_size=False,
                num_records_bytes=arith.index_cast(T.i64, eid_nbytes_idx),
            )
            bx_m = bx * fx.Index(tile_m)

            # Early-exit guard (as in 2ce65fb): some routing paths can produce extra/garbage
            # expert blocks beyond `num_valid_ids`. Skip those blocks entirely to avoid OOB.
            numids_rsrc = buffer_ops.create_buffer_resource(
                arg_num_valid_ids, max_size=False, num_records_bytes=fx.Index(4)
            )
            num_valid_i32 = buffer_ops.buffer_load(
                numids_rsrc, fx.Index(0), vec_width=1, dtype=T.i32
            )
            bx_m_i32 = arith.index_cast(T.i32, bx_m)
            blk_valid = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, num_valid_i32)

            def _moe_gemm2_then_body():
                # Expert id for this M tile.
                expert_i32 = buffer_ops.buffer_load(
                    expert_rsrc, bx, vec_width=1, dtype=T.i32
                )
                expert_idx = arith.index_cast(T.index, expert_i32)
                n_idx = fx.Index(model_dim)
                expert_off_idx = expert_idx * n_idx  # index

                # ---- X gmem->reg prefetch (match preshuffle GEMM mapping) ----
                # Prefer 16B buffer-load (dwordx4). If the per-thread byte count isn't divisible by
                # 16, fall back to 8B (dwordx2) or 4B (dword) loads. For fp16 we require 16B.
                if is_f16:
                    if bytes_per_thread_x % 16 != 0:
                        raise ValueError(
                            f"[fp16] bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 16"
                        )
                    x_load_bytes = 16
                else:
                    if bytes_per_thread_x % 16 == 0:
                        x_load_bytes = 16
                    elif bytes_per_thread_x % 8 == 0:
                        x_load_bytes = 8
                    elif bytes_per_thread_x % 4 == 0:
                        x_load_bytes = 4
                    else:
                        raise ValueError(
                            f"bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 4 to use the dword-indexed load mapping."
                        )
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4  # dwords per chunk (1/2/4)

                c_k_div4 = (k_in * arith.index(int(elem_bytes))) // fx.Index(4)
                c_k_div4_i32 = arith.index_cast(T.i32, c_k_div4)
                layout_x_div4 = fx.make_layout(
                    (m_i32_v, c_k_div4_i32), stride=(c_k_div4_i32, 1)
                )
                tile_k_dwords = (int(tile_k) * int(elem_bytes)) // 4
                layout_x_tile_div4 = fx.make_layout(
                    (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                )
                c_chunk_i32 = fx.Index(chunk_i32)
                tx_i32_base = tx * c_chunk_i32

                topk_i32 = fx.Int32(topk)
                mask24 = fx.Int32(0xFFFFFF)
                # Sentinel clamp uses `tokens` as the upper bound: t_valid = (t < tokens).
                tokens_i32 = arith.index_cast(T.i32, tokens_in)

                def x_tile_chunk_coord_i32(i: int):
                    return tile_chunk_coord_i32(
                        arith,
                        tx_i32_base=tx_i32_base,
                        i=i,
                        total_threads=total_threads,
                        layout_tile_div4=layout_x_tile_div4,
                        chunk_i32=chunk_i32,
                    )

                vec1_i32 = T.vec(1, T.i32)
                vec2_i32 = T.vec(2, T.i32)
                vec4_x = T.vec(4, x_elem)

                def load_x(idx_i32):
                    if x_load_bytes == 16:
                        idx_elem = (
                            idx_i32 if elem_bytes == 1 else (idx_i32 * fx.Index(2))
                        )
                        return buffer_copy_gmem16_dwordx4(
                            buffer_ops,
                            vector,
                            elem_type=x_elem,
                            idx_i32=idx_elem,
                            rsrc=x_rsrc,
                            vec_elems=vec16_elems,
                            elem_bytes=elem_bytes,
                        )
                    if x_load_bytes == 8:
                        return buffer_ops.buffer_load(
                            x_rsrc, idx_i32, vec_width=2, dtype=T.i32
                        )
                    return buffer_ops.buffer_load(
                        x_rsrc, idx_i32, vec_width=1, dtype=T.i32
                    )

                # decode routed token once (per thread's M-slice) and build a base offset.
                x_row_base_div4 = []
                x_col_local_i32 = []
                x_row_local = []
                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_row_local.append(row_local)
                    x_col_local_i32.append(col_local_i32)

                    sorted_row_i = bx_m + row_local
                    fused_i = buffer_ops.buffer_load(
                        sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32
                    )
                    t_i32 = fused_i & mask24
                    s_i32 = fused_i >> 24
                    # aiter moe_sorting uses sentinel token_id == tokens for padding.
                    # Do NOT rely on buffer OOB semantics for A2/scale loads; explicitly mask.
                    t_valid = arith.cmpi(arith.CmpIPredicate.ult, t_i32, tokens_i32)
                    s_valid = arith.cmpi(arith.CmpIPredicate.ult, s_i32, topk_i32)
                    ts_valid = t_valid & s_valid
                    t_safe = ts_valid.select(t_i32, fx.Int32(0))
                    s_safe = ts_valid.select(s_i32, fx.Int32(0))
                    row_ts_i32 = t_safe * topk_i32 + s_safe
                    row_ts_idx = arith.index_cast(T.index, row_ts_i32)
                    # Base row offset in dword units: row_ts_idx * (k_in/4)
                    x_row_base_div4.append(row_ts_idx * c_k_div4)

                def load_x_tile(base_k):
                    base_k_div4 = (base_k * arith.index(int(elem_bytes))) // fx.Index(4)
                    parts = []
                    for i in range_constexpr(num_x_loads):
                        idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                        x_vec = load_x(idx_i32)
                        if x_load_bytes == 16:
                            parts.append(vector.bitcast(T.i32x4, x_vec))
                        elif x_load_bytes == 8:
                            parts.append(x_vec)
                        else:
                            parts.append(x_vec)
                    return parts

                # tx -> wave/lane (GEMM-style decomposition).
                coord_wl = fx.idx2crd(tx, layout_tx_wave_lane)
                wave_id = fx.get(coord_wl, 0)
                lane_id = fx.get(coord_wl, 1)
                coord_l16 = fx.idx2crd(lane_id, layout_lane16)
                lane_div_16 = fx.get(coord_l16, 0)
                lane_mod_16 = fx.get(coord_l16, 1)

                row_a_lds = lane_mod_16
                a_kpack_elems = 16 // elem_bytes
                col_offset_base = lane_div_16 * arith.index(int(a_kpack_elems))
                col_offset_base_bytes = (
                    col_offset_base
                    if elem_bytes == 1
                    else (col_offset_base * arith.index(int(elem_bytes)))
                )

                # Dynamic N tiling within block.
                by_n = by * fx.Index(tile_n)
                num_waves = 4
                n_per_wave = tile_n // num_waves
                num_acc_n = n_per_wave // 16
                c_n_per_wave = fx.Index(n_per_wave)
                wave_mod_4 = wave_id % fx.Index(4)
                n_tile_base = wave_mod_4 * c_n_per_wave

                # Precompute (n_blk, n_intra) for B, and col indices for output.
                n_intra_list = []
                n_blk_list = []
                col_g_list = []
                c_n0 = c_n_total // fx.Index(16)
                c_n0_i32 = arith.index_cast(T.i32, c_n0)
                layout_n_blk_intra = fx.make_layout((c_n0_i32, 16), stride=(16, 1))
                for ni in range_constexpr(num_acc_n):
                    offset = arith.index(ni * 16)
                    col_g = by_n + n_tile_base + offset + lane_mod_16
                    col_g_list.append(col_g)

                    row_w = expert_off_idx + col_g
                    coord_w = fx.idx2crd(row_w, layout_n_blk_intra)
                    n_blk_list.append(fx.get(coord_w, 0))
                    n_intra_list.append(fx.get(coord_w, 1))

                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 64  # K64-byte micro-step (2x MFMA)

                # --- B Load Logic (K64) ---
                def load_b_pack(base_k, ki_step, ni):
                    return load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=ki_step,
                        n_blk=n_blk_list[ni],
                        n_intra=n_intra_list[ni],
                        lane_div_16=lane_div_16,  # 0..3
                        elem_type=w_elem,
                        kpack_bytes=kpack_bytes,
                        elem_bytes=elem_bytes,
                        unpack_int4=is_int4,
                    )

                def load_b_tile(base_k):
                    """Prefetch the entire per-thread B tile (gmem -> regs) for a given K base.

                    Returns a list of length `k_unroll`, where each entry is a tuple:
                      (packs_half0[ni], packs_half1[ni])  for the K64 micro-step.
                    """
                    b_tile = []
                    for ku in range_constexpr(k_unroll):
                        packs0 = []
                        packs1 = []
                        for ni in range_constexpr(num_acc_n):
                            ki0 = (ku * 2) + 0
                            ki1 = (ku * 2) + 1
                            b0 = load_b_pack(base_k, ki0, ni)
                            b1 = load_b_pack(base_k, ki1, ni)
                            packs0.append(b0)
                            packs1.append(b1)
                        b_tile.append((packs0, packs1))
                    return b_tile

                # ---- Pipeline helpers: store X tile to LDS with ping-pong base ----
                def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                    for i in range_constexpr(num_x_loads):
                        row_local = x_row_local[i]
                        col_local_i32 = x_col_local_i32[i]
                        if x_load_bytes == 16:
                            lds_store_16b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec16_ty=vec16_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x4=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )
                        elif x_load_bytes == 8:
                            lds_store_8b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec8_ty=vec8_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x2=vec_x_in_parts[i],
                            )
                        else:
                            lds_store_4b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec4_ty=vec4_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x1=vec_x_in_parts[i],
                            )

                # --- A LDS load helper for K64 (load 16B once, extract 2x i64 halves) ---
                def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_base):
                    col_base_swz_bytes = swizzle_xor16(
                        curr_row_a_lds, col_base_bytes, k_blocks16
                    )
                    col_base_swz = (
                        col_base_swz_bytes
                        if elem_bytes == 1
                        else (col_base_swz_bytes // arith.index(int(elem_bytes)))
                    )
                    idx_a16 = crd2idx((curr_row_a_lds, col_base_swz), layout_lds)
                    idx_a16 = idx_a16 + lds_base
                    loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                    a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                    a0 = vector.extract(
                        a_i64x2, static_position=[0], dynamic_position=[]
                    )
                    a1 = vector.extract(
                        a_i64x2, static_position=[1], dynamic_position=[]
                    )
                    return a0, a1

                # --- Blockscale pre-decode and helpers (stage2) ---
                c_scale_block_k_s2 = fx.Index(scale_block_k)
                c_128_s2 = fx.Index(128)
                c_nblk_k_w2 = fx.Index(nblk_k_w2)
                row_off_base_s2 = lane_div_16 * fx.Index(4)
                n_idx_s2 = fx.Index(model_dim)

                # Pre-decode sorted token IDs for stage2 (constant across all K-tiles).
                # OOB buffer loads return 0, so no validity masking needed for scale values.
                _pre_ts_safe_i32_s2 = []
                for _mi in range_constexpr(m_repeat):
                    _mi_safe = []
                    for _ii in range_constexpr(4):
                        _row_in_tile = (
                            arith.index(_mi * 16) + row_off_base_s2 + fx.Index(_ii)
                        )
                        _sorted_row = bx_m + _row_in_tile
                        _fused_pre = buffer_ops.buffer_load(
                            sorted_rsrc, _sorted_row, vec_width=1, dtype=T.i32
                        )
                        _t_id_pre = _fused_pre & mask24
                        _s_id_pre = _fused_pre >> 24
                        _t_valid_pre = arith.cmpi(
                            arith.CmpIPredicate.ult, _t_id_pre, tokens_i32
                        )
                        _s_valid_pre = arith.cmpi(
                            arith.CmpIPredicate.ult, _s_id_pre, topk_i32
                        )
                        _ts_valid_pre = _t_valid_pre & _s_valid_pre
                        _t_safe_pre = _ts_valid_pre.select(_t_id_pre, fx.Int32(0))
                        _s_safe_pre = _ts_valid_pre.select(_s_id_pre, fx.Int32(0))
                        _ts_i32_pre = _t_safe_pre * topk_i32 + _s_safe_pre
                        _mi_safe.append(_ts_i32_pre)
                    _pre_ts_safe_i32_s2.append(_mi_safe)

                # Pre-compute N-block indices for scale_w (constant per CTA)
                _pre_n_block_s2 = []
                for _ni in range_constexpr(num_acc_n):
                    _col_base_ni_pre = by_n + n_tile_base + arith.index(_ni * 16)
                    _pre_n_block_s2.append(
                        (expert_off_idx + _col_base_ni_pre) // c_128_s2
                    )

                m_in_s2 = tokens_in * fx.Index(topk)

                def load_scales_s2(k_base):
                    all_combined = []
                    for sb in range_constexpr(sb_per_tile_s2):
                        kb = k_base // c_scale_block_k_s2 + fx.Index(sb)
                        sa_base_offset = kb * m_in_s2

                        s_a_vecs = []
                        sa_base_i32 = arith.index_cast(T.i32, sa_base_offset)
                        for mi in range_constexpr(m_repeat):
                            s_a_row = []
                            for ii in range_constexpr(4):
                                ts_safe_i32 = _pre_ts_safe_i32_s2[mi][ii]
                                sa_idx_i32 = sa_base_i32 + ts_safe_i32
                                sa_idx = arith.index_cast(T.index, sa_idx_i32)
                                s_a_val = buffer_ops.buffer_load(
                                    sx_rsrc, sa_idx, vec_width=1, dtype=T.f32
                                )
                                s_a_row.append(s_a_val)
                            s_a_vecs.append(s_a_row)

                        _sw_shared_n_s2 = n_per_wave <= 128
                        s_w_vals = []
                        for ni in range_constexpr(num_acc_n):
                            if ni == 0 or not _sw_shared_n_s2:
                                sw_idx = _pre_n_block_s2[ni] * c_nblk_k_w2 + kb
                                s_w = buffer_ops.buffer_load(
                                    sw_rsrc, sw_idx, vec_width=1, dtype=T.f32
                                )
                            s_w_vals.append(s_w)

                        s_a_vec4_list = []
                        for mi in range_constexpr(m_repeat):
                            s_a_vec4_list.append(
                                vector.from_elements(T.f32x4, s_a_vecs[mi])
                            )
                        all_combined.append((s_a_vec4_list, s_w_vals))
                    return all_combined

                def compute_tile_bs_s2(
                    acc_in, b_tile_in, lds_base, pre_scales, *, a0_prefetch=None
                ):
                    current_acc = list(acc_in)
                    mfma_res_ty = T.f32x4

                    if _is_gfx950:

                        def _pack128(x0, x1, x2, x3):
                            v4 = vector.from_elements(T.vec(4, T.i64), [x0, x1, x2, x3])
                            return vector.bitcast(T.vec(8, T.i32), v4)

                        for sb in range_constexpr(sb_per_tile_s2):
                            s_a_vec4_list, s_w_vals = pre_scales[sb]
                            ku0 = sb * ku_per_sb_s2
                            ku1 = ku0 + 1
                            b0_p0, b0_p1 = b_tile_in[ku0]
                            b1_p0, b1_p1 = b_tile_in[ku1]
                            col0 = col_offset_base_bytes + arith.index(ku0 * 64)
                            col1 = col_offset_base_bytes + arith.index(ku1 * 64)
                            for mi in range_constexpr(m_repeat):
                                curr_row = row_a_lds + arith.index(mi * 16)
                                if a0_prefetch is not None and sb == 0 and mi == 0:
                                    a0, a1 = a0_prefetch
                                else:
                                    a0, a1 = lds_load_packs_k64(
                                        curr_row, col0, lds_base
                                    )
                                a2, a3 = lds_load_packs_k64(curr_row, col1, lds_base)
                                a128 = _pack128(a0, a1, a2, a3)
                                s_a_v4 = s_a_vec4_list[mi]
                                pending_acc = None
                                for ni in range_constexpr(num_acc_n):
                                    acc_idx = mi * num_acc_n + ni
                                    b128 = _pack128(
                                        b0_p0[ni], b0_p1[ni], b1_p0[ni], b1_p1[ni]
                                    )
                                    blk = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                        mfma_res_ty,
                                        [
                                            a128,
                                            b128,
                                            acc_init,
                                            0,
                                            0,
                                            0,
                                            0x7F7F7F7F,
                                            0,
                                            0x7F7F7F7F,
                                        ],
                                    )
                                    rocdl.sched_barrier(0)
                                    if pending_acc is not None:
                                        prev_acc_idx, prev_blk, prev_ni = pending_acc
                                        s_w_bc = vector.broadcast(
                                            T.f32x4, s_w_vals[prev_ni]
                                        )
                                        scale = ArithValue(s_a_v4) * ArithValue(s_w_bc)
                                        current_acc[prev_acc_idx] = math_dialect.fma(
                                            prev_blk, scale, current_acc[prev_acc_idx]
                                        )
                                    pending_acc = (acc_idx, blk, ni)
                                if pending_acc is not None:
                                    prev_acc_idx, prev_blk, prev_ni = pending_acc
                                    s_w_bc = vector.broadcast(
                                        T.f32x4, s_w_vals[prev_ni]
                                    )
                                    scale = ArithValue(s_a_v4) * ArithValue(s_w_bc)
                                    current_acc[prev_acc_idx] = math_dialect.fma(
                                        prev_blk, scale, current_acc[prev_acc_idx]
                                    )
                    else:
                        mfma_fn = (
                            mfma_i32_k32
                            if is_int8
                            else (
                                rocdl.mfma_f32_16x16x16f16
                                if is_f16
                                else rocdl.mfma_f32_16x16x32_fp8_fp8
                            )
                        )

                        def _i64_to_v4f16(x_i64):
                            v1 = vector.from_elements(T.vec(1, T.i64), [x_i64])
                            return vector.bitcast(T.f16x4, v1)

                        def mfma_k64(acc0, a0, a1, b0, b1):
                            if is_f16:
                                a0v = _i64_to_v4f16(a0)
                                a1v = _i64_to_v4f16(a1)
                                b0v = _i64_to_v4f16(b0)
                                b1v = _i64_to_v4f16(b1)
                                acc1 = mfma_fn(mfma_res_ty, [a0v, b0v, acc0, 0, 0, 0])
                                return mfma_fn(mfma_res_ty, [a1v, b1v, acc1, 0, 0, 0])
                            acc1 = mfma_fn(mfma_res_ty, [a0, b0, acc0, 0, 0, 0])
                            return mfma_fn(mfma_res_ty, [a1, b1, acc1, 0, 0, 0])

                        for sb in range_constexpr(sb_per_tile_s2):
                            s_a_vec4_list, s_w_vals = pre_scales[sb]
                            for mi in range_constexpr(m_repeat):
                                for ni in range_constexpr(num_acc_n):
                                    acc_idx = mi * num_acc_n + ni
                                    blk = acc_init
                                    for ku_local in range_constexpr(ku_per_sb_s2):
                                        ku = sb * ku_per_sb_s2 + ku_local
                                        b_packs0, b_packs1 = b_tile_in[ku]
                                        ki64 = arith.index(ku * 64)
                                        col_base = col_offset_base_bytes + ki64
                                        if (
                                            (a0_prefetch is not None)
                                            and (sb == 0)
                                            and (ku_local == 0)
                                            and (mi == 0)
                                        ):
                                            a0, a1 = a0_prefetch
                                        else:
                                            a0, a1 = lds_load_packs_k64(
                                                row_a_lds + arith.index(mi * 16),
                                                col_base,
                                                lds_base,
                                            )
                                        blk = mfma_k64(
                                            blk, a0, a1, b_packs0[ni], b_packs1[ni]
                                        )
                                    s_w_bc = vector.broadcast(T.f32x4, s_w_vals[ni])
                                    scale = ArithValue(s_a_vec4_list[mi]) * ArithValue(
                                        s_w_bc
                                    )
                                    current_acc[acc_idx] = math_dialect.fma(
                                        blk, scale, current_acc[acc_idx]
                                    )
                    return current_acc

                def compute_tile(
                    acc_in,
                    b_tile_in,
                    lds_base,
                    *,
                    prefetch_epilogue: bool = False,
                    a0_prefetch=None,
                ):
                    acc_list = list(acc_in)
                    mfma_res_ty = T.i32x4 if is_int8 else T.f32x4
                    mfma_fn = (
                        mfma_i32_k32
                        if is_int8
                        else (
                            rocdl.mfma_f32_16x16x16f16
                            if is_f16
                            else rocdl.mfma_f32_16x16x32_fp8_fp8
                        )
                    )

                    epilogue_pf = None
                    if prefetch_epilogue:
                        expert_off_pf = expert_off_idx
                        sw_pf = []
                        for ni in range_constexpr(num_acc_n):
                            col_g = col_g_list[ni]
                            row_w_idx = expert_off_pf + col_g
                            sw_pf.append(
                                fx.Float32(1.0)
                                if is_f16
                                else buffer_ops.buffer_load(
                                    sw_rsrc, row_w_idx, vec_width=1, dtype=T.f32
                                )
                            )
                        # Also prefetch per-row routed/topk weights (sorted_weights) when enabled.
                        tw_pf = None
                        if doweight_stage2:
                            tw_pf = []
                            lane_div_16_mul4_pf = lane_div_16 * fx.Index(4)
                            ii_idx_list_pf = [fx.Index(ii) for ii in range(4)]
                            for mi in range_constexpr(m_repeat):
                                mi_base_pf = arith.index(mi * 16)
                                for ii in range_constexpr(4):
                                    row_off_pf = (
                                        lane_div_16_mul4_pf + ii_idx_list_pf[ii]
                                    )
                                    row_in_tile_pf = mi_base_pf + row_off_pf
                                    sorted_row_pf = bx_m + row_in_tile_pf
                                    tw_pf.append(
                                        buffer_ops.buffer_load(
                                            sorted_w_rsrc,
                                            sorted_row_pf,
                                            vec_width=1,
                                            dtype=T.f32,
                                        )
                                    )
                        epilogue_pf = (sw_pf, tw_pf)

                    def _i64_to_v4f16(x_i64):
                        v1 = vector.from_elements(T.vec(1, T.i64), [x_i64])
                        return vector.bitcast(T.f16x4, v1)

                    def mfma_k64(acc0, a0, a1, b0, b1):
                        if is_f16:
                            a0v = _i64_to_v4f16(a0)
                            a1v = _i64_to_v4f16(a1)
                            b0v = _i64_to_v4f16(b0)
                            b1v = _i64_to_v4f16(b1)
                            acc1 = mfma_fn(mfma_res_ty, [a0v, b0v, acc0, 0, 0, 0])
                            return mfma_fn(mfma_res_ty, [a1v, b1v, acc1, 0, 0, 0])
                        acc1 = mfma_fn(mfma_res_ty, [a0, b0, acc0, 0, 0, 0])
                        return mfma_fn(mfma_res_ty, [a1, b1, acc1, 0, 0, 0])

                    for ku in range_constexpr(k_unroll):
                        b_packs0, b_packs1 = b_tile_in[ku]
                        ki64 = arith.index(ku * 64)
                        col_base = col_offset_base_bytes + ki64

                        for mi in range_constexpr(m_repeat):
                            mi_val = arith.index(mi * 16)
                            curr_row_a_lds = row_a_lds + mi_val

                            if (a0_prefetch is not None) and (ku == 0) and (mi == 0):
                                a0, a1 = a0_prefetch
                            else:
                                a0, a1 = lds_load_packs_k64(
                                    curr_row_a_lds, col_base, lds_base
                                )

                            for ni in range_constexpr(num_acc_n):
                                acc_idx = mi * num_acc_n + ni
                                acc_list[acc_idx] = mfma_k64(
                                    acc_list[acc_idx],
                                    a0,
                                    a1,
                                    b_packs0[ni],
                                    b_packs1[ni],
                                )
                    return acc_list, epilogue_pf

                # ---------------- 2-stage pipeline (ping-pong LDS + B tile prefetch) ----------------
                lds_tile_elems = arith.index(tile_m * lds_stride)
                lds_base_cur = fx.Index(0)
                lds_base_nxt = lds_tile_elems

                rocdl.sched_barrier(0)

                # def hot_loop_scheduler():
                #     mfma_group = num_acc_n
                #     # K64 micro-step: 2x K32 MFMA per accumulator update.
                #     mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                #     mfma_per_iter = 2 * mfma_group
                #     sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                #     rocdl.sched_dsrd(2)
                #     rocdl.sched_mfma(1)
                #     rocdl.sched_mfma(1)
                #     if num_acc_n < 4:
                #         rocdl.sched_dsrd(1)
                #         rocdl.sched_mfma(1)
                #         rocdl.sched_dsrd(1)
                #         rocdl.sched_mfma(1)
                #         rocdl.sched_vmem(1)
                #         rocdl.sched_mfma(1)
                #         rocdl.sched_vmem(1)
                #         rocdl.sched_mfma(2)
                #         rocdl.sched_dsrd(1)
                #         rocdl.sched_mfma(2)
                #         rocdl.sched_vmem(1)

                #     dswr_tail = num_x_loads
                #     if dswr_tail > sche_iters:
                #         dswr_tail = sche_iters
                #     dswr_start = sche_iters - dswr_tail
                #     for sche_i in range_constexpr(sche_iters):
                #         rocdl.sched_mfma(mfma_group // 2)
                #         rocdl.sched_dsrd(1)
                #         rocdl.sched_mfma(mfma_group // 2)
                #         rocdl.sched_vmem(1)
                #         rocdl.sched_mfma(mfma_group)
                #         if sche_i >= dswr_start - 1:
                #             rocdl.sched_dswr(1)
                #     rocdl.sched_barrier(0)

                def hot_loop_scheduler():
                    mfma_per_ku = m_repeat * num_acc_n * 2  # m * n_acc * 2(k32)
                    total_mfma = k_unroll * mfma_per_ku
                    rocdl.sched_group_barrier(
                        rocdl.mask_dsrd, ku_per_sb_s2 * m_repeat, 0
                    )
                    rocdl.sched_group_barrier(rocdl.mask_mfma, total_mfma, 1)
                    rocdl.sched_group_barrier(rocdl.mask_vmem_rd, num_x_loads, 2)
                    rocdl.sched_group_barrier(rocdl.mask_dswr, num_x_loads, 3)
                    rocdl.sched_barrier(0)

                # Prologue.
                k0 = fx.Index(0)
                x_regs0 = load_x_tile(k0)
                b_cur = load_b_tile(k0)
                store_x_tile_to_lds(x_regs0, lds_base_cur)
                gpu.barrier()

                acc = [arith.constant_vector(0.0, T.f32x4)] * (num_acc_n * m_repeat)
                lds_base_pong = lds_base_cur
                lds_base_ping = lds_base_nxt

                # Cross-tile A0 LDS prefetch (default-on): prefetch the first A-pack (K64) for the
                # tile we are about to compute from LDS, to overlap with upcoming VMEM.
                a0_prefetch_pong = lds_load_packs_k64(
                    row_a_lds, col_offset_base_bytes, lds_base_pong
                )

                # Main loop: process K tiles in 2-tile ping-pong steps.
                #
                # IMPORTANT: for odd number of K tiles, leave **1** tail tile; for even, leave **2**.
                # Otherwise the 2-tile tail below would double-count the last tile when num_tiles is odd
                # (e.g. inter_dim=192, tile_k=64 -> 3 tiles).
                num_k_tiles_py = int(inter_dim) // int(tile_k)
                odd_k_tiles = (num_k_tiles_py % 2) == 1
                tail_tiles = 1 if odd_k_tiles else 2
                k_main2_py = (num_k_tiles_py - tail_tiles) * int(tile_k)
                if k_main2_py < 0:
                    k_main2_py = 0

                c2_tile_k = arith.index(tile_k * 2)
                pair_iters = k_main2_py // (int(tile_k) * 2)
                for pair_i in range_constexpr(pair_iters):
                    k_iv = arith.index(pair_i * (tile_k * 2))
                    # Issue scale loads FIRST so their latency hides behind heavy tile VMEM.
                    pre_scales_pong = load_scales_s2(k_iv)
                    next_k1 = k_iv + tile_k
                    x_regs_ping = load_x_tile(next_k1)
                    b_ping = load_b_tile(next_k1)

                    acc = compute_tile_bs_s2(
                        acc,
                        b_cur,
                        lds_base_pong,
                        pre_scales_pong,
                        a0_prefetch=a0_prefetch_pong,
                    )
                    a0_prefetch_pong = None
                    store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    hot_loop_scheduler()
                    gpu.barrier()

                    # Cross-tile prefetch for the ping tile we are about to compute.
                    a0_prefetch_ping = lds_load_packs_k64(
                        row_a_lds, col_offset_base_bytes, lds_base_ping
                    )

                    # Issue scale loads FIRST so their latency hides behind heavy tile VMEM.
                    pre_scales_ping = load_scales_s2(next_k1)
                    next_k2 = k_iv + c2_tile_k
                    x_regs_pong = load_x_tile(next_k2)
                    b_next = load_b_tile(next_k2)

                    acc = compute_tile_bs_s2(
                        acc,
                        b_ping,
                        lds_base_ping,
                        pre_scales_ping,
                        a0_prefetch=a0_prefetch_ping,
                    )
                    a0_prefetch_ping = None
                    store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                    hot_loop_scheduler()
                    gpu.barrier()

                    # Cross-tile prefetch for the next pong tile.
                    a0_prefetch_pong = lds_load_packs_k64(
                        row_a_lds, col_offset_base_bytes, lds_base_pong
                    )

                    b_cur = b_next

                if odd_k_tiles:
                    # Tail: single remaining tile (already in `b_cur` / `lds_base_pong`).
                    k_last = arith.index((num_k_tiles_py - 1) * int(tile_k))
                    pre_scales_last = load_scales_s2(k_last)
                    acc = compute_tile_bs_s2(
                        acc,
                        b_cur,
                        lds_base_pong,
                        pre_scales_last,
                        a0_prefetch=a0_prefetch_pong,
                    )
                else:
                    # Tail: 2 remaining tiles.
                    k_tail0 = k_in - c2_tile_k
                    k_tail1 = k_in - tile_k
                    # Issue scale loads FIRST so their latency hides behind heavy tile VMEM.
                    pre_scales_tail0 = load_scales_s2(k_tail0)
                    x_regs_ping = load_x_tile(k_tail1)
                    b_ping = load_b_tile(k_tail1)

                    acc = compute_tile_bs_s2(
                        acc,
                        b_cur,
                        lds_base_pong,
                        pre_scales_tail0,
                        a0_prefetch=a0_prefetch_pong,
                    )
                    a0_prefetch_pong = None
                    store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    hot_loop_scheduler()
                    gpu.barrier()

                    # Epilogue tile (blockscale already dequantized).
                    a0_prefetch_ping = lds_load_packs_k64(
                        row_a_lds, col_offset_base_bytes, lds_base_ping
                    )
                    pre_scales_tail1 = load_scales_s2(k_tail1)
                    acc = compute_tile_bs_s2(
                        acc,
                        b_ping,
                        lds_base_ping,
                        pre_scales_tail1,
                        a0_prefetch=a0_prefetch_ping,
                    )

                # ---------------- Epilogue: LDS CShuffle + atomic half2 (x2) ----------------
                # Reuse the shared helper so GEMM / MoE kernels share the exact same CShuffle skeleton.
                expert_off = expert_off_idx
                mask24_i32 = fx.Int32(0xFFFFFF)
                model_i32 = fx.Int32(model_dim)
                topk_i32_v = topk_i32

                zero_i32 = fx.Int32(0)
                c2_i32 = fx.Int32(2)  # 2B element size for f16/bf16
                mask_even_i32 = fx.Int32(
                    0xFFFFFFFE
                )  # align element index to even for half2 atomics

                e_vec = _e_vec

                def atomic_add_f16x2(val_f16x2, byte_off_i32):
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        val_f16x2,
                        out_rsrc,
                        byte_off_i32,
                        zero_i32,
                        zero_i32,
                    )

                # Blockscale: dequant already done in compute_tile_bs_s2, no sw/sx needed here.

                if out_is_f32:
                    # origin/dev_a16w4: f32 output uses scalar f32 atomics and skips CShuffle/LDS.
                    c4_i32 = fx.Int32(4)

                    def atomic_add_f32(val_f32, byte_off_i32):
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            val_f32,
                            out_rsrc,
                            byte_off_i32,
                            zero_i32,
                            zero_i32,
                        )

                    def _stage2_row_atomic(*, mi: int, ii: int, row_in_tile, row):
                        # Blockscale: dequant already done in compute_tile_bs_s2.
                        fused2 = buffer_ops.buffer_load(
                            sorted_rsrc, row, vec_width=1, dtype=T.i32
                        )
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24

                        if doweight_stage2:
                            tw = buffer_ops.buffer_load(
                                sorted_w_rsrc, row, vec_width=1, dtype=T.f32
                            )

                        idx0 = t2 * model_i32  # i32 element index base

                        for ni in range_constexpr(num_acc_n):
                            col_g = col_g_list[ni]
                            acc_idx = mi * num_acc_n + ni
                            v = vector.extract(
                                acc[acc_idx], static_position=[ii], dynamic_position=[]
                            )
                            if doweight_stage2:
                                v = v * tw
                            col_i32 = arith.index_cast(T.i32, col_g)
                            idx_elem = idx0 + col_i32
                            byte_off = idx_elem * c4_i32
                            atomic_add_f32(v, byte_off)

                    default_epilog(
                        arith=arith,
                        range_constexpr=range_constexpr,
                        m_repeat=m_repeat,
                        lane_div_16=lane_div_16,
                        bx_m=bx_m,
                        body_row=_stage2_row_atomic,
                    )
                else:
                    if lds_out is None:
                        raise RuntimeError(
                            "FLYDSL_MOE_STAGE2_CSHUFFLE=1 but lds_out is not allocated/aliased."
                        )

                    # For bf16 global atomics (gfx942 only), precompute the output base address.
                    # gfx950+ has buffer_atomic_pk_add_bf16, so bf16 uses buffer atomics there.
                    out_base_idx = None
                    if _needs_global_atomic_bf16:
                        out_base_idx = buffer_ops.extract_base_index(arg_out)

                    def write_row_to_lds(
                        *,
                        mi: int,
                        ii: int,
                        row_in_tile,
                        row,
                        row_base_lds,
                        col_base_local,
                        num_acc_n: int,
                        lds_out,
                    ):
                        # Blockscale: dequant already done in compute_tile_bs_s2.
                        if doweight_stage2:
                            tw = buffer_ops.buffer_load(
                                sorted_w_rsrc, row, vec_width=1, dtype=T.f32
                            )

                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            acc_idx = mi * num_acc_n + ni
                            v = vector.extract(
                                acc[acc_idx], static_position=[ii], dynamic_position=[]
                            )
                            if doweight_stage2:
                                v = v * tw
                            v_out = arith.trunc_f(out_elem(), v)

                            lds_idx = row_base_lds + col_local
                            vec1_out = T.vec(1, out_elem())
                            v1 = vector.from_elements(vec1_out, [v_out])
                            vector.store(v1, lds_out, [lds_idx], alignment=2)

                    def precompute_row(*, row_local, row):
                        # Precompute row context for cshuffle stores.
                        # Return (fused_i32, row_valid_i1) so the epilogue can skip the entire row
                        # for invalid tail rows (CK-style), avoiding per-store branching.
                        fused2 = buffer_ops.buffer_load(
                            sorted_rsrc, row, vec_width=1, dtype=T.i32
                        )
                        row_i32 = arith.index_cast(T.i32, row)
                        row_valid0 = arith.cmpi(
                            arith.CmpIPredicate.ult, row_i32, num_valid_i32
                        )
                        t = fused2 & mask24_i32
                        s = fused2 >> 24
                        t_ok = arith.cmpi(arith.CmpIPredicate.ult, t, tokens_i32)
                        s_ok = arith.cmpi(arith.CmpIPredicate.ult, s, topk_i32_v)
                        row_valid = row_valid0 & t_ok & s_ok
                        return (fused2, row_valid)

                    def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                        fused = row_ctx
                        t = fused & mask24_i32
                        s = fused >> 24
                        idx0 = t * model_i32
                        if not bool(accumulate):
                            ts = t * topk_i32_v + s
                            idx0 = ts * model_i32
                        col_i32 = arith.index_cast(T.i32, col_g0)
                        idx_elem = idx0 + col_i32
                        idx_elem_even = idx_elem & mask_even_i32
                        if _needs_global_atomic_bf16:
                            # gfx942: no buffer_atomic_pk_add_bf16, use global atomicrmw fadd
                            if bool(accumulate):
                                byte_off = idx_elem_even * c2_i32
                                byte_off_idx = arith.index_cast(T.index, byte_off)
                                ptr_addr_idx = out_base_idx + byte_off_idx
                                out_ptr = buffer_ops.create_llvm_ptr(
                                    ptr_addr_idx, address_space=1
                                )
                                out_ptr_v = (
                                    out_ptr._value
                                    if hasattr(out_ptr, "_value")
                                    else out_ptr
                                )
                                frag_v = (
                                    frag._value if hasattr(frag, "_value") else frag
                                )
                                llvm.AtomicRMWOp(
                                    llvm.AtomicBinOp.fadd,
                                    out_ptr_v,
                                    frag_v,
                                    llvm.AtomicOrdering.monotonic,
                                    syncscope="agent",
                                    alignment=4,
                                )
                            else:
                                buffer_ops.buffer_store(frag, out_rsrc, idx_elem_even)
                        else:
                            # f16, or bf16 on gfx950+ (has buffer_atomic_pk_add_bf16)
                            byte_off = idx_elem_even * c2_i32
                            if bool(accumulate):
                                atomic_add_f16x2(frag, byte_off)
                            else:
                                buffer_ops.buffer_store(frag, out_rsrc, idx_elem_even)

                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=e_vec,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=(T.bf16 if out_is_bf16 else T.f16),
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                    )

            _if_blk = scf.IfOp(blk_valid)
            with _if_then(_if_blk):
                _moe_gemm2_then_body()

    # ── Host launcher (flyc.jit + .launch) ────────────────────────────────
    @flyc.jit
    def launch_moe_blockscale_gemm2(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        n_in = arith.index_cast(T.index, i32_n_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        gx = n_in // fx.Index(tile_n)
        gy = size_expert_ids_in

        launcher2 = moe_blockscale_gemm2(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_num_valid_ids,
            i32_tokens_in,
            i32_n_in,
            i32_k_in,
            i32_size_expert_ids_in,
        )
        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            T.i32, _wpe
                        )
        launcher2.launch(grid=(gx, gy, 1), block=(256, 1, 1), stream=stream)

    return launch_moe_blockscale_gemm2


# MoE Reduction Kernel (reduce sum over topk dimension)
@functools.lru_cache(maxsize=1024)
def compile_moe_reduction(
    *,
    topk: int,
    model_dim: int,
    dtype_str: str = "f16",
    use_mask: bool = False,
):
    """Compile a reduction kernel that sums over the topk dimension.

    Input:  X [tokens, topk, model_dim]
            valid_mask [tokens, topk] (optional, if use_mask=True)
    Output: Y [tokens, model_dim]

    This kernel performs: Y[t, d] = sum(X[t, :, d]) for all t, d.
    When use_mask=True, only sums slots where valid_mask[t,k]=1.
    Used in conjunction with compile_moe_blockscale_gemm2(accumulate=False) to avoid atomic contention.
    """
    gpu_arch = get_hip_arch()
    DYN = ir.ShapedType.get_dynamic_size()

    # Kernel Config
    BLOCK_SIZE = 256
    VEC_WIDTH = 8
    USE_NONTEMPORAL = True
    VEC_ALIGN = 16

    masked = "masked" if use_mask else ""

    module_name = f"bs_moe_reduce_topk{topk}_{dtype_str}{masked}"

    if dtype_str == "f32":
        elem_type_tag = "f32"
    elif dtype_str == "f16":
        elem_type_tag = "f16"
    elif dtype_str == "bf16":
        elem_type_tag = "bf16"
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    compute_type = lambda: T.f32
    i32_type = lambda: T.i32
    i8_type = lambda: T.i8

    def elem_type():
        ty = (
            T.f32
            if elem_type_tag == "f32"
            else (T.f16 if elem_type_tag == "f16" else T.bf16)
        )
        return ty() if callable(ty) else ty

    if True:

        @flyc.kernel(name=module_name)
        def moe_reduction_kernel(
            X: fx.Tensor,
            Y: fx.Tensor,
            valid_mask: fx.Tensor,
            i32_m_tokens: fx.Int32,
        ):
            m_tokens = arith.index_cast(T.index, i32_m_tokens)
            c_topk = fx.Index(topk)
            c_model_dim = fx.Index(model_dim)
            c_elem_bytes = arith.index(4 if dtype_str == "f32" else 2)
            x_nbytes_idx = m_tokens * c_topk * c_model_dim * c_elem_bytes
            y_nbytes_idx = m_tokens * c_model_dim * c_elem_bytes
            mask_nbytes_idx = m_tokens * c_topk
            x_rsrc = buffer_ops.create_buffer_resource(
                X, max_size=False, num_records_bytes=x_nbytes_idx
            )
            y_rsrc = buffer_ops.create_buffer_resource(
                Y, max_size=False, num_records_bytes=y_nbytes_idx
            )
            mask_rsrc = buffer_ops.create_buffer_resource(
                valid_mask, max_size=False, num_records_bytes=mask_nbytes_idx
            )

            token_idx = gpu.block_id("x")
            tile_idx = gpu.block_id("y")
            tid = gpu.thread_id("x")

            token_i32 = arith.index_cast(i32_type(), token_idx)
            m_tokens_i32 = arith.index_cast(i32_type(), m_tokens)
            tok_ok = arith.cmpi(arith.CmpIPredicate.ult, token_i32, m_tokens_i32)
            _if_tok = scf.IfOp(tok_ok)
            with _if_then(_if_tok):
                tile_cols = BLOCK_SIZE * VEC_WIDTH
                c_tile_cols = fx.Index(tile_cols)
                c_vecw = fx.Index(VEC_WIDTH)

                col_base = tile_idx * c_tile_cols + tid * c_vecw

                col_ok = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    arith.index_cast(i32_type(), col_base),
                    arith.index_cast(i32_type(), c_model_dim),
                )
                _if_col = scf.IfOp(col_ok)
                with _if_then(_if_col):
                    end_ok = arith.cmpi(
                        arith.CmpIPredicate.ule,
                        arith.index_cast(i32_type(), col_base + c_vecw),
                        arith.index_cast(i32_type(), c_model_dim),
                    )
                    _if_full = scf.IfOp(end_ok, has_else=True)
                    with _if_then(_if_full):
                        c0_i8 = arith.constant(0, type=i8_type())
                        token_base = token_idx * c_topk
                        for lane in range_constexpr(VEC_WIDTH):
                            col = col_base + fx.Index(lane)
                            a = arith.constant(0.0, type=compute_type())
                            for k in range_constexpr(topk):
                                k_idx = fx.Index(k)
                                x_idx = (token_base + k_idx) * c_model_dim + col
                                x_idx_i32 = arith.index_cast(i32_type(), x_idx)
                                if use_mask:
                                    m_idx = token_base + k_idx
                                    m_idx_i32 = arith.index_cast(i32_type(), m_idx)
                                    mv = buffer_ops.buffer_load(
                                        mask_rsrc,
                                        m_idx_i32,
                                        vec_width=1,
                                        dtype=i8_type(),
                                    )
                                    mv_ok = arith.cmpi(
                                        arith.CmpIPredicate.ne, mv, c0_i8
                                    )
                                    v = arith.select(
                                        mv_ok,
                                        buffer_ops.buffer_load(
                                            x_rsrc,
                                            x_idx_i32,
                                            vec_width=1,
                                            dtype=elem_type(),
                                        ),
                                        arith.constant(0.0, type=elem_type()),
                                    )
                                else:
                                    v = buffer_ops.buffer_load(
                                        x_rsrc,
                                        x_idx_i32,
                                        vec_width=1,
                                        dtype=elem_type(),
                                    )
                                if dtype_str in ("f16", "bf16"):
                                    v = arith.extf(compute_type(), v)
                                a = a + v
                            v = a
                            if dtype_str in ("f16", "bf16"):
                                v = arith.trunc_f(elem_type(), v)
                            y_idx = token_idx * c_model_dim + col
                            y_idx_i32 = arith.index_cast(i32_type(), y_idx)
                            buffer_ops.buffer_store(v, y_rsrc, y_idx_i32)

                    with _if_else(_if_full):
                        for lane in range_constexpr(VEC_WIDTH):
                            col = col_base + fx.Index(lane)
                            lane_ok = arith.cmpi(
                                arith.CmpIPredicate.ult,
                                arith.index_cast(i32_type(), col),
                                arith.index_cast(i32_type(), c_model_dim),
                            )
                            _if_lane = scf.IfOp(lane_ok)
                            with _if_then(_if_lane):
                                a = arith.constant(0.0, type=compute_type())
                                token_base = token_idx * c_topk
                                c0_i8 = arith.constant(0, type=i8_type())
                                for k in range_constexpr(topk):
                                    k_idx = fx.Index(k)
                                    x_idx = (token_base + k_idx) * c_model_dim + col
                                    x_idx_i32 = arith.index_cast(i32_type(), x_idx)
                                    if use_mask:
                                        m_idx = token_base + k_idx
                                        m_idx_i32 = arith.index_cast(i32_type(), m_idx)
                                        mv = buffer_ops.buffer_load(
                                            mask_rsrc,
                                            m_idx_i32,
                                            vec_width=1,
                                            dtype=i8_type(),
                                        )
                                        mv_ok = arith.cmpi(
                                            arith.CmpIPredicate.ne, mv, c0_i8
                                        )
                                        v = arith.select(
                                            mv_ok,
                                            buffer_ops.buffer_load(
                                                x_rsrc,
                                                x_idx_i32,
                                                vec_width=1,
                                                dtype=elem_type(),
                                            ),
                                            arith.constant(0.0, type=elem_type()),
                                        )
                                    else:
                                        v = buffer_ops.buffer_load(
                                            x_rsrc,
                                            x_idx_i32,
                                            vec_width=1,
                                            dtype=elem_type(),
                                        )
                                    if dtype_str in ("f16", "bf16"):
                                        v = arith.extf(compute_type(), v)
                                    a = a + v

                                out = a
                                if dtype_str in ("f16", "bf16"):
                                    out = arith.trunc_f(elem_type(), out)
                                y_idx = token_idx * c_model_dim + col
                                y_idx_i32 = arith.index_cast(i32_type(), y_idx)
                                buffer_ops.buffer_store(out, y_rsrc, y_idx_i32)

    # ── Host launcher (flyc.jit + .launch) ────────────────────────────────
    tile_size = BLOCK_SIZE * VEC_WIDTH
    gy_static = (model_dim + tile_size - 1) // tile_size

    @flyc.jit
    def launch_moe_reduction(
        X: fx.Tensor,
        Y: fx.Tensor,
        valid_mask: fx.Tensor,
        i32_m_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        gx = arith.index_cast(T.index, i32_m_tokens)
        moe_reduction_kernel(X, Y, valid_mask, i32_m_tokens).launch(
            grid=(gx, gy_static, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_moe_reduction


# MoE GEMM2 Execution Modes
class MoeGemm2Mode:
    """Execution mode for MoE GEMM2."""

    ATOMIC = "atomic"  # Use atomic accumulation (default)
    REDUCE = "reduce"  # Use non-atomic write + reduce kernel


class _MoeGemm2ReduceWrapper:
    """Wrapper combining GEMM2 (no atomics) with reduction kernel.

    This wrapper handles the intermediate buffer allocation and orchestrates
    the two-phase computation:
    1. GEMM2 outputs to [tokens*topk, model_dim] without atomics
    2. Reduce sums over topk to produce [tokens, model_dim]
    """

    def __init__(
        self,
        gemm2_exe,
        reduce_exe,
        topk: int,
        model_dim: int,
        out_dtype_str: str = "f16",
        use_mask: bool = False,
    ):
        self._gemm2_exe = gemm2_exe
        self._reduce_exe = reduce_exe
        self._topk = topk
        self._model_dim = model_dim
        self._out_dtype_str = out_dtype_str
        self._use_mask = use_mask

    def _get_torch_dtype(self):
        """Convert dtype string to torch dtype."""
        import torch

        dtype_map = {
            "f16": torch.float16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "f32": torch.float32,
        }
        return dtype_map.get(self._out_dtype_str, torch.float16)

    def __call__(
        self,
        arg_out,
        arg_x,
        arg_w,
        arg_scale_x,
        arg_scale_w,
        arg_sorted_token_ids,
        arg_expert_ids,
        arg_sorted_weights,
        arg_num_valid_ids,
        tokens_in,
        n_in,
        k_in,
        size_expert_ids_in,
        valid_mask=None,
        stream=None,
    ):
        """Execute GEMM2 + reduce.

        Args match moe_gemm2 kernel signature (see compile_moe_blockscale_gemm2).
        """
        import torch

        if stream is None:
            stream = torch.cuda.current_stream()
        intermediate = torch.empty(
            tokens_in * self._topk,
            self._model_dim,
            device=arg_out.device,
            dtype=self._get_torch_dtype(),
        )
        if not self._use_mask:
            intermediate.zero_()
        # Phase 1: GEMM2 (no atomics) -> [tokens*topk, model_dim]
        self._gemm2_exe(
            intermediate.view(-1),
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_num_valid_ids,
            tokens_in,
            n_in,
            k_in,
            size_expert_ids_in,
            stream,
        )
        # Phase 2: Reduce over topk -> [tokens, model_dim]
        X = intermediate.view(tokens_in, self._topk, self._model_dim)
        Y = arg_out.view(tokens_in, self._model_dim)
        if not self._use_mask:
            if valid_mask is not None:
                logging.warning(
                    "valid_mask provided but use_mask=False; ignoring valid_mask"
                )
            valid_mask = torch.empty(
                (0, self._topk), device=arg_out.device, dtype=torch.uint8
            )
        self._reduce_exe(X, Y, valid_mask, tokens_in, stream)

    @property
    def mode(self) -> str:
        """Return the execution mode."""
        return MoeGemm2Mode.REDUCE


def compile_moe_blockscale_gemm2_ex(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    in_dtype: str = "fp8",
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    # Extended parameters for mode control
    mode: str = MoeGemm2Mode.ATOMIC,
    valid_mask=None,
):
    """Compile MoE GEMM2 kernel with optional reduction.

    This is the extended interface that supports explicit mode control.

    Args:
        mode: Execution mode selection:
            - "atomic": Use atomic accumulation (original behavior)
            - "reduce": Use non-atomic write + reduce kernel

    Returns:
        Compiled executable (either wrapped or raw depending on mode).
    """
    # Compile based on mode
    if mode == MoeGemm2Mode.REDUCE:
        # Determine if we need masked reduction
        use_mask = valid_mask is not None

        # Compile GEMM2 with accumulate=False
        gemm2_exe = compile_moe_blockscale_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            use_cshuffle_epilog=use_cshuffle_epilog,
            accumulate=False,
        )
        # Compile reduction kernel with masking support
        out_s = str(out_dtype).strip().lower()
        if out_s in ("f16", "fp16", "half"):
            dtype_str = "f16"
        elif out_s in ("bf16", "bfloat16"):
            dtype_str = "bf16"
        else:
            dtype_str = "f32"
        reduce_exe = compile_moe_reduction(
            topk=topk,
            model_dim=model_dim,
            dtype_str=dtype_str,
            use_mask=use_mask,
        )
        return _MoeGemm2ReduceWrapper(
            gemm2_exe=gemm2_exe,
            reduce_exe=reduce_exe,
            topk=topk,
            model_dim=model_dim,
            out_dtype_str=dtype_str,
            use_mask=use_mask,
        )
    else:
        # Compile GEMM2 with accumulate=True (atomic mode)
        return compile_moe_blockscale_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            use_cshuffle_epilog=use_cshuffle_epilog,
            accumulate=True,
        )


# ---------------------------------------------------------------------------
# Public Python wrappers (mirrors of `flydsl_moe_stage1` / `flydsl_moe_stage2`
# from `aiter/ops/flydsl/moe_kernels.py`, but specialised to the per_1x128
# blockscale FP8/FP8 path used by DSV4).
# ---------------------------------------------------------------------------


def _bs_run_compiled(exe, args):
    """Call the JitFunction with the given args, scrubbing leaked MLIR
    contexts on failure to avoid poisoning subsequent compilations.

    Mirrors the helper in ``aiter.ops.flydsl.moe_kernels._run_compiled`` —
    we keep a private copy here so this module remains self-contained and
    does not depend on a sibling import order.
    """
    try:
        exe(*args)
    except Exception:
        try:
            while ir.Context.current is not None:
                ir.Context.current.__exit__(None, None, None)
        except Exception:
            pass
        raise


def _bs_normalize_out_dtype(out_dtype: str) -> str:
    """Map aiter-style dtype strings to the kernel-internal short form."""
    s = str(out_dtype).strip().lower()
    if s in ("bf16", "bfloat16"):
        return "bf16"
    if s in ("f16", "fp16", "half"):
        return "f16"
    if s in ("f32", "fp32", "float"):
        return "f32"
    raise ValueError(f"unsupported blockscale out_dtype={out_dtype!r}")


def flydsl_moe_blockscale_stage1(
    a: "torch.Tensor",
    w1: "torch.Tensor",
    sorted_token_ids: "torch.Tensor",
    sorted_expert_ids: "torch.Tensor",
    num_valid_ids: "torch.Tensor",
    out: Optional["torch.Tensor"] = None,
    topk: int = 1,
    *,
    tile_m: int = 32,
    tile_n: int = 128,
    tile_k: int = 128,
    a_dtype: str = "fp8",
    b_dtype: str = "fp8",
    out_dtype: str = "bf16",
    act: str = "silu",
    w1_scale: Optional["torch.Tensor"] = None,
    a1_scale: Optional["torch.Tensor"] = None,
    sorted_weights: Optional["torch.Tensor"] = None,
    persist_m: int = 0,
    use_async_copy: bool = False,
    k_batch: int = 1,
    waves_per_eu: int = 3,
    b_nt: int = 0,
    gate_mode: str = "interleave",
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    bias: Optional["torch.Tensor"] = None,
    a_scale_one: bool = False,
    xcd_swizzle: int = 0,
    scale_block_k: int = 128,
    use_cshuffle_epilog: Optional[bool] = None,
):
    """Stage1 (gate+up GEMM) for FlyDSL blockscale FP8/FP8 MoE.

    Mirrors the API of :func:`aiter.ops.flydsl.moe_kernels.flydsl_moe_stage1`
    so the AITER ``fused_moe`` dispatcher can call this wrapper with the same
    signature regardless of the underlying kernel family.

    Differences vs the FP4 wrapper:

    * ``a_dtype`` / ``b_dtype`` are always ``"fp8"`` (per_1x128 blockscale).
    * ``out_dtype`` is always one of ``"bf16"`` / ``"f16"``.
    * ``scale_block_k`` is fixed at 128 (per_1x128 quant scheme).
    * No fused activation+quant epilog: stage1 returns bf16/f16 SiLU(g)*u.
    """
    del (
        act,
        persist_m,
        use_async_copy,
        k_batch,
        gate_mode,
    )  # unused (silu fused in kernel)
    del b_nt, model_dim_pad, inter_dim_pad, bias, a_scale_one, xcd_swizzle

    if a_dtype != "fp8" or b_dtype != "fp8":
        raise ValueError(
            "flydsl_moe_blockscale_stage1 requires a_dtype=b_dtype='fp8'; "
            f"got a_dtype={a_dtype!r}, b_dtype={b_dtype!r}"
        )

    out_short = _bs_normalize_out_dtype(out_dtype)
    if out_short not in ("f16", "bf16"):
        raise ValueError(
            f"flydsl_moe_blockscale_stage1 out_dtype must be 'bf16' or 'f16', got {out_dtype!r}"
        )

    token_num = a.shape[0]
    E = w1.shape[0]
    inter_dim = w1.shape[1] // 2
    model_dim = a.shape[1]

    torch_out_dtype = torch.bfloat16 if out_short == "bf16" else torch.float16
    dev = a.device

    if out is None:
        out = torch.empty(
            (token_num, topk, inter_dim), dtype=torch_out_dtype, device=dev
        )

    flat_a_scale = (
        a1_scale.view(-1)
        if a1_scale is not None
        else torch.empty(0, device=dev, dtype=torch.float32)
    )
    flat_w_scale = (
        w1_scale.view(-1)
        if w1_scale is not None
        else torch.empty(0, device=dev, dtype=torch.float32)
    )
    sw = (
        sorted_weights
        if sorted_weights is not None
        else torch.empty(0, device=dev, dtype=torch.float32)
    )

    _sort_block_m = tile_m
    _all_blks = sorted_expert_ids.shape[0]
    _dense_blks = (
        min(token_num * topk * _sort_block_m, sorted_token_ids.shape[0])
        // _sort_block_m
    )
    _grid_y = min(_dense_blks, _all_blks)

    stream = torch.cuda.current_stream()

    args = (
        out.view(-1),
        a.view(-1),
        w1.view(-1),
        flat_a_scale,
        flat_w_scale,
        sorted_token_ids,
        sorted_expert_ids,
        sw,
        num_valid_ids,
        token_num,
        inter_dim,
        model_dim,
        _grid_y,
        stream,
    )

    exe = compile_moe_blockscale_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=(sorted_weights is not None),
        scale_block_k=scale_block_k,
        out_dtype=out_short,
        use_cshuffle_epilog=use_cshuffle_epilog,
        waves_per_eu=waves_per_eu,
    )
    _bs_run_compiled(exe, args)

    return out


def flydsl_moe_blockscale_stage2(
    inter_states: "torch.Tensor",
    w2: "torch.Tensor",
    sorted_token_ids: "torch.Tensor",
    sorted_expert_ids: "torch.Tensor",
    num_valid_ids: "torch.Tensor",
    out: Optional["torch.Tensor"] = None,
    topk: int = 1,
    *,
    tile_m: int = 32,
    tile_n: int = 128,
    tile_k: int = 128,
    a_dtype: str = "fp8",
    b_dtype: str = "fp8",
    out_dtype: str = "bf16",
    mode: str = "atomic",
    w2_scale: Optional["torch.Tensor"] = None,
    a2_scale: Optional["torch.Tensor"] = None,
    sorted_weights: Optional["torch.Tensor"] = None,
    sort_block_m: int = 0,
    persist: Optional[bool] = None,
    b_nt: int = 0,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    xcd_swizzle: int = 0,
    bias: Optional["torch.Tensor"] = None,
    scale_block_k: int = 128,
    use_cshuffle_epilog: Optional[bool] = None,
    waves_per_eu: Optional[int] = None,
) -> "torch.Tensor":
    """Stage2 (down-projection GEMM) for FlyDSL blockscale FP8/FP8 MoE.

    Mirrors the API of :func:`aiter.ops.flydsl.moe_kernels.flydsl_moe_stage2`.

    Currently only the ``mode="atomic"`` path is supported (atomic-add into
    the output buffer). The ``mode="reduce"`` path is left to a follow-up
    once the upstream ``compile_moe_blockscale_gemm2_ex`` REDUCE wiring is
    fixed (it currently passes an unsupported ``in_dtype=`` kwarg through).
    """
    del persist, b_nt, model_dim_pad, inter_dim_pad, xcd_swizzle, bias

    if a_dtype != "fp8" or b_dtype != "fp8":
        raise ValueError(
            "flydsl_moe_blockscale_stage2 requires a_dtype=b_dtype='fp8'; "
            f"got a_dtype={a_dtype!r}, b_dtype={b_dtype!r}"
        )

    out_short = _bs_normalize_out_dtype(out_dtype)
    accumulate = mode != "reduce"
    if not accumulate:
        # The reduce wrapper relies on `compile_moe_blockscale_gemm2_ex`,
        # which currently has an upstream bug (in_dtype kwarg mismatch).
        # Bail explicitly so callers can pick the atomic path until that
        # is reconciled in FlyDSL upstream.
        raise NotImplementedError(
            "flydsl_moe_blockscale_stage2(mode='reduce') is not wired up in "
            "this aiter port yet; use mode='atomic' (the default)."
        )

    token_num = inter_states.shape[0]
    E = w2.shape[0]
    model_dim = w2.shape[1]
    inter_dim = inter_states.shape[2]

    torch_out_dtype = torch.bfloat16 if out_short == "bf16" else torch.float16
    dev = inter_states.device

    if out is None:
        # Atomic-mode kernel accumulates into `out`, so it must start zeroed.
        out = torch.zeros((token_num, model_dim), dtype=torch_out_dtype, device=dev)

    flat_a_scale = (
        a2_scale.view(-1)
        if a2_scale is not None
        else torch.empty(0, device=dev, dtype=torch.float32)
    )
    flat_w_scale = (
        w2_scale.view(-1)
        if w2_scale is not None
        else torch.empty(0, device=dev, dtype=torch.float32)
    )
    sw = (
        sorted_weights
        if sorted_weights is not None
        else torch.empty(sorted_token_ids.shape, dtype=torch.float32, device=dev)
    )

    _sbm = sort_block_m if sort_block_m > 0 else tile_m
    if _sbm == tile_m:
        m_blocks = min(sorted_expert_ids.shape[0], token_num * topk)
    else:
        total_sorted = sorted_expert_ids.shape[0] * _sbm
        m_blocks = (total_sorted + tile_m - 1) // tile_m

    stream = torch.cuda.current_stream()

    args = (
        out,
        inter_states,
        w2,
        flat_a_scale,
        flat_w_scale,
        sorted_token_ids,
        sorted_expert_ids,
        sw,
        num_valid_ids,
        token_num,
        model_dim,
        inter_dim,
        m_blocks,
        stream,
    )

    exe = compile_moe_blockscale_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=(sorted_weights is not None),
        scale_block_k=scale_block_k,
        out_dtype=out_short,
        use_cshuffle_epilog=use_cshuffle_epilog,
        accumulate=True,
        waves_per_eu=waves_per_eu,
    )
    _bs_run_compiled(exe, args)

    return out
