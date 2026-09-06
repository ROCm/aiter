# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE stage1 gate-up kernel builder."""

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm, vector
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch

from . import layout_helpers as fxh
from .common import get_device_cache_key


def _build_moe_gemm1(
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    stage="gateup",
    alg="splitk",
    E=None,
    USE_ATOMIC_WRITE=True,
    act_quant_type=None,
    tile_k=None,
    activation="silu",
    swiglu_limit=None,
    down_path="default",
    down_output_padding_bytes=None,
    METADATA_TILE_SIZE_M=None,
    mxfp4_gate_up_interleaved=True,
    fused_down_clear=False,
):
    assert stage == "gateup"
    is_gfx950 = "gfx950" in str(get_rocm_arch())
    TILE_K = 64
    # Optional TILE_K override for the prefill_1x4 alg. The env fallback lets test_moe.py /
    # profile scripts pick BK without threading a kwarg through every caller. bf16 prefill_1x4
    # supports BK in {64, 128} (the per-ki gemm loop); fp8 stays 128.
    if tile_k is None and os.environ.get("MOE_PREFILL_TILE_K"):
        tile_k = int(os.environ["MOE_PREFILL_TILE_K"])
    # weight_quant_type governs the WEIGHT scale form; act_quant_type governs the ACTIVATION
    # scale form (native-fp8 prefill only) and defaults to weight_quant_type (previous behavior
    # where a single quant_type drove both).
    if act_quant_type is None:
        act_quant_type = "no" if weight_dtype == "fp4" else weight_quant_type
    assert (
        BLOCK_TILE_SIZE_M <= 256
    ), "BLOCK_SIZE_M must be less than or equal to 256 due to LDS size limit for sorted ids."
    assert weight_dtype in [
        "bf16",
        "fp8",
        "fp4",
    ], "weight_dtype must be one of 'bf16', 'fp8' or 'fp4'"
    assert weight_quant_type in [
        "no",
        "ptpc",
        "per_tensor",
        "mxfp4",
    ], "weight_quant_type must be one of 'no', 'ptpc', 'per_tensor' or 'mxfp4'"
    assert act_quant_type in [
        "no",
        "ptpc",
        "per_tensor",
    ], "act_quant_type must be either 'no', 'ptpc' or 'per_tensor'"
    assert activation in [
        "silu",
        "swiglu",
        "situv2",
    ], "activation must be 'silu', 'swiglu' or 'situv2'"
    if activation in ("swiglu", "situv2"):
        swiglu_limit = float(swiglu_limit) if swiglu_limit else 7.0
    if weight_dtype == "fp4":
        assert (
            weight_quant_type == "mxfp4" and act_quant_type == "no"
        ), "fp4 requires mxfp4 weights and bf16 activations"
        assert K % 512 == 0, f"fp4 gateup K must be a multiple of 512, got {K}"
    else:
        assert weight_quant_type != "mxfp4", "mxfp4 quantization requires fp4 weights"
    if METADATA_TILE_SIZE_M is None:
        METADATA_TILE_SIZE_M = BLOCK_TILE_SIZE_M
    if stage == "gateup" and alg == "prefill_1x4":
        assert 32 <= BLOCK_TILE_SIZE_M <= METADATA_TILE_SIZE_M
        assert BLOCK_TILE_SIZE_M % 32 == 0
        assert METADATA_TILE_SIZE_M % BLOCK_TILE_SIZE_M == 0
    else:
        assert (
            BLOCK_TILE_SIZE_M == METADATA_TILE_SIZE_M
        ), "only gateup prefill_1x4 supports different kernel/metadata M tiles"
    gateup_tasks_per_metadata = METADATA_TILE_SIZE_M // BLOCK_TILE_SIZE_M
    assert down_path == "default"
    assert down_output_padding_bytes is None
    # Supported native-fp8 prefill (weight, act) combos: weight ptpc requires act ptpc;
    # weight per_tensor allows act ptpc or per_tensor.
    if weight_dtype == "fp8" and alg == "prefill_1x4":
        assert (weight_quant_type == "ptpc" and act_quant_type == "ptpc") or (
            weight_quant_type == "per_tensor"
            and act_quant_type in ("ptpc", "per_tensor")
        ), (
            f"unsupported prefill quant combo (weight={weight_quant_type}, "
            f"act={act_quant_type})"
        )

    if alg == "splitk":
        assert (
            BLOCK_TILE_SIZE_N % 64 == 0
        ), "For split-k, BLOCK_TILE_SIZE_N needs to be multiple of 64 due to reduce layout."
        assert K % (32 * 4) == 0, "K must be a multiple of 128 for split-k algorithm."
        c_reduce_lds_size = (
            16 * 64 * 4
        )  # save LDS size instead of BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_N * 4

        @fx.union
        class SharedStorage:
            sorted_lds: fx.Array[fx.Int32, 256, 16]
            c_reduce_lds: fx.Array[fx.Float32, c_reduce_lds_size, 16]

    elif alg == "batch1":
        c_reduce_lds_size = (
            16 * 64 * 4
        )  # save LDS size instead of BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_N * 4

        @fx.struct
        class SharedStorage:
            c_reduce_lds: fx.Array[fx.Float32, c_reduce_lds_size, 16]

    elif alg == "prefill_1x4":
        # 1x4: the 4 waves tile the N(channel) direction, and the full TILE_M is shared across
        # all waves (no M-split). B (weight gate/up)
        # loads direct global->register (no LDS); only A (activation) is staged through LDS
        # with a ping-pong double buffer (a_ping / a_pong).
        assert 128 <= BLOCK_TILE_SIZE_N <= 256 and BLOCK_TILE_SIZE_N % 128 == 0, (
            "For prefill_1x4 alg, BLOCK_TILE_SIZE_N must be in [128, 256] and a multiple of 128 "
            "(each wave owns contiguous_n//4 = BN//8 output channels, a multiple of 16)"
        )
        assert N % BLOCK_TILE_SIZE_N == 0, (
            f"gateup prefill_1x4 requires a complete N tile, got N={N}, "
            f"BLOCK_TILE_SIZE_N={BLOCK_TILE_SIZE_N}"
        )
        # fp8 issues 128-bit g2r loads (16 fp8/thread) -> widen K-tile to 128; bf16 keeps 64.
        # tile_k override (default 64 for bf16 / 128 for fp8) enables the BK sweep; the per-ki
        # gemm loop in _gemm_1x4 handles bf16 TILE_K in {64, 128} and fp8 TILE_K in {128, 256}.
        TILE_K = (
            tile_k if tile_k is not None else (128 if weight_dtype == "fp8" else 64)
        )
        if weight_dtype == "fp8":
            assert TILE_K in (
                128,
                256,
            ), f"prefill_1x4 fp8 TILE_K must be 128 or 256, got {TILE_K}"
        else:
            assert TILE_K in (
                64,
                128,
            ), f"prefill_1x4 bf16 TILE_K must be 64 or 128, got {TILE_K}"
        assert (
            K % TILE_K == 0
        ), f"prefill_1x4 K={K} must be a multiple of TILE_K={TILE_K}"
        # The 2-stage software pipeline in _gemm_1x4 consumes K-tiles two at a time (main loop
        # + a 2-stage tail), so the tile count must be even or the middle tile(s) get skipped.
        assert (K // TILE_K) % 2 == 0, (
            f"prefill_1x4 needs an even K-tile count for the 2-stage pipeline, got "
            f"K//TILE_K={K // TILE_K} (K={K}, TILE_K={TILE_K})"
        )
        # BLOCK_TILE_SIZE_M lower bound comes from two independent sources:
        #   (1) A global->register load: 256 threads tile the (BM, TILE_K) A-tile, each issuing
        #       one 128-bit buffer_load (val_per_thr = 16B/elem = 8 bf16 / 16 fp8). That covers
        #       _thrs_m = 256 // (TILE_K // val_per_thr) M-rows per pass, so BM must be >= and a
        #       multiple of _thrs_m. _thrs_m = 32 for (bf16 TK=64 / fp8 TK=128); 16 for
        #       (bf16 TK=128 / fp8 TK=256).
        #   (2) CShuffle epilogue read: the (32 token x 64 channel) read tile walks the staged
        #       LDS with rep_token = BM // 32, so BM must be a multiple of 32 (>= 32).
        # (2) always dominates (1) (32 >= _thrs_m and 32 % _thrs_m == 0), so the effective
        # range is a 32-multiple in [32, 256] regardless of TILE_K / dtype.
        _val_per_thr = 16 if weight_dtype == "fp8" else 8
        _thrs_m_aload = 256 // (TILE_K // _val_per_thr)
        assert 32 <= BLOCK_TILE_SIZE_M <= 256 and BLOCK_TILE_SIZE_M % 32 == 0, (
            f"For prefill_1x4 alg, BLOCK_TILE_SIZE_M must be a multiple of 32 in [32, 256] "
            f"(A g2r load needs BM >= {_thrs_m_aload}; the CShuffle 32-token read tile forces a "
            f"32-multiple, so 32 is the effective minimum). got BM={BLOCK_TILE_SIZE_M}"
        )
        a_lds_size = (
            BLOCK_TILE_SIZE_M * TILE_K
        )  # full A tile; ping-pong needs two buffers
        lds_elem = fx.Float8E4M3FNUZ if weight_dtype == "fp8" else fx.BFloat16

        @fx.struct
        class GemmBuffers:
            a_ping: fx.Array[lds_elem, a_lds_size, 16]
            a_pong: fx.Array[lds_elem, a_lds_size, 16]

        @fx.union
        class SharedStorage:
            sorted_lds: fx.Array[fx.Int32, 256, 16]
            gemm: GemmBuffers

        # The CShuffle epilogue stages the single (BM x contiguous_n) bf16 output into the
        # A LDS (reused after the main loop). Confirm it fits.
        _cshuffle_elem_bytes = 1 if weight_dtype == "fp8" else 2
        _gemm_lds_bytes = 2 * a_lds_size * _cshuffle_elem_bytes
        _cshuffle_bytes = BLOCK_TILE_SIZE_M * (BLOCK_TILE_SIZE_N // 2) * 2
        assert _cshuffle_bytes <= _gemm_lds_bytes, (
            f"CShuffle needs {_cshuffle_bytes} B of LDS but GemmBuffers only allocates "
            f"{_gemm_lds_bytes} B (BM={BLOCK_TILE_SIZE_M}, BN={BLOCK_TILE_SIZE_N})"
        )
        # gfx942 (MI300X) has only 64 KB LDS per workgroup. The A ping-pong (a_ping + a_pong
        # = 2 x BM x TILE_K x sizeof(elem)) is the dominant consumer (sorted_lds is unioned
        # with it and far smaller), so _gemm_lds_bytes is effectively the whole LDS usage. A
        # large BM x wide TILE_K (e.g. fp8 BM=256 TK=256 -> 128 KB) overflows it; reject early.
        assert _gemm_lds_bytes <= 64 * 1024, (
            f"prefill_1x4 A ping-pong needs {_gemm_lds_bytes} B of LDS but gfx942 allows "
            f"{64 * 1024} B (BM={BLOCK_TILE_SIZE_M}, TILE_K={TILE_K}, "
            f"dtype={'fp8' if weight_dtype == 'fp8' else 'bf16'})"
        )

    if weight_dtype == "bf16":
        weight_dtype = fx.BFloat16
    elif weight_dtype == "fp8":
        weight_dtype = fx.Float8E4M3FNUZ
    elif weight_dtype == "fp4":
        assert alg in ("batch1", "splitk"), "fp4 is only supported by batch1/splitk"
        assert is_gfx950, "fp4 batch1/splitk is only supported on gfx950"
        weight_dtype = fx.Float4E2M1FN

    def _encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
        """Encode s_waitcnt bitfield for CDNA3 (gfx94x)."""
        vm_lo = vmcnt & 0xF
        vm_hi = (vmcnt >> 4) & 0x3
        return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)

    class TensorWithIndex:
        # view: real tensor
        # tile_m, tile_k: tile size in M/K dimension for each copy from global to shared
        # index_frag: pre-read sorted-M-row index fragment (built by _read_sorted_index). The
        #     caller reads it BEFORE constructing this object because sorted_lds is reused
        #     (e.g. overwritten by the CShuffle epilogue) before copy() runs.
        # tiled_copy: thread mapping for copy
        # tid: thread id for copy
        def __init__(
            self,
            view,
            tile_m,
            tile_k,
            index_frag,
            tiled_copy: fx.TiledCopy,
            tid,
            is_read_from_mem=True,
            TOPK=None,
            is_atomic_write=False,
        ):
            assert not (is_atomic_write and is_read_from_mem)
            self.view = view
            self.tile_m = tile_m
            self.tile_k = tile_k
            self.is_read_from_mem = is_read_from_mem
            self.TOPK = TOPK
            self.is_atomic_write = is_atomic_write
            self.index_frag = index_frag

            # split into (1, tile_k) blocks
            rank = fx.get_shape(self.view).rank
            dims = [1] * (rank - 1)
            # shape: [(1, tile_k), (m, rep_k)]
            self.tensor_blocks_in_k = fx.zipped_divide(
                view, fx.make_tile(*dims, tile_k)
            )

            dtype = fx.PointerType.get(fx.Int8.ir_type, 1, 512)
            ptr = fx.inttoptr(dtype, fx.Int32(0))
            self.fake_tensor = fx.make_view(
                ptr, fx.make_layout((tile_m, tile_k), (1, tile_m))
            )
            self.fake_tensor_thr = (
                tiled_copy.get_slice(tid).partition_S(self.fake_tensor)
                if is_read_from_mem
                else tiled_copy.get_slice(tid).partition_D(self.fake_tensor)
            )
            # since init ptr is zero, it will be the offset of the thread in the tile after partition_S
            offset_thread = fx.Int32(fx.ptrtoint(fx.get_iter(self.fake_tensor_thr)))
            self.offset_thread_k = offset_thread // tile_m

        @flyc.jit
        def copy(self, copy_atom, k_idx, frag: fx.Tensor, extra_offset=0):
            layout = fx.get_layout(self.fake_tensor_thr)
            shape = fx.get_shape(self.fake_tensor_thr)
            rep_m = fx.size(shape[1]).to_py_value()
            rep_k = fx.size(shape[2]).to_py_value()
            value_size = fx.get_shape(frag)[0].to_py_value()
            stride_size = fx.get_stride(frag)[0].to_py_value()

            rank = fx.get_shape(self.view).rank
            block_cord = [None] * (rank - 1) + [k_idx]
            # current iter block (M dimension is not indexed), shape: [(1, tile_k), m]
            tensor_block = self.tensor_blocks_in_k[None, (*block_cord,)]
            for m in range_constexpr(rep_m):
                # current iter subblock with correct M index, shape: [(1, tile_k)]
                if const_expr(rank == 2):
                    tensor_sub_block = tensor_block[
                        None, self.index_frag[0, m] & 0xFFFFFF
                    ]
                else:
                    tensor_sub_block = tensor_block[
                        None,
                        self.index_frag[0, m] & 0xFFFFFF,
                        (self.index_frag[0, m] >> 24),
                    ]
                if const_expr(not self.is_atomic_write):
                    for k in range_constexpr(rep_k):
                        # get block k index
                        offset_block = fx.crd2idx((0, m, k), layout).to_py_value()
                        offset_block_k = offset_block // self.tile_m
                        # NOTE: assume K is linear in memory
                        offset_k_in_tile = offset_block_k + self.offset_thread_k
                        reg = frag[None, m, k]
                        mem = fx.make_view(
                            fx.get_iter(tensor_sub_block)
                            + offset_k_in_tile
                            + extra_offset,
                            fx.make_layout(value_size, stride_size),
                        )
                        if const_expr(self.is_read_from_mem):
                            fx.copy(copy_atom, mem, reg)
                        else:
                            fx.copy(copy_atom, reg, mem)
                else:
                    # fx.UniversalAtomic(fx.AtomicOp.Add) could not lower to `global_atomic_pk_add_bf16`, hack to emit
                    if (self.index_frag[0, m] >> 24) < TOPK:
                        for k in range_constexpr(rep_k):
                            # get block k index
                            offset_block = fx.crd2idx((0, m, k), layout).to_py_value()
                            offset_block_k = offset_block // self.tile_m
                            # NOTE: assume K is linear in memory
                            offset_k_in_tile = offset_block_k + self.offset_thread_k
                            reg = frag[None, m, k]
                            mem = fx.make_view(
                                fx.get_iter(tensor_sub_block) + offset_k_in_tile,
                                fx.make_layout(value_size, stride_size),
                            )
                            reg_vec = reg.load()
                            ptr_base = fx.get_iter(mem)
                            fxh.atomic_add_bf16(ptr_base, reg_vec)

    def _read_sorted_index(
        tiled_copy_index, tid, lds_index, index_size=None, index_offset=0
    ):
        # Read the sorted M-row index from LDS into a per-thread register fragment. Kept out
        # of TensorWithIndex so the read happens at an explicit, caller-controlled point:
        # sorted_lds is reused (e.g. overwritten by the CShuffle epilogue), so the index must
        # be captured before that. tiled_copy_index maps threads to the index tile.
        if index_size is None:
            index_size = BLOCK_TILE_SIZE_M
        lds = fx.make_view(lds_index.ptr + index_offset, fx.make_layout(index_size, 1))
        cp_atom_lds = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
        lds_thr = tiled_copy_index.get_slice(tid).partition_S(lds)
        index_frag = fx.make_fragment_like(lds_thr)
        fx.copy(cp_atom_lds, lds_thr, index_frag)
        return index_frag

    def _select(tensor: fx.Tensor, order):
        rank = fx.get_shape(tensor).rank
        assert len(order) == rank
        stride = fx.get_stride(tensor)
        shape = fx.get_shape(tensor)
        new_layout = fx.make_layout(
            [shape[i] for i in order], [stride[i] for i in order]
        )
        return fx.make_view(fx.get_iter(tensor), new_layout)

    def _cvt_fp8_bf16(src_tensor: fx.Tensor, dst_tensor: fx.Tensor):
        # src_tensor is a packed-uint32 fragment (4 fp8 per dword) loaded straight from
        # memory, so each dword feeds v_cvt_pk_f32_fp8 directly -- no whole-vector load +
        # bitcast, which would emit shufflevector / v_lshrrev to repack the bytes.
        n_dwords = fx.size(fx.get_shape(src_tensor)).to_py_value()

        items = []
        src_vec = src_tensor.load()
        for i in range_constexpr(n_dwords):
            src_val = src_vec[i]
            pk0_f32 = llvm.inline_asm(
                T.f32x2,
                [as_ir_value(src_val)],
                "v_cvt_pk_f32_fp8 $0, $1",
                "=v,v",
                has_side_effects=False,
            )
            pk1_f32 = llvm.inline_asm(
                T.f32x2,
                [as_ir_value(src_val)],
                "v_cvt_pk_f32_fp8_sdwa $0, $1 src0_sel:WORD_1",
                "=v,v",
                has_side_effects=False,
            )
            tmp = (pk0_f32.bitcast(fx.Uint32) >> 16).to(fx.Uint16).bitcast(fx.BFloat16)
            items.append(tmp[0])
            items.append(tmp[1])
            tmp = (pk1_f32.bitcast(fx.Uint32) >> 16).to(fx.Uint16).bitcast(fx.BFloat16)
            items.append(tmp[0])
            items.append(tmp[1])
        vec = Vec.from_elements(items, fx.BFloat16)
        layout = fx.get_layout(dst_tensor)
        for i in range_constexpr(4 * n_dwords):
            crd = fx.idx2crd(i, layout)
            dst_tensor[crd] = vec[i]

    def _mxfp4_scale_index(
        expert_id,
        blk_n,
        local_n,
        k_group,
        gateup_contiguous_n,
    ):
        groups_padded = ((K // 32 + 7) // 8) * 8
        if const_expr(gateup_contiguous_n is not None):
            grouped_n = blk_n * BLOCK_TILE_SIZE_N + local_n
            group_n = grouped_n // (2 * gateup_contiguous_n)
            within_group = grouped_n % (2 * gateup_contiguous_n)
            gate_up_idx = within_group // gateup_contiguous_n
            channel = group_n * gateup_contiguous_n + within_group % gateup_contiguous_n
            if const_expr(mxfp4_gate_up_interleaved):
                group = fx.Int64(k_group)
                return (
                    fx.Int64(expert_id) * N * groups_padded
                    + (fx.Int64(channel) // 16 * groups_padded) * 32
                    + (group // 8) * 256
                    + (group % 4) * 64
                    + (fx.Int64(channel) % 16) * 4
                    + (group % 8 // 4) * 2
                    + gate_up_idx
                )
            row_in_expert = channel + gate_up_idx * (N // 2)
            row = fx.Int64(expert_id) * N + fx.Int64(row_in_expert)
        else:
            row = fx.Int64(expert_id) * N + fx.Int64(
                blk_n * BLOCK_TILE_SIZE_N + local_n
            )

        group = fx.Int64(k_group)
        return (
            (row // 32 * groups_padded) * 32
            + (group // 8) * 256
            + (group % 4) * 64
            + (row % 16) * 4
            + (group % 8 // 4) * 2
            + (row % 32 // 16)
        )

    def _mxfp4_scale_from_dword(packed_scale, byte_idx):
        shift = fx.Uint32(byte_idx) * 8
        scale_bits = ((packed_scale >> shift) & 0xFF) << 23
        return scale_bits.bitcast(fx.Float32)

    def _load_mxfp4_packed_scales(
        src_tensor,
        p_w_scale,
        expert_id,
        blk_n,
        k_idx,
        tile_k_per_wg,
        tid,
        gateup_contiguous_n,
    ):
        n_rows = fx.size(fx.get_shape(src_tensor)).to_py_value() // 4
        if const_expr(gateup_contiguous_n is not None and tile_k_per_wg == 512):
            wave_id = tid // 64
            lane_group = tid % 64 // 16
            k_group = wave_id * (K // 4 // 32) + k_idx * (128 // 32) + lane_group
        else:
            k_group = k_idx * (tile_k_per_wg // 32) + tid // 16
        scale_byte_ptr = fx.recast_iter(fx.Uint8, fxh._as_ptr(p_w_scale))
        scale_u32_type = fx.PointerType.get(
            fx.Uint32.ir_type, scale_byte_ptr.memspace, 4
        )
        scale_u32_ptr = fx.recast_iter(scale_u32_type, scale_byte_ptr)
        packed_scale_rows = [None] * n_rows
        if const_expr(gateup_contiguous_n is not None and mxfp4_gate_up_interleaved):
            rows_per_half = n_rows // 2
            for channel_block in range_constexpr(rows_per_half):
                local_n = tid % 16 + channel_block * 16
                scale_idx = _mxfp4_scale_index(
                    expert_id,
                    blk_n,
                    local_n,
                    k_group,
                    gateup_contiguous_n,
                )
                packed_scale = scale_u32_ptr[scale_idx // 4]
                packed_scale_rows[channel_block] = packed_scale
                packed_scale_rows[channel_block + rows_per_half] = packed_scale
        elif const_expr(
            n_rows >= 2 and (gateup_contiguous_n is None or BLOCK_TILE_SIZE_N >= 64)
        ):
            for row_pair in range_constexpr(n_rows // 2):
                local_n = tid % 16 + row_pair * 32
                scale_idx = _mxfp4_scale_index(
                    expert_id,
                    blk_n,
                    local_n,
                    k_group,
                    gateup_contiguous_n,
                )
                packed_scale = scale_u32_ptr[scale_idx // 4]
                packed_scale_rows[row_pair * 2] = packed_scale
                packed_scale_rows[row_pair * 2 + 1] = packed_scale
        else:
            for row in range_constexpr(n_rows):
                local_n = tid % 16 + row * 16
                scale_idx = _mxfp4_scale_index(
                    expert_id,
                    blk_n,
                    local_n,
                    k_group,
                    gateup_contiguous_n,
                )
                packed_scale_rows[row] = scale_u32_ptr[scale_idx // 4]
        return packed_scale_rows

    def _load_mxfp4_inputs(
        src_tensor,
        p_w_scale,
        expert_id,
        blk_n,
        k_idx,
        tile_k_per_wg,
        tid,
        gateup_contiguous_n,
        packed_scale_rows=None,
    ):
        n_dwords = fx.size(fx.get_shape(src_tensor)).to_py_value()
        src_vec = src_tensor.load()
        if const_expr(gateup_contiguous_n is not None and tile_k_per_wg == 512):
            wave_id = tid // 64
            lane_group = tid % 64 // 16
            k_group = wave_id * (K // 4 // 32) + k_idx * (128 // 32) + lane_group
        else:
            k_group = k_idx * (tile_k_per_wg // 32) + tid // 16
        n_rows = n_dwords // 4
        if const_expr(packed_scale_rows is None):
            packed_scale_rows = _load_mxfp4_packed_scales(
                src_tensor,
                p_w_scale,
                expert_id,
                blk_n,
                k_idx,
                tile_k_per_wg,
                tid,
                gateup_contiguous_n,
            )

        scales = []
        for row in range_constexpr(n_rows):
            scale_idx = _mxfp4_scale_index(
                expert_id,
                blk_n,
                tid % 16 + row * 16,
                k_group,
                gateup_contiguous_n,
            )
            scales.append(
                _mxfp4_scale_from_dword(packed_scale_rows[row], scale_idx % 4)
            )
        return src_vec, scales, packed_scale_rows

    def _decode_mxfp4_dword(packed, scale):
        items = []
        for byte_idx in range_constexpr(4):
            pair = llvm.call_intrinsic(
                T.vec(2, T.bf16),
                "llvm.amdgcn.cvt.scalef32.pk.bf16.fp4",
                [
                    as_ir_value(packed),
                    as_ir_value(scale),
                    as_ir_value(fx.Int32(byte_idx)),
                ],
                [],
                [],
            )
            items.append(
                fx.BFloat16(
                    vector.extract(pair, static_position=[0], dynamic_position=[])
                )
            )
            items.append(
                fx.BFloat16(
                    vector.extract(pair, static_position=[1], dynamic_position=[])
                )
            )
        return Vec.from_elements(items, fx.BFloat16)

    def _cvt_mxfp4_bf16(
        src_tensor,
        dst_tensor,
        p_w_scale,
        expert_id,
        blk_n,
        k_idx,
        tile_k_per_wg,
        tid,
        gateup_contiguous_n,
        packed_scale_rows=None,
    ):
        src_vec, scales, _ = _load_mxfp4_inputs(
            src_tensor,
            p_w_scale,
            expert_id,
            blk_n,
            k_idx,
            tile_k_per_wg,
            tid,
            gateup_contiguous_n,
            packed_scale_rows,
        )
        n_dwords = src_vec.numel
        items = []
        for i in range_constexpr(n_dwords):
            decoded = _decode_mxfp4_dword(src_vec[i], scales[i // 4])
            for value_idx in range_constexpr(8):
                items.append(decoded[value_idx])
        vec = Vec.from_elements(items, fx.BFloat16)
        layout = fx.get_layout(dst_tensor)
        for i in range_constexpr(8 * n_dwords):
            dst_tensor[fx.idx2crd(i, layout)] = vec[i]

    def _clamp_gateup(gate, up, neg_limit):
        gate = -((-gate).maximumf(neg_limit))
        up = (-((-up).maximumf(neg_limit))).maximumf(neg_limit)
        return gate, up

    def _swiglu_oai(gate, up, neg_limit):
        gate, up = _clamp_gateup(gate, up, neg_limit)
        neg_alpha_log2e = -1.702 * 1.4426950408889634
        tmp = rocdl.exp2(T.f32, _raw(gate * neg_alpha_log2e))
        return (gate * rocdl.rcp(T.f32, 1.0 + tmp)) * (up + 1.0)

    def _sigmoid(value):
        tmp = rocdl.exp2(T.f32, _raw(value * -1.4426950408889634))
        return rocdl.rcp(T.f32, 1.0 + tmp)

    def _tanh(value):
        abs_value = value.maximumf(-value)
        exp_value = rocdl.exp2(T.f32, _raw(abs_value * -2.8853900817779268))
        tanh_abs = (1.0 - exp_value) * rocdl.rcp(T.f32, 1.0 + exp_value)
        return (value > fx.Float32(0.0)).select(tanh_abs, -tanh_abs)

    def _situv2(
        gate,
        up,
        beta,
        beta_rcp,
        linear_beta,
        linear_beta_rcp,
        neg_limit,
    ):
        gate, up = _clamp_gateup(gate, up, neg_limit)
        situ_gate = beta * _tanh(gate * beta_rcp) * _sigmoid(gate)
        up_scaled = linear_beta * _tanh(up * linear_beta_rcp)
        return situ_gate * up_scaled

    def _apply_scale_gateup_bf16(
        c_frag,
        tid,
        expert_id,
        blk_n,
        contiguous_n,
        p_w_scale,
        situ_beta_f,
        situ_beta_rcp_f,
        situ_linear_beta_f,
        situ_linear_beta_rcp_f,
        activation_limit_f,
    ):
        # The reduce makes gate/up adjacent (2i, 2i+1).
        v_reps = fx.size(fx.get_shape(c_frag)[0]).to_py_value()
        m_reps = fx.size(fx.get_shape(c_frag)[1]).to_py_value()
        n_reps = fx.size(fx.get_shape(c_frag)[2]).to_py_value()

        if const_expr(weight_dtype != fx.BFloat16):
            if const_expr(weight_quant_type == "ptpc"):
                group_layout_silu = fx.make_layout(
                    ((contiguous_n, 2, N // (2 * contiguous_n)), 1),
                    ((1, N // 2, contiguous_n), 0),
                )
                arg_p_scale = fx.make_view(
                    fxh._as_ptr(p_w_scale) + expert_id * N,
                    fx.composition(fx.make_layout(N, 1), group_layout_silu),
                )
                scale_tile = fx.flat_divide(
                    arg_p_scale, fx.make_tile(BLOCK_TILE_SIZE_N, 1)
                )[None, None, blk_n, 0]
                cp_atom_scale = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
                tiled_copy_scale = fx.make_tiled_copy(
                    cp_atom_scale,
                    fx.make_layout(
                        ((16, 4, 4), contiguous_n // 16),
                        ((contiguous_n // 16, 0, 0), 1),
                    ),
                    fx.make_tile(contiguous_n, 1),
                )
                scale_frag_tensor = tiled_copy_scale.get_slice(tid).partition_S(
                    scale_tile
                )
                scale_frag = fx.make_fragment_like(scale_frag_tensor)
                fx.copy(cp_atom_scale, scale_frag_tensor, scale_frag)
                for n in range_constexpr(n_reps):
                    scale_vec = scale_frag[None, n, 0].load()
                    for m in range_constexpr(m_reps):
                        c_vec = c_frag[None, m, n].load()
                        vec = c_vec * scale_vec
                        c_frag[None, m, n].store(vec)
            elif const_expr(weight_quant_type == "per_tensor"):
                arg_p_scale = fx.make_view(
                    fxh._as_ptr(p_w_scale) + expert_id, fx.make_layout(1, 1)
                )
                scale = arg_p_scale[0]
                c_frag.store(c_frag.load() * scale)

        # c_frag_bf16 stores the gateup activation result (N/2 outputs).
        n_half = n_reps // 2
        if const_expr(v_reps == 1):
            # v_reps==1 (TILE_N=32): flat value mode with stride 0 to avoid
            # ((1,1),...):((1,0),...) producing two stride-1 leaves in findContigSegment.
            c_frag_bf16 = fx.make_rmem_tensor(
                fx.make_layout((1, m_reps, n_half), (0, n_half, 1)), fx.BFloat16
            )
        else:
            c_frag_bf16 = fx.make_rmem_tensor(
                fx.make_layout(
                    ((v_reps, 1), m_reps, n_half), ((1, 0), n_half * v_reps, v_reps)
                ),
                fx.BFloat16,
            )

        for i in range_constexpr(n_reps // 2):
            gate = c_frag[None, None, 2 * i + 0].load()
            up = c_frag[None, None, 2 * i + 1].load()
            acc = []
            neg_limit = -activation_limit_f
            if const_expr(activation == "swiglu"):
                for j in range_constexpr(gate.numel):
                    acc.append(_swiglu_oai(gate[j], up[j], neg_limit))
            elif const_expr(activation == "situv2"):
                for j in range_constexpr(gate.numel):
                    acc.append(
                        _situv2(
                            gate[j],
                            up[j],
                            situ_beta_f,
                            situ_beta_rcp_f,
                            situ_linear_beta_f,
                            situ_linear_beta_rcp_f,
                            neg_limit,
                        )
                    )
            else:
                log2_exp1 = -1.4426950408889634
                gate_log2 = gate * log2_exp1
                for j in range_constexpr(gate.numel):
                    tmp = rocdl.exp2(T.f32, _raw(gate_log2[j]))
                    acc.append((gate[j] * rocdl.rcp(T.f32, 1.0 + tmp)) * up[j])
            acc = Vec.from_elements(acc, fx.Float32)
            round_bit = fx.Uint32(0x8000)
            acc = (
                ((acc.bitcast(fx.Uint32) + round_bit) >> 16)
                .to(fx.Uint16)
                .bitcast(fx.BFloat16)
            )
            c_frag_bf16[None, None, i].store(acc)

        return c_frag_bf16

    def _gateup_pair_bf16(
        gate_frag,
        up_frag,
        situ_beta_f,
        situ_beta_rcp_f,
        situ_linear_beta_f,
        situ_linear_beta_rcp_f,
        activation_limit_f,
        gate_scale=None,
        up_scale=None,
        a_scale=None,
    ):
        # Apply the selected activation over identically-laid-out gate/up fragments.
        # Used by the 4-wave compute path where gate (left N-half) and up (right N-half)
        # land in separate quadrant fragments with matching layout. Iterate (m, n)
        # explicitly so the result keeps the fragment's [v, m, n] positions. Optional
        # per-N-channel fp8 weight scales (shape [value, rep_n]) and an optional per-row
        # fp8 activation scale (a_scale[m], one per C M-row) are folded into the read so
        log2_exp1 = -1.4426950408889634
        round_bit = fx.Uint32(0x8000)
        out_bf16 = fx.make_fragment_like(gate_frag, dtype=fx.BFloat16)
        m_reps = fx.size(fx.get_shape(gate_frag)[1]).to_py_value()
        n_reps = fx.size(fx.get_shape(gate_frag)[2]).to_py_value()
        for m in range_constexpr(m_reps):
            if const_expr(a_scale is not None):
                a_sc = a_scale[m]
            for n in range_constexpr(n_reps):
                gate = gate_frag[None, m, n].load()
                up = up_frag[None, m, n].load()
                if const_expr(gate_scale is not None):
                    sc_g = gate_scale[None, n].load()
                    sc_u = up_scale[None, n].load()
                acc = []
                for j in range_constexpr(gate.numel):
                    g = gate[j]
                    u = up[j]
                    if const_expr(gate_scale is not None):
                        g = g * sc_g[j]
                        u = u * sc_u[j]
                    if const_expr(a_scale is not None):
                        g = g * a_sc
                        u = u * a_sc
                    if const_expr(activation == "swiglu"):
                        acc.append(_swiglu_oai(g, u, -activation_limit_f))
                    elif const_expr(activation == "situv2"):
                        acc.append(
                            _situv2(
                                g,
                                u,
                                situ_beta_f,
                                situ_beta_rcp_f,
                                situ_linear_beta_f,
                                situ_linear_beta_rcp_f,
                                -activation_limit_f,
                            )
                        )
                    else:
                        tmp = rocdl.exp2(T.f32, _raw(g * log2_exp1))
                        acc.append((g * rocdl.rcp(T.f32, 1.0 + tmp)) * u)
                acc = Vec.from_elements(acc, fx.Float32)
                acc = (
                    ((acc.bitcast(fx.Uint32) + round_bit) >> 16)
                    .to(fx.Uint16)
                    .bitcast(fx.BFloat16)
                )
                out_bf16[None, m, n].store(acc)
        return out_bf16

    def _make_gateup_weight_view(p_weight, expert_id, contiguous_n):
        storage_k = K // 2 if const_expr(weight_dtype == fx.Float4E2M1FN) else K
        group_layout_silu = fx.make_layout(
            ((contiguous_n, 2, N // (contiguous_n * 2)), storage_k),
            ((1, N // 2, contiguous_n), N),
        )
        if const_expr(weight_dtype == fx.Float4E2M1FN and mxfp4_gate_up_interleaved):
            return fx.make_view(
                p_weight + fx.Int64(expert_id) * N * storage_k,
                fx.composition(
                    fx.make_layout(
                        (((16, N // 32), 2), (16, 4, storage_k // 64)),
                        (
                            ((16, 32 * storage_k), 16 * storage_k),
                            (1, 256, 1024),
                        ),
                    ),
                    group_layout_silu,
                ),
            )

        element_num = 16 // (p_weight.dtype.width // 8)
        return fx.make_view(
            p_weight + fx.Int64(expert_id) * N * storage_k,
            fx.composition(
                fx.make_layout(
                    ((16, N // 16), (element_num, storage_k // element_num)),
                    ((element_num, 16 * storage_k), (1, 16 * element_num)),
                ),
                group_layout_silu,
            ),
        )

    def _setup_b_operand(
        arg_p_weight, arg_p_input, tiled_mma, blk_n, TILE_N, tile_k_per_wg, tid
    ):
        # B (weight) operand setup for _gemm_splitk. bf16: load directly as the MFMA B-operand
        # (b_frag and b_frag_retile are two views of the same storage). fp8/mxfp4 load packed
        # uint32 values for decompression in the main loop. Returns
        # (b_cp_atom_r, b_tensor_thr, b_frag, b_frag_retile).
        if weight_dtype == fx.BFloat16:
            b_tensor = fx.rocdl.make_buffer_tensor(arg_p_weight, max_size=False)
            # shape: [n_in_tile, k_in_tile, k_tile]
            b_tile = fx.flat_divide(b_tensor, fx.make_tile(TILE_N, tile_k_per_wg))[
                None, None, blk_n, None
            ]
            b_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), weight_dtype)
            b_tiled_thr = fx.make_tiled_copy_B(b_cp_atom_r, tiled_mma).get_slice(tid)
            b_tensor_thr = b_tiled_thr.partition_S(b_tile)
            b_frag = [
                tiled_mma.make_fragment_B(b_tile[None, None, 0]),
                tiled_mma.make_fragment_B(b_tile[None, None, 0]),
            ]
            b_frag_retile = [
                b_tiled_thr.retile(b_frag[0]),
                b_tiled_thr.retile(b_frag[1]),
            ]
            return b_cp_atom_r, b_tensor_thr, b_frag, b_frag_retile

        # b_frag will be decompressed from fp8 or packed mxfp4.
        b_fake_tensor = fx.make_view(
            fx.get_iter(arg_p_input),
            fx.make_layout((TILE_N, tile_k_per_wg), (tile_k_per_wg, 1)),
        )
        b_frag = [
            tiled_mma.make_fragment_B(b_fake_tensor),
            tiled_mma.make_fragment_B(b_fake_tensor),
        ]

        # Load the fp8 weights as packed uint32 dwords (4 fp8 / dword) so cvt_fp8_bf16 can
        # feed each dword straight into v_cvt_pk_f32_fp8 -- avoids the whole-vector load +
        # bitcast that LLVM lowers to shufflevector / v_lshrrev byte repacking.
        #
        # Recast fp8 -> uint32 at the SOURCE (before make_buffer_tensor): arg_p_weight's
        # iter is a plain pointer whose expert offset is already a byte address, so
        # recast_iter (reinterpret_cast) keeps the address while recast_layout collapses
        # the contiguous 16 fp8 into 4 dwords (/4 on every stride). Recasting the buffer
        # descriptor AFTER partition is wrong: the block/thread offset is baked into the
        # descriptor in fp8 ELEMENTS, and recast_iter would not divide it by 4 (-> 4x
        # address error). The fp8 pointer is align=1, so build the uint32 pointer
        # explicitly with a 16B alignment (the 128b tiles are already 16B-aligned).
        _w_it = fx.get_iter(arg_p_weight)
        _w_u32_ptr = fx.PointerType.get(fx.Uint32.ir_type, _w_it.memspace, 16)
        arg_w_u32 = fx.make_view(
            fx.recast_iter(_w_u32_ptr, _w_it),
            fx.recast_layout(fx.get_layout(arg_p_weight), 8, 32),
        )
        b_tensor_u32 = fx.rocdl.make_buffer_tensor(arg_w_u32, max_size=False)
        packed_per_dword = 8 if const_expr(weight_dtype == fx.Float4E2M1FN) else 4
        b_tile = fx.flat_divide(
            b_tensor_u32, fx.make_tile(TILE_N, tile_k_per_wg // packed_per_dword)
        )[None, None, blk_n, None]
        b_cp_atom_r = fx.make_copy_atom(
            (
                fx.rocdl.BufferCopy128b(cache_modifier=3)
                if const_expr(weight_dtype == fx.Float4E2M1FN)
                else fx.rocdl.BufferCopy128b()
            ),
            fx.Uint32,
        )
        # uint32 thread-value layout mirrors the fp8 tv_layout_B_tiled. Recasting the fp8
        # weight to uint32 keeps the thread's contiguous-inner stride (1) but divides the
        # K-group stride by 4 (4 fp8 = 1 dword), so derive the uint32 thread strides from
        # the fp8 tv (divide any stride >= 4 by 4). value = 4 contiguous uint32 (= the 16
        # fp8 each thread loads with one 128b buffer_load). Using tile_k_per_wg//4 for the
        # K-group stride is wrong for splitk_waves=1 (the preshuffle K stride is fixed by
        # the weight layout, not the tile width).
        n_mma = fx.get_scalar(fx.size(fx.select(tiled_mma.tile_size_mnk, [1])))
        _tvB = tiled_mma.tv_layout_B_tiled
        _n0 = fx.get_scalar(_tvB.shape[0][0])
        _n1 = fx.get_scalar(_tvB.shape[0][1])
        _s0 = fx.get_scalar(_tvB.stride[0][0])
        _s1 = fx.get_scalar(_tvB.stride[0][1])
        _s0 = _s0 if _s0 < packed_per_dword else _s0 // packed_per_dword
        _s1 = _s1 if _s1 < packed_per_dword else _s1 // packed_per_dword
        values_per_copy = 4
        tv_u32 = fx.make_layout(((_n0, _n1), values_per_copy), ((_s0, _s1), n_mma))
        tile_mn = fx.make_tile(
            fx.make_layout(n_mma, 1),
            fx.make_layout(tile_k_per_wg // packed_per_dword, 1),
        )
        b_tiled_thr = fx.make_tiled_copy(b_cp_atom_r, tv_u32, tile_mn).get_slice(tid)
        b_tensor_thr = b_tiled_thr.partition_S(b_tile)
        b_frag_retile = [
            fx.make_fragment_like(b_tensor_thr[None, None, None, 0], fx.Uint32),
            fx.make_fragment_like(b_tensor_thr[None, None, None, 0], fx.Uint32),
        ]
        return b_cp_atom_r, b_tensor_thr, b_frag, b_frag_retile

    @flyc.jit
    def gemm_splitk(
        TILE_M,
        TILE_N,
        TILE_K,
        blk_n: int,  # block index for N dimension
        arg_p_input: fx.Tensor,  # [M, K] or [M, TOPK, K]
        arg_p_weight: fx.Tensor,  # [(16,N/16), (8, K/8)]
        lds,
        splitk_waves=4,
        a_with_index=True,
        p_w_scale=None,
        expert_id=None,
        gateup_contiguous_n=None,
    ):
        tid = gpu.thread_idx.x

        tile_k_per_wave = 128 if const_expr(weight_dtype == fx.Float4E2M1FN) else TILE_K
        tile_k_per_wg = tile_k_per_wave * splitk_waves
        fp4_jit_kmap = (
            weight_dtype == fx.Float4E2M1FN
            and splitk_waves == 4
            and gateup_contiguous_n is not None
        )

        a_tensor = fx.rocdl.make_buffer_tensor(arg_p_input, max_size=False)
        a_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), arg_p_input.dtype)

        # tiled copy is created based on the tiled_mma, so the tiled_mma should be same size for tiled copy
        if const_expr(weight_dtype == fx.Float4E2M1FN):
            k_perm = fx.make_tile(
                None,
                None,
                fx.make_layout((8, 4 * splitk_waves, 4), (1, 32, 8)),
            )
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        else:
            rep_k_per_lane = 4 if const_expr(weight_dtype != fx.BFloat16) else 2
            k_perm = fx.make_tile(
                None,
                None,
                fx.make_layout(
                    (4, 4 * splitk_waves, rep_k_per_lane),
                    (1, 4 * rep_k_per_lane, 4),
                ),
            )
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            # splitk for gateup/down
            fx.make_layout((1, 1, splitk_waves), (0, 0, 1)),
            k_perm,
        )
        if const_expr(a_with_index):
            cp_atom_lds = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
            tiled_copy_sortid_lds = fx.make_tiled_copy(
                cp_atom_lds,
                fx.make_layout(((16, 4 * splitk_waves), 1), ((1, 0), 0)),
                fx.make_tile(16),
            )
            a_index_frag = _read_sorted_index(
                tiled_copy_sortid_lds, tid, lds.sorted_lds
            )
            a_tensor_thr = TensorWithIndex(
                a_tensor,
                TILE_M,
                tile_k_per_wg,
                a_index_frag,
                fx.make_tiled_copy_A(a_cp_atom_r, tiled_mma),
                tid,
            )
            a_fake_tensor = fx.make_view(
                fx.get_iter(arg_p_input),
                fx.make_layout((TILE_M, tile_k_per_wg), (tile_k_per_wg, 1)),
            )
            a_frag = [
                tiled_mma.make_fragment_A(a_fake_tensor),
                tiled_mma.make_fragment_A(a_fake_tensor),
            ]
        else:
            a_tile = fx.flat_divide(a_tensor, fx.make_tile(TILE_M, tile_k_per_wg))[
                None, None, 0, None
            ]
            a_tiled_thr = fx.make_tiled_copy_A(a_cp_atom_r, tiled_mma).get_slice(tid)
            a_tensor_thr = a_tiled_thr.partition_S(a_tile)
            a_frag = [
                tiled_mma.make_fragment_A(a_tile[None, None, 0]),
                tiled_mma.make_fragment_A(a_tile[None, None, 0]),
            ]

        a_frag_retile = [
            fx.make_tiled_copy_A(a_cp_atom_r, tiled_mma)
            .get_slice(tid)
            .retile(a_frag[0]),
            fx.make_tiled_copy_A(a_cp_atom_r, tiled_mma)
            .get_slice(tid)
            .retile(a_frag[1]),
        ]

        b_cp_atom_r, b_tensor_thr, b_frag, b_frag_retile = _setup_b_operand(
            arg_p_weight, arg_p_input, tiled_mma, blk_n, TILE_N, tile_k_per_wg, tid
        )

        c_fake_tensor = fx.make_view(
            fx.get_iter(arg_p_input), fx.make_layout((TILE_N, TILE_M), (TILE_M, 1))
        )
        c_frag = tiled_mma.make_fragment_C(c_fake_tensor)
        c_frag.fill(0)

        num_k_iters = K // tile_k_per_wg

        def _a_k_offset(k_idx):
            if const_expr(fp4_jit_kmap):
                wave_id = tid // 64
                return wave_id * (K // splitk_waves - tile_k_per_wave) + k_idx * (
                    tile_k_per_wave - tile_k_per_wg
                )
            return 0

        def _b_k_offset(k_idx):
            if const_expr(fp4_jit_kmap):
                wave_id = tid // 64
                return wave_id * (K // 2 - 2 * tile_k_per_wave) + k_idx * (
                    2 * tile_k_per_wave - 2 * tile_k_per_wg
                )
            return 0

        def _gemm_stage(buf, k_idx, packed_scale_rows=None):
            if const_expr(weight_dtype == fx.Float4E2M1FN):
                _cvt_mxfp4_bf16(
                    b_frag_retile[buf],
                    b_frag[buf],
                    p_w_scale,
                    expert_id,
                    blk_n,
                    k_idx,
                    tile_k_per_wg,
                    tid,
                    gateup_contiguous_n,
                    packed_scale_rows,
                )
            else:
                if const_expr(weight_dtype != fx.BFloat16):
                    _cvt_fp8_bf16(b_frag_retile[buf], b_frag[buf])
            fx.gemm(tiled_mma, c_frag, b_frag[buf], a_frag[buf], c_frag)

        def _prefetch_a(k_idx, buf):
            extra_offset = _a_k_offset(k_idx)
            if const_expr(a_with_index):
                a_tensor_thr.copy(
                    a_cp_atom_r,
                    k_idx,
                    a_frag_retile[buf],
                    extra_offset=extra_offset,
                )
            else:
                src = a_tensor_thr[None, None, None, k_idx]
                if const_expr(fp4_jit_kmap):
                    src = fx.make_view(
                        fx.add_offset(fx.get_iter(src), extra_offset),
                        fx.get_layout(src),
                    )
                fx.copy(
                    a_cp_atom_r,
                    src,
                    a_frag_retile[buf],
                )

        def _prefetch_b(k_idx, buf):
            src = b_tensor_thr[None, None, None, k_idx]
            if const_expr(fp4_jit_kmap):
                src = fx.make_view(
                    fx.add_offset(fx.get_iter(src), _b_k_offset(k_idx)),
                    fx.get_layout(src),
                )
            fx.copy(b_cp_atom_r, src, b_frag_retile[buf])

        down_pair_scales = weight_dtype == fx.Float4E2M1FN and tile_k_per_wg == 128
        if const_expr(down_pair_scales):
            _prefetch_b(fx.Int32(0), 0)
            _prefetch_a(fx.Int32(0), 0)
            rocdl.sched_barrier(0)

        down_packed_scale_pairs = None
        if const_expr(down_pair_scales):
            down_packed_scale_pairs = []
            for pair_idx in range_constexpr((num_k_iters + 1) // 2):
                down_packed_scale_pairs.append(
                    _load_mxfp4_packed_scales(
                        b_frag_retile[0],
                        p_w_scale,
                        expert_id,
                        blk_n,
                        fx.Int32(pair_idx * 2),
                        tile_k_per_wg,
                        tid,
                        gateup_contiguous_n,
                    )
                )

        gate_pair_scales = fp4_jit_kmap and BLOCK_TILE_SIZE_N == 32 and K % 1024 == 0
        if const_expr(gate_pair_scales):
            _prefetch_b(fx.Int32(0), 0)
            _prefetch_a(fx.Int32(0), 0)

        gate_packed_scale_rows = None
        if const_expr(gate_pair_scales):
            gate_packed_scale_rows = _load_mxfp4_packed_scales(
                b_frag_retile[0],
                p_w_scale,
                expert_id,
                blk_n,
                fx.Int32(0),
                tile_k_per_wg,
                tid,
                gateup_contiguous_n,
            )

        if const_expr(not down_pair_scales and not gate_pair_scales):
            if const_expr(weight_dtype == fx.Float4E2M1FN):
                _prefetch_b(fx.Int32(0), 0)
                _prefetch_a(fx.Int32(0), 0)
            else:
                _prefetch_a(fx.Int32(0), 0)
                _prefetch_b(fx.Int32(0), 0)

        acc_init = c_frag.load()

        # Instruction counts for scheduling
        # 128-bit buffer_loads per prefetch: A loads + B loads
        a_load_bytes = arg_p_input.dtype.width // 8
        a_vmem_cnt = a_frag_retile[0].load().numel * a_load_bytes // 16
        if const_expr(weight_dtype == fx.BFloat16):
            b_vmem_cnt = b_frag_retile[0].load().numel * weight_dtype.width // 8 // 16
        else:
            b_vmem_cnt = b_frag_retile[0].load().numel * 4 // 16
        vmcnt_per_prefetch = a_vmem_cnt + b_vmem_cnt
        gate_scale_vmem_cnt = b_frag_retile[0].load().numel // 4

        rocdl.sched_barrier(0)

        if const_expr(weight_dtype == fx.Float4E2M1FN):
            c_frag.store(acc_init)
            if const_expr(fp4_jit_kmap and K % 1024 == 0):
                for pair_idx in range_constexpr(num_k_iters // 2):
                    even_idx = fx.Int32(pair_idx * 2)
                    odd_idx = even_idx + 1
                    if const_expr(BLOCK_TILE_SIZE_N == 32):
                        packed_scale_rows = gate_packed_scale_rows
                    else:
                        packed_scale_rows = _load_mxfp4_packed_scales(
                            b_frag_retile[0],
                            p_w_scale,
                            expert_id,
                            blk_n,
                            even_idx,
                            tile_k_per_wg,
                            tid,
                            gateup_contiguous_n,
                        )
                    _prefetch_b(odd_idx, 1)
                    _prefetch_a(odd_idx, 1)
                    rocdl.s_waitcnt(_encode_waitcnt(vmcnt=vmcnt_per_prefetch))
                    rocdl.sched_barrier(0)
                    rocdl.s_setprio(1)
                    _gemm_stage(0, even_idx, packed_scale_rows)
                    rocdl.s_setprio(0)
                    rocdl.sched_barrier(0)
                    if const_expr(pair_idx + 1 < num_k_iters // 2):
                        next_even_idx = even_idx + 2
                        if const_expr(BLOCK_TILE_SIZE_N == 32):
                            next_gate_packed_scale_rows = _load_mxfp4_packed_scales(
                                b_frag_retile[0],
                                p_w_scale,
                                expert_id,
                                blk_n,
                                next_even_idx,
                                tile_k_per_wg,
                                tid,
                                gateup_contiguous_n,
                            )
                        _prefetch_b(next_even_idx, 0)
                        _prefetch_a(next_even_idx, 0)
                        rocdl.s_waitcnt(
                            _encode_waitcnt(
                                vmcnt=(
                                    vmcnt_per_prefetch + gate_scale_vmem_cnt
                                    if BLOCK_TILE_SIZE_N == 32
                                    else vmcnt_per_prefetch
                                )
                            )
                        )
                    else:
                        rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
                    rocdl.sched_barrier(0)
                    rocdl.s_setprio(1)
                    _gemm_stage(1, odd_idx, packed_scale_rows)
                    rocdl.s_setprio(0)
                    rocdl.sched_barrier(0)
                    if const_expr(
                        BLOCK_TILE_SIZE_N == 32 and pair_idx + 1 < num_k_iters // 2
                    ):
                        gate_packed_scale_rows = next_gate_packed_scale_rows
            elif const_expr(fp4_jit_kmap):
                for k_idx in range_constexpr(num_k_iters):
                    read_buf = k_idx & 1
                    packed_scale_rows = _load_mxfp4_packed_scales(
                        b_frag_retile[read_buf],
                        p_w_scale,
                        expert_id,
                        blk_n,
                        fx.Int32(k_idx),
                        tile_k_per_wg,
                        tid,
                        gateup_contiguous_n,
                    )
                    if const_expr(k_idx + 1 < num_k_iters):
                        write_buf = read_buf ^ 1
                        next_idx = fx.Int32(k_idx + 1)
                        _prefetch_b(next_idx, write_buf)
                        _prefetch_a(next_idx, write_buf)
                        rocdl.s_waitcnt(_encode_waitcnt(vmcnt=vmcnt_per_prefetch))
                    else:
                        rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
                    rocdl.sched_barrier(0)
                    rocdl.s_setprio(1)
                    _gemm_stage(read_buf, fx.Int32(k_idx), packed_scale_rows)
                    rocdl.s_setprio(0)
                    rocdl.sched_barrier(0)
            elif const_expr(tile_k_per_wg == 128):
                for pair_idx in range_constexpr((num_k_iters + 1) // 2):
                    even_idx = fx.Int32(pair_idx * 2)
                    odd_idx = even_idx + 1
                    packed_scale_rows = down_packed_scale_pairs[pair_idx]
                    if const_expr(pair_idx * 2 + 1 < num_k_iters):
                        _prefetch_b(odd_idx, 1)
                        _prefetch_a(odd_idx, 1)
                        rocdl.s_waitcnt(_encode_waitcnt(vmcnt=vmcnt_per_prefetch))
                    else:
                        rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
                    rocdl.sched_barrier(0)
                    rocdl.s_setprio(1)
                    _gemm_stage(0, even_idx, packed_scale_rows)
                    rocdl.s_setprio(0)
                    rocdl.sched_barrier(0)
                    if const_expr(pair_idx * 2 + 1 < num_k_iters):
                        if const_expr(pair_idx * 2 + 2 < num_k_iters):
                            next_even_idx = even_idx + 2
                            _prefetch_b(next_even_idx, 0)
                            _prefetch_a(next_even_idx, 0)
                            rocdl.s_waitcnt(_encode_waitcnt(vmcnt=vmcnt_per_prefetch))
                        else:
                            rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
                        rocdl.sched_barrier(0)
                        rocdl.s_setprio(1)
                        _gemm_stage(1, odd_idx, packed_scale_rows)
                        rocdl.s_setprio(0)
                        rocdl.sched_barrier(0)
            else:
                for k_idx in range_constexpr(num_k_iters):
                    read_buf = k_idx & 1
                    if const_expr(k_idx + 1 < num_k_iters):
                        write_buf = read_buf ^ 1
                        next_idx = fx.Int32(k_idx + 1)
                        _prefetch_a(next_idx, write_buf)
                        _prefetch_b(next_idx, write_buf)
                        rocdl.s_waitcnt(_encode_waitcnt(vmcnt=vmcnt_per_prefetch))
                    else:
                        rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
                    rocdl.sched_barrier(0)
                    _gemm_stage(read_buf, fx.Int32(k_idx))
                    rocdl.sched_barrier(0)
        else:
            for k2, state in range(0, num_k_iters // 2, 1, init=[acc_init]):
                c_frag.store(state[0])
                k_base = fx.Int32(k2 * 2)
                _prefetch_a(k_base + 1, 1)
                _prefetch_b(k_base + 1, 1)
                rocdl.s_waitcnt(_encode_waitcnt(vmcnt=vmcnt_per_prefetch))
                rocdl.sched_barrier(0)
                _gemm_stage(0, k_base)
                rocdl.sched_barrier(0)
                _prefetch_a(k_base + 2, 0)
                _prefetch_b(k_base + 2, 0)
                rocdl.s_waitcnt(_encode_waitcnt(vmcnt=vmcnt_per_prefetch))
                rocdl.sched_barrier(0)
                _gemm_stage(1, k_base + 1)
                rocdl.sched_barrier(0)
                results = yield [c_frag.load()]
            c_frag.store(results)
            if const_expr(num_k_iters % 2 == 1):
                _gemm_stage(0, fx.Int32(num_k_iters - 1))

        # [v, n, m] -> [v, m, n]
        c_frag = _select(c_frag, [0, 2, 1])

        if const_expr(splitk_waves == 1):
            return c_frag

        if const_expr(TILE_N == 32):
            c_lds = fx.make_view(
                lds.c_reduce_lds.ptr, fx.make_ordered_layout((16 * 4, 32), order=(1, 0))
            )
            cp_atom_lds_w = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
            c_tiled_lds_w = fx.make_tiled_copy(
                cp_atom_lds_w,
                # (4wave*16)*4
                fx.make_layout(((16, 4, 4), (4, 2)), ((1, 256, 16), (64, 1024))),
                fx.make_tile(16 * 4, 16 * 2),
            )
            c_tensor_thr_lds_w = c_tiled_lds_w.get_slice(tid).partition_D(c_lds)
        else:
            # Reduce across 4 waves. To save lds size, will reuse (16*4)x64 floats for one loop
            swz = fx.SwizzleType.get(3, 3, 3)
            c_lds = fx.make_view(
                lds.c_reduce_lds.ptr,
                fx.make_composed_layout(
                    fx.static(swz), fx.make_ordered_layout((16 * 4, 64), order=(1, 0))
                ),
            )
            cp_atom_lds_w = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
            c_tiled_lds_w = fx.make_tiled_copy(
                cp_atom_lds_w,
                # (4wave*16)*4
                fx.make_layout(((16, 4, 4), (4, 4)), ((1, 256, 16), (64, 1024))),
                fx.make_tile(16 * 4, 16 * 4),
            )
            c_tensor_thr_lds_w = c_tiled_lds_w.get_slice(tid).partition_D(c_lds)

        if const_expr(TILE_N == 32):
            cp_atom_lds_r = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
            c_tiled_lds_r = fx.make_tiled_copy(
                cp_atom_lds_r,
                # thread mapping: (4wavex4)x16, repeat 4 times in m dimension for reduce
                fx.make_layout(((16, 4, 4), (1, 4)), ((32 * 2, 1, 4), (32, 16))),
                fx.make_tile(16 * 4, 16 * 1),
            )
            tile_sub_n = 16
        elif const_expr(TILE_N == 64):
            cp_atom_lds_r = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Float32)
            c_tiled_lds_r = fx.make_tiled_copy(
                cp_atom_lds_r,
                # thread mapping: (4wavex4)x16, repeat 4 times in m dimension for reduce
                fx.make_layout(((16, 4, 4), (2, 4)), ((64 * 2, 1, 4), (64, 16))),
                fx.make_tile(16 * 4, 16 * 2),
            )
            tile_sub_n = 32
        else:
            cp_atom_lds_r = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
            c_tiled_lds_r = fx.make_tiled_copy(
                cp_atom_lds_r,
                # thread mapping: (4wavex4)x16, repeat 4 times in m dimension for reduce
                fx.make_layout(((16, 4, 4), (4, 4)), ((256, 1, 4), (64, 16))),
                fx.make_tile(16 * 4, 16 * 4),
            )
            tile_sub_n = 64
        c_tensor_thr_lds_r = c_tiled_lds_r.get_slice(tid).partition_S(c_lds)

        # shape: [(4, 1), rep_m, rep_n]
        c_frag_vec = c_frag.load()
        # shape: [v, rm, rn]
        shape_v = fx.size(fx.get_shape(c_tensor_thr_lds_r)[0][0]).to_py_value()
        read_rep_n = fx.size(fx.get_shape(c_tensor_thr_lds_r)[2]).to_py_value()
        if const_expr(shape_v == 1):
            # TILE_N==32: flat layout (0-stride value mode) to avoid two stride-1 leaves
            c_frag_reduce = fx.make_rmem_tensor(
                fx.make_layout((1, TILE_M // 16, read_rep_n), (0, read_rep_n, 1)),
                fx.Float32,
            )
        else:
            stride_v = 1
            stride_sub_rn = shape_v * stride_v
            stride_rn = stride_sub_rn * (64 // tile_sub_n)
            stride_rm = stride_rn * TILE_N // tile_sub_n
            c_frag_reduce = fx.make_rmem_tensor(
                fx.make_layout(
                    (shape_v, TILE_M // (4 * 4), (64 // tile_sub_n, TILE_N // 64)),
                    (stride_v, stride_rm, (stride_sub_rn, stride_rn)),
                ),
                fx.Float32,
            )
        n_blocks = max(1, TILE_N // 64)
        w_size = fx.size(fx.get_shape(c_tensor_thr_lds_w)).to_py_value()
        for m in range_constexpr(TILE_M // 16):
            for n in range_constexpr(n_blocks):
                items = []
                for i in range_constexpr(w_size):
                    n_idx = n * (w_size // 4) + i // 4
                    idx = fx.get_scalar(fx.crd2idx((i % 4, m, n_idx), c_frag.layout))
                    items.append(c_frag_vec[idx])
                sub_c_frag = fx.make_fragment_like(c_tensor_thr_lds_w)
                sub_c_frag.store(Vec.from_elements(items, fx.Float32))
                fx.copy(cp_atom_lds_w, sub_c_frag, c_tensor_thr_lds_w)
                gpu.barrier()

                sub_c_frag_reduce = fx.make_fragment_like(c_tensor_thr_lds_r)
                fx.copy(cp_atom_lds_r, c_tensor_thr_lds_r, sub_c_frag_reduce)
                acc = sub_c_frag_reduce[(None, 0), None, None].load()
                for i in range_constexpr(1, 4):
                    acc += sub_c_frag_reduce[(None, i), None, None].load()

                if const_expr(shape_v == 1):
                    c_frag_reduce[0, m, None].store(acc)
                else:
                    c_frag_reduce[None, m, (None, n)].store(acc)
                if const_expr(m * n_blocks + n + 1 < (TILE_M // 16) * n_blocks):
                    gpu.barrier()

        return c_frag_reduce

    def _make_1x4_tiled_mma():
        # Shared B-first (mma_M=channel, mma_N=token) 1x4 tiled_mma, used by both _gemm_1x4
        # and the CShuffle epilogue's make_tiled_copy_C. One definition keeps the two sites
        # from drifting apart. bf16 = MFMA(16,16,16); native fp8 = MFMA(16,16,32).
        if weight_dtype == fx.BFloat16:
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
            k_perm = fx.make_layout((4, 4, 2), (1, 8, 4))
        else:
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, weight_dtype))
            k_perm = fx.make_layout((8, 4, 2), (1, 16, 8))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((4, 1, 1), (1, 0, 0)),
            fx.make_tile(None, None, k_perm),
        )
        return mma_atom, tiled_mma

    @flyc.jit
    def gemm_1x4(
        TILE_M,
        TILE_N,
        TILE_K,
        blk_n: int,  # block index for N dimension (in units of TILE_N)
        arg_p_input: fx.Tensor,  # [M, K]; A rows are gathered via lds.sorted_lds
        arg_p_weight: fx.Tensor,  # preshuffle layout with gate/up grouping composed
        lds,  # SharedStorage with sorted_lds, a_ping, a_pong
    ):
        """1x4 tiled GEMM: the 4 waves tile N(channel); the full TILE_M is shared across
        all waves (no M-split). Each wave owns contiguous_n//4 output channels of BOTH the
        gate and the up projection (two C fragments) so activation stays wave-internal. A
        (activation) is gathered via sorted_lds and staged through an LDS ping-pong
        (a_ping/a_pong); B (weight gate/up) loads direct global->register (no LDS). Pipeline
        mirrors preshuffle_gemm_v2 (A 2-stage LDS ping-pong). B-first MFMA (weight is the
        MFMA M-side) so each C fragment's value dim runs along channel (4 contiguous
        channels/lane), letting the epilogue store 64-bit instead of the A-first 16-bit.
        Convention inside this function: m = channel (mma_M), n = token (mma_N)."""
        tid = gpu.thread_idx.x
        contiguous_n = TILE_N // 2

        a_tensor = fx.rocdl.make_buffer_tensor(arg_p_input, max_size=False)
        b_tensor = fx.rocdl.make_buffer_tensor(arg_p_weight, max_size=False)

        # tiled_mma: B-first (mma_M=channel from weight, mma_N=token from activation); the 4
        # waves tile M(channel) so each wave still owns contiguous_n//4 output channels.
        mma_atom, tiled_mma = _make_1x4_tiled_mma()

        # ---- A (activation): gather + LDS ping-pong ----
        # Static (TILE_M, K) fake keeps flat_divide static; real rows gathered below.
        a_size_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(arg_p_input), fx.make_layout((TILE_M, K), (K, 1))),
            max_size=False,
        )
        a_tile = fx.flat_divide(a_size_buf, fx.make_tile(TILE_M, TILE_K))[
            None, None, 0, None
        ]
        buf_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), weight_dtype)
        _val_per_thr = 8 if const_expr(weight_dtype == fx.BFloat16) else 16
        _thrs_k = TILE_K // _val_per_thr
        _thrs_m = 256 // _thrs_k
        g2r_tv_layout = fx.make_layout(
            ((_thrs_k, _thrs_m), (1, _val_per_thr)),
            ((_thrs_m * _val_per_thr, 1), (1, _thrs_m)),
        )
        a_mem_cp_g2r = fx.make_tiled_copy(
            buf_cp_atom_r, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K)
        )
        # index copy for A gather: M-row mapping matches g2r M-tile (_thrs_m).
        _m_per_wave = _thrs_m // 4
        cp_atom_sortid_a = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
        tiled_copy_sortid_a = fx.make_tiled_copy(
            cp_atom_sortid_a,
            fx.make_layout(((_thrs_k, _m_per_wave, 4), 1), ((0, 1, _m_per_wave), 0)),
            fx.make_tile(_thrs_m),
        )
        a_index_frag = _read_sorted_index(
            tiled_copy_sortid_a, tid, lds.sorted_lds, index_size=TILE_M
        )
        a_idx = TensorWithIndex(
            a_tensor,
            TILE_M,
            TILE_K,
            a_index_frag,
            a_mem_cp_g2r,
            tid,
        )
        a_mem_thr = a_mem_cp_g2r.get_slice(tid).partition_S(a_tile)
        a_cp_frag = fx.make_fragment_like(a_mem_thr[None, None, None, 0])

        # sorted_lds is unioned with a_ping: seed all index_frag reads (caller's c_out index
        # + a_idx above) before overwriting that LDS region with the A tile below.
        gpu.barrier()

        if const_expr(weight_dtype == fx.BFloat16):
            swz = fx.SwizzleType.get(3, 3, 3)
        else:
            swz = fx.SwizzleType.get(3, 4, 3)
        a_ping = fx.make_view(
            lds.gemm.a_ping.ptr,
            fx.make_composed_layout(
                fx.static(swz), fx.make_ordered_layout((TILE_M, TILE_K), order=(1, 0))
            ),
        )
        a_pong = fx.make_view(
            lds.gemm.a_pong.ptr,
            fx.make_composed_layout(
                fx.static(swz), fx.make_ordered_layout((TILE_M, TILE_K), order=(1, 0))
            ),
        )

        # One 128-bit universal copy shared by A r2s (ds_write_b128) and A LDS read (ds_read_b128).
        uni_cp_atom = fx.make_copy_atom(fx.UniversalCopy128b(), weight_dtype)
        a_r2s = fx.make_tiled_copy(
            uni_cp_atom, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K)
        )
        a_lds_w = [
            a_r2s.get_slice(tid).partition_D(a_ping),
            a_r2s.get_slice(tid).partition_D(a_pong),
        ]
        a_cp_frag_retile = a_r2s.get_slice(tid).retile(a_cp_frag)
        # B-first: activation is the MFMA B-operand (make_fragment_B / make_tiled_copy_B).
        a_lds_r = [
            fx.make_tiled_copy_B(uni_cp_atom, tiled_mma)
            .get_slice(tid)
            .partition_S(a_ping),
            fx.make_tiled_copy_B(uni_cp_atom, tiled_mma)
            .get_slice(tid)
            .partition_S(a_pong),
        ]
        a_frag = tiled_mma.make_fragment_B(a_ping)
        a_frag_retile = (
            fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).retile(a_frag)
        )

        # ---- B (weight gate/up): direct global->register (no LDS), 2-stage double buffer ----
        # B-first: weight is the MFMA A-operand (make_fragment_A / make_tiled_copy_A).
        bl_tile = fx.flat_divide(b_tensor, fx.make_tile(contiguous_n, TILE_K))[
            None, None, blk_n * 2 + 0, None
        ]
        br_tile = fx.flat_divide(b_tensor, fx.make_tile(contiguous_n, TILE_K))[
            None, None, blk_n * 2 + 1, None
        ]
        b_g2r = fx.make_tiled_copy_A(buf_cp_atom_r, tiled_mma).get_slice(tid)
        bl_g2r = b_g2r.partition_S(bl_tile)
        br_g2r = b_g2r.partition_S(br_tile)
        bl_frag_st = [
            tiled_mma.make_fragment_A(bl_tile[None, None, 0]),
            tiled_mma.make_fragment_A(bl_tile[None, None, 0]),
        ]
        br_frag_st = [
            tiled_mma.make_fragment_A(br_tile[None, None, 0]),
            tiled_mma.make_fragment_A(br_tile[None, None, 0]),
        ]
        bl_ret_st = [b_g2r.retile(bl_frag_st[0]), b_g2r.retile(bl_frag_st[1])]
        br_ret_st = [b_g2r.retile(br_frag_st[0]), b_g2r.retile(br_frag_st[1])]

        # ---- C fragments (gate + up), one make_fragment_C each ----
        # B-first: make_fragment_C over the (channel, token) tile; the value dim then runs
        # along channel (4 contiguous channels/lane) for a 64-bit epilogue store.
        c_fake_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.get_iter(arg_p_input),
                fx.make_layout((contiguous_n, TILE_M), (TILE_M, 1)),
            ),
            max_size=False,
        )
        c_fake = fx.flat_divide(c_fake_buf, fx.make_tile(contiguous_n, TILE_M))[
            None, None, 0, 0
        ]
        c_gate = tiled_mma.make_fragment_C(c_fake)
        c_up = tiled_mma.make_fragment_C(c_fake)
        c_gate.fill(0)
        c_up.fill(0)

        num_tiles = K // TILE_K

        # ---- instruction-scheduling hints ----
        # 128-bit loads / ds ops per stage; MFMA count for the two gemms.
        k_per_mma = 16 if const_expr(weight_dtype == fx.BFloat16) else 32
        _m_reps = fx.size(fx.get_shape(c_gate)[1]).to_py_value()
        _n_reps = fx.size(fx.get_shape(c_gate)[2]).to_py_value()
        mfma_per_gemm = _m_reps * _n_reps * (TILE_K // k_per_mma)
        mem_a_cnt = a_cp_frag.load().numel * weight_dtype.width // 8 // 16
        mem_b_cnt = bl_frag_st[0].load().numel * weight_dtype.width // 8 // 16
        # per-ki interleave: k_perm groups 2 MFMA-K atoms, so k_iters = TILE_K / (2*k_per_mma).
        # fragment K dim is (2 atoms, k_iters) -> gemm coord = (None, ki); the retile/LDS-read
        # views have a flat k_iters dim -> coord = ki. This is what lets TILE_K scale to 128.
        k_iters = TILE_K // (2 * k_per_mma)
        # full A(tile) LDS read (ds_read), done once per stage (cross-stage rotation)
        lds_a_cnt = a_frag.load().numel * weight_dtype.width // 8 // 16

        def hot_loop_scheduler():
            # Fixed interleave: each buffer_load(vmem)+4 mfma; each ds_read(dsrd)+1 mfma;
            # each ds_write(dswr)+2 mfma (dsrd before dswr); then the remaining mfma.
            mfma_cnt = 2 * mfma_per_gemm
            n_vmem = mem_a_cnt + 2 * mem_b_cnt  # A g2r + B gate/up g2r (buffer_load)
            n_dswr = mem_a_cnt  # A staging -> LDS store (ds_write)
            n_dsrd = lds_a_cnt  # A LDS -> register full tile (ds_read)
            used = 0
            rocdl.sched_dsrd(2)
            for _ in range_constexpr(n_vmem):
                rocdl.sched_dsrd(1)
                rocdl.sched_vmem(1)
                rocdl.sched_mfma(2)
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(2)
                used += 4
            for _ in range_constexpr(n_dsrd - 2 * n_vmem - 2):
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(1)
                used += 1
            if const_expr(mfma_cnt - n_dswr * 2 - used > 0):
                rocdl.sched_mfma(mfma_cnt - n_dswr * 2 - used)
            for _ in range_constexpr(n_dswr):
                rocdl.sched_dswr(1)
                rocdl.sched_mfma(2)
                used += 2
            if const_expr(mfma_cnt - used > 0):
                rocdl.sched_mfma(mfma_cnt - used)

        def pipeline_stage(read_i, k_next, do_prefetch):
            write_i = read_i ^ 1
            # prefetch next B (gate/up) + A (global -> register)
            if const_expr(do_prefetch):
                a_idx.copy(buf_cp_atom_r, k_next, a_cp_frag)
                fx.copy(
                    buf_cp_atom_r, bl_g2r[None, None, None, k_next], bl_ret_st[write_i]
                )
                fx.copy(
                    buf_cp_atom_r, br_g2r[None, None, None, k_next], br_ret_st[write_i]
                )
            # read this stage's own A tile LDS[read_i] -> a_frag at the head, then compute
            for ki in range_constexpr(k_iters):
                fx.copy(
                    uni_cp_atom,
                    a_lds_r[read_i][None, None, ki],
                    a_frag_retile[None, None, ki],
                )
                for n in range_constexpr(_n_reps):
                    for m in range_constexpr(_m_reps):
                        for k in range_constexpr(2):
                            fx.mma_atom_call(
                                mma_atom,
                                c_gate[None, m, n],
                                bl_frag_st[read_i][None, m, (k, ki)],
                                a_frag[None, n, (k, ki)],
                                c_gate[None, m, n],
                            )
                            fx.mma_atom_call(
                                mma_atom,
                                c_up[None, m, n],
                                br_frag_st[read_i][None, m, (k, ki)],
                                a_frag[None, n, (k, ki)],
                                c_up[None, m, n],
                            )
            if const_expr(do_prefetch):
                # A(k_next) staging -> LDS[write] for a later stage's head read
                fx.copy(uni_cp_atom, a_cp_frag_retile, a_lds_w[write_i])
                hot_loop_scheduler()
            rocdl.sched_barrier(0)
            gpu.barrier()

        # Prologue: gather A(0) -> LDS[0]; load B(0) -> stage 0.
        a_idx.copy(buf_cp_atom_r, fx.Int32(0), a_cp_frag)
        fx.copy(buf_cp_atom_r, bl_g2r[None, None, None, fx.Int32(0)], bl_ret_st[0])
        fx.copy(buf_cp_atom_r, br_g2r[None, None, None, fx.Int32(0)], br_ret_st[0])
        rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
        fx.copy(uni_cp_atom, a_cp_frag_retile, a_lds_w[0])
        gpu.barrier()

        acc_init = [c_gate.load(), c_up.load()]
        for iv, state in range(0, num_tiles // 2 - 1, 1, init=acc_init):
            c_gate.store(state[0])
            c_up.store(state[1])
            kb = fx.Int32(iv * 2)
            pipeline_stage(0, kb + 1, True)
            pipeline_stage(1, kb + 2, True)
            results = yield [c_gate.load(), c_up.load()]
        c_gate.store(results[0])
        c_up.store(results[1])
        kb = fx.Int32(num_tiles - 2)
        pipeline_stage(0, kb + 1, True)
        pipeline_stage(1, fx.Int32(0), False)
        return c_gate, c_up

    def _apply_1x4_fp8_dequant(
        c_gate_frag,
        c_up_frag,
        tid,
        expert_id,
        blk_n,
        contiguous_n,
        asc_idx,
        M,
        p_w_scale,
        p_a_scale,
    ):
        # Native-fp8 dequant folded into c_gate/c_up IN PLACE, before activation. Caller
        # guards on weight_dtype (bf16 -> not called). B-first layout: value dim = 4 contiguous
        # channels, m_rep = channel_rep, n_rep = token_rep.
        #   act ptpc: a_scale is per token (one per token_rep n, shared by the 4 channel values,
        #     gathered via asc_idx). weight b_scale is per-output-channel (ptpc) or a per_tensor
        #     scalar.
        #   act per_tensor: b_scale * a_scale is a single scalar pre-multiply.
        if const_expr(act_quant_type == "ptpc"):
            m_reps = fx.size(fx.get_shape(c_gate_frag)[1]).to_py_value()
            n_reps = fx.size(fx.get_shape(c_gate_frag)[2]).to_py_value()
            if const_expr(weight_quant_type == "ptpc"):
                scale_gate = fx.make_view(
                    fxh._as_ptr(p_w_scale) + expert_id * N + blk_n * contiguous_n,
                    fx.make_layout(contiguous_n, 1),
                )
                scale_up = fx.make_view(
                    fxh._as_ptr(p_w_scale)
                    + expert_id * N
                    + N // 2
                    + blk_n * contiguous_n,
                    fx.make_layout(contiguous_n, 1),
                )
                cp_atom_scale = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
                # channel = v + 4*(lane//16) + 16*wave (+ 64*channel_rep): gather 4 per
                # value into [v, channel_rep] to match the C fragment channel layout.
                scale_copy = fx.make_tiled_copy(
                    cp_atom_scale,
                    fx.make_layout(((16, 4, 4), 4), ((0, 4, 16), 1)),
                    fx.make_tile(64),
                )
                sg_thr = scale_copy.get_slice(tid).partition_S(scale_gate)
                su_thr = scale_copy.get_slice(tid).partition_S(scale_up)
                gate_scale = fx.make_fragment_like(sg_thr)
                up_scale = fx.make_fragment_like(su_thr)
                fx.copy(cp_atom_scale, sg_thr, gate_scale)
                fx.copy(cp_atom_scale, su_thr, up_scale)
            else:
                b_scalar = fx.make_view(
                    fxh._as_ptr(p_w_scale) + expert_id, fx.make_layout(1, 1)
                )[0]

            a_scale_tensor = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)),
                    fx.make_layout(M, 1),
                ),
                max_size=False,
            )
            # a_scale is per token = per token_rep n (independent of channel_rep m), so
            # gather all n up front: removes the redundant gather across m and lets the
            # indexed loads issue together instead of serializing inside the inner loop.
            a_sc_n = [
                a_scale_tensor[asc_idx[0, n] & 0xFFFFFF]
                for n in range_constexpr(n_reps)
            ]
            for m in range_constexpr(m_reps):
                if const_expr(weight_quant_type == "ptpc"):
                    sg_v = gate_scale[None, m].load()
                    su_v = up_scale[None, m].load()
                for n in range_constexpr(n_reps):
                    a_sc = a_sc_n[n]
                    cg = c_gate_frag[None, m, n].load()
                    cu = c_up_frag[None, m, n].load()
                    cg_items = []
                    cu_items = []
                    for v in range_constexpr(4):
                        if const_expr(weight_quant_type == "ptpc"):
                            sg = sg_v[v]
                            su = su_v[v]
                        else:
                            sg = b_scalar
                            su = b_scalar
                        cg_items.append(cg[v] * sg * a_sc)
                        cu_items.append(cu[v] * su * a_sc)
                    c_gate_frag[None, m, n].store(
                        Vec.from_elements(cg_items, fx.Float32)
                    )
                    c_up_frag[None, m, n].store(Vec.from_elements(cu_items, fx.Float32))
        elif const_expr(act_quant_type == "per_tensor"):
            b_scale = fx.make_view(
                fxh._as_ptr(p_w_scale) + expert_id, fx.make_layout(1, 1)
            )[0]
            a_scale0 = fx.make_view(
                fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)), fx.make_layout(1, 1)
            )[0]
            scale = b_scale * a_scale0
            c_gate_frag.store(c_gate_frag.load() * scale)
            c_up_frag.store(c_up_frag.load() * scale)

    @flyc.kernel
    def moe_2stage_gateup_splitk(
        p_input: fx.Pointer,  # bf16 [M, K]
        p_weight: fx.Pointer,  # quantized/bf16 [N/16, K/8 * 16 * 8]
        p_output: fx.Pointer,  # bf16 [M, TOPK, N//2]
        # sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        M: fx.Int32,
        situ_beta_f: fx.Float32,
        situ_beta_rcp_f: fx.Float32,
        situ_linear_beta_f: fx.Float32,
        situ_linear_beta_rcp_f: fx.Float32,
        activation_limit_f: fx.Float32,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y

        arg_p_input = fx.make_view(fxh._as_ptr(p_input), fx.make_layout((M, K), (K, 1)))
        num_valid_buf = fx.make_view(
            fx.recast_iter(fx.Int32, fxh._as_ptr(p_num_valid_ids)), fx.make_layout(1, 1)
        )
        max_valid_id = num_valid_buf[0]
        if const_expr(weight_dtype != fx.BFloat16):
            p_weight = fx.recast_iter(fx.Uint8, fxh._as_ptr(p_weight))
        if e_idx * BLOCK_TILE_SIZE_M < max_valid_id:
            lds = fx.SharedAllocator().allocate(SharedStorage)
            lds.sorted_lds = lds.sorted_lds.peek()
            lds.c_reduce_lds = lds.c_reduce_lds.peek()
            arg_p_sorted_ids = fx.make_view(
                fx.recast_iter(
                    fx.Int32, fxh._as_ptr(p_sorted_ids) + e_idx * BLOCK_TILE_SIZE_M
                ),
                fx.make_layout(BLOCK_TILE_SIZE_M, 1),
            )
            arg_p_sorted_expert_ids = fx.recast_iter(
                fx.Int32, fxh._as_ptr(p_sorted_expert_ids)
            )
            expert_id = arg_p_sorted_expert_ids[e_idx]
            # there is a reduce in gemm_splitk which will read/write from lds, the BLOCK_TILE_SIZE_N will impact the coalesced access:
            # BLOCK_TILE_SIZE_N BLOCK_TILE_SIZE_N//2(after activation) LDS_read_per_lane  MEM_write_per_lane
            # 64                32                               2=(32/16 threads)  2=(32/16 threads)
            # 128               64                               4=(64/16 threads)  4=(64/16 threads)
            # 256: will split into 2x128
            contiguous_n = 64 if const_expr(BLOCK_TILE_SIZE_N % 128 == 0) else 32

            # NOTE: assume permuted adjacent 32 rows will fall in the same wave for activation
            arg_p_weight = _make_gateup_weight_view(p_weight, expert_id, contiguous_n)

            # sorted ids: global -> LDS (scalar load/store, only first BLOCK_TILE_SIZE_M threads participate)
            sorted_ids_buf = fx.rocdl.make_buffer_tensor(
                arg_p_sorted_ids, max_size=False
            )
            if tid < BLOCK_TILE_SIZE_M:
                lds_view = fx.make_view(
                    lds.sorted_lds.ptr, fx.make_layout(BLOCK_TILE_SIZE_M, 1)
                )
                # fx.memref_store(val, lds_view, tid)
                lds_view[tid] = sorted_ids_buf[tid]
            gpu.barrier()

            # prepare c_tensor(reuse lds.c_reduce_lds before gemm)
            cp_atom_w = fx.make_copy_atom(
                (
                    fx.rocdl.BufferCopy64b()
                    if const_expr(BLOCK_TILE_SIZE_N % 128 == 0)
                    else fx.rocdl.BufferCopy32b()
                ),
                fx.BFloat16,
            )
            c_tiled_g = fx.make_tiled_copy(
                cp_atom_w,
                # thread mapping: 4 wavex(4x16), (contiguous_n // 16) elements per lane
                fx.make_layout(
                    ((16, 4, 4), contiguous_n // 16), ((contiguous_n, 1, 4), 16)
                ),
                fx.make_tile(16, contiguous_n),
            )
            arg_p_output = fx.make_view(
                fxh._as_ptr(p_output),
                fx.make_layout((M, TOPK, N // 2), (TOPK * N // 2, N // 2, 1)),
            )
            out_tensor = fx.rocdl.make_buffer_tensor(
                arg_p_output,
                max_size=False,
                num_records_bytes=fx.Int64(M)
                * (TOPK * N // 2 * fx.BFloat16.width // 8),
            )
            tiled_copy_sortid_lds = fx.make_tiled_copy(
                fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
                fx.make_layout(((16, 16), 1), ((0, 1), 0)),
                fx.make_tile(16),
            )
            c_index_frag = _read_sorted_index(
                tiled_copy_sortid_lds, tid, lds.sorted_lds
            )
            c_tensor = TensorWithIndex(
                out_tensor,
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N // 2,
                c_index_frag,
                c_tiled_g,
                tid,
                is_read_from_mem=False,
            )

            c_frag = gemm_splitk(
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N,
                TILE_K,
                blk_n,
                arg_p_input,
                arg_p_weight,
                lds,
                splitk_waves=4,
                p_w_scale=p_w_scale,
                expert_id=expert_id,
                gateup_contiguous_n=contiguous_n,
            )

            c_frag_bf16 = _apply_scale_gateup_bf16(
                c_frag,
                tid,
                expert_id,
                blk_n,
                contiguous_n,
                p_w_scale,
                situ_beta_f,
                situ_beta_rcp_f,
                situ_linear_beta_f,
                situ_linear_beta_rcp_f,
                activation_limit_f,
            )

            c_tensor.copy(
                cp_atom_w, blk_n, c_tiled_g.get_slice(tid).retile(c_frag_bf16)
            )

    @flyc.kernel
    def moe_2stage_gateup_batch1(
        p_input: fx.Pointer,  # bf16 [M, K]
        p_weight: fx.Pointer,  # quantized/bf16 [N/16, K/8 * 16 * 8]
        p_output: fx.Pointer,  # bf16 [M, TOPK, N//2]
        p_topk_ids: fx.Pointer,  # int32 [M, TOPK]
        p_clear_output: fx.Pointer,
        p_w_scale: fx.Pointer,
        situ_beta_f: fx.Float32,
        situ_beta_rcp_f: fx.Float32,
        situ_linear_beta_f: fx.Float32,
        situ_linear_beta_rcp_f: fx.Float32,
        activation_limit_f: fx.Float32,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y
        batch_idx = gpu.block_idx.z
        route_idx = batch_idx * TOPK + e_idx

        arg_p_input = fx.make_view(
            fxh._as_ptr(p_input) + fx.Int64(batch_idx * K),
            fx.make_layout((1, K), (K, 1)),
        )
        if const_expr(weight_dtype != fx.BFloat16):
            p_weight = fx.recast_iter(fx.Uint8, fxh._as_ptr(p_weight))
        arg_p_expert_ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_topk_ids))
        expert_id = arg_p_expert_ids[route_idx]
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # gate/up group width. BN==32 keeps the 4-wave split-K but uses a dedicated reduce below;
        # BN>=64 uses the coalesced reduce in gemm_splitk.
        contiguous_n = min(64, BLOCK_TILE_SIZE_N // 2)

        # NOTE: assume permuted adjacent 32 rows will fall in the same wave to do silu
        arg_p_weight = _make_gateup_weight_view(p_weight, expert_id, contiguous_n)

        c_frag = gemm_splitk(
            BLOCK_TILE_SIZE_M,
            BLOCK_TILE_SIZE_N,
            TILE_K,
            blk_n,
            arg_p_input,
            arg_p_weight,
            lds,
            splitk_waves=4,
            a_with_index=False,
            p_w_scale=p_w_scale,
            expert_id=expert_id,
            gateup_contiguous_n=contiguous_n,
        )

        c_frag_bf16 = _apply_scale_gateup_bf16(
            c_frag,
            tid,
            expert_id,
            blk_n,
            contiguous_n,
            p_w_scale,
            situ_beta_f,
            situ_beta_rcp_f,
            situ_linear_beta_f,
            situ_linear_beta_rcp_f,
            activation_limit_f,
        )

        arg_p_output = fx.make_view(
            fxh._as_ptr(p_output) + fx.Int64(batch_idx * TOPK * (N // 2)),
            fx.make_layout((1, TOPK, N // 2), (TOPK * N // 2, N // 2, 1)),
        )
        out_tensor = fx.rocdl.make_buffer_tensor(
            arg_p_output,
            max_size=False,
            num_records_bytes=1 * TOPK * N // 2 * fx.BFloat16.width // 8,
        )
        cp_atom_w = fx.make_copy_atom(
            (
                fx.rocdl.BufferCopy64b()
                if const_expr(BLOCK_TILE_SIZE_N % 128 == 0)
                else (
                    fx.rocdl.BufferCopy32b()
                    if const_expr(BLOCK_TILE_SIZE_N >= 64)
                    else fx.rocdl.BufferCopy16b()
                )
            ),
            fx.BFloat16,
        )
        c_tiled_g = fx.make_tiled_copy(
            cp_atom_w,
            # thread mapping: 4 wavex(4x16), (contiguous_n // 16) elements per lane
            fx.make_layout(
                ((16, 4, 4), max(1, contiguous_n // 16)),
                ((max(1, contiguous_n), 1, 4), 16),
            ),
            fx.make_tile(16, max(16, contiguous_n)),
        )
        c_tile = fx.flat_divide(
            out_tensor[None, e_idx, None],
            fx.make_tile(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 2),
        )[None, None, None, blk_n]
        c_dst = c_tiled_g.get_slice(tid).partition_S(c_tile)
        c_src = c_tiled_g.get_slice(tid).retile(c_frag_bf16)

        fx.copy(cp_atom_w, c_src, c_dst[None, None, None, 0])

        if const_expr(fused_down_clear):
            clear_blocks = (N + BLOCK_TILE_SIZE_N - 1) // BLOCK_TILE_SIZE_N
            clear_threads = clear_blocks * 256
            clear_rounds = (K // 8 + clear_threads - 1) // clear_threads
            for clear_round in range_constexpr(clear_rounds):
                clear_idx = blk_n * 256 + tid + clear_round * clear_threads
                if e_idx == 0 and clear_idx < K // 8:
                    clear_output = fx.make_view(
                        fxh._as_ptr(p_clear_output)
                        + fx.Int64(batch_idx * K + clear_idx * 8),
                        fx.make_layout(8, 1),
                    )
                    clear_frag = fx.make_fragment_like(clear_output)
                    clear_frag.fill(0)
                    fx.copy(
                        fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16),
                        clear_frag,
                        clear_output,
                    )

    @flyc.kernel
    def moe_2stage_gateup_prefill_1x4(
        p_input: fx.Pointer,  # bf16 or native-fp8 [M, K]
        p_weight: fx.Pointer,  # bf16/fp8 [N/16, K/8 * 16 * 8]
        p_output: fx.Pointer,  # bf16 [M, TOPK, N//2]
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,  # weight fp8 scale (per-output-channel ptpc / per-tensor)
        p_a_scale: fx.Pointer,  # input fp8 scale (per-token ptpc / per-tensor)
        M: fx.Int32,
        situ_beta_f: fx.Float32,
        situ_beta_rcp_f: fx.Float32,
        situ_linear_beta_f: fx.Float32,
        situ_linear_beta_rcp_f: fx.Float32,
        activation_limit_f: fx.Float32,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y

        if const_expr(weight_dtype != fx.BFloat16):
            in_ptr = fx.recast_iter(weight_dtype, fxh._as_ptr(p_input))
        else:
            in_ptr = fxh._as_ptr(p_input)
        arg_p_input = fx.make_view(in_ptr, fx.make_layout((M, K), (K, 1)))
        num_valid_buf = fx.make_view(
            fx.recast_iter(fx.Int32, fxh._as_ptr(p_num_valid_ids)), fx.make_layout(1, 1)
        )
        max_valid_id = num_valid_buf[0]
        if const_expr(weight_dtype != fx.BFloat16):
            p_weight = fx.recast_iter(weight_dtype, fxh._as_ptr(p_weight))
        if e_idx * BLOCK_TILE_SIZE_M < max_valid_id:
            lds = fx.SharedAllocator().allocate(SharedStorage)
            lds.sorted_lds = lds.sorted_lds.peek()
            lds.gemm = lds.gemm.peek()
            arg_p_sorted_ids = fx.make_view(
                fx.recast_iter(
                    fx.Int32, fxh._as_ptr(p_sorted_ids) + e_idx * BLOCK_TILE_SIZE_M
                ),
                fx.make_layout(BLOCK_TILE_SIZE_M, 1),
            )
            arg_p_sorted_expert_ids = fx.recast_iter(
                fx.Int32, fxh._as_ptr(p_sorted_expert_ids)
            )
            expert_id = arg_p_sorted_expert_ids[e_idx // gateup_tasks_per_metadata]

            contiguous_n = BLOCK_TILE_SIZE_N // 2
            arg_p_weight = _make_gateup_weight_view(p_weight, expert_id, contiguous_n)

            sorted_ids_buf = fx.rocdl.make_buffer_tensor(
                arg_p_sorted_ids, max_size=False
            )
            if tid < BLOCK_TILE_SIZE_M:
                lds_view = fx.make_view(
                    lds.sorted_lds.ptr, fx.make_layout(BLOCK_TILE_SIZE_M, 1)
                )
                lds_view[tid] = sorted_ids_buf[tid]
            gpu.barrier()

            # Output [M, TOPK, N//2] + the per-row scatter index, built BEFORE gemm_1x4:
            # sorted_lds is unioned with a_ping, so c_out must seed its index_frag from
            # sorted_lds now; gemm_1x4 then overwrites that LDS region with the A tile.
            arg_p_output = fx.make_view(
                fxh._as_ptr(p_output),
                fx.make_layout((M, TOPK, N // 2), (TOPK * N // 2, N // 2, 1)),
            )
            out_tensor = fx.rocdl.make_buffer_tensor(
                arg_p_output,
                max_size=False,
                num_records_bytes=fx.Int64(M)
                * (TOPK * N // 2 * fx.BFloat16.width // 8),
            )
            buf_atom_w128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
            # CShuffle read/scatter over the single (BM x contiguous_n) region. The read uses a
            # 4-wave 2x2 thread grid (token = lane//4 + 16*waveM, 8 contiguous channels/lane,
            # 128b), decoupled from the gemm's 1x4 wave layout (it just walks the staged LDS),
            # so rep_token = BM//32 and rep_channel = contiguous_n//64.
            c_rw_copy = fx.make_tiled_copy(
                buf_atom_w128,
                fx.make_layout(((4, 16, 2, 2), 8), ((256, 1, 16, 1024), 32)),
                fx.make_tile(32, 64),
            )
            c_index_copy = fx.make_tiled_copy(
                fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
                fx.make_layout(((4, 16, 2, 2), 1), ((0, 1, 16, 0), 0)),
                fx.make_tile(32),
            )
            c_out_index_frag = _read_sorted_index(c_index_copy, tid, lds.sorted_lds)
            c_out = TensorWithIndex(
                out_tensor,
                BLOCK_TILE_SIZE_M,
                contiguous_n,
                c_out_index_frag,
                c_rw_copy,
                tid,
                is_read_from_mem=False,
                TOPK=TOPK,
            )

            # ptpc a_scale is per-token; B-first packs 4 CONTIGUOUS channels per lane in the
            # value dim, so token = lane%16 + 16*token_rep (one id per token_rep, shared by
            # the 4 channel values). Gather the per-token_rep sorted id here, before gemm_1x4
            # overwrites sorted_lds.
            asc_idx = None
            if const_expr(weight_dtype != fx.BFloat16 and act_quant_type == "ptpc"):
                asc_index_copy = fx.make_tiled_copy(
                    fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
                    fx.make_layout(((16, 4, 4), 1), ((1, 0, 0), 0)),
                    fx.make_tile(16),
                )
                cp_atom_idx = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
                asc_lds = fx.make_view(
                    lds.sorted_lds.ptr, fx.make_layout(BLOCK_TILE_SIZE_M, 1)
                )
                asc_thr = asc_index_copy.get_slice(tid).partition_S(asc_lds)
                asc_idx = fx.make_fragment_like(asc_thr)
                fx.copy(cp_atom_idx, asc_thr, asc_idx)

            c_gate_frag, c_up_frag = gemm_1x4(
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N,
                TILE_K,
                blk_n,
                arg_p_input,
                arg_p_weight,
                lds,
            )

            if const_expr(weight_dtype != fx.BFloat16):
                _apply_1x4_fp8_dequant(
                    c_gate_frag,
                    c_up_frag,
                    tid,
                    expert_id,
                    blk_n,
                    contiguous_n,
                    asc_idx,
                    M,
                    p_w_scale,
                    p_a_scale,
                )

            c_out_bf16 = _gateup_pair_bf16(
                c_gate_frag,
                c_up_frag,
                situ_beta_f,
                situ_beta_rcp_f,
                situ_linear_beta_f,
                situ_linear_beta_rcp_f,
                activation_limit_f,
            )

            # 128-bit CShuffle epilogue (single region). Stage c_out_bf16 into the A LDS via
            # make_tiled_copy_C (framework-consistent with the make_fragment_C layout), reusing
            # the same 1x4 tiled_mma as _gemm_1x4. B-first makes the value dim 4 contiguous
            # channels, so the store is 64-bit; read it back channel-contiguous (8 bf16/lane)
            # so the scatter issues 128-bit writes.
            _, _tiled_mma = _make_1x4_tiled_mma()
            cshuf_atom_w = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
            cshuf_atom_r = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
            cshuf_ptr = fx.recast_iter(fx.BFloat16, lds.gemm.a_ping.ptr)
            # B-first: store through the (channel=M, token=N) transpose view so the value dim
            # (4 contiguous channels) is channel-contiguous -> 64-bit ds_write; read back the
            # aliased (token, channel) view channel-contiguous (8 bf16/lane) for the 128b
            # scatter. Both views share the same LDS bytes AND linear-offset formula, so the
            # same XOR swizzle keeps them consistent. The swizzle is required: the token stride
            # (contiguous_n elems) is bank-aligned, so an unswizzled 64-bit store is 16-way
            # bank-conflicted; the swizzle spreads it (needs no extra LDS, unlike padding).
            # C-staging is bf16 in both the bf16 and fp8 paths (it holds the bf16 output), so
            # the de-conflict swizzle is bf16's (3,3,3) in both cases -- NOT the fp8 input swz.
            swz_c = fx.SwizzleType.get(3, 3, 3)
            lds_c_store = fx.make_view(
                cshuf_ptr,
                fx.make_composed_layout(
                    fx.static(swz_c),
                    fx.make_ordered_layout(
                        (contiguous_n, BLOCK_TILE_SIZE_M), order=(0, 1)
                    ),
                ),
            )
            lds_c = fx.make_view(
                cshuf_ptr,
                fx.make_composed_layout(
                    fx.static(swz_c),
                    fx.make_ordered_layout(
                        (BLOCK_TILE_SIZE_M, contiguous_n), order=(1, 0)
                    ),
                ),
            )

            gpu.barrier()  # gemm_1x4's LDS reads must finish before reusing GemmBuffers
            store_c = fx.make_tiled_copy_C(cshuf_atom_w, _tiled_mma).get_slice(tid)
            fx.copy(
                cshuf_atom_w,
                store_c.retile(c_out_bf16),
                store_c.partition_D(lds_c_store),
            )
            gpu.barrier()
            rd = fx.make_fragment_like(c_rw_copy.get_slice(tid).partition_S(lds_c))
            fx.copy(cshuf_atom_r, c_rw_copy.get_slice(tid).partition_S(lds_c), rd)
            c_out.copy(buf_atom_w128, blk_n, rd)

    @flyc.jit
    def launch_splitk(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        situ_beta_f: fx.Float32,
        situ_beta_rcp_f: fx.Float32,
        situ_linear_beta_f: fx.Float32,
        situ_linear_beta_rcp_f: fx.Float32,
        activation_limit_f: fx.Float32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        num_n_blocks = fxh.div_up(N, BLOCK_TILE_SIZE_N)
        if const_expr(E is not None) and M * TOPK <= E:
            task_num = M * TOPK
        moe_2stage_gateup_splitk(
            p_input,
            p_weight,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            M,
            situ_beta_f,
            situ_beta_rcp_f,
            situ_linear_beta_f,
            situ_linear_beta_rcp_f,
            activation_limit_f,
        ).launch(grid=(num_n_blocks, task_num, 1), block=(256, 1, 1), stream=stream)

    @flyc.jit
    def launch_batch1(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_topk_ids: fx.Pointer,
        p_topk_weights: fx.Pointer,
        p_w_scale: fx.Pointer,
        M: fx.Int32,
        situ_beta_f: fx.Float32,
        situ_beta_rcp_f: fx.Float32,
        situ_linear_beta_f: fx.Float32,
        situ_linear_beta_rcp_f: fx.Float32,
        activation_limit_f: fx.Float32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        num_n_blocks = fxh.div_up(N, BLOCK_TILE_SIZE_N)
        moe_2stage_gateup_batch1(
            p_input,
            p_weight,
            p_output,
            p_topk_ids,
            p_topk_weights,
            p_w_scale,
            situ_beta_f,
            situ_beta_rcp_f,
            situ_linear_beta_f,
            situ_linear_beta_rcp_f,
            activation_limit_f,
        ).launch(grid=(num_n_blocks, TOPK, M), block=(256, 1, 1), stream=stream)

    @flyc.jit
    def launch_prefill_1x4(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        situ_beta_f: fx.Float32,
        situ_beta_rcp_f: fx.Float32,
        situ_linear_beta_f: fx.Float32,
        situ_linear_beta_rcp_f: fx.Float32,
        activation_limit_f: fx.Float32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        num_n_blocks = fxh.div_up(N, BLOCK_TILE_SIZE_N)
        task_num *= gateup_tasks_per_metadata
        if const_expr(E is not None) and M * TOPK <= E:
            task_num = M * TOPK * gateup_tasks_per_metadata
        moe_2stage_gateup_prefill_1x4(
            p_input,
            p_weight,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            p_a_scale,
            M,
            situ_beta_f,
            situ_beta_rcp_f,
            situ_linear_beta_f,
            situ_linear_beta_rcp_f,
            activation_limit_f,
        ).launch(grid=(num_n_blocks, task_num, 1), block=(256, 1, 1), stream=stream)

    if const_expr(alg == "prefill_1x4"):
        return launch_prefill_1x4
    if const_expr(alg == "batch1"):
        return launch_batch1
    return launch_splitk


@functools.cache
def _compile_moe_gemm1_cached(
    device_cache_key,
    *,
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    alg="splitk",
    E=None,
    act_quant_type=None,
    tile_k=None,
    activation="silu",
    swiglu_limit=None,
    mxfp4_gate_up_interleaved=True,
    fused_down_clear=False,
    METADATA_TILE_SIZE_M=None,
):
    del device_cache_key
    return _build_moe_gemm1(
        N=N,
        K=K,
        weight_dtype=weight_dtype,
        weight_quant_type=weight_quant_type,
        TOPK=TOPK,
        BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M,
        BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N,
        stage="gateup",
        alg=alg,
        E=E,
        act_quant_type=act_quant_type,
        tile_k=tile_k,
        activation=activation,
        swiglu_limit=swiglu_limit,
        down_path="default",
        down_output_padding_bytes=None,
        METADATA_TILE_SIZE_M=METADATA_TILE_SIZE_M,
        mxfp4_gate_up_interleaved=mxfp4_gate_up_interleaved,
        fused_down_clear=fused_down_clear,
    )


def compile_moe_gemm1(
    *,
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    alg="splitk",
    E=None,
    act_quant_type=None,
    tile_k=None,
    activation="silu",
    swiglu_limit=None,
    mxfp4_gate_up_interleaved=True,
    fused_down_clear=False,
    METADATA_TILE_SIZE_M=None,
):
    """Build and cache a stage1 gate-up launcher for one static configuration."""
    return _compile_moe_gemm1_cached(
        get_device_cache_key(),
        N=N,
        K=K,
        weight_dtype=weight_dtype,
        weight_quant_type=weight_quant_type,
        TOPK=TOPK,
        BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M,
        BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N,
        alg=alg,
        E=E,
        act_quant_type=act_quant_type,
        tile_k=tile_k,
        activation=activation,
        swiglu_limit=swiglu_limit,
        mxfp4_gate_up_interleaved=mxfp4_gate_up_interleaved,
        fused_down_clear=fused_down_clear,
        METADATA_TILE_SIZE_M=METADATA_TILE_SIZE_M,
    )


compile_moe_gemm1.cache_clear = _compile_moe_gemm1_cached.cache_clear
compile_moe_gemm1.cache_info = _compile_moe_gemm1_cached.cache_info
