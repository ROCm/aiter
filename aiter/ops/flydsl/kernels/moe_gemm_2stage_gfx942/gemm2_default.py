# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE stage2 default down-projection kernel builder."""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm, vector
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from . import layout_helpers as fxh
from .common import get_down_device_config as _get_down_device_config

# gfx942 raw-buffer aux bit 1 selects the non-temporal policy.
_DOWN_STORE_CACHE_MODIFIER = 2


def _build_moe_gemm2_default(
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    stage="down",
    alg="splitk",
    E=None,
    USE_ATOMIC_WRITE=True,
    act_quant_type=None,
    tile_k=None,
    activation="silu",
    swiglu_limit=None,
    mxfp4_gate_up_interleaved=True,
    fused_down_clear=False,
    down_path="default",
    down_output_padding_bytes=None,
    METADATA_TILE_SIZE_M=None,
):
    assert stage == "down"
    assert down_path == "default"
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
        assert K % 128 == 0, f"fp4 down K must be a multiple of 128, got {K}"
    else:
        assert weight_quant_type != "mxfp4", "mxfp4 quantization requires fp4 weights"
    if METADATA_TILE_SIZE_M is None:
        METADATA_TILE_SIZE_M = BLOCK_TILE_SIZE_M
    assert (
        BLOCK_TILE_SIZE_M == METADATA_TILE_SIZE_M
    ), "only gateup prefill_1x4 supports different kernel/metadata M tiles"
    assert down_path in ("default", "1x4_64x256", "2x4", "1x8")
    use_1x4_64x256 = down_path == "1x4_64x256"
    use_2x4 = down_path == "2x4"
    use_1x8 = down_path == "1x8"
    _topology_enabled, _generic_xcd_count = (
        _get_down_device_config()
        if use_1x4_64x256 or use_2x4 or use_1x8
        else (False, 8)
    )
    block_m_per_4wave_group = 64

    # Down path selection. 1x4_64x256 uses a 1x4-wave M64xN256 workgroup;
    # 2x4 combines two independent 4-wave subgroups into one M128xN256
    # workgroup; 1x8 uses a 1x8-wave M64xN512 workgroup.
    if use_1x4_64x256:
        assert BLOCK_TILE_SIZE_M == 64
        assert BLOCK_TILE_SIZE_N == 256
    elif use_2x4:
        assert BLOCK_TILE_SIZE_M == 128
        assert BLOCK_TILE_SIZE_N == 256
    elif use_1x8:
        assert BLOCK_TILE_SIZE_M == 64
        assert BLOCK_TILE_SIZE_N == 512
        assert weight_quant_type == "per_tensor"
        assert act_quant_type == "per_tensor"
    else:
        assert down_output_padding_bytes is None

    cshuffle_2x4_bytes = 8 * 16 * 64 * 2

    # MI308X有4个XCC，每个XCC包含4个SE，每个SE包含5个CU。
    # topology map仅在MI308X上启用；每个分区的任务数由运行时有效任务数推导。
    if use_1x4_64x256 or use_2x4 or use_1x8:
        assert stage == "down" and alg == "prefill_1x4"
        assert N % BLOCK_TILE_SIZE_N == 0
        assert K % 64 == 0
        assert weight_dtype == "fp8"
        assert weight_quant_type in ("ptpc", "per_tensor")
        assert down_output_padding_bytes in (0, 32, 64, 128)
        if use_1x4_64x256:
            activation_bytes = block_m_per_4wave_group * K
            scale_bytes = BLOCK_TILE_SIZE_N * 4 if weight_quant_type == "ptpc" else 0
            cshuffle_bytes = 4 * 16 * 64 * (fx.BFloat16.width // 8)
            assert activation_bytes + scale_bytes + cshuffle_bytes <= 64 * 1024, (
                "1x4_64x256 exceeds gfx942 LDS capacity; "
                f"activation={activation_bytes}B, scale={scale_bytes}B, "
                f"cshuffle={cshuffle_bytes}B"
            )
        if use_2x4:
            activation_bytes = 2 * block_m_per_4wave_group * K
            assert activation_bytes + cshuffle_2x4_bytes <= 64 * 1024, (
                "2x4 requires activation and row-major CShuffle "
                "to fit gfx942 LDS; "
                f"activation={activation_bytes}B, "
                f"cshuffle={cshuffle_2x4_bytes}B"
            )
        if use_1x8:
            activation_bytes = block_m_per_4wave_group * K
            cshuffle_bytes = 8 * 16 * 64 * (fx.BFloat16.width // 8)
            assert activation_bytes + cshuffle_bytes <= 64 * 1024, (
                "1x8 exceeds gfx942 LDS capacity; "
                f"activation={activation_bytes}B, cshuffle={cshuffle_bytes}B"
            )
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

        @fx.struct
        class SharedStorage:
            sorted_lds: fx.Array[fx.Int32, 256, 16]

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

    def _mxfp4_scale_index(expert_id, blk_n, local_n, k_group):
        groups_padded = ((K // 32 + 7) // 8) * 8
        row = fx.Int64(expert_id) * N + fx.Int64(blk_n * BLOCK_TILE_SIZE_N + local_n)
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
    ):
        n_rows = fx.size(fx.get_shape(src_tensor)).to_py_value() // 4
        k_group = k_idx * (tile_k_per_wg // 32) + tid // 16
        scale_byte_ptr = fx.recast_iter(fx.Uint8, fxh._as_ptr(p_w_scale))
        scale_u32_type = fx.PointerType.get(
            fx.Uint32.ir_type, scale_byte_ptr.memspace, 4
        )
        scale_u32_ptr = fx.recast_iter(scale_u32_type, scale_byte_ptr)
        packed_scale_rows = [None] * n_rows
        if const_expr(n_rows >= 2 and BLOCK_TILE_SIZE_N >= 64):
            for row_pair in range_constexpr(n_rows // 2):
                local_n = tid % 16 + row_pair * 32
                scale_idx = _mxfp4_scale_index(expert_id, blk_n, local_n, k_group)
                packed_scale = scale_u32_ptr[scale_idx // 4]
                packed_scale_rows[row_pair * 2] = packed_scale
                packed_scale_rows[row_pair * 2 + 1] = packed_scale
        else:
            for row in range_constexpr(n_rows):
                local_n = tid % 16 + row * 16
                scale_idx = _mxfp4_scale_index(expert_id, blk_n, local_n, k_group)
                packed_scale_rows[row] = scale_u32_ptr[scale_idx // 4]
        return packed_scale_rows

    def _cvt_mxfp4_bf16(
        src_tensor,
        dst_tensor,
        p_w_scale,
        expert_id,
        blk_n,
        k_idx,
        tile_k_per_wg,
        tid,
        packed_scale_rows=None,
    ):
        n_dwords = fx.size(fx.get_shape(src_tensor)).to_py_value()
        src_vec = src_tensor.load()
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
            )

        scales = []
        for row in range_constexpr(n_rows):
            scale_idx = _mxfp4_scale_index(
                expert_id,
                blk_n,
                tid % 16 + row * 16,
                k_group,
            )
            scales.append(
                _mxfp4_scale_from_dword(packed_scale_rows[row], scale_idx % 4)
            )

        items = []
        for dword_idx in range_constexpr(n_dwords):
            for byte_idx in range_constexpr(4):
                pair = llvm.call_intrinsic(
                    T.vec(2, T.bf16),
                    "llvm.amdgcn.cvt.scalef32.pk.bf16.fp4",
                    [
                        as_ir_value(src_vec[dword_idx]),
                        as_ir_value(scales[dword_idx // 4]),
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
        vec = Vec.from_elements(items, fx.BFloat16)
        layout = fx.get_layout(dst_tensor)
        for i in range_constexpr(8 * n_dwords):
            dst_tensor[fx.idx2crd(i, layout)] = vec[i]

    def _apply_down_scale(c_frag, tid, expert_id, blk_n, p_w_scale):
        if const_expr(weight_dtype != fx.BFloat16):
            if const_expr(weight_quant_type == "ptpc"):
                arg_p_scale = fx.make_view(
                    fxh._as_ptr(p_w_scale) + expert_id * N, fx.make_layout(N, 1)
                )
                scale_tile = fx.flat_divide(
                    arg_p_scale, fx.make_tile(BLOCK_TILE_SIZE_N)
                )[None, blk_n]
                cp_atom_scale = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
                tiled_copy_scale = fx.make_tiled_copy(
                    cp_atom_scale,
                    fx.make_layout(((16, 4), 4), ((0, 4), 1)),
                    fx.make_tile(16),
                )
                scale_frag_tensor = tiled_copy_scale.get_slice(tid).partition_S(
                    scale_tile
                )
                scale_frag = fx.make_fragment_like(scale_frag_tensor)
                fx.copy(cp_atom_scale, scale_frag_tensor, scale_frag)
                m_reps = fx.size(fx.get_shape(c_frag)[1]).to_py_value()
                n_reps = fx.size(fx.get_shape(c_frag)[2]).to_py_value()
                for n in range_constexpr(n_reps):
                    scale_vec = scale_frag[None, n].load()
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

    def _cvt_f32_to_bf16(c_frag):
        c_frag_bf16 = fx.make_fragment_like(c_frag, dtype=fx.BFloat16)
        if const_expr(is_gfx950):
            c_frag_bf16.store(c_frag.load().to(fx.BFloat16))
        else:
            round_bit = fx.Uint32(0x8000)
            c_frag_bf16.store(
                ((c_frag.load().bitcast(fx.Uint32) + round_bit) >> 16)
                .to(fx.Uint16)
                .bitcast(fx.BFloat16)
            )
        return c_frag_bf16

    def _make_down_weight_view(p_weight, expert_id):
        # Preshuffle weight [16, (element_num, K//element_num)] without silu grouping. Shared
        # by the splitk / batch1 down kernels.
        storage_k = K // 2 if const_expr(weight_dtype == fx.Float4E2M1FN) else K
        element_num = 16 // (p_weight.dtype.width // 8)
        return fx.make_view(
            p_weight + fx.Int64(expert_id) * N * storage_k,
            fx.make_layout(
                ((16, N // 16), (element_num, storage_k // element_num)),
                ((element_num, 16 * storage_k), (1, 16 * element_num)),
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
    ):
        tid = gpu.thread_idx.x

        tile_k_per_wave = 128 if const_expr(weight_dtype == fx.Float4E2M1FN) else TILE_K
        tile_k_per_wg = tile_k_per_wave * splitk_waves

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
                    packed_scale_rows,
                )
            else:
                if const_expr(weight_dtype != fx.BFloat16):
                    _cvt_fp8_bf16(b_frag_retile[buf], b_frag[buf])
            fx.gemm(tiled_mma, c_frag, b_frag[buf], a_frag[buf], c_frag)

        def _prefetch_a(k_idx, buf):
            if const_expr(a_with_index):
                a_tensor_thr.copy(a_cp_atom_r, k_idx, a_frag_retile[buf])
            else:
                fx.copy(
                    a_cp_atom_r,
                    a_tensor_thr[None, None, None, k_idx],
                    a_frag_retile[buf],
                )

        def _prefetch_b(k_idx, buf):
            fx.copy(
                b_cp_atom_r,
                b_tensor_thr[None, None, None, k_idx],
                b_frag_retile[buf],
            )

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
                    )
                )

        if const_expr(not down_pair_scales):
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

        splitk_stage_mask = 0x20 if stage == "down" and alg == "batch1" else 0
        rocdl.sched_barrier(0 if weight_dtype == fx.Float4E2M1FN else splitk_stage_mask)

        if const_expr(weight_dtype == fx.Float4E2M1FN):
            c_frag.store(acc_init)
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
            for k2, state in range(0, num_k_iters // 2, 1, init=[acc_init]):
                c_frag.store(state[0])
                k_base = fx.Int32(k2 * 2)
                _prefetch_a(k_base + 1, 1)
                _prefetch_b(k_base + 1, 1)
                rocdl.s_waitcnt(_encode_waitcnt(vmcnt=vmcnt_per_prefetch))
                rocdl.sched_barrier(splitk_stage_mask)
                _gemm_stage(0, k_base)
                rocdl.sched_barrier(splitk_stage_mask)
                _prefetch_a(k_base + 2, 0)
                _prefetch_b(k_base + 2, 0)
                rocdl.s_waitcnt(_encode_waitcnt(vmcnt=vmcnt_per_prefetch))
                rocdl.sched_barrier(splitk_stage_mask)
                _gemm_stage(1, k_base + 1)
                rocdl.sched_barrier(splitk_stage_mask)
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

    @flyc.kernel
    def moe_2stage_down_splitk(
        p_input: fx.Pointer,  # bf16 [M, TOPK, K]
        p_weight: fx.Pointer,  # quantized/bf16 [N/16, K/8 * 16 * 8]
        p_output: fx.Pointer,  # bf16 [M, N]
        # sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y

        arg_p_input = fx.make_view(
            fxh._as_ptr(p_input), fx.make_layout((M, TOPK, K), (TOPK * K, K, 1))
        )
        num_valid_buf = fx.make_view(
            fx.recast_iter(fx.Int32, fxh._as_ptr(p_num_valid_ids)), fx.make_layout(1, 1)
        )
        max_valid_id = num_valid_buf[0]
        if const_expr(weight_dtype != fx.BFloat16):
            p_weight = fx.recast_iter(fx.Uint8, fxh._as_ptr(p_weight))
        if e_idx * BLOCK_TILE_SIZE_M < max_valid_id:
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
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
            arg_p_weight = _make_down_weight_view(p_weight, expert_id)

            # sorted ids: global -> LDS (scalar load/store, only first BLOCK_TILE_SIZE_M threads participate)
            sorted_ids_buf = fx.rocdl.make_buffer_tensor(
                arg_p_sorted_ids, max_size=False
            )
            lds_view = fx.make_view(
                lds.sorted_lds.ptr, fx.make_layout(BLOCK_TILE_SIZE_M, 1)
            )
            for idx in range(tid, BLOCK_TILE_SIZE_M, 64):
                # fx.memref_store(val, lds_view, tid)
                lds_view[idx] = sorted_ids_buf[idx]
            gpu.barrier()

            cp_atom_weight = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            arg_p_sorted_weights = fx.make_view(
                fx.recast_iter(
                    fx.Float32,
                    fxh._as_ptr(p_sorted_weights) + e_idx * BLOCK_TILE_SIZE_M,
                ),
                fx.make_layout(BLOCK_TILE_SIZE_M, 1),
            )
            sorted_weights_buf = fx.rocdl.make_buffer_tensor(
                arg_p_sorted_weights, max_size=False
            )
            cp_atom_lds = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
            tiled_copy_sortid_lds = fx.make_tiled_copy(
                cp_atom_lds, fx.make_layout(((16, 4), 1), ((1, 0), 0)), fx.make_tile(16)
            )
            sorted_weights_tensor = tiled_copy_sortid_lds.get_slice(tid).partition_S(
                sorted_weights_buf
            )
            sorted_weight_frag = fx.make_fragment_like(
                sorted_weights_tensor, fx.Float32
            )
            fx.copy(cp_atom_weight, sorted_weights_tensor, sorted_weight_frag)

            c_frag = gemm_splitk(
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N,
                TILE_K,
                blk_n,
                arg_p_input,
                arg_p_weight,
                lds,
                splitk_waves=1,
                p_w_scale=p_w_scale,
                expert_id=expert_id,
            )

            _apply_down_scale(c_frag, tid, expert_id, blk_n, p_w_scale)

            # mul weight
            sorted_weight_frag_vec = sorted_weight_frag.load()
            for m in range_constexpr(BLOCK_TILE_SIZE_M // 16):
                w = sorted_weight_frag_vec[m]
                v = c_frag[None, m, None].load()
                v *= w
                c_frag[None, m, None].store(v)

            c_frag_bf16 = _cvt_f32_to_bf16(c_frag)

            # write to mem
            if const_expr(not USE_ATOMIC_WRITE):  # gateup output shape: [M, TOPK, N]
                arg_p_output = fx.make_view(
                    fxh._as_ptr(p_output),
                    fx.make_layout((M, TOPK, N), (TOPK * N, N, 1)),
                )
                arg_p_output = fx.rocdl.make_buffer_tensor(
                    arg_p_output,
                    max_size=False,
                    num_records_bytes=fx.Int64(M) * (TOPK * N * fx.BFloat16.width // 8),
                )
                cp_atom_w = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.BFloat16)
                is_atomic_write = False
            else:
                arg_p_output = fx.make_view(
                    fxh._as_ptr(p_output), fx.make_layout((M, N), (N, 1))
                )
                # arg_p_output = fx.rocdl.make_buffer_tensor(arg_p_output, max_size=False, num_records_bytes=M * TOPK * N * fx.BFloat16.width // 8)
                # cp_atom_w = fx.make_copy_atom(fx.rocdl.BufferAtomicAdd(fx.BFloat16), fx.BFloat16)
                cp_atom_w = fx.make_copy_atom(
                    fx.UniversalAtomic(fx.AtomicOp.Add, fx.BFloat16), fx.BFloat16
                )
                is_atomic_write = True
            c_tiled_g = fx.make_tiled_copy(
                cp_atom_w,
                # 16x4 threads, each writes 4 points in N dimension
                fx.make_layout(((16, 4), 4), ((1, 64), 16)),
                fx.make_tile(16, 16),
            )
            c_index_frag = _read_sorted_index(
                tiled_copy_sortid_lds, tid, lds.sorted_lds
            )
            c_tensor = TensorWithIndex(
                arg_p_output,
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N,
                c_index_frag,
                c_tiled_g,
                tid,
                is_read_from_mem=False,
                TOPK=TOPK,
                is_atomic_write=is_atomic_write,
            )
            c_tensor.copy(
                cp_atom_w, blk_n, c_tiled_g.get_slice(tid).retile(c_frag_bf16)
            )

    @flyc.kernel
    def moe_2stage_down_batch1(
        p_input: fx.Pointer,  # bf16 [M, TOPK, K]
        p_weight: fx.Pointer,  # quantized/bf16 [N/16, K/8 * 16 * 8]
        p_output: fx.Pointer,  # bf16 [M, N]
        p_topk_ids: fx.Pointer,
        p_topk_weights: fx.Pointer,
        p_w_scale: fx.Pointer,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x
        e_idx = gpu.block_idx.y
        batch_idx = gpu.block_idx.z
        route_idx = batch_idx * TOPK + e_idx

        # batch1: input is gemm1_out[0, e_idx, :] (single token, expert slot e_idx). Point at that
        # row and broadcast it across the TILE_M MFMA rows (stride 0); every computed row is then
        # identical, so any single row is the real result.
        arg_p_input = fx.make_view(
            fxh._as_ptr(p_input) + fx.Int64(route_idx * K),
            fx.make_layout((BLOCK_TILE_SIZE_M, K), (0, 1)),
        )
        if const_expr(weight_dtype != fx.BFloat16):
            p_weight = fx.recast_iter(fx.Uint8, fxh._as_ptr(p_weight))
        arg_p_topk_ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_topk_ids))
        arg_p_topk_weights = fx.recast_iter(fx.Float32, fxh._as_ptr(p_topk_weights))
        expert_id = arg_p_topk_ids[route_idx]
        topk_weight = arg_p_topk_weights[route_idx]
        arg_p_weight = _make_down_weight_view(p_weight, expert_id)

        c_frag = gemm_splitk(
            BLOCK_TILE_SIZE_M,
            BLOCK_TILE_SIZE_N,
            TILE_K,
            blk_n,
            arg_p_input,
            arg_p_weight,
            None,
            splitk_waves=1,
            a_with_index=False,
            p_w_scale=p_w_scale,
            expert_id=expert_id,
        )

        _apply_down_scale(c_frag, tid, expert_id, blk_n, p_w_scale)

        # mul weight
        c_frag.store(c_frag.load() * topk_weight)

        c_frag_bf16 = _cvt_f32_to_bf16(c_frag)

        # write to mem
        arg_p_output = fx.make_view(
            fxh._as_ptr(p_output) + fx.Int64(batch_idx * N),
            fx.make_layout((1, N), (N, 1)),
        )
        cp_atom_w = fx.make_copy_atom(
            fx.UniversalAtomic(fx.AtomicOp.Add, fx.BFloat16), fx.BFloat16
        )
        c_tiled_g = fx.make_tiled_copy(
            cp_atom_w,
            # 16x4 threads, each writes 4 points in N dimension
            fx.make_layout(((16, 4), 4), ((1, 64), 16)),
            fx.make_tile(16, 16),
        )
        c_tile = fx.flat_divide(
            arg_p_output, fx.make_tile(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N)
        )[None, None, None, blk_n]
        c_dst = c_tiled_g.get_slice(tid).partition_S(c_tile)
        c_src = c_tiled_g.get_slice(tid).retile(c_frag_bf16)
        rep_m = fx.size(fx.get_shape(c_src)[1]).to_py_value()
        rep_n = fx.size(fx.get_shape(c_src)[2]).to_py_value()
        if tid % 16 == 0:
            for m in range_constexpr(rep_m):
                for n in range_constexpr(rep_n):
                    reg_vec = c_src[None, m, n].load()
                    ptr_base = fx.get_iter(c_dst[None, m, n, 0])
                    fxh.atomic_add_bf16(ptr_base, reg_vec)

    flyobj = fxh.FlyObjCache()

    @flyc.kernel
    def moe_2stage_down_prefill_1x4(
        p_input: fx.Pointer,  # bf16/fp8 [M, TOPK, K]       K = HIDDEN_STATES//TP
        p_weight: fx.Pointer,  # quantized/bf16 [E, N, K]   N = HIDDEN_STATES
        p_output: fx.Pointer,  # bf16 [M, TOPK, N]
        p_sorted_ids: fx.Pointer,  # int32 [num_tokens_sorted]
        p_sorted_weights: fx.Pointer,  # f32 [num_tokens_sorted]
        p_sorted_expert_ids: fx.Pointer,  # int32 [num_blocks] num_tokens_sorted <= num_blocks * BLOCK_TILE_SIZE_M
        p_num_valid_ids: fx.Pointer,  # int32 [2]  value: (sorting valid rows incl. expert padding, M)
        p_w_scale: fx.Pointer,  # weight fp8 scale (per-output-channel ptpc / per-tensor)
        p_a_scale: fx.Pointer,  # input fp8 scale (per-token ptpc / per-tensor)
        M: fx.Int32,
    ):
        e_idx = fx.gpu.block_idx.y

        max_valid_id = fxh.view_as_torch_tensor(p_num_valid_ids, (1,), fx.Int32)[0]

        if e_idx * BLOCK_TILE_SIZE_M < max_valid_id:
            arg_p_input = fxh.view_as_torch_tensor(p_input, (M, TOPK, K), weight_dtype)
            arg_p_output = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_output, fx.BFloat16)
                + fx.Int64(e_idx) * (BLOCK_TILE_SIZE_M * N),
                (BLOCK_TILE_SIZE_M, N),
            )
            arg_p_sorted_ids = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_sorted_ids) + e_idx * BLOCK_TILE_SIZE_M,
                (BLOCK_TILE_SIZE_M,),
                fx.Int32,
            )
            arg_p_sorted_weights = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_sorted_weights) + e_idx * BLOCK_TILE_SIZE_M,
                (BLOCK_TILE_SIZE_M,),
                fx.Float32,
            )
            expert_id = fxh.view_as_torch_tensor(p_sorted_expert_ids, (1,), fx.Int32)[
                e_idx
            ]

            # 16bytes/DW4
            element_num = 16 // (weight_dtype.width // 8)
            arg_p_weight = fx.make_view(
                fxh._as_ptr(p_weight, weight_dtype) + fx.Int64(expert_id * N * K),
                fx.make_layout(
                    ((16, N // 16), (element_num, K // element_num)),
                    ((element_num, 16 * K), (1, 16 * element_num)),
                ),
            )

            arg_p_weight = fx.rocdl.make_buffer_tensor(arg_p_weight, max_size=False)
            arg_p_output = fx.rocdl.make_buffer_tensor(arg_p_output, max_size=False)

            BLOCK_M = BLOCK_TILE_SIZE_M
            BLOCK_N = 64
            BLOCK_K = 64 // (weight_dtype.width // 8)

            # mask,base,shift, swizzle always in unit of 128b,
            swz_base = ((128 // weight_dtype.width) - 1).bit_length()
            swz = fx.SwizzleType.get(3, swz_base, 3)

            act_dtype = weight_dtype  # fp8 / bf16

            @fx.union
            class SharedStorage:
                A: fx.Array[act_dtype, BLOCK_M * K]
                C: fx.Array[fx.BFloat16, 2 * BLOCK_M * BLOCK_N]

            lds = fx.SharedAllocator().allocate(SharedStorage)
            ldsA0 = lds.A.peek().view(
                fx.make_composed_layout(fx.static(swz), fxh.torch_layout(BLOCK_M, K))
            )
            layoutC = fx.make_composed_layout(
                fx.static(swz),
                fx.make_ordered_layout((BLOCK_M, BLOCK_N, 2), (1, 0, 2)),
            )
            layoutCt = fx.make_composed_layout(
                fx.static(swz), fx.make_ordered_layout((BLOCK_N, BLOCK_M, 2), (0, 1, 2))
            )
            ldsC = lds.C.peek().view(layoutC)
            ldsCt = lds.C.peek().view(layoutCt)

            arg_p_input = fx.rocdl.make_buffer_tensor(
                arg_p_input,
                max_size=False,
                num_records_bytes=fx.Int64(M)
                * (TOPK * K)
                * (arg_p_input.dtype.width // 8),
            )
            cp_atom = flyobj.get_buffer_copy_atom(arg_p_input.dtype, 128)

            def flatten_A(x):
                # second mode is innermost, so swap before flattening
                # to get the right order for the tiled copy
                x = fx.select(x, [1, 0])
                return fx.group(x, 0, -1)

            cp_ldsA0 = flatten_A(ldsA0)
            cp_rows = flatten_A(
                fxh.make_1d_coord_tensor(ldsA0, 0, fx.get_iter(arg_p_sorted_ids))
            )
            cp_cols = flatten_A(
                fxh.make_1d_coord_tensor(ldsA0, 1, fx.make_int_tuple(0))
            )
            for dst, row, col in fxh.all_copy_atoms(
                cp_ldsA0, cp_rows, cp_cols, atom_bits=128, num_threads=256
            ):
                sorted_id = row[0].bitcast(fx.Uint32)
                atom_A = fxh.atom_tensor(
                    arg_p_input, (sorted_id & 0xFFFFFF, sorted_id >> 24, col[0]), 128
                )
                fx.copy(cp_atom, atom_A, dst)
            fx.gpu.barrier()

            # (BLOCK_N, BLOCK_K, num_blocks_N, num_blocks_K)
            weight = fx.flat_divide(arg_p_weight, (BLOCK_N, BLOCK_K))
            ldsA = fx.flat_divide(ldsA0, (BLOCK_M, BLOCK_K))

            nBN = fxh.div_up(N, BLOCK_N)
            nBK = fxh.div_up(K, BLOCK_K)

            mm = flyobj.create_thr_mma(weight_dtype, (4, 1, 1))

            c_fake_tensor = fx.make_view(
                fx.get_iter(arg_p_input),
                fx.make_ordered_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            fragC = [
                mm.make_fragment_C(c_fake_tensor),
                mm.make_fragment_C(c_fake_tensor),
            ]
            fragC_bf16 = fx.make_fragment_like(fragC[0], fx.BFloat16)

            frag_act = flyobj.load_tiled_mma_fragB(mm, ldsA, copy_atom_bits=128)
            fx.gpu.barrier()  # make sure all threads finished using ldsA (since it's reused by ldsC)

            arg_w_scale = None
            if const_expr(weight_quant_type == "per_tensor"):
                arg_w_scale = fx.make_view(
                    fxh._as_ptr(p_w_scale) + expert_id, fx.make_layout((N, 1), (0, 0))
                )
                arg_w_scale = fx.flat_divide(arg_w_scale, (BLOCK_N, 1))
            if const_expr(weight_quant_type == "ptpc"):
                arg_w_scale = fx.make_view(
                    fxh._as_ptr(p_w_scale) + expert_id * N,
                    fx.make_layout((N, 1), (1, 0)),
                )
                # (BLOCK_N, 1, num_block_N, 1)
                arg_w_scale = fx.flat_divide(arg_w_scale, (BLOCK_N, 1))

            arg_a_scale = None
            if const_expr(act_quant_type == "per_tensor"):
                arg_a_scale = fx.make_view(
                    fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)),
                    fx.make_layout((M, TOPK), (0, 0)),
                )
                arg_a_scale = fx.rocdl.make_buffer_tensor(
                    arg_a_scale,
                    max_size=False,
                    num_records_bytes=fx.Int64(1) * (arg_a_scale.dtype.width // 8),
                )
            if const_expr(act_quant_type == "ptpc"):
                arg_a_scale = fx.make_view(
                    fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)),
                    fx.make_layout((M, TOPK), (TOPK, 1)),
                )
                arg_a_scale = fx.rocdl.make_buffer_tensor(
                    arg_a_scale,
                    max_size=False,
                    num_records_bytes=fx.Int64(M)
                    * TOPK
                    * (arg_a_scale.dtype.width // 8),
                )

            sorted_weights = fx.make_view(
                fx.get_iter(arg_p_sorted_weights),
                fx.make_layout((BLOCK_N, BLOCK_M), (0, 1)),
            )
            # load rows/token weights using C layout
            frag_sorted_weight = flyobj.load_tiled_mma_fragC(
                mm, sorted_weights, copy_atom_bits=32
            )

            if fx.const_expr(arg_a_scale is not None):
                """load & combine per-token scales with per-token weights, and store into lds.C"""
                cp_atom = flyobj.get_buffer_copy_atom(p_a_scale.dtype, 32)
                coord_tensor = fx.make_view(
                    fx.get_iter(arg_p_sorted_ids),
                    fx.make_layout((BLOCK_N, BLOCK_M), (0, 1)),
                )
                frag_coord = flyobj.load_tiled_mma_fragC(
                    mm, coord_tensor, copy_atom_bits=32
                )
                frag_pt_scales = mm.make_fragment_C(coord_tensor)
                frag_pt_scalesr = flyobj.get_tiled_mma_retile(
                    mm, frag_pt_scales, "C", copy_atom=cp_atom
                )

                for dst, coord in fxh.all_elements(frag_pt_scalesr, frag_coord):
                    sorted_id = coord[0].bitcast(fx.Uint32)
                    atom_A = fxh.atom_tensor(
                        arg_a_scale,
                        (sorted_id & 0xFFFFFF, sorted_id >> 24),
                        32,
                    )
                    fx.copy(cp_atom, atom_A, dst)

                # combine per-token scales with per-token weights
                for frag_pt, frag_sw in fxh.all_elements(
                    frag_pt_scales, frag_sorted_weight
                ):
                    frag_pt.store(frag_pt.load() * frag_sw.load())

                frag_sorted_weight = frag_pt_scales

            def f32_to_bf16(x):
                round_bit = as_ir_value(fx.Uint32(0x8000)).bitcast(fx.Float32.ir_type)
                return (
                    ((x + round_bit).bitcast(fx.Uint32) >> 16)
                    .to(fx.Uint16)
                    .bitcast(fx.BFloat16)
                )

            def gemm_compute(fragW, fragPCS, fragC):
                fragC.fill(0)
                for k in fx.range_constexpr(nBK):
                    fx.gemm(
                        mm,
                        fragC,
                        fragW[None, None, None, k],
                        frag_act[None, None, None, 0, k],
                        fragC,
                    )
                if fx.const_expr(fragPCS is not None):
                    for fc, fpc in fxh.all_elements(fragC, fragPCS):
                        fc.store(fc.load() * fpc.load())

            col_tensor = fx.make_view(
                fx.make_int_tuple(0), fx.make_layout((BLOCK_M, N), (0, 1))
            )
            col_tensor = fx.flat_divide(col_tensor, (BLOCK_M, BLOCK_N))

            tcopyLDS, cp_ldsc = flyobj.get_tiled_copy_coalesced_mn(
                ldsC[None, None, 0], copy_atom_bits=128, num_threads=256
            )

            thrv_ldsC = tcopyLDS.partition_S(ldsC)

            copy_atom_ = flyobj.get_universal_copy_atom(fragC_bf16.dtype, 64)
            tcopy = flyobj.get_tiled_mma_copy(copy_atom_, mm, "C")
            fragC_bf16r = flyobj.get_retile(tcopy, fragC_bf16)

            thrv_ldsCt = flyobj.get_partition_D(tcopy, ldsCt)

            def postprocess_store2lds(fragC, ldsc_idx):
                for fc, fsw in fxh.all_elements(fragC, frag_sorted_weight):
                    fc.store(fc.load() * fsw.load())
                vec_f32 = fragC.load()
                fragC_bf16.store(f32_to_bf16(vec_f32))
                fx.copy(copy_atom_, fragC_bf16r, thrv_ldsCt[None, None, None, ldsc_idx])

            arg_p_output = fx.flat_divide(arg_p_output, (BLOCK_M, BLOCK_N))
            cp_atom_out_128b = flyobj.get_buffer_copy_atom(fx.BFloat16, 128)
            thrv_out = tcopyLDS.partition_D(arg_p_output)
            fragOut = fx.make_fragment_like(thrv_ldsC[None, None, None, 0])

            def postprocess_store2vmem(n, ldsc_idx):
                fx.copy(cp_ldsc, thrv_ldsC[None, None, None, ldsc_idx], fragOut)
                fx.copy(cp_atom_out_128b, fragOut, thrv_out[None, None, None, 0, n])

            """
            apply per-token scale & weights, cvt-bf16, write-fragC to LDS
            load fragC from LDS, write to global memory
            """

            def hot_loop_scheduler():
                """
                // to cross the SCHED_BARRIER during scheduling.
                //     MASK = 0x0000 0000: No instructions may be scheduled across SCHED_BARRIER.
                //     MASK = 0x0000 0001: ALL, non-memory, non-side-effect producing instructions may be
                //                         scheduled across SCHED_BARRIER, i.e. allow ALU instructions to pass.
                //     MASK = 0x0000 0002: VALU instructions may be scheduled across SCHED_BARRIER.
                //     MASK = 0x0000 0004: SALU instructions may be scheduled across SCHED_BARRIER.
                //     MASK = 0x0000 0008: MFMA/WMMA instructions may be scheduled across SCHED_BARRIER.
                //     MASK = 0x0000 0010: ALL VMEM instructions may be scheduled across SCHED_BARRIER.
                //     MASK = 0x0000 0020: VMEM read instructions may be scheduled across SCHED_BARRIER.
                //     MASK = 0x0000 0040: VMEM write instructions may be scheduled across SCHED_BARRIER.
                //     MASK = 0x0000 0080: ALL DS instructions may be scheduled across SCHED_BARRIER.
                //     MASK = 0x0000 0100: ALL DS read instructions may be scheduled accoss SCHED_BARRIER.
                //     MASK = 0x0000 0200: ALL DS write instructions may be scheduled across SCHED_BARRIER.
                """
                num_mfma_inst = (BLOCK_M // 16) * (
                    K // (16 if weight_dtype.width == 16 else 32)
                )
                num_stores = BLOCK_M // (256 // (BLOCK_N // 8))
                num_loads = K // ((4 * 8) if weight_dtype.width == 16 else (4 * 16))

                # print(num_loads, num_stores, num_mfma_inst)
                nloads = num_loads
                nstores = num_stores
                mfma_step = num_mfma_inst // (nloads + nstores)

                nmfma = num_mfma_inst - mfma_step * (nloads + nstores)
                if nmfma > 0:
                    fx.rocdl.sched_mfma(nmfma)

                for _ in fx.range_constexpr(nloads):
                    fx.rocdl.sched_mfma(mfma_step)
                    fx.rocdl.sched_group_barrier(0x10, 1, 0)

                for _ in fx.range_constexpr(nstores):
                    fx.rocdl.sched_mfma(mfma_step)
                    fx.rocdl.sched_group_barrier(0x10, 1, 0)

                fx.rocdl.sched_barrier(0)

            # prelog
            frag_weights = [None, None]
            frag_pc_scales = [None, None]
            frag_weights[0] = flyobj.load_tiled_mma_fragA(
                mm, weight, [None, None, 0, None]
            )
            if fx.const_expr(arg_w_scale is not None):
                frag_pc_scales[0] = flyobj.load_tiled_mma_fragC(
                    mm,
                    arg_w_scale,
                    [None, None, 0, 0],
                    copy_atom_bits=32 if weight_quant_type == "per_tensor" else 128,
                )

            # prelog
            gemm_compute(frag_weights[0], frag_pc_scales[0], fragC[0])
            frag_weights[1] = flyobj.load_tiled_mma_fragA(
                mm, weight, [None, None, 1, None]
            )
            if fx.const_expr(arg_w_scale is not None):
                frag_pc_scales[1] = flyobj.load_tiled_mma_fragC(
                    mm,
                    arg_w_scale,
                    [None, None, 1, 0],
                    copy_atom_bits=32 if weight_quant_type == "per_tensor" else 128,
                )

            postprocess_store2lds(fragC[0], 0)
            fx.gpu.barrier()
            """
            sched_group_barrier only search independent instructions within current basic-block
            and syn-threads/barrier is a boundary of basic-block, we need to respect this rules
            and clearly organize instructions into natural basic-blocks and apply sched_group_barrier
            to each of them:
                basic-block1: post-process & LDS write | prefetch part of next weight block
                    wait barrier
                basic-block2: gemm compute | prefetch part of next weight block | LDS-read + global-write
            """
            for n, state in range(0, nBN - 2, 2, init=[]):
                fxh.asm_mark("aaa")
                postprocess_store2vmem(n, 0)
                flyobj.load_tiled_mma_fragA(
                    mm, weight, [None, None, n + 2, None], frag_weights[0]
                )
                if fx.const_expr(
                    arg_w_scale is not None and weight_quant_type != "per_tensor"
                ):
                    flyobj.load_tiled_mma_fragC(
                        mm, arg_w_scale, [None, None, n + 2, 0], frag_pc_scales[0]
                    )
                gemm_compute(frag_weights[1], frag_pc_scales[1], fragC[1])
                postprocess_store2lds(fragC[1], 1)

                hot_loop_scheduler()
                fx.gpu.barrier()

                fxh.asm_mark("bbb")

                postprocess_store2vmem(n + 1, 1)
                flyobj.load_tiled_mma_fragA(
                    mm, weight, [None, None, n + 3, None], frag_weights[1]
                )
                # fxh.asm_mark("ccc")

                if fx.const_expr(
                    arg_w_scale is not None and weight_quant_type != "per_tensor"
                ):
                    flyobj.load_tiled_mma_fragC(
                        mm, arg_w_scale, [None, None, n + 3, 0], frag_pc_scales[1]
                    )
                gemm_compute(frag_weights[0], frag_pc_scales[0], fragC[0])
                postprocess_store2lds(fragC[0], 0)

                hot_loop_scheduler()
                fx.gpu.barrier()

            # epilogue
            postprocess_store2vmem(nBN - 2, 0)
            gemm_compute(frag_weights[1], frag_pc_scales[1], fragC[1])
            postprocess_store2lds(fragC[1], 1)
            fx.gpu.barrier()
            postprocess_store2vmem(nBN - 1, 1)

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
        moe_2stage_down_splitk(
            p_input,
            p_weight,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            M,
        ).launch(grid=(num_n_blocks, task_num, 1), block=(64, 1, 1), stream=stream)

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
        moe_2stage_down_batch1(
            p_input, p_weight, p_output, p_topk_ids, p_topk_weights, p_w_scale
        ).launch(grid=(num_n_blocks, TOPK, M), block=(64, 1, 1), stream=stream)

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
        moe_2stage_down_prefill_1x4(
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
            value_attrs=None,
        ).launch(grid=(1, task_num, 1), block=(256, 1, 1), stream=stream)

    if const_expr(alg == "prefill_1x4"):
        return launch_prefill_1x4
    if const_expr(alg == "batch1"):
        return launch_batch1
    return launch_splitk
