"""MoE GEMM stage1/stage2 kernel implementations (FLIR MFMA FP8).

This module intentionally contains the **kernel builder code** for:
- `moe_gemm1` (stage1)
- `moe_gemm2` (stage2)

It is extracted from `tests/kernels/test_moe_gemm.py` so that:
- `kernels/` holds the implementation
- `tests/` holds correctness/perf harnesses
"""

import os
import functools

import flydsl
from flydsl.dialects.ext import flir
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator, SmemPtr

from _mlir import ir
import _mlir.extras.types as T
from flydsl.lang.ir.types import T as I

from flydsl.dialects.ext import arith, gpu, buffer_ops, llvm, vector, rocdl, scf, memref

from kernels.mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    lds_load_pack_k32,
    lds_store_4b_xor16,
    lds_store_8b_xor16,
    lds_store_16b_xor16,
    make_preshuffle_b_layout,
    load_b_pack_k32,
    tile_chunk_coord_i32,
)
from kernels.mfma_epilogues import c_shuffle_epilog, default_epilog, mfma_epilog
from kernels.kernels_common import stream_ptr_to_async_token


@functools.lru_cache(maxsize=1024)
def compile_moe_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    # NOTE: aiter swap passes these for API symmetry; stage1 uses dynamic memrefs so they are ignored.
    doweight_stage1: bool,
    in_dtype: str = "fp8",
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
):
    """Compile stage1 kernel (`moe_gemm1`) and return the compiled executable.

    in_dtype:
      - "fp8": X/W are fp8
      - "fp16": X/W are fp16
      - "int8": X/W are int8
      - "int4": W4A8 path: X is int8, W is packed int4 (2 values per byte) unpacked to int8 in-kernel
    """

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    if in_dtype not in ("fp8", "fp16", "int8", "int4"):
        raise ValueError(f"in_dtype must be one of ('fp8','fp16','int8','int4'), got {in_dtype!r}")
    is_f16 = in_dtype == "fp16"
    elem_bytes = 2 if is_f16 else 1
    if out_dtype not in ("f16", "bf16"):
        raise ValueError(f"out_dtype must be 'f16' or 'bf16', got {out_dtype!r}")
    # NOTE: don't materialize MLIR types outside an active MLIR Context.
    out_mlir = lambda: (T.f16() if out_dtype == "f16" else T.bf16())
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
    size_w = (experts * (2 * inter_dim) * model_dim) // 2 if is_int4 else (experts * (2 * inter_dim) * model_dim)
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
    # split into <=16B pieces and use `flir.copy(load-only)` for buffer_load_dwordx4.
    # (Compute the split lens inside the kernel so the code matches GEMM structure.)

    # LDS128 mode (same idea as test_preshuffle_gemm.py):
    # - LDS stride == tile_k (no extra padding) + XOR16 swizzle
    # - Use ds_{read,write}_b128 (16B) and extract 8B halves for MFMA steps
    _ck_lds128 = os.environ.get("FLIR_CK_LDS128", "1") in ("1", "true", "True", "YES", "yes")
    pad_k = 0 if _ck_lds128 else 8
    lds_stride = tile_k + pad_k
    if use_cshuffle_epilog is None:
        use_cshuffle_epilog = os.environ.get("FLIR_MOE_STAGE1_CSHUFFLE", "1") in ("1", "true", "True", "YES", "yes")
    use_cshuffle_epilog = bool(use_cshuffle_epilog)
    if out_dtype != "f16" and use_cshuffle_epilog:
        raise ValueError("stage1 cshuffle epilog currently supports only f16 output (out_dtype='f16')")

    epilog_tag = "cshuffle" if use_cshuffle_epilog else "direct"
    # IMPORTANT: module name participates in FlyDSL's compile cache key.
    # Keep an explicit ABI tag so signature changes can't accidentally reuse an old binary.
    module_name = (
        f"mfma_moe1_{in_dtype}_{out_dtype}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_abi3"  # also mask sentinel token ids on loads (X/scale_x) to avoid illegal address faults
    ).replace("-", "_")

    class _MOE1(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            # Optional epilogue CShuffle (LDS + vectorized buffer stores).
            # Reuse the same LDS bytes for both:
            # - ping-pong X tiles (2 * tile_m * lds_stride bytes; fp8/int8)
            # - epilogue CShuffle tile (tile_m * tile_n f16 -> 2 * tile_m * tile_n bytes)
            _use_cshuffle_epilog = bool(use_cshuffle_epilog)
            lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(elem_bytes)
            lds_out_bytes = 2 * tile_m * tile_n if _use_cshuffle_epilog else 0
            lds_total_bytes = max(lds_x_bytes, lds_out_bytes)
            lds_total_elems = lds_total_bytes if elem_bytes == 1 else (lds_total_bytes // 2)
            x_lds_elem = I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)
            _state["lds_x_decl"] = allocator.allocate_array(x_lds_elem, lds_total_elems)
            allocator.finalize()

        @flir.kernel
        def moe_gemm1(
            self: flir.T.i64,
            arg_out: lambda: T.memref(DYN, out_mlir()),
            arg_x: lambda: T.memref(DYN, I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)),
            arg_w: lambda: T.memref(DYN, I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)),
            arg_scale_x: lambda: T.memref(DYN, T.f32()),
            arg_scale_w: lambda: T.memref(experts * (2 * inter_dim), T.f32()),
            arg_sorted_token_ids: lambda: T.memref(DYN, T.i32()),
            arg_expert_ids: lambda: T.memref(DYN, T.i32()),
            arg_sorted_weights: lambda: T.memref(DYN, T.f32()),
            arg_max_token_ids: lambda: T.memref(DYN, T.i32()),
            tokens_in: lambda: T.index(),
            inter_in: lambda: T.index(),
            k_in: lambda: T.index(),
            size_expert_ids_in: lambda: T.index(),
        ):
            x_elem = I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)
            # For int4, weights are stored as packed bytes (i8) and unpacked to i8 packs.
            w_elem = I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)
            f16 = I.f16
            f32 = I.f32
            i32 = I.i32
            i64 = I.i64
            vec4_f32 = I.vec(4, f32)
            vec4_i32 = I.vec(4, i32)
            vec1_f16 = I.vec(1, f16)
            vec4_f16 = I.vec(4, f16)
            vec16_elems = 16 if elem_bytes == 1 else 8
            vec8_elems = 8 if elem_bytes == 1 else 4
            vec4_elems = 4 if elem_bytes == 1 else 2
            vec8_x = I.vec(vec8_elems, x_elem)
            vec16_x = I.vec(vec16_elems, x_elem)
            vec1_i64 = I.vec(1, i64)
            vec2_i64 = I.vec(2, i64)

            def silu(x):
                # device fast path:
                #   emu = exp(-x)  ~= exp2(log2e * (-x))  -> v_exp_f32
                #   sig = rcp(1 + emu)                   -> v_rcp_f32
                #   y = x * sig
                #
                # Using llvm.amdgcn intrinsics prevents lowering to the div_scale/div_fixup
                # sequences that introduce extra compares/cndmasks.
                t = x * (-1.4426950408889634)  # -log2(e)
                emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                den = 1.0 + emu
                sig = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [den], [], [])
                return x * sig

            acc_init = (
                arith.constant_vector(0, vec4_i32)
                if is_int8
                else arith.constant_vector(0.0, vec4_f32)
            )

            # Layouts
            layout_x = flir.make_layout((tokens_in, k_in), stride=(k_in, 1))

            # B preshuffle layout: match GEMM test helper exactly.
            c_n_total = arith.constant(experts * (2 * inter_dim), index=True)
            kpack_bytes = 8 if is_int4 else 16
            b_layout = make_preshuffle_b_layout(
                flir, arith, c_n=c_n_total, c_k=k_in, kpack_bytes=kpack_bytes, elem_bytes=elem_bytes
            )
            layout_b = b_layout.layout_b
            c_k0 = (k_in * arith.constant(int(elem_bytes), index=True)) / arith.index(64)

            shape_lds = flir.make_shape(tile_m, tile_k)
            stride_lds = flir.make_stride(lds_stride, 1)
            layout_lds = flir.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            # Align with Aiter launch mapping (NSwizzle==false):
            # - blockIdx.x -> N dimension (tile along inter_dim)
            # - blockIdx.y -> expert-block id / M dimension (tile along sorted M)
            by = gpu.block_id("x")  # tile along inter_dim
            bx = gpu.block_id("y")  # tile along sorted M

            # Block validity: compute as early as possible so invalid blocks skip all buffer-resource
            # setup, LDS pointer math, and gmem prefetch work.
            bx_m = bx * arith.constant(tile_m, index=True)
            maxids_rsrc = buffer_ops.create_buffer_resource(
                arg_max_token_ids, max_size=False, num_records_bytes=arith.i32(4)
            )
            max_token_id_i32 = buffer_ops.buffer_load(
                maxids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=i32
            )
            bx_m_i32 = arith.index_cast(i32, bx_m)
            blk_valid = arith.cmpu(bx_m_i32, max_token_id_i32, "ult")
            # Common constants/atoms (hoisted): keep IR small like GEMM.
            # XOR16 swizzle parameter (in bytes; constant, power-of-two in our configs).
            k_blocks16 = arith.constant(tile_k_bytes // 16, index=True)
            atom_x_s16 = flir.make_copy_atom(x_elem, vector_size=vec16_elems)
            atom_x_s8 = flir.make_copy_atom(x_elem, vector_size=8)
            atom_x_s4 = flir.make_copy_atom(x_elem, vector_size=4)
            atom_x_g2r16 = flir.make_copy_atom(x_elem, vector_size=vec16_elems)
            atom_x_g2r8 = flir.make_copy_atom(x_elem, vector_size=vec8_elems)
            atom_x_g2r4 = flir.make_copy_atom(x_elem, vector_size=vec4_elems)
            layout_tx_wave_lane = flir.make_layout((4, 64), stride=(64, 1))
            layout_lane16 = flir.make_layout((4, 16), stride=(16, 1))

            # Everything below is gated by `blk_valid` to avoid doing buffer-resource setup and
            # gmem work for padding blocks.
            _if_blk = scf.IfOp(blk_valid)
            with _if_blk.then():
                base_ptr = allocator.get_base()
                lds_x_ptr = _state["lds_x_decl"](base_ptr)
                lds_x = lds_x_ptr.get()
                # Alias LDS bytes as fp16 for optional CShuffle epilogue.
                _use_cshuffle_epilog = bool(use_cshuffle_epilog)
                lds_out = (
                    SmemPtr(base_ptr, lds_x_ptr.byte_offset, I.f16, shape=(tile_m * tile_n,)).get()
                    if _use_cshuffle_epilog
                    else None
                )

                # Buffer resources: for dynamic memrefs, provide `num_records_bytes` explicitly so
                # hardware OOB behavior is stable (otherwise it falls back to a large max size).
                c_topk = arith.constant(topk, index=True)

                # X: [tokens, k] bytes = tokens*k*elem_bytes
                x_nbytes_idx = tokens_in * k_in * arith.constant(int(elem_bytes), index=True)
                x_nbytes_i32 = arith.index_cast(i32, x_nbytes_idx)
                x_rsrc = buffer_ops.create_buffer_resource(
                    arg_x, max_size=False, num_records_bytes=x_nbytes_i32
                )

                w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)

                # OUT: [tokens, topk, inter] f16/bf16 -> bytes = tokens*topk*inter*out_elem_bytes
                out_elem_bytes = 2  # f16/bf16
                out_nbytes_idx = tokens_in * c_topk * inter_in * arith.constant(out_elem_bytes, index=True)
                out_nbytes_i32 = arith.index_cast(i32, out_nbytes_idx)
                out_rsrc = buffer_ops.create_buffer_resource(
                    arg_out, max_size=False, num_records_bytes=out_nbytes_i32
                )

                # fp16 path ignores scales completely (implicit scale=1.0).
                if is_f16:
                    sx_rsrc = None
                    sw_rsrc = None
                else:
                    # scale_x: [tokens] f32 -> bytes = tokens*4
                    sx_nbytes_idx = tokens_in * arith.constant(4, index=True)
                    sx_nbytes_i32 = arith.index_cast(i32, sx_nbytes_idx)
                    sx_rsrc = buffer_ops.create_buffer_resource(
                        arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_i32
                    )
                    sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)

                sorted_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids, max_size=False)
                sorted_w_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights, max_size=False)

                # expert ids: [blocks] i32 -> bytes = size_expert_ids_in*4
                eid_nbytes_i32 = arith.index_cast(i32, size_expert_ids_in * arith.constant(4, index=True))
                expert_rsrc = buffer_ops.create_buffer_resource(
                    arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes_i32
                )

                # Expert id for this M tile (keep address math in `index`)
                expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=i32)
                expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
                inter2_idx = arith.constant(2 * inter_dim, index=True)
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
    
                c_k_div4 = (k_in * arith.constant(int(elem_bytes), index=True)) / arith.index(4)
                layout_x_div4 = flir.make_layout((tokens_in, c_k_div4), stride=(c_k_div4, 1))
                tile_k_dwords = (int(tile_k) * int(elem_bytes)) // 4
                layout_x_tile_div4 = flir.make_layout((tile_m, tile_k_dwords), stride=(tile_k_dwords, 1))
                c_chunk_i32 = arith.constant(chunk_i32, index=True)
                tx_i32_base = tx * c_chunk_i32
                mask24 = arith.i32(0xFFFFFF)
                # Keep i32 constants available for epilogue index math.
                tokens_i32 = arith.index_cast(i32, tokens_in)
                topk_i32 = arith.i32(topk)
    
                def x_tile_chunk_coord_i32(i: int):
                    return tile_chunk_coord_i32(
                        flir,
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
                    fused_i = buffer_ops.buffer_load(sorted_rsrc, sorted_row_i, vec_width=1, dtype=i32)
                    t_raw = arith.andi(fused_i, mask24)
                    t_idx = arith.index_cast(ir.IndexType.get(), t_raw)
                    # NOTE: aiter moe_sorting uses sentinel token_id == tokens for padding.
                    # Do NOT rely on buffer OOB semantics for X loads; explicitly mask to a safe row.
                    t_valid_i32 = arith.cmpu(t_raw, tokens_i32, "ult")
                    t_safe = arith.select(t_valid_i32, t_idx, arith.index(0))
                    x_row_base_div4.append(t_safe * c_k_div4)
    
                vec1_i32 = I.vec(1, i32)
                vec2_i32 = I.vec(2, i32)
                vec4_i32 = I.vec(4, i32)
                vec4_x = I.vec(4, x_elem)
    
                def load_x(idx_i32):
                    """Load `x_load_bytes` bytes from X (gmem) into regs.
    
                    For 16B, keep the fast dwordx4 path. For 8B/4B, use byte offsets.
                    """
                    if x_load_bytes == 16:
                        idx_elem = idx_i32 if elem_bytes == 1 else (idx_i32 * arith.index(2))
                        return buffer_copy_gmem16_dwordx4(
                            flir,
                            arg=arg_x,
                            elem_type=x_elem,
                            idx_i32=idx_elem,
                            atom_g2r16=atom_x_g2r16,
                            rsrc=x_rsrc,
                            vec_elems=vec16_elems,
                        )
                    idx_bytes = idx_i32 * arith.index(4)
                    atom = atom_x_g2r8 if x_load_bytes == 8 else atom_x_g2r4
                    view = flir.TensorView(
                        arg_x,
                        (x_load_bytes,),
                        strides=(1,),
                        base_indices=(idx_bytes,),
                        element_type=x_elem,
                    )
                    return flir.copy(
                        atom,
                        view,
                        None,
                        alignment=x_load_bytes,
                        return_vector=True,
                        src_buffer_resource=x_rsrc,
                        src_buffer_offset_in_bytes=True,
                    )
    
                def load_x_tile(base_k):
                    """Prefetch the per-thread X tile portion (gmem -> regs) for a given K base (in elements)."""
                    base_k_div4 = (base_k * arith.constant(int(elem_bytes), index=True)) / arith.index(4)
                    parts = []
                    for i in range_constexpr(num_x_loads):
                        idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                        x_vec = load_x(idx_i32)
                        if x_load_bytes == 16:
                            parts.append(vector.bitcast(vec4_i32, x_vec))
                        elif x_load_bytes == 8:
                            parts.append(vector.bitcast(vec2_i32, x_vec))
                        else:
                            parts.append(vector.bitcast(vec1_i32, x_vec))
                    return parts
    
                # tx -> wave/lane (GEMM-style decomposition).
                coord_wl = flir.idx2crd(tx, layout_tx_wave_lane)
                wave_id = flir.get(coord_wl, 0)
                lane_id = flir.get(coord_wl, 1)
                coord_l16 = flir.idx2crd(lane_id, layout_lane16)
                lane_div_16 = flir.get(coord_l16, 0)
                lane_mod_16 = flir.get(coord_l16, 1)
    
                # Match GEMM naming/pattern: row in LDS is lane_mod_16, and col base is lane_div_16*16.
                row_a_lds = lane_mod_16
                col_offset_base = flir.crd2idx(flir.make_coord(lane_div_16, 0), layout_lane16)
                col_offset_base_bytes = (
                    col_offset_base
                    if elem_bytes == 1
                    else (col_offset_base * arith.constant(int(elem_bytes), index=True))
                )
    
                # Dynamic N tiling within block (same as existing kernels)
                by_n = by * arith.constant(tile_n, index=True)
                num_waves = 4
                n_per_wave = tile_n // num_waves
                num_acc_n = n_per_wave // 16
                c_n_per_wave = arith.constant(n_per_wave, index=True)
                wave_mod_4 = wave_id % arith.index(4)
                n_tile_base = wave_mod_4 * c_n_per_wave
    
                # Precompute n_blk/n_intra for gate and up rows (GEMM-style: idx2crd/get)
                n_intra_gate = []
                n_blk_gate = []
                n_intra_up = []
                n_blk_up = []
                col_g_list = []
                inter_idx = arith.constant(inter_dim, index=True)
                # layout for (row -> (blk,intra)) where intra is 0..15
                c_n0 = c_n_total / arith.index(16)
                layout_n_blk_intra = flir.make_layout((c_n0, 16), stride=(16, 1))
                for ni in range_constexpr(num_acc_n):
                    offset = arith.constant(ni * 16, index=True)
                    col_g = by_n + n_tile_base
                    col_g = col_g + offset
                    col_g = col_g + lane_mod_16
                    col_g_list.append(col_g)
    
                    row_gate = expert_off_idx + col_g
                    row_up = row_gate + inter_idx
    
                    coord_gate = flir.idx2crd(row_gate, layout_n_blk_intra)
                    n_blk_gate.append(flir.get(coord_gate, 0))
                    n_intra_gate.append(flir.get(coord_gate, 1))
    
                    coord_up = flir.idx2crd(row_up, layout_n_blk_intra)
                    n_blk_up.append(flir.get(coord_up, 0))
                    n_intra_up.append(flir.get(coord_up, 1))
    
    
                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 64  # K64-byte micro-step (2x MFMA)
    
                # --- B Load Logic (K64) - shared layout with preshuffle GEMM ---
                def load_b_pack(base_k, ki_step, ni, blk_list, intra_list):
                    return load_b_pack_k32(
                        buffer_ops,
                        flir,
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
    
                acc_gate = [acc_init] * (num_acc_n * m_repeat)
                acc_up = [acc_init] * (num_acc_n * m_repeat)
    
                # ---- Pipeline helpers: store X tile to LDS with ping-pong base ----
                def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                    for i in range_constexpr(num_x_loads):
                        row_local = x_row_local[i]
                        col_local_i32 = x_col_local_i32[i]
                        if x_load_bytes == 16:
                            lds_store_16b_xor16(
                                flir,
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec16_ty=vec16_x,
                                elem_type=x_elem,
                                atom_s16=atom_x_s16,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x4=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )
                        elif x_load_bytes == 8:
                            lds_store_8b_xor16(
                                flir,
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec8_ty=vec8_x,
                                elem_type=x_elem,
                                atom_s8=atom_x_s8,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x2=vec_x_in_parts[i],
                            )
                        else:
                            lds_store_4b_xor16(
                                flir,
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec4_ty=vec4_x,
                                elem_type=x_elem,
                                atom_s4=atom_x_s4,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x1=vec_x_in_parts[i],
                            )
    
                # --- A LDS load helper for K64 (load 16B once, extract 2x i64 halves) ---
                def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_base):
                    col_base_swz_bytes = flir.swizzle_xor16(curr_row_a_lds, col_base_bytes, k_blocks16)
                    col_base_swz = (
                        col_base_swz_bytes
                        if elem_bytes == 1
                        else (col_base_swz_bytes / arith.constant(int(elem_bytes), index=True))
                    )
                    coord_a16 = flir.make_coord(curr_row_a_lds, col_base_swz)
                    idx_a16 = flir.crd2idx(coord_a16, layout_lds)
                    idx_a16 = idx_a16 + lds_base
                    loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                    a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                    a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                    a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                    return a0, a1
    
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
                    mfma_res_ty = vec4_i32 if is_int8 else vec4_f32
                    mfma_fn = (
                        mfma_i32_k32
                        if is_int8
                        else (rocdl.mfma_f32_16x16x16f16 if is_f16 else rocdl.mfma_f32_16x16x32_fp8_fp8)
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
                                arith.f32(1.0)
                                if is_f16
                                else buffer_ops.buffer_load(sw_rsrc, row_gate_idx, vec_width=1, dtype=f32)
                            )
                            sw_up_pf.append(
                                arith.f32(1.0)
                                if is_f16
                                else buffer_ops.buffer_load(sw_rsrc, row_up_idx, vec_width=1, dtype=f32)
                            )
                        epilogue_pf = (sw_gate_pf, sw_up_pf)
    
                    def _i64_to_v4f16(x_i64):
                        v1 = vector.from_elements(vec1_i64, [x_i64])
                        return vector.bitcast(vec4_f16, v1)

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
                        ki64 = arith.constant(ku * 64, index=True)
                        col_base = col_offset_base_bytes + ki64
    
                        for mi in range_constexpr(m_repeat):
                            mi_val = arith.constant(mi * 16, index=True)
                            curr_row_a_lds = row_a_lds + mi_val
    
                            if (a0_prefetch is not None) and (ku == 0) and (mi == 0):
                                a0, a1 = a0_prefetch
                            else:
                                a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base, lds_base)
    
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
    
                # ---------------- 2-stage pipeline (ping-pong LDS + B tile prefetch) ----------------
                lds_tile_elems = arith.constant(tile_m * lds_stride, index=True)
                lds_base_cur = arith.index(0)
                lds_base_nxt = lds_tile_elems
    
                # Optional scheduler hints (copied from tuned GEMM); can be disabled via env.
                rocdl.sched_barrier(0)
    
                def hot_loop_scheduler():
                    mfma_group = num_acc_n * 2
                    # K64 micro-step: 2x K32 MFMA per gemm.
                    mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                    mfma_per_iter = 2 * mfma_group
                    sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
    
                    rocdl.sched_dsrd(2)
                    rocdl.sched_mfma(2)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
    
                    # DS-write hints near the end: match total X LDS-store micro-ops per thread.
                    dswr_tail = num_x_loads
                    if dswr_tail > sche_iters:
                        dswr_tail = sche_iters
                    dswr_start = sche_iters - dswr_tail
                    for sche_i in range_constexpr(sche_iters):
                        rocdl.sched_vmem(1)
                        rocdl.sched_mfma(mfma_group)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(mfma_group)
                        if sche_i >= dswr_start - 1:
                            rocdl.sched_dswr(1)
                    rocdl.sched_barrier(0)
    
                # Prologue: prefetch tile0, store to LDS(cur), sync.
                k0 = arith.index(0)
                x_regs0 = load_x_tile(k0)
                b_gate_cur = load_b_tile(k0, n_blk_gate, n_intra_gate)
                b_up_cur = load_b_tile(k0, n_blk_up, n_intra_up)
                store_x_tile_to_lds(x_regs0, lds_base_cur)
                gpu.barrier()
    
                # Loop-carried ping/pong state.
                lds_base_pong = lds_base_cur  # current/compute
                lds_base_ping = lds_base_nxt  # next/load+store
    
                # Cross-tile A0 LDS prefetch (default-on): prefetch the first A-pack (K64) for the
                # tile we are about to compute from LDS, to overlap with upcoming VMEM.
                a0_prefetch_pong = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_pong)
    
                # Unrolled ping-pong main loop (2 tiles per iteration), leaving 2 tail tiles.
                c2_tile_k = arith.constant(tile_k * 2, index=True)
                c_k_main2 = k_in - c2_tile_k
    
                for k_iv in range(arith.index(0), c_k_main2, c2_tile_k):
                    # ---- stage 0: prefetch+store ping, compute pong ----
                    next_k1 = k_iv + tile_k
                    x_regs_ping = load_x_tile(next_k1)
                    b_gate_ping = load_b_tile(next_k1, n_blk_gate, n_intra_gate)
                    b_up_ping = load_b_tile(next_k1, n_blk_up, n_intra_up)
    
                    acc_gate, acc_up, _ = compute_tile(
                        acc_gate,
                        acc_up,
                        b_gate_cur,
                        b_up_cur,
                        lds_base_pong,
                        a0_prefetch=a0_prefetch_pong,
                    )
                    a0_prefetch_pong = None
                    store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    hot_loop_scheduler()
                    gpu.barrier()
    
                    # Cross-tile prefetch for the ping tile we are about to compute.
                    a0_prefetch_ping = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_ping)
    
                    # ---- stage 1: prefetch+store pong, compute ping ----
                    next_k2 = k_iv + c2_tile_k
                    x_regs_pong = load_x_tile(next_k2)
                    b_gate_next = load_b_tile(next_k2, n_blk_gate, n_intra_gate)
                    b_up_next = load_b_tile(next_k2, n_blk_up, n_intra_up)
    
                    acc_gate, acc_up, _ = compute_tile(
                        acc_gate,
                        acc_up,
                        b_gate_ping,
                        b_up_ping,
                        lds_base_ping,
                        a0_prefetch=a0_prefetch_ping,
                    )
                    a0_prefetch_ping = None
                    store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                    hot_loop_scheduler()
                    gpu.barrier()
    
                    # Cross-tile prefetch for the next pong tile.
                    a0_prefetch_pong = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_pong)
    
                    # Advance pong state to next_k2 for next iteration.
                    b_gate_cur = b_gate_next
                    b_up_cur = b_up_next
    
                # Tail: 2 remaining tiles at (k_in - 2*tile_k) and (k_in - tile_k).
                k_tail1 = k_in - tile_k
                x_regs_ping = load_x_tile(k_tail1)
                b_gate_ping = load_b_tile(k_tail1, n_blk_gate, n_intra_gate)
                b_up_ping = load_b_tile(k_tail1, n_blk_up, n_intra_up)
    
                acc_gate, acc_up, _ = compute_tile(
                    acc_gate,
                    acc_up,
                    b_gate_cur,
                    b_up_cur,
                    lds_base_pong,
                    a0_prefetch=a0_prefetch_pong,
                )
                a0_prefetch_pong = None
                store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                hot_loop_scheduler()
                gpu.barrier()
    
                # Cross-tile prefetch for the final ping tile.
                a0_prefetch_ping = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_ping)
    
                # Epilogue: compute last tile with epilogue scale prefetch to overlap loads with MFMA.
                acc_gate, acc_up, epilogue_pf = compute_tile(
                    acc_gate,
                    acc_up,
                    b_gate_ping,
                    b_up_ping,
                    lds_base_ping,
                    prefetch_epilogue=True,
                    a0_prefetch=a0_prefetch_ping,
                )
    
                # Store epilogue to out[t, slot, inter]
                expert_off = expert_off_idx
                bx_m0 = bx_m
                tokens_i32_v = tokens_i32
                topk_i32_v = topk_i32
                inter_i32_v = arith.i32(inter_dim)
                mask24_i32 = arith.i32(0xFFFFFF)
    
                if epilogue_pf is not None:
                    sw_gate_vals, sw_up_vals = epilogue_pf
                else:
                    sw_gate_vals = []
                    sw_up_vals = []
                    for ni in range_constexpr(num_acc_n):
                        col_g = col_g_list[ni]
                        row_gate_idx = expert_off + col_g
                        row_up_idx = row_gate_idx + inter_idx
                        sw_gate_vals.append(
                            arith.f32(1.0)
                            if is_f16
                            else buffer_ops.buffer_load(sw_rsrc, row_gate_idx, vec_width=1, dtype=f32)
                        )
                        sw_up_vals.append(
                            arith.f32(1.0)
                            if is_f16
                            else buffer_ops.buffer_load(sw_rsrc, row_up_idx, vec_width=1, dtype=f32)
                        )
    
                # Epilogue hoists to keep IR + Python build time small:
                col_i32_list = []
                for ni in range_constexpr(num_acc_n):
                    col_i32_list.append(arith.index_cast(i32, col_g_list[ni]))
    
                lane_div_16_mul4 = lane_div_16 * arith.index(4)
                inter_i32_local = inter_i32_v
    
                # Uses EVec=4 (buffer store "x4" of fp16 elements).
                _use_cshuffle_epilog = bool(use_cshuffle_epilog)
    
                if _use_cshuffle_epilog:
                    if lds_out is None:
                        raise RuntimeError("CShuffle epilogue enabled but lds_out is not allocated/aliased.")
    
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
                        # `row` is the sorted-row index (bx_m + row_in_tile).
                        fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=i32)
                        t2 = fused2 & mask24_i32
                        # aiter moe_sorting uses sentinel token_id == tokens for padding.
                        # Do NOT rely on buffer OOB semantics for scale loads; explicitly mask.
                        t_valid = arith.cmpu(t2, tokens_i32_v, "ult")
                        sx = (
                            arith.f32(1.0)
                            if is_f16
                            else arith.select(
                                t_valid,
                                buffer_ops.buffer_load(sx_rsrc, t2, vec_width=1, dtype=f32),
                                arith.f32(0.0),
                            )
                        )
    
                        # Sorted weight aligned with `row` (matches aiter moe_sorting output).
                        if doweight_stage1:
                            tw = buffer_ops.buffer_load(sorted_w_rsrc, row, vec_width=1, dtype=f32)
    
                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            sw_gate = sw_gate_vals[ni]
                            sw_up = sw_up_vals[ni]

                            acc_idx = mi * num_acc_n + ni
                            vg = vector.extract(
                                acc_gate[acc_idx], static_position=[ii], dynamic_position=[]
                            )
                            vu = vector.extract(
                                acc_up[acc_idx], static_position=[ii], dynamic_position=[]
                            )

                            if is_int8:
                                vg = arith.sitofp(f32, vg)
                                vu = arith.sitofp(f32, vu)
                            vg = vg * sx * sw_gate
                            vu = vu * sx * sw_up

                            y = silu(vg) * vu
                            if doweight_stage1:
                                y = y * tw
                            y16 = arith.trunc_f(T.f16(), y)
    
                            lds_idx = row_base_lds + col_local
                            v1 = vector.from_elements(vec1_f16, [y16])
                            vector.store(v1, lds_out, [lds_idx], alignment=2)
    
                    def precompute_row(*, row_local, row):
                        fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=i32)
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24
                        return (t2 * topk_i32_v + s2) * inter_i32_local
    
                    def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                        # Guard against sentinel token ids (t == tokens) produced by aiter moe_sorting padding.
                        # OOB buffer stores are not guaranteed to be safe on all paths, so predicate explicitly.
                        fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=i32)
                        t2 = fused2 & mask24_i32
                        t_valid = arith.cmpu(t2, tokens_i32_v, "ult")
                        _if_valid = scf.IfOp(t_valid)
                        with _if_valid.then():
                            idx0 = row_ctx
                            col_i32 = arith.index_cast(i32, col_g0)
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
                    # `row` is the sorted-row index (bx_m + row_in_tile).
                    # Block-level early-exit already guards `bx_m` range.
                    # Here we rely on buffer OOB semantics for any tail rows.
                    fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=i32)
                    t2_raw = fused2 & mask24_i32
                    s2_raw = fused2 >> 24
                    t2 = t2_raw
                    s2 = s2_raw
                    t_valid = arith.cmpu(t2, tokens_i32_v, "ult")

                    # Do NOT rely on buffer OOB semantics for scale loads; explicitly mask.
                    sx0 = (
                        arith.f32(1.0)
                        if is_f16
                        else arith.select(
                            t_valid,
                            buffer_ops.buffer_load(sx_rsrc, t2, vec_width=1, dtype=f32),
                            arith.f32(0.0),
                        )
                    )
                    sx = sx0
                    zero_out = arith.constant(0.0, type=out_mlir())
    
                    # out linear index base = ((t*topk + s)*inter_dim) (invariant across ni)
                    idx0 = (t2 * topk_i32_v + s2) * inter_i32_local
    
                    # Sorted weight aligned with `row` (matches aiter moe_sorting output).
                    if doweight_stage1:
                        tw = buffer_ops.buffer_load(sorted_w_rsrc, row, vec_width=1, dtype=f32)
    
                    _if_valid = scf.IfOp(t_valid)
                    with _if_valid.then():
                        for ni in range_constexpr(num_acc_n):
                            col_i32 = col_i32_list[ni]
                            sw_gate = sw_gate_vals[ni]
                            sw_up = sw_up_vals[ni]
        
                            acc_idx = mi * num_acc_n + ni
                            vg = vector.extract(
                                acc_gate[acc_idx], static_position=[ii], dynamic_position=[]
                            )
                            vu = vector.extract(
                                acc_up[acc_idx], static_position=[ii], dynamic_position=[]
                            )
        
                            if is_int8:
                                vg = arith.sitofp(f32, vg)
                                vu = arith.sitofp(f32, vu)
                            vg = vg * sx * sw_gate
                            vu = vu * sx * sw_up
        
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
    
        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_out: lambda: T.memref(DYN, out_mlir()),
            arg_x: lambda: T.memref(DYN, I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)),
            arg_w: lambda: T.memref(DYN, I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)),
            arg_scale_x: lambda: T.memref(DYN, T.f32()),
            arg_scale_w: lambda: T.memref(experts * (2 * inter_dim), T.f32()),
            arg_sorted_token_ids: lambda: T.memref(DYN, T.i32()),
            arg_expert_ids: lambda: T.memref(DYN, T.i32()),
            arg_sorted_weights: lambda: T.memref(DYN, T.f32()),
            arg_max_token_ids: lambda: T.memref(DYN, T.i32()),
            tokens_in: lambda: T.index(),
            inter_in: lambda: T.index(),
            k_in: lambda: T.index(),
            size_expert_ids_in: lambda: T.index(),
        ):
            bdx = 256
            gx = inter_in / arith.index(tile_n)
            # Use host-provided upper bound for M blocks (same as aiter moe_sorting allocation).
            # This avoids device->host sync on num_valid_ids.
            gy = size_expert_ids_in
            flir.gpu_ext.LaunchFuncOp(
                [module_name, "moe_gemm1"],
                grid_size=(gx, gy, 1),
                block_size=(bdx, 1, 1),
                kernel_operands=[
                    arg_out,
                    arg_x,
                    arg_w,
                    arg_scale_x,
                    arg_scale_w,
                    arg_sorted_token_ids,
                    arg_expert_ids,
                    arg_sorted_weights,
                    arg_max_token_ids,
                    tokens_in,
                    inter_in,
                    k_in,
                    size_expert_ids_in,
                ],
            )

    m = _MOE1()
    exe = flydsl.compile(m)
    return exe


@functools.lru_cache(maxsize=1024)
def compile_moe_gemm2(
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
    # Optional experiment: write per-(token,slot) output (no atomics) into an output shaped
    # [tokens*topk, model_dim] (or [tokens, topk, model_dim] flattened), then reduce over topk outside.
    # This can reduce atomic contention for small tokens at the cost of extra bandwidth / reduction.
    accumulate: bool = True,
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
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    if in_dtype not in ("fp8", "fp16", "int8", "int4"):
        raise ValueError(f"in_dtype must be one of ('fp8','fp16','int8','int4'), got {in_dtype!r}")
    is_f16 = in_dtype == "fp16"
    elem_bytes = 2 if is_f16 else 1
    out_s = str(out_dtype).strip().lower()
    if out_s not in ("f16", "fp16", "half", "bf16", "bfloat16", "f32", "fp32", "float"):
        raise ValueError(f"out_dtype must be 'f16', 'bf16', or 'f32', got {out_dtype!r}")
    out_is_f32 = out_s in ("f32", "fp32", "float")
    out_is_bf16 = out_s in ("bf16", "bfloat16")
    if (not bool(accumulate)) and out_is_f32:
        raise ValueError("compile_moe_gemm2(accumulate=False) only supports out_dtype in {'f16','bf16'}")
    is_int4 = in_dtype == "int4"
    # INT4 here means W4A8: A2 is int8, W is packed int4 and unpacked to int8 in-kernel.
    is_int8 = (in_dtype == "int8") or is_int4

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
    size_w = (experts * model_dim * inter_dim) // 2 if is_int4 else (experts * model_dim * inter_dim)

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

    _ck_lds128 = os.environ.get("FLIR_CK_LDS128", "1") in ("1", "true", "True", "YES", "yes")
    pad_k = 0 if _ck_lds128 else 8
    lds_stride = tile_k + pad_k
    if out_is_bf16:
        if not (gpu_arch.startswith("gfx942") or gpu_arch.startswith("gfx950") or gpu_arch.startswith("gfx12")):
            raise ValueError(f"out_dtype='bf16' requires bf16 global atomics (gfx942/gfx950/gfx12), got arch={gpu_arch!r}")

    if out_is_f32:
        # Match origin/dev_a16w4: f32 output uses scalar atomics and does NOT use the CShuffle epilogue.
        _use_cshuffle_epilog = False if use_cshuffle_epilog is None else bool(use_cshuffle_epilog)
        if _use_cshuffle_epilog:
            raise ValueError("out_dtype='f32' does not support CShuffle epilogue (set use_cshuffle_epilog=False).")
    else:
        if use_cshuffle_epilog is None:
            _use_cshuffle_epilog = os.environ.get("FLIR_MOE_STAGE2_CSHUFFLE", "1") in (
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
                "stage2 f16 output currently requires CShuffle epilogue (FLIR_MOE_STAGE2_CSHUFFLE=1)."
            )

    # NOTE: Keep this as a callable so we don't require an MLIR Context at Python-time.
    out_elem = (T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16))
    epilog_tag = "cshuffle"
    # IMPORTANT: include tiling in the module name to avoid accidentally reusing a compiled
    # binary for a different (tile_m, tile_n, tile_k) configuration.
    # See stage1 note: include ABI tag to prevent binary reuse across signature changes.
    # IMPORTANT: module name participates in FlyDSL's compile cache key.
    # Dynamic-shape variant: safe to reuse across (tokens/sorted_size/size_expert_ids) at runtime.
    # Keep a distinct ABI tag so the compile cache never mixes with historical signatures.
    module_name = (
        f"mfma_moe2_{in_dtype}_{out_s}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_abi2"  # mask sentinel token ids on loads/stores to avoid illegal address faults
    ).replace("-", "_")

    class _MOE2(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            # LDS scratch: reuse a single allocation for both
            # - ping-pong A2 tiles (2 * tile_m * lds_stride * elem_bytes bytes)
            # - epilogue CShuffle tile (tile_m * tile_n f16 -> 2 * tile_m * tile_n bytes)
            #
            # This reduces LDS usage from sum(...) to max(...).
            lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(elem_bytes)
            lds_out_bytes = 2 * tile_m * tile_n if _use_cshuffle_epilog else 0  # f16 bytes
            lds_total_bytes = max(lds_x_bytes, lds_out_bytes)
            lds_total_elems = lds_total_bytes if elem_bytes == 1 else (lds_total_bytes // 2)
            x_lds_elem = I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)
            _state["lds_x_decl"] = allocator.allocate_array(x_lds_elem, lds_total_elems)
            allocator.finalize()

        @flir.kernel
        def moe_gemm2(
            self: flir.T.i64,
            arg_out: lambda: T.memref(size_out, out_elem()),
            arg_x: lambda: T.memref(size_x, I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)),
            arg_w: lambda: T.memref(size_w, I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)),
            arg_scale_x: lambda: T.memref(size_scale_x, T.f32()),
            arg_scale_w: lambda: T.memref(experts * model_dim, T.f32()),
            arg_sorted_token_ids: lambda: T.memref(size_sorted, T.i32()),
            arg_expert_ids: lambda: T.memref(size_expert_ids_shape, T.i32()),
            arg_sorted_weights: lambda: T.memref(size_sorted, T.f32()),
            arg_num_valid_ids: lambda: T.memref(DYN, T.i32()),
            tokens_in: lambda: T.index(),
            n_in: lambda: T.index(),
            k_in: lambda: T.index(),
            size_expert_ids_in: lambda: T.index(),
        ):
            x_elem = I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)
            # For int4, weights are stored as packed bytes (i8) and unpacked to i8 packs.
            w_elem = I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)
            f16 = I.f16
            f32 = I.f32
            i32 = I.i32
            i64 = I.i64
            vec4_f32 = I.vec(4, f32)
            vec4_i32 = I.vec(4, i32)
            vec1_f16 = I.vec(1, f16)
            vec2_f16 = I.vec(2, f16)
            vec4_f16 = I.vec(4, f16)
            vec16_elems = 16 if elem_bytes == 1 else 8
            vec8_elems = 8 if elem_bytes == 1 else 4
            vec4_elems = 4 if elem_bytes == 1 else 2
            vec8_x = I.vec(vec8_elems, x_elem)
            vec16_x = I.vec(vec16_elems, x_elem)
            vec1_i64 = I.vec(1, i64)
            vec2_i64 = I.vec(2, i64)

            acc_init = (
                arith.constant_vector(0, vec4_i32)
                if is_int8
                else arith.constant_vector(0.0, vec4_f32)
            )

            # A2 layout (flatten token-slot -> M).
            topk_idx = arith.constant(topk, index=True)
            m_in = tokens_in * topk_idx
            layout_x = flir.make_layout((m_in, k_in), stride=(k_in, 1))

            # B preshuffle layout: [experts*model_dim, inter_dim]
            c_n_total = arith.constant(experts * model_dim, index=True)
            kpack_bytes = 8 if is_int4 else 16
            b_layout = make_preshuffle_b_layout(
                flir, arith, c_n=c_n_total, c_k=k_in, kpack_bytes=kpack_bytes, elem_bytes=elem_bytes
            )
            layout_b = b_layout.layout_b
            c_k0 = (k_in * arith.constant(int(elem_bytes), index=True)) / arith.index(64)

            shape_lds = flir.make_shape(tile_m, tile_k)
            stride_lds = flir.make_stride(lds_stride, 1)
            layout_lds = flir.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            # Align with Aiter launch mapping:
            # - blockIdx.x -> N dimension (tile along model_dim)
            # - blockIdx.y -> expert-block id / M dimension (tile along sorted M)
            by = gpu.block_id("x")  # tile along model_dim
            bx = gpu.block_id("y")  # tile along sorted M

            # XOR16 swizzle parameter (in bytes; constant, power-of-two in our configs).
            k_blocks16 = arith.constant(tile_k_bytes // 16, index=True)
            atom_x_s16 = flir.make_copy_atom(x_elem, vector_size=vec16_elems)
            atom_x_s8 = flir.make_copy_atom(x_elem, vector_size=8)
            atom_x_s4 = flir.make_copy_atom(x_elem, vector_size=4)
            atom_x_g2r16 = flir.make_copy_atom(x_elem, vector_size=vec16_elems)
            atom_x_g2r8 = flir.make_copy_atom(x_elem, vector_size=vec8_elems)
            atom_x_g2r4 = flir.make_copy_atom(x_elem, vector_size=vec4_elems)
            layout_tx_wave_lane = flir.make_layout((4, 64), stride=(64, 1))
            layout_lane16 = flir.make_layout((4, 16), stride=(16, 1))
            layout_lin_rowcol = flir.make_layout((tile_m, tile_k), stride=(tile_k, 1))

            base_ptr = allocator.get_base()
            lds_x_ptr = _state["lds_x_decl"](base_ptr)
            lds_x = lds_x_ptr.get()
            # Alias the same underlying LDS bytes as f16/bf16 for epilogue shuffle.
            lds_out = (
                SmemPtr(
                    base_ptr,
                    lds_x_ptr.byte_offset,
                    (I.bf16 if out_is_bf16 else I.f16),
                    shape=(tile_m * tile_n,),
                ).get()
                if _use_cshuffle_epilog
                else None
            )

            # Buffer resources.
            # For dynamic memrefs, `max_size=False` cannot infer the logical size from the memref *type*,
            # so we should pass `num_records_bytes` explicitly for stable hardware OOB behavior.
            c_topk = arith.constant(topk, index=True)

            # X(A2): [tokens*topk, inter_dim] bytes = tokens*topk*k*elem_bytes
            x_nbytes_idx = (tokens_in * c_topk) * k_in * arith.constant(int(elem_bytes), index=True)
            x_nbytes_i32 = arith.index_cast(i32, x_nbytes_idx)
            x_rsrc = buffer_ops.create_buffer_resource(
                arg_x, max_size=False, num_records_bytes=x_nbytes_i32
            )

            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)

            # OUT: [tokens, model_dim] -> clamp to descriptor max (i32 bytes) to avoid overflow on huge tokens.
            out_elem_bytes = 4 if out_is_f32 else 2
            out_nbytes_idx = tokens_in * n_in * arith.constant(out_elem_bytes, index=True)
            if not bool(accumulate):
                out_nbytes_idx = (
                    tokens_in
                    * arith.index(topk)
                    * n_in
                    * arith.constant(out_elem_bytes, index=True)
                )
            out_nbytes_i32 = arith.index_cast(i32, out_nbytes_idx)
            out_rsrc = buffer_ops.create_buffer_resource(
                arg_out, max_size=False, num_records_bytes=out_nbytes_i32
            )
            # fp16 path ignores scales completely (implicit scale=1.0).
            if is_f16:
                sx_rsrc = None
                sw_rsrc = None
            else:
                # scale_x (A2 scale): [tokens*topk] f32 -> bytes = tokens*topk*4
                sx_nbytes_idx = (tokens_in * c_topk) * arith.constant(4, index=True)
                sx_nbytes_i32 = arith.index_cast(i32, sx_nbytes_idx)
                sx_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_i32
                )
                # scale_w: [experts*model_dim] f32 (static shape in practice)
                sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)

            # sorted_token_ids / sorted_weights: [blocks*tile_m] (CK-style padded length)
            sorted_nbytes_idx = (
                size_expert_ids_in
                * arith.constant(tile_m, index=True)
                * arith.constant(4, index=True)
            )
            sorted_nbytes_i32 = arith.index_cast(i32, sorted_nbytes_idx)
            sorted_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_token_ids, max_size=False, num_records_bytes=sorted_nbytes_i32
            )
            sorted_w_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_weights, max_size=False, num_records_bytes=sorted_nbytes_i32
            )

            # expert ids: [blocks] i32 -> bytes = size_expert_ids_in*4
            eid_nbytes_idx = size_expert_ids_in * arith.constant(4, index=True)
            eid_nbytes_i32 = arith.index_cast(i32, eid_nbytes_idx)
            expert_rsrc = buffer_ops.create_buffer_resource(
                arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes_i32
            )
            bx_m = bx * arith.constant(tile_m, index=True)

            # Early-exit guard (as in 2ce65fb): some routing paths can produce extra/garbage
            # expert blocks beyond `num_valid_ids`. Skip those blocks entirely to avoid OOB.
            numids_rsrc = buffer_ops.create_buffer_resource(
                arg_num_valid_ids, max_size=False, num_records_bytes=arith.i32(4)
            )
            num_valid_i32 = buffer_ops.buffer_load(
                numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=i32
            )
            bx_m_i32 = arith.index_cast(i32, bx_m)
            blk_valid = arith.cmpu(bx_m_i32, num_valid_i32, "ult")

            def _moe_gemm2_then_body():
                # Expert id for this M tile.
                expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=i32)
                expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
                n_idx = arith.constant(model_dim, index=True)
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
                vec4_i32 = I.vec(4, i32)
    
                c_k_div4 = (k_in * arith.constant(int(elem_bytes), index=True)) / arith.index(4)
                layout_x_div4 = flir.make_layout((m_in, c_k_div4), stride=(c_k_div4, 1))
                tile_k_dwords = (int(tile_k) * int(elem_bytes)) // 4
                layout_x_tile_div4 = flir.make_layout((tile_m, tile_k_dwords), stride=(tile_k_dwords, 1))
                c_chunk_i32 = arith.constant(chunk_i32, index=True)
                tx_i32_base = tx * c_chunk_i32
    
                topk_i32 = arith.i32(topk)
                mask24 = arith.i32(0xFFFFFF)
                # Sentinel clamp uses `tokens` as the upper bound: t_valid = (t < tokens).
                tokens_i32 = arith.index_cast(i32, tokens_in)
    
                def x_tile_chunk_coord_i32(i: int):
                    return tile_chunk_coord_i32(
                        flir,
                        arith,
                        tx_i32_base=tx_i32_base,
                        i=i,
                        total_threads=total_threads,
                        layout_tile_div4=layout_x_tile_div4,
                        chunk_i32=chunk_i32,
                    )
    
                vec1_i32 = I.vec(1, i32)
                vec2_i32 = I.vec(2, i32)
                vec4_x = I.vec(4, x_elem)
    
                def load_x(idx_i32):
                    if x_load_bytes == 16:
                        idx_elem = idx_i32 if elem_bytes == 1 else (idx_i32 * arith.index(2))
                        return buffer_copy_gmem16_dwordx4(
                            flir,
                            arg=arg_x,
                            elem_type=x_elem,
                            idx_i32=idx_elem,
                            atom_g2r16=atom_x_g2r16,
                            rsrc=x_rsrc,
                            vec_elems=vec16_elems,
                        )
                    idx_bytes = idx_i32 * arith.index(4)
                    atom = atom_x_g2r8 if x_load_bytes == 8 else atom_x_g2r4
                    view = flir.TensorView(
                        arg_x,
                        (x_load_bytes,),
                        strides=(1,),
                        base_indices=(idx_bytes,),
                        element_type=x_elem,
                    )
                    return flir.copy(
                        atom,
                        view,
                        None,
                        alignment=x_load_bytes,
                        return_vector=True,
                        src_buffer_resource=x_rsrc,
                        src_buffer_offset_in_bytes=True,
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
                    fused_i = buffer_ops.buffer_load(sorted_rsrc, sorted_row_i, vec_width=1, dtype=i32)
                    t_i32 = arith.andi(fused_i, mask24)
                    s_i32 = arith.shrui(fused_i, arith.i32(24))
                    # aiter moe_sorting uses sentinel token_id == tokens for padding.
                    # Do NOT rely on buffer OOB semantics for A2/scale loads; explicitly mask.
                    t_valid = arith.cmpu(t_i32, tokens_i32, "ult")
                    s_valid = arith.cmpu(s_i32, topk_i32, "ult")
                    ts_valid = arith.andi(t_valid, s_valid)
                    t_safe = arith.select(ts_valid, t_i32, arith.i32(0))
                    s_safe = arith.select(ts_valid, s_i32, arith.i32(0))
                    row_ts_i32 = t_safe * topk_i32 + s_safe
                    row_ts_idx = arith.index_cast(ir.IndexType.get(), row_ts_i32)
                    # Base row offset in dword units: row_ts_idx * (k_in/4)
                    x_row_base_div4.append(row_ts_idx * c_k_div4)
    
                def load_x_tile(base_k):
                    base_k_div4 = (base_k * arith.constant(int(elem_bytes), index=True)) / arith.index(4)
                    parts = []
                    for i in range_constexpr(num_x_loads):
                        idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                        x_vec = load_x(idx_i32)
                        if x_load_bytes == 16:
                            parts.append(vector.bitcast(vec4_i32, x_vec))
                        elif x_load_bytes == 8:
                            parts.append(vector.bitcast(vec2_i32, x_vec))
                        else:
                            parts.append(vector.bitcast(vec1_i32, x_vec))
                    return parts
    
                # tx -> wave/lane (GEMM-style decomposition).
                coord_wl = flir.idx2crd(tx, layout_tx_wave_lane)
                wave_id = flir.get(coord_wl, 0)
                lane_id = flir.get(coord_wl, 1)
                coord_l16 = flir.idx2crd(lane_id, layout_lane16)
                lane_div_16 = flir.get(coord_l16, 0)
                lane_mod_16 = flir.get(coord_l16, 1)
    
                row_a_lds = lane_mod_16
                col_offset_base = flir.crd2idx(flir.make_coord(lane_div_16, 0), layout_lane16)
                col_offset_base_bytes = (
                    col_offset_base
                    if elem_bytes == 1
                    else (col_offset_base * arith.constant(int(elem_bytes), index=True))
                )
    
                # Dynamic N tiling within block.
                by_n = by * arith.constant(tile_n, index=True)
                num_waves = 4
                n_per_wave = tile_n // num_waves
                num_acc_n = n_per_wave // 16
                c_n_per_wave = arith.constant(n_per_wave, index=True)
                wave_mod_4 = wave_id % arith.constant(4, index=True)
                n_tile_base = wave_mod_4 * c_n_per_wave
    
                # Precompute (n_blk, n_intra) for B, and col indices for output.
                n_intra_list = []
                n_blk_list = []
                col_g_list = []
                c_n0 = c_n_total / arith.index(16)
                layout_n_blk_intra = flir.make_layout((c_n0, 16), stride=(16, 1))
                for ni in range_constexpr(num_acc_n):
                    offset = arith.constant(ni * 16, index=True)
                    col_g = by_n + n_tile_base + offset + lane_mod_16
                    col_g_list.append(col_g)
    
                    row_w = expert_off_idx + col_g
                    coord_w = flir.idx2crd(row_w, layout_n_blk_intra)
                    n_blk_list.append(flir.get(coord_w, 0))
                    n_intra_list.append(flir.get(coord_w, 1))
    
                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 64  # K64-byte micro-step (2x MFMA)
    
                # --- B Load Logic (K64) ---
                def load_b_pack(base_k, ki_step, ni):
                    return load_b_pack_k32(
                        buffer_ops,
                        flir,
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
                                flir,
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec16_ty=vec16_x,
                                elem_type=x_elem,
                                atom_s16=atom_x_s16,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x4=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )
                        elif x_load_bytes == 8:
                            lds_store_8b_xor16(
                                flir,
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec8_ty=vec8_x,
                                elem_type=x_elem,
                                atom_s8=atom_x_s8,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x2=vec_x_in_parts[i],
                            )
                        else:
                            lds_store_4b_xor16(
                                flir,
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec4_ty=vec4_x,
                                elem_type=x_elem,
                                atom_s4=atom_x_s4,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x1=vec_x_in_parts[i],
                            )
    
                # --- A LDS load helper for K64 (load 16B once, extract 2x i64 halves) ---
                def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_base):
                    col_base_swz_bytes = flir.swizzle_xor16(curr_row_a_lds, col_base_bytes, k_blocks16)
                    col_base_swz = (
                        col_base_swz_bytes
                        if elem_bytes == 1
                        else (col_base_swz_bytes / arith.constant(int(elem_bytes), index=True))
                    )
                    coord_a16 = flir.make_coord(curr_row_a_lds, col_base_swz)
                    idx_a16 = flir.crd2idx(coord_a16, layout_lds)
                    idx_a16 = idx_a16 + lds_base
                    loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                    a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                    a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                    a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                    return a0, a1
    
                def compute_tile(acc_in, b_tile_in, lds_base, *, prefetch_epilogue: bool = False, a0_prefetch=None):
                    acc_list = list(acc_in)
                    mfma_res_ty = vec4_i32 if is_int8 else vec4_f32
                    mfma_fn = (
                        mfma_i32_k32
                        if is_int8
                        else (rocdl.mfma_f32_16x16x16f16 if is_f16 else rocdl.mfma_f32_16x16x32_fp8_fp8)
                    )
    
                    epilogue_pf = None
                    if prefetch_epilogue:
                        expert_off_pf = expert_off_idx
                        sw_pf = []
                        for ni in range_constexpr(num_acc_n):
                            col_g = col_g_list[ni]
                            row_w_idx = expert_off_pf + col_g
                            sw_pf.append(
                                arith.f32(1.0)
                                if is_f16
                                else buffer_ops.buffer_load(sw_rsrc, row_w_idx, vec_width=1, dtype=f32)
                            )
                        # Also prefetch per-row routed/topk weights (sorted_weights) when enabled.
                        tw_pf = None
                        if doweight_stage2:
                            tw_pf = []
                            lane_div_16_mul4_pf = lane_div_16 * arith.index(4)
                            ii_idx_list_pf = [arith.constant(ii, index=True) for ii in range(4)]
                            for mi in range_constexpr(m_repeat):
                                mi_base_pf = arith.constant(mi * 16, index=True)
                                for ii in range_constexpr(4):
                                    row_off_pf = lane_div_16_mul4_pf + ii_idx_list_pf[ii]
                                    row_in_tile_pf = mi_base_pf + row_off_pf
                                    sorted_row_pf = bx_m + row_in_tile_pf
                                    tw_pf.append(
                                        buffer_ops.buffer_load(
                                            sorted_w_rsrc, sorted_row_pf, vec_width=1, dtype=f32
                                        )
                                    )
                        epilogue_pf = (sw_pf, tw_pf)
    
                    def _i64_to_v4f16(x_i64):
                        v1 = vector.from_elements(vec1_i64, [x_i64])
                        return vector.bitcast(vec4_f16, v1)

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
                        ki64 = arith.constant(ku * 64, index=True)
                        col_base = col_offset_base_bytes + ki64
    
                        for mi in range_constexpr(m_repeat):
                            mi_val = arith.constant(mi * 16, index=True)
                            curr_row_a_lds = row_a_lds + mi_val
    
                            if (a0_prefetch is not None) and (ku == 0) and (mi == 0):
                                a0, a1 = a0_prefetch
                            else:
                                a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base, lds_base)
    
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
                lds_tile_elems = arith.constant(tile_m * lds_stride, index=True)
                lds_base_cur = arith.index(0)
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
                    # - MFMA group size per "slot": num_acc_n
                    # - Total MFMA per tile: (2*K32 per K64) * k_unroll * m_repeat * num_acc_n
                    # - We emit (mfma_group + dsrd + mfma_group) per scheduler iteration.
                    mfma_group = num_acc_n
                    mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                    mfma_per_iter = 2 * mfma_group
                    sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
    
                    rocdl.sched_dsrd(2)
                    rocdl.sched_mfma(1)
                    if tile_m == 16:
                        rocdl.sched_vmem(1)
                    rocdl.sched_mfma(1)
                    if tile_m == 16:
                        rocdl.sched_vmem(1)
                    if num_acc_n < 4:
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                        if tile_m == 16:
                            rocdl.sched_vmem(1)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                        if tile_m == 16:
                            rocdl.sched_vmem(1)
                        rocdl.sched_mfma(1)
    
                    # DS-write hints near the end: match total A LDS-store micro-ops per thread.
                    dswr_tail = num_x_loads
                    if dswr_tail > sche_iters:
                        dswr_tail = sche_iters
                    dswr_start = sche_iters - dswr_tail
    
                    for sche_i in range_constexpr(sche_iters):
                        rocdl.sched_vmem(1)
                        rocdl.sched_mfma(mfma_group)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(mfma_group)
                        if sche_i >= dswr_start - 1:
                            rocdl.sched_dswr(1)
    
                    rocdl.sched_barrier(0)
                # Prologue.
                k0 = arith.index(0)
                x_regs0 = load_x_tile(k0)
                b_cur = load_b_tile(k0)
                store_x_tile_to_lds(x_regs0, lds_base_cur)
                gpu.barrier()
    
                acc = [acc_init] * (num_acc_n * m_repeat)
                lds_base_pong = lds_base_cur
                lds_base_ping = lds_base_nxt
    
                # Cross-tile A0 LDS prefetch (default-on): prefetch the first A-pack (K64) for the
                # tile we are about to compute from LDS, to overlap with upcoming VMEM.
                a0_prefetch_pong = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_pong)
    
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
    
                c2_tile_k = arith.constant(tile_k * 2, index=True)
                c_k_main2 = arith.index(k_main2_py)
                for k_iv in range(arith.index(0), c_k_main2, arith.index(tile_k * 2)):
                    next_k1 = k_iv + tile_k
                    x_regs_ping = load_x_tile(next_k1)
                    b_ping = load_b_tile(next_k1)
    
                    acc, _ = compute_tile(acc, b_cur, lds_base_pong, a0_prefetch=a0_prefetch_pong)
                    a0_prefetch_pong = None
                    store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    hot_loop_scheduler()
                    gpu.barrier()
    
                    # Cross-tile prefetch for the ping tile we are about to compute.
                    a0_prefetch_ping = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_ping)
    
                    next_k2 = k_iv + c2_tile_k
                    x_regs_pong = load_x_tile(next_k2)
                    b_next = load_b_tile(next_k2)
    
                    acc, _ = compute_tile(acc, b_ping, lds_base_ping, a0_prefetch=a0_prefetch_ping)
                    a0_prefetch_ping = None
                    store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                    hot_loop_scheduler()
                    gpu.barrier()
    
                    # Cross-tile prefetch for the next pong tile.
                    a0_prefetch_pong = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_pong)
    
                    b_cur = b_next
    
                if odd_k_tiles:
                    # Tail: single remaining tile (already in `b_cur` / `lds_base_pong`).
                    acc, epilogue_pf = compute_tile(
                        acc,
                        b_cur,
                        lds_base_pong,
                        prefetch_epilogue=True,
                        a0_prefetch=a0_prefetch_pong,
                    )
                else:
                    # Tail: 2 remaining tiles.
                    k_tail1 = k_in - tile_k
                    x_regs_ping = load_x_tile(k_tail1)
                    b_ping = load_b_tile(k_tail1)
    
                    acc, _ = compute_tile(acc, b_cur, lds_base_pong, a0_prefetch=a0_prefetch_pong)
                    a0_prefetch_pong = None
                    store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    hot_loop_scheduler()
                    gpu.barrier()
    
                    # Epilogue tile with sw prefetch.
                    a0_prefetch_ping = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_base_ping)
                    acc, epilogue_pf = compute_tile(
                        acc, b_ping, lds_base_ping, prefetch_epilogue=True, a0_prefetch=a0_prefetch_ping
                    )
    
                # ---------------- Epilogue: LDS CShuffle + atomic half2 (x2) ----------------
                # Reuse the shared helper so GEMM / MoE kernels share the exact same CShuffle skeleton.
                expert_off = expert_off_idx
                mask24_i32 = arith.i32(0xFFFFFF)
                model_i32 = arith.i32(model_dim)
                topk_i32_v = topk_i32
    
                zero_i32 = arith.i32(0)
                c2_i32 = arith.i32(2)  # 2B element size for f16/bf16
                mask_even_i32 = arith.i32(0xFFFFFFFE)  # align element index to even for half2 atomics
    
                def atomic_add_f16x2(val_f16x2, byte_off_i32):
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        val_f16x2,
                        out_rsrc,
                        byte_off_i32,
                        zero_i32,
                        zero_i32,
                    )

    
                sw_pf = None
                tw_pf = None
                if epilogue_pf is not None:
                    sw_pf, tw_pf = epilogue_pf
    
                # Weight scales for the N tile (col_g depends on lane/wave/by but not on (t,s)).
                if sw_pf is not None:
                    sw_vals = sw_pf
                else:
                    sw_vals = []
                    for ni in range_constexpr(num_acc_n):
                        col_g = col_g_list[ni]
                        row_w_idx = expert_off + col_g
                        sw_vals.append(
                            arith.f32(1.0)
                            if is_f16
                            else buffer_ops.buffer_load(sw_rsrc, row_w_idx, vec_width=1, dtype=f32)
                        )
    
                if out_is_f32:
                    # origin/dev_a16w4: f32 output uses scalar f32 atomics and skips CShuffle/LDS.
                    c4_i32 = arith.i32(4)

                    def atomic_add_f32(val_f32, byte_off_i32):
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            val_f32,
                            out_rsrc,
                            byte_off_i32,
                            zero_i32,
                            zero_i32,
                        )

                    def _stage2_row_atomic(*, mi: int, ii: int, row_in_tile, row):
                        fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=i32)
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24

                        ts2 = t2 * topk_i32_v + s2
                        sx = arith.f32(1.0) if is_f16 else buffer_ops.buffer_load(sx_rsrc, ts2, vec_width=1, dtype=f32)

                        if doweight_stage2:
                            tw_idx = (mi * 4) + ii
                            if tw_pf is not None:
                                tw = tw_pf[tw_idx]
                            else:
                                tw = buffer_ops.buffer_load(sorted_w_rsrc, row, vec_width=1, dtype=f32)

                        idx0 = t2 * model_i32  # i32 element index base

                        for ni in range_constexpr(num_acc_n):
                            col_g = col_g_list[ni]
                            sw = sw_vals[ni]
                            acc_idx = mi * num_acc_n + ni
                            v = vector.extract(acc[acc_idx], static_position=[ii], dynamic_position=[])
                            if is_int8:
                                v = arith.sitofp(f32, v)
                            v = v * sx * sw
                            if doweight_stage2:
                                v = v * tw
                            col_i32 = arith.index_cast(i32, col_g)
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
                            "FLIR_MOE_STAGE2_CSHUFFLE=1 but lds_out is not allocated/aliased."
                        )

                    # For bf16 global atomics, precompute the output base address once.
                    # (We still need an inttoptr per atomic unless we can rely on GEPOp; keep the
                    # stable path here.)
                    out_base_idx = None
                    if out_is_bf16:
                        out_base_idx = memref.extract_aligned_pointer_as_index(arg_out)

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
                        fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=i32)
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24
                        # Explicitly mask sentinel token/slot to avoid OOB scale_x loads.
                        t_ok = arith.cmpu(t2, tokens_i32, "ult")
                        s_ok = arith.cmpu(s2, topk_i32_v, "ult")
                        ts_ok = arith.andi(t_ok, s_ok)
                        t2_safe = arith.select(ts_ok, t2, arith.i32(0))
                        s2_safe = arith.select(ts_ok, s2, arith.i32(0))
                        ts2 = t2_safe * topk_i32_v + s2_safe
                        sx = (
                            arith.f32(1.0)
                            if is_f16
                            else arith.select(
                                ts_ok,
                                buffer_ops.buffer_load(sx_rsrc, ts2, vec_width=1, dtype=f32),
                                arith.f32(0.0),
                            )
                        )

                        if doweight_stage2:
                            tw_idx = (mi * 4) + ii
                            if tw_pf is not None:
                                tw = tw_pf[tw_idx]
                            else:
                                tw = buffer_ops.buffer_load(
                                    sorted_w_rsrc, row, vec_width=1, dtype=f32
                                )

                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            sw = sw_vals[ni]
                            acc_idx = mi * num_acc_n + ni
                            v = vector.extract(acc[acc_idx], static_position=[ii], dynamic_position=[])
                            if is_int8:
                                v = arith.sitofp(f32, v)
                            v = v * sx * sw
                            if doweight_stage2:
                                v = v * tw
                            v_out = arith.trunc_f(out_elem(), v)

                            lds_idx = row_base_lds + col_local
                            vec1_out = I.vec(1, out_elem())
                            v1 = vector.from_elements(vec1_out, [v_out])
                            vector.store(v1, lds_out, [lds_idx], alignment=2)

                    def precompute_row(*, row_local, row):
                        # Precompute row context for cshuffle stores.
                        # Return (fused_i32, row_valid_i1) so the epilogue can skip the entire row
                        # for invalid tail rows (CK-style), avoiding per-store branching.
                        fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=i32)
                        row_i32 = arith.index_cast(i32, row)
                        row_valid0 = arith.cmpu(row_i32, num_valid_i32, "ult")
                        t = fused2 & mask24_i32
                        s = fused2 >> 24
                        t_ok = arith.cmpu(t, tokens_i32, "ult")
                        s_ok = arith.cmpu(s, topk_i32_v, "ult")
                        row_valid = arith.andi(row_valid0, arith.andi(t_ok, s_ok))
                        return (fused2, row_valid)

                    def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                        fused = row_ctx
                        t = fused & mask24_i32
                        s = fused >> 24
                        idx0 = t * model_i32
                        if not bool(accumulate):
                            ts = t * topk_i32_v + s
                            idx0 = ts * model_i32
                        col_i32 = arith.index_cast(i32, col_g0)
                        idx_elem = idx0 + col_i32
                        idx_elem_even = idx_elem & mask_even_i32
                        if out_is_bf16:
                            if bool(accumulate):
                                # Use global atomicrmw fadd on <2 x bf16> (CK path).
                                # Row-valid gating is handled at the row level by c_shuffle_epilog via `precompute_row`.
                                byte_off = idx_elem_even * c2_i32
                                byte_off_idx = arith.index_cast(ir.IndexType.get(), byte_off)
                                ptr_addr_idx = out_base_idx + byte_off_idx
                                out_ptr = buffer_ops.create_llvm_ptr(ptr_addr_idx, address_space=1)
                                out_ptr_v = out_ptr._value if hasattr(out_ptr, "_value") else out_ptr
                                frag_v = frag._value if hasattr(frag, "_value") else frag
                                llvm.AtomicRMWOp(
                                    llvm.AtomicBinOp.fadd,
                                    out_ptr_v,
                                    frag_v,
                                    llvm.AtomicOrdering.monotonic,
                                    syncscope="agent",
                                    alignment=4,
                                )
                            else:
                                # Scatter store (no atomic): store bf16x2 directly via buffer store.
                                buffer_ops.buffer_store(frag, out_rsrc, idx_elem_even)
                        else:
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
                        e_vec=2,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=(ir.BF16Type.get() if out_is_bf16 else ir.F16Type.get()),
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                    )
    
            _if_blk = scf.IfOp(blk_valid)
            with _if_blk.then():
                _moe_gemm2_then_body()
        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_out: lambda: T.memref(size_out, out_elem()),
            arg_x: lambda: T.memref(size_x, I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)),
            arg_w: lambda: T.memref(size_w, I.f16 if is_f16 else (I.i8 if is_int8 else I.f8)),
            arg_scale_x: lambda: T.memref(size_scale_x, T.f32()),
            arg_scale_w: lambda: T.memref(experts * model_dim, T.f32()),
            arg_sorted_token_ids: lambda: T.memref(size_sorted, T.i32()),
            arg_expert_ids: lambda: T.memref(size_expert_ids_shape, T.i32()),
            arg_sorted_weights: lambda: T.memref(size_sorted, T.f32()),
            arg_num_valid_ids: lambda: T.memref(DYN, T.i32()),
            tokens_in: lambda: T.index(),
            n_in: lambda: T.index(),
            k_in: lambda: T.index(),
            size_expert_ids_in: lambda: T.index(),
        ):
            bdx = 256
            gx = n_in / arith.index(tile_n)
            gy = size_expert_ids_in
            flir.gpu_ext.LaunchFuncOp(
                [module_name, "moe_gemm2"],
                grid_size=(gx, gy, 1),
                block_size=(bdx, 1, 1),
                kernel_operands=[
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
                ],
            )

    m = _MOE2()
    exe = flydsl.compile(m)
    return exe


# MoE Reduction Kernel (reduce sum over topk dimension)
@functools.lru_cache(maxsize=1024)
def compile_moe_reduction(
    *,
    topk: int,
    model_dim: int,
    dtype_str: str = "f16",
):
    """Compile a reduction kernel that sums over the topk dimension.

    Input:  X [tokens, topk, model_dim]
    Output: Y [tokens, model_dim]

    This kernel performs: Y[t, d] = sum(X[t, :, d]) for all t, d.
    Used in conjunction with compile_moe_gemm2(accumulate=False) to avoid atomic contention.
    """
    gpu_arch = get_hip_arch()
    DYN = ir.ShapedType.get_dynamic_size()

    # Kernel Config
    BLOCK_SIZE = 256
    VEC_WIDTH = 8
    USE_NONTEMPORAL = True
    VEC_ALIGN = 16

    _state = {}

    class _MoeReduction(flir.MlirModule):
        GPU_MODULE_NAME = f"moe_reduction_{topk}_{model_dim}_{dtype_str}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            if dtype_str == "f32":
                elem_type = T.f32()
            elif dtype_str == "f16":
                elem_type = T.f16()
            elif dtype_str == "bf16":
                elem_type = ir.BF16Type.get()
            else:
                raise ValueError(f"Unsupported dtype: {dtype_str}")

            compute_type = T.f32()
            _state["elem_type"] = elem_type
            _state["compute_type"] = compute_type

        @flir.kernel
        def moe_reduction_kernel(
            self: flir.T.i64,
            X: lambda: T.memref(DYN, topk, model_dim, _state["elem_type"]),
            Y: lambda: T.memref(DYN, model_dim, _state["elem_type"]),
            m_tokens: lambda: T.index(),
        ):
            from _mlir.dialects import vector as mlir_vector

            token_idx = flir.const_index(flir.block_idx("x"))
            tid = flir.const_index(flir.thread_idx("x"))

            elem_type = _state["elem_type"]
            compute_type = _state["compute_type"]

            tensor_X = flir.make_tensor(X, shape=(m_tokens, topk, model_dim), strides=(topk * model_dim, model_dim, 1))
            tensor_Y = flir.make_tensor(Y, shape=(m_tokens, model_dim), strides=(model_dim, 1))

            c0_idx = flir.const_index(0)
            tile_cols = BLOCK_SIZE * VEC_WIDTH
            gX = flir.zipped_divide(tensor_X, (1, 1, tile_cols))
            gY = flir.zipped_divide(tensor_Y, (1, tile_cols))

            copy_atom_load = flir.make_copy_atom(elem_type, vector_size=VEC_WIDTH)
            copy_atom_store = flir.make_copy_atom(elem_type, vector_size=VEC_WIDTH)
            tiled_copy_X = flir.make_tiled_copy_tv(
                copy_atom_load,
                flir.make_ordered_layout((1, 1, BLOCK_SIZE), order=(2, 1, 0)),
                flir.make_ordered_layout((1, 1, VEC_WIDTH), order=(2, 1, 0)),
                thr_shape=(1, 1, BLOCK_SIZE),
                val_shape=(1, 1, VEC_WIDTH),
            )
            tiled_copy_Y = flir.make_tiled_copy_tv(
                copy_atom_store,
                flir.make_ordered_layout((1, BLOCK_SIZE), order=(1, 0)),
                flir.make_ordered_layout((1, VEC_WIDTH), order=(1, 0)),
                thr_shape=(1, BLOCK_SIZE),
                val_shape=(1, VEC_WIDTH),
            )

            thr_copy_X = tiled_copy_X.get_slice(tid)
            thr_copy_Y = tiled_copy_Y.get_slice(tid)

            step = BLOCK_SIZE * VEC_WIDTH
            thread_offset_base = (arith.ArithValue(tid) * VEC_WIDTH).value

            vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
            vec_type_f32 = ir.VectorType.get([VEC_WIDTH], compute_type)

            c_model_dim = arith.index(model_dim).value

            for base_idx_int in range_constexpr(0, model_dim, step):
                c_base = arith.index(base_idx_int)
                curr_idx = (c_base + thread_offset_base).value

                # Bounds check
                c_end_idx = (arith.ArithValue(curr_idx) + arith.index(VEC_WIDTH)).value
                in_bounds = arith.CmpIOp(arith.CmpIPredicate.sle, c_end_idx, c_model_dim)

                tile_i = base_idx_int // tile_cols

                _if_in_bounds = scf.IfOp(arith.as_value(in_bounds))
                with _if_in_bounds.then():
                    c_zero_f32 = arith.constant(0.0, type=compute_type)
                    acc = mlir_vector.splat(vec_type_f32, arith.as_value(c_zero_f32))

                    for k in range_constexpr(topk):
                        blkX = gX[(token_idx, k, tile_i)]
                        thrX = thr_copy_X.partition_S(blkX)
                        frgX = flir.make_fragment_like(thrX, elem_type)
                        flir.copy(
                            tiled_copy_X,
                            thrX,
                            frgX,
                            nontemporal=USE_NONTEMPORAL,
                            alignment=VEC_ALIGN,
                        )
                        vec_val_e = mlir_vector.load(vec_type_e, frgX.memref, [c0_idx, c0_idx, c0_idx], alignment=VEC_ALIGN)
                        if dtype_str in ("f16", "bf16"):
                            vec_val = flir.arith.extf(vec_type_f32, arith.as_value(vec_val_e))
                        else:
                            vec_val = vec_val_e
                        acc = (arith.ArithValue(acc) + arith.ArithValue(vec_val)).value

                    if dtype_str in ("f16", "bf16"):
                        out_vec = flir.arith.truncf(vec_type_e, arith.as_value(acc))
                    else:
                        out_vec = acc

                    blkY = gY[(token_idx, tile_i)]
                    thrY = thr_copy_Y.partition_S(blkY)
                    frgY = flir.make_fragment_like(thrY, elem_type)
                    mlir_vector.store(arith.as_value(out_vec), frgY.memref, [c0_idx, c0_idx], alignment=VEC_ALIGN)
                    flir.copy(
                        tiled_copy_Y,
                        frgY,
                        thrY,
                        nontemporal=USE_NONTEMPORAL,
                        alignment=VEC_ALIGN,
                    )

        @flir.jit
        def __call__(
            self: flir.T.i64,
            X: lambda: T.memref(DYN, topk, model_dim, _state["elem_type"]),
            Y: lambda: T.memref(DYN, model_dim, _state["elem_type"]),
            m_tokens: lambda: T.index(),
            stream_ptr: lambda: T.i64(),  # PyTorch stream pointer
        ):
            from flydsl.dialects.ext import arith as arith_ext
            c1 = arith.as_value(arith_ext.index(1))
            gx = arith.as_value(m_tokens)
            bx = arith.as_value(arith_ext.index(BLOCK_SIZE))
            stream_token = stream_ptr_to_async_token(stream_ptr)
            flir.gpu_ext.LaunchFuncOp(
                [f"moe_reduction_{topk}_{model_dim}_{dtype_str}", "moe_reduction_kernel"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[X, Y, m_tokens],
                async_dependencies=[stream_token],
            )

    m = _MoeReduction()
    exe = flydsl.compile(m)
    return exe


# MoE GEMM2 Execution Modes
class MoeGemm2Mode:
    """Execution mode for MoE GEMM2."""
    ATOMIC = "atomic"       # Use atomic accumulation (default)
    REDUCE = "reduce"       # Use non-atomic write + reduce kernel
    AUTO = "auto"           # Auto-select based on shape heuristics


# Default reduce mode rules: list of (topk, model_dim, min_tokens, max_tokens) tuples
# Only exact (topk, model_dim) matches with tokens in [min_tokens, max_tokens] enable reduce mode
_DEFAULT_REDUCE_RULES = [
    # (topk, model_dim, min_tokens, max_tokens)
    (8, 7168, 16384, 32768),
]


def _should_use_reduce_mode(
    tokens: int,
    topk: int,
    model_dim: int,
    rules: list | None = None,
) -> bool:
    """Determine if reduce mode should be used based on shape matching rules.

    For specific (topk, model_dim) combinations with tokens in a certain range,
    atomic contention becomes significant. In these cases, using non-atomic writes
    followed by a reduce kernel is faster.

    Args:
        tokens: Number of tokens (must be in [min_tokens, max_tokens] of a rule)
        topk: Top-k value (must exactly match a rule)
        model_dim: Model dimension (must exactly match a rule)
        rules: Custom rules list, each entry is (topk, model_dim, min_tokens, max_tokens).
               If None, uses _DEFAULT_REDUCE_RULES.

    Returns:
        True if reduce mode should be used, False otherwise.
    """
    if rules is None:
        rules = _DEFAULT_REDUCE_RULES

    for rule_topk, rule_model_dim, min_tokens, max_tokens in rules:
        if (
            topk == rule_topk and
            model_dim == rule_model_dim and
            min_tokens <= tokens <= max_tokens
        ):
            return True
    return False


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
    ):
        self._gemm2_exe = gemm2_exe
        self._reduce_exe = reduce_exe
        self._topk = topk
        self._model_dim = model_dim
        self._out_dtype_str = out_dtype_str
        
        # Lazy-allocated intermediate buffer
        self._intermediate = None
        self._intermediate_capacity = 0

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

    def _ensure_intermediate_buffer(self, tokens: int, device):
        """Ensure intermediate buffer is large enough."""
        import torch
        required_size = tokens * self._topk * self._model_dim
        if self._intermediate is None or self._intermediate_capacity < required_size:
            self._intermediate = torch.empty(
                tokens * self._topk, self._model_dim,
                device=device,
                dtype=self._get_torch_dtype()
            )
            self._intermediate_capacity = required_size
        return self._intermediate[:tokens * self._topk, :self._model_dim]

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
    ):
        """Execute GEMM2 + reduce.

        Args match moe_gemm2 kernel signature (see compile_moe_gemm2).
        """
        # Allocate/reuse intermediate buffer
        intermediate = self._ensure_intermediate_buffer(tokens_in, arg_out.device)
        # Phase 1: GEMM2 (no atomics) -> [tokens*topk, model_dim]
        self._gemm2_exe(
            intermediate.view(-1),
            arg_x, arg_w, arg_scale_x, arg_scale_w,
            arg_sorted_token_ids, arg_expert_ids, arg_sorted_weights,
            arg_num_valid_ids, tokens_in, n_in, k_in, size_expert_ids_in,
        )
        # Phase 2: Reduce over topk -> [tokens, model_dim]
        X = intermediate.view(tokens_in, self._topk, self._model_dim)
        Y = arg_out.view(tokens_in, self._model_dim)
        self._reduce_exe(X, Y, tokens_in)

    @property
    def mode(self) -> str:
        """Return the execution mode."""
        return MoeGemm2Mode.REDUCE


def compile_moe_gemm2_ex(
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
    mode: str = MoeGemm2Mode.AUTO,
    tokens_hint: int | None = None,
    reduce_rules: list | None = None,
):
    """Compile MoE GEMM2 kernel with optional reduction.

    This is the extended interface that supports automatic mode selection
    or explicit mode control for testing.

    Args:
        mode: Execution mode selection:
            - "auto": Automatically select based on shape matching rules (requires tokens_hint)
            - "atomic": Force atomic accumulation (original behavior)
            - "reduce": Force non-atomic + reduce kernel
        tokens_hint: Expected token count for auto mode selection.
            Only used when mode="auto".
        reduce_rules: Custom rules for auto mode selection.
            List of (topk, model_dim, min_tokens, max_tokens) tuples.
            Reduce mode is enabled only when (topk, model_dim) exactly matches
            a rule and tokens is in [min_tokens, max_tokens].
            Default rule: (8, 7168, 16384, 32768)

    Returns:
        Compiled executable (either wrapped or raw depending on mode).
    """
    # Determine actual mode
    actual_mode = mode
    if mode == MoeGemm2Mode.AUTO:
        # TODO: Autotuning-based kernel selection
        # The current implementation uses static rule matching to select between
        # atomic and reduce modes. A more robust approach would be to perform
        # runtime autotuning during compilation.
        if tokens_hint is not None and _should_use_reduce_mode(
            tokens_hint, topk, model_dim, reduce_rules
        ):
            actual_mode = MoeGemm2Mode.REDUCE
        else:
            actual_mode = MoeGemm2Mode.ATOMIC

    # Compile based on mode
    if actual_mode == MoeGemm2Mode.REDUCE:
        # Compile GEMM2 with accumulate=False
        gemm2_exe = compile_moe_gemm2(
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
        # Compile reduction kernel
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
        )
        return _MoeGemm2ReduceWrapper(
            gemm2_exe=gemm2_exe,
            reduce_exe=reduce_exe,
            topk=topk,
            model_dim=model_dim,
            out_dtype_str=dtype_str,
        )
    else:
        # Compile GEMM2 with accumulate=True (atomic mode)
        return compile_moe_gemm2(
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
