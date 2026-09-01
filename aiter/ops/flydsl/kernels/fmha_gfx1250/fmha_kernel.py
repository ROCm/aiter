"""
Forward Kernel -- gfx1250, Unified FMHA Implementation.
Target: gfx1250 (MI450), wave32, 4 waves per TG (1TG), 1024 shared VGPRs.
Causal mask always on. num_tiles = bx + 1 (triangular).
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, rocdl
from flydsl.expr.primitive import const_expr
from flydsl.expr.typing import T

from ..tensor_shim import _run_compiled
from .fmha_utils import *  # constants, classes, prologue helpers


def compile_fmha_fwd(*, is_causal: bool = False, return_lse: bool = False):
    """Compile FMHA kernel variant. Cached per (is_causal, return_lse)."""
    IS_CAUSAL = int(is_causal)
    RETURN_LSE = int(return_lse)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fmha_fwd_kernel(
        ptr_O: fx.Tensor,
        ptr_Q: fx.Tensor,
        ptr_K: fx.Tensor,
        ptr_V: fx.Tensor,
        ptr_LSE: fx.Tensor,
        ptr_cu_seqlens_q: fx.Tensor,
        ptr_cu_seqlens_k: fx.Tensor,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        gqa: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
    ):
        """D128 BF16 FMHA Forward -- full kernel with dynamic KV loop."""
        mlir_types = get_types()
        setreg(2074, 2)  # WAVE_SCHED_MODE = 2
        rocdl.s_nop(0)
        thread_id = fx.Int32(fx.thread_idx.x)
        lane_id = thread_id & 31
        wave_id = thread_id >> 5

        # ── XCD remap ──
        raw_block_x = fx.Int32(fx.block_idx.x)
        raw_block_y = fx.Int32(fx.block_idx.y)
        raw_block_z = fx.Int32(fx.block_idx.z)
        grid_dim_x = fx.Int32(fx.grid_dim.x)
        grid_dim_y = fx.Int32(fx.grid_dim.y)
        grid_dim_z = fx.Int32(fx.grid_dim.z)
        bz, bx, by = xcd_remap(
            raw_block_x, raw_block_y, raw_block_z, grid_dim_x, grid_dim_y, grid_dim_z
        )
        m_start = bx * TILE_N

        # ── Load seqlens ──
        q_start_tok = load_scalar_from_tensor(ptr_cu_seqlens_q, bz)
        q_end_tok = load_scalar_from_tensor(ptr_cu_seqlens_q, bz + 1)
        k_start_tok = load_scalar_from_tensor(ptr_cu_seqlens_k, bz)
        k_end_tok = load_scalar_from_tensor(ptr_cu_seqlens_k, bz + 1)
        actual_q_len = q_end_tok - q_start_tok
        actual_kv_len = k_end_tok - k_start_tok

        # ── OOB descriptors for TDM K/V loads and D store ──
        k_stride_elems, v_stride_elems = stride_k_seq >> 1, stride_v_seq >> 1
        k_tdm_cfg, v_tdm_cfg = (1 << 16) | K_TDM_CONFIG, (1 << 16) | V_TDM_CONFIG
        k_oob_dg1 = TDM.build_oob_dg1_list(
            k_tdm_cfg, QK_HDIM, k_stride_elems, actual_kv_len, wave_id, dim0_stride=200
        )
        v_oob_dg1 = TDM.build_oob_dg1_list(
            v_tdm_cfg, 128, v_stride_elems, actual_kv_len, wave_id
        )
        q_remain_o = arith.maxsi(actual_q_len - m_start, fx.Int32(0).ir_value())
        o_oob_dim1 = TDM.per_warp_oob_dim1(q_remain_o, wave_id, 32)

        # ── Zero-fill output when KV is empty (seqlen_k == 0) ──
        wg_valid = m_start < actual_q_len
        if wg_valid & (actual_kv_len == 0):
            tid_z = wave_id * WAVE_SIZE + lane_id
            q_tok_z = q_start_tok + m_start + tid_z
            if tid_z < q_remain_o:
                o_addr_z = ptr_base_i64(ptr_O) + fx.Int64(
                    (by * stride_o_head + q_tok_z * stride_o_seq) * 2
                )
                for chunk_z in fx.range_constexpr(V_HDIM // 8):
                    llvm_dialect.store(
                        fx.constant_vector(0, T.vec(4, T.i32)),
                        llvm_dialect.inttoptr(
                            glb_ptr_ty(), o_addr_z + fx.Int64(chunk_z * 16)
                        ),
                    )
                if const_expr(RETURN_LSE):
                    lse_addr_z = ptr_base_i64(ptr_LSE) + fx.Int64(
                        (q_tok_z * grid_dim_z + by) * 4
                    )
                    llvm_dialect.store(
                        fx.Float32(float("-inf")).ir_value(),
                        llvm_dialect.inttoptr(glb_ptr_ty(), lse_addr_z),
                    )

        if wg_valid & (actual_kv_len > 0):
            # ── Prologue: Q load + address setup ──
            q_frags = prologue_q_load_and_rearrange(
                lane_id,
                wave_id,
                ptr_Q,
                stride_q_seq,
                by,
                stride_q_head,
                q_start_tok,
                q_end_tok,
                bx,
            )
            head_index = head_index_div(by, gqa)
            k_offset = k_start_tok * stride_k_seq + head_index * stride_k_head
            v_offset = k_start_tok * stride_v_seq + head_index * stride_v_head
            k_lds_base_a = extract_lds_base_i32(lds_alloc_k_a.get_base())
            k_lds_base_b = extract_lds_base_i32(lds_alloc_k_b.get_base())
            v_lds_base_a = extract_lds_base_i32(lds_alloc_v_a.get_base())
            v_lds_base_b = extract_lds_base_i32(lds_alloc_v_b.get_base())
            rocdl.sched_barrier(0)
            kv_lds_addrs_a = build_kv_lds_addrs(lane_id, k_lds_base_a, v_lds_base_a)
            kv_lds_addrs_b = build_kv_lds_addrs(lane_id, k_lds_base_b, v_lds_base_b)
            stride_k_32, stride_v_32 = stride_k_seq * 32, stride_v_seq * 32
            scale = (fx.Float32(LOG2_E) * softmax_scale).ir_value()
            sgpr_state = {
                "s_log2e_scl": scale,
                "s_log2e_scl_pair": vector.broadcast(T.vec(2, T.f32), scale),
            }
            ctx = {
                "mlir_types": mlir_types,
                "lane_id": lane_id,
                "wave_id": wave_id,
                "m_start": m_start,
                "bx": bx,
                "by": by,
                "grid_dim_z": grid_dim_z,
                "ptr_K": ptr_K,
                "ptr_V": ptr_V,
                "ptr_O": ptr_O,
                "ptr_LSE": ptr_LSE,
                "softmax_scale": softmax_scale,
                "stride_k_seq": stride_k_seq,
                "stride_v_seq": stride_v_seq,
                "stride_o_seq": stride_o_seq,
                "stride_o_head": stride_o_head,
                "stride_k_32": stride_k_32,
                "stride_v_32": stride_v_32,
                "k_offset": k_offset,
                "v_offset": v_offset,
                "actual_kv_len": actual_kv_len,
                "actual_q_len": actual_q_len,
                "q_start_tok": q_start_tok,
                "o_oob_dim1": o_oob_dim1,
                "q_frags": q_frags,
                "sgpr_state": sgpr_state,
                "RETURN_LSE": RETURN_LSE,
            }
            # ── Prologue: Tile 0 QK + softmax ──
            (
                softmax_state_prologue,
                sp_pairs_prologue,
                all_su_sp_tiles,
                causal_offset,
                zero_v8f32,
            ) = prologue_tile0(
                ctx,
                mlir_types,
                q_frags,
                kv_lds_addrs_a,
                k_lds_base_a,
                v_lds_base_a,
                k_oob_dg1,
                v_oob_dg1,
                IS_CAUSAL,
                sgpr_state,
            )
            ctx["zero_v8f32"] = zero_v8f32

            # ── Core loop setup: tile counts + K prefetch + init args ──
            (
                init_args,
                num_tiles,
                num_tiles_idx,
                num_tiles_minus1_idx,
                first_causal_tile_idx,
            ) = core_loop_setup(
                ctx,
                ptr_K,
                stride_k_32,
                kv_lds_addrs_b,
                k_lds_base_b,
                v_lds_base_a,
                k_lds_base_a,
                v_lds_base_b,
                softmax_state_prologue,
                sp_pairs_prologue,
                all_su_sp_tiles,
                causal_offset,
                IS_CAUSAL,
                k_tdm_cfg,
                k_stride_elems,
            )

            # ── Main KV Loop: non-causal tiles ──
            noncausal_loop_results = init_args
            for tile_idx, iter_args in range(
                1, first_causal_tile_idx, 1, init=init_args
            ):
                noncausal_loop_results = yield tile_iteration(ctx, tile_idx, iter_args)

            # ── Main KV Loop: causal tiles ──
            loop_results = noncausal_loop_results
            for tile_idx, iter_args in range(
                first_causal_tile_idx,
                num_tiles_minus1_idx,
                1,
                init=noncausal_loop_results,
            ):
                tile_idx_i32 = fx.Int32(tile_idx)
                causal_n = tile_idx_i32 * TILE_N - causal_offset
                loop_results = yield tile_iteration(
                    ctx, tile_idx, iter_args, causal_n_start=causal_n
                )

            # ── Epilogue ──
            epilogue_state = unpack_loop_results(loop_results, lane_id)
            emit_void("s_wait_idle")
            rocdl.s_barrier_signal(-1)
            rocdl.s_barrier_wait(-1)
            if num_tiles >= 2:
                epilogue_endtile(
                    ctx,
                    mlir_types,
                    epilogue_state,
                    q_frags,
                    sgpr_state,
                    num_tiles,
                    num_tiles_idx,
                    TILE_N,
                    causal_offset,
                    IS_CAUSAL,
                    v_tdm_cfg,
                    zero_v8f32,
                )
            else:
                epilogue_single_tile(ctx, epilogue_state)

    return fmha_fwd_kernel


HEAD_DIM_QK = 192
HEAD_DIM_V = 128
BLOCK_M = 128
KV_TILE_N = 128
BPP = 2  # bytes per element (bf16)
launch_fns = {}  # {(is_causal, return_lse): launch_fn}


def patch_reusable_slot_specs():
    import ctypes

    from flydsl.expr.numeric import Float32, Float64

    for Cls, cty in [(Float32, ctypes.c_float), (Float64, ctypes.c_double)]:
        if not hasattr(Cls, "_reusable_slot_spec"):

            @classmethod
            def _slot_spec(cls, arg, _c=cty):
                return _c, lambda a: a.value if hasattr(a, "value") else a

            Cls._reusable_slot_spec = _slot_spec
            Cls._reusable_ctype = cty


def ensure_kernel(is_causal: bool, return_lse: bool = False):
    key = (is_causal, return_lse)
    if key in launch_fns:
        return
    patch_reusable_slot_specs()
    kernel = compile_fmha_fwd(is_causal=is_causal, return_lse=return_lse)

    @flyc.jit
    def _launch(
        ptr_O: fx.Tensor,
        ptr_Q: fx.Tensor,
        ptr_K: fx.Tensor,
        ptr_V: fx.Tensor,
        ptr_LSE: fx.Tensor,
        ptr_cu_seqlens_q: fx.Tensor,
        ptr_cu_seqlens_k: fx.Tensor,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        gqa: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        num_heads: fx.Int32,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        for alloc in [lds_alloc_k_a, lds_alloc_k_b, lds_alloc_v_a, lds_alloc_v_b]:
            alloc.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            for alloc in [lds_alloc_k_a, lds_alloc_k_b, lds_alloc_v_a, lds_alloc_v_b]:
                alloc.finalize()
        launcher = kernel(
            ptr_O,
            ptr_Q,
            ptr_K,
            ptr_V,
            ptr_LSE,
            ptr_cu_seqlens_q,
            ptr_cu_seqlens_k,
            softmax_scale,
            stride_q_seq,
            stride_k_seq,
            stride_v_seq,
            stride_o_seq,
            stride_q_head,
            stride_k_head,
            stride_v_head,
            stride_o_head,
            gqa,
            max_seqlen_q,
            max_seqlen_k,
        )
        launcher.launch(
            grid=(
                fx.Index(batch_size),
                fx.Index((max_seqlen_q + (BLOCK_M - 1)) // BLOCK_M),
                fx.Index(num_heads),
            ),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _launch.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}
    launch_fns[key] = _launch


def flash_attn_varlen_d192_gfx1250(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale=None,
    causal=False,
    out=None,
    return_lse=False,
):
    assert q.dtype == torch.bfloat16, f"Expected bf16, got {q.dtype}"
    assert (
        q.shape[-1] == HEAD_DIM_QK
    ), f"Expected headdim_qk={HEAD_DIM_QK}, got {q.shape[-1]}"
    assert (
        v.shape[-1] == HEAD_DIM_V
    ), f"Expected headdim_v={HEAD_DIM_V}, got {v.shape[-1]}"
    total_q_tokens, batch = q.shape[0], cu_seqlens_q.shape[0] - 1
    nheads_q, nheads_k = q.shape[1], k.shape[1]
    gqa = nheads_q // nheads_k
    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_DIM_QK**0.5)
    if out is None:
        out = torch.empty(
            (total_q_tokens, nheads_q, HEAD_DIM_V),
            dtype=torch.bfloat16,
            device=q.device,
        )
    lse_shape = (
        (total_q_tokens, nheads_q) if return_lse else (batch, nheads_q, max_seqlen_q)
    )
    lse = torch.empty(lse_shape, dtype=torch.float32, device=q.device)
    stride_q_bytes, stride_k_bytes, stride_v_bytes = (
        q.stride(0) * BPP,
        k.stride(0) * BPP,
        v.stride(0) * BPP,
    )
    ensure_kernel(bool(causal), bool(return_lse))
    _run_compiled(
        launch_fns[(bool(causal), bool(return_lse))],
        out,
        q,
        k,
        v,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale,
        stride_q_bytes,
        stride_k_bytes,
        stride_v_bytes,
        out.stride(0),
        q.stride(1) * BPP,
        k.stride(1) * BPP,
        v.stride(1) * BPP,
        out.stride(1),
        gqa,
        max_seqlen_q,
        max_seqlen_k,
        nheads_q,
        batch,
        torch.cuda.current_stream(),
    )
    return (out, lse) if return_lse else out
