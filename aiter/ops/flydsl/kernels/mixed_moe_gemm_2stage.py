"""MoE GEMM stage1/stage2 kernel implementations (FLIR MFMA FP8/FP16).

This module intentionally contains the **kernel builder code** for:
- `moe_gemm1` (stage1)
- `moe_gemm2` (stage2)

It is extracted from `tests/kernels/test_moe_gemm.py` so that:
- `kernels/` holds the implementation
- `tests/` holds correctness/perf harnesses
"""

import os
from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch


from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from flydsl._mlir import ir
from flydsl.expr.typing import T

from flydsl.expr import arith, gpu, buffer_ops, vector, rocdl
from flydsl._mlir.dialects import llvm, scf, memref
from flydsl._mlir.dialects.arith import CmpIPredicate

from .mfma_preshuffle_pipeline import (
    _buffer_load_vec,
    buffer_copy_gmem16_dwordx4,
    lds_store_16b_xor16,
    lds_store_8b_xor16,
    lds_store_4b_xor16,
    make_preshuffle_b_layout,
    make_preshuffle_scale_layout,
    tile_chunk_coord_i32,
    swizzle_xor16,
)
from .mfma_epilogues import c_shuffle_epilog
from .layout_utils import crd2idx, idx2crd, get as layout_get

import functools


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


def _barrier(vmcnt=63, lgkmcnt=63):
    """Emit s_waitcnt + s_barrier via inline asm.

    Bypasses LLVM SIInsertWaitcnts which would insert a conservative
    s_waitcnt vmcnt(0) lgkmcnt(0) before every S_BARRIER MI.
    """
    parts = []
    needs_waitcnt = vmcnt < 63 or lgkmcnt < 63
    if needs_waitcnt:
        wc = []
        if vmcnt < 63:
            wc.append(f"vmcnt({vmcnt})")
        if lgkmcnt < 63:
            wc.append(f"lgkmcnt({lgkmcnt})")
        parts.append("s_waitcnt " + " ".join(wc))
    parts.append("s_barrier")
    llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string="\n".join(parts),
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
    )


@functools.lru_cache(maxsize=None)
def compile_mixed_moe_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "f16",
    act: str = "silu",
    use_cshuffle_epilog: bool | None = None,
    enable_bias: bool = False,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    persist_m: int = 1,
    fuse_fp4_quant: bool = False,
    fuse_sort_scale: bool = False,
    use_async_copy: bool = False,
    waves_per_eu: int = 3,
    k_batch: int = 1,
    b_nt: int = 0,
    gate_only: bool = False,
):
    """Compile stage1 kernel (gate+up with silu) based on stage2 structure.

    GEMM: silu(X @ W_gate.T) * (X @ W_up.T) -> [tokens*topk, inter_dim]
    Direct store (no atomic).  When k_batch>1 (split-K), each CTA
    computes a K-slice and atomically adds gate/up partials.
    Note: persist_m=1 (no persistence) is optimal for stage1 because K=model_dim
    is large, so each CTA is already compute-heavy. persist_m>1 serializes M blocks
    that the GPU can process in parallel.

    When gate_only=True (requires k_batch>1), each workgroup computes
    only one B-tile stream instead of interleaving gate and up.
    The grid X dimension doubles (inter_in / tile_n instead of
    inter_in / 2 / tile_n) so that by_n covers the full [0, 2*inter_dim)
    range, naturally selecting gate or up rows by position.
    This halves per-WG B-VMEM traffic and MFMA count, and the
    doubled block count compensates.
    """
    gpu_arch = get_hip_arch()
    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")
    _state = {}

    if a_dtype not in ("fp8", "fp16", "int8", "fp4"):
        raise ValueError(
            f"a_dtype must be one of ('fp8','fp16','int8','fp4'), got {a_dtype!r}"
        )
    if b_dtype not in ("fp8", "fp16", "int8", "int4", "fp4"):
        raise ValueError(
            f"b_dtype must be one of ('fp8','fp16','int8','int4','fp4'), got {b_dtype!r}"
        )

    is_f16_a = a_dtype == "fp16"
    is_f16_b = b_dtype == "fp16"
    is_f16 = is_f16_a or is_f16_b
    is_f8_a = a_dtype == "fp8"
    is_f4_a = a_dtype == "fp4"
    is_f4_b = b_dtype == "fp4"

    sort_block_m = max(32, tile_m)
    num_waves = tile_n // 32
    total_threads = num_waves * 64
    pack_M = 1 if tile_m < 32 else 2
    n_per_wave = tile_n // num_waves
    pack_N = min(2, n_per_wave // 16)
    pack_K = 2
    scale_mn_pack = 2
    elem_bytes = 1
    a_elem_bytes = 2 if is_f16_a else 1
    b_elem_bytes = 1
    tile_k_bytes = int(tile_k) * int(a_elem_bytes)
    a_elem_vec_pack = 2 if is_f4_a else 1
    cbsz = 0 if is_f8_a else 4
    blgp = 4

    if (tile_k_bytes % 64) != 0:
        raise ValueError(f"tile_k_bytes must be divisible by 64, got {tile_k_bytes}")

    out_s = str(out_dtype).strip().lower()
    out_is_f32 = out_s in ("f32", "fp32", "float")
    out_is_bf16 = out_s in ("bf16", "bfloat16")
    is_int4 = b_dtype == "int4"
    is_int8 = False

    def _x_elem_type():
        if is_f4_b:
            return T.f8 if is_f8_a else T.i8
        return T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)

    def _w_elem_type():
        if is_f4_b:
            return T.i8
        return T.f16 if is_f16_b else (T.i8 if is_int8 else T.f8)

    def out_elem():
        return T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)

    # Split-K validation
    _is_splitk = k_batch > 1
    if gate_only and not _is_splitk:
        raise ValueError("gate_only requires k_batch > 1 (split-K)")
    if _is_splitk:
        _k_per_batch = model_dim // k_batch
        assert model_dim % k_batch == 0, f"model_dim={model_dim} not divisible by k_batch={k_batch}"
        assert _k_per_batch % tile_k == 0, f"K_per_batch={_k_per_batch} not divisible by tile_k={tile_k}"

        fuse_fp4_quant = False
    else:
        _k_per_batch = model_dim
    _k_dim = _k_per_batch

    # Stage1 gate-only: output = [tokens*topk, inter_dim], direct store (accumulate=False)
    # Weight layout: [E * 2*inter_dim, model_dim] pre-shuffled; gate = first inter_dim rows per expert
    # GEMM: X[tokens, model_dim] @ W_gate[inter_dim, model_dim].T -> [tokens*topk, inter_dim]
    accumulate = _is_splitk

    DYN = ir.ShapedType.get_dynamic_size()
    size_w = experts * (2 * inter_dim) * model_dim
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(a_elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            f"tile_m*tile_k*elem_bytes must be divisible by {total_threads}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads

    _use_lds128 = os.environ.get("FLIR_CK_LDS128", "1") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    pad_k = 0 if _use_lds128 else 8
    lds_stride = tile_k + pad_k

    if use_cshuffle_epilog is None:
        _use_cshuffle_epilog = os.environ.get("FLIR_MOE_STAGE1_CSHUFFLE", "1") in (
            "1",
            "true",
            "True",
            "YES",
            "yes",
        )
    else:
        _use_cshuffle_epilog = bool(use_cshuffle_epilog)

    _need_quant = fuse_fp4_quant
    _need_sort = _need_quant and fuse_sort_scale

    if _need_quant:
        _use_cshuffle_epilog = True

    _fp4q_tag = "_fp4q" if _need_quant else ""
    _sort_tag = "_sort" if _need_sort else ""
    _async_tag = "_async" if use_async_copy else ""
    _sk_tag = f"_sk{k_batch}" if _is_splitk else ""
    _go_tag = "_go" if gate_only else ""
    module_name = (
        f"mfma_moe1_silu_mul_a{a_dtype}_w{b_dtype}_{out_s}"
        f"_t{tile_m}x{tile_n}x{tile_k}_pm{persist_m}{_fp4q_tag}{_sort_tag}{_async_tag}{_sk_tag}{_go_tag}_v32"
    ).replace("-", "_")

    # -- LDS sizing (split ping/pong allocators) --
    _cshuffle_elem_bytes = 4 if _need_quant else (4 if out_is_f32 else 2)
    _single_x_bytes = int(tile_m) * int(lds_stride) * int(a_elem_bytes)
    lds_out_bytes = _cshuffle_elem_bytes * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    lds_tid_bytes = int(tile_m) * 4
    _buffer_bytes = max(_single_x_bytes, lds_out_bytes)
    _buffer_elems = _buffer_bytes if a_elem_bytes == 1 else (_buffer_bytes // 2)

    def x_lds_elem():
        return T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)

    lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = lds_pong_offset + _buffer_bytes
    _lds_tid_offset_pong = allocator_pong._align(allocator_pong.ptr, 4)
    allocator_pong.ptr = _lds_tid_offset_pong + lds_tid_bytes

    lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = lds_ping_offset + _buffer_bytes

    # if tile_m == 16:
    #     waves_per_eu = 1

    if waves_per_eu is not None and waves_per_eu >= 1:
        _total_cu_lds = 160 * 1024
        _min_lds = _total_cu_lds // (waves_per_eu + 1) + 1
        _pong_sz = allocator_pong._align(allocator_pong.ptr, 128)
        _ping_sz = allocator_ping._align(allocator_ping.ptr, 128)
        _cur_lds = _pong_sz + _ping_sz
        if _cur_lds < _min_lds:
            allocator_ping.ptr += _min_lds - _cur_lds

    kpack_bytes = 8 if is_int4 else 16
    out_elem_bytes = 4 if out_is_f32 else 2

    _e_vec_s1 = min(tile_n // 32, 8)
    if _need_quant:
        _e_vec_s1 = max(2, _e_vec_s1)
    _num_threads_per_quant_blk_s1 = 32 // _e_vec_s1
    _shuffle_dists_s1 = []
    _sh_val = 1
    while _sh_val < _num_threads_per_quant_blk_s1:
        _shuffle_dists_s1.append(_sh_val)
        _sh_val *= 2
    _num_shuffle_steps_s1 = len(_shuffle_dists_s1)

    # ---- Unified pipeline schedule (outside @flyc.kernel) ----
    # Each scheduling phase is a dict:
    #   mfma:      [(k_idx, mi_idx, ikxdl, imxdl, asv_idx), ...]
    #   a_reads:   [(k, mi), ...]       # A ds_read subtiles
    #   b_loads:   [('gate'/'up', ku, ni), ...]  # B VMEM loads
    #   has_scale: bool                  # A/B scale VMEM loads
    _pipe_m_repeat = tile_m // 16
    _pipe_k_unroll = tile_k_bytes // 128
    _pipe_k_unroll_packed = _pipe_k_unroll // pack_K
    _pipe_m_repeat_packed = _pipe_m_repeat // pack_M
    _pipe_num_acc_n = n_per_wave // 16

    # A ds_read groups: group by mi (same mi, all k values together)
    _pipe_a_groups = []
    for _mi in range(_pipe_m_repeat):
        _grp = []
        for _k in range(_pipe_k_unroll):
            _grp.append((_k, _mi))
            if len(_grp) == 2:
                _pipe_a_groups.append(_grp)
                _grp = []
        if _grp:
            _pipe_a_groups.append(_grp)

    # B VMEM loads: individual gate/up loads
    _pipe_b_loads = []
    for ku in range(_pipe_k_unroll):
        for ni in range(_pipe_num_acc_n):
            _pipe_b_loads.append(('gate', ku, ni))
            if not gate_only:
                _pipe_b_loads.append(('up', ku, ni))

    # MFMA order: B-major (fix B, cycle all A tiles before next B)
    # Each entry: one (k, ni) pair; the compute function loops over all mi.
    # This keeps B operands (from VMEM) fixed while cycling A (from LDS, no wait).
    _pipe_all_mfma = []
    for _ku128 in range(_pipe_k_unroll_packed):
        for _ikxdl in range(pack_K):
            for _inxdl in range(pack_N):
                _k_idx = _ku128 * pack_K + _ikxdl
                _ni_idx = _inxdl
                _pipe_all_mfma.append(
                    (_k_idx, _ni_idx, _ikxdl, _inxdl, _ku128)
                )

    # Group MFMAs per scheduling phase (wider M → more MFMAs per phase)
    _pipe_mfma_per_phase = max(1, len(_pipe_all_mfma) // 4)
    _pipe_n_phases = len(_pipe_all_mfma) // _pipe_mfma_per_phase

    # Build unified phase descriptors
    _a_groups_per_phase = (len(_pipe_a_groups) + _pipe_n_phases - 1) // _pipe_n_phases
    _pipe_phases = []
    _mfma_i = 0
    _a_i = 0
    for _p in range(_pipe_n_phases):
        _a_reads = []
        for _ in range(_a_groups_per_phase):
            if _a_i < len(_pipe_a_groups):
                _a_reads.extend(_pipe_a_groups[_a_i])
                _a_i += 1
        _phase = {
            'mfma': _pipe_all_mfma[_mfma_i : _mfma_i + _pipe_mfma_per_phase],
            'a_reads': _a_reads,
            'b_loads': [],
            'has_scale': (_p == 0),
        }
        _mfma_i += _pipe_mfma_per_phase
        _pipe_phases.append(_phase)

    # Distribute B loads evenly across phases 1..n-1 (phase 0 has scales)
    _bi = 0
    for _p in range(1, _pipe_n_phases):
        _rem_b = len(_pipe_b_loads) - _bi
        _rem_p = _pipe_n_phases - _p
        _n_b = (_rem_b + _rem_p - 1) // _rem_p if _rem_p > 0 else 0
        for _ in range(_n_b):
            if _bi < len(_pipe_b_loads):
                _pipe_phases[_p]['b_loads'].append(_pipe_b_loads[_bi])
                _bi += 1

    # Extract flat lists for kernel access (avoids dict access in AST rewriter)
    _pp_mfma = [p['mfma'] for p in _pipe_phases]
    _pp_a_reads = [p['a_reads'] for p in _pipe_phases]
    _pp_b_loads = [p['b_loads'] for p in _pipe_phases]
    _pp_has_scale = [p['has_scale'] for p in _pipe_phases]

    if True:

        @flyc.kernel
        def moe_gemm1(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_sorted_token_ids: fx.Tensor,
            arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor,
            arg_num_valid_ids: fx.Tensor,
            arg_bias: fx.Tensor,
            arg_out_scale_sorted: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_n_in: fx.Int32,
            i32_k_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
        ):
            arg_out = arg_out.value
            arg_x = arg_x.value
            arg_w = arg_w.value
            arg_scale_x = arg_scale_x.value
            arg_scale_w = arg_scale_w.value
            arg_sorted_token_ids = arg_sorted_token_ids.value
            arg_expert_ids = arg_expert_ids.value
            arg_sorted_weights = arg_sorted_weights.value
            arg_num_valid_ids = arg_num_valid_ids.value
            arg_bias = arg_bias.value
            arg_out_scale_sorted = arg_out_scale_sorted.value

            tokens_in = arith.index_cast(ir.IndexType.get(), i32_tokens_in.ir_value())
            n_in = arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
            k_in = arith.index_cast(ir.IndexType.get(), i32_k_in.ir_value())
            size_expert_ids_in = arith.index_cast(
                ir.IndexType.get(), i32_size_expert_ids_in.ir_value()
            )

            x_elem = T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)
            w_elem = T.f16 if is_f16_b else (T.i8 if is_int8 else T.f8)
            f16 = T.f16
            f32 = T.f32
            i32 = T.i32
            i64 = T.i64
            vec4_f32 = T.vec(4, f32)
            vec4_i32 = T.vec(4, i32)
            vec16_elems = 16 if a_elem_bytes == 1 else 8
            vec8_elems = 8 if a_elem_bytes == 1 else 4
            vec4_elems = 4 if a_elem_bytes == 1 else 2
            vec16_x = T.vec(vec16_elems, x_elem)
            vec8_x = T.vec(vec8_elems, x_elem)
            vec1_i64 = T.vec(1, i64)
            vec2_i64 = T.vec(2, i64)

            acc_init = arith.constant_vector(0.0, vec4_f32)

            # --- Stage1 dimension mapping ---
            # X: [tokens, model_dim] -- M = sorted tokens, K = model_dim
            # W: [E*2*inter_dim, model_dim] gate portion -- N = inter_dim
            # Out: [tokens*topk, inter_dim]
            topk_idx = arith.constant(topk, index=True)
            # X layout: row = token_id, col = model_dim
            tokens_i32_v = i32_tokens_in.ir_value()
            k_i32_v = i32_k_in.ir_value()

            # B preshuffle layout: [E*2*inter_dim, model_dim]
            # Gate rows for expert e: [e*2*inter_dim, e*2*inter_dim + inter_dim)
            c_n_total = arith.constant(experts * (2 * inter_dim), index=True)
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=k_in // pack_K,
                kpack_bytes=kpack_bytes,
                elem_bytes=b_elem_bytes,
                # k_major=True,
            )
            layout_b = b_layout.layout_b

            # A-scale: [sorted_size, K/32] -- pre-scattered by caller into sorted layout
            # Same as stage2: indexed by sorted_row position, not by token_id.
            sorted_m = size_expert_ids_in * arith.constant(sort_block_m, index=True)
            layout_a_scale = make_preshuffle_scale_layout(
                arith, c_mn=sorted_m, c_k=arith.constant(model_dim, index=True)
            )
            # B-scale: [E*2*inter_dim, K/32]
            layout_b_scale = make_preshuffle_scale_layout(
                arith, c_mn=c_n_total, c_k=arith.constant(model_dim, index=True)
            )

            if use_async_copy and a_elem_vec_pack > 1:
                _eff_lds_stride = lds_stride // a_elem_vec_pack
                _eff_tile_k_bytes = tile_k_bytes // a_elem_vec_pack
            else:
                _eff_lds_stride = lds_stride
                _eff_tile_k_bytes = tile_k_bytes

            shape_lds = fx.make_shape(tile_m, _eff_lds_stride)
            stride_lds = fx.make_stride(_eff_lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            by = gpu.block_id("x")  # tile along inter_dim (N)
            bx_persist = gpu.block_id("y")  # persistent WG index
            by_n = by * arith.constant(tile_n, index=True)

            if _is_splitk:
                bz = gpu.block_id("z")  # K-batch id
                k_base_idx = bz * arith.constant(_k_dim, index=True)
            else:
                k_base_idx = arith.index(0)

            k_blocks16 = arith.constant(_eff_tile_k_bytes // 16, index=True)
            layout_tx_wave_lane = fx.make_layout((num_waves, 64), stride=(64, 1))
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

            base_ptr_pong = allocator_pong.get_base()
            base_ptr_ping = allocator_ping.get_base()
            lds_x_pong = SmemPtr(
                base_ptr_pong, lds_pong_offset, x_lds_elem(), shape=(_buffer_elems,)
            ).get()
            lds_x_ping = SmemPtr(
                base_ptr_ping, lds_ping_offset, x_lds_elem(), shape=(_buffer_elems,)
            ).get()
            _lds_out_elem_type = T.f32 if _need_quant else (T.bf16 if out_is_bf16 else T.f16)
            lds_out = (
                SmemPtr(
                    base_ptr_pong,
                    lds_pong_offset,
                    _lds_out_elem_type,
                    shape=(tile_m * tile_n,),
                ).get()
                if _use_cshuffle_epilog
                else None
            )
            lds_tid = SmemPtr(
                base_ptr_pong, _lds_tid_offset_pong, T.i32, shape=(tile_m,)
            ).get()

            # Buffer resources
            c_a_pack = arith.constant(int(a_elem_vec_pack), index=True)
            c_elem_bytes = arith.constant(int(a_elem_bytes), index=True)

            # X: [tokens, model_dim]
            x_nbytes_idx = (tokens_in * k_in * c_elem_bytes) / c_a_pack
            x_nbytes_i32 = arith.index_cast(T.i32, x_nbytes_idx)
            x_rsrc = buffer_ops.create_buffer_resource(
                arg_x, max_size=False, num_records_bytes=x_nbytes_i32
            )

            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)

            # Out: [tokens*topk, inter_dim]
            out_nbytes_idx = (
                tokens_in
                * arith.index(topk)
                * n_in
                * arith.constant(out_elem_bytes, index=True)
            )
            out_nbytes_i32 = arith.index_cast(T.i32, out_nbytes_idx)
            out_rsrc = buffer_ops.create_buffer_resource(
                arg_out, max_size=False, num_records_bytes=out_nbytes_i32
            )

            numids_rsrc = buffer_ops.create_buffer_resource(
                arg_num_valid_ids,
                max_size=False,
                num_records_bytes=arith.constant(4, type=T.i32),
            )
            num_valid_i32 = buffer_ops.buffer_load(
                numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=T.i32
            )
            num_valid_idx = arith.index_cast(ir.IndexType.get(), num_valid_i32)

            if is_f16_a:
                sx_rsrc = None
            else:
                # A scale: [sorted_size, model_dim/32] pre-scattered by caller
                c32 = arith.constant(32, index=True)
                kblk = k_in / c32
                sx_nbytes_idx = sorted_m * kblk
                sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
                sx_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_i32
                )

            if is_f16_b:
                sw_rsrc = None
            else:
                c32 = arith.constant(32, index=True)
                kblk_w = k_in / c32
                mn_w = arith.constant(experts * (2 * inter_dim), index=True)
                sw_nbytes_idx = mn_w * kblk_w
                sw_nbytes_i32 = arith.index_cast(T.i32, sw_nbytes_idx)
                sw_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_w, max_size=False, num_records_bytes=sw_nbytes_i32
                )

            sorted_nbytes_idx = size_expert_ids_in * arith.constant(
                sort_block_m * 4, index=True
            )
            sorted_nbytes_i32 = arith.index_cast(T.i32, sorted_nbytes_idx)
            sorted_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_token_ids,
                max_size=False,
                num_records_bytes=sorted_nbytes_i32,
            )
            sorted_w_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_weights, max_size=False, num_records_bytes=sorted_nbytes_i32
            )

            eid_nbytes_idx = size_expert_ids_in * arith.constant(4, index=True)
            eid_nbytes_i32 = arith.index_cast(T.i32, eid_nbytes_idx)
            expert_rsrc = buffer_ops.create_buffer_resource(
                arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes_i32
            )

            # Sorted-scale buffer resource for fused mxfp4 quantization
            _sorted_scale_cols = inter_dim // 32
            _sorted_scale_cols_i32 = arith.constant(_sorted_scale_cols, type=T.i32)
            if _need_sort:
                sorted_scale_rsrc = buffer_ops.create_buffer_resource(
                    arg_out_scale_sorted, max_size=False
                )
            else:
                sorted_scale_rsrc = None

            # ---- persist_m loop (same pattern as stage2) ----
            _PERSIST_M = persist_m
            _c0_p = arith.constant(0, index=True)
            _c1_p = arith.constant(1, index=True)
            _c_pm = arith.constant(_PERSIST_M, index=True)
            _for_persist = scf.ForOp(_c0_p, _c_pm, _c1_p)
            _for_ip = ir.InsertionPoint(_for_persist.body)
            _for_ip.__enter__()
            _mi_p = _for_persist.induction_variable
            bx = bx_persist * _c_pm + _mi_p
            bx_m = bx * arith.constant(sort_block_m, index=True)

            # Block validity
            bx_m_i32 = arith.index_cast(T.i32, bx_m)
            blk_valid = arith.cmpi(CmpIPredicate.ult, bx_m_i32, num_valid_i32)
            expert_i32 = buffer_ops.buffer_load(
                expert_rsrc, bx, vec_width=1, dtype=T.i32
            )
            expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
            exp_valid = arith.cmpi(
                CmpIPredicate.ult, expert_i32, arith.constant(experts, type=T.i32)
            )

            def _moe_gemm1_body():
                # Gate expert offset: first inter_dim rows of each expert's 2*inter_dim block
                expert_off_idx = expert_idx * arith.constant(2 * inter_dim, index=True)

                # X loading -- KEY DIFFERENCE from stage2: X row = token_id only
                x_load_bytes = 16
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4
                vec4_i32_t = T.vec(4, i32)

                c_k_div4 = (
                    (k_in / c_a_pack) * arith.constant(int(a_elem_bytes), index=True)
                ) / arith.index(4)
                c_k_div4_i32 = arith.index_cast(T.i32, c_k_div4)
                tile_k_dwords = (int(tile_k) * int(a_elem_bytes)) // (
                    4 * int(a_elem_vec_pack)
                )
                layout_x_tile_div4 = fx.make_layout(
                    (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                )
                c_chunk_i32 = arith.constant(chunk_i32, index=True)
                tx_i32_base = tx * c_chunk_i32

                topk_i32 = arith.constant(topk)
                mask24 = arith.constant(0xFFFFFF)
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

                def load_x(idx_i32):
                    idx_elem = (
                        idx_i32 if a_elem_bytes == 1 else (idx_i32 * arith.index(2))
                    )
                    return buffer_copy_gmem16_dwordx4(
                        buffer_ops,
                        vector,
                        elem_type=x_elem,
                        idx_i32=idx_elem,
                        rsrc=x_rsrc,
                        vec_elems=vec16_elems,
                    )

                # Decode sorted token ids -- stage1: X row = token_id (not t*topk+s)
                x_row_base_div4 = []
                x_col_local_i32 = []
                x_row_local = []
                # Also store token_id and slot_id for output indexing
                x_token_idx = []
                x_slot_idx = []
                x_row_valid = []

                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_row_local.append(row_local)
                    x_col_local_i32.append(col_local_i32)

                    sorted_row_i = bx_m + row_local
                    fused_i = buffer_ops.buffer_load(
                        sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32
                    )
                    t_i32 = arith.andi(fused_i, mask24)
                    s_i32 = arith.shrui(fused_i, arith.constant(24))
                    t_valid = arith.cmpi(CmpIPredicate.ult, t_i32, tokens_i32)
                    s_valid = arith.cmpi(CmpIPredicate.ult, s_i32, topk_i32)
                    ts_valid = arith.andi(t_valid, s_valid)
                    t_safe = arith.select(ts_valid, t_i32, arith.constant(0))

                    # KEY: X row base uses token_id only (not t*topk+s)
                    t_idx = arith.index_cast(ir.IndexType.get(), t_safe)
                    x_row_base_div4.append(t_idx * c_k_div4)

                def load_x_tile(base_k):
                    base_k_div4 = (
                        (base_k / c_a_pack)
                        * arith.constant(int(a_elem_bytes), index=True)
                    ) / arith.index(4)
                    parts = []
                    for i in range_constexpr(num_x_loads):
                        idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                        x_vec = load_x(idx_i32)
                        parts.append(vector.bitcast(T.vec(4, i32), x_vec))
                    return parts

                # Wave/lane decomposition (identical to stage2)
                coord_wl = idx2crd(tx, layout_tx_wave_lane)
                wave_id = layout_get(coord_wl, 0)
                lane_id = layout_get(coord_wl, 1)
                coord_l16 = idx2crd(lane_id, layout_lane16)
                lane_div_16 = layout_get(coord_l16, 0)
                lane_mod_16 = layout_get(coord_l16, 1)
                row_a_lds = lane_mod_16
                col_offset_base = lane_div_16 * arith.constant(16, index=True)

                num_acc_n = n_per_wave // 16
                c_n_per_wave = arith.constant(n_per_wave, index=True)
                wave_n_id = wave_id % arith.constant(num_waves, index=True)
                n_tile_base = wave_n_id * c_n_per_wave

                # N-tile precompute for gate AND up weights
                gate_n_intra_list = []
                gate_n_blk_list = []
                if not gate_only:
                    up_n_intra_list = []
                    up_n_blk_list = []
                col_g_list = []
                c_n0_static = experts * (2 * inter_dim) // 16
                layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))
                inter_idx = arith.constant(inter_dim, index=True)

                for i in range_constexpr(num_acc_n):
                    offset = i * 16
                    c_offset = arith.constant(offset, index=True)
                    col_g = by_n + n_tile_base + c_offset + lane_mod_16
                    col_g_list.append(col_g)

                    global_n = by_n + n_tile_base + c_offset + lane_mod_16
                    # Gate: rows [expert_off, expert_off + inter_dim)
                    # For gate_only, by_n covers [0, 2*inter_dim) so this
                    # indexes into both gate and up regions naturally.
                    gate_row_w = expert_off_idx + global_n
                    gate_coord = idx2crd(gate_row_w, layout_n_blk_intra)
                    gate_n_blk_list.append(layout_get(gate_coord, 0))
                    gate_n_intra_list.append(layout_get(gate_coord, 1))
                    if not gate_only:
                        # Up: rows [expert_off + inter_dim, expert_off + 2*inter_dim)
                        up_row_w = gate_row_w + inter_idx
                        up_coord = idx2crd(up_row_w, layout_n_blk_intra)
                        up_n_blk_list.append(layout_get(up_coord, 0))
                        up_n_intra_list.append(layout_get(up_coord, 1))

                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 128
                k_unroll_packed = k_unroll // pack_K
                m_repeat_packed = m_repeat // pack_M
                num_acc_n_packed = num_acc_n // pack_N

                # B load for gate and up separately
                def load_b_packs_k64(base_k, ku: int, n_blk, n_intra):
                    c64 = arith.constant(64, index=True)
                    base_k_bytes = base_k * arith.constant(
                        int(b_elem_bytes), index=True
                    )
                    k0 = base_k_bytes // c64 + arith.constant(ku, index=True)
                    k1 = lane_div_16
                    coord_pack = (n_blk, k0, k1, n_intra, arith.constant(0, index=True))
                    idx_pack = crd2idx(coord_pack, layout_b)
                    vec_elems = kpack_bytes // int(b_elem_bytes)
                    b16 = _buffer_load_vec(
                        buffer_ops,
                        vector,
                        w_rsrc,
                        idx_pack,
                        elem_type=_w_elem_type(),
                        vec_elems=vec_elems,
                        elem_bytes=b_elem_bytes,
                        offset_in_bytes=(b_elem_bytes == 1),
                        cache_modifier=b_nt,
                    )
                    b_i64x2 = vector.bitcast(vec2_i64, b16)
                    b0 = vector.extract(
                        b_i64x2, static_position=[0], dynamic_position=[]
                    )
                    b1 = vector.extract(
                        b_i64x2, static_position=[1], dynamic_position=[]
                    )
                    return b0, b1

                def load_b_tile(base_k):
                    """Load B tiles. Returns (gate_b_tile, up_b_tile).
                    When gate_only, up_b_tile is None."""
                    gate_b_tile = []
                    up_b_tile = [] if not gate_only else None
                    for ku in range_constexpr(k_unroll):
                        g_packs0, g_packs1 = [], []
                        u_packs0, u_packs1 = [], []
                        for ni in range_constexpr(num_acc_n):
                            gb0, gb1 = load_b_packs_k64(
                                base_k, ku, gate_n_blk_list[ni], gate_n_intra_list[ni]
                            )
                            g_packs0.append(gb0)
                            g_packs1.append(gb1)
                            if not gate_only:
                                ub0, ub1 = load_b_packs_k64(
                                    base_k, ku, up_n_blk_list[ni], up_n_intra_list[ni]
                                )
                                u_packs0.append(ub0)
                                u_packs1.append(ub1)
                        gate_b_tile.append((g_packs0, g_packs1))
                        if not gate_only:
                            up_b_tile.append((u_packs0, u_packs1))
                    return gate_b_tile, up_b_tile

                # Pre-compute scale base element indices (K-loop invariant).
                # idx = mni * stride_n0 + ku * stride_k0 + k_lane * stride_klane + n_lane
                # Split into: base_elem = mni * stride_n0 + lane_elem (invariant)
                #              k_elem    = ku * stride_k0             (per-iteration)
                _scale_lane_elem = (
                    lane_div_16 * layout_b_scale.stride_klane + lane_mod_16
                )

                _gate_scale_bases = []
                if not gate_only:
                    _up_scale_bases = []
                for _ni in range_constexpr(num_acc_n_packed):
                    _col_base = by_n + n_tile_base + arith.constant(
                        _ni * 16 * pack_N, index=True
                    )
                    _gate_mni = (
                        expert_off_idx + _col_base
                    ) // arith.constant(32, index=True)
                    _gate_scale_bases.append(
                        _gate_mni * layout_b_scale.stride_n0 + _scale_lane_elem
                    )
                    if not gate_only:
                        _up_mni = (
                            expert_off_idx + inter_idx + _col_base
                        ) // arith.constant(32, index=True)
                        _up_scale_bases.append(
                            _up_mni * layout_b_scale.stride_n0 + _scale_lane_elem
                        )

                _a_scale_bases = []
                for _mi in range_constexpr(m_repeat_packed):
                    _a_mni = _mi + bx_m // scale_mn_pack // 16
                    _a_scale_bases.append(
                        _a_mni * layout_a_scale.stride_n0 + _scale_lane_elem
                    )

                _c16_idx = arith.constant(16, index=True)
                _c2_idx = arith.constant(2, index=True)
                _scale_mask_lo = arith.constant(0xFF, type=T.i32)

                if pack_M < scale_mn_pack:
                    _m_half_idx = (bx_m // _c16_idx) % _c2_idx
                    _m_half_i32 = arith.index_cast(T.i32, _m_half_idx)
                    _scale_shift = _m_half_i32 * arith.constant(8, type=T.i32)
                    _scale_shift_hi = _scale_shift + arith.constant(16, type=T.i32)

                if pack_N < scale_mn_pack:
                    _n_half_idx = (n_tile_base // _c16_idx) % _c2_idx
                    _n_half_i32 = arith.index_cast(T.i32, _n_half_idx)
                    _bscale_shift = _n_half_i32 * arith.constant(8, type=T.i32)
                    _bscale_shift_hi = _bscale_shift + arith.constant(16, type=T.i32)

                def _rearrange_a_scale(raw_i32):
                    """Rearrange scale bytes for pack_M=1: extract m_half's k0,k1 bytes."""
                    if pack_M >= scale_mn_pack:
                        return raw_i32
                    b_k0 = arith.andi(arith.shrui(raw_i32, _scale_shift), _scale_mask_lo)
                    b_k1 = arith.andi(arith.shrui(raw_i32, _scale_shift_hi), _scale_mask_lo)
                    return arith.ori(b_k0, arith.shli(b_k1, arith.constant(8, type=T.i32)))

                def _rearrange_b_scale(raw_i32):
                    """Rearrange scale bytes for pack_N=1: extract n_half's k0,k1 bytes."""
                    if pack_N >= scale_mn_pack:
                        return raw_i32
                    b_k0 = arith.andi(arith.shrui(raw_i32, _bscale_shift), _scale_mask_lo)
                    b_k1 = arith.andi(arith.shrui(raw_i32, _bscale_shift_hi), _scale_mask_lo)
                    return arith.ori(b_k0, arith.shli(b_k1, arith.constant(8, type=T.i32)))

                def prefetch_ab_scale_tile(base_k):
                    a_scale_tile = []
                    gate_b_scale = []
                    up_b_scale = [] if not gate_only else None
                    for ku in range_constexpr(k_unroll_packed):
                        k_off = (ku + base_k) * layout_b_scale.stride_k0
                        for mi in range_constexpr(m_repeat_packed):
                            s = buffer_ops.buffer_load(
                                sx_rsrc,
                                _a_scale_bases[mi] + k_off,
                                vec_width=1, dtype=T.i32,
                                cache_modifier=0,
                            )
                            s = _rearrange_a_scale(s)
                            a_scale_tile.append(
                                vector.from_elements(T.vec(1, T.i32), [s])
                            )
                        for ni in range_constexpr(num_acc_n_packed):
                            gs = buffer_ops.buffer_load(
                                sw_rsrc,
                                _gate_scale_bases[ni] + k_off,
                                vec_width=1, dtype=T.i32,
                                cache_modifier=0,
                            )
                            gs = _rearrange_b_scale(gs)
                            gate_b_scale.append(
                                vector.from_elements(T.vec(1, T.i32), [gs])
                            )
                            if not gate_only:
                                us = buffer_ops.buffer_load(
                                    sw_rsrc,
                                    _up_scale_bases[ni] + k_off,
                                    vec_width=1, dtype=T.i32,
                                    cache_modifier=0,
                                )
                                us = _rearrange_b_scale(us)
                                up_b_scale.append(
                                    vector.from_elements(T.vec(1, T.i32), [us])
                                )
                    return [a_scale_tile, gate_b_scale, up_b_scale]

                vec4_x_lds = T.vec(vec4_elems, x_elem)
                _lds_base_zero = arith.index(0)

                def store_x_tile_to_lds(vec_x_in_parts, lds_buffer):
                    for i in range_constexpr(num_x_loads):
                        row_local = x_row_local[i]
                        col_local_i32 = x_col_local_i32[i]
                        if x_load_bytes == 16:
                            lds_store_16b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_buffer,
                                vec16_ty=vec16_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=_lds_base_zero,
                                vec_part_i32x4=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )

                if use_async_copy:
                    _dma_bytes = 16
                    _wave_size = 64
                    _eff_bytes_per_buffer = int(tile_m) * int(_eff_lds_stride) * int(a_elem_bytes)
                    _num_dma_loads = max(1, _eff_bytes_per_buffer // (total_threads * _dma_bytes))

                    def dma_x_tile_to_lds(base_k, lds_buffer):
                        c4_idx = arith.index(4)
                        base_k_div4 = ((base_k / c_a_pack) * arith.constant(int(elem_bytes), index=True)) / arith.index(4)

                        lds_ptr_i64 = None
                        for i in range_constexpr(_num_dma_loads):
                            row_local_i = x_row_local[i]
                            col_local_i32_i = x_col_local_i32[i]
                            col_local_sw = swizzle_xor16(row_local_i, col_local_i32_i * c4_idx, k_blocks16)
                            row_k_dw = x_row_base_div4[i] + base_k_div4
                            global_byte_idx = row_k_dw * c4_idx + col_local_sw
                            global_offset = arith.index_cast(T.i32, global_byte_idx)

                            if i == 0:
                                lds_addr = (
                                    memref.extract_aligned_pointer_as_index(lds_buffer)
                                    + wave_id * arith.constant(_wave_size * _dma_bytes, index=True)
                                )
                                lds_ptr_i64 = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, lds_addr))
                            else:
                                lds_ptr_i64 = lds_ptr_i64 + arith.constant(total_threads * _dma_bytes, type=T.i64)

                            lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                            lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64)

                            rocdl.raw_ptr_buffer_load_lds(
                                x_rsrc,
                                lds_ptr,
                                arith.constant(_dma_bytes, type=T.i32),
                                global_offset,
                                arith.constant(0, type=T.i32),
                                arith.constant(0, type=T.i32),
                                arith.constant(0, type=T.i32),
                            )

                    def prefetch_x_to_lds(base_k, lds_buffer):
                        dma_x_tile_to_lds(base_k, lds_buffer)

                def lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer):
                    col_base_swz_bytes = swizzle_xor16(
                        curr_row_a_lds, col_base, k_blocks16
                    )
                    col_base_swz = (
                        col_base_swz_bytes
                        if elem_bytes == 1
                        else (col_base_swz_bytes / arith.index(2))
                    )
                    idx_a16 = crd2idx([curr_row_a_lds, col_base_swz], layout_lds)
                    loaded_a16 = vector.load_op(vec16_x, lds_buffer, [idx_a16])
                    a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                    a0 = vector.extract(
                        a_i64x2, static_position=[0], dynamic_position=[]
                    )
                    a1 = vector.extract(
                        a_i64x2, static_position=[1], dynamic_position=[]
                    )
                    return a0, a1

                def prefetch_full_a_from_lds(lds_buffer):
                    """Load entire A tile from LDS into registers before compute."""
                    a_regs = []
                    for k_idx in range_constexpr(k_unroll):
                        col_base = (
                            col_offset_base
                            + (k_idx * 128) // a_elem_vec_pack
                        )
                        for mi_idx in range_constexpr(m_repeat):
                            mi_val = arith.constant(mi_idx * 16, index=True)
                            curr_row = row_a_lds + mi_val
                            a0, a1 = lds_load_packs_k64(
                                curr_row, col_base, lds_buffer
                            )
                            if is_f8_a:
                                a2, a3 = lds_load_packs_k64(
                                    curr_row, col_base + 64, lds_buffer
                                )
                                a_regs.append((a0, a1, a2, a3))
                            else:
                                a_regs.append((a0, a1))
                    return a_regs

                # Compute tile: gate + up MFMA interleaved, same A data, different B data.
                # Two accumulator sets; after all K tiles, acc = acc_gate + acc_up (f32 add).
                def compute_tile(
                    acc_gate_in,
                    acc_up_in,
                    gate_b_tile_in,
                    up_b_tile_in,
                    a_tile_regs,
                    a_scale=None,
                    gate_b_scale=None,
                    up_b_scale=None,
                    *,
                    prefetch_epilogue=False,
                ):
                    gate_list = list(acc_gate_in)
                    up_list = list(acc_up_in) if not gate_only else None
                    mfma_res_ty = vec4_f32
                    epilogue_pf = None
                    if prefetch_epilogue and doweight_stage1:
                        tw_pf = []
                        lane_div_16_mul4_pf = lane_div_16 * arith.index(4)
                        ii_idx_list_pf = [
                            arith.constant(ii, index=True) for ii in range(4)
                        ]
                        for mi in range_constexpr(m_repeat):
                            mi_base_pf = arith.constant(mi * 16, index=True)
                            for ii in range_constexpr(4):
                                row_off_pf = lane_div_16_mul4_pf + ii_idx_list_pf[ii]
                                sorted_row_pf = bx_m + mi_base_pf + row_off_pf
                                tw_pf.append(
                                    buffer_ops.buffer_load(
                                        sorted_w_rsrc,
                                        sorted_row_pf,
                                        vec_width=1,
                                        dtype=f32,
                                    )
                                )
                        epilogue_pf = (None, tw_pf, None)

                    c0_i64 = arith.constant(0, type=T.i64)
                    vec4_i64 = T.vec(4, T.i64)
                    vec8_i32 = T.vec(8, T.i32)

                    def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                        v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                        return vector.bitcast(vec8_i32, v4)

                    # B-major: fix B (ni), cycle A (mi) — B from VMEM stays
                    # in registers while A from LDS is repacked per mi.
                    for ku128 in range_constexpr(k_unroll_packed):
                        for ni in range_constexpr(num_acc_n_packed):
                            gate_bs_i32 = gate_b_scale[
                                ku128 * num_acc_n_packed + ni
                            ]
                            gate_bs_val = vector.extract(
                                gate_bs_i32,
                                static_position=[0],
                                dynamic_position=[],
                            )
                            if not gate_only:
                                up_bs_i32 = up_b_scale[ku128 * num_acc_n_packed + ni]
                                up_bs_val = vector.extract(
                                    up_bs_i32, static_position=[0], dynamic_position=[]
                                )
                            for ikxdl in range_constexpr(pack_K):
                                k_idx = ku128 * pack_K + ikxdl
                                gate_bp0, gate_bp1 = gate_b_tile_in[k_idx]
                                if not gate_only:
                                    up_bp0, up_bp1 = up_b_tile_in[k_idx]
                                for inxdl in range_constexpr(pack_N):
                                    ni_idx = ni * pack_N + inxdl
                                    gb0 = gate_bp0[ni_idx]
                                    gb1 = gate_bp1[ni_idx]
                                    gb128 = pack_i64x4_to_i32x8(
                                        gb0, gb1, c0_i64, c0_i64
                                    )
                                    if not gate_only:
                                        ub0 = up_bp0[ni_idx]
                                        ub1 = up_bp1[ni_idx]
                                        ub128 = pack_i64x4_to_i32x8(
                                            ub0, ub1, c0_i64, c0_i64
                                        )
                                    for mi in range_constexpr(m_repeat_packed):
                                        a_scale_i32 = a_scale[ku128 * m_repeat_packed + mi]
                                        a_scale_val = vector.extract(
                                            a_scale_i32, static_position=[0], dynamic_position=[]
                                        )
                                        for imxdl in range_constexpr(pack_M):
                                            mi_idx = mi * pack_M + imxdl
                                            _a_reg_idx = k_idx * m_repeat + mi_idx
                                            if is_f8_a:
                                                a0, a1, a2, a3 = a_tile_regs[_a_reg_idx]
                                                a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)
                                            else:
                                                a0, a1 = a_tile_regs[_a_reg_idx]
                                                a128 = pack_i64x4_to_i32x8(
                                                    a0, a1, c0_i64, c0_i64
                                                )
                                            acc_idx = mi_idx * num_acc_n + ni_idx
                                            gate_list[acc_idx] = (
                                                rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                                    mfma_res_ty,
                                                    [
                                                        a128,
                                                        gb128,
                                                        gate_list[acc_idx],
                                                        cbsz,
                                                        blgp,
                                                        ikxdl * pack_M + imxdl,
                                                        a_scale_val,
                                                        ikxdl * pack_N + inxdl,
                                                        gate_bs_val,
                                                    ],
                                                )
                                            )
                                            if not gate_only:
                                                up_list[acc_idx] = (
                                                    rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                                        mfma_res_ty,
                                                        [
                                                            a128,
                                                            ub128,
                                                            up_list[acc_idx],
                                                            cbsz,
                                                            blgp,
                                                            ikxdl * pack_M + imxdl,
                                                            a_scale_val,
                                                            ikxdl * pack_N + inxdl,
                                                            up_bs_val,
                                                        ],
                                                    )
                                                )
                    return gate_list, up_list, epilogue_pf

                def load_a_subtile(k_idx, mi_idx, lds_buffer):
                    """Load a single A sub-tile from LDS (one ds_read)."""
                    col_base = (
                        col_offset_base
                        + (k_idx * 128) // a_elem_vec_pack
                    )
                    mi_val = arith.constant(mi_idx * 16, index=True)
                    curr_row = row_a_lds + mi_val
                    a0, a1 = lds_load_packs_k64(
                        curr_row, col_base, lds_buffer
                    )
                    if is_f8_a:
                        a2, a3 = lds_load_packs_k64(
                            curr_row, col_base + 64, lds_buffer
                        )
                        return (a0, a1, a2, a3)
                    else:
                        return (a0, a1)

                def compute_bmajor_mfma_phase(
                    all_a_tiles, gate_b_single, up_b_single,
                    a_scale_vals, gate_bs_val, up_bs_val,
                    gate_list, up_list,
                    k_idx, ni_idx, ikxdl, inxdl,
                ):
                    """B-major MFMA: fix one B (ni), cycle all A tiles (mi).

                    Packs B once and reuses across all mi iterations.
                    A tiles come from LDS (already available, no VMEM wait).

                    all_a_tiles: flat list indexed by [k*m_repeat + mi].
                    gate_b_single/up_b_single: (b0, b1) for one specific ni.
                      When gate_only, up_b_single is None.
                    a_scale_vals: list of A scale scalars indexed by mi_packed.
                    """
                    c0_i64 = arith.constant(0, type=T.i64)
                    vec4_i64 = T.vec(4, T.i64)
                    vec8_i32 = T.vec(8, T.i32)

                    def _pack(x0, x1, x2, x3):
                        v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                        return vector.bitcast(vec8_i32, v4)

                    mfma_res_ty = vec4_f32
                    gb128 = _pack(gate_b_single[0], gate_b_single[1], c0_i64, c0_i64)
                    if not gate_only:
                        ub128 = _pack(up_b_single[0], up_b_single[1], c0_i64, c0_i64)

                    for mi_p in range_constexpr(m_repeat_packed):
                        a_scale_val = a_scale_vals[mi_p]
                        for imxdl in range_constexpr(pack_M):
                            mi_idx = mi_p * pack_M + imxdl
                            a_reg = all_a_tiles[k_idx * m_repeat + mi_idx]

                            if is_f8_a:
                                a128 = _pack(a_reg[0], a_reg[1], a_reg[2], a_reg[3])
                            else:
                                a128 = _pack(a_reg[0], a_reg[1], c0_i64, c0_i64)

                            acc_idx = mi_idx * num_acc_n + ni_idx
                            gate_list[acc_idx] = (
                                rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                    mfma_res_ty,
                                    [
                                        a128,
                                        gb128,
                                        gate_list[acc_idx],
                                        cbsz,
                                        blgp,
                                        ikxdl * pack_M + imxdl,
                                        a_scale_val,
                                        ikxdl * pack_N + inxdl,
                                        gate_bs_val,
                                    ],
                                )
                            )
                            if not gate_only:
                                up_list[acc_idx] = (
                                    rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                        mfma_res_ty,
                                        [
                                            a128,
                                            ub128,
                                            up_list[acc_idx],
                                            cbsz,
                                            blgp,
                                            ikxdl * pack_M + imxdl,
                                            a_scale_val,
                                            ikxdl * pack_N + inxdl,
                                            up_bs_val,
                                        ],
                                    )
                                )

                def _interleaved_half(
                    lds_read, lds_write, next_k_dma_py, next_k_load,
                    prev_a_tile, prev_gate_w, prev_up_w, prev_a_scale,
                    prev_gate_bs, prev_up_bs, acc_gate, acc_up,
                ):
                    """One flatmm-style interleaved half-iteration (deep pipeline).

                    Generalized for arbitrary m_repeat (block_m=32, 64, ...).
                    DMA targets lds_write (OTHER buffer) while ds_read uses
                    lds_read (already DMA'd in previous half).

                    Interleaving schedule (per half):
                      Phase 0: scale VMEM + 2 ds_read(A) → 4 MFMA(prev)
                      Phase 1..N: B VMEM(distributed) + 2 ds_read(A, if avail) → 4 MFMA(prev)
                      Phase N+1..: remaining B VMEM → 4 MFMA(prev)
                    """
                    _abs_k = k_base_idx + arith.constant(next_k_load, index=True)
                    _bk = _abs_k // arith.constant(2, index=True)
                    _sk = _abs_k // arith.constant(pack_K * 128, index=True)
                    _k_off = _sk * layout_b_scale.stride_k0

                    rocdl.sched_barrier(0)
                    rocdl.s_waitcnt(3)
                    _barrier()
                    rocdl.sched_barrier(0)

                    # DMA A to OTHER buffer (for next half), non-blocking
                    _abs_k_dma = k_base_idx + arith.constant(next_k_dma_py, index=True)
                    if use_async_copy and next_k_dma_py < int(_k_dim):
                        prefetch_x_to_lds(_abs_k_dma, lds_write)
                    if not use_async_copy:
                        _x_regs = load_x_tile(_abs_k_dma)

                    # ---- Extract previous scale values ----
                    _prev_asvs = []
                    for _mi_p in range_constexpr(m_repeat_packed):
                        _prev_asvs.append(vector.extract(
                            prev_a_scale[_mi_p],
                            static_position=[0], dynamic_position=[],
                        ))
                    _prev_gsv = vector.extract(
                        prev_gate_bs[0], static_position=[0], dynamic_position=[],
                    )
                    if not gate_only:
                        _prev_usv = vector.extract(
                            prev_up_bs[0], static_position=[0], dynamic_position=[],
                        )

                    # ---- Execute phases from unified schedule ----
                    _a_all = {}
                    _b_gate_all = {}
                    _b_up_all = {}

                    for _p in range_constexpr(_pipe_n_phases):
                        # Scale VMEM loads (phase 0 only)
                        if _pp_has_scale[_p]:
                            _new_as_list = []
                            for _mi_p in range_constexpr(m_repeat_packed):
                                _raw_as = buffer_ops.buffer_load(
                                    sx_rsrc, _a_scale_bases[_mi_p] + _k_off,
                                    vec_width=1, dtype=T.i32, cache_modifier=0,
                                )
                                _new_as_list.append(_rearrange_a_scale(_raw_as))
                            _new_gs = buffer_ops.buffer_load(
                                sw_rsrc, _gate_scale_bases[0] + _k_off,
                                vec_width=1, dtype=T.i32, cache_modifier=0,
                            )
                            _new_gs = _rearrange_b_scale(_new_gs)
                            if not gate_only:
                                _new_us = buffer_ops.buffer_load(
                                    sw_rsrc, _up_scale_bases[0] + _k_off,
                                    vec_width=1, dtype=T.i32, cache_modifier=0,
                                )
                                _new_us = _rearrange_b_scale(_new_us)

                        # B VMEM loads
                        for _b_j in range_constexpr(len(_pp_b_loads[_p])):
                            _b_type, _b_ku, _b_ni = _pp_b_loads[_p][_b_j]
                            if _b_type == 'gate':
                                _b_gate_all[(_b_ku, _b_ni)] = load_b_packs_k64(
                                    _bk, _b_ku,
                                    gate_n_blk_list[_b_ni],
                                    gate_n_intra_list[_b_ni],
                                )
                            else:
                                _b_up_all[(_b_ku, _b_ni)] = load_b_packs_k64(
                                    _bk, _b_ku,
                                    up_n_blk_list[_b_ni],
                                    up_n_intra_list[_b_ni],
                                )

                        # A ds_reads
                        rocdl.sched_barrier(0)
                        for _a_j in range_constexpr(len(_pp_a_reads[_p])):
                            _ak, _ami = _pp_a_reads[_p][_a_j]
                            _a_all[(_ak, _ami)] = load_a_subtile(
                                _ak, _ami, lds_read,
                            )
                        rocdl.sched_barrier(0)

                        # MFMAs on prev data
                        rocdl.s_setprio(1)
                        for _m_j in range_constexpr(len(_pp_mfma[_p])):
                            _k_idx, _ni_idx, _ikxdl, _inxdl, _ku128 = _pp_mfma[_p][_m_j]
                            _up_b_single = (
                                (prev_up_w[_k_idx][0][_ni_idx], prev_up_w[_k_idx][1][_ni_idx])
                                if not gate_only else None
                            )
                            compute_bmajor_mfma_phase(
                                prev_a_tile,
                                (prev_gate_w[_k_idx][0][_ni_idx], prev_gate_w[_k_idx][1][_ni_idx]),
                                _up_b_single,
                                _prev_asvs, _prev_gsv,
                                _prev_usv if not gate_only else None,
                                acc_gate, acc_up,
                                _k_idx, _ni_idx, _ikxdl, _inxdl,
                            )
                        rocdl.s_setprio(0)
                        rocdl.sched_barrier(0)

                    # ---- Assemble loaded data for next half-iteration ----
                    cur_a_tile = []
                    for _k in range_constexpr(k_unroll):
                        for _mi in range_constexpr(m_repeat):
                            cur_a_tile.append(_a_all[(_k, _mi)])

                    cur_gate_w = []
                    cur_up_w = None if gate_only else []
                    for ku in range_constexpr(k_unroll):
                        g_packs0, g_packs1 = [], []
                        u_packs0, u_packs1 = [], []
                        for ni in range_constexpr(num_acc_n):
                            g = _b_gate_all[(ku, ni)]
                            g_packs0.append(g[0])
                            g_packs1.append(g[1])
                            if not gate_only:
                                u = _b_up_all[(ku, ni)]
                                u_packs0.append(u[0])
                                u_packs1.append(u[1])
                        cur_gate_w.append((g_packs0, g_packs1))
                        if not gate_only:
                            cur_up_w.append((u_packs0, u_packs1))

                    cur_a_scale = []
                    for _mi_p in range_constexpr(m_repeat_packed):
                        cur_a_scale.append(vector.from_elements(
                            T.vec(1, T.i32), [_new_as_list[_mi_p]],
                        ))
                    cur_gate_bs = [
                        vector.from_elements(T.vec(1, T.i32), [_new_gs])
                    ]
                    if not gate_only:
                        cur_up_bs = [
                            vector.from_elements(T.vec(1, T.i32), [_new_us])
                        ]
                    else:
                        cur_up_bs = None

                    if not use_async_copy:
                        store_x_tile_to_lds(_x_regs, lds_write)

                    return (
                        cur_a_tile, cur_gate_w, cur_up_w,
                        cur_a_scale, cur_gate_bs, cur_up_bs,
                        acc_gate, acc_up,
                    )

                # Pipeline (split ping/pong allocators)
                rocdl.sched_barrier(0)

                k0 = k_base_idx
                if use_async_copy:
                    prefetch_x_to_lds(k0, lds_x_pong)
                else:
                    x_regs0 = load_x_tile(k0)
                    store_x_tile_to_lds(x_regs0, lds_x_pong)
                rocdl.sched_barrier(0)
                _k0_scale = k_base_idx // arith.constant(pack_K * 128, index=True)
                a_scale_pong, gate_bs_pong, up_bs_pong = prefetch_ab_scale_tile(
                    _k0_scale
                )
                _c_tile_m_idx = arith.constant(tile_m, index=True)
                _tid_in_range = arith.cmpi(CmpIPredicate.ult, tx, _c_tile_m_idx)
                _if_tid = scf.IfOp(_tid_in_range)
                with ir.InsertionPoint(_if_tid.then_block):
                    _tid_row = bx_m + tx
                    _tid_val = buffer_ops.buffer_load(
                        sorted_rsrc, _tid_row, vec_width=1, dtype=T.i32
                    )
                    _tid_vec1 = vector.from_elements(T.vec(1, T.i32), [_tid_val])
                    vector.store(_tid_vec1, lds_tid, [tx])
                    scf.YieldOp([])

                acc_gate = [acc_init] * num_acc_n * m_repeat
                acc_up = [acc_init] * num_acc_n * m_repeat if not gate_only else None


                _k1 = k_base_idx + arith.constant(tile_k, index=True)
                rocdl.sched_barrier(0)
                if use_async_copy:
                    prefetch_x_to_lds(_k1, lds_x_ping)
                else:
                    _x_regs_prime = load_x_tile(_k1)
                    store_x_tile_to_lds(_x_regs_prime, lds_x_ping)

                _k0_b = k_base_idx // arith.constant(2, index=True)
                gate_w0, up_w0 = load_b_tile(_k0_b)
                # Prime the deep pipeline: DMA K=tile_k → ping (1 tile ahead)
                # rocdl.s_waitcnt(8)
                gpu.barrier()
                rocdl.sched_barrier(0)
                a_tile_pong = prefetch_full_a_from_lds(lds_x_pong)

                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(6)


                num_k_tiles_py = int(_k_dim) // int(tile_k)
                odd_k_tiles = (num_k_tiles_py % 2) == 1
                tail_tiles = 1 if odd_k_tiles else 2
                k_main2_py = (num_k_tiles_py - tail_tiles) * int(tile_k)
                if k_main2_py < 0:
                    k_main2_py = 0

                c2_tile_k = arith.constant(tile_k * 2, index=True)
                gate_w_pong = gate_w0
                up_w_pong = up_w0

                def _sched_hints_stage1_gate_up():
                    """Stage1 hot-loop scheduler adapted from the gate/up gufusion pipeline.

                    The original hot loop doubles the B-side VMEM and MFMA streams:
                    - gate B load + up B load
                    - gate B-scale load + up B-scale load
                    - gate MFMA + up MFMA

                    The scheduler API here is less expressive than the original
                    `__builtin_amdgcn_sched_group_barrier`, so we encode the same
                    idea with a compact heuristic:
                    - always double MFMA groups (`num_acc_n * 2`)
                    - use 2 VMEM groups only when the N tile is wide enough to
                      sustain the extra B-side traffic (`num_acc_n >= 4`)
                    - otherwise keep 1 VMEM group to avoid over-throttling the
                      smaller `tile_n=128` kernels
                    """
                    # mfma_group = num_acc_n * 2
                    # mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                    # mfma_per_iter = 2 * mfma_group
                    # sche_iters = (
                    #     0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                    # )

                    # # Approximate the doubled B-side prefetch pressure.
                    # vmem_groups = 2 if int(num_acc_n) >= 4 else 1

                    # rocdl.sched_dsrd(2)
                    # rocdl.sched_mfma(2)
                    # rocdl.sched_dsrd(1)
                    # rocdl.sched_mfma(1)
                    # rocdl.sched_dsrd(1)
                    # rocdl.sched_mfma(1)

                    # dswr_tail = num_x_loads
                    # if dswr_tail > sche_iters:
                    #     dswr_tail = sche_iters
                    # dswr_start = sche_iters - dswr_tail

                    # for sche_i in range_constexpr(sche_iters):
                    #     rocdl.sched_vmem(vmem_groups)
                    #     rocdl.sched_mfma(mfma_group)
                    #     rocdl.sched_dsrd(1)
                    #     rocdl.sched_mfma(mfma_group)
                    #     if sche_i >= dswr_start - 1:
                    #         rocdl.sched_dswr(1)
                    # rocdl.sched_barrier(0)

                    if use_async_copy:
                        a_vmem_load = max(1, tile_m // 32)
                        mfma_group = a_vmem_load
                        rocdl.sched_vmem(a_vmem_load)

                        rocdl.sched_mfma(mfma_group)

                        ds_read_count = max(2, a_vmem_load * 4)

                        b_vmem_total = k_unroll * num_acc_n * 2
                        vmem_count = b_vmem_total + 2 + a_vmem_load

                        if tile_m == 16:
                            for i in range_constexpr(2):
                                rocdl.sched_dsrd(1)
                                rocdl.sched_mfma(1)
                                rocdl.sched_vmem(1)
                                rocdl.sched_mfma(1)
                            for i in range_constexpr(9):
                                rocdl.sched_vmem(1)
                                rocdl.sched_mfma(1)
                        else:
                            for i in range_constexpr(a_vmem_load * 4):
                                rocdl.sched_dsrd(1)
                                rocdl.sched_mfma(1)
                                rocdl.sched_vmem(1)
                                rocdl.sched_mfma(mfma_group)

                            if tile_m == 32:
                                for i in range_constexpr(vmem_count - a_vmem_load * 4):
                                    rocdl.sched_vmem(1)
                                    rocdl.sched_mfma(mfma_group)
                            elif tile_m == 64:
                                rocdl.sched_vmem(1)
                                rocdl.sched_mfma(1)
                                rocdl.sched_vmem(1)
                                rocdl.sched_mfma(2)
                                rocdl.sched_vmem(1)
                                rocdl.sched_mfma(1)
                                rocdl.sched_vmem(1)
                                rocdl.sched_mfma(2)

                    rocdl.sched_barrier(0)
                
                rocdl.sched_barrier(0)

                if k_main2_py > 0:
                    for k_iv_py in range_constexpr(0, k_main2_py, tile_k * 2):
                        next_k_load_1 = k_iv_py + tile_k
                        next_k_load_2 = k_iv_py + tile_k * 2
                        next_k_dma_1 = k_iv_py + tile_k * 2
                        next_k_dma_2 = k_iv_py + tile_k * 3

                        # Half 1: read ping (DMA'd prev half), DMA→pong, MFMA(pong)
                        (
                            a_tile_ping, gate_w_ping, up_w_ping,
                            a_scale_ping, gate_bs_ping, up_bs_ping,
                            acc_gate, acc_up,
                        ) = _interleaved_half(
                            lds_x_ping, lds_x_pong,
                            next_k_dma_1, next_k_load_1,
                            a_tile_pong, gate_w_pong, up_w_pong,
                            a_scale_pong, gate_bs_pong, up_bs_pong,
                            acc_gate, acc_up,
                        )

                        # Half 2: read pong (DMA'd Half 1), DMA→ping, MFMA(ping)
                        (
                            a_tile_pong, gate_w_pong, up_w_pong,
                            a_scale_pong, gate_bs_pong, up_bs_pong,
                            acc_gate, acc_up,
                        ) = _interleaved_half(
                            lds_x_pong, lds_x_ping,
                            next_k_dma_2, next_k_load_2,
                            a_tile_ping, gate_w_ping, up_w_ping,
                            a_scale_ping, gate_bs_ping, up_bs_ping,
                            acc_gate, acc_up,
                        )

                # _wave_mod2_b = wave_id % arith.constant(2, index=True)
                # _wave_odd = arith.cmpi(
                #     CmpIPredicate.eq, _wave_mod2_b, arith.constant(1, index=True)
                # )
                # _if_wave_odd = scf.IfOp(_wave_odd)
                # with ir.InsertionPoint(_if_wave_odd.then_block):
                #     # gpu.barrier()
                #     _barrier()
                #     scf.YieldOp([])

                if odd_k_tiles:
                    acc_gate, acc_up, epilogue_pf = compute_tile(
                        acc_gate,
                        acc_up,
                        gate_w_pong,
                        up_w_pong,
                        a_tile_pong,
                        a_scale_pong,
                        gate_bs_pong,
                        up_bs_pong,
                        prefetch_epilogue=True,
                    )
                else:
                    _k_tail_rel = arith.constant(_k_dim - tile_k, index=True)
                    k_tail1 = k_base_idx + _k_tail_rel
                    if use_async_copy:
                        prefetch_x_to_lds(k_tail1, lds_x_ping)
                    else:
                        x_regs_ping = load_x_tile(k_tail1)
                    gate_w_ping, up_w_ping = load_b_tile(
                        k_tail1 // arith.constant(2, index=True)
                    )
                    a_scale_ping, gate_bs_ping, up_bs_ping = prefetch_ab_scale_tile(
                        k_tail1 // arith.constant(pack_K * 128, index=True)
                    )
                    acc_gate, acc_up, _ = compute_tile(
                        acc_gate,
                        acc_up,
                        gate_w_pong,
                        up_w_pong,
                        a_tile_pong,
                        a_scale_pong,
                        gate_bs_pong,
                        up_bs_pong,
                    )
                    if not use_async_copy:
                        store_x_tile_to_lds(x_regs_ping, lds_x_ping)
                    rocdl.s_waitcnt(0)
                    _barrier()
                    a_tile_ping = prefetch_full_a_from_lds(lds_x_ping)
                    acc_gate, acc_up, epilogue_pf = compute_tile(
                        acc_gate,
                        acc_up,
                        gate_w_ping,
                        up_w_ping,
                        a_tile_ping,
                        a_scale_ping,
                        gate_bs_ping,
                        up_bs_ping,
                        prefetch_epilogue=True,
                    )

                # silu(gate) * up in f32 before epilogue
                # silu(x) = x * sigmoid(x); use HW fast path: exp2, rcp
                def _silu_mul_vec4(gate_v4, up_v4):
                    """Element-wise silu(gate) * up on vec4_f32."""
                    result_elems = []
                    for ei in range_constexpr(4):
                        g = vector.extract(
                            gate_v4, static_position=[ei], dynamic_position=[]
                        )
                        u = vector.extract(
                            up_v4, static_position=[ei], dynamic_position=[]
                        )
                        neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                        t = g * neg_log2e
                        emu = llvm.call_intrinsic(
                            f32, "llvm.amdgcn.exp2.f32", [t], [], []
                        )
                        one = arith.constant(1.0, type=f32)
                        den = one + emu
                        sig = llvm.call_intrinsic(
                            f32, "llvm.amdgcn.rcp.f32", [den], [], []
                        )
                        result_elems.append(g * sig * u)
                    return vector.from_elements(vec4_f32, result_elems)

                if not _is_splitk:
                    acc = [None] * (int(num_acc_n) * int(m_repeat))
                    for _mi in range_constexpr(m_repeat):
                        for _ni in range_constexpr(num_acc_n):
                            _aidx = _mi * num_acc_n + _ni
                            acc[_aidx] = _silu_mul_vec4(acc_gate[_aidx], acc_up[_aidx])

                # ---- Epilogue: CShuffle + direct store (accumulate=False) ----
                # Output: out[(t*topk+s) * inter_dim + col] = silu(gate) * up
                # For split-K: skip silu, output gate/up separately with atomic add
                tw_pf = None
                if epilogue_pf is not None:
                    _, tw_pf, _ = epilogue_pf

                mask24_i32 = arith.constant(0xFFFFFF)
                topk_i32_v = topk_i32
                tokens_i32_v = tokens_i32

                from flydsl._mlir.dialects import fly as _fly

                _llvm_ptr_ty = ir.Type.parse("!llvm.ptr")
                out_base_ptr = _fly.extract_aligned_pointer_as_index(
                    _llvm_ptr_ty, arg_out
                )
                out_base_i64 = llvm.ptrtoint(T.i64, out_base_ptr)
                out_base_idx = arith.index_cast(ir.IndexType.get(), out_base_i64)

                if lds_out is None:
                    raise RuntimeError("CShuffle epilogue requires lds_out")

                _apply_weight = doweight_stage1 and not _is_splitk

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
                    if _apply_weight:
                        tw_idx = (mi * 4) + ii
                        if tw_pf is not None:
                            tw = tw_pf[tw_idx]
                        else:
                            tw = buffer_ops.buffer_load(
                                sorted_w_rsrc, row, vec_width=1, dtype=f32
                            )
                    for ni in range_constexpr(num_acc_n):
                        col_local = col_base_local + (ni * 16)
                        acc_idx = mi * num_acc_n + ni
                        v = vector.extract(
                            acc[acc_idx], static_position=[ii], dynamic_position=[]
                        )
                        if _apply_weight:
                            v = v * tw
                        if _need_quant:
                            lds_idx = row_base_lds + col_local
                            vec1_f32 = T.vec(1, f32)
                            v1 = vector.from_elements(vec1_f32, [v])
                            vector.store(v1, lds_out, [lds_idx], alignment=4)
                        else:
                            v_out = arith.trunc_f(out_elem(), v)
                            lds_idx = row_base_lds + col_local
                            vec1_out = T.vec(1, out_elem())
                            v1 = vector.from_elements(vec1_out, [v_out])
                            vector.store(v1, lds_out, [lds_idx], alignment=2)

                _out_row_stride = (
                    inter_dim * 2 * out_elem_bytes if _is_splitk
                    else (inter_dim // 2 if _need_quant else inter_dim * out_elem_bytes)
                )

                def precompute_row(*, row_local, row):
                    fused2 = memref.load(lds_tid, [row_local])
                    row_i32 = arith.index_cast(T.i32, row)
                    row_valid0 = arith.cmpi(CmpIPredicate.ult, row_i32, num_valid_i32)
                    t = fused2 & mask24_i32
                    s = fused2 >> 24
                    t_ok = arith.cmpi(CmpIPredicate.ult, t, tokens_i32_v)
                    s_ok = arith.cmpi(CmpIPredicate.ult, s, topk_i32_v)
                    row_valid = arith.andi(row_valid0, arith.andi(t_ok, s_ok))
                    t_idx = arith.index_cast(ir.IndexType.get(), t)
                    s_idx = arith.index_cast(ir.IndexType.get(), s)
                    ts_idx = t_idx * arith.constant(topk, index=True) + s_idx
                    row_byte_base = out_base_idx + ts_idx * arith.constant(
                        _out_row_stride, index=True
                    )
                    return ((fused2, row_byte_base), row_valid)

                def _idx_to_llvm_ptr(idx_val, addr_space=1):
                    idx_v = idx_val._value if hasattr(idx_val, "_value") else idx_val
                    i64_v = arith.index_cast(T.i64, idx_v)
                    i64_raw = i64_v._value if hasattr(i64_v, "_value") else i64_v
                    ptr_ty = ir.Type.parse(f"!llvm.ptr<{addr_space}>")
                    return llvm.inttoptr(ptr_ty, i64_raw)

                _e_vec = _e_vec_s1
                _e_vec_sk = 2
                _cshuffle_nlane = min(32, tile_n // _e_vec)
                _cshuffle_nlane_sk = min(32, tile_n // _e_vec_sk)
                _num_threads_per_quant_blk = _num_threads_per_quant_blk_s1

                _c0_i32 = arith.constant(0, type=T.i32)
                _c1_i32 = arith.constant(1, type=T.i32)
                _c2_i32 = arith.constant(2, type=T.i32)
                _c3_i32 = arith.constant(3, type=T.i32)
                _c4_i32 = arith.constant(4, type=T.i32)
                _c5_i32 = arith.constant(5, type=T.i32)
                _c7_i32 = arith.constant(7, type=T.i32)
                _c15_i32 = arith.constant(15, type=T.i32)
                _c21_i32 = arith.constant(21, type=T.i32)
                _c23_i32 = arith.constant(23, type=T.i32)
                _c28_i32 = arith.constant(28, type=T.i32)
                _c31_i32 = arith.constant(31, type=T.i32)
                _c32_i32 = arith.constant(32, type=T.i32)
                _c64_i32 = arith.constant(64, type=T.i32)
                _c126_i32 = arith.constant(126, type=T.i32)
                _c127_i32 = arith.constant(127, type=T.i32)
                _c254_i32 = arith.constant(254, type=T.i32)
                _c256_i32 = arith.constant(256, type=T.i32)
                _c0xFF_i32 = arith.constant(0xFF, type=T.i32)
                _c0x200000_i32 = arith.constant(0x200000, type=T.i32)
                _c0xFF800000_i32 = arith.constant(0xFF800000, type=T.i32)
                _c0x400000_i32 = arith.constant(0x400000, type=T.i32)
                _c0x7FFFFF_i32 = arith.constant(0x7FFFFF, type=T.i32)
                _c0x80000000_i32 = arith.constant(0x80000000, type=T.i32)
                _c0_f32 = arith.constant(0.0, type=T.f32)

                def _f32_to_e2m1(qx_f32):
                    """Convert a scaled f32 value to fp4 (e2m1) 4-bit integer."""
                    qx = qx_f32.bitcast(T.i32)
                    s = qx & _c0x80000000_i32
                    e = (qx >> _c23_i32) & _c0xFF_i32
                    m = qx & _c0x7FFFFF_i32
                    adj_exp = arith.maxsi(_c126_i32 - e, _c0_i32)
                    m_denorm = (_c0x400000_i32 | (m >> _c1_i32)) >> adj_exp
                    is_denorm = arith.cmpi(CmpIPredicate.ult, e, _c127_i32)
                    m = arith.select(is_denorm, m_denorm, m)
                    e = arith.maxsi(e - _c126_i32, _c0_i32)
                    combined = (e << _c2_i32) | (m >> _c21_i32)
                    rounded = (combined + _c1_i32) >> _c1_i32
                    e2m1 = arith.minui(rounded, _c7_i32)
                    return (s >> _c28_i32) | e2m1

                if _need_sort:
                    _n32_sort = _sorted_scale_cols_i32 * _c32_i32

                # Mutable slot for split-K N-offset (gate=0, up=inter_dim)
                _sk_n_offset = [0]

                def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                    fused, row_byte_base = row_ctx
                    if _need_quant and not _is_splitk:
                        frag_vals = []
                        for i in range_constexpr(_e_vec):
                            frag_vals.append(vector.extract(frag, static_position=[i], dynamic_position=[]))

                        local_max = _c0_f32
                        for i in range_constexpr(_e_vec):
                            abs_v = llvm.call_intrinsic(
                                f32, "llvm.fabs.f32", [frag_vals[i]], [], []
                            )
                            local_max = arith.maximumf(local_max, abs_v)

                        for _si in range_constexpr(_num_shuffle_steps_s1):
                            off = arith.constant(_shuffle_dists_s1[_si], type=T.i32)
                            peer = local_max.shuffle_xor(off, _c64_i32)
                            local_max = arith.maximumf(local_max, peer)

                        max_i32 = local_max.bitcast(T.i32)
                        max_rounded = (max_i32 + _c0x200000_i32) & _c0xFF800000_i32
                        exp_field = max_rounded >> _c23_i32
                        e8m0_biased = arith.maxsi(exp_field - _c2_i32, _c0_i32)

                        quant_exp = _c254_i32 - e8m0_biased
                        quant_scale = (quant_exp << _c23_i32).bitcast(T.f32)

                        fp4_vals = []
                        for i in range_constexpr(_e_vec):
                            scaled_v = frag_vals[i] * quant_scale
                            fp4_vals.append(_f32_to_e2m1(scaled_v))

                        packed_i32 = fp4_vals[0] | (fp4_vals[1] << _c4_i32)
                        for k in range_constexpr(1, _e_vec // 2):
                            byte_k = fp4_vals[2 * k] | (fp4_vals[2 * k + 1] << _c4_i32)
                            packed_i32 = packed_i32 | (byte_k << arith.constant(k * 8, type=T.i32))

                        ptr_addr_idx = row_byte_base + col_g0 / arith.constant(2, index=True)
                        out_ptr_v = _idx_to_llvm_ptr(ptr_addr_idx)
                        _pack_bytes = _e_vec // 2
                        if _pack_bytes == 1:
                            store_val = arith.TruncIOp(T.i8, packed_i32)
                            store_raw = store_val._value if hasattr(store_val, "_value") else store_val
                            llvm.StoreOp(store_raw, out_ptr_v, alignment=1, nontemporal=True)
                        elif _pack_bytes == 2:
                            store_val = arith.TruncIOp(T.i16, packed_i32)
                            store_raw = store_val._value if hasattr(store_val, "_value") else store_val
                            llvm.StoreOp(store_raw, out_ptr_v, alignment=2, nontemporal=True)
                        else:
                            packed_raw = packed_i32._value if hasattr(packed_i32, "_value") else packed_i32
                            llvm.StoreOp(packed_raw, out_ptr_v, alignment=4, nontemporal=True)

                        if _need_sort:
                            col_g0_i32 = arith.index_cast(T.i32, col_g0)
                            is_scale_writer = arith.cmpi(CmpIPredicate.eq,
                                col_g0_i32 & _c31_i32, _c0_i32)
                            _if_scale = scf.IfOp(is_scale_writer)
                            with ir.InsertionPoint(_if_scale.then_block):
                                row_i32_s = arith.index_cast(T.i32, row)
                                col_s_i32 = col_g0_i32 >> _c5_i32
                                d0 = row_i32_s >> _c5_i32
                                d1 = (row_i32_s >> _c4_i32) & _c1_i32
                                d2 = row_i32_s & _c15_i32
                                d3 = col_s_i32 >> _c3_i32
                                d4 = (col_s_i32 >> _c2_i32) & _c1_i32
                                d5 = col_s_i32 & _c3_i32
                                byte_off = (d0 * _n32_sort + d3 * _c256_i32
                                            + d5 * _c64_i32 + d2 * _c4_i32
                                            + d4 * _c2_i32 + d1)
                                e8m0_i8 = arith.TruncIOp(T.i8, e8m0_biased)
                                buffer_ops.buffer_store(e8m0_i8, sorted_scale_rsrc, byte_off, offset_is_bytes=True)
                                scf.YieldOp([])
                    elif _is_splitk:
                        col_idx = col_g0 + arith.constant(_sk_n_offset[0], index=True)
                        byte_off_col = col_idx * arith.constant(out_elem_bytes, index=True)
                        ptr_addr_idx = row_byte_base + byte_off_col
                        out_ptr_v = _idx_to_llvm_ptr(ptr_addr_idx)
                        frag_v = frag._value if hasattr(frag, "_value") else frag
                        llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.fadd,
                            out_ptr_v,
                            frag_v,
                            llvm.AtomicOrdering.monotonic,
                            syncscope="agent",
                            alignment=_e_vec_sk * out_elem_bytes,
                        )
                    else:
                        col_idx = col_g0
                        byte_off_col = col_idx * arith.constant(out_elem_bytes, index=True)
                        ptr_addr_idx = row_byte_base + byte_off_col
                        out_ptr_v = _idx_to_llvm_ptr(ptr_addr_idx)
                        frag_v = frag._value if hasattr(frag, "_value") else frag
                        llvm.StoreOp(
                            frag_v,
                            out_ptr_v,
                            alignment=_e_vec * out_elem_bytes,
                            nontemporal=True,
                        )

                _frag_elem = ir.F32Type.get() if _need_quant else (ir.BF16Type.get() if out_is_bf16 else ir.F16Type.get())

                if gate_only:
                    # gate_only: single pass, by_n covers full [0, 2*inter_dim)
                    _eff_e_vec = _e_vec_sk
                    acc = acc_gate
                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=_eff_e_vec,
                        cshuffle_nlane=_cshuffle_nlane_sk,
                        block_size=total_threads,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=_frag_elem,
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                    )
                elif _is_splitk:
                    # Two-pass epilogue: gate then up, each with atomic add
                    _eff_e_vec = _e_vec_sk

                    # Pass 1: gate
                    acc = acc_gate
                    _sk_n_offset[0] = 0
                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=_eff_e_vec,
                        cshuffle_nlane=_cshuffle_nlane_sk,
                        block_size=total_threads,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=_frag_elem,
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                    )

                    gpu.barrier()

                    # Pass 2: up
                    acc = acc_up
                    _sk_n_offset[0] = inter_dim
                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=_eff_e_vec,
                        cshuffle_nlane=_cshuffle_nlane_sk,
                        block_size=total_threads,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=_frag_elem,
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                    )
                else:
                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=_e_vec,
                        cshuffle_nlane=_cshuffle_nlane,
                        block_size=total_threads,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=_frag_elem,
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                    )

            _if_blk = scf.IfOp(blk_valid)
            with ir.InsertionPoint(_if_blk.then_block):
                _ifexpert_of = scf.IfOp(exp_valid)
                with ir.InsertionPoint(_ifexpert_of.then_block):
                    _moe_gemm1_body()
                    scf.YieldOp([])
                scf.YieldOp([])

            gpu.barrier()
            scf.YieldOp([])
            _for_ip.__exit__(None, None, None)

    # -- Host launcher --
    _cache_tag = (
        module_name,
        a_dtype,
        b_dtype,
        out_dtype,
        tile_m,
        tile_n,
        tile_k,
        doweight_stage1,
        act,
        enable_bias,
        model_dim_pad,
        inter_dim_pad,
        use_cshuffle_epilog,
        persist_m,
        fuse_fp4_quant,
        fuse_sort_scale,
        use_async_copy,
        waves_per_eu,
        k_batch,
        gate_only,
    )

    @flyc.jit
    def launch_mixed_moe_gemm1(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_max_token_ids: fx.Tensor,
        arg_bias: fx.Tensor,
        arg_out_scale_sorted: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        _ = _cache_tag
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()

        inter_in = arith.index_cast(ir.IndexType.get(), i32_inter_in.ir_value())
        if gate_only:
            gx = inter_in / arith.constant(tile_n, index=True)
        else:
            gx = (
                inter_in
                / arith.constant(2, index=True)
                / arith.constant(tile_n, index=True)
            )
        _c_pm_l = arith.constant(persist_m, index=True)
        gy = (
            arith.index_cast(ir.IndexType.get(), i32_size_expert_ids_in.ir_value())
            + _c_pm_l
            - arith.constant(1, index=True)
        ) / _c_pm_l

        launcher = moe_gemm1(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_max_token_ids,
            arg_bias,
            arg_out_scale_sorted,
            i32_tokens_in,
            i32_inter_in,
            i32_k_in,
            i32_size_expert_ids_in,
        ).launch(grid=(gx, gy, k_batch), block=(total_threads, 1, 1), stream=stream)

    return launch_mixed_moe_gemm1


@functools.lru_cache(maxsize=None)
def compile_mixed_moe_gemm2(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    # Optional experiment: write per-(token,slot) output (no atomics) into an output shaped
    # [tokens*topk, model_dim] (or [tokens, topk, model_dim] flattened), then reduce over topk outside.
    # This can reduce atomic contention for small tokens at the cost of extra bandwidth / reduction.
    accumulate: bool = True,
    enable_bias: bool = False,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    persist_m: int = 4,
    sort_block_m: int = 0,
):
    """Compile stage2 kernel (`moe_gemm2`) and return the compiled executable.

    a_dtype:
      - "fp8": A2 is fp8
      - "fp16": A2 is fp16 (caller uses tile_k halved vs fp8 to match MFMA K halving)
      - "int8": A2 is int8
      - "fp4": A2 is fp4

    b_dtype:
      - "fp8": W is fp8
      - "fp16": W is fp16 (caller uses tile_k halved vs fp8 to match MFMA K halving)
      - "int8": W is int8
      - "int4": W4A8 path: A2 is int8, W is packed int4 (2 values per byte) unpacked to int8 in-kernel
      - "fp4": W is fp4

    Stage2 output supports:
      - out_dtype="f16": fp16 half2 atomics (fast, can overflow to +/-inf for bf16 workloads)
      - out_dtype="f32": fp32 scalar atomics (slower, but avoids fp16 atomic overflow)

    `use_cshuffle_epilog` controls whether we use the LDS CShuffle epilogue before
    global atomics (recommended for performance).

    `sort_block_m` is the block_size used by moe_sorting / stage1. When 0 (default),
    assumed equal to `tile_m`. When set, stage2 can use a different tile_m from
    sorting/stage1. Requires sort_block_m % tile_m == 0.
    """
    _sort_block_m = tile_m if sort_block_m <= 0 else sort_block_m
    if _sort_block_m != tile_m and _sort_block_m % tile_m != 0:
        raise ValueError(
            f"sort_block_m ({_sort_block_m}) must be a multiple of tile_m ({tile_m})"
        )

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    _state = {}

    if a_dtype not in ("fp8", "fp16", "int8", "fp4"):
        raise ValueError(
            f"a_dtype must be one of ('fp8','fp16','int8','fp4'), got {a_dtype!r}"
        )
    if b_dtype not in ("fp8", "fp16", "int8", "int4", "fp4"):
        raise ValueError(
            f"b_dtype must be one of ('fp8','fp16','int8','int4','fp4'), got {b_dtype!r}"
        )

    is_f16_a = a_dtype == "fp16"
    is_f16_b = b_dtype == "fp16"
    is_f16 = is_f16_a or is_f16_b

    is_f8_a = a_dtype == "fp8"
    is_f4_a = a_dtype == "fp4"
    is_f4_b = b_dtype == "fp4"

    _scale_pack_m = 2  # physical mn_pack in preshuffle microscale layout
    _scale_pack_n = 2
    _scale_pack_k = 2  # physical k_pack in preshuffle scale layout
    pack_M = min(_scale_pack_m, tile_m // 16)
    pack_N = min(_scale_pack_n, tile_n // 64)
    _k_unroll_raw = (int(tile_k) * (2 if a_dtype == "fp16" else 1)) // 128
    pack_K = min(_scale_pack_k, _k_unroll_raw)

    elem_bytes = 1

    a_elem_bytes = 2 if is_f16_a else 1
    b_elem_bytes = 1
    tile_k_bytes = int(tile_k) * int(a_elem_bytes)

    a_elem_vec_pack = 2 if is_f4_a else 1
    cbsz = 0 if is_f8_a else 4
    blgp = 4

    # ---- Static B preshuffle strides (compile-time) ----
    # All values below are Python ints computable at kernel-compile time.
    # Using them in an explicit multiply-add replaces the fly dialect's
    # dynamic ``crd2idx`` path which emits Barrett reduction for the
    # non-power-of-2 ``n0 = experts*model_dim//16`` shape.
    _b_kpack_bytes_s = 8 if (b_dtype == "int4") else 16
    _b_kpack_elems_s = _b_kpack_bytes_s // b_elem_bytes
    _b_c_k_s = inter_dim // _scale_pack_k
    _b_c_k0_s = (_b_c_k_s * b_elem_bytes) // 64
    _b_stride_nlane = _b_kpack_elems_s  # 16
    _b_stride_klane = 16 * _b_stride_nlane  # 256
    _b_stride_k0 = 4 * _b_stride_klane  # 1024
    _b_stride_n0 = _b_c_k0_s * _b_stride_k0  # c_k0 * 1024
    assert model_dim % 16 == 0, "model_dim must be divisible by 16"
    _expert_b_stride = (model_dim // 16) * _b_stride_n0

    # K64-byte micro-step: always 64 bytes per `ku`. For fp16, this is 32 elements (2xK16 MFMA).
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={a_elem_bytes})"
        )

    out_s = str(out_dtype).strip().lower()
    if out_s not in ("f16", "fp16", "half", "bf16", "bfloat16", "f32", "fp32", "float"):
        raise ValueError(
            f"out_dtype must be 'f16', 'bf16', or 'f32', got {out_dtype!r}"
        )
    out_is_f32 = out_s in ("f32", "fp32", "float")
    out_is_bf16 = out_s in ("bf16", "bfloat16")
    if (not bool(accumulate)) and out_is_f32:
        raise ValueError(
            "compile_moe_gemm2(accumulate=False) only supports out_dtype in {'f16','bf16'}"
        )
    is_int4 = b_dtype == "int4"
    # INT4 here means W4A8: A2 is int8, W is packed int4 and unpacked to int8 in-kernel.
    is_int8 = False

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

    def _x_elem_type():
        if is_f4_b:
            return T.f8 if is_f8_a else T.i8
        return T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)

    def _w_elem_type():
        if is_f4_b:
            return T.i8
        return T.f16 if is_f16_b else (T.i8 if is_int8 else T.f8)

    def _scale_elem_type():
        return T.i32

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
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(a_elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={a_elem_bytes}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads

    _use_lds128 = os.environ.get("FLIR_CK_LDS128", "1") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    pad_k = 0 if _use_lds128 else 8
    lds_stride = tile_k + pad_k

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
    def out_elem():
        return T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)

    epilog_tag = "cshuffle"
    # IMPORTANT: include tiling in the module name to avoid accidentally reusing a compiled
    # binary for a different (tile_m, tile_n, tile_k) configuration.
    # See stage1 note: include ABI tag to prevent binary reuse across signature changes.
    # IMPORTANT: module name participates in the compiler cache key.
    # Dynamic-shape variant: safe to reuse across (tokens/sorted_size/size_expert_ids) at runtime.
    # Keep a distinct ABI tag so the compile cache never mixes with historical signatures.
    _sbm_tag = "" if _sort_block_m == tile_m else f"_sbm{_sort_block_m}"
    module_name = (
        f"mfma_moe2_a{a_dtype}_w{b_dtype}_{out_s}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_vscale_fix3_pm{persist_m}{_sbm_tag}"
    ).replace("-", "_")
    # -- LDS sizing (pure Python; no MLIR Context needed) ---------------------
    # Reuse a single allocation for both:
    # - ping-pong A2 tiles (2 * tile_m * lds_stride * elem_bytes bytes)
    # - epilogue CShuffle tile (tile_m * tile_n f16 -> 2 * tile_m * tile_n bytes)
    lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(a_elem_bytes)
    lds_out_bytes = (
        2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    )  # f16 bytes
    lds_tid_bytes = int(tile_m) * 4
    lds_total_bytes = max(lds_x_bytes, lds_out_bytes) + lds_tid_bytes
    lds_total_elems = lds_total_bytes if a_elem_bytes == 1 else (lds_total_bytes // 2)

    def x_lds_elem():
        return T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)

    lds_alloc_bytes = int(lds_total_elems) * int(a_elem_bytes)
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_alloc_bytes

    if True:

        @flyc.kernel
        def moe_gemm2(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_sorted_token_ids: fx.Tensor,
            arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor,
            arg_num_valid_ids: fx.Tensor,
            arg_bias: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_n_in: fx.Int32,
            i32_k_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
        ):
            arg_out = arg_out.value
            arg_x = arg_x.value
            arg_w = arg_w.value
            arg_scale_x = arg_scale_x.value
            arg_scale_w = arg_scale_w.value
            arg_sorted_token_ids = arg_sorted_token_ids.value
            arg_expert_ids = arg_expert_ids.value
            arg_sorted_weights = arg_sorted_weights.value
            arg_num_valid_ids = arg_num_valid_ids.value
            arg_bias = arg_bias.value

            tokens_in = arith.index_cast(ir.IndexType.get(), i32_tokens_in.ir_value())
            n_in = arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
            k_in = arith.index_cast(ir.IndexType.get(), i32_k_in.ir_value())
            size_expert_ids_in = arith.index_cast(
                ir.IndexType.get(), i32_size_expert_ids_in.ir_value()
            )
            x_elem = T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)
            # For int4, weights are stored as packed bytes (i8) and unpacked to i8 packs.
            w_elem = T.f16 if is_f16_b else (T.i8 if is_int8 else T.f8)
            f16 = T.f16
            f32 = T.f32
            i32 = T.i32
            i64 = T.i64
            vec4_f32 = T.vec(4, f32)
            vec4_i32 = T.vec(4, i32)
            vec1_f16 = T.vec(1, f16)
            vec2_f16 = T.vec(2, f16)
            vec4_f16 = T.vec(4, f16)
            vec16_elems = 16 if a_elem_bytes == 1 else 8
            vec8_elems = 8 if a_elem_bytes == 1 else 4
            vec4_elems = 4 if a_elem_bytes == 1 else 2
            vec8_x = T.vec(vec8_elems, x_elem)
            vec16_x = T.vec(vec16_elems, x_elem)
            vec1_i64 = T.vec(1, i64)
            vec2_i64 = T.vec(2, i64)

            acc_init = (
                arith.constant_vector(0, vec4_i32)
                if is_int8
                else arith.constant_vector(0.0, vec4_f32)
            )

            # A2 layout (flatten token-slot -> M; use i32 for fly.make_shape).
            topk_idx = arith.constant(topk, index=True)
            m_in = tokens_in * topk_idx
            # fly.make_shape requires i32/i64, not index
            m_i32_v = arith.index_cast(T.i32, m_in)
            k_i32_v = i32_k_in.ir_value()
            layout_x = fx.make_layout((m_i32_v, k_i32_v), stride=(k_i32_v, 1))

            # B preshuffle layout: [experts*model_dim, inter_dim]
            c_n_total = arith.constant(experts * model_dim, index=True)
            kpack_bytes = 8 if is_int4 else 16
            from .layout_utils import _div_pow2, _mod_pow2

            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=_div_pow2(k_in, _scale_pack_k),
                kpack_bytes=kpack_bytes,
                elem_bytes=b_elem_bytes,
            )
            layout_b = b_layout.layout_b
            c_k0 = _div_pow2(k_in * arith.constant(int(a_elem_bytes), index=True), 64)

            def check_c_n_valid_gate(base_n):
                return arith.cmpi(CmpIPredicate.ult, base_n, model_dim - model_dim_pad)

            def check_c_k_valid_gate(base_k):
                return arith.cmpi(CmpIPredicate.ult, base_k, inter_dim - inter_dim_pad)

            # A&B's scale preshuffle layout
            # For fp4, k_in is already packed (inter_dim // a_elem_vec_pack), so we need original inter_dim
            c_k_orig = arith.constant(inter_dim, index=True)
            layout_a_scale = make_preshuffle_scale_layout(
                arith, c_mn=m_in, c_k=c_k_orig
            )
            layout_b_scale = make_preshuffle_scale_layout(
                arith, c_mn=c_n_total, c_k=c_k_orig
            )

            shape_lds = fx.make_shape(tile_m, tile_k)
            stride_lds = fx.make_stride(lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            # Align with Aiter launch mapping:
            # - blockIdx.x -> N dimension (tile along model_dim)
            # - blockIdx.y -> expert-block id / M dimension (tile along sorted M)
            by = gpu.block_id("x")  # tile along model_dim
            bx_persist = gpu.block_id("y")  # persistent WG index

            # XOR16 swizzle parameter (in bytes; constant, power-of-two in our configs).
            k_blocks16 = arith.constant(tile_k_bytes // 16, index=True)
            layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

            base_ptr = allocator.get_base()
            lds_x_ptr = SmemPtr(
                base_ptr,
                lds_alloc_offset,
                x_lds_elem(),
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

            # lds_tid: alias LDS after max(x, out) for sorted_idx preload
            _lds_x_b = 2 * int(tile_m) * int(lds_stride) * int(a_elem_bytes)
            _lds_out_b = 2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
            _lds_tid_off = max(_lds_x_b, _lds_out_b)
            lds_tid = SmemPtr(
                base_ptr, lds_x_ptr.byte_offset + _lds_tid_off, T.i32, shape=(tile_m,)
            ).get()

            # Buffer resources.
            # For dynamic memrefs, `max_size=False` cannot infer the logical size from the memref *type*,
            # so we should pass `num_records_bytes` explicitly for stable hardware OOB behavior.
            c_topk = arith.constant(topk, index=True)

            # X(A2): buffer size in bytes, accounting for FP4 packing (2 elements per byte).
            # fp8/int8: 1 byte per element  -> bytes = tokens*topk * K
            # fp4:      2 elements per byte -> bytes = tokens*topk * K / 2
            c_a_pack = arith.constant(int(a_elem_vec_pack), index=True)
            c_elem_bytes = arith.constant(int(a_elem_bytes), index=True)
            x_nbytes_idx = _div_pow2(
                (tokens_in * c_topk) * k_in * c_elem_bytes, int(a_elem_vec_pack)
            )
            x_nbytes_i32 = arith.index_cast(T.i32, x_nbytes_idx)
            x_rsrc = buffer_ops.create_buffer_resource(
                arg_x, max_size=False, num_records_bytes=x_nbytes_i32
            )

            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)

            # OUT: [tokens, model_dim] -> clamp to descriptor max (i32 bytes) to avoid overflow on huge tokens.
            out_elem_bytes = 4 if out_is_f32 else 2
            out_nbytes_idx = (
                tokens_in * n_in * arith.constant(out_elem_bytes, index=True)
            )
            if not bool(accumulate):
                out_nbytes_idx = (
                    tokens_in
                    * arith.index(topk)
                    * n_in
                    * arith.constant(out_elem_bytes, index=True)
                )
            out_nbytes_i32 = arith.index_cast(T.i32, out_nbytes_idx)
            out_rsrc = buffer_ops.create_buffer_resource(
                arg_out, max_size=False, num_records_bytes=out_nbytes_i32
            )

            # num_valid_ids (sorted padded MN) for scale sizing / guards.
            numids_rsrc = buffer_ops.create_buffer_resource(
                arg_num_valid_ids,
                max_size=False,
                num_records_bytes=arith.constant(4, type=T.i32),
            )
            num_valid_i32 = buffer_ops.buffer_load(
                numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=T.i32
            )
            # num_valid_ids is a scalar (same value for all lanes) loaded into
            # VGPR.  Promote to SGPR so downstream buffer resource descriptors
            # that use it for num_records stay in SGPRs, eliminating the
            # expensive waterfall loop the compiler would otherwise emit.
            num_valid_i32 = rocdl.ReadfirstlaneOp(T.i32, num_valid_i32).res
            num_valid_idx = arith.index_cast(ir.IndexType.get(), num_valid_i32)

            # fp16 path ignores scales completely (implicit scale=1.0).
            if is_f16_a:
                sx_rsrc = None
            else:
                if is_f4_a:
                    # A2 microscale: e8m0 in sorted layout [sorted_size, K/32].
                    # Caller must pre-scatter a2_scale via moe_mxfp4_sort.
                    c32 = arith.constant(32, index=True)
                    kblk = _div_pow2(k_in, 32)
                    sx_nbytes_idx = num_valid_idx * kblk
                    sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
                    sx_rsrc = buffer_ops.create_buffer_resource(
                        arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_i32
                    )
                else:
                    # scale_x (A2 scale): [tokens*topk] f32 -> bytes = tokens*topk*4
                    sx_nbytes_idx = (tokens_in * c_topk) * arith.constant(4, index=True)
                    sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
                    sx_rsrc = buffer_ops.create_buffer_resource(
                        arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_i32
                    )

            if is_f16_b:
                sw_rsrc = None
            else:
                # Weight microscale buffer (packed i32 holding e8m0 bytes).
                # Use an exact descriptor size so hardware OOB checking works.
                c32 = arith.constant(32, index=True)
                kblk_w = _div_pow2(k_in, 32)  # K/32
                mn_w = arith.constant(experts * model_dim, index=True)
                sw_nbytes_idx = mn_w * kblk_w  # bytes (e8m0)
                sw_nbytes_i32 = arith.index_cast(T.i32, sw_nbytes_idx)
                sw_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_w, max_size=False, num_records_bytes=sw_nbytes_i32
                )

            # sorted_token_ids / sorted_weights: [blocks*tile_m] (padded length)
            sorted_nbytes_idx = (
                size_expert_ids_in
                * arith.constant(tile_m, index=True)
                * arith.constant(4, index=True)
            )
            sorted_nbytes_i32 = arith.index_cast(T.i32, sorted_nbytes_idx)
            sorted_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_token_ids,
                max_size=False,
                num_records_bytes=sorted_nbytes_i32,
            )
            sorted_w_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_weights, max_size=False, num_records_bytes=sorted_nbytes_i32
            )

            # expert ids: [sort_blocks] i32.
            _c_sbm = arith.constant(_sort_block_m, index=True)
            _c_tm = arith.constant(tile_m, index=True)
            _c1 = arith.constant(1, index=True)
            _sort_blocks_ub = _div_pow2(
                size_expert_ids_in * _c_tm + _c_sbm - _c1, _sort_block_m
            )
            eid_nbytes_idx = _sort_blocks_ub * arith.constant(4, index=True)
            eid_nbytes_i32 = arith.index_cast(T.i32, eid_nbytes_idx)
            expert_rsrc = buffer_ops.create_buffer_resource(
                arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes_i32
            )
            bias_rsrc = (
                buffer_ops.create_buffer_resource(arg_bias, max_size=False)
                if enable_bias
                else None
            )

            # ---- persist_m loop (with incremental expert B-base) ----
            _PERSIST_M = persist_m
            _c0_p = arith.constant(0, index=True)
            _c1_p = arith.constant(1, index=True)
            _c_pm = arith.constant(_PERSIST_M, index=True)
            # iter_args: [prev_expert_i32, prev_expert_b_base]
            _init_prev_expert = arith.constant(0, type=T.i32)
            _init_prev_b_base = arith.constant(0, index=True)
            _for_persist = scf.ForOp(
                _c0_p,
                _c_pm,
                _c1_p,
                [_init_prev_expert, _init_prev_b_base],
            )
            _for_ip = ir.InsertionPoint(_for_persist.body)
            _for_ip.__enter__()
            _mi_p = _for_persist.induction_variable
            _prev_expert_i32 = _for_persist.inner_iter_args[0]
            _prev_expert_b_base = _for_persist.inner_iter_args[1]
            bx = bx_persist * _c_pm + _mi_p
            bx_m = bx * arith.constant(tile_m, index=True)

            # Early-exit guard: skip garbage expert blocks beyond `num_valid_ids`.
            bx_m_i32 = arith.index_cast(T.i32, bx_m)
            blk_valid = arith.cmpi(CmpIPredicate.ult, bx_m_i32, num_valid_i32)

            sort_blk = _div_pow2(bx_m, _sort_block_m)
            expert_i32 = buffer_ops.buffer_load(
                expert_rsrc, sort_blk, vec_width=1, dtype=T.i32
            )
            expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
            exp_valid = arith.cmpi(
                CmpIPredicate.ult, expert_i32, arith.constant(experts, type=T.i32)
            )

            # Incremental expert B-base: delta = (cur - prev) * stride
            _delta_expert = arith.subi(expert_i32, _prev_expert_i32)
            _delta_expert_idx = arith.index_cast(ir.IndexType.get(), _delta_expert)
            _delta_b = _delta_expert_idx * arith.constant(_expert_b_stride, index=True)
            _expert_b_base = _prev_expert_b_base + _delta_b

            # Early-exit: if the first row of this tile is a sentinel (all-padding tile),
            # skip the entire GEMM.
            _first_tok = buffer_ops.buffer_load(
                sorted_rsrc, bx_m, vec_width=1, dtype=T.i32
            )
            _first_tid = arith.andi(_first_tok, arith.constant(0xFFFFFF, type=T.i32))
            _tokens_i32_guard = arith.index_cast(T.i32, tokens_in)
            tile_has_tokens = arith.cmpi(
                CmpIPredicate.ult, _first_tid, _tokens_i32_guard
            )

            # For tile_m < 32 (pack_M < _scale_pack_m): shift a_scale i32 so the
            # correct bytes land at the op_sel positions we use.
            if pack_M < _scale_pack_m:
                _m_off = _mod_pow2(_div_pow2(bx_m, 16), _scale_pack_m)
                _m_scale_shift_i32 = arith.index_cast(
                    T.i32, _m_off * arith.constant(8, index=True)
                )
            else:
                _m_scale_shift_i32 = None

            def _moe_gemm2_then_body():
                # Expert id for this M tile.
                n_idx = arith.constant(model_dim, index=True)
                expert_off_idx = expert_idx * n_idx  # index

                # ---- X gmem->reg prefetch (match preshuffle GEMM mapping) ----
                # Prefer 16B buffer-load (dwordx4). If the per-thread byte count isn't divisible by
                # 16, fall back to 8B (dwordx2) or 4B (dword) loads. For fp16 we require 16B.
                if is_f16_a:
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
                vec4_i32 = T.vec(4, i32)

                c_k_div4 = _div_pow2(
                    _div_pow2(k_in, int(a_elem_vec_pack))
                    * arith.constant(int(a_elem_bytes), index=True),
                    4,
                )
                c_k_div4_i32 = arith.index_cast(T.i32, c_k_div4)
                layout_x_div4 = fx.make_layout(
                    (m_i32_v, c_k_div4_i32), stride=(c_k_div4_i32, 1)
                )
                tile_k_dwords = (int(tile_k) * int(a_elem_bytes)) // (
                    4 * int(a_elem_vec_pack)
                )
                layout_x_tile_div4 = fx.make_layout(
                    (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                )
                c_chunk_i32 = arith.constant(chunk_i32, index=True)
                tx_i32_base = tx * c_chunk_i32

                topk_i32 = arith.constant(topk)
                mask24 = arith.constant(0xFFFFFF)
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

                vec1_i32 = T.vec(1, i32)
                vec2_i32 = T.vec(2, i32)
                vec4_x = T.vec(4, x_elem)
                x_load_vec_elems = (
                    x_load_bytes if a_elem_bytes == 1 else x_load_bytes // a_elem_bytes
                )

                def load_x(idx_i32):
                    """Load `x_load_bytes` bytes from X (gmem) into regs.

                    For 16B, keep the fast dwordx4 path. For 8B/4B, use byte offsets.
                    """
                    if x_load_bytes == 16:
                        idx_elem = (
                            idx_i32 if a_elem_bytes == 1 else (idx_i32 * arith.index(2))
                        )
                        return buffer_copy_gmem16_dwordx4(
                            buffer_ops,
                            vector,
                            elem_type=x_elem,
                            idx_i32=idx_elem,
                            rsrc=x_rsrc,
                            vec_elems=vec16_elems,
                        )
                    # 8B/4B: convert dword index to byte offset and use offset_in_bytes path.
                    idx_bytes = idx_i32 * arith.index(4)
                    return _buffer_load_vec(
                        buffer_ops,
                        vector,
                        x_rsrc,
                        idx_bytes,
                        elem_type=x_elem,
                        vec_elems=x_load_vec_elems,
                        elem_bytes=a_elem_bytes,
                        offset_in_bytes=True,
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
                    t_i32 = arith.andi(fused_i, mask24)
                    s_i32 = arith.shrui(fused_i, arith.constant(24))

                    t_valid = arith.cmpi(CmpIPredicate.ult, t_i32, tokens_i32)
                    s_valid = arith.cmpi(CmpIPredicate.ult, s_i32, topk_i32)
                    ts_valid = arith.andi(t_valid, s_valid)
                    t_safe = arith.select(ts_valid, t_i32, arith.constant(0))
                    s_safe = arith.select(ts_valid, s_i32, arith.constant(0))
                    row_ts_i32 = t_safe * topk_i32 + s_safe
                    row_ts_idx = arith.index_cast(ir.IndexType.get(), row_ts_i32)

                    x_row_base_div4.append(row_ts_idx * c_k_div4)

                def load_x_tile(base_k):
                    base_k_div4 = _div_pow2(
                        _div_pow2(base_k, int(a_elem_vec_pack))
                        * arith.constant(int(a_elem_bytes), index=True),
                        4,
                    )
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
                coord_wl = idx2crd(tx, layout_tx_wave_lane)
                wave_id = layout_get(coord_wl, 0)
                lane_id = layout_get(coord_wl, 1)
                coord_l16 = idx2crd(lane_id, layout_lane16)
                lane_div_16 = layout_get(coord_l16, 0)
                lane_mod_16 = layout_get(coord_l16, 1)

                row_a_lds = lane_mod_16

                col_offset_base = lane_div_16 * arith.constant(16, index=True)

                # Dynamic N tiling within block.
                by_n = by * arith.constant(tile_n, index=True)
                num_waves = 4
                n_per_wave = tile_n // num_waves
                num_acc_n = n_per_wave // 16
                c_n_per_wave = arith.constant(n_per_wave, index=True)
                wave_mod_4 = _mod_pow2(wave_id, 4)
                n_tile_base = wave_mod_4 * c_n_per_wave

                if pack_N < _scale_pack_n:
                    _global_n_base = expert_off_idx + by_n + n_tile_base
                    _n_off = _mod_pow2(_div_pow2(_global_n_base, 16), _scale_pack_n)
                    _n_scale_shift_i32 = arith.index_cast(
                        T.i32, _n_off * arith.constant(8, index=True)
                    )
                else:
                    _n_scale_shift_i32 = None

                # Precompute (local n_blk, n_intra) for B -- expert contribution
                # is factored out into _expert_b_base (incremental, carried
                # across persist_m iterations).  model_dim % 16 == 0 guarantees
                # the expert offset does not affect n_intra.
                n_intra_list = []
                n_blk_list = []
                col_g_list = []
                c_n0 = _div_pow2(c_n_total, 16)
                c_n0_static = experts * model_dim // 16

                for i in range_constexpr(num_acc_n):
                    offset = i * 16

                    col_g = by_n + n_tile_base
                    col_g = _div_pow2(col_g, 2) + offset
                    col_g = col_g + lane_mod_16
                    col_g_list.append(col_g)

                    c_offset = arith.constant(offset, index=True)
                    global_n = by_n + n_tile_base + c_offset + lane_mod_16
                    # Only the LOCAL part (no expert offset): expert is in
                    # _expert_b_base which is added in load_b_packs_k64.
                    n_blk_list.append(_div_pow2(global_n, 16))
                    n_intra_list.append(_mod_pow2(global_n, 16))

                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 128  # K64-byte micro-step (2x MFMA)

                # fp4 pack
                k_unroll_packed = k_unroll // pack_K
                m_repeat_packed = m_repeat // pack_M
                num_acc_n_packed = num_acc_n // pack_N

                # --- B Load Logic (K64) - shared layout with preshuffle GEMM ---
                def load_b_packs_k64(base_k, ku: int, ni: int):
                    """Load one K64-byte B micro-step: single 16B load, split into 2x i64."""
                    base_k_bytes = base_k * arith.constant(
                        int(b_elem_bytes), index=True
                    )
                    k0_base = _div_pow2(base_k_bytes, 64)
                    k0 = k0_base + arith.constant(ku, index=True)
                    k1 = lane_div_16
                    # Incremental B addressing: _expert_b_base carries the
                    # expert's preshuffle offset (updated via delta each
                    # persist_m iteration); local n_blk/n_intra contribute
                    # the per-lane within-tile offset.  All strides are
                    # compile-time constants -> shift/mul, no Barrett.
                    idx_pack = (
                        _expert_b_base
                        + n_blk_list[ni] * arith.constant(_b_stride_n0, index=True)
                        + k0 * arith.constant(_b_stride_k0, index=True)
                        + k1 * arith.constant(_b_stride_klane, index=True)
                        + n_intra_list[ni] * arith.constant(_b_stride_nlane, index=True)
                    )

                    vec_elems = kpack_bytes // int(b_elem_bytes)
                    b16 = _buffer_load_vec(
                        buffer_ops,
                        vector,
                        w_rsrc,
                        idx_pack,
                        elem_type=_w_elem_type(),
                        vec_elems=vec_elems,
                        elem_bytes=b_elem_bytes,
                        offset_in_bytes=(b_elem_bytes == 1),
                    )
                    b_i64x2 = vector.bitcast(vec2_i64, b16)
                    b0 = vector.extract(
                        b_i64x2, static_position=[0], dynamic_position=[]
                    )
                    b1 = vector.extract(
                        b_i64x2, static_position=[1], dynamic_position=[]
                    )
                    return b0, b1

                def load_b_tile(base_k):
                    b_tile = []
                    for ku in range_constexpr(k_unroll):
                        packs0 = []
                        packs1 = []
                        for ni in range_constexpr(num_acc_n):
                            b0, b1 = load_b_packs_k64(base_k, ku, ni)
                            packs0.append(b0)
                            packs1.append(b1)
                        b_tile.append((packs0, packs1))
                    return b_tile

                def load_scale(arg_scale, rsrc, scale_info, ku, mni):
                    k_lane = lane_div_16
                    n_lane = lane_mod_16
                    # Direct arith crd2idx: idx = mni*stride_n0 + ku*stride_k0 + k_lane*stride_klane + n_lane
                    idx_pack = (
                        mni * scale_info.stride_n0
                        + ku * scale_info.stride_k0
                        + k_lane * scale_info.stride_klane
                        + n_lane
                    )
                    s = buffer_ops.buffer_load(rsrc, idx_pack, vec_width=1, dtype=T.i32)
                    return vector.from_elements(T.vec(1, T.i32), [s])

                def _apply_k_shift(scale_vec, k_shift_bits):
                    if k_shift_bits > 0:
                        val = vector.extract(
                            scale_vec, static_position=[0], dynamic_position=[]
                        )
                        val = arith.shrui(val, arith.constant(k_shift_bits, type=T.i32))
                        return vector.from_elements(T.vec(1, T.i32), [val])
                    return scale_vec

                def load_b_scale_tile(base_k, k_shift_bits=0):
                    b_scale_tile = []
                    for ku in range_constexpr(k_unroll_packed):
                        for ni in range_constexpr(num_acc_n_packed):
                            scale = load_scale(
                                arg_scale_w,
                                sw_rsrc,
                                layout_b_scale,
                                ku + base_k,
                                ni
                                + _div_pow2(
                                    _div_pow2(
                                        expert_off_idx + by_n + n_tile_base,
                                        _scale_pack_n,
                                    ),
                                    16,
                                ),
                            )
                            scale = _apply_k_shift(scale, k_shift_bits)
                            b_scale_tile.append(scale)
                    return b_scale_tile

                def load_a_scale_tile(base_k, k_shift_bits=0):
                    a_scale_tile = []
                    for ku in range_constexpr(k_unroll_packed):
                        for mi in range_constexpr(m_repeat_packed):
                            scale = load_scale(
                                arg_scale_x,
                                sx_rsrc,
                                layout_a_scale,
                                ku + base_k,
                                mi + _div_pow2(_div_pow2(bx_m, _scale_pack_m), 16),
                            )
                            scale = _apply_k_shift(scale, k_shift_bits)
                            a_scale_tile.append(scale)
                    return a_scale_tile

                def prefetch_ab_scale_tile(base_k, k_shift_bits=0):
                    return [
                        load_a_scale_tile(base_k, k_shift_bits),
                        load_b_scale_tile(base_k, k_shift_bits),
                    ]

                vec8_x = T.vec(vec8_elems, x_elem)
                vec4_x_lds = T.vec(vec4_elems, x_elem)

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
                                tx_c4=arith.index(4),
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
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x2=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )
                        else:  # x_load_bytes == 4
                            lds_store_4b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec4_ty=vec4_x_lds,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x1=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )

                # --- A LDS load helper for K64 (load 16B once, extract 2x i64 halves) ---
                def lds_load_packs_k64(curr_row_a_lds, col_base, lds_base):
                    # Swizzle in bytes, then convert to element offset for memref indexing.
                    col_base_swz_bytes = swizzle_xor16(
                        curr_row_a_lds, col_base, k_blocks16
                    )
                    col_base_swz = (
                        col_base_swz_bytes
                        if elem_bytes == 1
                        else (col_base_swz_bytes / arith.index(2))
                    )
                    # Pass as list so layout_utils.crd2idx uses static arith path
                    idx_a16 = crd2idx([curr_row_a_lds, col_base_swz], layout_lds)
                    idx_a16 = idx_a16 + lds_base
                    loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                    a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                    a0 = vector.extract(
                        a_i64x2, static_position=[0], dynamic_position=[]
                    )
                    a1 = vector.extract(
                        a_i64x2, static_position=[1], dynamic_position=[]
                    )
                    return a0, a1

                def compute_tile(
                    acc_in,
                    b_tile_in,
                    lds_base,
                    a_scale=None,
                    b_scale=None,
                    *,
                    prefetch_epilogue: bool = False,
                    a0_prefetch=None,
                    a1_prefetch=None,
                ):
                    acc_list = list(acc_in)
                    mfma_res_ty = vec4_i32 if is_int8 else vec4_f32

                    epilogue_pf = None
                    bias = None
                    if prefetch_epilogue:
                        if enable_bias:
                            bias = []
                            for ni in range_constexpr(num_acc_n):
                                global_n = by_n + n_tile_base + ni * 16 + lane_mod_16
                                bias_offset = expert_off_idx + global_n
                                bias.append(
                                    buffer_ops.buffer_load(
                                        bias_rsrc, bias_offset, vec_width=1, dtype=f32
                                    )
                                )
                        tw_pf = None
                        if doweight_stage2:
                            tw_pf = []
                            lane_div_16_mul4_pf = lane_div_16 * arith.index(4)
                            ii_idx_list_pf = [
                                arith.constant(ii, index=True) for ii in range(4)
                            ]
                            for mi in range_constexpr(m_repeat):
                                mi_base_pf = arith.constant(mi * 16, index=True)
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
                                            dtype=f32,
                                        )
                                    )
                        epilogue_pf = (None, tw_pf, bias)

                    c0_i64 = arith.constant(0, type=T.i64)
                    vec4_i64 = T.vec(4, T.i64)
                    vec8_i32 = T.vec(8, T.i32)

                    def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                        v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                        return vector.bitcast(vec8_i32, v4)

                    # fp4 path
                    for ku128 in range_constexpr(k_unroll_packed):
                        for mi in range_constexpr(m_repeat_packed):
                            a_scale_i32 = a_scale[ku128 * m_repeat_packed + mi]
                            a_scale_val = vector.extract(
                                a_scale_i32, static_position=[0], dynamic_position=[]
                            )
                            if _m_scale_shift_i32 is not None:
                                a_scale_val = arith.shrui(
                                    a_scale_val, _m_scale_shift_i32
                                )
                            for ni in range_constexpr(num_acc_n_packed):
                                b_scale_i32 = b_scale[ku128 * num_acc_n_packed + ni]
                                b_scale_val = vector.extract(
                                    b_scale_i32,
                                    static_position=[0],
                                    dynamic_position=[],
                                )
                                if _n_scale_shift_i32 is not None:
                                    b_scale_val = arith.shrui(
                                        b_scale_val, _n_scale_shift_i32
                                    )
                                for ikxdl in range_constexpr(pack_K):
                                    k_idx = ku128 * pack_K + ikxdl

                                    b_packs0, b_packs1 = b_tile_in[k_idx]

                                    col_base = (
                                        col_offset_base
                                        + (k_idx * 128) // a_elem_vec_pack
                                    )

                                    for imxdl in range_constexpr(pack_M):
                                        col_base0 = col_base
                                        mi_idx = mi * pack_M + imxdl
                                        mi_val = arith.constant(mi_idx * 16, index=True)
                                        curr_row_a_lds = row_a_lds + mi_val

                                        if (
                                            (a0_prefetch is not None)
                                            and (k_idx == 0)
                                            and (mi_idx == 0)
                                        ):
                                            a0, a1 = a0_prefetch
                                        elif (
                                            (a1_prefetch is not None)
                                            and (k_idx == 1)
                                            and (mi_idx == 0)
                                        ):
                                            a0, a1 = a1_prefetch
                                        else:
                                            a0, a1 = lds_load_packs_k64(
                                                curr_row_a_lds, col_base0, lds_base
                                            )

                                        if is_f8_a:
                                            col_base1 = col_base + 64
                                            a2, a3 = lds_load_packs_k64(
                                                curr_row_a_lds, col_base1, lds_base
                                            )
                                            a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)
                                        else:
                                            a128 = pack_i64x4_to_i32x8(
                                                a0, a1, c0_i64, c0_i64
                                            )

                                        for inxdl in range_constexpr(pack_N):
                                            ni_idx = ni * pack_N + inxdl

                                            b0 = b_packs0[ni_idx]
                                            b1 = b_packs1[ni_idx]
                                            b128 = pack_i64x4_to_i32x8(
                                                b0, b1, c0_i64, c0_i64
                                            )

                                            acc_idx = mi_idx * num_acc_n + ni_idx
                                            rocdl.sched_barrier(0)
                                            acc_list[acc_idx] = (
                                                rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                                    mfma_res_ty,
                                                    [
                                                        a128,
                                                        b128,
                                                        acc_list[acc_idx],
                                                        cbsz,
                                                        blgp,
                                                        ikxdl * _scale_pack_m + imxdl,
                                                        a_scale_val,
                                                        ikxdl * _scale_pack_n + inxdl,
                                                        b_scale_val,
                                                    ],
                                                )
                                            )

                    return acc_list, epilogue_pf

                # ---------------- 2-stage pipeline (ping-pong LDS + B tile prefetch) ----------------
                lds_tile_elems = arith.constant(tile_m * lds_stride, index=True)
                lds_base_cur = arith.index(0)
                lds_base_nxt = lds_tile_elems

                rocdl.sched_barrier(0)

                def hot_loop_scheduler():
                    # - MFMA group size per "slot": num_acc_n
                    # - Total MFMA per tile: (2*K32 per K64) * k_unroll * m_repeat * num_acc_n
                    # - We emit (mfma_group + dsrd + mfma_group) per scheduler iteration.
                    mfma_group = num_acc_n
                    mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                    mfma_per_iter = 2 * mfma_group
                    sche_iters = (
                        0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                    )

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

                def _k_shift_bits(k_py):
                    if pack_K >= _scale_pack_k:
                        return 0
                    return ((k_py // 128) % _scale_pack_k) * _scale_pack_m * 8

                def _k_base(k_py):
                    return k_py // _scale_pack_k // 128

                # Prologue -- B-first.
                k0 = arith.index(0)
                b_cur = load_b_tile(k0)
                a_scale_pong, b_scale_pong = prefetch_ab_scale_tile(
                    _k_base(0), _k_shift_bits(0)
                )
                # scheduling fence to prevent LLVM from deferring
                # the scale buffer_loads past the upcoming barrier.
                rocdl.sched_barrier(0)
                x_regs0 = load_x_tile(k0)
                store_x_tile_to_lds(x_regs0, lds_base_cur)
                # Preload sorted_idx into lds_tid for epilogue precompute_row
                _c_tile_m_idx = arith.constant(tile_m, index=True)
                _tid_in_range = arith.cmpi(CmpIPredicate.ult, tx, _c_tile_m_idx)
                _if_tid = scf.IfOp(_tid_in_range)
                with ir.InsertionPoint(_if_tid.then_block):
                    _tid_row = bx_m + tx
                    _tid_val = buffer_ops.buffer_load(
                        sorted_rsrc, _tid_row, vec_width=1, dtype=T.i32
                    )
                    _tid_vec1 = vector.from_elements(T.vec(1, T.i32), [_tid_val])
                    vector.store(_tid_vec1, lds_tid, [tx])
                    scf.YieldOp([])
                gpu.barrier()

                acc = [acc_init] * num_acc_n * m_repeat
                lds_base_pong = lds_base_cur
                lds_base_ping = lds_base_nxt

                # Cross-tile A0+A1 LDS prefetch: issue both ds_reads back-to-back
                # (??2) so LDS bandwidth is fully utilized and the second read
                # completes during MFMA #1/#2 execution.
                a0_prefetch_pong = lds_load_packs_k64(
                    row_a_lds, col_offset_base, lds_base_pong
                )
                _a1_col_base = col_offset_base + 128 // a_elem_vec_pack
                a1_prefetch_pong = (
                    lds_load_packs_k64(row_a_lds, _a1_col_base, lds_base_pong)
                    if pack_K >= 2
                    else None
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

                c2_tile_k = arith.constant(tile_k * 2, index=True)
                b_pong = b_cur
                # Only emit the scf.for when there are actually iterations to run.
                # When k_main2_py == 0 the loop body is empty; emitting an scf.for
                # would create a region whose internal SSA values cannot be used
                # by the post-loop tail code.
                if k_main2_py > 0:
                    for k_iv_py in range_constexpr(0, k_main2_py, tile_k * 2):
                        k_iv = k_iv_py
                        next_k1 = k_iv + tile_k
                        x_regs_ping = load_x_tile(next_k1)
                        b_ping = load_b_tile(next_k1 // 2)
                        a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(
                            _k_base(next_k1), _k_shift_bits(next_k1)
                        )

                        acc, _ = compute_tile(
                            acc,
                            b_pong,
                            lds_base_pong,
                            a_scale_pong,
                            b_scale_pong,
                            a0_prefetch=a0_prefetch_pong,
                            a1_prefetch=a1_prefetch_pong,
                        )
                        store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                        # hot_loop_scheduler()
                        gpu.barrier()

                        # Cross-tile prefetch for the ping tile we are about to compute.
                        a0_prefetch_ping = lds_load_packs_k64(
                            row_a_lds, col_offset_base, lds_base_ping
                        )
                        a1_prefetch_ping = (
                            lds_load_packs_k64(row_a_lds, _a1_col_base, lds_base_ping)
                            if pack_K >= 2
                            else None
                        )

                        next_k2 = k_iv + c2_tile_k
                        next_k2_py = k_iv_py + tile_k * 2
                        x_regs_pong = load_x_tile(next_k2)
                        b_pong = load_b_tile(next_k2 // 2)
                        a_scale_pong, b_scale_pong = prefetch_ab_scale_tile(
                            _k_base(next_k2_py), _k_shift_bits(next_k2_py)
                        )

                        acc, _ = compute_tile(
                            acc,
                            b_ping,
                            lds_base_ping,
                            a_scale_ping,
                            b_scale_ping,
                            a0_prefetch=a0_prefetch_ping,
                            a1_prefetch=a1_prefetch_ping,
                        )
                        store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                        # hot_loop_scheduler()
                        gpu.barrier()

                        # Cross-tile prefetch for the next pong tile.
                        a0_prefetch_pong = lds_load_packs_k64(
                            row_a_lds, col_offset_base, lds_base_pong
                        )
                        a1_prefetch_pong = (
                            lds_load_packs_k64(row_a_lds, _a1_col_base, lds_base_pong)
                            if pack_K >= 2
                            else None
                        )

                if odd_k_tiles:
                    # Tail: single remaining tile (already in `b_cur` / `lds_base_pong`).
                    acc, epilogue_pf = compute_tile(
                        acc,
                        b_pong,
                        lds_base_pong,
                        a_scale_pong,
                        b_scale_pong,
                        a0_prefetch=a0_prefetch_pong,
                        a1_prefetch=a1_prefetch_pong,
                        prefetch_epilogue=True,
                    )

                else:
                    # Tail: 2 remaining tiles.
                    k_tail1 = (k_in + tile_k - 1) // tile_k * tile_k - tile_k
                    k_tail1_py = (
                        int(inter_dim) + tile_k - 1
                    ) // tile_k * tile_k - tile_k
                    x_regs_ping = load_x_tile(k_tail1)
                    b_ping = load_b_tile(k_tail1 // 2)
                    a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(
                        _k_base(k_tail1_py), _k_shift_bits(k_tail1_py)
                    )

                    acc, _ = compute_tile(
                        acc,
                        b_pong,
                        lds_base_pong,
                        a_scale_pong,
                        b_scale_pong,
                        a0_prefetch=a0_prefetch_pong,
                        a1_prefetch=a1_prefetch_pong,
                    )

                    store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    # hot_loop_scheduler()
                    gpu.barrier()

                    # Epilogue tile with sw prefetch.
                    a0_prefetch_ping = lds_load_packs_k64(
                        row_a_lds, col_offset_base, lds_base_ping
                    )
                    a1_prefetch_ping = (
                        lds_load_packs_k64(row_a_lds, _a1_col_base, lds_base_ping)
                        if pack_K >= 2
                        else None
                    )
                    acc, epilogue_pf = compute_tile(
                        acc,
                        b_ping,
                        lds_base_ping,
                        a_scale_ping,
                        b_scale_ping,
                        a0_prefetch=a0_prefetch_ping,
                        a1_prefetch=a1_prefetch_ping,
                        prefetch_epilogue=True,
                    )

                # ---------------- Epilogue: LDS CShuffle + atomic half2 (x2) ----------------
                # Reuse the shared helper so GEMM / MoE kernels share the exact same CShuffle skeleton.

                sw_pf = None
                tw_pf = None
                bias_pf = None
                if epilogue_pf is not None:
                    sw_pf, tw_pf, bias_pf = epilogue_pf

                mask24_i32 = arith.constant(0xFFFFFF)
                topk_i32_v = topk_i32

                zero_i32 = arith.constant(0)

                def atomic_add_f16x2(val_f16x2, byte_off_i32):
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        val_f16x2,
                        out_rsrc,
                        byte_off_i32,
                        zero_i32,
                        zero_i32,
                    )

                # Weight scales for the N tile (col_g depends on lane/wave/by but not on (t,s)).
                if lds_out is None:
                    raise RuntimeError(
                        "FLIR_MOE_STAGE2_CSHUFFLE=1 but lds_out is not allocated/aliased."
                    )

                # Precompute the output base address (i64 index) for ALL paths.
                # Both accumulate=True (global atomic) and accumulate=False (global store)
                # need 64-bit addressing to avoid i32 offset overflow when
                # tokens * model_dim * elem_bytes > INT32_MAX (~150K tokens for model_dim=7168).
                from flydsl._mlir.dialects import fly as _fly

                _llvm_ptr_ty = ir.Type.parse("!llvm.ptr")
                out_base_ptr = _fly.extract_aligned_pointer_as_index(
                    _llvm_ptr_ty, arg_out
                )
                out_base_i64 = llvm.ptrtoint(T.i64, out_base_ptr)
                out_base_idx = arith.index_cast(ir.IndexType.get(), out_base_i64)

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
                    # Match origin/dev_a16w4: rely on sentinel padded rows + hardware OOB behavior.
                    fused2 = buffer_ops.buffer_load(
                        sorted_rsrc, row, vec_width=1, dtype=T.i32
                    )
                    t2 = fused2 & mask24_i32
                    s2 = fused2 >> 24

                    t_ok = arith.cmpi(CmpIPredicate.ult, t2, tokens_i32)
                    s_ok = arith.cmpi(CmpIPredicate.ult, s2, topk_i32_v)
                    ts_ok = arith.andi(t_ok, s_ok)
                    t2_safe = arith.select(ts_ok, t2, arith.constant(0))
                    s2_safe = arith.select(ts_ok, s2, arith.constant(0))
                    t2_safe * topk_i32_v + s2_safe

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
                        acc_idx = mi * num_acc_n + ni
                        v = vector.extract(
                            acc[acc_idx], static_position=[ii], dynamic_position=[]
                        )
                        if is_int8:
                            v = arith.sitofp(f32, v)
                        if enable_bias:
                            v = v + bias_pf[ni]

                        if doweight_stage2:
                            v = v * tw
                        v_out = arith.trunc_f(out_elem(), v)

                        lds_idx = row_base_lds + col_local
                        vec1_out = T.vec(1, out_elem())
                        v1 = vector.from_elements(vec1_out, [v_out])

                        vector.store(v1, lds_out, [lds_idx], alignment=2)

                def precompute_row(*, row_local, row):
                    # Use lds_tid (sorted_idx preloaded to LDS) instead of buffer_load
                    # to avoid extra VMEM round-trips in the epilogue.
                    fused2 = memref.load(lds_tid, [row_local])
                    row_i32 = arith.index_cast(T.i32, row)
                    row_valid0 = arith.cmpi(CmpIPredicate.ult, row_i32, num_valid_i32)
                    t = fused2 & mask24_i32
                    s = fused2 >> 24
                    t_ok = arith.cmpi(CmpIPredicate.ult, t, tokens_i32)
                    s_ok = arith.cmpi(CmpIPredicate.ult, s, topk_i32_v)
                    row_valid = arith.andi(row_valid0, arith.andi(t_ok, s_ok))
                    t_idx = arith.index_cast(ir.IndexType.get(), t)
                    s_idx = arith.index_cast(ir.IndexType.get(), s)
                    ts_idx = t_idx * arith.constant(topk, index=True) + s_idx
                    if accumulate:
                        row_byte_base = out_base_idx + t_idx * arith.constant(
                            model_dim * out_elem_bytes, index=True
                        )
                    else:
                        row_byte_base = out_base_idx + ts_idx * arith.constant(
                            model_dim * out_elem_bytes, index=True
                        )
                    return ((fused2, row_byte_base), row_valid)

                def _idx_to_llvm_ptr(idx_val, addr_space=1):
                    """Convert an index-typed byte address to !llvm.ptr<addr_space>."""
                    idx_v = idx_val._value if hasattr(idx_val, "_value") else idx_val
                    i64_v = arith.index_cast(T.i64, idx_v)
                    i64_raw = i64_v._value if hasattr(i64_v, "_value") else i64_v
                    ptr_ty = ir.Type.parse(f"!llvm.ptr<{addr_space}>")
                    return llvm.inttoptr(ptr_ty, i64_raw)

                def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                    fused, row_byte_base = row_ctx
                    if not bool(accumulate):
                        # ---- 64-bit global store path (avoids i32 offset overflow) ----
                        col_idx = col_g0
                        byte_off_col = col_idx * arith.constant(
                            out_elem_bytes, index=True
                        )
                        ptr_addr_idx = row_byte_base + byte_off_col
                        out_ptr_v = _idx_to_llvm_ptr(ptr_addr_idx)
                        frag_v = frag._value if hasattr(frag, "_value") else frag
                        llvm.StoreOp(
                            frag_v,
                            out_ptr_v,
                            alignment=_e_vec * out_elem_bytes,
                            nontemporal=True,
                        )
                    else:
                        # ---- accumulate=True: 64-bit global atomic path ----
                        col_idx = col_g0
                        byte_off_col = col_idx * arith.constant(
                            out_elem_bytes, index=True
                        )
                        ptr_addr_idx = row_byte_base + byte_off_col
                        out_ptr_v = _idx_to_llvm_ptr(ptr_addr_idx)
                        frag_v = frag._value if hasattr(frag, "_value") else frag
                        llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.fadd,
                            out_ptr_v,
                            frag_v,
                            llvm.AtomicOrdering.monotonic,
                            syncscope="agent",
                            alignment=_e_vec * out_elem_bytes,
                        )

                _e_vec = 2 if accumulate else min(tile_n // 32, 8)
                c_shuffle_epilog(
                    arith=arith,
                    vector=vector,
                    gpu=gpu,
                    scf=scf,
                    range_constexpr=range_constexpr,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    e_vec=_e_vec,
                    m_repeat=m_repeat,
                    num_acc_n=num_acc_n,
                    tx=tx,
                    lane_div_16=lane_div_16,
                    lane_mod_16=lane_mod_16,
                    bx_m=bx_m,
                    by_n=by_n,
                    n_tile_base=n_tile_base,
                    lds_out=lds_out,
                    frag_elem_type=(
                        ir.BF16Type.get() if out_is_bf16 else ir.F16Type.get()
                    ),
                    write_row_to_lds=write_row_to_lds,
                    precompute_row=precompute_row,
                    store_pair=store_pair,
                )

            _all_valid = arith.andi(blk_valid, arith.andi(exp_valid, tile_has_tokens))
            _if_valid = scf.IfOp(_all_valid)
            with ir.InsertionPoint(_if_valid.then_block):
                _moe_gemm2_then_body()
                scf.YieldOp([])

            gpu.barrier()
            # Yield updated expert state for next persist_m iteration.
            scf.YieldOp([expert_i32, _expert_b_base])
            _for_ip.__exit__(None, None, None)

    # -- Host launcher (flyc.jit + .launch) --------------------------------
    _cache_tag = (
        module_name,
        a_dtype,
        b_dtype,
        out_dtype,
        tile_m,
        tile_n,
        tile_k,
        doweight_stage2,
        accumulate,
        enable_bias,
        model_dim_pad,
        inter_dim_pad,
        use_cshuffle_epilog,
        persist_m,
        _sort_block_m,
    )

    @flyc.jit
    def launch_mixed_moe_gemm2(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        _ = _cache_tag
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        n_in = arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
        gx = n_in / arith.constant(tile_n, index=True)
        _c_pm_l = arith.constant(persist_m, index=True)
        gy = (
            arith.index_cast(ir.IndexType.get(), i32_size_expert_ids_in.ir_value())
            + _c_pm_l
            - arith.constant(1, index=True)
        ) / _c_pm_l

        moe_gemm2(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_num_valid_ids,
            arg_bias,
            i32_tokens_in,
            i32_n_in,
            i32_k_in,
            i32_size_expert_ids_in,
        ).launch(
            grid=(gx, gy, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_mixed_moe_gemm2