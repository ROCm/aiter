"""MoE GEMM stage1/stage2 kernel implementations (FLIR MFMA FP8/FP16).

This module intentionally contains the **kernel builder code** for:
- `moe_gemm1` (stage1)
- `moe_gemm2` (stage2)

It is extracted from `tests/kernels/test_moe_gemm.py` so that:
- `kernels/` holds the implementation
- `tests/` holds correctness/perf harnesses
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch


from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from flydsl._mlir import ir
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from flydsl.expr import arith, gpu, buffer_ops, vector, rocdl, const_expr
from flydsl._mlir.dialects import llvm, scf, memref

from .mfma_preshuffle_pipeline import (
    _buffer_load_vec,
    buffer_copy_gmem16_dwordx4,
    lds_store_16b_xor16,
    load_b_tile_range_k64,
    load_scale_tile,
    make_preshuffle_b_layout,
    make_preshuffle_scale_layout,
    tile_chunk_coord_i32,
    swizzle_xor16,
    xcd_remap_bx_by_grid,
)
from .mfma_epilogues import (
    c_shuffle_epilog,
    f32_to_e2m1,
    idx_to_llvm_ptr,
    moe_act_vec4,
    raw_value,
    run_moe_stage1_cshuffle_epilog,
    store_or_atomic_add,
)
from .layout_utils import crd2idx, idx2crd, get as layout_get

import functools

from aiter.ops.flydsl.moe_common import (
    GateMode,
)  # noqa: F401  re-exported for back-compat


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
    use_async_copy: bool = False,
    waves_per_eu: int = 4,
    k_batch: int = 1,
    b_nt: int = 0,
    gate_mode: GateMode = GateMode.SEPARATED,
    a_scale_one: bool = False,
    xcd_swizzle: int = 0,
    swiglu_limit: float = 0.0,
):
    """Compile stage1 kernel (gate+up with silu/swiglu).

    GEMM: act(X @ W_gate.T, X @ W_up.T) -> [tokens*topk, inter_dim]
    Direct store (no atomic).  When k_batch>1 (split-K), each CTA
    computes a K-slice and atomically adds gate/up partials.
    Note: persist_m=1 (no persistence) is optimal for stage1 because K=model_dim
    is large, so each CTA is already compute-heavy. persist_m>1 serializes M blocks
    that the GPU can process in parallel.

    gate_mode controls the gate/up computation strategy — see GateMode enum.
    """
    gpu_arch = get_hip_arch()
    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")
    _state = {}

    # Specialized to a4w4 (a=fp4) and a8w4 (a=fp8); weights are always fp4.
    if a_dtype not in ("fp4", "fp8"):
        raise ValueError(
            f"mixed MoE supports a_dtype in ('fp4','fp8'), got {a_dtype!r}"
        )
    if b_dtype != "fp4":
        raise ValueError(f"mixed MoE supports b_dtype='fp4' only, got {b_dtype!r}")

    is_f8_a = a_dtype == "fp8"
    is_f4_a = a_dtype == "fp4"
    is_f4_b = True

    sort_block_m = max(32, tile_m)
    num_waves = min(4, tile_n // 32)
    total_threads = num_waves * 64
    pack_M = 1 if tile_m < 32 else 2
    n_per_wave = tile_n // num_waves
    pack_N = min(2, n_per_wave // 16)
    pack_K = 2
    scale_mn_pack = 2
    elem_bytes = 1
    a_elem_bytes = 1
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

    def _w_elem_type():
        return T.i8

    def out_elem():
        return T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)

    def _load_bias_scalar(bias_rsrc, offset):
        return buffer_ops.buffer_load(bias_rsrc, offset, vec_width=1, dtype=T.f32)

    mock_gate_only = gate_mode is GateMode.MOCK_GATE_ONLY
    gate_up_interleave = gate_mode is GateMode.INTERLEAVE
    gate_only = gate_mode is GateMode.GATE_ONLY

    # Padding semantics: model_dim and inter_dim INCLUDE padding.
    #   model_dim = model_dim_true + model_dim_pad   (K direction)
    #   inter_dim = inter_dim_true + inter_dim_pad   (N direction)
    # Tensor sizes use the padded dimensions (inter_dim, model_dim).
    # Padding only affects kernel internal logic and grid computation.
    _inter_dim_valid = inter_dim - inter_dim_pad

    # Split-K validation
    _is_splitk = k_batch > 1
    if mock_gate_only and not _is_splitk:
        raise ValueError("mock_gate_only requires k_batch > 1 (split-K)")
    if _is_splitk:
        _k_per_batch = model_dim // k_batch
        assert (
            model_dim % k_batch == 0
        ), f"model_dim={model_dim} not divisible by k_batch={k_batch}"
        assert (
            _k_per_batch % tile_k == 0
        ), f"K_per_batch={_k_per_batch} not divisible by tile_k={tile_k}"

        out_dtype = "bf16"
    else:
        _k_per_batch = model_dim
    _k_dim = _k_per_batch

    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(a_elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            f"tile_m*tile_k*elem_bytes must be divisible by {total_threads}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads

    lds_stride = tile_k

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

    _need_fp4 = out_dtype == "fp4"
    _need_fp8 = out_dtype == "fp8"
    _need_quant = _need_fp4 or _need_fp8
    _need_sort = _need_quant

    if _need_quant:
        _use_cshuffle_epilog = True

    _fp4q_tag = "_fp4q" if _need_fp4 else ""
    _fp8q_tag = "_fp8q" if _need_fp8 else ""
    _sort_tag = "_sort" if _need_sort else ""
    _async_tag = "_async" if use_async_copy else ""
    _sk_tag = f"_sk{k_batch}" if _is_splitk else ""
    _go_tag = "_go" if mock_gate_only else ""
    _gui_tag = "_gui" if gate_up_interleave else ""
    _as1_tag = "_as1" if a_scale_one else ""
    _xcd_tag = f"_xcd{xcd_swizzle}" if xcd_swizzle > 0 else ""
    module_name = (
        f"mfma_moe1_silu_mul_a{a_dtype}_w{b_dtype}_{out_s}"
        f"_t{tile_m}x{tile_n}x{tile_k}_pm{persist_m}{_fp4q_tag}{_fp8q_tag}{_sort_tag}{_async_tag}{_sk_tag}{_go_tag}{_gui_tag}{_as1_tag}{_xcd_tag}_v32"
    ).replace("-", "_")

    # -- LDS sizing --
    _cshuffle_elem_bytes = 4 if _need_quant else (4 if out_is_f32 else 2)
    _single_x_bytes = int(tile_m) * int(lds_stride) * int(a_elem_bytes)
    lds_out_bytes = (
        _cshuffle_elem_bytes * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    )
    lds_tid_bytes = int(tile_m) * 4
    _input_elems = _single_x_bytes if a_elem_bytes == 1 else (_single_x_bytes // 2)

    # Determine whether we need wave-group split for lds_out.
    # Standard layout: pong = max(input, lds_out) + tid, ping = input.
    # When this overflows, split lds_out into two halves across pong & ping.
    _GLOBAL_ALIGN = 1024
    _std_pong = max(_single_x_bytes, lds_out_bytes) + lds_tid_bytes
    _std_ping = _single_x_bytes
    _std_pong_aligned = allocator_pong._align(_std_pong, 128)
    _std_total = allocator_pong._align(
        _std_pong_aligned, _GLOBAL_ALIGN
    ) + allocator_pong._align(_std_ping, 128)
    _lds_limit = {"gfx950": 163840, "gfx942": 65536}.get(gpu_arch, 0)

    _split_lds_out = (
        _lds_limit > 0
        and lds_out_bytes > 0
        and _std_total > _lds_limit
        and num_waves >= 2
    )

    if _split_lds_out:
        _half_out_bytes = _cshuffle_elem_bytes * int(tile_m) * (int(tile_n) // 2)
        _pong_buffer_bytes = max(_single_x_bytes, _half_out_bytes)
        _ping_buffer_bytes = max(_single_x_bytes, _half_out_bytes)
    else:
        _pong_buffer_bytes = max(_single_x_bytes, lds_out_bytes)
        _ping_buffer_bytes = _single_x_bytes

    def x_lds_elem():
        return T.f8

    lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = lds_pong_offset + _pong_buffer_bytes
    _lds_tid_offset_pong = allocator_pong._align(allocator_pong.ptr, 4)
    allocator_pong.ptr = _lds_tid_offset_pong + lds_tid_bytes

    lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = lds_ping_offset + _ping_buffer_bytes

    if waves_per_eu is not None and waves_per_eu >= 1:
        _total_cu_lds = 160 * 1024
        _min_lds = _total_cu_lds // (waves_per_eu + 1) + 1
        _pong_sz = allocator_pong._align(allocator_pong.ptr, 128)
        _ping_sz = allocator_ping._align(allocator_ping.ptr, 128)
        _cur_lds = _pong_sz + _ping_sz
        if _cur_lds < _min_lds:
            allocator_ping.ptr += _min_lds - _cur_lds

    kpack_bytes = 16
    out_elem_bytes = 4 if out_is_f32 else 2
    w_elem_bytes = 1
    w_elem_pack = 2
    w_nbytes = (experts * (2 * inter_dim) * model_dim * w_elem_bytes) // w_elem_pack
    bias_nbytes = experts * (2 * inter_dim) * 4

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
            _pipe_b_loads.append(("gate", ku, ni))
            if not mock_gate_only and not gate_up_interleave:
                _pipe_b_loads.append(("up", ku, ni))

    # MFMA order: B-major (fix B, cycle all A tiles before next B)
    # Each entry: one (k, ni) pair; the compute function loops over all mi.
    # This keeps B operands (from VMEM) fixed while cycling A (from LDS, no wait).
    _pipe_num_acc_n_packed = _pipe_num_acc_n // pack_N
    _pipe_all_mfma = []
    for _ku128 in range(_pipe_k_unroll_packed):
        for _ni_packed in range(_pipe_num_acc_n_packed):
            for _ikxdl in range(pack_K):
                for _inxdl in range(pack_N):
                    _k_idx = _ku128 * pack_K + _ikxdl
                    _ni_idx = _ni_packed * pack_N + _inxdl
                    _pipe_all_mfma.append((_k_idx, _ni_idx, _ikxdl, _inxdl, _ku128))

    # Group MFMAs per scheduling phase (wider M -> more MFMAs per phase)
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
            "mfma": _pipe_all_mfma[_mfma_i : _mfma_i + _pipe_mfma_per_phase],
            "a_reads": _a_reads,
            "b_loads": [],
            "has_scale": (_p == 0),
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
                _pipe_phases[_p]["b_loads"].append(_pipe_b_loads[_bi])
                _bi += 1

    # Extract flat lists for kernel access (avoids dict access in AST rewriter)
    _pp_mfma = [p["mfma"] for p in _pipe_phases]
    _pp_a_reads = [p["a_reads"] for p in _pipe_phases]
    _pp_b_loads = [p["b_loads"] for p in _pipe_phases]
    _pp_has_scale = [p["has_scale"] for p in _pipe_phases]

    fp4_ratio = 2 if a_dtype == "fp4" else 1
    gui_ratio = 1 if gate_up_interleave else 2
    _vmcnt_before_barrier = tile_m // 32 // fp4_ratio + tile_n // 32 * gui_ratio

    @flyc.kernel(name=module_name)
    def moe_gemm1(
        arg_out: fx.Pointer,
        arg_x: fx.Pointer,
        arg_w: fx.Pointer,
        arg_scale_x: fx.Pointer,
        arg_scale_w: fx.Pointer,
        arg_sorted_token_ids: fx.Pointer,
        arg_expert_ids: fx.Pointer,
        arg_sorted_weights: fx.Pointer,
        arg_num_valid_ids: fx.Pointer,
        arg_bias: fx.Pointer,
        arg_out_scale_sorted: fx.Pointer,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
    ):

        tokens_in = fx.Index(i32_tokens_in.ir_value()).ir_value()
        n_in = fx.Index(i32_n_in.ir_value())
        k_in = fx.Index(i32_k_in.ir_value())
        size_expert_ids_in = fx.Index(i32_size_expert_ids_in.ir_value())

        x_elem = T.f8
        f32 = T.f32
        i32 = T.i32
        i64 = T.i64
        vec4_f32 = T.vec(4, f32)
        vec16_elems = 16 if a_elem_bytes == 1 else 8
        vec16_x = T.vec(vec16_elems, x_elem)
        vec2_i64 = T.vec(2, i64)

        def _ptr_buffer_resource(ptr, num_records_bytes):
            addr = fx.ptrtoint(ptr)
            addr_i64 = fx.Int64(addr)
            return buffer_ops.create_buffer_resource_from_addr(
                addr_i64, num_records_bytes=num_records_bytes
            )

        # keep raw: emits a dense-splat arith.constant; Vec.filled would emit
        # a vector.broadcast instead, changing the generated IR.
        acc_init = arith.constant_vector(0.0, vec4_f32)

        # --- Stage1 dimension mapping ---
        # X: [tokens, model_dim] -- M = sorted tokens, K = model_dim
        # W: [E*2*inter_dim, model_dim] gate portion -- N = inter_dim
        # Out: [tokens*topk, inter_dim]

        # B preshuffle layout: [E*2*inter_dim, model_dim]
        # Gate rows for expert e: [e*2*inter_dim, e*2*inter_dim + inter_dim)
        c_n_total = fx.Index(experts * (2 * inter_dim))
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
        sorted_m = size_expert_ids_in * sort_block_m
        layout_a_scale = make_preshuffle_scale_layout(
            c_mn=sorted_m, c_k=fx.Index(model_dim)
        )
        # B-scale: [E*2*inter_dim, K/32]
        layout_b_scale = make_preshuffle_scale_layout(
            c_mn=c_n_total, c_k=fx.Index(model_dim)
        )

        _eff_lds_stride = lds_stride
        _eff_tile_k_bytes = tile_k_bytes
        if const_expr(use_async_copy and a_elem_vec_pack > 1):
            _eff_lds_stride = lds_stride // a_elem_vec_pack
            _eff_tile_k_bytes = tile_k_bytes // a_elem_vec_pack

        shape_lds = fx.make_shape(tile_m, _eff_lds_stride)
        stride_lds = fx.make_stride(_eff_lds_stride, 1)
        layout_lds = fx.make_layout(shape_lds, stride_lds)

        tx = gpu.thread_id("x")
        by = gpu.block_id("x")  # tile along inter_dim (N)
        bx_persist = gpu.block_id("y")  # persistent WG index

        if const_expr(xcd_swizzle > 0):
            _NUM_XCDS_S1 = 8
            _c1_sw = fx.Index(1)
            _c_tn_sw = fx.Index(tile_n)
            _c_idp_sw = fx.Index(2 * inter_dim_pad)
            if const_expr(mock_gate_only or gate_up_interleave):
                _gx = (n_in - _c_idp_sw + _c_tn_sw - _c1_sw) // _c_tn_sw
            else:
                _c2_sw = fx.Index(2)
                _gx = (
                    (n_in - _c_idp_sw + _c2_sw * _c_tn_sw - _c1_sw)
                    // _c_tn_sw
                    // _c2_sw
                )
            _c_pm_sw = fx.Index(persist_m)
            _gy = (size_expert_ids_in + _c_pm_sw - _c1_sw) // _c_pm_sw

            bx_persist, by = xcd_remap_bx_by_grid(
                bx_persist, by, _gx, _gy, xcd_swizzle=xcd_swizzle
            )

        by_n = by * tile_n

        k_base_idx = fx.Index(0)
        if const_expr(_is_splitk):
            bz = gpu.block_id("z")  # K-batch id
            k_base_idx = bz * _k_dim

        k_blocks16 = fx.Index(_eff_tile_k_bytes // 16)
        layout_tx_wave_lane = fx.make_layout((num_waves, 64), stride=(64, 1))
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

        base_ptr_pong = allocator_pong.get_base()
        base_ptr_ping = allocator_ping.get_base()
        lds_x_pong = SmemPtr(
            base_ptr_pong, lds_pong_offset, x_lds_elem(), shape=(_input_elems,)
        ).get()
        lds_x_ping = SmemPtr(
            base_ptr_ping, lds_ping_offset, x_lds_elem(), shape=(_input_elems,)
        ).get()
        _lds_out_elem_type = (
            T.f32 if _need_quant else (T.bf16 if out_is_bf16 else T.f16)
        )
        if const_expr(_split_lds_out and _use_cshuffle_epilog):
            _half_out_elems = int(tile_m) * (int(tile_n) // 2)
            lds_out = SmemPtr(
                base_ptr_pong,
                lds_pong_offset,
                _lds_out_elem_type,
                shape=(_half_out_elems,),
            ).get()
            lds_out_B = SmemPtr(
                base_ptr_ping,
                lds_ping_offset,
                _lds_out_elem_type,
                shape=(_half_out_elems,),
            ).get()
        else:
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
            lds_out_B = None
        lds_tid = SmemPtr(
            base_ptr_pong, _lds_tid_offset_pong, T.i32, shape=(tile_m,)
        ).get()

        # Buffer resources
        c_a_pack = fx.Index(int(a_elem_vec_pack))
        c_elem_bytes = fx.Index(int(a_elem_bytes))

        # X: [tokens, model_dim]
        x_nbytes_idx = (tokens_in * k_in * c_elem_bytes) // c_a_pack
        x_nbytes_i32 = fx.Int32(x_nbytes_idx)
        x_rsrc = _ptr_buffer_resource(arg_x, x_nbytes_i32)

        w_rsrc = _ptr_buffer_resource(arg_w, w_nbytes)

        # Out: [tokens*topk, inter_dim]
        numids_rsrc = _ptr_buffer_resource(arg_num_valid_ids, fx.Int32(4))
        num_valid_i32 = buffer_ops.buffer_load(
            numids_rsrc, fx.Index(0), vec_width=1, dtype=T.i32
        )

        sx_rsrc = 1
        sw_rsrc = 1
        if const_expr(not a_scale_one):
            # A scale: [sorted_size, model_dim/32] pre-scattered by caller
            c32 = fx.Index(32)
            kblk = k_in // c32
            sx_nbytes_idx = sorted_m * kblk
            sx_nbytes_i32 = fx.Int32(sx_nbytes_idx)
            sx_rsrc = _ptr_buffer_resource(arg_scale_x, sx_nbytes_i32)

        c32 = fx.Index(32)
        kblk_w = k_in // c32
        mn_w = fx.Index(experts * (2 * inter_dim))
        sw_nbytes_idx = mn_w * kblk_w
        sw_nbytes_i32 = fx.Int32(sw_nbytes_idx)
        sw_rsrc = _ptr_buffer_resource(arg_scale_w, sw_nbytes_i32)

        sorted_nbytes_idx = size_expert_ids_in * (sort_block_m * 4)
        sorted_nbytes_i32 = fx.Int32(sorted_nbytes_idx)
        sorted_rsrc = _ptr_buffer_resource(arg_sorted_token_ids, sorted_nbytes_i32)
        sorted_w_rsrc = _ptr_buffer_resource(arg_sorted_weights, sorted_nbytes_i32)

        eid_nbytes_idx = size_expert_ids_in * 4
        eid_nbytes_i32 = fx.Int32(eid_nbytes_idx)
        expert_rsrc = _ptr_buffer_resource(arg_expert_ids, eid_nbytes_i32)
        bias_rsrc = _ptr_buffer_resource(arg_bias, bias_nbytes) if enable_bias else None

        # Sorted-scale buffer resource for fused mxfp4 quantization
        _sorted_scale_cols = inter_dim // 32
        _sorted_scale_cols_i32 = fx.Int32(_sorted_scale_cols)
        sorted_scale_rsrc = None
        if const_expr(_need_sort):
            _sort_rows_idx = size_expert_ids_in * sort_block_m
            _sort_padded_rows = (_sort_rows_idx + 255) // 256 * 256
            _sort_padded_cols = fx.Index(((_sorted_scale_cols + 7) // 8) * 8)
            _sort_scale_nbytes = fx.Int32(_sort_padded_rows * _sort_padded_cols)
            sorted_scale_rsrc = _ptr_buffer_resource(
                arg_out_scale_sorted, _sort_scale_nbytes
            )

        # ---- persist_m loop (same pattern as stage2) ----
        _PERSIST_M = persist_m
        _c0_p = fx.Index(0).ir_value()
        _c1_p = fx.Index(1).ir_value()
        _c_pm = fx.Index(_PERSIST_M).ir_value()
        # keep raw scf.ForOp: the persistent loop is built imperatively with
        # __enter__/__exit__ + induction_variable + inner_iter_args spanning the
        # whole body; rewriting to `for ... in range(init=)` would require
        # restructuring the entire body and risks changing the generated IR.
        _for_persist = scf.ForOp(_c0_p, _c_pm, _c1_p)
        _for_ip = ir.InsertionPoint(_for_persist.body)
        _for_ip.__enter__()
        _mi_p = _for_persist.induction_variable
        bx = bx_persist * _c_pm + _mi_p
        bx_m = bx * sort_block_m

        # Block validity
        bx_m_i32 = fx.Int32(bx_m).ir_value()
        blk_valid = fx.Uint32(bx_m_i32) < num_valid_i32
        expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=T.i32)
        expert_idx = fx.Index(expert_i32)
        exp_valid = fx.Uint32(expert_i32) < experts

        def _moe_gemm1_body():
            # Gate expert offset: first inter_dim rows of each expert's 2*inter_dim block
            expert_off_idx = expert_idx * (2 * inter_dim)

            # X loading -- KEY DIFFERENCE from stage2: X row = token_id only
            x_load_bytes = 16
            num_x_loads = bytes_per_thread_x // x_load_bytes
            chunk_i32 = x_load_bytes // 4

            c_k_div4 = ((k_in // c_a_pack) * fx.Index(int(a_elem_bytes))) // 4
            tile_k_dwords = (int(tile_k) * int(a_elem_bytes)) // (
                4 * int(a_elem_vec_pack)
            )
            layout_x_tile_div4 = fx.make_layout(
                (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
            )
            c_chunk_i32 = fx.Index(chunk_i32)
            tx_i32_base = tx * c_chunk_i32

            topk_i32 = fx.Int32(topk)
            mask24 = fx.Int32(0xFFFFFF)
            tokens_i32 = fx.Int32(tokens_in)

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
                idx_elem = idx_i32 if a_elem_bytes == 1 else (idx_i32 * 2)
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

            for i in range_constexpr(num_x_loads):
                row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                x_row_local.append(row_local)
                x_col_local_i32.append(col_local_i32)

                sorted_row_i = bx_m + row_local
                fused_i = buffer_ops.buffer_load(
                    sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32
                )
                t_i32 = fused_i & mask24
                s_i32 = (fx.Uint32(fused_i) >> 24).ir_value()
                t_valid = fx.Uint32(t_i32) < tokens_i32
                s_valid = fx.Uint32(s_i32) < topk_i32
                ts_valid = t_valid & s_valid
                t_safe = ts_valid.select(t_i32, 0)

                # KEY: X row base uses token_id only (not t*topk+s)
                t_idx = fx.Index(t_safe)
                x_row_base_div4.append(t_idx * c_k_div4)

            def load_x_tile(base_k):
                base_k_div4 = ((base_k // c_a_pack) * fx.Index(int(a_elem_bytes))) // 4
                parts = []
                for i in range_constexpr(num_x_loads):
                    idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                    x_vec = load_x(idx_i32)
                    parts.append(Vec(x_vec).bitcast(fx.Int32))
                return parts

            # Wave/lane decomposition (identical to stage2)
            coord_wl = idx2crd(tx, layout_tx_wave_lane)
            wave_id = layout_get(coord_wl, 0)
            lane_id = layout_get(coord_wl, 1)
            coord_l16 = idx2crd(lane_id, layout_lane16)
            lane_div_16 = layout_get(coord_l16, 0)
            lane_mod_16 = layout_get(coord_l16, 1)
            row_a_lds = lane_mod_16
            col_offset_base = lane_div_16 * 16

            num_acc_n = n_per_wave // 16
            c_n_per_wave = fx.Index(n_per_wave)
            wave_n_id = wave_id % num_waves
            n_tile_base = wave_n_id * c_n_per_wave

            # N-tile precompute for gate AND up weights
            gate_n_intra_list = []
            gate_n_blk_list = []
            up_n_intra_list = []
            up_n_blk_list = []
            col_g_list = []
            c_n0_static = experts * (2 * inter_dim) // 16
            layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))
            inter_idx = fx.Index(inter_dim)

            for i in range_constexpr(num_acc_n):
                offset = i * 16
                c_offset = fx.Index(offset)
                if const_expr(not gate_up_interleave):
                    col_g = by_n + n_tile_base + c_offset + lane_mod_16
                    col_g_list.append(col_g)

                global_n = by_n + n_tile_base + c_offset + lane_mod_16
                # Gate/interleave: rows [expert_off, expert_off + 2*inter_dim)
                gate_row_w = expert_off_idx + global_n
                gate_coord = idx2crd(gate_row_w, layout_n_blk_intra)
                gate_n_blk_list.append(layout_get(gate_coord, 0))
                gate_n_intra_list.append(layout_get(gate_coord, 1))
                if const_expr(not mock_gate_only and not gate_up_interleave):
                    up_row_w = gate_row_w + inter_idx
                    up_coord = idx2crd(up_row_w, layout_n_blk_intra)
                    up_n_blk_list.append(layout_get(up_coord, 0))
                    up_n_intra_list.append(layout_get(up_coord, 1))

            if const_expr(gate_up_interleave):
                _gui_num_acc_n_out = num_acc_n // pack_N
                for _gui_i in range_constexpr(_gui_num_acc_n_out):
                    _gui_offset = _gui_i * 16
                    _gui_c_offset = fx.Index(_gui_offset)
                    _gui_col_g = (by_n + n_tile_base) // 2 + _gui_c_offset + lane_mod_16
                    col_g_list.append(_gui_col_g)

            m_repeat = tile_m // 16
            k_unroll = tile_k_bytes // 128
            k_unroll_packed = k_unroll // pack_K
            m_repeat_packed = m_repeat // pack_M
            num_acc_n_packed = num_acc_n // pack_N

            _K_per_ku = tile_k // k_unroll
            _pad_k_elems = (
                (model_dim_pad % tile_k)
                if (not _is_splitk and model_dim_pad > 0)
                else 0
            )
            _pad_ku_skip = _pad_k_elems // _K_per_ku
            _tail_ku = k_unroll - _pad_ku_skip
            _tail_ku_packed = (
                (_tail_ku + pack_K - 1) // pack_K if _pad_ku_skip > 0 else None
            )

            # B load for gate and up separately
            def load_b_packs_k64(base_k, ku: int, n_blk, n_intra):
                c64 = fx.Index(64)
                base_k_bytes = base_k * fx.Index(int(b_elem_bytes))
                k0 = base_k_bytes // c64 + ku
                k1 = lane_div_16
                coord_pack = (n_blk, k0, k1, n_intra, fx.Index(0))
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
                b_i64x2 = Vec(b16).bitcast(fx.Int64)
                b0 = Vec(b_i64x2)[0]
                b1 = Vec(b_i64x2)[1]
                return b0, b1

            def load_b_tile(base_k, ku_limit=k_unroll):
                """Load B tiles. Returns (gate_b_tile, up_b_tile).
                When mock_gate_only or gate_up_interleave, up_b_tile is None."""
                gate_b_tile = []
                up_b_tile = (
                    [] if (not mock_gate_only and not gate_up_interleave) else None
                )
                for ku in range_constexpr(ku_limit):
                    g_packs0, g_packs1 = [], []
                    u_packs0, u_packs1 = [], []
                    for ni in range_constexpr(num_acc_n):
                        gb0, gb1 = load_b_packs_k64(
                            base_k, ku, gate_n_blk_list[ni], gate_n_intra_list[ni]
                        )
                        g_packs0.append(gb0)
                        g_packs1.append(gb1)
                        if const_expr(not mock_gate_only and not gate_up_interleave):
                            ub0, ub1 = load_b_packs_k64(
                                base_k, ku, up_n_blk_list[ni], up_n_intra_list[ni]
                            )
                            u_packs0.append(ub0)
                            u_packs1.append(ub1)
                    gate_b_tile.append((g_packs0, g_packs1))
                    if const_expr(not mock_gate_only and not gate_up_interleave):
                        up_b_tile.append((u_packs0, u_packs1))
                return gate_b_tile, up_b_tile

            # Pre-compute scale base element indices (K-loop invariant).
            # idx = mni * stride_n0 + ku * stride_k0 + k_lane * stride_klane + n_lane
            # Split into: base_elem = mni * stride_n0 + lane_elem (invariant)
            #              k_elem    = ku * stride_k0             (per-iteration)
            _scale_lane_elem = lane_div_16 * layout_b_scale.stride_klane + lane_mod_16

            _gate_scale_bases = []
            _up_scale_bases = []
            for _ni in range_constexpr(num_acc_n_packed):
                _col_base = by_n + n_tile_base + (_ni * 16 * pack_N)
                _gate_mni = (expert_off_idx + _col_base) // 32
                _gate_scale_bases.append(
                    _gate_mni * layout_b_scale.stride_n0 + _scale_lane_elem
                )
                if const_expr(not mock_gate_only and not gate_up_interleave):
                    _up_mni = (expert_off_idx + inter_idx + _col_base) // 32
                    _up_scale_bases.append(
                        _up_mni * layout_b_scale.stride_n0 + _scale_lane_elem
                    )

            if const_expr(not a_scale_one):
                _a_scale_bases = []
                for _mi in range_constexpr(m_repeat_packed):
                    _a_mni = _mi + bx_m // scale_mn_pack // 16
                    _a_scale_bases.append(
                        _a_mni * layout_a_scale.stride_n0 + _scale_lane_elem
                    )

            _scale_mask_lo = fx.Int32(0xFF)

            _m_half_idx = fx.Int32(0).ir_value()
            _m_half_i32 = fx.Int32(0)
            _scale_shift = fx.Int32(0)
            _scale_shift_hi = fx.Int32(0)
            _n_half_idx = fx.Int32(0).ir_value()
            _n_half_i32 = fx.Int32(0)
            _bscale_shift = fx.Int32(0)
            _bscale_shift_hi = fx.Int32(0)
            if const_expr(pack_M < scale_mn_pack):
                _m_half_idx = (bx_m // 16) % 2
                _m_half_i32 = fx.Int32(_m_half_idx)
                _scale_shift = _m_half_i32 * 8
                _scale_shift_hi = _scale_shift + 16

            if const_expr(pack_N < scale_mn_pack):
                _n_half_idx = (n_tile_base // 16) % 2
                _n_half_i32 = fx.Int32(_n_half_idx)
                _bscale_shift = _n_half_i32 * 8
                _bscale_shift_hi = _bscale_shift + 16

            def _rearrange_a_scale(raw_i32):
                """Rearrange scale bytes for pack_M=1: extract m_half's k0,k1 bytes."""
                if const_expr(pack_M >= scale_mn_pack):
                    return raw_i32
                b_k0 = (fx.Uint32(raw_i32) >> _scale_shift) & _scale_mask_lo
                b_k1 = (fx.Uint32(raw_i32) >> _scale_shift_hi) & _scale_mask_lo
                return b_k0 | (b_k1 << 8)

            def _rearrange_b_scale(raw_i32):
                """Rearrange scale bytes for pack_N=1: extract n_half's k0,k1 bytes."""
                if const_expr(pack_N >= scale_mn_pack):
                    return raw_i32
                b_k0 = (fx.Uint32(raw_i32) >> _bscale_shift) & _scale_mask_lo
                b_k1 = (fx.Uint32(raw_i32) >> _bscale_shift_hi) & _scale_mask_lo
                return b_k0 | (b_k1 << 8)

            if const_expr(a_scale_one):
                _as1_const = fx.Int32(0x7F7F7F7F)
                _as1_vec = Vec.from_elements([_as1_const], fx.Int32)

            def prefetch_ab_scale_tile(base_k, ku_packed_limit=k_unroll_packed):
                a_scale_tile = []
                gate_b_scale = []
                up_b_scale = (
                    [] if (not mock_gate_only and not gate_up_interleave) else None
                )
                for ku in range_constexpr(ku_packed_limit):
                    k_off = (ku + base_k) * layout_b_scale.stride_k0
                    for mi in range_constexpr(m_repeat_packed):
                        if const_expr(a_scale_one):
                            a_scale_tile.append(_as1_vec)
                        else:
                            s = buffer_ops.buffer_load(
                                sx_rsrc,
                                _a_scale_bases[mi] + k_off,
                                vec_width=1,
                                dtype=T.i32,
                                cache_modifier=0,
                            )
                            s = _rearrange_a_scale(s)
                            a_scale_tile.append(Vec.from_elements([s], fx.Int32))
                    for ni in range_constexpr(num_acc_n_packed):
                        gs = buffer_ops.buffer_load(
                            sw_rsrc,
                            _gate_scale_bases[ni] + k_off,
                            vec_width=1,
                            dtype=T.i32,
                            cache_modifier=0,
                        )
                        gs = _rearrange_b_scale(gs)
                        gate_b_scale.append(Vec.from_elements([gs], fx.Int32))
                        if const_expr(not mock_gate_only and not gate_up_interleave):
                            us = buffer_ops.buffer_load(
                                sw_rsrc,
                                _up_scale_bases[ni] + k_off,
                                vec_width=1,
                                dtype=T.i32,
                                cache_modifier=0,
                            )
                            us = _rearrange_b_scale(us)
                            up_b_scale.append(Vec.from_elements([us], fx.Int32))
                return [a_scale_tile, gate_b_scale, up_b_scale]

            _lds_base_zero = fx.Index(0)

            def store_x_tile_to_lds(vec_x_in_parts, lds_buffer):
                for i in range_constexpr(num_x_loads):
                    row_local = x_row_local[i]
                    col_local_i32 = x_col_local_i32[i]
                    if const_expr(x_load_bytes == 16):
                        lds_store_16b_xor16(
                            arith,
                            vector,
                            lds_memref=lds_buffer,
                            vec16_ty=vec16_x,
                            layout_lds=layout_lds,
                            row_local=row_local,
                            col_local_i32=col_local_i32,
                            tx_c4=fx.Index(4),
                            k_blocks16=k_blocks16,
                            lds_base=_lds_base_zero,
                            vec_part_i32x4=vec_x_in_parts[i],
                            elem_bytes=elem_bytes,
                        )

            if const_expr(use_async_copy):
                _dma_bytes = 16
                _wave_size = 64
                _eff_bytes_per_buffer = (
                    int(tile_m) * int(_eff_lds_stride) * int(a_elem_bytes)
                )
                _num_dma_loads = max(
                    1, _eff_bytes_per_buffer // (total_threads * _dma_bytes)
                )

                def dma_x_tile_to_lds(base_k, lds_buffer):
                    c4_idx = fx.Index(4)
                    base_k_div4 = (
                        (base_k // c_a_pack) * fx.Index(int(elem_bytes))
                    ) // 4

                    lds_ptr_i64 = None
                    for i in range_constexpr(_num_dma_loads):
                        row_local_i = x_row_local[i]
                        col_local_i32_i = x_col_local_i32[i]
                        col_local_sw = swizzle_xor16(
                            row_local_i, col_local_i32_i * c4_idx, k_blocks16
                        )
                        row_k_dw = x_row_base_div4[i] + base_k_div4
                        global_byte_idx = row_k_dw * c4_idx + col_local_sw
                        global_offset = fx.Int32(global_byte_idx)

                        if const_expr(i == 0):
                            lds_addr = memref.extract_aligned_pointer_as_index(
                                lds_buffer
                            ) + wave_id * (_wave_size * _dma_bytes)
                            lds_ptr_i64 = rocdl.readfirstlane(T.i64, fx.Int64(lds_addr))
                        else:
                            lds_ptr_i64 = lds_ptr_i64 + (total_threads * _dma_bytes)

                        lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                        lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64)

                        rocdl.raw_ptr_buffer_load_lds(
                            x_rsrc,
                            lds_ptr,
                            fx.Int32(_dma_bytes),
                            global_offset,
                            fx.Int32(0),
                            fx.Int32(0),
                            fx.Int32(0),
                        )

                def prefetch_x_to_lds(base_k, lds_buffer):
                    dma_x_tile_to_lds(base_k, lds_buffer)

            def lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer):
                col_base_swz_bytes = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
                col_base_swz = (
                    col_base_swz_bytes if elem_bytes == 1 else (col_base_swz_bytes // 2)
                )
                idx_a16 = crd2idx([curr_row_a_lds, col_base_swz], layout_lds)
                loaded_a16 = Vec.load(vec16_x, lds_buffer, [idx_a16])
                a_i64x2 = Vec(loaded_a16).bitcast(fx.Int64)
                a0 = Vec(a_i64x2)[0]
                a1 = Vec(a_i64x2)[1]
                return a0, a1

            def prefetch_full_a_from_lds(lds_buffer, ku_limit=k_unroll):
                """Load entire A tile from LDS into registers before compute."""
                a_regs = []
                for k_idx in range_constexpr(ku_limit):
                    col_base = col_offset_base + (k_idx * 128) // a_elem_vec_pack
                    for mi_idx in range_constexpr(m_repeat):
                        mi_val = fx.Index(mi_idx * 16)
                        curr_row = row_a_lds + mi_val
                        a0, a1 = lds_load_packs_k64(curr_row, col_base, lds_buffer)
                        if const_expr(is_f8_a):
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
                ku_count=k_unroll,
            ):
                gate_list = list(acc_gate_in)
                _single_b = mock_gate_only or gate_up_interleave
                up_list = None if _single_b else list(acc_up_in)
                mfma_res_ty = vec4_f32
                epilogue_pf = None
                bias_pf = None
                if const_expr(prefetch_epilogue):
                    if const_expr(enable_bias):
                        if const_expr(gate_up_interleave):
                            bias_pf = []
                            for ni in range_constexpr(num_acc_n):
                                _logical_col = (
                                    (by_n + n_tile_base) // 2
                                    + fx.Index((ni // 2) * 16)
                                    + lane_mod_16
                                )
                                _up_off = inter_idx if (ni % 2 == 1) else fx.Index(0)
                                bias_offset = expert_off_idx + _up_off + _logical_col
                                bias_pf.append(
                                    _load_bias_scalar(bias_rsrc, bias_offset)
                                )
                        else:
                            gate_bias_pf = []
                            up_bias_pf = [] if const_expr(not mock_gate_only) else None
                            for ni in range_constexpr(num_acc_n):
                                global_n = by_n + n_tile_base + (ni * 16) + lane_mod_16
                                gate_bias_pf.append(
                                    _load_bias_scalar(
                                        bias_rsrc, expert_off_idx + global_n
                                    )
                                )
                                if const_expr(not mock_gate_only):
                                    up_bias_pf.append(
                                        _load_bias_scalar(
                                            bias_rsrc,
                                            expert_off_idx + inter_idx + global_n,
                                        )
                                    )
                            bias_pf = (gate_bias_pf, up_bias_pf)
                    tw_pf = None
                    if const_expr(doweight_stage1):
                        tw_pf = []
                        lane_div_16_mul4_pf = lane_div_16 * 4
                        ii_idx_list_pf = [fx.Index(ii) for ii in range(4)]
                        for mi in range_constexpr(m_repeat):
                            mi_base_pf = fx.Index(mi * 16)
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
                    epilogue_pf = (None, tw_pf, bias_pf)

                c0_i64 = fx.Int64(0)
                vec4_i64 = T.vec(4, T.i64)
                vec8_i32 = T.vec(8, T.i32)

                def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                    v4 = Vec.from_elements([x0, x1, x2, x3], fx.Int64)
                    return Vec(v4).bitcast(fx.Int32)

                _eff_packed = (ku_count + pack_K - 1) // pack_K
                # B-major: fix B (ni), cycle A (mi) -- B from VMEM stays
                # in registers while A from LDS is repacked per mi.
                for ku128 in range_constexpr(_eff_packed):
                    for ni in range_constexpr(num_acc_n_packed):
                        gate_bs_i32 = gate_b_scale[ku128 * num_acc_n_packed + ni]
                        gate_bs_val = Vec(gate_bs_i32)[0]
                        if const_expr(not _single_b):
                            up_bs_i32 = up_b_scale[ku128 * num_acc_n_packed + ni]
                            up_bs_val = Vec(up_bs_i32)[0]
                        for ikxdl in range_constexpr(pack_K):
                            k_idx = ku128 * pack_K + ikxdl
                            if const_expr(k_idx < ku_count):
                                gate_bp0, gate_bp1 = gate_b_tile_in[k_idx]
                                if const_expr(not _single_b):
                                    up_bp0, up_bp1 = up_b_tile_in[k_idx]
                                for inxdl in range_constexpr(pack_N):
                                    ni_idx = ni * pack_N + inxdl
                                    gb0 = gate_bp0[ni_idx]
                                    gb1 = gate_bp1[ni_idx]
                                    gb128 = pack_i64x4_to_i32x8(
                                        gb0, gb1, c0_i64, c0_i64
                                    )
                                    if const_expr(not _single_b):
                                        ub0 = up_bp0[ni_idx]
                                        ub1 = up_bp1[ni_idx]
                                        ub128 = pack_i64x4_to_i32x8(
                                            ub0, ub1, c0_i64, c0_i64
                                        )
                                    for mi in range_constexpr(m_repeat_packed):
                                        a_scale_i32 = a_scale[
                                            ku128 * m_repeat_packed + mi
                                        ]
                                        a_scale_val = Vec(a_scale_i32)[0].ir_value()
                                        for imxdl in range_constexpr(pack_M):
                                            mi_idx = mi * pack_M + imxdl
                                            _a_reg_idx = k_idx * m_repeat + mi_idx
                                            if const_expr(is_f8_a):
                                                a0, a1, a2, a3 = a_tile_regs[_a_reg_idx]
                                                a128 = pack_i64x4_to_i32x8(
                                                    a0, a1, a2, a3
                                                )
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
                                            if const_expr(not _single_b):
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
                col_base = col_offset_base + (k_idx * 128) // a_elem_vec_pack
                mi_val = fx.Index(mi_idx * 16)
                curr_row = row_a_lds + mi_val
                a0, a1 = lds_load_packs_k64(curr_row, col_base, lds_buffer)
                if const_expr(is_f8_a):
                    a2, a3 = lds_load_packs_k64(curr_row, col_base + 64, lds_buffer)
                    return (a0, a1, a2, a3)
                else:
                    return (a0, a1)

            _single_b_pipe = mock_gate_only or gate_up_interleave

            def compute_bmajor_mfma_phase(
                all_a_tiles,
                gate_b_single,
                up_b_single,
                a_scale_vals,
                gate_bs_val,
                up_bs_val,
                gate_list,
                up_list,
                k_idx,
                ni_idx,
                ikxdl,
                inxdl,
            ):
                """B-major MFMA: fix one B (ni), cycle all A tiles (mi).

                Packs B once and reuses across all mi iterations.
                A tiles come from LDS (already available, no VMEM wait).

                all_a_tiles: flat list indexed by [k*m_repeat + mi].
                gate_b_single/up_b_single: (b0, b1) for one specific ni.
                  When _single_b_pipe (mock_gate_only or interleave), up_b_single is None.
                a_scale_vals: list of A scale scalars indexed by mi_packed.
                """
                c0_i64 = fx.Int64(0)
                vec4_i64 = T.vec(4, T.i64)
                vec8_i32 = T.vec(8, T.i32)

                def _pack(x0, x1, x2, x3):
                    v4 = Vec.from_elements([x0, x1, x2, x3], fx.Int64)
                    return Vec(v4).bitcast(fx.Int32)

                mfma_res_ty = vec4_f32
                gb128 = _pack(gate_b_single[0], gate_b_single[1], c0_i64, c0_i64)
                if const_expr(not _single_b_pipe):
                    ub128 = _pack(up_b_single[0], up_b_single[1], c0_i64, c0_i64)

                for mi_p in range_constexpr(m_repeat_packed):
                    a_scale_val = a_scale_vals[mi_p]
                    for imxdl in range_constexpr(pack_M):
                        mi_idx = mi_p * pack_M + imxdl
                        a_reg = all_a_tiles[k_idx * m_repeat + mi_idx]

                        if const_expr(is_f8_a):
                            a128 = _pack(a_reg[0], a_reg[1], a_reg[2], a_reg[3])
                        else:
                            a128 = _pack(a_reg[0], a_reg[1], c0_i64, c0_i64)

                        acc_idx = mi_idx * num_acc_n + ni_idx
                        gate_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
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
                        if const_expr(not _single_b_pipe):
                            up_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
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

            def _interleaved_half(
                lds_read,
                lds_write,
                next_k_dma_py,
                next_k_load,
                prev_a_tile,
                prev_gate_w,
                prev_up_w,
                prev_a_scale,
                prev_gate_bs,
                prev_up_bs,
                acc_gate,
                acc_up,
            ):
                """One flatmm-style interleaved half-iteration (deep pipeline).

                Generalized for arbitrary m_repeat (block_m=32, 64, ...).
                DMA targets lds_write (OTHER buffer) while ds_read uses
                lds_read (already DMA'd in previous half).

                Interleaving schedule (per half):
                  Phase 0: scale VMEM + 2 ds_read(A) -> 4 MFMA(prev)
                  Phase 1..N: B VMEM(distributed) + 2 ds_read(A, if avail) -> 4 MFMA(prev)
                  Phase N+1..: remaining B VMEM -> 4 MFMA(prev)
                """
                _abs_k = k_base_idx + next_k_load
                _bk = _abs_k // 2
                _sk = _abs_k // (pack_K * 128)
                _k_off = _sk * layout_b_scale.stride_k0

                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(_vmcnt_before_barrier)
                _barrier()
                rocdl.sched_barrier(0)

                # DMA A to OTHER buffer (for next half), non-blocking
                _abs_k_dma = k_base_idx + next_k_dma_py
                if const_expr(use_async_copy and next_k_dma_py < int(_k_dim)):
                    prefetch_x_to_lds(_abs_k_dma, lds_write)
                if const_expr(not use_async_copy):
                    _x_regs = load_x_tile(_abs_k_dma)

                # ---- Extract previous scale values ----
                _prev_asvs = []
                for _mi_p in range_constexpr(m_repeat_packed):
                    _prev_asvs.append(Vec(prev_a_scale[_mi_p])[0])
                _prev_gsv_list = []
                for _gs_ni in range_constexpr(num_acc_n_packed):
                    _prev_gsv_list.append(Vec(prev_gate_bs[_gs_ni])[0])
                if const_expr(not _single_b_pipe):
                    _prev_usv_list = []
                    for _us_ni in range_constexpr(num_acc_n_packed):
                        _prev_usv_list.append(Vec(prev_up_bs[_us_ni])[0])

                # ---- Execute phases from unified schedule ----
                _a_all = {}
                _b_gate_all = {}
                _b_up_all = {}

                for _p in range_constexpr(_pipe_n_phases):
                    # Scale VMEM loads (phase 0 only)
                    if const_expr(_pp_has_scale[_p]):
                        _new_as_list = []
                        for _mi_p in range_constexpr(m_repeat_packed):
                            if const_expr(a_scale_one):
                                _new_as_list.append(_as1_const)
                            else:
                                _raw_as = buffer_ops.buffer_load(
                                    sx_rsrc,
                                    _a_scale_bases[_mi_p] + _k_off,
                                    vec_width=1,
                                    dtype=T.i32,
                                    cache_modifier=0,
                                )
                                _new_as_list.append(_rearrange_a_scale(_raw_as))
                        _new_gs_list = []
                        for _gs_ni in range_constexpr(num_acc_n_packed):
                            _gs_raw = buffer_ops.buffer_load(
                                sw_rsrc,
                                _gate_scale_bases[_gs_ni] + _k_off,
                                vec_width=1,
                                dtype=T.i32,
                                cache_modifier=0,
                            )
                            _new_gs_list.append(_rearrange_b_scale(_gs_raw))
                        if const_expr(not _single_b_pipe):
                            _new_us_list = []
                            for _us_ni in range_constexpr(num_acc_n_packed):
                                _us_raw = buffer_ops.buffer_load(
                                    sw_rsrc,
                                    _up_scale_bases[_us_ni] + _k_off,
                                    vec_width=1,
                                    dtype=T.i32,
                                    cache_modifier=0,
                                )
                                _new_us_list.append(_rearrange_b_scale(_us_raw))

                    # B VMEM loads
                    for _b_j in range_constexpr(len(_pp_b_loads[_p])):
                        _b_type, _b_ku, _b_ni = _pp_b_loads[_p][_b_j]
                        if const_expr(_b_type == "gate"):
                            _b_gate_all[(_b_ku, _b_ni)] = load_b_packs_k64(
                                _bk,
                                _b_ku,
                                gate_n_blk_list[_b_ni],
                                gate_n_intra_list[_b_ni],
                            )
                        else:
                            _b_up_all[(_b_ku, _b_ni)] = load_b_packs_k64(
                                _bk,
                                _b_ku,
                                up_n_blk_list[_b_ni],
                                up_n_intra_list[_b_ni],
                            )

                    # A ds_reads
                    rocdl.sched_barrier(0)
                    for _a_j in range_constexpr(len(_pp_a_reads[_p])):
                        _ak, _ami = _pp_a_reads[_p][_a_j]
                        _a_all[(_ak, _ami)] = load_a_subtile(
                            _ak,
                            _ami,
                            lds_read,
                        )
                    rocdl.sched_barrier(0)

                    # MFMAs on prev data
                    rocdl.s_setprio(1)
                    for _m_j in range_constexpr(len(_pp_mfma[_p])):
                        _k_idx, _ni_idx, _ikxdl, _inxdl, _ku128 = _pp_mfma[_p][_m_j]
                        _ni_packed_idx = _ni_idx // pack_N
                        _up_b_single = (
                            (
                                prev_up_w[_k_idx][0][_ni_idx],
                                prev_up_w[_k_idx][1][_ni_idx],
                            )
                            if not _single_b_pipe
                            else None
                        )
                        compute_bmajor_mfma_phase(
                            prev_a_tile,
                            (
                                prev_gate_w[_k_idx][0][_ni_idx],
                                prev_gate_w[_k_idx][1][_ni_idx],
                            ),
                            _up_b_single,
                            _prev_asvs,
                            _prev_gsv_list[_ni_packed_idx],
                            (
                                _prev_usv_list[_ni_packed_idx]
                                if not _single_b_pipe
                                else None
                            ),
                            acc_gate,
                            acc_up,
                            _k_idx,
                            _ni_idx,
                            _ikxdl,
                            _inxdl,
                        )
                    rocdl.s_setprio(0)
                    rocdl.sched_barrier(0)

                # ---- Assemble loaded data for next half-iteration ----
                cur_a_tile = []
                for _k in range_constexpr(k_unroll):
                    for _mi in range_constexpr(m_repeat):
                        cur_a_tile.append(_a_all[(_k, _mi)])

                cur_gate_w = []
                cur_up_w = None if _single_b_pipe else []
                for ku in range_constexpr(k_unroll):
                    g_packs0, g_packs1 = [], []
                    u_packs0, u_packs1 = [], []
                    for ni in range_constexpr(num_acc_n):
                        g = _b_gate_all[(ku, ni)]
                        g_packs0.append(g[0])
                        g_packs1.append(g[1])
                        if const_expr(not _single_b_pipe):
                            u = _b_up_all[(ku, ni)]
                            u_packs0.append(u[0])
                            u_packs1.append(u[1])
                    cur_gate_w.append((g_packs0, g_packs1))
                    if const_expr(not _single_b_pipe):
                        cur_up_w.append((u_packs0, u_packs1))

                cur_a_scale = []
                for _mi_p in range_constexpr(m_repeat_packed):
                    cur_a_scale.append(
                        Vec.from_elements([_new_as_list[_mi_p]], fx.Int32)
                    )
                cur_gate_bs = []
                for _gs_ni in range_constexpr(num_acc_n_packed):
                    cur_gate_bs.append(
                        Vec.from_elements([_new_gs_list[_gs_ni]], fx.Int32)
                    )
                if const_expr(not _single_b_pipe):
                    cur_up_bs = []
                    for _us_ni in range_constexpr(num_acc_n_packed):
                        cur_up_bs.append(
                            Vec.from_elements([_new_us_list[_us_ni]], fx.Int32)
                        )
                else:
                    cur_up_bs = None

                if const_expr(not use_async_copy):
                    store_x_tile_to_lds(_x_regs, lds_write)

                return (
                    cur_a_tile,
                    cur_gate_w,
                    cur_up_w,
                    cur_a_scale,
                    cur_gate_bs,
                    cur_up_bs,
                    acc_gate,
                    acc_up,
                )

            # Pipeline (split ping/pong allocators)
            rocdl.sched_barrier(0)

            k0 = k_base_idx
            if const_expr(use_async_copy):
                prefetch_x_to_lds(k0, lds_x_pong)
            else:
                x_regs0 = load_x_tile(k0)
                store_x_tile_to_lds(x_regs0, lds_x_pong)
            rocdl.sched_barrier(0)
            _k0_scale = k_base_idx // (pack_K * 128)
            a_scale_pong, gate_bs_pong, up_bs_pong = prefetch_ab_scale_tile(_k0_scale)
            if fx.Index(tx) < fx.Index(tile_m):
                _tid_row = bx_m + tx
                _tid_val = buffer_ops.buffer_load(
                    sorted_rsrc, _tid_row, vec_width=1, dtype=T.i32
                )
                _tid_vec1 = Vec.from_elements([_tid_val], fx.Int32)
                Vec(_tid_vec1).store(lds_tid, [tx])

            acc_gate = [acc_init] * num_acc_n * m_repeat
            acc_up = [acc_init] * num_acc_n * m_repeat if not _single_b_pipe else None

            _k1 = k_base_idx + tile_k
            rocdl.sched_barrier(0)
            if const_expr(use_async_copy):
                prefetch_x_to_lds(_k1, lds_x_ping)
            else:
                _x_regs_prime = load_x_tile(_k1)
                store_x_tile_to_lds(_x_regs_prime, lds_x_ping)

            _k0_b = k_base_idx // 2
            gate_w0, up_w0 = load_b_tile(_k0_b)
            # Prime the deep pipeline: DMA K=tile_k -> ping (1 tile ahead)
            if const_expr(use_async_copy):
                rocdl.s_waitcnt(0)
            gpu.barrier()
            rocdl.sched_barrier(0)
            a_tile_pong = prefetch_full_a_from_lds(lds_x_pong)

            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(6)

            num_k_tiles_py = int(_k_dim) // int(tile_k)
            odd_k_tiles = (num_k_tiles_py % 2) == 1
            tail_tiles = 1 if odd_k_tiles else 2
            k_main2_py = (num_k_tiles_py - tail_tiles) * int(tile_k)
            if const_expr(k_main2_py < 0):
                k_main2_py = 0

            gate_w_pong = gate_w0
            up_w_pong = up_w0

            rocdl.sched_barrier(0)

            if const_expr(k_main2_py > 0):
                for k_iv_py in range_constexpr(0, k_main2_py, tile_k * 2):
                    next_k_load_1 = k_iv_py + tile_k
                    next_k_load_2 = k_iv_py + tile_k * 2
                    next_k_dma_1 = k_iv_py + tile_k * 2
                    next_k_dma_2 = k_iv_py + tile_k * 3

                    # Half 1: read ping (DMA'd prev half), DMA->pong, MFMA(pong)
                    (
                        a_tile_ping,
                        gate_w_ping,
                        up_w_ping,
                        a_scale_ping,
                        gate_bs_ping,
                        up_bs_ping,
                        acc_gate,
                        acc_up,
                    ) = _interleaved_half(
                        lds_x_ping,
                        lds_x_pong,
                        next_k_dma_1,
                        next_k_load_1,
                        a_tile_pong,
                        gate_w_pong,
                        up_w_pong,
                        a_scale_pong,
                        gate_bs_pong,
                        up_bs_pong,
                        acc_gate,
                        acc_up,
                    )

                    # Half 2: read pong (DMA'd Half 1), DMA->ping, MFMA(ping)
                    (
                        a_tile_pong,
                        gate_w_pong,
                        up_w_pong,
                        a_scale_pong,
                        gate_bs_pong,
                        up_bs_pong,
                        acc_gate,
                        acc_up,
                    ) = _interleaved_half(
                        lds_x_pong,
                        lds_x_ping,
                        next_k_dma_2,
                        next_k_load_2,
                        a_tile_ping,
                        gate_w_ping,
                        up_w_ping,
                        a_scale_ping,
                        gate_bs_ping,
                        up_bs_ping,
                        acc_gate,
                        acc_up,
                    )

            # _wave_mod2_b = wave_id % 2
            # _wave_odd = arith.cmpi(
            #     CmpIPredicate.eq, _wave_mod2_b, fx.Index(1)
            # )
            # _if_wave_odd = scf.IfOp(_wave_odd)
            # with ir.InsertionPoint(_if_wave_odd.then_block):
            #     # gpu.barrier()
            #     _barrier()
            #     scf.YieldOp([])

            if const_expr(odd_k_tiles):
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
                    ku_count=_tail_ku if _pad_ku_skip > 0 else k_unroll,
                )
            else:
                _k_tail_rel = fx.Index(_k_dim - tile_k)
                k_tail1 = k_base_idx + _k_tail_rel
                x_regs_ping = []
                if const_expr(use_async_copy):
                    prefetch_x_to_lds(k_tail1, lds_x_ping)
                else:
                    x_regs_ping = load_x_tile(k_tail1)
                if const_expr(_pad_ku_skip > 0):
                    gate_w_ping, up_w_ping = load_b_tile(
                        k_tail1 // 2,
                        ku_limit=_tail_ku,
                    )
                    a_scale_ping, gate_bs_ping, up_bs_ping = prefetch_ab_scale_tile(
                        k_tail1 // (pack_K * 128),
                        ku_packed_limit=_tail_ku_packed,
                    )
                else:
                    gate_w_ping, up_w_ping = load_b_tile(k_tail1 // 2)
                    a_scale_ping, gate_bs_ping, up_bs_ping = prefetch_ab_scale_tile(
                        k_tail1 // (pack_K * 128)
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
                if const_expr(not use_async_copy):
                    store_x_tile_to_lds(x_regs_ping, lds_x_ping)
                rocdl.s_waitcnt(0)
                _barrier()
                if const_expr(_pad_ku_skip > 0):
                    a_tile_ping = prefetch_full_a_from_lds(
                        lds_x_ping, ku_limit=_tail_ku
                    )
                else:
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
                    ku_count=_tail_ku if _pad_ku_skip > 0 else k_unroll,
                )

            bias_pf = None
            if const_expr(epilogue_pf is not None):
                _, _, bias_pf = epilogue_pf

            def _act_vec4(gate_v4, up_v4):
                """Dispatch activation based on `act` parameter."""
                return moe_act_vec4(gate_v4, up_v4, act=act, swiglu_limit=swiglu_limit)

            # Add bias to raw GEMM accumulators before activation.
            # bias layout: [E, 2*inter_dim] flat f32 (non-interleaved: gate then up).
            # For gate_up_interleave, map physical column to logical bias offset.
            if const_expr(enable_bias and not _is_splitk):
                _bias_up_vals = None
                if const_expr(bias_pf is not None):
                    if const_expr(gate_up_interleave):
                        _bias_gate_vals = bias_pf
                    else:
                        _bias_gate_vals, _bias_up_vals = bias_pf
                else:
                    _bias_gate_vals = []
                    for _ni in range_constexpr(num_acc_n):
                        if const_expr(gate_up_interleave):
                            _logical_col = (
                                (by_n + n_tile_base) // 2
                                + fx.Index((_ni // 2) * 16)
                                + lane_mod_16
                            )
                            _up_off = inter_idx if (_ni % 2 == 1) else fx.Index(0)
                            _bias_off = expert_off_idx + _up_off + _logical_col
                        else:
                            _bn = by_n + n_tile_base + (_ni * 16) + lane_mod_16
                            _bias_off = expert_off_idx + _bn
                        _bias_gate_vals.append(_load_bias_scalar(bias_rsrc, _bias_off))
                    if const_expr(not (mock_gate_only or gate_up_interleave)):
                        _bias_up_vals = []
                        for _ni in range_constexpr(num_acc_n):
                            _bn = by_n + n_tile_base + (_ni * 16) + lane_mod_16
                            _bias_up_vals.append(
                                _load_bias_scalar(
                                    bias_rsrc, expert_off_idx + inter_idx + _bn
                                )
                            )
                for _mi in range_constexpr(m_repeat):
                    for _ni in range_constexpr(num_acc_n):
                        _aidx = _mi * num_acc_n + _ni
                        _bsplat = Vec.from_elements(
                            [_bias_gate_vals[_ni]] * 4, fx.Float32
                        )
                        acc_gate[_aidx] = acc_gate[_aidx] + _bsplat

                if const_expr(not (mock_gate_only or gate_up_interleave)):
                    for _mi in range_constexpr(m_repeat):
                        for _ni in range_constexpr(num_acc_n):
                            _aidx = _mi * num_acc_n + _ni
                            _bsplat = Vec.from_elements(
                                [_bias_up_vals[_ni]] * 4, fx.Float32
                            )
                            acc_up[_aidx] = acc_up[_aidx] + _bsplat

            if const_expr(gate_up_interleave and not _is_splitk):
                _gui_out_n = num_acc_n // pack_N
                acc = [None] * (_gui_out_n * m_repeat)
                for _mi in range_constexpr(m_repeat):
                    for _ni in range_constexpr(_gui_out_n):
                        _g_idx = _mi * num_acc_n + _ni * pack_N
                        _u_idx = _g_idx + 1
                        _out_idx = _mi * _gui_out_n + _ni
                        acc[_out_idx] = _act_vec4(acc_gate[_g_idx], acc_gate[_u_idx])
            elif const_expr(not _is_splitk):
                acc = [None] * (int(num_acc_n) * int(m_repeat))
                for _mi in range_constexpr(m_repeat):
                    for _ni in range_constexpr(num_acc_n):
                        _aidx = _mi * num_acc_n + _ni
                        acc[_aidx] = _act_vec4(acc_gate[_aidx], acc_up[_aidx])

            # ---- Epilogue: CShuffle + direct store (accumulate=False) ----
            # Output: out[(t*topk+s) * inter_dim + col] = silu(gate) * up
            # For split-K: skip silu, output gate/up separately with atomic add
            tw_pf = None
            bias_pf = None
            if const_expr(epilogue_pf is not None):
                _, tw_pf, bias_pf = epilogue_pf

            mask24_i32 = fx.Int32(0xFFFFFF)
            topk_i32_v = topk_i32
            tokens_i32_v = tokens_i32

            out_base_i64 = fx.Int64(fx.ptrtoint(arg_out)).ir_value()
            out_base_idx = fx.Index(out_base_i64)

            if const_expr(lds_out is None):
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
                if const_expr(_apply_weight):
                    tw_idx = (mi * 4) + ii
                    if const_expr(tw_pf is not None):
                        tw = tw_pf[tw_idx]
                    else:
                        tw = buffer_ops.buffer_load(
                            sorted_w_rsrc, row, vec_width=1, dtype=f32
                        )
                for ni in range_constexpr(num_acc_n):
                    col_local = col_base_local + (ni * 16)
                    acc_idx = mi * num_acc_n + ni
                    v = Vec(acc[acc_idx])[ii].ir_value()
                    if const_expr(_apply_weight):
                        v = v * tw
                    if const_expr(_need_quant):
                        lds_idx = row_base_lds + col_local
                        vec1_f32 = T.vec(1, f32)
                        v1 = Vec.from_elements([v], fx.Float32)
                        Vec(v1).store(lds_out, [lds_idx], alignment=4)
                    else:
                        v_out = v.truncf(out_elem())
                        lds_idx = row_base_lds + col_local
                        vec1_out = T.vec(1, out_elem())
                        v1 = Vec.from_elements([v_out])
                        Vec(v1).store(lds_out, [lds_idx], alignment=2)

            _out_row_stride = (
                inter_dim * 2 * out_elem_bytes
                if _is_splitk
                else (
                    inter_dim // 2
                    if _need_fp4
                    else (inter_dim if _need_fp8 else inter_dim * out_elem_bytes)
                )
            )

            def precompute_row(*, row_local, row):
                # keep raw: fx.memref_load expects fly-level memref types,
                # not this raw LDS memref
                fused2 = memref.load(lds_tid, [row_local])
                row_i32 = fx.Int32(row).ir_value()
                row_valid0 = fx.Uint32(row_i32) < num_valid_i32
                t = fused2 & mask24_i32
                s = fused2 >> 24
                t_ok = fx.Uint32(t) < tokens_i32_v
                s_ok = fx.Uint32(s) < topk_i32_v
                row_valid = (row_valid0 & (t_ok & s_ok)).ir_value()
                t_idx = fx.Index(t)
                s_idx = fx.Index(s)
                ts_idx = t_idx * topk + s_idx
                row_byte_base = out_base_idx + ts_idx * _out_row_stride
                return ((fused2, row_byte_base), row_valid)

            _e_vec = _e_vec_s1
            _e_vec_sk = 2
            _cshuffle_nlane = min(32, tile_n // _e_vec)
            _cshuffle_nlane_sk = min(32, tile_n // _e_vec_sk)
            _num_threads_per_quant_blk = _num_threads_per_quant_blk_s1

            _c0_f32 = fx.Float32(0.0).ir_value()

            _fp_headroom = 2 if _need_fp4 else (8 if _need_fp8 else 0)
            _c_headroom_i32 = fx.Int32(_fp_headroom)

            if const_expr(_need_sort):
                (
                    _c0_i32,
                    _c1_i32,
                    _c2_i32,
                    _c3_i32,
                    _c4_i32,
                    _c5_i32,
                    _c15_i32,
                    _c31_i32,
                    _c32_i32,
                    _c64_i32,
                    _c256_i32,
                ) = (
                    arith.constant(v, type=T.i32)
                    for v in (0, 1, 2, 3, 4, 5, 15, 31, 32, 64, 256)
                )
                _n32_sort = _sorted_scale_cols_i32 * _c32_i32

            # Mutable slot for split-K N-offset (gate=0, up=inter_dim)
            _sk_n_offset = [0]

            def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                fused, row_byte_base = row_ctx
                if const_expr(_need_quant and not _is_splitk):
                    frag_vals = []
                    for i in range_constexpr(_e_vec):
                        frag_vals.append(Vec(frag)[i])

                    local_max = _c0_f32
                    for i in range_constexpr(_e_vec):
                        abs_v = llvm.call_intrinsic(
                            f32, "llvm.fabs.f32", [raw_value(frag_vals[i])], [], []
                        )
                        local_max = local_max.maximumf(abs_v)

                    for _si in range_constexpr(_num_shuffle_steps_s1):
                        off = fx.Int32(_shuffle_dists_s1[_si])
                        peer = local_max.shuffle_xor(off, 64)
                        local_max = local_max.maximumf(peer)

                    max_i32 = local_max.bitcast(T.i32)
                    # Match fp4_utils.f32_to_e8m0(max_abs / 4): round the
                    # exponent at the 1.5x threshold before dropping mantissa.
                    max_rounded = (max_i32 + 0x400000) & fx.Int32(0xFF800000)
                    exp_field = max_rounded >> 23
                    # keep raw: no ArithValue/fx signed-integer-max equivalent
                    e8m0_biased = arith.maxsi(exp_field - _c_headroom_i32, _c0_i32)

                    quant_exp = 254 - e8m0_biased
                    quant_scale = arith.bitcast(T.f32, raw_value(quant_exp << 23))

                    if const_expr(_need_fp4):
                        fp4_vals = []
                        for i in range_constexpr(_e_vec):
                            scaled_v = frag_vals[i] * quant_scale
                            fp4_vals.append(f32_to_e2m1(scaled_v))

                        packed_i32 = fp4_vals[0] | (fp4_vals[1] << 4)
                        for k in range_constexpr(1, _e_vec // 2):
                            byte_k = fp4_vals[2 * k] | (fp4_vals[2 * k + 1] << 4)
                            packed_i32 = packed_i32 | (byte_k << (k * 8))

                        ptr_addr_idx = row_byte_base + col_g0 // 2
                        out_ptr_v = idx_to_llvm_ptr(ptr_addr_idx)
                        _pack_bytes = _e_vec // 2
                        if const_expr(_pack_bytes == 1):
                            store_val = packed_i32.trunci(T.i8)
                            store_raw = (
                                store_val._value
                                if hasattr(store_val, "_value")
                                else store_val
                            )
                            llvm.StoreOp(
                                store_raw, out_ptr_v, alignment=1, nontemporal=True
                            )
                        elif const_expr(_pack_bytes == 2):
                            store_val = packed_i32.trunci(T.i16)
                            store_raw = (
                                store_val._value
                                if hasattr(store_val, "_value")
                                else store_val
                            )
                            llvm.StoreOp(
                                store_raw, out_ptr_v, alignment=2, nontemporal=True
                            )
                        else:
                            packed_raw = (
                                packed_i32._value
                                if hasattr(packed_i32, "_value")
                                else packed_i32
                            )
                            llvm.StoreOp(
                                packed_raw, out_ptr_v, alignment=4, nontemporal=True
                            )

                    elif const_expr(_need_fp8):
                        scaled_vals = []
                        for i in range_constexpr(_e_vec):
                            scaled_vals.append(frag_vals[i] * quant_scale)

                        ptr_addr_idx = row_byte_base + col_g0
                        if const_expr(_e_vec <= 4):
                            packed_i32 = 0
                            for _w in range_constexpr(_e_vec // 2):
                                packed_i32 = rocdl.cvt_pk_fp8_f32(
                                    T.i32,
                                    scaled_vals[2 * _w],
                                    scaled_vals[2 * _w + 1],
                                    packed_i32,
                                    _w,
                                )
                            out_ptr_v = idx_to_llvm_ptr(ptr_addr_idx)
                            if const_expr(_e_vec == 2):
                                store_val = packed_i32.trunci(T.i16)
                                store_raw = (
                                    store_val._value
                                    if hasattr(store_val, "_value")
                                    else store_val
                                )
                                llvm.StoreOp(
                                    store_raw,
                                    out_ptr_v,
                                    alignment=2,
                                    nontemporal=True,
                                )
                            else:
                                packed_raw = (
                                    packed_i32._value
                                    if hasattr(packed_i32, "_value")
                                    else packed_i32
                                )
                                llvm.StoreOp(
                                    packed_raw,
                                    out_ptr_v,
                                    alignment=4,
                                    nontemporal=True,
                                )
                        else:
                            for _wg in range_constexpr(_e_vec // 4):
                                _b = _wg * 4
                                packed_w = 0
                                packed_w = rocdl.cvt_pk_fp8_f32(
                                    T.i32,
                                    scaled_vals[_b],
                                    scaled_vals[_b + 1],
                                    packed_w,
                                    0,
                                )
                                packed_w = rocdl.cvt_pk_fp8_f32(
                                    T.i32,
                                    scaled_vals[_b + 2],
                                    scaled_vals[_b + 3],
                                    packed_w,
                                    1,
                                )
                                word_ptr = ptr_addr_idx + (_wg * 4)
                                out_ptr_v = idx_to_llvm_ptr(word_ptr)
                                packed_raw = (
                                    packed_w._value
                                    if hasattr(packed_w, "_value")
                                    else packed_w
                                )
                                llvm.StoreOp(
                                    packed_raw,
                                    out_ptr_v,
                                    alignment=4,
                                    nontemporal=True,
                                )

                    if const_expr(_need_sort):
                        col_g0_i32 = arith.index_cast(T.i32, col_g0)
                        if fx.Int32(col_g0_i32 & _c31_i32) == 0:
                            row_i32_s = arith.index_cast(T.i32, row)
                            col_s_i32 = col_g0_i32 >> _c5_i32
                            d0 = row_i32_s >> _c5_i32
                            d1 = (row_i32_s >> _c4_i32) & _c1_i32
                            d2 = row_i32_s & _c15_i32
                            d3 = col_s_i32 >> _c3_i32
                            d4 = (col_s_i32 >> _c2_i32) & _c1_i32
                            d5 = col_s_i32 & _c3_i32
                            byte_off = (
                                d0 * _n32_sort
                                + d3 * _c256_i32
                                + d5 * _c64_i32
                                + d2 * _c4_i32
                                + d4 * _c2_i32
                                + d1
                            )
                            e8m0_i8 = arith.TruncIOp(T.i8, e8m0_biased)
                            buffer_ops.buffer_store(
                                e8m0_i8,
                                sorted_scale_rsrc,
                                byte_off,
                                offset_is_bytes=True,
                            )
                elif const_expr(_is_splitk):
                    col_idx = col_g0 + fx.Index(_sk_n_offset[0])
                    byte_off_col = col_idx * out_elem_bytes
                    ptr_addr_idx = row_byte_base + byte_off_col
                    out_ptr_v = idx_to_llvm_ptr(ptr_addr_idx)
                    store_or_atomic_add(
                        out_ptr_v,
                        frag,
                        accumulate=True,
                        alignment=_e_vec_sk * out_elem_bytes,
                    )
                else:
                    col_idx = col_g0
                    byte_off_col = col_idx * out_elem_bytes
                    ptr_addr_idx = row_byte_base + byte_off_col
                    out_ptr_v = idx_to_llvm_ptr(ptr_addr_idx)
                    store_or_atomic_add(
                        out_ptr_v,
                        frag,
                        accumulate=False,
                        alignment=_e_vec * out_elem_bytes,
                    )

            _frag_elem = (
                ir.F32Type.get()
                if _need_quant
                else (ir.BF16Type.get() if out_is_bf16 else ir.F16Type.get())
            )

            def _run_cshuffle_epilog(
                *,
                acc_value,
                tile_n_value,
                e_vec_value,
                cshuffle_nlane_value,
                num_acc_n_value,
                by_n_value=by_n,
                n_tile_base_value=n_tile_base,
                lds_out_split_value=lds_out_B,
                sk_n_offset=None,
            ):
                nonlocal acc
                acc = acc_value
                if sk_n_offset is not None:
                    _sk_n_offset[0] = sk_n_offset
                c_shuffle_epilog(
                    tile_m=tile_m,
                    tile_n=tile_n_value,
                    e_vec=e_vec_value,
                    cshuffle_nlane=cshuffle_nlane_value,
                    block_size=total_threads,
                    m_repeat=m_repeat,
                    num_acc_n=num_acc_n_value,
                    tx=tx,
                    lane_div_16=lane_div_16,
                    lane_mod_16=lane_mod_16,
                    bx_m=bx_m,
                    by_n=by_n_value,
                    n_tile_base=n_tile_base_value,
                    lds_out=lds_out,
                    frag_elem_type=_frag_elem,
                    write_row_to_lds=write_row_to_lds,
                    precompute_row=precompute_row,
                    store_pair=store_pair,
                    lds_out_split=lds_out_split_value,
                )

            _gui_out_n_for_epilog = num_acc_n
            if const_expr(gate_up_interleave and not _is_splitk):
                _gui_out_n_for_epilog = _gui_out_n

            _acc_for_epilog = acc_gate
            if const_expr(not _is_splitk and not mock_gate_only):
                _acc_for_epilog = acc

            run_moe_stage1_cshuffle_epilog(
                gate_up_interleave=gate_up_interleave,
                is_splitk=_is_splitk,
                mock_gate_only=mock_gate_only,
                acc=_acc_for_epilog,
                acc_gate=acc_gate,
                acc_up=acc_up,
                gui_out_n=_gui_out_n_for_epilog,
                tile_n=tile_n,
                e_vec=_e_vec,
                e_vec_sk=_e_vec_sk,
                cshuffle_nlane=_cshuffle_nlane,
                cshuffle_nlane_sk=_cshuffle_nlane_sk,
                num_acc_n=num_acc_n,
                by_n=by_n,
                n_tile_base=n_tile_base,
                inter_dim=inter_dim,
                run_cshuffle_epilog=_run_cshuffle_epilog,
            )

        if blk_valid:
            if exp_valid:
                _moe_gemm1_body()

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
        use_async_copy,
        waves_per_eu,
        k_batch,
        gate_mode,
        a_scale_one,
        xcd_swizzle,
    )

    @flyc.jit
    def launch_mixed_moe_gemm1(
        arg_out: fx.Pointer,
        arg_x: fx.Pointer,
        arg_w: fx.Pointer,
        arg_scale_x: fx.Pointer,
        arg_scale_w: fx.Pointer,
        arg_sorted_token_ids: fx.Pointer,
        arg_expert_ids: fx.Pointer,
        arg_sorted_weights: fx.Pointer,
        arg_max_token_ids: fx.Pointer,
        arg_bias: fx.Pointer,
        arg_out_scale_sorted: fx.Pointer,
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

        inter_dim_pad_total = fx.Index(2 * inter_dim_pad)
        tile2_pad = 0
        if const_expr(not gate_only):
            tile_k_stage2 = tile_k // 2
            tile2_pad = (
                tile_k_stage2 - (inter_dim - inter_dim_pad) % tile_k_stage2
            ) % tile_k_stage2

        inter_in = fx.Index(i32_inter_in.ir_value())
        tile_n_index = fx.Index(tile_n)
        if const_expr(mock_gate_only or gate_up_interleave):
            gx = (
                inter_in - inter_dim_pad_total + tile2_pad + tile_n_index - 1
            ) // tile_n_index
        else:
            gx = (
                (inter_in - inter_dim_pad_total + tile2_pad + 2 * tile_n_index - 1)
                // tile_n_index
                // 2
            )

        _c_pm_l = fx.Index(persist_m)
        gy = (fx.Index(i32_size_expert_ids_in.ir_value()) + _c_pm_l - 1) // _c_pm_l

        moe_gemm1(
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
    b_nt: int = 2,
    xcd_swizzle: int = 0,
):
    """Compile stage2 kernel (`moe_gemm2`) and return the compiled executable.

    persist_m:
      - > 0: legacy mode -- each CTA processes exactly persist_m consecutive M tiles.
      - <= 0: **persistent mode** -- grid_y = cu_num (auto-detected), each CTA
        round-robins over M tiles with stride cu_num.

    Supported dtypes (specialized): a8w4 (a_dtype="fp8") and a4w4
    (a_dtype="fp4"); b_dtype is always "fp4".

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
    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")
    _state = {}

    # Specialized to a4w4 (a=fp4) and a8w4 (a=fp8); weights are always fp4.
    if a_dtype not in ("fp4", "fp8"):
        raise ValueError(
            f"mixed MoE supports a_dtype in ('fp4','fp8'), got {a_dtype!r}"
        )
    if b_dtype != "fp4":
        raise ValueError(f"mixed MoE supports b_dtype='fp4' only, got {b_dtype!r}")

    is_f8_a = a_dtype == "fp8"
    is_f4_a = a_dtype == "fp4"
    is_f4_b = True

    _scale_pack_m = 2  # physical mn_pack in preshuffle microscale layout
    _scale_pack_n = 2
    _scale_pack_k = 2  # physical k_pack in preshuffle scale layout
    pack_M = min(_scale_pack_m, tile_m // 16)
    pack_N = min(_scale_pack_n, tile_n // 64)
    _k_unroll_raw = int(tile_k) // 128
    pack_K = min(_scale_pack_k, _k_unroll_raw)

    elem_bytes = 1

    a_elem_bytes = 1
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
    _b_kpack_bytes_s = 16
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
    w_elem_bytes = 1
    w_elem_pack = 2
    w_nbytes = (experts * model_dim * inter_dim * w_elem_bytes) // w_elem_pack
    bias_nbytes = experts * model_dim * 4

    def _w_elem_type():
        return T.i8

    total_threads = 256
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(a_elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={a_elem_bytes}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads

    lds_stride = tile_k

    if a_elem_vec_pack > 1:
        _eff_lds_stride = lds_stride // a_elem_vec_pack
        _eff_tile_k_bytes = tile_k_bytes // a_elem_vec_pack
    else:
        _eff_lds_stride = lds_stride
        _eff_tile_k_bytes = tile_k_bytes

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
        _use_cshuffle_epilog = (
            True if use_cshuffle_epilog is None else bool(use_cshuffle_epilog)
        )
        if not _use_cshuffle_epilog:
            raise ValueError(
                "stage2 non-f32 output currently requires CShuffle epilogue."
            )

    # NOTE: Keep this as a callable so we don't require an MLIR Context at Python-time.
    def out_elem():
        return T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)

    def _load_bias_scalar(bias_rsrc, offset):
        return buffer_ops.buffer_load(bias_rsrc, offset, vec_width=1, dtype=T.f32)

    epilog_tag = "cshuffle"
    # IMPORTANT: include tiling in the module name to avoid accidentally reusing a compiled
    # binary for a different (tile_m, tile_n, tile_k) configuration.
    # See stage1 note: include ABI tag to prevent binary reuse across signature changes.
    # IMPORTANT: module name participates in the compiler cache key.
    # Dynamic-shape variant: safe to reuse across (tokens/sorted_size/size_expert_ids) at runtime.
    # Keep a distinct ABI tag so the compile cache never mixes with historical signatures.
    _persistent = persist_m <= 0
    if _persistent:
        from aiter.jit.utils.chip_info import get_cu_num

        _cu_num = get_cu_num()
    else:
        _cu_num = 0
    _sbm_tag = "" if _sort_block_m == tile_m else f"_sbm{_sort_block_m}"
    _pm_tag = f"_persist_cu{_cu_num}" if _persistent else f"_pm{persist_m}"
    _xcd_tag = f"_xcd{xcd_swizzle}" if xcd_swizzle > 0 else ""
    module_name = (
        f"mfma_moe2_a{a_dtype}_w{b_dtype}_{out_s}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_vscale_fix3{_pm_tag}{_sbm_tag}{_xcd_tag}"
    ).replace("-", "_")
    # -- LDS sizing (pure Python; no MLIR Context needed) ---------------------
    # Ping-pong A2 tiles via separate allocators (like stage1).
    _single_x_bytes = int(tile_m) * int(_eff_lds_stride) * int(a_elem_bytes)
    _cshuffle_elem_bytes_s2 = 2  # f16/bf16 = 2 bytes
    lds_out_bytes = (
        _cshuffle_elem_bytes_s2 * int(tile_m) * int(tile_n)
        if _use_cshuffle_epilog
        else 0
    )
    lds_tid_bytes = int(tile_m) * 4
    _input_elems = _single_x_bytes if a_elem_bytes == 1 else (_single_x_bytes // 2)

    _pong_buffer_bytes = max(_single_x_bytes, lds_out_bytes)
    _ping_buffer_bytes = _single_x_bytes

    def x_lds_elem():
        return T.f8

    lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = lds_pong_offset + _pong_buffer_bytes
    _lds_tid_offset_pong = allocator_pong._align(allocator_pong.ptr, 4)
    allocator_pong.ptr = _lds_tid_offset_pong + lds_tid_bytes

    lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = lds_ping_offset + _ping_buffer_bytes

    @flyc.kernel(name=module_name)
    def moe_gemm2(
        arg_out: fx.Pointer,
        arg_x: fx.Pointer,
        arg_w: fx.Pointer,
        arg_scale_x: fx.Pointer,
        arg_scale_w: fx.Pointer,
        arg_sorted_token_ids: fx.Pointer,
        arg_expert_ids: fx.Pointer,
        arg_sorted_weights: fx.Pointer,
        arg_num_valid_ids: fx.Pointer,
        arg_bias: fx.Pointer,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
    ):

        tokens_in = fx.Index(i32_tokens_in.ir_value()).ir_value()
        n_in = fx.Index(i32_n_in.ir_value())
        k_in = fx.Index(i32_k_in.ir_value())
        size_expert_ids_in = fx.Index(i32_size_expert_ids_in.ir_value())
        x_elem = T.f8
        f32 = T.f32
        i32 = T.i32
        i64 = T.i64
        vec4_f32 = T.vec(4, f32)
        vec4_i32 = T.vec(4, i32)
        vec16_elems = 16 if a_elem_bytes == 1 else 8
        vec8_elems = 8 if a_elem_bytes == 1 else 4
        vec4_elems = 4 if a_elem_bytes == 1 else 2
        vec16_x = T.vec(vec16_elems, x_elem)
        vec2_i64 = T.vec(2, i64)

        def _ptr_buffer_resource(ptr, num_records_bytes):
            addr = fx.ptrtoint(ptr)
            addr_i64 = fx.Int64(addr)
            return buffer_ops.create_buffer_resource_from_addr(
                addr_i64, num_records_bytes=num_records_bytes
            )

        acc_init = arith.constant_vector(0.0, vec4_f32)

        # A2 layout (flatten token-slot -> M; use i32 for fly.make_shape).
        topk_idx = fx.Index(topk)
        m_in = tokens_in * topk_idx

        # B preshuffle layout: [experts*model_dim, inter_dim]
        c_n_total = fx.Index(experts * model_dim)
        kpack_bytes = 16
        from .layout_utils import _div_pow2, _mod_pow2

        # A&B's scale preshuffle layout
        # For fp4, k_in is already packed (inter_dim // a_elem_vec_pack), so we need original inter_dim
        c_k_orig = fx.Index(inter_dim)
        layout_a_scale = make_preshuffle_scale_layout(c_mn=m_in, c_k=c_k_orig)
        layout_b_scale = make_preshuffle_scale_layout(c_mn=c_n_total, c_k=c_k_orig)

        shape_lds = fx.make_shape(tile_m, _eff_lds_stride)
        stride_lds = fx.make_stride(_eff_lds_stride, 1)
        layout_lds = fx.make_layout(shape_lds, stride_lds)

        tx = gpu.thread_id("x")
        by = gpu.block_id("x")  # tile along model_dim (N-dim)
        bx_persist = gpu.block_id("y")  # persistent WG index (M-dim)

        if const_expr(xcd_swizzle > 0):
            _NUM_XCDS_S = 8
            _c1_sw = fx.Index(1)
            _c_tn_sw = fx.Index(tile_n)
            _c_mdp_sw = fx.Index(model_dim_pad)
            _gx = (n_in - _c_mdp_sw + _c_tn_sw - _c1_sw) // _c_tn_sw
            if const_expr(_persistent):
                _gy = fx.Index(_cu_num)
            else:
                _c_pm_sw = fx.Index(persist_m)
                _gy = (size_expert_ids_in + _c_pm_sw - _c1_sw) // _c_pm_sw

            bx_persist, by = xcd_remap_bx_by_grid(
                bx_persist, by, _gx, _gy, xcd_swizzle=xcd_swizzle
            )

        # XOR16 swizzle parameter (in bytes; constant, power-of-two in our configs).
        k_blocks16 = fx.Index(_eff_tile_k_bytes // 16)
        layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

        base_ptr_pong = allocator_pong.get_base()
        base_ptr_ping = allocator_ping.get_base()
        lds_x_pong = SmemPtr(
            base_ptr_pong, lds_pong_offset, x_lds_elem(), shape=(_input_elems,)
        ).get()
        lds_x_ping = SmemPtr(
            base_ptr_ping, lds_ping_offset, x_lds_elem(), shape=(_input_elems,)
        ).get()
        lds_out = (
            SmemPtr(
                base_ptr_pong,
                lds_pong_offset,
                (T.bf16 if out_is_bf16 else T.f16),
                shape=(tile_m * tile_n,),
            ).get()
            if _use_cshuffle_epilog
            else None
        )
        lds_tid = SmemPtr(
            base_ptr_pong, _lds_tid_offset_pong, T.i32, shape=(tile_m,)
        ).get()

        # Buffer resources.
        # For dynamic memrefs, `max_size=False` cannot infer the logical size from the memref *type*,
        # so we should pass `num_records_bytes` explicitly for stable hardware OOB behavior.
        c_topk = fx.Index(topk)

        # X(A2): buffer size in bytes, accounting for FP4 packing (2 elements per byte).
        # fp8/int8: 1 byte per element  -> bytes = tokens*topk * K
        # fp4:      2 elements per byte -> bytes = tokens*topk * K / 2
        c_elem_bytes = fx.Index(int(a_elem_bytes))
        x_nbytes_idx = _div_pow2(
            (tokens_in * c_topk) * k_in * c_elem_bytes, int(a_elem_vec_pack)
        )
        x_nbytes_i32 = fx.Int32(x_nbytes_idx)
        x_rsrc = _ptr_buffer_resource(arg_x, x_nbytes_i32)

        w_rsrc = _ptr_buffer_resource(arg_w, w_nbytes)

        # OUT: [tokens, model_dim] -> clamp to descriptor max (i32 bytes) to avoid overflow on huge tokens.
        out_elem_bytes = 4 if out_is_f32 else 2
        out_nbytes_idx = tokens_in * n_in * out_elem_bytes
        if const_expr(not bool(accumulate)):
            out_nbytes_idx = tokens_in * topk * n_in * out_elem_bytes
        out_nbytes_i32 = fx.Int32(out_nbytes_idx)
        out_rsrc = _ptr_buffer_resource(arg_out, out_nbytes_i32)

        # num_valid_ids (sorted padded MN) for scale sizing / guards.
        numids_rsrc = _ptr_buffer_resource(arg_num_valid_ids, fx.Int32(4))
        num_valid_i32 = buffer_ops.buffer_load(
            numids_rsrc, fx.Index(0), vec_width=1, dtype=T.i32
        )
        # num_valid_ids is a scalar (same value for all lanes) loaded into
        # VGPR.  Promote to SGPR so downstream buffer resource descriptors
        # that use it for num_records stay in SGPRs, eliminating the
        # expensive waterfall loop the compiler would otherwise emit.
        num_valid_i32 = rocdl.ReadfirstlaneOp(T.i32, num_valid_i32).res
        num_valid_idx = fx.Index(num_valid_i32)

        sx_rsrc = 1
        sw_rsrc = 1
        # A2 microscale: e8m0 in sorted layout [sorted_size, K/32].
        # Caller must pre-scatter a2_scale via moe_mxfp4_sort.
        kblk = _div_pow2(k_in, 32)
        sx_nbytes_idx = num_valid_idx * kblk
        sx_nbytes_i32 = fx.Int32(sx_nbytes_idx)
        sx_rsrc = _ptr_buffer_resource(arg_scale_x, sx_nbytes_i32)

        # Weight microscale buffer (packed i32 holding e8m0 bytes).
        # Use an exact descriptor size so hardware OOB checking works.
        kblk_w = _div_pow2(k_in, 32)  # K/32
        mn_w = fx.Index(experts * model_dim)
        sw_nbytes_idx = mn_w * kblk_w  # bytes (e8m0)
        sw_nbytes_i32 = fx.Int32(sw_nbytes_idx)
        sw_rsrc = _ptr_buffer_resource(arg_scale_w, sw_nbytes_i32)

        # sorted_token_ids / sorted_weights: [blocks*tile_m] (padded length)
        sorted_nbytes_idx = size_expert_ids_in * tile_m * 4
        sorted_nbytes_i32 = fx.Int32(sorted_nbytes_idx)
        sorted_rsrc = _ptr_buffer_resource(arg_sorted_token_ids, sorted_nbytes_i32)
        sorted_w_rsrc = _ptr_buffer_resource(arg_sorted_weights, sorted_nbytes_i32)

        # expert ids: [sort_blocks] i32.
        _c_sbm = fx.Index(_sort_block_m)
        _c_tm = fx.Index(tile_m)
        _c1 = fx.Index(1)
        _sort_blocks_ub = _div_pow2(
            size_expert_ids_in * _c_tm + _c_sbm - _c1, _sort_block_m
        )
        eid_nbytes_idx = _sort_blocks_ub * 4
        eid_nbytes_i32 = fx.Int32(eid_nbytes_idx)
        expert_rsrc = _ptr_buffer_resource(arg_expert_ids, eid_nbytes_i32)
        bias_rsrc = _ptr_buffer_resource(arg_bias, bias_nbytes) if enable_bias else None

        # ---- persist loop ----
        _c0_p = fx.Index(0).ir_value()
        _c1_p = fx.Index(1).ir_value()

        if const_expr(_persistent):
            # Expert-phase scheduling: contiguous M-tile dispatch.
            # grid_y = cu_num, each CTA handles a contiguous chunk of M-tiles:
            #   [bx_persist * tiles_per_block, ..., (bx_persist+1) * tiles_per_block - 1]
            # Adjacent blocks process adjacent M-tiles -> same expert -> B weight L2 reuse.
            _c_cu = fx.Index(_cu_num)
            _c_tm_p = fx.Index(tile_m)
            _num_valid_idx = fx.Index(num_valid_i32)
            _total_m_tiles = (_num_valid_idx + _c_tm_p - _c1_p) // _c_tm_p
            _tiles_per_block = (_total_m_tiles + _c_cu - _c1_p) // _c_cu
            _init_active = fx.Boolean(1).ir_value()
            _for_persist = scf.ForOp(_c0_p, _tiles_per_block, _c1_p, [_init_active])
        else:
            # Legacy mode: fixed persist_m consecutive tiles.
            _c_pm = fx.Index(persist_m).ir_value()
            _init_prev_expert = fx.Int32(0).ir_value()
            _init_prev_b_base = fx.Index(0).ir_value()
            _for_persist = scf.ForOp(
                _c0_p,
                _c_pm,
                _c1_p,
                [_init_prev_expert, _init_prev_b_base],
            )

        _for_ip = ir.InsertionPoint(_for_persist.body)
        _for_ip.__enter__()
        _mi_p = _for_persist.induction_variable

        if const_expr(_persistent):
            _still_active = _for_persist.inner_iter_args[0]
            bx = bx_persist * _tiles_per_block + _mi_p
        else:
            _prev_expert_i32 = _for_persist.inner_iter_args[0]
            _prev_expert_b_base = _for_persist.inner_iter_args[1]
            bx = bx_persist * persist_m + _mi_p

        bx_m = bx * tile_m

        # Early-exit guard: skip garbage expert blocks beyond `num_valid_ids`.
        bx_m_i32 = fx.Int32(bx_m).ir_value()
        blk_valid = fx.Uint32(bx_m_i32) < num_valid_i32

        sort_blk = _div_pow2(bx_m, _sort_block_m)
        expert_i32 = buffer_ops.buffer_load(
            expert_rsrc, sort_blk, vec_width=1, dtype=T.i32
        )
        expert_idx = fx.Index(expert_i32)
        exp_valid = fx.Uint32(expert_i32) < experts

        if const_expr(_persistent):
            # Absolute B-base: no cross-iteration state needed.
            _expert_b_base = expert_idx * _expert_b_stride
        else:
            # Legacy incremental B-base: delta = (cur - prev) * stride
            _delta_expert = expert_i32 - _prev_expert_i32
            _delta_expert_idx = fx.Index(_delta_expert)
            _delta_b = _delta_expert_idx * _expert_b_stride
            _expert_b_base = _prev_expert_b_base + _delta_b

        # Early-exit: if the first row of this tile is a sentinel (all-padding tile),
        # skip the entire GEMM.
        _first_tok = buffer_ops.buffer_load(sorted_rsrc, bx_m, vec_width=1, dtype=T.i32)
        _first_tid = _first_tok & fx.Int32(0xFFFFFF)
        _tokens_i32_guard = fx.Int32(tokens_in)
        tile_has_tokens = fx.Uint32(_first_tid) < _tokens_i32_guard

        # For tile_m < 32 (pack_M < _scale_pack_m): shift a_scale i32 so the
        # correct bytes land at the op_sel positions we use.
        if const_expr(pack_M < _scale_pack_m):
            _m_off = _mod_pow2(_div_pow2(bx_m, 16), _scale_pack_m)
            _m_scale_shift_i32 = fx.Int32(_m_off * 8)
        else:
            _m_scale_shift_i32 = None

        def _moe_gemm2_then_body():
            # Expert id for this M tile.
            n_idx = fx.Index(model_dim)
            expert_off_idx = expert_idx * n_idx  # index

            # ---- X gmem->reg prefetch (match preshuffle GEMM mapping) ----
            # Prefer 16B buffer-load (dwordx4). If the per-thread byte count isn't divisible by
            # 16, fall back to 8B (dwordx2) or 4B (dword) loads. For fp16 we require 16B.
            if const_expr(bytes_per_thread_x % 16 == 0):
                x_load_bytes = 16
            elif const_expr(bytes_per_thread_x % 8 == 0):
                x_load_bytes = 8
            elif const_expr(bytes_per_thread_x % 4 == 0):
                x_load_bytes = 4
            else:
                raise ValueError(
                    f"bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 4 to use the dword-indexed load mapping."
                )
            num_x_loads = bytes_per_thread_x // x_load_bytes
            chunk_i32 = x_load_bytes // 4  # dwords per chunk (1/2/4)

            c_k_div4 = _div_pow2(
                _div_pow2(k_in, int(a_elem_vec_pack)) * fx.Index(int(a_elem_bytes)),
                4,
            )
            tile_k_dwords = (int(tile_k) * int(a_elem_bytes)) // (
                4 * int(a_elem_vec_pack)
            )
            layout_x_tile_div4 = fx.make_layout(
                (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
            )
            c_chunk_i32 = fx.Index(chunk_i32)
            tx_i32_base = tx * c_chunk_i32

            topk_i32 = fx.Int32(topk)
            mask24 = fx.Int32(0xFFFFFF)
            # Sentinel clamp uses `tokens` as the upper bound: t_valid = (t < tokens).
            tokens_i32 = fx.Int32(tokens_in)

            def x_tile_chunk_coord_i32(i: int):
                return tile_chunk_coord_i32(
                    arith,
                    tx_i32_base=tx_i32_base,
                    i=i,
                    total_threads=total_threads,
                    layout_tile_div4=layout_x_tile_div4,
                    chunk_i32=chunk_i32,
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
                s_i32 = (fx.Uint32(fused_i) >> 24).ir_value()

                t_valid = fx.Uint32(t_i32) < tokens_i32
                s_valid = fx.Uint32(s_i32) < topk_i32
                ts_valid = t_valid & s_valid
                t_safe = ts_valid.select(t_i32, 0)
                s_safe = ts_valid.select(s_i32, 0)
                row_ts_i32 = t_safe * topk_i32 + s_safe
                row_ts_idx = fx.Index(row_ts_i32)

                x_row_base_div4.append(row_ts_idx * c_k_div4)

            # tx -> wave/lane (GEMM-style decomposition).
            coord_wl = idx2crd(tx, layout_tx_wave_lane)
            wave_id = layout_get(coord_wl, 0)
            lane_id = layout_get(coord_wl, 1)
            coord_l16 = idx2crd(lane_id, layout_lane16)
            lane_div_16 = layout_get(coord_l16, 0)
            lane_mod_16 = layout_get(coord_l16, 1)

            row_a_lds = lane_mod_16

            col_offset_base = lane_div_16 * 16

            # Dynamic N tiling within block.
            num_waves = 4
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16
            c_n_per_wave = fx.Index(n_per_wave)
            wave_mod_4 = _mod_pow2(wave_id, 4)
            n_tile_base = wave_mod_4 * c_n_per_wave

            by_n = by * tile_n

            if const_expr(pack_N < _scale_pack_n):
                _global_n_base = expert_off_idx + by_n + n_tile_base
                _n_off = _mod_pow2(_div_pow2(_global_n_base, 16), _scale_pack_n)
                _n_scale_shift_i32 = fx.Int32(_n_off * 8)
            else:
                _n_scale_shift_i32 = None
            n_intra_list = [None] * num_acc_n
            n_blk_list = [None] * num_acc_n
            col_g_list = [None] * num_acc_n
            for i in range_constexpr(num_acc_n):
                offset = i * 16
                col_g = by_n + n_tile_base
                col_g = _div_pow2(col_g, 2) + offset
                col_g = col_g + lane_mod_16
                col_g_list[i] = col_g
                c_offset = fx.Index(offset)
                global_n = by_n + n_tile_base + c_offset + lane_mod_16
                n_blk_list[i] = _div_pow2(global_n, 16)
                n_intra_list[i] = _mod_pow2(global_n, 16)

            m_repeat = tile_m // 16
            k_unroll = tile_k_bytes // 128  # K64-byte micro-step (2x MFMA)

            # fp4 pack
            k_unroll_packed = k_unroll // pack_K
            m_repeat_packed = m_repeat // pack_M
            num_acc_n_packed = num_acc_n // pack_N

            _K_per_ku_s2 = tile_k // k_unroll
            _pad_k_elems_s2 = (inter_dim_pad % tile_k) if inter_dim_pad > 0 else 0
            _pad_ku_skip_s2 = _pad_k_elems_s2 // _K_per_ku_s2
            _tail_ku_s2 = k_unroll - _pad_ku_skip_s2
            _tail_ku_packed_s2 = (
                (_tail_ku_s2 + pack_K - 1) // pack_K if _pad_ku_skip_s2 > 0 else None
            )

            # --- B Load Logic (K64) - shared layout with preshuffle GEMM ---
            def load_b_packs_k64(base_k, ku: int, ni: int):
                """Load one K64-byte B micro-step: single 16B load, split into 2x i64."""
                base_k_bytes = base_k * fx.Index(int(b_elem_bytes))
                k0_base = _div_pow2(base_k_bytes, 64)
                k0 = k0_base + ku
                k1 = lane_div_16
                # Incremental B addressing: _expert_b_base carries the
                # expert's preshuffle offset (updated via delta each
                # persist_m iteration); local n_blk/n_intra contribute
                # the per-lane within-tile offset.  All strides are
                # compile-time constants -> shift/mul, no Barrett.
                idx_pack = (
                    _expert_b_base
                    + n_blk_list[ni] * _b_stride_n0
                    + k0 * _b_stride_k0
                    + k1 * _b_stride_klane
                    + n_intra_list[ni] * _b_stride_nlane
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
                    cache_modifier=b_nt,
                )
                b_i64x2 = Vec(b16).bitcast(fx.Int64)
                b0 = Vec(b_i64x2)[0]
                b1 = Vec(b_i64x2)[1]
                return b0, b1

            def load_b_tile_range(base_k, ku_begin, ku_end):
                return load_b_tile_range_k64(
                    load_b_packs_k64,
                    base_k=base_k,
                    ku_begin=ku_begin,
                    ku_end=ku_end,
                    num_acc_n=num_acc_n,
                )

            def load_b_tile(base_k, ku_limit=k_unroll):
                return load_b_tile_range(base_k, 0, ku_limit)

            _b_split_enabled = k_unroll >= 2
            _b_split_ku = k_unroll // 2 if _b_split_enabled else k_unroll

            def load_b_tile_lo(base_k):
                """Load first half of B tile (ku < _b_split_ku)."""
                return load_b_tile_range(base_k, 0, _b_split_ku)

            def load_b_tile_hi(base_k):
                """Load second half of B tile (ku >= _b_split_ku)."""
                return load_b_tile_range(base_k, _b_split_ku, k_unroll)

            def load_b_scale_tile(
                base_k, k_shift_bits=0, ku_packed_limit=k_unroll_packed
            ):
                return load_scale_tile(
                    buffer_ops,
                    sw_rsrc,
                    layout_b_scale,
                    base_k=base_k,
                    mni_base=_div_pow2(
                        _div_pow2(
                            expert_off_idx + by_n + n_tile_base,
                            _scale_pack_n,
                        ),
                        16,
                    ),
                    mni_count=num_acc_n_packed,
                    ku_packed_limit=ku_packed_limit,
                    lane_div_16=lane_div_16,
                    lane_mod_16=lane_mod_16,
                    k_shift_bits=k_shift_bits,
                )

            def load_a_scale_tile(
                base_k, k_shift_bits=0, ku_packed_limit=k_unroll_packed
            ):
                return load_scale_tile(
                    buffer_ops,
                    sx_rsrc,
                    layout_a_scale,
                    base_k=base_k,
                    mni_base=_div_pow2(_div_pow2(bx_m, _scale_pack_m), 16),
                    mni_count=m_repeat_packed,
                    ku_packed_limit=ku_packed_limit,
                    lane_div_16=lane_div_16,
                    lane_mod_16=lane_mod_16,
                    k_shift_bits=k_shift_bits,
                )

            def prefetch_ab_scale_tile(
                base_k, k_shift_bits=0, ku_packed_limit=k_unroll_packed
            ):
                return [
                    load_a_scale_tile(
                        base_k, k_shift_bits, ku_packed_limit=ku_packed_limit
                    ),
                    load_b_scale_tile(
                        base_k, k_shift_bits, ku_packed_limit=ku_packed_limit
                    ),
                ]

            vec8_x = T.vec(vec8_elems, x_elem)
            vec4_x_lds = T.vec(vec4_elems, x_elem)

            # --- A LDS load helper for K64 (load 16B once, extract 2x i64 halves) ---
            def lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer):
                col_base_swz_bytes = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
                col_base_swz = (
                    col_base_swz_bytes if elem_bytes == 1 else (col_base_swz_bytes // 2)
                )
                idx_a16 = crd2idx([curr_row_a_lds, col_base_swz], layout_lds)
                loaded_a16 = Vec.load(vec16_x, lds_buffer, [idx_a16])
                a_i64x2 = Vec(loaded_a16).bitcast(fx.Int64)
                a0 = Vec(a_i64x2)[0]
                a1 = Vec(a_i64x2)[1]
                return a0, a1

            def compute_tile(
                acc_in,
                b_tile_in,
                lds_buffer,
                a_scale=None,
                b_scale=None,
                *,
                prefetch_epilogue: bool = False,
                a0_prefetch=None,
                a1_prefetch=None,
                b_hi_loader=None,
                ku_count=k_unroll,
            ):
                if const_expr(b_hi_loader is not None):
                    b_tile_full = [None] * k_unroll
                    for i in range_constexpr(_b_split_ku):
                        b_tile_full[i] = b_tile_in[i]
                else:
                    b_tile_full = b_tile_in
                acc_list = list(acc_in)
                mfma_res_ty = vec4_f32

                epilogue_pf = None
                bias = None
                if const_expr(prefetch_epilogue):
                    if const_expr(enable_bias):
                        bias = []
                        for ni in range_constexpr(num_acc_n):
                            global_n = by_n + n_tile_base + ni * 16 + lane_mod_16
                            bias_offset = expert_off_idx + global_n
                            bias.append(_load_bias_scalar(bias_rsrc, bias_offset))
                    tw_pf = None
                    if const_expr(doweight_stage2):
                        tw_pf = []
                        lane_div_16_mul4_pf = lane_div_16 * 4
                        ii_idx_list_pf = [fx.Index(ii) for ii in range(4)]
                        for mi in range_constexpr(m_repeat):
                            mi_base_pf = fx.Index(mi * 16)
                            for ii in range_constexpr(4):
                                row_off_pf = lane_div_16_mul4_pf + ii_idx_list_pf[ii]
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

                c0_i64 = fx.Int64(0)
                vec4_i64 = T.vec(4, T.i64)
                vec8_i32 = T.vec(8, T.i32)

                def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                    v4 = Vec.from_elements([x0, x1, x2, x3], fx.Int64)
                    return Vec(v4).bitcast(fx.Int32)

                # fp4 path -- single k_idx loop [0, k_unroll).
                # b_hi load is issued at the very start so all k_unroll
                # MFMAs can overlap the VMEM latency.
                _pack_K_shift = (pack_K - 1).bit_length()
                _pack_K_mask = pack_K - 1

                if const_expr(b_hi_loader is not None):
                    _b_hi = b_hi_loader()
                    for _bhi_i in range_constexpr(len(_b_hi)):
                        b_tile_full[_b_split_ku + _bhi_i] = _b_hi[_bhi_i]

                for k_idx in range_constexpr(ku_count):
                    ku128 = k_idx >> _pack_K_shift
                    ikxdl = k_idx & _pack_K_mask

                    b_packs0, b_packs1 = b_tile_full[k_idx]

                    col_base = col_offset_base + (k_idx * 128) // a_elem_vec_pack

                    for mi in range_constexpr(m_repeat_packed):
                        a_scale_i32 = a_scale[ku128 * m_repeat_packed + mi]
                        a_scale_val = Vec(a_scale_i32)[0].ir_value()
                        if const_expr(_m_scale_shift_i32 is not None):
                            a_scale_val = fx.Uint32(a_scale_val) >> _m_scale_shift_i32
                        for ni in range_constexpr(num_acc_n_packed):
                            b_scale_i32 = b_scale[ku128 * num_acc_n_packed + ni]
                            b_scale_val = Vec(b_scale_i32)[0].ir_value()
                            if const_expr(_n_scale_shift_i32 is not None):
                                b_scale_val = (
                                    fx.Uint32(b_scale_val) >> _n_scale_shift_i32
                                )

                            for imxdl in range_constexpr(pack_M):
                                col_base0 = col_base
                                mi_idx = mi * pack_M + imxdl
                                mi_val = fx.Index(mi_idx * 16)
                                curr_row_a_lds = row_a_lds + mi_val

                                if const_expr(
                                    (a0_prefetch is not None)
                                    and (k_idx == 0)
                                    and (mi_idx == 0)
                                ):
                                    a0, a1 = a0_prefetch
                                elif const_expr(
                                    (a1_prefetch is not None)
                                    and (k_idx == 1)
                                    and (mi_idx == 0)
                                ):
                                    a0, a1 = a1_prefetch
                                else:
                                    a0, a1 = lds_load_packs_k64(
                                        curr_row_a_lds, col_base0, lds_buffer
                                    )

                                if const_expr(is_f8_a):
                                    col_base1 = col_base + 64
                                    a2, a3 = lds_load_packs_k64(
                                        curr_row_a_lds, col_base1, lds_buffer
                                    )
                                    a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)
                                else:
                                    a128 = pack_i64x4_to_i32x8(a0, a1, c0_i64, c0_i64)

                                for inxdl in range_constexpr(pack_N):
                                    ni_idx = ni * pack_N + inxdl

                                    b0 = b_packs0[ni_idx]
                                    b1 = b_packs1[ni_idx]
                                    b128 = pack_i64x4_to_i32x8(b0, b1, c0_i64, c0_i64)

                                    acc_idx = mi_idx * num_acc_n + ni_idx
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
            # ---- Async DMA: GMEM -> LDS (bypasses VGPR, like stage1) ----
            _dma_bytes = 16
            _wave_size = 64
            _eff_bytes_per_buffer = (
                int(tile_m) * int(_eff_lds_stride) * int(a_elem_bytes)
            )
            _num_dma_loads = max(
                1, _eff_bytes_per_buffer // (total_threads * _dma_bytes)
            )

            def dma_x_tile_to_lds(base_k, lds_buffer):
                c4_idx = fx.Index(4)
                base_k_div4 = _div_pow2(
                    _div_pow2(base_k, int(a_elem_vec_pack))
                    * fx.Index(int(a_elem_bytes)),
                    4,
                )

                lds_ptr_i64 = None
                for i in range_constexpr(_num_dma_loads):
                    row_local_i = x_row_local[i]
                    col_local_i32_i = x_col_local_i32[i]
                    col_local_sw = swizzle_xor16(
                        row_local_i, col_local_i32_i * c4_idx, k_blocks16
                    )
                    row_k_dw = x_row_base_div4[i] + base_k_div4
                    global_byte_idx = row_k_dw * c4_idx + col_local_sw
                    global_offset = fx.Int32(global_byte_idx)

                    if const_expr(i == 0):
                        lds_addr = memref.extract_aligned_pointer_as_index(
                            lds_buffer
                        ) + wave_id * (_wave_size * _dma_bytes)
                        lds_ptr_i64 = rocdl.readfirstlane(T.i64, fx.Int64(lds_addr))
                    else:
                        lds_ptr_i64 = lds_ptr_i64 + (total_threads * _dma_bytes)

                    lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                    lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64)

                    rocdl.raw_ptr_buffer_load_lds(
                        x_rsrc,
                        lds_ptr,
                        fx.Int32(_dma_bytes),
                        global_offset,
                        fx.Int32(0),
                        fx.Int32(0),
                        fx.Int32(0),
                    )

            def prefetch_x_to_lds(base_k, lds_buffer):
                dma_x_tile_to_lds(base_k, lds_buffer)

            rocdl.sched_barrier(0)

            def hot_loop_scheduler():
                rocdl.sched_barrier(0)

            def _k_shift_bits(k_py):
                if const_expr(pack_K >= _scale_pack_k):
                    return 0
                return ((k_py // 128) % _scale_pack_k) * _scale_pack_m * 8

            def _k_base(k_py):
                return k_py // _scale_pack_k // 128

            # Preload sorted_idx into lds_tid for epilogue precompute_row
            # (N-independent; placed before N-tile loop so it's done once per M-tile.)
            if fx.Index(tx) < fx.Index(tile_m):
                _tid_row = bx_m + tx
                _tid_val = buffer_ops.buffer_load(
                    sorted_rsrc, _tid_row, vec_width=1, dtype=T.i32
                )
                _tid_vec1 = Vec.from_elements([_tid_val], fx.Int32)
                Vec(_tid_vec1).store(lds_tid, [tx])

            gpu.barrier()

            # Prologue -- B-first + async DMA X(0) -> pong.
            k0 = fx.Index(0)
            if const_expr(_b_split_enabled):
                b_cur = load_b_tile_lo(k0)
            else:
                b_cur = load_b_tile(k0)
            a_scale_pong, b_scale_pong = prefetch_ab_scale_tile(
                _k_base(0), _k_shift_bits(0)
            )
            rocdl.sched_barrier(0)
            prefetch_x_to_lds(k0, lds_x_pong)
            rocdl.s_waitcnt(0)
            gpu.barrier()

            acc = [acc_init] * num_acc_n * m_repeat

            # Cross-tile A0+A1 LDS prefetch from pong buffer.
            a0_prefetch_pong = lds_load_packs_k64(
                row_a_lds, col_offset_base, lds_x_pong
            )
            _a1_col_base = col_offset_base + 128 // a_elem_vec_pack
            a1_prefetch_pong = (
                lds_load_packs_k64(row_a_lds, _a1_col_base, lds_x_pong)
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
            if const_expr(k_main2_py < 0):
                k_main2_py = 0

            c2_tile_k = fx.Index(tile_k * 2)
            b_pong = b_cur
            k0_pong_bk = k0

            # Only emit the scf.for when there are actually iterations to run.
            # When k_main2_py == 0 the loop body is empty; emitting an scf.for
            # would create a region whose internal SSA values cannot be used
            # by the post-loop tail code.
            def _make_b_hi_loader(base_k):
                """Create a b_hi_loader callable for a given base_k."""
                return lambda _bk=base_k: load_b_tile_hi(_bk)

            if const_expr(k_main2_py > 0):
                for k_iv_py in range_constexpr(0, k_main2_py, tile_k * 2):
                    rocdl.sched_barrier(0)
                    k_iv = fx.Index(k_iv_py)
                    next_k1 = k_iv + tile_k
                    next_k1_bk = next_k1 // 2
                    # DMA X(next_k1) -> ping (non-blocking, overlaps with compute)
                    prefetch_x_to_lds(next_k1, lds_x_ping)
                    b_ping_lo = (
                        load_b_tile_lo(next_k1_bk)
                        if _b_split_enabled
                        else load_b_tile(next_k1_bk)
                    )
                    a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(
                        _k_base(next_k1), _k_shift_bits(next_k1)
                    )

                    acc, _ = compute_tile(
                        acc,
                        b_pong,
                        lds_x_pong,
                        a_scale_pong,
                        b_scale_pong,
                        a0_prefetch=a0_prefetch_pong,
                        a1_prefetch=a1_prefetch_pong,
                        b_hi_loader=(
                            _make_b_hi_loader(k0_pong_bk) if _b_split_enabled else None
                        ),
                    )
                    hot_loop_scheduler()
                    rocdl.s_waitcnt(0)
                    gpu.barrier()

                    # Cross-tile prefetch for the ping tile we are about to compute.
                    a0_prefetch_ping = lds_load_packs_k64(
                        row_a_lds, col_offset_base, lds_x_ping
                    )
                    a1_prefetch_ping = (
                        lds_load_packs_k64(row_a_lds, _a1_col_base, lds_x_ping)
                        if pack_K >= 2
                        else None
                    )

                    next_k2 = k_iv + c2_tile_k
                    next_k2_py = k_iv_py + tile_k * 2
                    next_k2_bk = next_k2 // 2
                    # DMA X(next_k2) -> pong (non-blocking, overlaps with compute)
                    prefetch_x_to_lds(next_k2, lds_x_pong)
                    b_pong = (
                        load_b_tile_lo(next_k2_bk)
                        if _b_split_enabled
                        else load_b_tile(next_k2_bk)
                    )
                    a_scale_pong, b_scale_pong = prefetch_ab_scale_tile(
                        _k_base(next_k2_py), _k_shift_bits(next_k2_py)
                    )

                    acc, _ = compute_tile(
                        acc,
                        b_ping_lo,
                        lds_x_ping,
                        a_scale_ping,
                        b_scale_ping,
                        a0_prefetch=a0_prefetch_ping,
                        a1_prefetch=a1_prefetch_ping,
                        b_hi_loader=(
                            _make_b_hi_loader(next_k1_bk) if _b_split_enabled else None
                        ),
                    )
                    k0_pong_bk = next_k2_bk
                    hot_loop_scheduler()
                    gpu.barrier()

                    # Cross-tile prefetch for the next pong tile.
                    a0_prefetch_pong = lds_load_packs_k64(
                        row_a_lds, col_offset_base, lds_x_pong
                    )
                    a1_prefetch_pong = (
                        lds_load_packs_k64(row_a_lds, _a1_col_base, lds_x_pong)
                        if pack_K >= 2
                        else None
                    )

            if const_expr(odd_k_tiles):
                # Tail: single remaining tile (already in pong buffer).
                acc, epilogue_pf = compute_tile(
                    acc,
                    b_pong,
                    lds_x_pong,
                    a_scale_pong,
                    b_scale_pong,
                    a0_prefetch=a0_prefetch_pong,
                    a1_prefetch=a1_prefetch_pong,
                    prefetch_epilogue=True,
                    b_hi_loader=(
                        _make_b_hi_loader(k0_pong_bk) if _b_split_enabled else None
                    ),
                    ku_count=_tail_ku_s2 if _pad_ku_skip_s2 > 0 else k_unroll,
                )

            else:
                # Tail: 2 remaining tiles.
                k_tail1 = (k_in + tile_k - 1) // tile_k * tile_k - tile_k
                k_tail1_py = (int(inter_dim) + tile_k - 1) // tile_k * tile_k - tile_k
                k_tail1_bk = k_tail1 // 2
                # DMA tail X -> ping
                prefetch_x_to_lds(k_tail1, lds_x_ping)
                if const_expr(_pad_ku_skip_s2 > 0):
                    b_ping_lo = load_b_tile(k_tail1_bk, ku_limit=_tail_ku_s2)
                    a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(
                        _k_base(k_tail1_py),
                        _k_shift_bits(k_tail1_py),
                        ku_packed_limit=_tail_ku_packed_s2,
                    )
                else:
                    b_ping_lo = (
                        load_b_tile_lo(k_tail1_bk)
                        if _b_split_enabled
                        else load_b_tile(k_tail1_bk)
                    )
                    a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(
                        _k_base(k_tail1_py), _k_shift_bits(k_tail1_py)
                    )

                acc, _ = compute_tile(
                    acc,
                    b_pong,
                    lds_x_pong,
                    a_scale_pong,
                    b_scale_pong,
                    a0_prefetch=a0_prefetch_pong,
                    a1_prefetch=a1_prefetch_pong,
                    b_hi_loader=(
                        _make_b_hi_loader(k0_pong_bk) if _b_split_enabled else None
                    ),
                )

                # hot_loop_scheduler()
                rocdl.s_waitcnt(0)
                gpu.barrier()

                # Epilogue tile with sw prefetch.
                a0_prefetch_ping = lds_load_packs_k64(
                    row_a_lds, col_offset_base, lds_x_ping
                )
                a1_prefetch_ping = (
                    lds_load_packs_k64(row_a_lds, _a1_col_base, lds_x_ping)
                    if pack_K >= 2 and (_pad_ku_skip_s2 == 0 or _tail_ku_s2 >= 2)
                    else None
                )
                acc, epilogue_pf = compute_tile(
                    acc,
                    b_ping_lo,
                    lds_x_ping,
                    a_scale_ping,
                    b_scale_ping,
                    a0_prefetch=a0_prefetch_ping,
                    a1_prefetch=a1_prefetch_ping,
                    prefetch_epilogue=True,
                    b_hi_loader=(
                        None
                        if _pad_ku_skip_s2 > 0
                        else (
                            _make_b_hi_loader(k_tail1_bk) if _b_split_enabled else None
                        )
                    ),
                    ku_count=_tail_ku_s2 if _pad_ku_skip_s2 > 0 else k_unroll,
                )

            # ---------------- Epilogue: LDS CShuffle + atomic half2 (x2) ----------------
            # Reuse the shared helper so GEMM / MoE kernels share the exact same CShuffle skeleton.

            sw_pf = None
            tw_pf = None
            bias_pf = None
            if const_expr(epilogue_pf is not None):
                sw_pf, tw_pf, bias_pf = epilogue_pf

            mask24_i32 = fx.Int32(0xFFFFFF)
            topk_i32_v = topk_i32

            # Weight scales for the N tile (col_g depends on lane/wave/by but not on (t,s)).
            if const_expr(lds_out is None):
                raise RuntimeError(
                    "stage2 CShuffle epilogue requires lds_out to be allocated/aliased."
                )

            # Precompute the output base address (i64 index) for ALL paths.
            # Both accumulate=True (global atomic) and accumulate=False (global store)
            # need 64-bit addressing to avoid i32 offset overflow when
            # tokens * model_dim * elem_bytes > INT32_MAX (~150K tokens for model_dim=7168).
            out_base_i64 = fx.Int64(fx.ptrtoint(arg_out)).ir_value()
            out_base_idx = fx.Index(out_base_i64)

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
                if const_expr(doweight_stage2):
                    tw_idx = (mi * 4) + ii
                    if const_expr(tw_pf is not None):
                        tw = tw_pf[tw_idx]
                    else:
                        tw = buffer_ops.buffer_load(
                            sorted_w_rsrc, row, vec_width=1, dtype=f32
                        )

                for ni in range_constexpr(num_acc_n):
                    col_local = col_base_local + (ni * 16)
                    acc_idx = mi * num_acc_n + ni
                    v = Vec(acc[acc_idx])[ii].ir_value()
                    if const_expr(enable_bias):
                        v = v + bias_pf[ni]

                    if const_expr(doweight_stage2):
                        v = v * tw
                    v_out = v.truncf(out_elem())

                    lds_idx = row_base_lds + col_local
                    vec1_out = T.vec(1, out_elem())
                    v1 = Vec.from_elements([v_out])

                    Vec(v1).store(lds_out, [lds_idx], alignment=2)

            def precompute_row(*, row_local, row):
                # Use lds_tid (sorted_idx preloaded to LDS) instead of buffer_load
                # to avoid extra VMEM round-trips in the epilogue.
                # keep raw: fx.memref_load expects fly-level memref types,
                # not this raw LDS memref
                fused2 = memref.load(lds_tid, [row_local])
                row_i32 = fx.Int32(row).ir_value()
                row_valid0 = fx.Uint32(row_i32) < num_valid_i32
                t = fused2 & mask24_i32
                s = fused2 >> 24
                t_ok = fx.Uint32(t) < tokens_i32
                s_ok = fx.Uint32(s) < topk_i32_v
                row_valid = (row_valid0 & (t_ok & s_ok)).ir_value()
                t_idx = fx.Index(t)
                s_idx = fx.Index(s)
                ts_idx = t_idx * topk + s_idx
                if const_expr(accumulate):
                    row_byte_base = out_base_idx + t_idx * (model_dim * out_elem_bytes)
                else:
                    row_byte_base = out_base_idx + ts_idx * (model_dim * out_elem_bytes)
                return ((fused2, row_byte_base), row_valid)

            def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                fused, row_byte_base = row_ctx
                if const_expr(not bool(accumulate)):
                    # ---- 64-bit global store path (avoids i32 offset overflow) ----
                    col_idx = col_g0
                    byte_off_col = col_idx * out_elem_bytes
                    ptr_addr_idx = row_byte_base + byte_off_col
                    out_ptr_v = idx_to_llvm_ptr(ptr_addr_idx)
                    store_or_atomic_add(
                        out_ptr_v,
                        frag,
                        accumulate=False,
                        alignment=_e_vec * out_elem_bytes,
                    )
                else:
                    # ---- accumulate=True: 64-bit global atomic path ----
                    col_idx = col_g0
                    byte_off_col = col_idx * out_elem_bytes
                    ptr_addr_idx = row_byte_base + byte_off_col
                    out_ptr_v = idx_to_llvm_ptr(ptr_addr_idx)
                    store_or_atomic_add(
                        out_ptr_v,
                        frag,
                        accumulate=True,
                        alignment=_e_vec * out_elem_bytes,
                    )

            _e_vec = 2 if accumulate else min(tile_n // 32, 8)
            c_shuffle_epilog(
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
                frag_elem_type=(ir.BF16Type.get() if out_is_bf16 else ir.F16Type.get()),
                write_row_to_lds=write_row_to_lds,
                precompute_row=precompute_row,
                store_pair=store_pair,
            )

        _all_valid = blk_valid & (exp_valid & tile_has_tokens)

        if const_expr(_persistent):
            # Short-circuit: contiguous tiles are monotonically increasing,
            # so once bx_m >= num_valid_ids all remaining tiles are invalid.
            # _cur_active is raw i1: it is yielded as the scf.ForOp iter_arg.
            _cur_active = (fx.Boolean(_still_active) & blk_valid).ir_value()
            _do_gemm = fx.Boolean(_cur_active) & (exp_valid & tile_has_tokens)
            if _do_gemm:
                _moe_gemm2_then_body()

            gpu.barrier()
            scf.YieldOp([_cur_active])
        else:
            if _all_valid:
                _moe_gemm2_then_body()

            gpu.barrier()
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
        _cu_num if _persistent else 0,
        xcd_swizzle,
    )

    @flyc.jit
    def launch_mixed_moe_gemm2(
        arg_out: fx.Pointer,
        arg_x: fx.Pointer,
        arg_w: fx.Pointer,
        arg_scale_x: fx.Pointer,
        arg_scale_w: fx.Pointer,
        arg_sorted_token_ids: fx.Pointer,
        arg_expert_ids: fx.Pointer,
        arg_sorted_weights: fx.Pointer,
        arg_num_valid_ids: fx.Pointer,
        arg_bias: fx.Pointer,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
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

        n_in = fx.Index(i32_n_in.ir_value())
        _tile_n_idx = fx.Index(tile_n)
        _model_dim_pad_idx = fx.Index(model_dim_pad)
        gx = (n_in - _model_dim_pad_idx + _tile_n_idx - 1) // _tile_n_idx
        if const_expr(_persistent):
            gy = fx.Index(_cu_num)
        else:
            _c_pm_l = fx.Index(persist_m)
            gy = (fx.Index(i32_size_expert_ids_in.ir_value()) + _c_pm_l - 1) // _c_pm_l

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
