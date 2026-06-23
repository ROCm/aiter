"""MoE GEMM stage1/stage2 kernel builders (FLIR MFMA, mxfp4/mxfp8)."""

from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch


from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from flydsl._mlir import ir
from flydsl.expr.typing import T

from flydsl.expr import arith, gpu, buffer_ops, vector, rocdl, const_expr
from flydsl.expr.gpu import lds_space as _lds_space
from flydsl._mlir.extras import types as _mT
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
from .mfma_epilogues import c_shuffle_epilog, default_epilog
from .layout_utils import crd2idx, idx2crd, get as layout_get

import functools

from aiter.ops.flydsl.moe_common import (
    GateMode,
)


def barrier(vmcnt=63, lgkmcnt=63):
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
    swiglu_limit: float | None = None,
    k_wave: int = 1,
):
    """Compile stage1 kernel: act(X @ W_gate.T, X @ W_up.T) -> [tokens*topk, inter_dim]."""
    gpu_arch = get_hip_arch()
    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")

    if a_dtype not in ("fp8", "fp4"):
        raise ValueError(f"a_dtype must be one of ('fp8','fp4'), got {a_dtype!r}")
    if b_dtype not in ("fp8", "fp4"):
        raise ValueError(f"b_dtype must be one of ('fp8','fp4'), got {b_dtype!r}")

    is_f8_a = a_dtype == "fp8"
    is_f4_a = a_dtype == "fp4"
    is_f4_b = b_dtype == "fp4"
    is_f8_b = b_dtype == "fp8"

    sort_block_m = max(32, tile_m)
    num_waves = min(4, tile_n // 32)
    # accumulators are reduced in LDS before the epilogue. k_wave=1 keeps the
    num_n_waves = num_waves
    num_waves_total = num_n_waves * k_wave
    # threads, so the cooperative-load striding is group-local.
    a_load_threads = num_n_waves * 64
    total_threads = num_waves_total * 64
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
    blgp = 0 if is_f8_b else 4
    b_byte_div = 2 if is_f4_b else 1
    b_cells_per_ku = 2 if is_f8_b else 1

    if (tile_k_bytes % 64) != 0:
        raise ValueError(f"tile_k_bytes must be divisible by 64, got {tile_k_bytes}")

    out_s = str(out_dtype).strip().lower()
    out_is_f32 = out_s in ("f32", "fp32", "float")
    out_is_bf16 = out_s in ("bf16", "bfloat16")

    def x_elem_type():
        if is_f4_b:
            return T.f8 if is_f8_a else T.i8
        return T.f8

    def w_elem_type():
        if is_f4_b:
            return T.i8
        return T.f8

    def out_elem():
        return T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)

    def load_bias_scalar(bias_rsrc, offset):
        return buffer_ops.buffer_load(bias_rsrc, offset, vec_width=1, dtype=T.f32)

    mock_gate_only = gate_mode is GateMode.MOCK_GATE_ONLY
    gate_up_interleave = gate_mode is GateMode.INTERLEAVE
    gate_only = gate_mode is GateMode.GATE_ONLY

    is_splitk = k_batch > 1
    if mock_gate_only and not is_splitk:
        raise ValueError("mock_gate_only requires k_batch > 1 (split-K)")
    if is_splitk:
        k_per_batch = model_dim // k_batch
        assert (
            model_dim % k_batch == 0
        ), f"model_dim={model_dim} not divisible by k_batch={k_batch}"
        assert (
            k_per_batch % tile_k == 0
        ), f"K_per_batch={k_per_batch} not divisible by tile_k={tile_k}"

        out_dtype = "bf16"
    else:
        k_per_batch = model_dim
    k_dim = k_per_batch

    if k_wave > 1:
        if k_dim % k_wave != 0:
            raise ValueError(f"model_dim={k_dim} not divisible by k_wave={k_wave}")
        klen = k_dim // k_wave
        if klen % tile_k != 0:
            raise ValueError(f"K per group={klen} not divisible by tile_k={tile_k}")
    else:
        klen = k_dim

    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(a_elem_bytes)
    # For k_wave=1 this equals total_threads (unchanged behaviour).
    if bytes_x_per_tile % a_load_threads != 0:
        raise ValueError(
            f"tile_m*tile_k*elem_bytes must be divisible by {a_load_threads}"
        )
    bytes_per_thread_x = bytes_x_per_tile // a_load_threads

    lds_stride = tile_k

    _use_cshuffle_epilog = True

    need_fp4 = out_dtype == "fp4"
    need_fp8 = out_dtype == "fp8"
    need_quant = need_fp4 or need_fp8
    need_sort = need_quant

    fp4q_tag = "_fp4q" if need_fp4 else ""
    fp8q_tag = "_fp8q" if need_fp8 else ""
    sort_tag = "_sort" if need_sort else ""
    async_tag = "_async" if use_async_copy else ""
    sk_tag = f"_sk{k_batch}" if is_splitk else ""
    kw_tag = f"_kw{k_wave}" if k_wave > 1 else ""
    go_tag = "_go" if mock_gate_only else ""
    gui_tag = "_gui" if gate_up_interleave else ""
    as1_tag = "_as1" if a_scale_one else ""
    xcd_tag = f"_xcd{xcd_swizzle}" if xcd_swizzle > 0 else ""
    module_name = (
        f"mfma_moe1_silu_mul_a{a_dtype}_w{b_dtype}_{out_s}"
        f"_t{tile_m}x{tile_n}x{tile_k}_pm{persist_m}{fp4q_tag}{fp8q_tag}{sort_tag}{async_tag}{sk_tag}{kw_tag}{go_tag}{gui_tag}{as1_tag}{xcd_tag}_v32"
    ).replace("-", "_")

    cshuffle_elem_bytes = 4 if need_quant else (4 if out_is_f32 else 2)
    single_x_bytes = int(tile_m) * int(lds_stride) * int(a_elem_bytes)
    x_region_bytes = k_wave * single_x_bytes
    lds_out_bytes = (
        cshuffle_elem_bytes * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    )
    lds_tid_bytes = int(tile_m) * 4
    input_elems = single_x_bytes if a_elem_bytes == 1 else (single_x_bytes // 2)

    GLOBAL_ALIGN = 1024
    std_pong = max(x_region_bytes, lds_out_bytes) + lds_tid_bytes
    std_ping = x_region_bytes
    std_pong_aligned = allocator_pong._align(std_pong, 128)
    std_total = allocator_pong._align(
        std_pong_aligned, GLOBAL_ALIGN
    ) + allocator_pong._align(std_ping, 128)
    lds_limit = {"gfx950": 163840, "gfx942": 65536}.get(gpu_arch, 0)

    split_lds_out = (
        k_wave == 1
        and lds_limit > 0
        and lds_out_bytes > 0
        and std_total > lds_limit
        and num_waves >= 2
    )

    if split_lds_out:
        half_out_bytes = cshuffle_elem_bytes * int(tile_m) * (int(tile_n) // 2)
        pong_buffer_bytes = max(single_x_bytes, half_out_bytes)
        ping_buffer_bytes = max(single_x_bytes, half_out_bytes)
    else:
        pong_buffer_bytes = max(x_region_bytes, lds_out_bytes)
        ping_buffer_bytes = x_region_bytes

    def x_lds_elem():
        return T.f8

    lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = lds_pong_offset + pong_buffer_bytes
    lds_tid_offset_pong = allocator_pong._align(allocator_pong.ptr, 4)
    allocator_pong.ptr = lds_tid_offset_pong + lds_tid_bytes

    lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = lds_ping_offset + ping_buffer_bytes

    if waves_per_eu is not None and waves_per_eu >= 1:
        total_cu_lds = 160 * 1024
        min_lds = total_cu_lds // (waves_per_eu + 1) + 1
        pong_sz = allocator_pong._align(allocator_pong.ptr, 128)
        ping_sz = allocator_ping._align(allocator_ping.ptr, 128)
        cur_lds = pong_sz + ping_sz
        if cur_lds < min_lds:
            allocator_ping.ptr += min_lds - cur_lds

    kpack_bytes = 16
    out_elem_bytes = 4 if out_is_f32 else 2
    w_elem_bytes = 1
    w_elem_pack = 2 if is_f4_b else 1
    w_nbytes = (experts * (2 * inter_dim) * model_dim * w_elem_bytes) // w_elem_pack
    bias_nbytes = experts * (2 * inter_dim) * 4

    e_vec_s1 = min(tile_n // 32, 8)
    if need_quant:
        e_vec_s1 = max(2, e_vec_s1)
    num_threads_per_quant_blk_s1 = 32 // e_vec_s1
    shuffle_dists_s1 = []
    sh_val = 1
    while sh_val < num_threads_per_quant_blk_s1:
        shuffle_dists_s1.append(sh_val)
        sh_val *= 2
    num_shuffle_steps_s1 = len(shuffle_dists_s1)

    pipe_m_repeat = tile_m // 16
    pipe_k_unroll = tile_k_bytes // 128
    pipe_k_unroll_packed = pipe_k_unroll // pack_K
    pipe_num_acc_n = n_per_wave // 16

    pipe_a_groups = []
    for mi in range(pipe_m_repeat):
        grp = []
        for k in range(pipe_k_unroll):
            grp.append((k, mi))
            if len(grp) == 2:
                pipe_a_groups.append(grp)
                grp = []
        if grp:
            pipe_a_groups.append(grp)

    pipe_b_loads = []
    for ku in range(pipe_k_unroll):
        for ni in range(pipe_num_acc_n):
            pipe_b_loads.append(("gate", ku, ni))
            if not mock_gate_only and not gate_up_interleave:
                pipe_b_loads.append(("up", ku, ni))

    pipe_num_acc_n_packed = pipe_num_acc_n // pack_N
    pipe_all_mfma = []
    for ku128 in range(pipe_k_unroll_packed):
        for ni_packed in range(pipe_num_acc_n_packed):
            for ikxdl in range(pack_K):
                for inxdl in range(pack_N):
                    k_idx = ku128 * pack_K + ikxdl
                    ni_idx = ni_packed * pack_N + inxdl
                    pipe_all_mfma.append((k_idx, ni_idx, ikxdl, inxdl, ku128))

    pipe_mfma_per_phase = max(1, len(pipe_all_mfma) // 4)
    pipe_n_phases = len(pipe_all_mfma) // pipe_mfma_per_phase

    a_groups_per_phase = (len(pipe_a_groups) + pipe_n_phases - 1) // pipe_n_phases
    pipe_phases = []
    mfma_i = 0
    a_i = 0
    for _p in range(pipe_n_phases):
        a_reads = []
        for _ in range(a_groups_per_phase):
            if a_i < len(pipe_a_groups):
                a_reads.extend(pipe_a_groups[a_i])
                a_i += 1
        phase = {
            "mfma": pipe_all_mfma[mfma_i : mfma_i + pipe_mfma_per_phase],
            "a_reads": a_reads,
            "b_loads": [],
            "has_scale": (_p == 0),
        }
        mfma_i += pipe_mfma_per_phase
        pipe_phases.append(phase)

    bi = 0
    for _p in range(1, pipe_n_phases):
        rem_b = len(pipe_b_loads) - bi
        rem_p = pipe_n_phases - _p
        n_b = (rem_b + rem_p - 1) // rem_p if rem_p > 0 else 0
        for _ in range(n_b):
            if bi < len(pipe_b_loads):
                pipe_phases[_p]["b_loads"].append(pipe_b_loads[bi])
                bi += 1

    pp_mfma = [p["mfma"] for p in pipe_phases]
    pp_a_reads = [p["a_reads"] for p in pipe_phases]
    pp_b_loads = [p["b_loads"] for p in pipe_phases]
    pp_has_scale = [p["has_scale"] for p in pipe_phases]

    fp4_ratio = 2 if a_dtype == "fp4" else 1
    gui_ratio = 1 if gate_up_interleave else 2
    b_load_mult = 2 if is_f8_b else 1
    vmcnt_before_barrier = (
        tile_m // 32 // fp4_ratio + tile_n // 32 * gui_ratio * b_load_mult
    )

    if True:

        @flyc.kernel(name=module_name, known_block_size=[total_threads, 1, 1])
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

            tokens_in = arith.index_cast(ir.IndexType.get(), i32_tokens_in.ir_value())
            n_in = arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
            k_in = arith.index_cast(ir.IndexType.get(), i32_k_in.ir_value())
            size_expert_ids_in = arith.index_cast(
                ir.IndexType.get(), i32_size_expert_ids_in.ir_value()
            )

            x_elem = T.f8
            f32 = T.f32
            i32 = T.i32
            i64 = T.i64
            vec4_f32 = T.vec(4, f32)
            vec16_elems = 16 if a_elem_bytes == 1 else 8
            vec16_x = T.vec(vec16_elems, x_elem)
            vec2_i64 = T.vec(2, i64)

            def ptr_buffer_resource(ptr, num_records_bytes):
                addr = fx.ptrtoint(ptr)
                addr_i64 = arith.index_cast(T.i64, addr)
                return buffer_ops.create_buffer_resource_from_addr(
                    addr_i64, num_records_bytes=num_records_bytes
                )

            acc_init = arith.constant_vector(0.0, vec4_f32)

            c_n_total = arith.constant(experts * (2 * inter_dim), index=True)
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=k_in // b_byte_div,
                kpack_bytes=kpack_bytes,
                elem_bytes=b_elem_bytes,
            )
            layout_b = b_layout.layout_b

            sorted_m = size_expert_ids_in * arith.constant(sort_block_m, index=True)
            layout_a_scale = make_preshuffle_scale_layout(
                arith, c_mn=sorted_m, c_k=arith.constant(model_dim, index=True)
            )
            layout_b_scale = make_preshuffle_scale_layout(
                arith, c_mn=c_n_total, c_k=arith.constant(model_dim, index=True)
            )

            eff_lds_stride = lds_stride
            eff_tile_k_bytes = tile_k_bytes
            if const_expr(use_async_copy and a_elem_vec_pack > 1):
                eff_lds_stride = lds_stride // a_elem_vec_pack
                eff_tile_k_bytes = tile_k_bytes // a_elem_vec_pack

            shape_lds = fx.make_shape(tile_m, eff_lds_stride)
            stride_lds = fx.make_stride(eff_lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            by = gpu.block_id("x")
            bx_persist = gpu.block_id("y")

            if const_expr(xcd_swizzle > 0):
                NUM_XCDS_S1 = 8
                c1_sw = arith.constant(1, index=True)
                c_tn_sw = arith.constant(tile_n, index=True)
                c_idp_sw = arith.constant(2 * inter_dim_pad, index=True)
                if const_expr(mock_gate_only or gate_up_interleave):
                    gx = (n_in - c_idp_sw + c_tn_sw - c1_sw) / c_tn_sw
                else:
                    c2_sw = arith.constant(2, index=True)
                    gx = (n_in - c_idp_sw + c2_sw * c_tn_sw - c1_sw) / c_tn_sw / c2_sw
                c_pm_sw = arith.constant(persist_m, index=True)
                gy = (size_expert_ids_in + c_pm_sw - c1_sw) / c_pm_sw

                linear_id = bx_persist * gx + by
                num_wgs = gx * gy

                c_xcds = arith.constant(NUM_XCDS_S1, index=True)
                wgs_per_xcd = num_wgs / c_xcds
                wgid = (linear_id % c_xcds) * wgs_per_xcd + (linear_id / c_xcds)

                WGM_S1 = xcd_swizzle
                c_wgm = arith.constant(WGM_S1, index=True)
                num_wgid_in_group = c_wgm * gx
                group_id = wgid / num_wgid_in_group
                first_pid_m = group_id * c_wgm
                remaining_m = gy - first_pid_m
                cmp_m = arith.cmpi(CmpIPredicate.ult, remaining_m, c_wgm)
                group_size_m = arith.select(cmp_m, remaining_m, c_wgm)

                wgid_in_group = wgid % num_wgid_in_group
                bx_persist = first_pid_m + (wgid_in_group % group_size_m)
                by = wgid_in_group / group_size_m

            by_n = by * arith.constant(tile_n, index=True)

            k_base_idx = arith.index(0)
            if const_expr(is_splitk):
                bz = gpu.block_id("z")
                k_base_idx = bz * arith.constant(k_dim, index=True)

            k_blocks16 = arith.constant(eff_tile_k_bytes // 16, index=True)
            layout_tx_wave_lane = fx.make_layout((num_waves_total, 64), stride=(64, 1))
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

            base_ptr_pong = allocator_pong.get_base()
            base_ptr_ping = allocator_ping.get_base()
            lds_x_pong = SmemPtr(
                base_ptr_pong, lds_pong_offset, x_lds_elem(), shape=(input_elems,)
            ).get()
            lds_x_ping = SmemPtr(
                base_ptr_ping, lds_ping_offset, x_lds_elem(), shape=(input_elems,)
            ).get()
            lds_out_elem_type = (
                T.f32 if need_quant else (T.bf16 if out_is_bf16 else T.f16)
            )
            if const_expr(split_lds_out and _use_cshuffle_epilog):
                half_out_elems = int(tile_m) * (int(tile_n) // 2)
                lds_out = SmemPtr(
                    base_ptr_pong,
                    lds_pong_offset,
                    lds_out_elem_type,
                    shape=(half_out_elems,),
                ).get()
                lds_out_B = SmemPtr(
                    base_ptr_ping,
                    lds_ping_offset,
                    lds_out_elem_type,
                    shape=(half_out_elems,),
                ).get()
            else:
                lds_out = (
                    SmemPtr(
                        base_ptr_pong,
                        lds_pong_offset,
                        lds_out_elem_type,
                        shape=(tile_m * tile_n,),
                    ).get()
                    if _use_cshuffle_epilog
                    else None
                )
                lds_out_B = None
            lds_tid = SmemPtr(
                base_ptr_pong, lds_tid_offset_pong, T.i32, shape=(tile_m,)
            ).get()

            c_a_pack = arith.constant(int(a_elem_vec_pack), index=True)
            c_elem_bytes = arith.constant(int(a_elem_bytes), index=True)

            x_nbytes_idx = (tokens_in * k_in * c_elem_bytes) / c_a_pack
            x_nbytes_i32 = arith.index_cast(T.i32, x_nbytes_idx)
            x_rsrc = ptr_buffer_resource(arg_x, x_nbytes_i32)

            w_rsrc = ptr_buffer_resource(arg_w, w_nbytes)

            numids_rsrc = ptr_buffer_resource(
                arg_num_valid_ids, arith.constant(4, type=T.i32)
            )
            num_valid_i32 = buffer_ops.buffer_load(
                numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=T.i32
            )

            sx_rsrc = 1
            sw_rsrc = 1
            if const_expr(not a_scale_one):
                c32 = arith.constant(32, index=True)
                kblk = k_in / c32
                sx_nbytes_idx = sorted_m * kblk
                sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
                sx_rsrc = ptr_buffer_resource(arg_scale_x, sx_nbytes_i32)

            c32 = arith.constant(32, index=True)
            kblk_w = k_in / c32
            mn_w = arith.constant(experts * (2 * inter_dim), index=True)
            sw_nbytes_idx = mn_w * kblk_w
            sw_nbytes_i32 = arith.index_cast(T.i32, sw_nbytes_idx)
            sw_rsrc = ptr_buffer_resource(arg_scale_w, sw_nbytes_i32)

            sorted_nbytes_idx = size_expert_ids_in * arith.constant(
                sort_block_m * 4, index=True
            )
            sorted_nbytes_i32 = arith.index_cast(T.i32, sorted_nbytes_idx)
            sorted_rsrc = ptr_buffer_resource(arg_sorted_token_ids, sorted_nbytes_i32)
            sorted_w_rsrc = ptr_buffer_resource(arg_sorted_weights, sorted_nbytes_i32)

            eid_nbytes_idx = size_expert_ids_in * arith.constant(4, index=True)
            eid_nbytes_i32 = arith.index_cast(T.i32, eid_nbytes_idx)
            expert_rsrc = ptr_buffer_resource(arg_expert_ids, eid_nbytes_i32)
            bias_rsrc = (
                ptr_buffer_resource(arg_bias, bias_nbytes) if enable_bias else None
            )

            # #3476: pad group-N (= inter_dim/32) up to a multiple of 8 so it
            sorted_scale_cols = ((inter_dim // 32) + 7) // 8 * 8
            sorted_scale_cols_i32 = arith.constant(sorted_scale_cols, type=T.i32)
            sorted_scale_rsrc = None
            if const_expr(need_sort):
                sort_rows_idx = size_expert_ids_in * arith.constant(
                    sort_block_m, index=True
                )
                sort_padded_rows = (
                    (sort_rows_idx + arith.constant(255, index=True))
                    / arith.constant(256, index=True)
                    * arith.constant(256, index=True)
                )
                sort_padded_cols = arith.constant(
                    ((sorted_scale_cols + 7) // 8) * 8, index=True
                )
                sort_scale_nbytes = arith.index_cast(
                    T.i32, sort_padded_rows * sort_padded_cols
                )
                sorted_scale_rsrc = ptr_buffer_resource(
                    arg_out_scale_sorted, sort_scale_nbytes
                )

            PERSIST_M = persist_m
            c0_p = arith.constant(0, index=True)
            c1_p = arith.constant(1, index=True)
            c_pm = arith.constant(PERSIST_M, index=True)
            for_persist = scf.ForOp(c0_p, c_pm, c1_p)
            for_ip = ir.InsertionPoint(for_persist.body)
            for_ip.__enter__()
            mi_p = for_persist.induction_variable
            bx = bx_persist * c_pm + mi_p
            bx_m = bx * arith.constant(sort_block_m, index=True)

            bx_m_i32 = arith.index_cast(T.i32, bx_m)
            blk_valid = arith.cmpi(CmpIPredicate.ult, bx_m_i32, num_valid_i32)
            expert_i32 = buffer_ops.buffer_load(
                expert_rsrc, bx, vec_width=1, dtype=T.i32
            )
            expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
            exp_valid = arith.cmpi(
                CmpIPredicate.ult, expert_i32, arith.constant(experts, type=T.i32)
            )

            def moe_gemm1_body():
                nonlocal k_base_idx, lds_x_pong, lds_x_ping
                expert_off_idx = expert_idx * arith.constant(2 * inter_dim, index=True)

                x_load_bytes = 16
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4

                c_k_div4 = (
                    (k_in / c_a_pack) * arith.constant(int(a_elem_bytes), index=True)
                ) / arith.index(4)
                tile_k_dwords = (int(tile_k) * int(a_elem_bytes)) // (
                    4 * int(a_elem_vec_pack)
                )
                layout_x_tile_div4 = fx.make_layout(
                    (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                )
                c_chunk_i32 = arith.constant(chunk_i32, index=True)
                if const_expr(k_wave > 1):
                    x_load_tid = tx % arith.constant(a_load_threads, index=True)
                else:
                    x_load_tid = tx
                tx_i32_base = x_load_tid * c_chunk_i32

                topk_i32 = arith.constant(topk)
                mask24 = arith.constant(0xFFFFFF)
                tokens_i32 = arith.index_cast(T.i32, tokens_in)

                def x_tile_chunk_coord_i32(i: int):
                    return tile_chunk_coord_i32(
                        arith,
                        tx_i32_base=tx_i32_base,
                        i=i,
                        total_threads=a_load_threads,
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

                coord_wl = idx2crd(tx, layout_tx_wave_lane)
                wave_id = layout_get(coord_wl, 0)
                lane_id = layout_get(coord_wl, 1)
                coord_l16 = idx2crd(lane_id, layout_lane16)
                lane_div_16 = layout_get(coord_l16, 0)
                lane_mod_16 = layout_get(coord_l16, 1)
                row_a_lds = lane_mod_16
                col_offset_base = lane_div_16 * arith.constant(16, index=True)

                if const_expr(k_wave > 1):
                    wave_k_id = wave_id / arith.constant(num_n_waves, index=True)
                    k_base_idx = k_base_idx + wave_k_id * arith.constant(
                        klen, index=True
                    )
                    grp_x_bytes = wave_k_id * arith.constant(single_x_bytes, index=True)
                    x_view_ty = _mT.memref(
                        input_elems, x_lds_elem(), memory_space=_lds_space()
                    )
                    pong_off = arith.constant(lds_pong_offset, index=True) + grp_x_bytes
                    ping_off = arith.constant(lds_ping_offset, index=True) + grp_x_bytes
                    lds_x_pong = memref.view(
                        x_view_ty, base_ptr_pong, pong_off, sizes=[]
                    )
                    lds_x_ping = memref.view(
                        x_view_ty, base_ptr_ping, ping_off, sizes=[]
                    )
                else:
                    wave_k_id = arith.index(0)

                num_acc_n = n_per_wave // 16
                c_n_per_wave = arith.constant(n_per_wave, index=True)
                wave_n_id = wave_id % arith.constant(num_waves, index=True)
                n_tile_base = wave_n_id * c_n_per_wave

                gate_n_intra_list = []
                gate_n_blk_list = []
                up_n_intra_list = []
                up_n_blk_list = []
                col_g_list = []
                c_n0_static = experts * (2 * inter_dim) // 16
                layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))
                inter_idx = arith.constant(inter_dim, index=True)

                for i in range_constexpr(num_acc_n):
                    offset = i * 16
                    c_offset = arith.constant(offset, index=True)
                    if const_expr(not gate_up_interleave):
                        col_g = by_n + n_tile_base + c_offset + lane_mod_16
                        col_g_list.append(col_g)

                    global_n = by_n + n_tile_base + c_offset + lane_mod_16
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
                    gui_num_acc_n_out = num_acc_n // pack_N
                    for gui_i in range_constexpr(gui_num_acc_n_out):
                        gui_offset = gui_i * 16
                        gui_c_offset = arith.constant(gui_offset, index=True)
                        gui_col_g = (
                            (by_n + n_tile_base) // arith.constant(2, index=True)
                            + gui_c_offset
                            + lane_mod_16
                        )
                        col_g_list.append(gui_col_g)

                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 128
                k_unroll_packed = k_unroll // pack_K
                m_repeat_packed = m_repeat // pack_M
                num_acc_n_packed = num_acc_n // pack_N

                K_per_ku = tile_k // k_unroll
                pad_k_elems = (
                    (model_dim_pad % tile_k)
                    if (not is_splitk and model_dim_pad > 0)
                    else 0
                )
                pad_ku_skip = pad_k_elems // K_per_ku
                tail_ku = k_unroll - pad_ku_skip
                tail_ku_packed = (
                    (tail_ku + pack_K - 1) // pack_K if pad_ku_skip > 0 else None
                )

                def load_b_packs_k64(base_k, ku: int, n_blk, n_intra):
                    c64 = arith.constant(64, index=True)
                    c1 = arith.constant(1, index=True)
                    base_k_bytes = base_k * arith.constant(
                        int(b_elem_bytes), index=True
                    )
                    k0_base = base_k_bytes // c64 + arith.constant(
                        ku * b_cells_per_ku, index=True
                    )
                    k1 = lane_div_16
                    vec_elems = kpack_bytes // int(b_elem_bytes)

                    def load_cell(k0):
                        coord_pack = (
                            n_blk,
                            k0,
                            k1,
                            n_intra,
                            arith.constant(0, index=True),
                        )
                        idx_pack = crd2idx(coord_pack, layout_b)
                        b16 = _buffer_load_vec(
                            buffer_ops,
                            vector,
                            w_rsrc,
                            idx_pack,
                            elem_type=w_elem_type(),
                            vec_elems=vec_elems,
                            elem_bytes=b_elem_bytes,
                            offset_in_bytes=(b_elem_bytes == 1),
                            cache_modifier=b_nt,
                        )
                        b_i64x2 = vector.bitcast(vec2_i64, b16)
                        return (
                            vector.extract(
                                b_i64x2, static_position=[0], dynamic_position=[]
                            ),
                            vector.extract(
                                b_i64x2, static_position=[1], dynamic_position=[]
                            ),
                        )

                    b0, b1 = load_cell(k0_base)
                    if const_expr(is_f8_b):
                        b2, b3 = load_cell(k0_base + c1)
                        return b0, b1, b2, b3
                    return b0, b1

                def load_b_tile(base_k, ku_limit=k_unroll):
                    """Load B tiles -> (gate_b_tile, up_b_tile); up is None for
                    interleave/mock. Tile entry (packs0..3); packs2/3 fp8-B only."""
                    gate_b_tile = []
                    up_b_tile = (
                        [] if (not mock_gate_only and not gate_up_interleave) else None
                    )
                    for ku in range_constexpr(ku_limit):
                        g_packs0, g_packs1, g_packs2, g_packs3 = [], [], [], []
                        u_packs0, u_packs1, u_packs2, u_packs3 = [], [], [], []
                        for ni in range_constexpr(num_acc_n):
                            gb = load_b_packs_k64(
                                base_k, ku, gate_n_blk_list[ni], gate_n_intra_list[ni]
                            )
                            g_packs0.append(gb[0])
                            g_packs1.append(gb[1])
                            if const_expr(is_f8_b):
                                g_packs2.append(gb[2])
                                g_packs3.append(gb[3])
                            if const_expr(
                                not mock_gate_only and not gate_up_interleave
                            ):
                                ub = load_b_packs_k64(
                                    base_k, ku, up_n_blk_list[ni], up_n_intra_list[ni]
                                )
                                u_packs0.append(ub[0])
                                u_packs1.append(ub[1])
                                if const_expr(is_f8_b):
                                    u_packs2.append(ub[2])
                                    u_packs3.append(ub[3])
                        gate_b_tile.append((g_packs0, g_packs1, g_packs2, g_packs3))
                        if const_expr(not mock_gate_only and not gate_up_interleave):
                            up_b_tile.append((u_packs0, u_packs1, u_packs2, u_packs3))
                    return gate_b_tile, up_b_tile

                scale_lane_elem = (
                    lane_div_16 * layout_b_scale.stride_klane + lane_mod_16
                )

                gate_scale_bases = []
                up_scale_bases = []
                for ni in range_constexpr(num_acc_n_packed):
                    col_base = (
                        by_n
                        + n_tile_base
                        + arith.constant(ni * 16 * pack_N, index=True)
                    )
                    gate_mni = (expert_off_idx + col_base) // arith.constant(
                        32, index=True
                    )
                    gate_scale_bases.append(
                        gate_mni * layout_b_scale.stride_n0 + scale_lane_elem
                    )
                    if const_expr(not mock_gate_only and not gate_up_interleave):
                        up_mni = (
                            expert_off_idx + inter_idx + col_base
                        ) // arith.constant(32, index=True)
                        up_scale_bases.append(
                            up_mni * layout_b_scale.stride_n0 + scale_lane_elem
                        )

                if const_expr(not a_scale_one):
                    a_scale_bases = []
                    for mi in range_constexpr(m_repeat_packed):
                        a_mni = mi + bx_m // scale_mn_pack // 16
                        a_scale_bases.append(
                            a_mni * layout_a_scale.stride_n0 + scale_lane_elem
                        )

                c16_idx = arith.constant(16, index=True)
                c2_idx = arith.constant(2, index=True)
                scale_mask_lo = arith.constant(0xFF, type=T.i32)

                m_half_idx = arith.constant(0, type=T.i32)
                m_half_i32 = arith.constant(0, type=T.i32)
                scale_shift = arith.constant(0, type=T.i32)
                scale_shift_hi = arith.constant(0, type=T.i32)
                n_half_idx = arith.constant(0, type=T.i32)
                n_half_i32 = arith.constant(0, type=T.i32)
                bscale_shift = arith.constant(0, type=T.i32)
                bscale_shift_hi = arith.constant(0, type=T.i32)
                if const_expr(pack_M < scale_mn_pack):
                    m_half_idx = (bx_m // c16_idx) % c2_idx
                    m_half_i32 = arith.index_cast(T.i32, m_half_idx)
                    scale_shift = m_half_i32 * arith.constant(8, type=T.i32)
                    scale_shift_hi = scale_shift + arith.constant(16, type=T.i32)

                if const_expr(pack_N < scale_mn_pack):
                    n_half_idx = (n_tile_base // c16_idx) % c2_idx
                    n_half_i32 = arith.index_cast(T.i32, n_half_idx)
                    bscale_shift = n_half_i32 * arith.constant(8, type=T.i32)
                    bscale_shift_hi = bscale_shift + arith.constant(16, type=T.i32)

                def rearrange_a_scale(raw_i32):
                    """Rearrange scale bytes for pack_M=1: extract m_half's k0,k1 bytes."""
                    if const_expr(pack_M >= scale_mn_pack):
                        return raw_i32
                    b_k0 = arith.andi(arith.shrui(raw_i32, scale_shift), scale_mask_lo)
                    b_k1 = arith.andi(
                        arith.shrui(raw_i32, scale_shift_hi), scale_mask_lo
                    )
                    return arith.ori(
                        b_k0, arith.shli(b_k1, arith.constant(8, type=T.i32))
                    )

                def rearrange_b_scale(raw_i32):
                    """Rearrange scale bytes for pack_N=1: extract n_half's k0,k1 bytes."""
                    if const_expr(pack_N >= scale_mn_pack):
                        return raw_i32
                    b_k0 = arith.andi(arith.shrui(raw_i32, bscale_shift), scale_mask_lo)
                    b_k1 = arith.andi(
                        arith.shrui(raw_i32, bscale_shift_hi), scale_mask_lo
                    )
                    return arith.ori(
                        b_k0, arith.shli(b_k1, arith.constant(8, type=T.i32))
                    )

                if const_expr(a_scale_one):
                    as1_const = arith.constant(0x7F7F7F7F, type=T.i32)
                    as1_vec = vector.from_elements(T.vec(1, T.i32), [as1_const])

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
                                a_scale_tile.append(as1_vec)
                            else:
                                s = buffer_ops.buffer_load(
                                    sx_rsrc,
                                    a_scale_bases[mi] + k_off,
                                    vec_width=1,
                                    dtype=T.i32,
                                    cache_modifier=0,
                                )
                                s = rearrange_a_scale(s)
                                a_scale_tile.append(
                                    vector.from_elements(T.vec(1, T.i32), [s])
                                )
                        for ni in range_constexpr(num_acc_n_packed):
                            gs = buffer_ops.buffer_load(
                                sw_rsrc,
                                gate_scale_bases[ni] + k_off,
                                vec_width=1,
                                dtype=T.i32,
                                cache_modifier=0,
                            )
                            gs = rearrange_b_scale(gs)
                            gate_b_scale.append(
                                vector.from_elements(T.vec(1, T.i32), [gs])
                            )
                            if const_expr(
                                not mock_gate_only and not gate_up_interleave
                            ):
                                us = buffer_ops.buffer_load(
                                    sw_rsrc,
                                    up_scale_bases[ni] + k_off,
                                    vec_width=1,
                                    dtype=T.i32,
                                    cache_modifier=0,
                                )
                                us = rearrange_b_scale(us)
                                up_b_scale.append(
                                    vector.from_elements(T.vec(1, T.i32), [us])
                                )
                    return [a_scale_tile, gate_b_scale, up_b_scale]

                lds_base_zero = arith.index(0)

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
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base_zero,
                                vec_part_i32x4=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )

                if const_expr(use_async_copy):
                    dma_bytes = 16
                    wave_size = 64
                    eff_bytes_per_buffer = (
                        int(tile_m) * int(eff_lds_stride) * int(a_elem_bytes)
                    )
                    # A-LDS buffer with a_load_threads (== total_threads at k_wave=1).
                    num_dma_loads = max(
                        1, eff_bytes_per_buffer // (a_load_threads * dma_bytes)
                    )

                    def dma_x_tile_to_lds(base_k, lds_buffer):
                        c4_idx = arith.index(4)
                        base_k_div4 = (
                            (base_k / c_a_pack)
                            * arith.constant(int(elem_bytes), index=True)
                        ) / arith.index(4)

                        lds_ptr_i64 = None
                        for i in range_constexpr(num_dma_loads):
                            row_local_i = x_row_local[i]
                            col_local_i32_i = x_col_local_i32[i]
                            col_local_sw = swizzle_xor16(
                                row_local_i, col_local_i32_i * c4_idx, k_blocks16
                            )
                            row_k_dw = x_row_base_div4[i] + base_k_div4
                            global_byte_idx = row_k_dw * c4_idx + col_local_sw
                            global_offset = arith.index_cast(T.i32, global_byte_idx)

                            if const_expr(i == 0):
                                lds_addr = memref.extract_aligned_pointer_as_index(
                                    lds_buffer
                                ) + wave_n_id * arith.constant(
                                    wave_size * dma_bytes, index=True
                                )
                                lds_ptr_i64 = rocdl.readfirstlane(
                                    T.i64, arith.index_cast(T.i64, lds_addr)
                                )
                            else:
                                lds_ptr_i64 = lds_ptr_i64 + arith.constant(
                                    a_load_threads * dma_bytes, type=T.i64
                                )

                            lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                            lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64)

                            rocdl.raw_ptr_buffer_load_lds(
                                x_rsrc,
                                lds_ptr,
                                arith.constant(dma_bytes, type=T.i32),
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

                def prefetch_full_a_from_lds(lds_buffer, ku_limit=k_unroll):
                    """Load entire A tile from LDS into registers before compute."""
                    a_regs = []
                    for k_idx in range_constexpr(ku_limit):
                        col_base = col_offset_base + (k_idx * 128) // a_elem_vec_pack
                        for mi_idx in range_constexpr(m_repeat):
                            mi_val = arith.constant(mi_idx * 16, index=True)
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
                    single_b = mock_gate_only or gate_up_interleave
                    up_list = None if single_b else list(acc_up_in)
                    mfma_res_ty = vec4_f32
                    epilogue_pf = None
                    bias_pf = None
                    if const_expr(prefetch_epilogue):
                        if const_expr(enable_bias):
                            if const_expr(gate_up_interleave):
                                bias_pf = []
                                for ni in range_constexpr(num_acc_n):
                                    logical_col = (
                                        (by_n + n_tile_base)
                                        // arith.constant(2, index=True)
                                        + arith.constant((ni // 2) * 16, index=True)
                                        + lane_mod_16
                                    )
                                    up_off = (
                                        inter_idx
                                        if (ni % 2 == 1)
                                        else arith.constant(0, index=True)
                                    )
                                    bias_offset = expert_off_idx + up_off + logical_col
                                    bias_pf.append(
                                        load_bias_scalar(bias_rsrc, bias_offset)
                                    )
                            else:
                                gate_bias_pf = []
                                up_bias_pf = (
                                    [] if const_expr(not mock_gate_only) else None
                                )
                                for ni in range_constexpr(num_acc_n):
                                    global_n = (
                                        by_n
                                        + n_tile_base
                                        + arith.constant(ni * 16, index=True)
                                        + lane_mod_16
                                    )
                                    gate_bias_pf.append(
                                        load_bias_scalar(
                                            bias_rsrc, expert_off_idx + global_n
                                        )
                                    )
                                    if const_expr(not mock_gate_only):
                                        up_bias_pf.append(
                                            load_bias_scalar(
                                                bias_rsrc,
                                                expert_off_idx + inter_idx + global_n,
                                            )
                                        )
                                bias_pf = (gate_bias_pf, up_bias_pf)
                        tw_pf = None
                        if const_expr(doweight_stage1):
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

                    c0_i64 = arith.constant(0, type=T.i64)
                    vec4_i64 = T.vec(4, T.i64)
                    vec8_i32 = T.vec(8, T.i32)

                    def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                        v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                        return vector.bitcast(vec8_i32, v4)

                    eff_packed = (ku_count + pack_K - 1) // pack_K
                    for ku128 in range_constexpr(eff_packed):
                        for ni in range_constexpr(num_acc_n_packed):
                            gate_bs_i32 = gate_b_scale[ku128 * num_acc_n_packed + ni]
                            gate_bs_val = vector.extract(
                                gate_bs_i32,
                                static_position=[0],
                                dynamic_position=[],
                            )
                            if const_expr(not single_b):
                                up_bs_i32 = up_b_scale[ku128 * num_acc_n_packed + ni]
                                up_bs_val = vector.extract(
                                    up_bs_i32, static_position=[0], dynamic_position=[]
                                )
                            for ikxdl in range_constexpr(pack_K):
                                k_idx = ku128 * pack_K + ikxdl
                                if const_expr(k_idx < ku_count):
                                    gate_bp = gate_b_tile_in[k_idx]
                                    if const_expr(not single_b):
                                        up_bp = up_b_tile_in[k_idx]
                                    for inxdl in range_constexpr(pack_N):
                                        ni_idx = ni * pack_N + inxdl
                                        if const_expr(is_f8_b):
                                            gb128 = pack_i64x4_to_i32x8(
                                                gate_bp[0][ni_idx],
                                                gate_bp[1][ni_idx],
                                                gate_bp[2][ni_idx],
                                                gate_bp[3][ni_idx],
                                            )
                                        else:
                                            gb128 = pack_i64x4_to_i32x8(
                                                gate_bp[0][ni_idx],
                                                gate_bp[1][ni_idx],
                                                c0_i64,
                                                c0_i64,
                                            )
                                        if const_expr(not single_b):
                                            if const_expr(is_f8_b):
                                                ub128 = pack_i64x4_to_i32x8(
                                                    up_bp[0][ni_idx],
                                                    up_bp[1][ni_idx],
                                                    up_bp[2][ni_idx],
                                                    up_bp[3][ni_idx],
                                                )
                                            else:
                                                ub128 = pack_i64x4_to_i32x8(
                                                    up_bp[0][ni_idx],
                                                    up_bp[1][ni_idx],
                                                    c0_i64,
                                                    c0_i64,
                                                )
                                        for mi in range_constexpr(m_repeat_packed):
                                            a_scale_i32 = a_scale[
                                                ku128 * m_repeat_packed + mi
                                            ]
                                            a_scale_val = vector.extract(
                                                a_scale_i32,
                                                static_position=[0],
                                                dynamic_position=[],
                                            )
                                            for imxdl in range_constexpr(pack_M):
                                                mi_idx = mi * pack_M + imxdl
                                                a_reg_idx = k_idx * m_repeat + mi_idx
                                                if const_expr(is_f8_a):
                                                    a0, a1, a2, a3 = a_tile_regs[
                                                        a_reg_idx
                                                    ]
                                                    a128 = pack_i64x4_to_i32x8(
                                                        a0, a1, a2, a3
                                                    )
                                                else:
                                                    a0, a1 = a_tile_regs[a_reg_idx]
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
                                                if const_expr(not single_b):
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
                    mi_val = arith.constant(mi_idx * 16, index=True)
                    curr_row = row_a_lds + mi_val
                    a0, a1 = lds_load_packs_k64(curr_row, col_base, lds_buffer)
                    if const_expr(is_f8_a):
                        a2, a3 = lds_load_packs_k64(curr_row, col_base + 64, lds_buffer)
                        return (a0, a1, a2, a3)
                    else:
                        return (a0, a1)

                single_b_pipe = mock_gate_only or gate_up_interleave

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
                    gate_b_single/up_b_single: (b0,b1) fp4 or (b0,b1,b2,b3) fp8 for
                      one ni; up is None under _single_b_pipe.
                    a_scale_vals: list of A scale scalars indexed by mi_packed.
                    """
                    c0_i64 = arith.constant(0, type=T.i64)
                    vec4_i64 = T.vec(4, T.i64)
                    vec8_i32 = T.vec(8, T.i32)

                    def pack(x0, x1, x2, x3):
                        v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                        return vector.bitcast(vec8_i32, v4)

                    def pack_b(b_single):
                        if const_expr(is_f8_b):
                            return pack(
                                b_single[0], b_single[1], b_single[2], b_single[3]
                            )
                        return pack(b_single[0], b_single[1], c0_i64, c0_i64)

                    mfma_res_ty = vec4_f32
                    gb128 = pack_b(gate_b_single)
                    if const_expr(not single_b_pipe):
                        ub128 = pack_b(up_b_single)

                    for mi_p in range_constexpr(m_repeat_packed):
                        a_scale_val = a_scale_vals[mi_p]
                        for imxdl in range_constexpr(pack_M):
                            mi_idx = mi_p * pack_M + imxdl
                            a_reg = all_a_tiles[k_idx * m_repeat + mi_idx]

                            if const_expr(is_f8_a):
                                a128 = pack(a_reg[0], a_reg[1], a_reg[2], a_reg[3])
                            else:
                                a128 = pack(a_reg[0], a_reg[1], c0_i64, c0_i64)

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
                            if const_expr(not single_b_pipe):
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

                def interleaved_half(
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
                    abs_k = k_base_idx + arith.constant(next_k_load, index=True)
                    bk = abs_k // arith.constant(b_byte_div, index=True)
                    sk = abs_k // arith.constant(pack_K * 128, index=True)
                    k_off = sk * layout_b_scale.stride_k0

                    rocdl.sched_barrier(0)
                    rocdl.s_waitcnt(vmcnt_before_barrier)
                    barrier()
                    rocdl.sched_barrier(0)

                    abs_k_dma = k_base_idx + arith.constant(next_k_dma_py, index=True)
                    if const_expr(use_async_copy and next_k_dma_py < int(k_dim)):
                        prefetch_x_to_lds(abs_k_dma, lds_write)
                    if const_expr(not use_async_copy):
                        x_regs = load_x_tile(abs_k_dma)

                    prev_asvs = []
                    for i_as in range_constexpr(len(prev_a_scale)):
                        prev_asvs.append(
                            vector.extract(
                                prev_a_scale[i_as],
                                static_position=[0],
                                dynamic_position=[],
                            )
                        )
                    prev_gsv_list = []
                    for i_gs in range_constexpr(len(prev_gate_bs)):
                        prev_gsv_list.append(
                            vector.extract(
                                prev_gate_bs[i_gs],
                                static_position=[0],
                                dynamic_position=[],
                            )
                        )
                    if const_expr(not single_b_pipe):
                        prev_usv_list = []
                        for i_us in range_constexpr(len(prev_up_bs)):
                            prev_usv_list.append(
                                vector.extract(
                                    prev_up_bs[i_us],
                                    static_position=[0],
                                    dynamic_position=[],
                                )
                            )

                    a_all = {}
                    b_gate_all = {}
                    b_up_all = {}

                    for _p in range_constexpr(pipe_n_phases):
                        if const_expr(pp_has_scale[_p]):
                            new_as_list = []
                            new_gs_list = []
                            new_us_list = [] if const_expr(not single_b_pipe) else None
                            for ku_s in range_constexpr(k_unroll_packed):
                                ku_off = (
                                    k_off
                                    + arith.constant(ku_s, index=True)
                                    * layout_b_scale.stride_k0
                                )
                                for mi_p in range_constexpr(m_repeat_packed):
                                    if const_expr(a_scale_one):
                                        new_as_list.append(as1_const)
                                    else:
                                        raw_as = buffer_ops.buffer_load(
                                            sx_rsrc,
                                            a_scale_bases[mi_p] + ku_off,
                                            vec_width=1,
                                            dtype=T.i32,
                                            cache_modifier=0,
                                        )
                                        new_as_list.append(rearrange_a_scale(raw_as))
                                for gs_ni in range_constexpr(num_acc_n_packed):
                                    gs_raw = buffer_ops.buffer_load(
                                        sw_rsrc,
                                        gate_scale_bases[gs_ni] + ku_off,
                                        vec_width=1,
                                        dtype=T.i32,
                                        cache_modifier=0,
                                    )
                                    new_gs_list.append(rearrange_b_scale(gs_raw))
                                if const_expr(not single_b_pipe):
                                    for us_ni in range_constexpr(num_acc_n_packed):
                                        us_raw = buffer_ops.buffer_load(
                                            sw_rsrc,
                                            up_scale_bases[us_ni] + ku_off,
                                            vec_width=1,
                                            dtype=T.i32,
                                            cache_modifier=0,
                                        )
                                        new_us_list.append(rearrange_b_scale(us_raw))

                        for b_j in range_constexpr(len(pp_b_loads[_p])):
                            b_type, b_ku, b_ni = pp_b_loads[_p][b_j]
                            if const_expr(b_type == "gate"):
                                b_gate_all[(b_ku, b_ni)] = load_b_packs_k64(
                                    bk,
                                    b_ku,
                                    gate_n_blk_list[b_ni],
                                    gate_n_intra_list[b_ni],
                                )
                            else:
                                b_up_all[(b_ku, b_ni)] = load_b_packs_k64(
                                    bk,
                                    b_ku,
                                    up_n_blk_list[b_ni],
                                    up_n_intra_list[b_ni],
                                )

                        rocdl.sched_barrier(0)
                        for a_j in range_constexpr(len(pp_a_reads[_p])):
                            ak, ami = pp_a_reads[_p][a_j]
                            a_all[(ak, ami)] = load_a_subtile(
                                ak,
                                ami,
                                lds_read,
                            )
                        rocdl.sched_barrier(0)

                        rocdl.s_setprio(1)
                        for m_j in range_constexpr(len(pp_mfma[_p])):
                            k_idx, ni_idx, ikxdl, inxdl, ku128 = pp_mfma[_p][m_j]
                            ni_packed_idx = ni_idx // pack_N

                            def mk_single(tile_entry, ni):
                                if const_expr(is_f8_b):
                                    return (
                                        tile_entry[0][ni],
                                        tile_entry[1][ni],
                                        tile_entry[2][ni],
                                        tile_entry[3][ni],
                                    )
                                return (tile_entry[0][ni], tile_entry[1][ni])

                            up_b_single = (
                                mk_single(prev_up_w[k_idx], ni_idx)
                                if not single_b_pipe
                                else None
                            )
                            as_off = ku128 * m_repeat_packed
                            bs_idx = ku128 * num_acc_n_packed + ni_packed_idx
                            compute_bmajor_mfma_phase(
                                prev_a_tile,
                                mk_single(prev_gate_w[k_idx], ni_idx),
                                up_b_single,
                                prev_asvs[as_off : as_off + m_repeat_packed],
                                prev_gsv_list[bs_idx],
                                (prev_usv_list[bs_idx] if not single_b_pipe else None),
                                acc_gate,
                                acc_up,
                                k_idx,
                                ni_idx,
                                ikxdl,
                                inxdl,
                            )
                        rocdl.s_setprio(0)
                        rocdl.sched_barrier(0)

                    cur_a_tile = []
                    for k in range_constexpr(k_unroll):
                        for mi in range_constexpr(m_repeat):
                            cur_a_tile.append(a_all[(k, mi)])

                    cur_gate_w = []
                    cur_up_w = None if single_b_pipe else []
                    for ku in range_constexpr(k_unroll):
                        g_packs0, g_packs1, g_packs2, g_packs3 = [], [], [], []
                        u_packs0, u_packs1, u_packs2, u_packs3 = [], [], [], []
                        for ni in range_constexpr(num_acc_n):
                            g = b_gate_all[(ku, ni)]
                            g_packs0.append(g[0])
                            g_packs1.append(g[1])
                            if const_expr(is_f8_b):
                                g_packs2.append(g[2])
                                g_packs3.append(g[3])
                            if const_expr(not single_b_pipe):
                                u = b_up_all[(ku, ni)]
                                u_packs0.append(u[0])
                                u_packs1.append(u[1])
                                if const_expr(is_f8_b):
                                    u_packs2.append(u[2])
                                    u_packs3.append(u[3])
                        cur_gate_w.append((g_packs0, g_packs1, g_packs2, g_packs3))
                        if const_expr(not single_b_pipe):
                            cur_up_w.append((u_packs0, u_packs1, u_packs2, u_packs3))

                    cur_a_scale = []
                    for i_as in range_constexpr(len(new_as_list)):
                        cur_a_scale.append(
                            vector.from_elements(
                                T.vec(1, T.i32),
                                [new_as_list[i_as]],
                            )
                        )
                    cur_gate_bs = []
                    for i_gs in range_constexpr(len(new_gs_list)):
                        cur_gate_bs.append(
                            vector.from_elements(T.vec(1, T.i32), [new_gs_list[i_gs]])
                        )
                    if const_expr(not single_b_pipe):
                        cur_up_bs = []
                        for i_us in range_constexpr(len(new_us_list)):
                            cur_up_bs.append(
                                vector.from_elements(
                                    T.vec(1, T.i32), [new_us_list[i_us]]
                                )
                            )
                    else:
                        cur_up_bs = None

                    if const_expr(not use_async_copy):
                        store_x_tile_to_lds(x_regs, lds_write)

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

                rocdl.sched_barrier(0)

                k0 = k_base_idx
                if const_expr(use_async_copy):
                    prefetch_x_to_lds(k0, lds_x_pong)
                else:
                    x_regs0 = load_x_tile(k0)
                    store_x_tile_to_lds(x_regs0, lds_x_pong)
                rocdl.sched_barrier(0)
                k0_scale = k_base_idx // arith.constant(pack_K * 128, index=True)
                a_scale_pong, gate_bs_pong, up_bs_pong = prefetch_ab_scale_tile(
                    k0_scale
                )
                c_tile_m_idx = arith.constant(tile_m, index=True)
                tid_in_range = arith.cmpi(CmpIPredicate.ult, tx, c_tile_m_idx)
                if_tid = scf.IfOp(tid_in_range)
                with ir.InsertionPoint(if_tid.then_block):
                    tid_row = bx_m + tx
                    tid_val = buffer_ops.buffer_load(
                        sorted_rsrc, tid_row, vec_width=1, dtype=T.i32
                    )
                    tid_vec1 = vector.from_elements(T.vec(1, T.i32), [tid_val])
                    vector.store(tid_vec1, lds_tid, [tx])
                    scf.YieldOp([])

                acc_gate = [acc_init] * num_acc_n * m_repeat
                acc_up = (
                    [acc_init] * num_acc_n * m_repeat if not single_b_pipe else None
                )

                k1 = k_base_idx + arith.constant(tile_k, index=True)
                rocdl.sched_barrier(0)
                if const_expr(use_async_copy):
                    prefetch_x_to_lds(k1, lds_x_ping)
                else:
                    x_regs_prime = load_x_tile(k1)
                    store_x_tile_to_lds(x_regs_prime, lds_x_ping)

                k0_b = k_base_idx // arith.constant(b_byte_div, index=True)
                gate_w0, up_w0 = load_b_tile(k0_b)
                if const_expr(use_async_copy):
                    rocdl.s_waitcnt(0)
                gpu.barrier()
                rocdl.sched_barrier(0)
                a_tile_pong = prefetch_full_a_from_lds(lds_x_pong)

                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(6)

                num_k_tiles_py = int(klen) // int(tile_k)
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

                        (
                            a_tile_ping,
                            gate_w_ping,
                            up_w_ping,
                            a_scale_ping,
                            gate_bs_ping,
                            up_bs_ping,
                            acc_gate,
                            acc_up,
                        ) = interleaved_half(
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

                        (
                            a_tile_pong,
                            gate_w_pong,
                            up_w_pong,
                            a_scale_pong,
                            gate_bs_pong,
                            up_bs_pong,
                            acc_gate,
                            acc_up,
                        ) = interleaved_half(
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
                        ku_count=tail_ku if pad_ku_skip > 0 else k_unroll,
                    )
                else:
                    k_tail_rel = arith.constant(klen - tile_k, index=True)
                    k_tail1 = k_base_idx + k_tail_rel
                    x_regs_ping = []
                    if const_expr(use_async_copy):
                        prefetch_x_to_lds(k_tail1, lds_x_ping)
                    else:
                        x_regs_ping = load_x_tile(k_tail1)
                    if const_expr(pad_ku_skip > 0):
                        gate_w_ping, up_w_ping = load_b_tile(
                            k_tail1 // arith.constant(b_byte_div, index=True),
                            ku_limit=tail_ku,
                        )
                        a_scale_ping, gate_bs_ping, up_bs_ping = prefetch_ab_scale_tile(
                            k_tail1 // arith.constant(pack_K * 128, index=True),
                            ku_packed_limit=tail_ku_packed,
                        )
                    else:
                        gate_w_ping, up_w_ping = load_b_tile(
                            k_tail1 // arith.constant(b_byte_div, index=True)
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
                    if const_expr(not use_async_copy):
                        store_x_tile_to_lds(x_regs_ping, lds_x_ping)
                    rocdl.s_waitcnt(0)
                    barrier()
                    if const_expr(pad_ku_skip > 0):
                        a_tile_ping = prefetch_full_a_from_lds(
                            lds_x_ping, ku_limit=tail_ku
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
                        ku_count=tail_ku if pad_ku_skip > 0 else k_unroll,
                    )

                bias_pf = None
                if const_expr(epilogue_pf is not None):
                    _, _, bias_pf = epilogue_pf

                def silu_elem(g):
                    """silu(x) = x * sigmoid(x); HW fast path: exp2, rcp"""
                    neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                    t = g * neg_log2e
                    emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                    one = arith.constant(1.0, type=f32)
                    den = one + emu
                    sig = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [den], [], [])
                    return g * sig

                def silu_mul_vec4(gate_v4, up_v4):
                    """Element-wise silu(gate) * up on vec4_f32.
                    When swiglu_limit is set, clamp gate <= limit and
                    -limit <= up <= limit before applying silu(gate) * up.
                    """
                    result_elems = []
                    if const_expr(swiglu_limit is not None):
                        limit = arith.constant(float(swiglu_limit), type=f32)
                        neg_limit = arith.constant(-float(swiglu_limit), type=f32)
                    for ei in range_constexpr(4):
                        g = vector.extract(
                            gate_v4, static_position=[ei], dynamic_position=[]
                        )
                        u = vector.extract(
                            up_v4, static_position=[ei], dynamic_position=[]
                        )
                        if const_expr(swiglu_limit is not None):
                            g = arith.minimumf(g, limit)
                            u = arith.minimumf(u, limit)
                            u = arith.maximumf(u, neg_limit)
                        result_elems.append(silu_elem(g) * u)
                    return vector.from_elements(vec4_f32, result_elems)

                def swiglu_mul_vec4(gate_v4, up_v4):
                    """Element-wise swiglu(gate, up) on vec4_f32.
                    swiglu(g, u) = g * sigmoid(alpha * g) * (u + 1)
                    When swiglu_limit is set, clamp gate <= limit and
                    -limit <= up <= limit before the activation.
                    """
                    result_elems = []
                    alpha = arith.constant(1.702, type=f32)
                    one = arith.constant(1.0, type=f32)
                    neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                    if const_expr(swiglu_limit is not None):
                        limit = arith.constant(float(swiglu_limit), type=f32)
                        neg_limit = arith.constant(-float(swiglu_limit), type=f32)
                    else:
                        limit = arith.constant(float(7.0), type=f32)
                        neg_limit = arith.constant(-float(7.0), type=f32)

                    for ei in range_constexpr(4):
                        g = vector.extract(
                            gate_v4, static_position=[ei], dynamic_position=[]
                        )
                        u = vector.extract(
                            up_v4, static_position=[ei], dynamic_position=[]
                        )
                        g = arith.minimumf(g, limit)
                        u = arith.minimumf(u, limit)
                        u = arith.maximumf(u, neg_limit)
                        t = g * alpha * neg_log2e
                        emu = llvm.call_intrinsic(
                            f32, "llvm.amdgcn.exp2.f32", [t], [], []
                        )
                        den = one + emu
                        sig = llvm.call_intrinsic(
                            f32, "llvm.amdgcn.rcp.f32", [den], [], []
                        )
                        result_elems.append(g * sig * (u + one))
                    return vector.from_elements(vec4_f32, result_elems)

                def act_vec4(gate_v4, up_v4):
                    """Dispatch activation based on `act` parameter."""
                    if const_expr(act == "swiglu"):
                        return swiglu_mul_vec4(gate_v4, up_v4)
                    else:
                        return silu_mul_vec4(gate_v4, up_v4)

                def act_elem(g, u):
                    """Scalar activation, byte-identical per-element to _act_vec4.
                    Used by the fused k-split epilogue (operates on summed f32
                    gate/up scalars in the CShuffle read phase)."""
                    if const_expr(act == "swiglu"):
                        alpha = arith.constant(1.702, type=f32)
                        one = arith.constant(1.0, type=f32)
                        neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                        lim = arith.constant(
                            float(swiglu_limit) if swiglu_limit != 0 else 7.0, type=f32
                        )
                        nlim = arith.constant(
                            -float(swiglu_limit) if swiglu_limit != 0 else -7.0,
                            type=f32,
                        )
                        g = arith.minimumf(g, lim)
                        u = arith.minimumf(u, lim)
                        u = arith.maximumf(u, nlim)
                        t = g * alpha * neg_log2e
                        emu = llvm.call_intrinsic(
                            f32, "llvm.amdgcn.exp2.f32", [t], [], []
                        )
                        den = one + emu
                        sig = llvm.call_intrinsic(
                            f32, "llvm.amdgcn.rcp.f32", [den], [], []
                        )
                        return g * sig * (u + one)
                    else:
                        if const_expr(swiglu_limit != 0):
                            lim = arith.constant(float(swiglu_limit), type=f32)
                            nlim = arith.constant(-float(swiglu_limit), type=f32)
                            g = arith.minimumf(g, lim)
                            u = arith.minimumf(u, lim)
                            u = arith.maximumf(u, nlim)
                        return silu_elem(g) * u

                kwave_fused = const_expr(
                    k_wave > 1
                    and not enable_bias
                    and not is_splitk
                    and not gate_up_interleave
                    and need_quant
                )

                if const_expr(k_wave > 1 and not kwave_fused):
                    has_up = const_expr(acc_up is not None)
                    nm = num_acc_n * m_repeat
                    grp_stride = 64 * nm
                    scr_ty = _mT.memref(
                        num_waves_total * grp_stride * 4, f32, memory_space=_lds_space()
                    )
                    scr_g = memref.view(
                        scr_ty,
                        base_ptr_pong,
                        arith.constant(lds_pong_offset, index=True),
                        sizes=[],
                    )
                    if const_expr(has_up):
                        scr_u = memref.view(
                            scr_ty,
                            base_ptr_ping,
                            arith.constant(lds_ping_offset, index=True),
                            sizes=[],
                        )
                    c_gs = arith.constant(grp_stride, index=True)
                    c4 = arith.constant(4, index=True)
                    c64 = arith.constant(64, index=True)
                    my_base = wave_id * c_gs + lane_id
                    gpu.barrier()
                    for ai in range_constexpr(nm):
                        sidx = (my_base + arith.constant(ai, index=True) * c64) * c4
                        vector.store(acc_gate[ai], scr_g, [sidx], alignment=16)
                        if const_expr(has_up):
                            vector.store(acc_up[ai], scr_u, [sidx], alignment=16)
                    gpu.barrier()
                    for ai in range_constexpr(nm):
                        ai_off = arith.constant(ai, index=True) * c64 + lane_id
                        gvs = []
                        uvs = []
                        for g in range_constexpr(k_wave):
                            peer = (
                                arith.constant(g * num_n_waves, index=True) + wave_n_id
                            )
                            pidx = (peer * c_gs + ai_off) * c4
                            gvs.append(vector.load_op(vec4_f32, scr_g, [pidx]))
                            if const_expr(has_up):
                                uvs.append(vector.load_op(vec4_f32, scr_u, [pidx]))

                        sg = gvs[0]
                        for g in range_constexpr(1, k_wave):
                            sg = arith.addf(sg, gvs[g])
                        acc_gate[ai] = sg
                        if const_expr(has_up):
                            su = uvs[0]
                            for g in range_constexpr(1, k_wave):
                                su = arith.addf(su, uvs[g])
                            acc_up[ai] = su
                    # No trailing barrier: CShuffle's leading barrier already gates

                if const_expr(enable_bias and not is_splitk):
                    bias_up_vals = None
                    if const_expr(bias_pf is not None):
                        if const_expr(gate_up_interleave):
                            bias_gate_vals = bias_pf
                        else:
                            bias_gate_vals, bias_up_vals = bias_pf
                    else:
                        bias_gate_vals = []
                        for ni in range_constexpr(num_acc_n):
                            if const_expr(gate_up_interleave):
                                logical_col = (
                                    (by_n + n_tile_base)
                                    // arith.constant(2, index=True)
                                    + arith.constant((ni // 2) * 16, index=True)
                                    + lane_mod_16
                                )
                                up_off = (
                                    inter_idx
                                    if (ni % 2 == 1)
                                    else arith.constant(0, index=True)
                                )
                                bias_off = expert_off_idx + up_off + logical_col
                            else:
                                bn = (
                                    by_n
                                    + n_tile_base
                                    + arith.constant(ni * 16, index=True)
                                    + lane_mod_16
                                )
                                bias_off = expert_off_idx + bn
                            bias_gate_vals.append(load_bias_scalar(bias_rsrc, bias_off))
                        if const_expr(not (mock_gate_only or gate_up_interleave)):
                            bias_up_vals = []
                            for ni in range_constexpr(num_acc_n):
                                bn = (
                                    by_n
                                    + n_tile_base
                                    + arith.constant(ni * 16, index=True)
                                    + lane_mod_16
                                )
                                bias_up_vals.append(
                                    load_bias_scalar(
                                        bias_rsrc, expert_off_idx + inter_idx + bn
                                    )
                                )
                    for mi in range_constexpr(m_repeat):
                        for ni in range_constexpr(num_acc_n):
                            aidx = mi * num_acc_n + ni
                            bsplat = vector.from_elements(
                                vec4_f32, [bias_gate_vals[ni]] * 4
                            )
                            acc_gate[aidx] = arith.addf(acc_gate[aidx], bsplat)

                    if const_expr(not (mock_gate_only or gate_up_interleave)):
                        for mi in range_constexpr(m_repeat):
                            for ni in range_constexpr(num_acc_n):
                                aidx = mi * num_acc_n + ni
                                bsplat = vector.from_elements(
                                    vec4_f32, [bias_up_vals[ni]] * 4
                                )
                                acc_up[aidx] = arith.addf(acc_up[aidx], bsplat)

                if const_expr(gate_up_interleave and not is_splitk):
                    gui_out_n = num_acc_n // pack_N
                    acc = [None] * (gui_out_n * m_repeat)
                    for mi in range_constexpr(m_repeat):
                        for ni in range_constexpr(gui_out_n):
                            g_idx = mi * num_acc_n + ni * pack_N
                            u_idx = g_idx + 1
                            out_idx = mi * gui_out_n + ni
                            acc[out_idx] = act_vec4(acc_gate[g_idx], acc_gate[u_idx])
                elif const_expr(not is_splitk and not kwave_fused):
                    acc = [None] * (int(num_acc_n) * int(m_repeat))
                    for mi in range_constexpr(m_repeat):
                        for ni in range_constexpr(num_acc_n):
                            aidx = mi * num_acc_n + ni
                            acc[aidx] = act_vec4(acc_gate[aidx], acc_up[aidx])

                tw_pf = None
                bias_pf = None
                if const_expr(epilogue_pf is not None):
                    _, tw_pf, bias_pf = epilogue_pf

                mask24_i32 = arith.constant(0xFFFFFF)
                topk_i32_v = topk_i32
                tokens_i32_v = tokens_i32

                out_base_i64 = arith.index_cast(T.i64, fx.ptrtoint(arg_out))
                out_base_idx = arith.index_cast(ir.IndexType.get(), out_base_i64)

                if const_expr(lds_out is None):
                    raise RuntimeError("CShuffle epilogue requires lds_out")

                apply_weight = doweight_stage1 and not is_splitk

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
                    if const_expr(apply_weight):
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
                        v = vector.extract(
                            acc[acc_idx], static_position=[ii], dynamic_position=[]
                        )
                        if const_expr(apply_weight):
                            v = v * tw
                        if const_expr(need_quant):
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

                out_row_stride = (
                    inter_dim * 2 * out_elem_bytes
                    if is_splitk
                    else (
                        inter_dim // 2
                        if need_fp4
                        else (inter_dim if need_fp8 else inter_dim * out_elem_bytes)
                    )
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
                        out_row_stride, index=True
                    )
                    return ((fused2, row_byte_base), row_valid)

                def idx_to_llvm_ptr(idx_val, addr_space=1):
                    idx_v = idx_val._value if hasattr(idx_val, "_value") else idx_val
                    i64_v = arith.index_cast(T.i64, idx_v)
                    i64_raw = i64_v._value if hasattr(i64_v, "_value") else i64_v
                    ptr_ty = ir.Type.parse(f"!llvm.ptr<{addr_space}>")
                    return llvm.inttoptr(ptr_ty, i64_raw)

                e_vec = e_vec_s1
                e_vec_sk = 2
                cshuffle_nlane = min(32, tile_n // e_vec)
                cshuffle_nlane_sk = min(32, tile_n // e_vec_sk)

                c0_i32 = arith.constant(0, type=T.i32)
                c1_i32 = arith.constant(1, type=T.i32)
                c2_i32 = arith.constant(2, type=T.i32)
                c3_i32 = arith.constant(3, type=T.i32)
                c4_i32 = arith.constant(4, type=T.i32)
                c5_i32 = arith.constant(5, type=T.i32)
                c15_i32 = arith.constant(15, type=T.i32)
                c22_i32 = arith.constant(22, type=T.i32)
                c23_i32 = arith.constant(23, type=T.i32)
                c28_i32 = arith.constant(28, type=T.i32)
                c31_i32 = arith.constant(31, type=T.i32)
                c32_i32 = arith.constant(32, type=T.i32)
                c64_i32 = arith.constant(64, type=T.i32)
                c254_i32 = arith.constant(254, type=T.i32)
                c256_i32 = arith.constant(256, type=T.i32)
                c0xFF800000_i32 = arith.constant(0xFF800000, type=T.i32)
                c0x400000_i32 = arith.constant(0x400000, type=T.i32)
                c0x7FFFFFFF_i32 = arith.constant(0x7FFFFFFF, type=T.i32)
                c0x80000000_i32 = arith.constant(0x80000000, type=T.i32)
                c0x3F800000_i32 = arith.constant(0x3F800000, type=T.i32)
                c0x40C00000_i32 = arith.constant(0x40C00000, type=T.i32)
                c0x4A800000_i32 = arith.constant(0x4A800000, type=T.i32)
                c0xC11FFFFF_i32 = arith.constant(0xC11FFFFF, type=T.i32)
                c0x7_i32 = arith.constant(0x7, type=T.i32)
                c0_f32 = arith.constant(0.0, type=T.f32)

                fp_headroom = 2 if need_fp4 else (8 if need_fp8 else 0)
                c_headroom_i32 = arith.constant(fp_headroom, type=T.i32)

                def f32_to_e2m1(qx_f32):
                    """Convert a scaled f32 value to fp4 (e2m1) 4-bit integer."""
                    qx = qx_f32.bitcast(T.i32)
                    s = qx & c0x80000000_i32
                    qx_abs = qx & c0x7FFFFFFF_i32
                    denormal_mask = arith.cmpi(
                        CmpIPredicate.ult, qx_abs, c0x3F800000_i32
                    )
                    normal_mask = arith.andi(
                        arith.cmpi(CmpIPredicate.ult, qx_abs, c0x40C00000_i32),
                        arith.cmpi(CmpIPredicate.uge, qx_abs, c0x3F800000_i32),
                    )

                    denorm_f32 = qx_abs.bitcast(T.f32) + c0x4A800000_i32.bitcast(T.f32)
                    denormal_x = denorm_f32.bitcast(T.i32) - c0x4A800000_i32

                    mant_odd = (qx_abs >> c22_i32) & c1_i32
                    normal_x = qx_abs + c0xC11FFFFF_i32 + mant_odd
                    normal_x = normal_x >> c22_i32

                    e2m1 = arith.select(normal_mask, normal_x, c0x7_i32)
                    e2m1 = arith.select(denormal_mask, denormal_x, e2m1)
                    return (s >> c28_i32) | e2m1

                if const_expr(need_sort):
                    n32_sort = sorted_scale_cols_i32 * c32_i32

                sk_n_offset = [0]

                def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                    fused, row_byte_base = row_ctx
                    if const_expr(need_quant and not is_splitk):
                        frag_vals = []
                        for i in range_constexpr(e_vec):
                            frag_vals.append(
                                vector.extract(
                                    frag, static_position=[i], dynamic_position=[]
                                )
                            )

                        local_max = c0_f32
                        for i in range_constexpr(e_vec):
                            abs_v = llvm.call_intrinsic(
                                f32, "llvm.fabs.f32", [frag_vals[i]], [], []
                            )
                            local_max = arith.maximumf(local_max, abs_v)

                        for si in range_constexpr(num_shuffle_steps_s1):
                            off = arith.constant(shuffle_dists_s1[si], type=T.i32)
                            peer = local_max.shuffle_xor(off, c64_i32)
                            local_max = arith.maximumf(local_max, peer)

                        max_i32 = local_max.bitcast(T.i32)
                        max_rounded = (max_i32 + c0x400000_i32) & c0xFF800000_i32
                        exp_field = max_rounded >> c23_i32
                        e8m0_biased = arith.maxsi(exp_field - c_headroom_i32, c0_i32)

                        quant_exp = c254_i32 - e8m0_biased
                        quant_scale = (quant_exp << c23_i32).bitcast(T.f32)

                        if const_expr(need_fp4):
                            fp4_vals = []
                            for i in range_constexpr(e_vec):
                                scaled_v = frag_vals[i] * quant_scale
                                fp4_vals.append(f32_to_e2m1(scaled_v))

                            packed_i32 = fp4_vals[0] | (fp4_vals[1] << c4_i32)
                            for k in range_constexpr(1, e_vec // 2):
                                byte_k = fp4_vals[2 * k] | (
                                    fp4_vals[2 * k + 1] << c4_i32
                                )
                                packed_i32 = packed_i32 | (
                                    byte_k << arith.constant(k * 8, type=T.i32)
                                )

                            ptr_addr_idx = row_byte_base + col_g0 / arith.constant(
                                2, index=True
                            )
                            out_ptr_v = idx_to_llvm_ptr(ptr_addr_idx)
                            pack_bytes = e_vec // 2
                            if const_expr(pack_bytes == 1):
                                store_val = arith.TruncIOp(T.i8, packed_i32)
                                store_raw = (
                                    store_val._value
                                    if hasattr(store_val, "_value")
                                    else store_val
                                )
                                llvm.StoreOp(
                                    store_raw, out_ptr_v, alignment=1, nontemporal=True
                                )
                            elif const_expr(pack_bytes == 2):
                                store_val = arith.TruncIOp(T.i16, packed_i32)
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

                        elif const_expr(need_fp8):
                            scaled_vals = []
                            for i in range_constexpr(e_vec):
                                scaled_vals.append(frag_vals[i] * quant_scale)

                            ptr_addr_idx = row_byte_base + col_g0
                            if const_expr(e_vec <= 4):
                                packed_i32 = c0_i32
                                for w in range_constexpr(e_vec // 2):
                                    packed_i32 = rocdl.cvt_pk_fp8_f32(
                                        T.i32,
                                        scaled_vals[2 * w],
                                        scaled_vals[2 * w + 1],
                                        packed_i32,
                                        w,
                                    )
                                out_ptr_v = idx_to_llvm_ptr(ptr_addr_idx)
                                if const_expr(e_vec == 2):
                                    store_val = arith.TruncIOp(T.i16, packed_i32)
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
                                for wg in range_constexpr(e_vec // 4):
                                    b = wg * 4
                                    packed_w = c0_i32
                                    packed_w = rocdl.cvt_pk_fp8_f32(
                                        T.i32,
                                        scaled_vals[b],
                                        scaled_vals[b + 1],
                                        packed_w,
                                        0,
                                    )
                                    packed_w = rocdl.cvt_pk_fp8_f32(
                                        T.i32,
                                        scaled_vals[b + 2],
                                        scaled_vals[b + 3],
                                        packed_w,
                                        1,
                                    )
                                    word_ptr = ptr_addr_idx + arith.constant(
                                        wg * 4, index=True
                                    )
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

                        if const_expr(need_sort):
                            col_g0_i32 = arith.index_cast(T.i32, col_g0)
                            is_scale_writer = arith.cmpi(
                                CmpIPredicate.eq, col_g0_i32 & c31_i32, c0_i32
                            )
                            if_scale = scf.IfOp(is_scale_writer)
                            with ir.InsertionPoint(if_scale.then_block):
                                row_i32_s = arith.index_cast(T.i32, row)
                                col_s_i32 = col_g0_i32 >> c5_i32
                                d0 = row_i32_s >> c5_i32
                                d1 = (row_i32_s >> c4_i32) & c1_i32
                                d2 = row_i32_s & c15_i32
                                d3 = col_s_i32 >> c3_i32
                                d4 = (col_s_i32 >> c2_i32) & c1_i32
                                d5 = col_s_i32 & c3_i32
                                byte_off = (
                                    d0 * n32_sort
                                    + d3 * c256_i32
                                    + d5 * c64_i32
                                    + d2 * c4_i32
                                    + d4 * c2_i32
                                    + d1
                                )
                                e8m0_i8 = arith.TruncIOp(T.i8, e8m0_biased)
                                buffer_ops.buffer_store(
                                    e8m0_i8,
                                    sorted_scale_rsrc,
                                    byte_off,
                                    offset_is_bytes=True,
                                )
                                scf.YieldOp([])
                    elif const_expr(is_splitk):
                        col_idx = col_g0 + arith.constant(sk_n_offset[0], index=True)
                        byte_off_col = col_idx * arith.constant(
                            out_elem_bytes, index=True
                        )
                        ptr_addr_idx = row_byte_base + byte_off_col
                        out_ptr_v = idx_to_llvm_ptr(ptr_addr_idx)
                        frag_v = frag._value if hasattr(frag, "_value") else frag
                        llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.fadd,
                            out_ptr_v,
                            frag_v,
                            llvm.AtomicOrdering.monotonic,
                            syncscope="agent",
                            alignment=e_vec_sk * out_elem_bytes,
                        )
                    else:
                        col_idx = col_g0
                        byte_off_col = col_idx * arith.constant(
                            out_elem_bytes, index=True
                        )
                        ptr_addr_idx = row_byte_base + byte_off_col
                        out_ptr_v = idx_to_llvm_ptr(ptr_addr_idx)
                        frag_v = frag._value if hasattr(frag, "_value") else frag
                        llvm.StoreOp(
                            frag_v,
                            out_ptr_v,
                            alignment=e_vec * out_elem_bytes,
                            nontemporal=True,
                        )

                frag_elem = (
                    ir.F32Type.get()
                    if need_quant
                    else (ir.BF16Type.get() if out_is_bf16 else ir.F16Type.get())
                )

                if const_expr(kwave_fused):
                    slab_n = tile_m * tile_n
                    slab_ty = _mT.memref(
                        k_wave * slab_n, f32, memory_space=_lds_space()
                    )
                    gate_slab = memref.view(
                        slab_ty,
                        base_ptr_pong,
                        arith.constant(lds_pong_offset, index=True),
                        sizes=[],
                    )
                    up_slab = memref.view(
                        slab_ty,
                        base_ptr_ping,
                        arith.constant(lds_ping_offset, index=True),
                        sizes=[],
                    )
                    c_tn = arith.constant(tile_n, index=True)
                    c_slabn = arith.constant(slab_n, index=True)
                    kg_base = wave_k_id * c_slabn
                    vec1_f32 = T.vec(1, f32)
                    vecev_f32 = T.vec(e_vec, f32)

                    gpu.barrier()

                    def fused_write(mi, ii, row_in_tile, row):
                        rb = row_in_tile * c_tn
                        for ni in range_constexpr(num_acc_n):
                            col = (
                                n_tile_base
                                + lane_mod_16
                                + arith.constant(ni * 16, index=True)
                            )
                            aidx = mi * num_acc_n + ni
                            gv = vector.extract(
                                acc_gate[aidx],
                                static_position=[ii],
                                dynamic_position=[],
                            )
                            uv = vector.extract(
                                acc_up[aidx], static_position=[ii], dynamic_position=[]
                            )
                            idx = kg_base + rb + col
                            vector.store(
                                vector.from_elements(vec1_f32, [gv]),
                                gate_slab,
                                [idx],
                                alignment=4,
                            )
                            vector.store(
                                vector.from_elements(vec1_f32, [uv]),
                                up_slab,
                                [idx],
                                alignment=4,
                            )

                    default_epilog(
                        arith=arith,
                        range_constexpr=range_constexpr,
                        m_repeat=m_repeat,
                        lane_div_16=lane_div_16,
                        bx_m=bx_m,
                        body_row=fused_write,
                    )
                    gpu.barrier()

                    cn = int(cshuffle_nlane)
                    cm = int(total_threads) // cn
                    mreps = int(tile_m) // cm
                    nreps = int(tile_n) // (cn * int(e_vec))
                    c_cn = arith.constant(cn, index=True)
                    c_ev = arith.constant(e_vec, index=True)
                    m_lane = tx / c_cn
                    n_lane = tx % c_cn
                    for mr in range_constexpr(mreps):
                        _row_local = arith.constant(mr * cm, index=True) + m_lane
                        row = bx_m + _row_local
                        # Unpack unconditionally (a Python `if` becomes scf.if and loses the binding).
                        rc, rp = precompute_row(row_local=_row_local, row=row)

                        def fused_read(_row_local=_row_local, row=row, rc=rc):
                            rb = _row_local * c_tn
                            for nr in range_constexpr(nreps):
                                cp0 = (
                                    arith.constant(nr * (cn * int(e_vec)), index=True)
                                    + n_lane * c_ev
                                )
                                base = rb + cp0
                                gsum = [None] * int(e_vec)
                                usum = [None] * int(e_vec)
                                for kg in range_constexpr(k_wave):
                                    ko = arith.constant(kg, index=True) * c_slabn + base
                                    gvv = vector.load_op(vecev_f32, gate_slab, [ko])
                                    uvv = vector.load_op(vecev_f32, up_slab, [ko])
                                    for e in range_constexpr(int(e_vec)):
                                        ge = vector.extract(
                                            gvv,
                                            static_position=[e],
                                            dynamic_position=[],
                                        )
                                        ue = vector.extract(
                                            uvv,
                                            static_position=[e],
                                            dynamic_position=[],
                                        )
                                        if kg == 0:
                                            gsum[e] = ge
                                            usum[e] = ue
                                        else:
                                            gsum[e] = arith.addf(gsum[e], ge)
                                            usum[e] = arith.addf(usum[e], ue)
                                fe = [
                                    act_elem(gsum[e], usum[e])
                                    for e in range_constexpr(int(e_vec))
                                ]
                                frag = vector.from_elements(vecev_f32, fe)
                                store_pair(
                                    row_local=_row_local,
                                    row=row,
                                    row_ctx=rc,
                                    col_pair0=cp0,
                                    col_g0=by_n + cp0,
                                    frag=frag,
                                )

                        ifr = scf.IfOp(rp)
                        with ir.InsertionPoint(ifr.then_block):
                            fused_read()
                            scf.YieldOp([])
                elif const_expr(gate_up_interleave and not is_splitk):
                    gui_eff_n = gui_out_n
                    gui_tile_n = tile_n // 2
                    gui_cshuffle_nlane = min(32, gui_tile_n // e_vec)
                    gui_by_n = by_n / arith.constant(2, index=True)
                    gui_n_tile_base = n_tile_base / arith.constant(2, index=True)
                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=gui_tile_n,
                        e_vec=e_vec,
                        cshuffle_nlane=gui_cshuffle_nlane,
                        block_size=total_threads,
                        m_repeat=m_repeat,
                        num_acc_n=gui_eff_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=gui_by_n,
                        n_tile_base=gui_n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=frag_elem,
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                    )
                elif const_expr(mock_gate_only or (gate_up_interleave and is_splitk)):
                    eff_e_vec = e_vec_sk
                    acc = acc_gate
                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=eff_e_vec,
                        cshuffle_nlane=cshuffle_nlane_sk,
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
                        frag_elem_type=frag_elem,
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                        lds_out_split=lds_out_B,
                    )
                elif const_expr(is_splitk):
                    eff_e_vec = e_vec_sk

                    acc = acc_gate
                    sk_n_offset[0] = 0
                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=eff_e_vec,
                        cshuffle_nlane=cshuffle_nlane_sk,
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
                        frag_elem_type=frag_elem,
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                        lds_out_split=lds_out_B,
                    )

                    gpu.barrier()

                    acc = acc_up
                    sk_n_offset[0] = inter_dim
                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=eff_e_vec,
                        cshuffle_nlane=cshuffle_nlane_sk,
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
                        frag_elem_type=frag_elem,
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                        lds_out_split=lds_out_B,
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
                        e_vec=e_vec,
                        cshuffle_nlane=cshuffle_nlane,
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
                        frag_elem_type=frag_elem,
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                        lds_out_split=lds_out_B,
                    )

            if_blk = scf.IfOp(blk_valid)
            with ir.InsertionPoint(if_blk.then_block):
                ifexpert_of = scf.IfOp(exp_valid)
                with ir.InsertionPoint(ifexpert_of.then_block):
                    moe_gemm1_body()
                    scf.YieldOp([])
                scf.YieldOp([])

            gpu.barrier()
            scf.YieldOp([])
            for_ip.__exit__(None, None, None)

    cache_tag = (
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
        _ = cache_tag
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()

        inter_dim_pad_total = arith.constant(2 * inter_dim_pad, index=True)
        tile2_pad = 0
        if const_expr(not gate_only):
            tile_k_stage2 = tile_k // 2
            tile2_pad = (
                tile_k_stage2 - (inter_dim - inter_dim_pad) % tile_k_stage2
            ) % tile_k_stage2

        inter_in = arith.index_cast(ir.IndexType.get(), i32_inter_in.ir_value())
        tile_n_index = arith.constant(tile_n, index=True)
        if const_expr(mock_gate_only or gate_up_interleave):
            gx = (
                inter_in - inter_dim_pad_total + tile2_pad + tile_n_index - 1
            ) / tile_n_index
        else:
            gx = (
                (inter_in - inter_dim_pad_total + tile2_pad + 2 * tile_n_index - 1)
                / tile_n_index
                / arith.constant(2, index=True)
            )

        c_pm_l = arith.constant(persist_m, index=True)
        gy = (
            arith.index_cast(ir.IndexType.get(), i32_size_expert_ids_in.ir_value())
            + c_pm_l
            - arith.constant(1, index=True)
        ) / c_pm_l

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
    accumulate: bool = True,
    enable_bias: bool = False,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    persist_m: int = 4,
    sort_block_m: int = 0,
    waves_per_eu: Optional[int] = None,
    use_async_copy: bool = False,
    cu_num_mul: int = 1,
    b_nt: int = 0,
    xcd_swizzle: int = 0,
):
    """Compile stage2 kernel (moe_gemm2): A2 @ W2.T -> [tokens, model_dim], atomic-add."""
    del b_nt
    _sort_block_m = tile_m if sort_block_m <= 0 else sort_block_m
    if const_expr(_sort_block_m != tile_m and _sort_block_m % tile_m != 0):
        raise ValueError(
            f"sort_block_m ({_sort_block_m}) must be a multiple of tile_m ({tile_m})"
        )

    r139_xdma_first = bool(
        use_async_copy
        and tile_m == 64
        and tile_n == 128
        and tile_k == 256
        and a_dtype == "fp4"
        and b_dtype == "fp4"
        and bool(accumulate)
    )

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")

    if const_expr(a_dtype not in ("fp8", "fp4")):
        raise ValueError(f"a_dtype must be one of ('fp8','fp4'), got {a_dtype!r}")
    if const_expr(b_dtype not in ("fp8", "fp4")):
        raise ValueError(f"b_dtype must be one of ('fp8','fp4'), got {b_dtype!r}")

    is_f8_a = a_dtype == "fp8"
    is_f4_a = a_dtype == "fp4"
    is_f4_b = b_dtype == "fp4"
    is_f8_b = b_dtype == "fp8"

    scale_pack_m = 2
    scale_pack_n = 2
    scale_pack_k = 2
    pack_M = min(scale_pack_m, tile_m // 16)
    pack_N = min(scale_pack_n, tile_n // 64)
    k_unroll_raw = int(tile_k) // 128
    pack_K = min(scale_pack_k, k_unroll_raw)

    elem_bytes = 1

    a_elem_bytes = 1
    b_elem_bytes = 1
    tile_k_bytes = int(tile_k) * int(a_elem_bytes)

    a_elem_vec_pack = 2 if is_f4_a else 1
    cbsz = 0 if is_f8_a else 4
    blgp = 0 if is_f8_b else 4
    b_byte_div = 2 if is_f4_b else 1
    b_cells_per_ku = 2 if is_f8_b else 1

    b_kpack_bytes_s = 16
    b_kpack_elems_s = b_kpack_bytes_s // b_elem_bytes
    b_c_k_s = inter_dim // b_byte_div
    b_c_k0_s = (b_c_k_s * b_elem_bytes) // 64
    b_stride_nlane = b_kpack_elems_s
    b_stride_klane = 16 * b_stride_nlane
    b_stride_k0 = 4 * b_stride_klane
    b_stride_n0 = b_c_k0_s * b_stride_k0
    assert model_dim % 16 == 0, "model_dim must be divisible by 16"
    expert_b_stride = (model_dim // 16) * b_stride_n0

    if const_expr((tile_k_bytes % 64) != 0):
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={a_elem_bytes})"
        )

    out_s = str(out_dtype).strip().lower()
    if const_expr(
        out_s not in ("f16", "fp16", "half", "bf16", "bfloat16", "f32", "fp32", "float")
    ):
        raise ValueError(
            f"out_dtype must be 'f16', 'bf16', or 'f32', got {out_dtype!r}"
        )
    out_is_f32 = out_s in ("f32", "fp32", "float")
    out_is_bf16 = out_s in ("bf16", "bfloat16")
    if const_expr((not bool(accumulate)) and out_is_f32):
        raise ValueError(
            "compile_moe_gemm2(accumulate=False) only supports out_dtype in {'f16','bf16'}"
        )
    w_elem_bytes = 1
    w_elem_pack = 2 if is_f4_b else 1
    w_nbytes = (experts * model_dim * inter_dim * w_elem_bytes) // w_elem_pack
    # #3476: host e8m0_shuffle pads scale group-N up to a multiple of 8, i.e.
    # 128- but not 256-aligned (e.g. 384) read OOB scales -> garbage e8m0 -> NaN.
    scale_k_padded = (inter_dim + 255) // 256 * 256
    scale_kblk_padded = scale_k_padded // 32
    bias_nbytes = experts * model_dim * 4

    def x_elem_type():
        if const_expr(is_f4_b):
            return T.f8 if is_f8_a else T.i8
        return T.f8

    def w_elem_type():
        if const_expr(is_f4_b):
            return T.i8
        return T.f8

    def scale_elem_type():
        return T.i32

    total_threads = 256
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(a_elem_bytes)
    if const_expr(bytes_x_per_tile % total_threads != 0):
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={a_elem_bytes}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads

    lds_stride = tile_k

    if const_expr(out_is_f32):
        _use_cshuffle_epilog = bool(use_cshuffle_epilog)
        if const_expr(_use_cshuffle_epilog):
            raise ValueError(
                "out_dtype='f32' does not support CShuffle epilogue (set use_cshuffle_epilog=False)."
            )
    else:
        _use_cshuffle_epilog = True

    def out_elem():
        return T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)

    def load_bias_scalar(bias_rsrc, offset):
        return buffer_ops.buffer_load(bias_rsrc, offset, vec_width=1, dtype=T.f32)

    epilog_tag = "cshuffle"
    persistent = persist_m <= 0
    if const_expr(not isinstance(cu_num_mul, int) or cu_num_mul < 1):
        raise ValueError(f"cu_num_mul must be int >= 1, got {cu_num_mul}")
    if const_expr(persistent):
        from aiter.jit.utils.chip_info import get_cu_num

        cu_num = get_cu_num() * int(cu_num_mul)
    else:
        cu_num = 0
    sbm_tag = "" if _sort_block_m == tile_m else f"_sbm{_sort_block_m}"
    pm_tag = f"_persist_cu{cu_num}" if persistent else f"_pm{persist_m}"
    wpe_tag = f"_w{waves_per_eu}" if waves_per_eu is not None else ""
    if const_expr(waves_per_eu is not None and not (1 <= int(waves_per_eu) <= 10)):
        raise ValueError(f"waves_per_eu must be in [1, 10] or None, got {waves_per_eu}")
    num_k_tiles_per_batch = int(inter_dim) // int(tile_k)
    async_tag = "_async" if use_async_copy else ""
    cumul_tag = f"_cumul{int(cu_num_mul)}" if int(cu_num_mul) != 1 else ""
    xcd_tag = f"_xcd{xcd_swizzle}" if xcd_swizzle > 0 else ""
    acc_tag = "" if accumulate else "_acc0"
    module_name = (
        f"mfma_moe2_a{a_dtype}_w{b_dtype}_{out_s}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_vscale_fix3_fp4opt_v1{pm_tag}{sbm_tag}{wpe_tag}{async_tag}{cumul_tag}{xcd_tag}{acc_tag}"
    ).replace("-", "_")
    lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(a_elem_bytes)
    lds_out_bytes = 2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    lds_tid_bytes = int(tile_m) * 4
    lds_tw_bytes = (int(tile_m) * 4) if bool(doweight_stage2) else 0
    lds_total_bytes = max(lds_x_bytes, lds_out_bytes) + lds_tid_bytes + lds_tw_bytes
    lds_total_elems = lds_total_bytes if a_elem_bytes == 1 else (lds_total_bytes // 2)

    def x_lds_elem():
        return T.f8

    lds_alloc_bytes = int(lds_total_elems) * int(a_elem_bytes)
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_alloc_bytes

    if const_expr(True):

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

            tokens_in = arith.index_cast(ir.IndexType.get(), i32_tokens_in.ir_value())
            n_in = arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
            k_in = arith.index_cast(ir.IndexType.get(), i32_k_in.ir_value())
            size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
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

            def ptr_buffer_resource(ptr, num_records_bytes):
                addr = fx.ptrtoint(ptr)
                addr_i64 = arith.index_cast(T.i64, addr)
                return buffer_ops.create_buffer_resource_from_addr(
                    addr_i64, num_records_bytes=num_records_bytes
                )

            acc_init = arith.constant_vector(0.0, vec4_f32)

            topk_idx = arith.constant(topk, index=True)
            m_in = tokens_in * topk_idx

            c_n_total = arith.constant(experts * model_dim, index=True)
            kpack_bytes = 16
            from .layout_utils import _div_pow2, _mod_pow2

            def check_c_n_valid_gate(base_n):
                return arith.cmpi(CmpIPredicate.ult, base_n, model_dim - model_dim_pad)

            def check_c_k_valid_gate(base_k):
                return arith.cmpi(CmpIPredicate.ult, base_k, inter_dim - inter_dim_pad)

            # A&B's scale preshuffle layout.  #3476: host e8m0_shuffle pads the
            c_k_orig = arith.constant(scale_k_padded, index=True)
            layout_a_scale = make_preshuffle_scale_layout(
                arith, c_mn=m_in, c_k=c_k_orig
            )
            layout_b_scale = make_preshuffle_scale_layout(
                arith, c_mn=c_n_total, c_k=c_k_orig
            )

            if const_expr(use_async_copy and a_elem_vec_pack > 1):
                eff_lds_stride = lds_stride // a_elem_vec_pack
                eff_tile_k_bytes = tile_k_bytes // a_elem_vec_pack
            else:
                eff_lds_stride = lds_stride
                eff_tile_k_bytes = tile_k_bytes

            shape_lds = fx.make_shape(tile_m, eff_lds_stride)
            stride_lds = fx.make_stride(eff_lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            by_outer = gpu.block_id("x")
            bx_persist = gpu.block_id("y")

            if const_expr(xcd_swizzle > 0):
                NUM_XCDS_S = 8
                c1_sw = arith.constant(1, index=True)
                c_tn_sw = arith.constant(tile_n, index=True)
                c_mdp_sw = arith.constant(model_dim_pad, index=True)
                gx = (n_in - c_mdp_sw + c_tn_sw - c1_sw) / c_tn_sw
                if const_expr(persistent):
                    gy = arith.constant(cu_num, index=True)
                else:
                    c_pm_sw = arith.constant(persist_m, index=True)
                    gy = (size_expert_ids_in + c_pm_sw - c1_sw) / c_pm_sw

                linear_id = bx_persist * gx + by_outer
                num_wgs = gx * gy

                c_xcds = arith.constant(NUM_XCDS_S, index=True)
                wgs_per_xcd = num_wgs / c_xcds
                wgid = (linear_id % c_xcds) * wgs_per_xcd + (linear_id / c_xcds)

                WGM_S = xcd_swizzle
                c_wgm = arith.constant(WGM_S, index=True)
                num_wgid_in_group = c_wgm * gx
                group_id = wgid / num_wgid_in_group
                first_pid_m = group_id * c_wgm
                remaining_m = gy - first_pid_m
                cmp_m = arith.cmpi(CmpIPredicate.ult, remaining_m, c_wgm)
                group_size_m = arith.select(cmp_m, remaining_m, c_wgm)

                wgid_in_group = wgid % num_wgid_in_group
                bx_persist = first_pid_m + (wgid_in_group % group_size_m)
                by_outer = wgid_in_group / group_size_m

            by = by_outer

            k_blocks16 = arith.constant(eff_tile_k_bytes // 16, index=True)
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

            lds_x_b = 2 * int(tile_m) * int(lds_stride) * int(a_elem_bytes)
            lds_out_b = 2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
            lds_tid_off = max(lds_x_b, lds_out_b)
            lds_tid = SmemPtr(
                base_ptr, lds_x_ptr.byte_offset + lds_tid_off, T.i32, shape=(tile_m,)
            ).get()
            # lds_tw aliases the LDS slot immediately after
            lds_tw_off = lds_tid_off + int(tile_m) * 4
            lds_tw = (
                SmemPtr(
                    base_ptr,
                    lds_x_ptr.byte_offset + lds_tw_off,
                    T.f32,
                    shape=(tile_m,),
                ).get()
                if doweight_stage2
                else None
            )

            c_topk = arith.constant(topk, index=True)

            c_elem_bytes = arith.constant(int(a_elem_bytes), index=True)
            c_a_pack = arith.constant(int(a_elem_vec_pack), index=True)
            x_nbytes_idx = _div_pow2(
                (tokens_in * c_topk) * k_in * c_elem_bytes, int(a_elem_vec_pack)
            )
            x_nbytes_i32 = arith.index_cast(T.i32, x_nbytes_idx)
            x_rsrc = ptr_buffer_resource(arg_x, x_nbytes_i32)

            w_rsrc = ptr_buffer_resource(arg_w, w_nbytes)

            out_elem_bytes = 4 if out_is_f32 else 2
            out_nbytes_idx = (
                tokens_in * n_in * arith.constant(out_elem_bytes, index=True)
            )
            if const_expr(not bool(accumulate)):
                out_nbytes_idx = (
                    tokens_in
                    * arith.index(topk)
                    * n_in
                    * arith.constant(out_elem_bytes, index=True)
                )
            out_nbytes_i32 = arith.index_cast(T.i32, out_nbytes_idx)
            out_rsrc = ptr_buffer_resource(arg_out, out_nbytes_i32)

            numids_rsrc = ptr_buffer_resource(
                arg_num_valid_ids, arith.constant(4, type=T.i32)
            )
            num_valid_i32 = buffer_ops.buffer_load(
                numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=T.i32
            )
            num_valid_i32 = rocdl.ReadfirstlaneOp(T.i32, num_valid_i32).res
            num_valid_idx = arith.index_cast(ir.IndexType.get(), num_valid_i32)

            if const_expr(is_f4_a or is_f8_a):
                # #3476: use 256-padded K/32 to match host scale padding.
                kblk = arith.constant(scale_kblk_padded, index=True)
                sx_nbytes_idx = num_valid_idx * kblk
                sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
                sx_rsrc = ptr_buffer_resource(arg_scale_x, sx_nbytes_i32)
            else:
                sx_nbytes_idx = (tokens_in * c_topk) * arith.constant(4, index=True)
                sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
                sx_rsrc = ptr_buffer_resource(arg_scale_x, sx_nbytes_i32)

            kblk_w = arith.constant(scale_kblk_padded, index=True)
            mn_w = arith.constant(experts * model_dim, index=True)
            sw_nbytes_idx = mn_w * kblk_w
            sw_nbytes_i32 = arith.index_cast(T.i32, sw_nbytes_idx)
            sw_rsrc = ptr_buffer_resource(arg_scale_w, sw_nbytes_i32)

            sorted_nbytes_idx = (
                size_expert_ids_in
                * arith.constant(tile_m, index=True)
                * arith.constant(4, index=True)
            )
            sorted_nbytes_i32 = arith.index_cast(T.i32, sorted_nbytes_idx)
            sorted_rsrc = ptr_buffer_resource(arg_sorted_token_ids, sorted_nbytes_i32)
            sorted_w_rsrc = ptr_buffer_resource(arg_sorted_weights, sorted_nbytes_i32)

            c_sbm = arith.constant(_sort_block_m, index=True)
            c_tm = arith.constant(tile_m, index=True)
            c1 = arith.constant(1, index=True)
            sort_blocks_ub = _div_pow2(
                size_expert_ids_in * c_tm + c_sbm - c1, _sort_block_m
            )
            eid_nbytes_idx = sort_blocks_ub * arith.constant(4, index=True)
            eid_nbytes_i32 = arith.index_cast(T.i32, eid_nbytes_idx)
            expert_rsrc = ptr_buffer_resource(arg_expert_ids, eid_nbytes_i32)
            bias_rsrc = (
                ptr_buffer_resource(arg_bias, bias_nbytes) if enable_bias else None
            )

            c0_p = arith.constant(0, index=True)
            c1_p = arith.constant(1, index=True)

            if const_expr(persistent):
                c_cu = arith.constant(cu_num, index=True)
                c_tm_p = arith.constant(tile_m, index=True)
                _num_valid_idx = arith.index_cast(ir.IndexType.get(), num_valid_i32)
                total_m_tiles = (_num_valid_idx + c_tm_p - c1_p) / c_tm_p
                tiles_per_block_base = total_m_tiles / c_cu
                tiles_remainder = total_m_tiles - (tiles_per_block_base * c_cu)
                has_extra_tile = arith.cmpi(
                    CmpIPredicate.ult, bx_persist, tiles_remainder
                )
                extra_tile = arith.select(has_extra_tile, c1_p, c0_p)
                tiles_per_block = tiles_per_block_base + extra_tile
                start_tail = arith.select(has_extra_tile, bx_persist, tiles_remainder)
                persist_start_tile = bx_persist * tiles_per_block_base + start_tail
                i1 = ir.IntegerType.get_signless(1)
                init_active = arith.constant(1, type=i1)
                for_persist = scf.ForOp(c0_p, tiles_per_block, c1_p, [init_active])
            else:
                c_pm = arith.constant(persist_m, index=True)
                init_prev_expert = arith.constant(0, type=T.i32)
                init_prev_b_base = arith.constant(0, index=True)
                for_persist = scf.ForOp(
                    c0_p,
                    c_pm,
                    c1_p,
                    [init_prev_expert, init_prev_b_base],
                )

            for_ip = ir.InsertionPoint(for_persist.body)
            for_ip.__enter__()
            mi_p = for_persist.induction_variable

            if const_expr(persistent):
                still_active = for_persist.inner_iter_args[0]
                bx = persist_start_tile + mi_p
            else:
                prev_expert_i32 = for_persist.inner_iter_args[0]
                prev_expert_b_base = for_persist.inner_iter_args[1]
                bx = bx_persist * arith.constant(persist_m, index=True) + mi_p

            bx_m = bx * arith.constant(tile_m, index=True)

            bx_m_i32 = arith.index_cast(T.i32, bx_m)
            blk_valid = arith.cmpi(CmpIPredicate.ult, bx_m_i32, num_valid_i32)

            sort_blk = _div_pow2(bx_m, _sort_block_m)
            expert_i32 = buffer_ops.buffer_load(
                expert_rsrc, sort_blk, vec_width=1, dtype=T.i32
            )
            expert_idx = arith.index_cast(T.index, expert_i32)
            exp_valid = arith.cmpi(
                CmpIPredicate.ult, expert_i32, arith.constant(experts, type=T.i32)
            )

            if const_expr(persistent):
                expert_b_base = expert_idx * arith.constant(expert_b_stride, index=True)
            else:
                delta_expert = arith.subi(expert_i32, prev_expert_i32)
                delta_expert_idx = arith.index_cast(ir.IndexType.get(), delta_expert)
                delta_b = delta_expert_idx * arith.constant(expert_b_stride, index=True)
                expert_b_base = prev_expert_b_base + delta_b

            first_tok = buffer_ops.buffer_load(
                sorted_rsrc, bx_m, vec_width=1, dtype=T.i32
            )
            first_tid = arith.andi(first_tok, arith.constant(0xFFFFFF, type=T.i32))
            tokens_i32_guard = arith.index_cast(T.i32, tokens_in)
            tile_has_tokens = arith.cmpi(CmpIPredicate.ult, first_tid, tokens_i32_guard)

            if const_expr(pack_M < scale_pack_m):
                m_off = _mod_pow2(_div_pow2(bx_m, 16), scale_pack_m)
                m_scale_shift_i32 = arith.index_cast(
                    T.i32, m_off * arith.constant(8, index=True)
                )
            else:
                m_scale_shift_i32 = None

            def moe_gemm2_then_body():
                n_idx = arith.constant(model_dim, index=True)
                expert_off_idx = expert_idx * n_idx

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
                chunk_i32 = x_load_bytes // 4
                vec4_i32 = T.vec(4, i32)

                c_k_div4 = _div_pow2(
                    _div_pow2(k_in, int(a_elem_vec_pack))
                    * arith.constant(int(a_elem_bytes), index=True),
                    4,
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
                x_load_vec_elems = (
                    x_load_bytes if a_elem_bytes == 1 else x_load_bytes // a_elem_bytes
                )

                def load_x(idx_i32):
                    """Load `x_load_bytes` bytes from X (gmem) into regs.

                    For 16B, keep the fast dwordx4 path. For 8B/4B, use byte offsets.
                    """
                    if const_expr(x_load_bytes == 16):
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

                if const_expr(use_async_copy and a_elem_vec_pack > 1):
                    dma_bytes_pre = 16
                    eff_bytes_pre = (
                        int(tile_m) * int(eff_lds_stride) * int(a_elem_bytes)
                    )
                    num_x_addr_loads = max(
                        1, eff_bytes_pre // (total_threads * dma_bytes_pre)
                    )
                else:
                    num_x_addr_loads = num_x_loads
                x_row_base_div4 = []
                x_col_local_i32 = []
                x_row_local = []
                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_row_local.append(row_local)
                    x_col_local_i32.append(col_local_i32)

                    if const_expr(i < num_x_addr_loads):
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
                        row_ts_idx = arith.index_cast(T.index, row_ts_i32)

                        x_row_base_div4.append(row_ts_idx * c_k_div4)
                    else:
                        x_row_base_div4.append(arith.index(0))

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

                        if const_expr(x_load_bytes == 16):
                            parts.append(vector.bitcast(vec4_i32, x_vec))
                        elif const_expr(x_load_bytes == 8):
                            parts.append(vector.bitcast(vec2_i32, x_vec))
                        else:
                            parts.append(vector.bitcast(vec1_i32, x_vec))
                    return parts

                coord_wl = idx2crd(tx, layout_tx_wave_lane)
                wave_id = layout_get(coord_wl, 0)
                lane_id = layout_get(coord_wl, 1)
                coord_l16 = idx2crd(lane_id, layout_lane16)
                lane_div_16 = layout_get(coord_l16, 0)
                lane_mod_16 = layout_get(coord_l16, 1)

                row_a_lds = lane_mod_16

                col_offset_base = lane_div_16 * arith.constant(16, index=True)

                num_waves = 4
                n_per_wave = tile_n // num_waves
                num_acc_n = n_per_wave // 16
                c_n_per_wave = arith.constant(n_per_wave, index=True)
                wave_mod_4 = _mod_pow2(wave_id, 4)
                n_tile_base = wave_mod_4 * c_n_per_wave

                by_n = by * arith.constant(tile_n, index=True)

                if const_expr(pack_N < scale_pack_n):
                    global_n_base = expert_off_idx + by_n + n_tile_base
                    n_off = _mod_pow2(_div_pow2(global_n_base, 16), scale_pack_n)
                    n_scale_shift_i32 = arith.index_cast(
                        T.i32, n_off * arith.constant(8, index=True)
                    )
                else:
                    n_scale_shift_i32 = None
                n_intra_list = [None] * num_acc_n
                n_blk_list = [None] * num_acc_n
                for i in range_constexpr(num_acc_n):
                    offset = i * 16
                    c_offset = arith.constant(offset, index=True)
                    global_n = by_n + n_tile_base + c_offset + lane_mod_16
                    n_blk_list[i] = _div_pow2(global_n, 16)
                    n_intra_list[i] = _mod_pow2(global_n, 16)

                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 128

                k_unroll_packed = k_unroll // pack_K
                m_repeat_packed = m_repeat // pack_M
                num_acc_n_packed = num_acc_n // pack_N

                def load_b_packs_k64(
                    base_k, ku: int, ni: int, *, n_blk_p=None, n_intra_p=None
                ):
                    """Load one B MFMA k-step.

                    fp4: single 16B load -> 2x i64 (lower MFMA half).
                    fp8: two adjacent k0 cells (32B) -> 4x i64 (full operand).

                    `n_blk_p` / `n_intra_p` allow callers to override the per-N
                    list. When ``None`` we fall back to the body-level defaults
                    (``n_blk_list`` / ``n_intra_list``).
                    """
                    blk = n_blk_p if n_blk_p is not None else n_blk_list
                    intra = n_intra_p if n_intra_p is not None else n_intra_list
                    base_k_bytes = base_k * arith.constant(
                        int(b_elem_bytes), index=True
                    )
                    k0_base = _div_pow2(base_k_bytes, 64) + arith.constant(
                        ku * b_cells_per_ku, index=True
                    )
                    k1 = lane_div_16
                    vec_elems = kpack_bytes // int(b_elem_bytes)

                    def load_cell(k0):
                        idx_pack = (
                            expert_b_base
                            + blk[ni] * arith.constant(b_stride_n0, index=True)
                            + k0 * arith.constant(b_stride_k0, index=True)
                            + k1 * arith.constant(b_stride_klane, index=True)
                            + intra[ni] * arith.constant(b_stride_nlane, index=True)
                        )
                        b16 = _buffer_load_vec(
                            buffer_ops,
                            vector,
                            w_rsrc,
                            idx_pack,
                            elem_type=w_elem_type(),
                            vec_elems=vec_elems,
                            elem_bytes=b_elem_bytes,
                            offset_in_bytes=(b_elem_bytes == 1),
                        )
                        b_i64x2 = vector.bitcast(vec2_i64, b16)
                        return (
                            vector.extract(
                                b_i64x2, static_position=[0], dynamic_position=[]
                            ),
                            vector.extract(
                                b_i64x2, static_position=[1], dynamic_position=[]
                            ),
                        )

                    b0, b1 = load_cell(k0_base)
                    if const_expr(is_f8_b):
                        b2, b3 = load_cell(k0_base + arith.constant(1, index=True))
                        return b0, b1, b2, b3
                    return b0, b1

                def accum_b_ku(b_tile, base_k, ku, n_blk_p, n_intra_p):
                    """Append one ku's (p0..p3) N-packs to b_tile (p2/p3 fp8-B only)."""
                    packs0, packs1, packs2, packs3 = [], [], [], []
                    for ni in range_constexpr(num_acc_n):
                        b = load_b_packs_k64(
                            base_k,
                            ku,
                            ni,
                            n_blk_p=n_blk_p,
                            n_intra_p=n_intra_p,
                        )
                        packs0.append(b[0])
                        packs1.append(b[1])
                        if const_expr(is_f8_b):
                            packs2.append(b[2])
                            packs3.append(b[3])
                    b_tile.append((packs0, packs1, packs2, packs3))

                def load_b_tile(base_k, *, n_blk_p=None, n_intra_p=None):
                    b_tile = []
                    for ku in range_constexpr(k_unroll):
                        accum_b_ku(b_tile, base_k, ku, n_blk_p, n_intra_p)
                    return b_tile

                b_split_enabled = k_unroll >= 2
                is_prefill_shape = (
                    int(tile_m) == 64
                    and int(tile_n) == 128
                    and int(tile_k) == 256
                    and bool(use_async_copy)
                    and a_elem_vec_pack > 1
                )
                if const_expr(b_split_enabled and is_prefill_shape and k_unroll >= 4):
                    b_split_ku = k_unroll - 1
                else:
                    b_split_ku = k_unroll // 2 if b_split_enabled else k_unroll

                def load_b_tile_lo(base_k, *, n_blk_p=None, n_intra_p=None):
                    """Load first half of B tile (ku < _b_split_ku)."""
                    b_tile = []
                    for ku in range_constexpr(b_split_ku):
                        accum_b_ku(b_tile, base_k, ku, n_blk_p, n_intra_p)
                    return b_tile

                def load_b_tile_hi(base_k, *, n_blk_p=None, n_intra_p=None):
                    """Load second half of B tile (ku >= _b_split_ku)."""
                    b_tile = []
                    for ku in range_constexpr(b_split_ku, k_unroll):
                        accum_b_ku(b_tile, base_k, ku, n_blk_p, n_intra_p)
                    return b_tile

                def load_scale(arg_scale, rsrc, scale_info, ku, mni):
                    k_lane = lane_div_16
                    n_lane = lane_mod_16
                    idx_pack = (
                        mni * scale_info.stride_n0
                        + ku * scale_info.stride_k0
                        + k_lane * scale_info.stride_klane
                        + n_lane
                    )
                    s = buffer_ops.buffer_load(rsrc, idx_pack, vec_width=1, dtype=T.i32)
                    return vector.from_elements(T.vec(1, T.i32), [s])

                def apply_k_shift(scale_vec, k_shift_bits):
                    if const_expr(k_shift_bits > 0):
                        val = vector.extract(
                            scale_vec, static_position=[0], dynamic_position=[]
                        )
                        val = arith.shrui(val, arith.constant(k_shift_bits, type=T.i32))
                        return vector.from_elements(T.vec(1, T.i32), [val])
                    return scale_vec

                def load_b_scale_tile(base_k, k_shift_bits=0, *, by_n_p=None):
                    """Load B-scale tile.

                    `by_n_p` overrides the body-default `by_n`.
                    """
                    byn = by_n_p if by_n_p is not None else by_n
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
                                        expert_off_idx + byn + n_tile_base,
                                        scale_pack_n,
                                    ),
                                    16,
                                ),
                            )
                            scale = apply_k_shift(scale, k_shift_bits)
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
                                mi + _div_pow2(_div_pow2(bx_m, scale_pack_m), 16),
                            )
                            scale = apply_k_shift(scale, k_shift_bits)
                            a_scale_tile.append(scale)
                    return a_scale_tile

                def prefetch_ab_scale_tile(base_k, k_shift_bits=0, *, by_n_p=None):
                    return [
                        load_a_scale_tile(base_k, k_shift_bits),
                        load_b_scale_tile(base_k, k_shift_bits, by_n_p=by_n_p),
                    ]

                vec8_x = T.vec(vec8_elems, x_elem)
                vec4_x_lds = T.vec(vec4_elems, x_elem)

                def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                    for i in range_constexpr(num_x_loads):
                        row_local = x_row_local[i]
                        col_local_i32 = x_col_local_i32[i]
                        if const_expr(x_load_bytes == 16):
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
                        elif const_expr(x_load_bytes == 8):
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
                        else:
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

                if const_expr(use_async_copy):
                    dma_bytes = 16
                    wave_size = 64
                    eff_bytes_per_buffer = (
                        int(tile_m) * int(eff_lds_stride) * int(a_elem_bytes)
                    )
                    num_dma_loads = max(
                        1, eff_bytes_per_buffer // (total_threads * dma_bytes)
                    )
                    c_a_elem_bytes_dma = arith.constant(int(a_elem_bytes), index=True)
                    c_wave_dma_bytes = arith.constant(wave_size * dma_bytes, index=True)

                    def dma_x_tile_to_lds(base_k, lds_base):
                        c4_idx = arith.index(4)
                        base_k_div4 = (
                            (base_k / c_a_pack)
                            * arith.constant(int(elem_bytes), index=True)
                        ) / arith.index(4)

                        lds_ptr_i64 = None
                        for i in range_constexpr(num_dma_loads):
                            row_local_i = x_row_local[i]
                            col_local_i32_i = x_col_local_i32[i]
                            col_local_sw = swizzle_xor16(
                                row_local_i, col_local_i32_i * c4_idx, k_blocks16
                            )
                            row_k_dw = x_row_base_div4[i] + base_k_div4
                            global_byte_idx = row_k_dw * c4_idx + col_local_sw
                            global_offset = arith.index_cast(T.i32, global_byte_idx)

                            if const_expr(i == 0):
                                lds_addr = (
                                    memref.extract_aligned_pointer_as_index(lds_x)
                                    + lds_base * c_a_elem_bytes_dma
                                    + wave_id * c_wave_dma_bytes
                                )
                                lds_ptr_i64 = rocdl.readfirstlane(
                                    T.i64, arith.index_cast(T.i64, lds_addr)
                                )
                            else:
                                lds_ptr_i64 = lds_ptr_i64 + arith.constant(
                                    total_threads * dma_bytes, type=T.i64
                                )

                            lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                            lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64)

                            rocdl.raw_ptr_buffer_load_lds(
                                x_rsrc,
                                lds_ptr,
                                arith.constant(dma_bytes, type=T.i32),
                                global_offset,
                                arith.constant(0, type=T.i32),
                                arith.constant(0, type=T.i32),
                                arith.constant(0, type=T.i32),
                            )

                    def prefetch_x_to_lds(base_k, lds_base):
                        dma_x_tile_to_lds(base_k, lds_base)

                def lds_load_packs_k64(curr_row_a_lds, col_base, lds_base):
                    col_base_swz_bytes = swizzle_xor16(
                        curr_row_a_lds, col_base, k_blocks16
                    )
                    col_base_swz = (
                        col_base_swz_bytes
                        if elem_bytes == 1
                        else (col_base_swz_bytes / arith.index(2))
                    )
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
                    b_hi_loader=None,
                    n_scale_shift_p=None,
                    ku_count=None,
                ):
                    nss = (
                        n_scale_shift_p
                        if n_scale_shift_p is not None
                        else n_scale_shift_i32
                    )
                    # PR3117 stage2 NaN fix: restrict the MFMA K-loop to the valid
                    # to mfma_scale, eliminating 0*NaN propagation into the output.
                    ku_loop = k_unroll if ku_count is None else ku_count
                    if const_expr(b_hi_loader is not None):
                        b_tile_full = [None] * k_unroll
                        for i in range_constexpr(b_split_ku):
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
                                bias.append(load_bias_scalar(bias_rsrc, bias_offset))
                        tw_pf = None
                        if const_expr(doweight_stage2):
                            tw_pf = []
                            lane_div_16_mul4_pf = lane_div_16 * arith.index(4)
                            vec4_f32_pf = T.vec(4, f32)
                            if const_expr(lds_tw is not None):
                                for mi in range_constexpr(m_repeat):
                                    mi_base_pf = arith.constant(mi * 16, index=True)
                                    lds_row_pf = mi_base_pf + lane_div_16_mul4_pf
                                    tw_v4 = vector.load_op(
                                        vec4_f32_pf, lds_tw, [lds_row_pf]
                                    )
                                    for ii in range_constexpr(4):
                                        tw_pf.append(
                                            vector.extract(
                                                tw_v4,
                                                static_position=[ii],
                                                dynamic_position=[],
                                            )
                                        )
                            else:
                                for mi in range_constexpr(m_repeat):
                                    mi_base_pf = arith.constant(mi * 16, index=True)
                                    base_row_pf = (
                                        bx_m + mi_base_pf + lane_div_16_mul4_pf
                                    )
                                    tw_v4 = buffer_ops.buffer_load(
                                        sorted_w_rsrc,
                                        base_row_pf,
                                        vec_width=4,
                                        dtype=f32,
                                    )
                                    for ii in range_constexpr(4):
                                        tw_pf.append(
                                            vector.extract(
                                                tw_v4,
                                                static_position=[ii],
                                                dynamic_position=[],
                                            )
                                        )
                        epilogue_pf = (None, tw_pf, bias)

                    c0_i64 = arith.constant(0, type=T.i64)
                    vec4_i64 = T.vec(4, T.i64)
                    vec8_i32 = T.vec(8, T.i32)

                    def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                        v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                        return vector.bitcast(vec8_i32, v4)

                    pack_K_shift = (pack_K - 1).bit_length()
                    pack_K_mask = pack_K - 1

                    xdl_arb_hint = (
                        int(tile_m) == 64
                        and int(tile_n) == 128
                        and int(tile_k) == 256
                        and bool(use_async_copy)
                        and a_elem_vec_pack > 1
                    )
                    if const_expr(xdl_arb_hint):
                        rocdl.disable_xdl_arb_stall()

                    if const_expr(b_hi_loader is not None):
                        b_hi = b_hi_loader()
                        for bhi_i in range_constexpr(len(b_hi)):
                            b_tile_full[b_split_ku + bhi_i] = b_hi[bhi_i]

                    rocdl.s_setprio(1)

                    for k_idx in range_constexpr(ku_loop):
                        ku128 = k_idx >> pack_K_shift
                        ikxdl = k_idx & pack_K_mask

                        b_packs = b_tile_full[k_idx]
                        b_packs0 = b_packs[0]
                        b_packs1 = b_packs[1]
                        if const_expr(is_f8_b):
                            b_packs2 = b_packs[2]
                            b_packs3 = b_packs[3]

                        col_base = col_offset_base + (k_idx * 128) // a_elem_vec_pack

                        for mi in range_constexpr(m_repeat_packed):
                            a_scale_i32 = a_scale[ku128 * m_repeat_packed + mi]
                            a_scale_val = vector.extract(
                                a_scale_i32, static_position=[0], dynamic_position=[]
                            )
                            if const_expr(m_scale_shift_i32 is not None):
                                a_scale_val = arith.shrui(
                                    a_scale_val, m_scale_shift_i32
                                )
                            a128_list = [None] * pack_M
                            for imxdl in range_constexpr(pack_M):
                                col_base0 = col_base
                                mi_idx = mi * pack_M + imxdl
                                mi_val = arith.constant(mi_idx * 16, index=True)
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
                                        curr_row_a_lds, col_base0, lds_base
                                    )

                                if const_expr(is_f8_a):
                                    col_base1 = col_base + 64
                                    a2, a3 = lds_load_packs_k64(
                                        curr_row_a_lds, col_base1, lds_base
                                    )
                                    a128_list[imxdl] = pack_i64x4_to_i32x8(
                                        a0, a1, a2, a3
                                    )
                                else:
                                    a128_list[imxdl] = pack_i64x4_to_i32x8(
                                        a0, a1, c0_i64, c0_i64
                                    )

                            for ni in range_constexpr(num_acc_n_packed):
                                b_scale_i32 = b_scale[ku128 * num_acc_n_packed + ni]
                                b_scale_val = vector.extract(
                                    b_scale_i32,
                                    static_position=[0],
                                    dynamic_position=[],
                                )
                                if const_expr(nss is not None):
                                    b_scale_val = arith.shrui(b_scale_val, nss)

                                b128_list = [None] * pack_N
                                for inxdl in range_constexpr(pack_N):
                                    ni_idx = ni * pack_N + inxdl
                                    if const_expr(is_f8_b):
                                        b128_list[inxdl] = pack_i64x4_to_i32x8(
                                            b_packs0[ni_idx],
                                            b_packs1[ni_idx],
                                            b_packs2[ni_idx],
                                            b_packs3[ni_idx],
                                        )
                                    else:
                                        b128_list[inxdl] = pack_i64x4_to_i32x8(
                                            b_packs0[ni_idx],
                                            b_packs1[ni_idx],
                                            c0_i64,
                                            c0_i64,
                                        )

                                for imxdl in range_constexpr(pack_M):
                                    mi_idx = mi * pack_M + imxdl
                                    a128 = a128_list[imxdl]

                                    for inxdl in range_constexpr(pack_N):
                                        ni_idx = ni * pack_N + inxdl

                                        b128 = b128_list[inxdl]

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
                                                    ikxdl * scale_pack_m + imxdl,
                                                    a_scale_val,
                                                    ikxdl * scale_pack_n + inxdl,
                                                    b_scale_val,
                                                ],
                                            )
                                        )

                    return acc_list, epilogue_pf

                lds_tile_elems = arith.constant(tile_m * lds_stride, index=True)
                lds_base_cur = arith.index(0)
                lds_base_nxt = lds_tile_elems

                rocdl.sched_barrier(0)

                def hot_loop_scheduler():
                    mfma_group = num_acc_n
                    mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                    mfma_per_iter = 2 * mfma_group
                    sche_iters = (
                        0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                    )

                    rocdl.sched_dsrd(2)
                    rocdl.sched_mfma(1)
                    if const_expr(tile_m == 16):
                        rocdl.sched_vmem(1)
                    rocdl.sched_mfma(1)
                    if const_expr(tile_m == 16):
                        rocdl.sched_vmem(1)
                    if const_expr(num_acc_n < 4):
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                        if const_expr(tile_m == 16):
                            rocdl.sched_vmem(1)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                        if const_expr(tile_m == 16):
                            rocdl.sched_vmem(1)
                        rocdl.sched_mfma(1)

                    if const_expr(use_async_copy):
                        dswr_tail = 0
                    else:
                        dswr_tail = num_x_loads
                        if const_expr(dswr_tail > sche_iters):
                            dswr_tail = sche_iters
                    dswr_start = sche_iters - dswr_tail

                    for sche_i in range_constexpr(sche_iters):
                        rocdl.sched_vmem(1)
                        rocdl.sched_mfma(mfma_group)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(mfma_group)
                        if const_expr(dswr_tail > 0 and sche_i >= dswr_start - 1):
                            rocdl.sched_dswr(1)

                    rocdl.sched_barrier(0)

                def k_shift_bits(k_py):
                    if const_expr(pack_K >= scale_pack_k):
                        return 0
                    return ((k_py // 128) % scale_pack_k) * scale_pack_m * 8

                def k_base(k_py):
                    return k_py // scale_pack_k // 128

                row_stride_bytes_pre = int(model_dim) * int(out_elem_bytes)
                use_buf_atomic_pre = bool(accumulate) and (
                    row_stride_bytes_pre <= 16384
                )
                c_tile_m_idx = arith.constant(tile_m, index=True)
                tid_in_range = arith.cmpi(CmpIPredicate.ult, tx, c_tile_m_idx)
                r216_defer_tid = bool(
                    r139_xdma_first
                    and use_async_copy
                    and int(tile_m) == 64
                    and int(tile_n) == 128
                    and int(tile_k) == 256
                )

                def emit_tid_lds_prologue():
                    if_tid = scf.IfOp(tid_in_range)
                    with ir.InsertionPoint(if_tid.then_block):
                        tid_row = bx_m + tx
                        tid_val = buffer_ops.buffer_load(
                            sorted_rsrc, tid_row, vec_width=1, dtype=T.i32
                        )
                        if const_expr(doweight_stage2):
                            tw_val_m = buffer_ops.buffer_load(
                                sorted_w_rsrc, tid_row, vec_width=1, dtype=f32
                            )
                        if const_expr(use_buf_atomic_pre):
                            t_pre = tid_val & arith.constant(0xFFFFFF, type=T.i32)
                            s_pre = arith.shrui(tid_val, arith.constant(24, type=T.i32))
                            row_byte_off = t_pre * arith.constant(
                                row_stride_bytes_pre, type=T.i32
                            )
                            global_row_i32 = arith.index_cast(T.i32, tid_row)
                            valid = arith.andi(
                                arith.andi(
                                    arith.cmpi(
                                        CmpIPredicate.ult,
                                        global_row_i32,
                                        num_valid_i32,
                                    ),
                                    arith.cmpi(
                                        CmpIPredicate.ult, t_pre, tokens_i32_guard
                                    ),
                                ),
                                arith.cmpi(
                                    CmpIPredicate.ult,
                                    s_pre,
                                    arith.constant(topk, type=T.i32),
                                ),
                            )
                            stored_val = arith.select(
                                valid,
                                row_byte_off,
                                arith.constant(0x7FFF0000, type=T.i32),
                            )
                        else:
                            stored_val = tid_val
                        tid_vec1 = vector.from_elements(T.vec(1, T.i32), [stored_val])
                        vector.store(tid_vec1, lds_tid, [tx])
                        if const_expr(doweight_stage2):
                            tw_vec1 = vector.from_elements(T.vec(1, T.f32), [tw_val_m])
                            vector.store(tw_vec1, lds_tw, [tx], alignment=4)
                        scf.YieldOp([])

                if const_expr(not r216_defer_tid):
                    emit_tid_lds_prologue()

                k0 = arith.index(0)
                k0_bk = k0
                if const_expr(r139_xdma_first):
                    prefetch_x_to_lds(k0, lds_base_cur)
                    rocdl.sched_barrier(0)
                if const_expr(b_split_enabled):
                    b_cur = load_b_tile_lo(k0_bk)
                else:
                    b_cur = load_b_tile(k0_bk)
                a_scale_pong, b_scale_pong = prefetch_ab_scale_tile(
                    k_base(0), k_shift_bits(0)
                )
                rocdl.sched_barrier(0)
                if const_expr(not r139_xdma_first):
                    if const_expr(use_async_copy):
                        prefetch_x_to_lds(k0, lds_base_cur)
                    else:
                        x_regs0 = load_x_tile(k0)
                        store_x_tile_to_lds(x_regs0, lds_base_cur)
                elif const_expr(not use_async_copy):
                    x_regs0 = load_x_tile(k0)
                    store_x_tile_to_lds(x_regs0, lds_base_cur)
                if const_expr(r216_defer_tid):
                    emit_tid_lds_prologue()
                    rocdl.sched_barrier(0)
                gpu.barrier()

                acc = [acc_init] * num_acc_n * m_repeat
                lds_base_pong = lds_base_cur
                lds_base_ping = lds_base_nxt

                a0_prefetch_pong = lds_load_packs_k64(
                    row_a_lds, col_offset_base, lds_base_pong
                )
                a1_col_base = col_offset_base + 128 // a_elem_vec_pack
                a1_prefetch_pong = (
                    lds_load_packs_k64(row_a_lds, a1_col_base, lds_base_pong)
                    if pack_K >= 2
                    else None
                )

                num_k_tiles_py = num_k_tiles_per_batch
                odd_k_tiles = (num_k_tiles_py % 2) == 1
                tail_tiles = 1 if odd_k_tiles else 2
                k_main2_py = (num_k_tiles_py - tail_tiles) * int(tile_k)
                if const_expr(k_main2_py < 0):
                    k_main2_py = 0

                # uninitialized -> feeding them to mfma_scale yields NaN.  Skip
                K_per_ku_s2 = int(tile_k) // int(k_unroll)
                pad_ku_skip_s2 = (
                    min(int(k_unroll), int(inter_dim_pad) // K_per_ku_s2)
                    if inter_dim_pad > 0
                    else 0
                )
                tail_ku_s2 = int(k_unroll) - pad_ku_skip_s2

                c2_tile_k = arith.constant(tile_k * 2, index=True)
                b_pong = b_cur
                k0_pong_bk = k0_bk

                # would create a region whose internal SSA values cannot be used
                def make_b_hi_loader(base_k):
                    """Create a b_hi_loader callable for a given base_k."""
                    return lambda bk=base_k: load_b_tile_hi(bk)

                if const_expr(k_main2_py > 0):
                    for k_iv_py in range_constexpr(0, k_main2_py, tile_k * 2):
                        k_iv = arith.index(k_iv_py)
                        next_k1 = k_iv + tile_k
                        next_k1_py = k_iv_py + tile_k
                        next_k1_bk = next_k1 // b_byte_div
                        if const_expr(use_async_copy):
                            prefetch_x_to_lds(next_k1, lds_base_ping)
                        else:
                            x_regs_ping = load_x_tile(next_k1)
                        a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(
                            k_base(next_k1_py), k_shift_bits(next_k1_py)
                        )
                        b_ping_lo = (
                            load_b_tile_lo(next_k1_bk)
                            if b_split_enabled
                            else load_b_tile(next_k1_bk)
                        )

                        acc, _ = compute_tile(
                            acc,
                            b_pong,
                            lds_base_pong,
                            a_scale_pong,
                            b_scale_pong,
                            a0_prefetch=a0_prefetch_pong,
                            a1_prefetch=a1_prefetch_pong,
                            b_hi_loader=(
                                make_b_hi_loader(k0_pong_bk)
                                if b_split_enabled
                                else None
                            ),
                        )
                        if const_expr(not use_async_copy):
                            store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                        gpu.barrier()

                        a0_prefetch_ping = lds_load_packs_k64(
                            row_a_lds, col_offset_base, lds_base_ping
                        )
                        a1_prefetch_ping = (
                            lds_load_packs_k64(row_a_lds, a1_col_base, lds_base_ping)
                            if pack_K >= 2
                            else None
                        )

                        next_k2 = k_iv + c2_tile_k
                        next_k2_py = k_iv_py + tile_k * 2
                        next_k2_bk = next_k2 // b_byte_div
                        if const_expr(use_async_copy):
                            prefetch_x_to_lds(next_k2, lds_base_pong)
                        else:
                            x_regs_pong = load_x_tile(next_k2)
                        a_scale_pong, b_scale_pong = prefetch_ab_scale_tile(
                            k_base(next_k2_py), k_shift_bits(next_k2_py)
                        )
                        b_pong = (
                            load_b_tile_lo(next_k2_bk)
                            if b_split_enabled
                            else load_b_tile(next_k2_bk)
                        )

                        acc, _ = compute_tile(
                            acc,
                            b_ping_lo,
                            lds_base_ping,
                            a_scale_ping,
                            b_scale_ping,
                            a0_prefetch=a0_prefetch_ping,
                            a1_prefetch=a1_prefetch_ping,
                            b_hi_loader=(
                                make_b_hi_loader(next_k1_bk)
                                if b_split_enabled
                                else None
                            ),
                        )
                        k0_pong_bk = next_k2_bk
                        if const_expr(not use_async_copy):
                            store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                        gpu.barrier()

                        a0_prefetch_pong = lds_load_packs_k64(
                            row_a_lds, col_offset_base, lds_base_pong
                        )
                        a1_prefetch_pong = (
                            lds_load_packs_k64(row_a_lds, a1_col_base, lds_base_pong)
                            if pack_K >= 2
                            else None
                        )

                if const_expr(odd_k_tiles):
                    acc, epilogue_pf = compute_tile(
                        acc,
                        b_pong,
                        lds_base_pong,
                        a_scale_pong,
                        b_scale_pong,
                        a0_prefetch=a0_prefetch_pong,
                        a1_prefetch=a1_prefetch_pong,
                        prefetch_epilogue=True,
                        ku_count=tail_ku_s2,
                        b_hi_loader=(
                            make_b_hi_loader(k0_pong_bk) if b_split_enabled else None
                        ),
                    )

                else:
                    k_tail1 = (k_in + tile_k - 1) // tile_k * tile_k - tile_k
                    k_tail1_py = (
                        int(inter_dim) + tile_k - 1
                    ) // tile_k * tile_k - tile_k
                    k_tail1_bk = k_tail1 // b_byte_div
                    if const_expr(use_async_copy):
                        prefetch_x_to_lds(k_tail1, lds_base_ping)
                    else:
                        x_regs_ping = load_x_tile(k_tail1)
                    b_ping_lo = (
                        load_b_tile_lo(k_tail1_bk)
                        if b_split_enabled
                        else load_b_tile(k_tail1_bk)
                    )
                    a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(
                        k_base(k_tail1_py), k_shift_bits(k_tail1_py)
                    )

                    acc, _ = compute_tile(
                        acc,
                        b_pong,
                        lds_base_pong,
                        a_scale_pong,
                        b_scale_pong,
                        a0_prefetch=a0_prefetch_pong,
                        a1_prefetch=a1_prefetch_pong,
                        b_hi_loader=(
                            make_b_hi_loader(k0_pong_bk) if b_split_enabled else None
                        ),
                    )

                    if const_expr(not use_async_copy):
                        store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    gpu.barrier()

                    a0_prefetch_ping = lds_load_packs_k64(
                        row_a_lds, col_offset_base, lds_base_ping
                    )
                    a1_prefetch_ping = (
                        lds_load_packs_k64(row_a_lds, a1_col_base, lds_base_ping)
                        if pack_K >= 2
                        else None
                    )
                    acc, epilogue_pf = compute_tile(
                        acc,
                        b_ping_lo,
                        lds_base_ping,
                        a_scale_ping,
                        b_scale_ping,
                        a0_prefetch=a0_prefetch_ping,
                        a1_prefetch=a1_prefetch_ping,
                        prefetch_epilogue=True,
                        ku_count=tail_ku_s2,
                        b_hi_loader=(
                            make_b_hi_loader(k_tail1_bk) if b_split_enabled else None
                        ),
                    )

                tw_pf = None
                bias_pf = None
                if const_expr(epilogue_pf is not None):
                    _, tw_pf, bias_pf = epilogue_pf

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

                if const_expr(lds_out is None):
                    raise RuntimeError(
                        "FLIR_MOE_STAGE2_CSHUFFLE=1 but lds_out is not allocated/aliased."
                    )

                out_base_i64 = arith.index_cast(T.i64, fx.ptrtoint(arg_out))
                out_base_idx = arith.index_cast(T.index, out_base_i64)

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
                        v = vector.extract(
                            acc[acc_idx], static_position=[ii], dynamic_position=[]
                        )
                        if const_expr(enable_bias):
                            v = v + bias_pf[ni]

                        if const_expr(doweight_stage2):
                            v = v * tw
                        v_out = arith.trunc_f(out_elem(), v)

                        lds_idx = row_base_lds + col_local
                        vec1_out = T.vec(1, out_elem())
                        v1 = vector.from_elements(vec1_out, [v_out])

                        vector.store(v1, lds_out, [lds_idx], alignment=2)

                row_stride_bytes_py = int(model_dim) * int(out_elem_bytes)
                use_buf_atomic = bool(accumulate) and (row_stride_bytes_py <= 16384)
                out_elem_bytes_i32 = (
                    arith.constant(int(out_elem_bytes), type=T.i32)
                    if use_buf_atomic
                    else None
                )

                def precompute_row(*, row_local, row):
                    if const_expr(use_buf_atomic):
                        row_byte_off_i32 = memref.load(lds_tid, [row_local])
                        return (
                            (None, None, row_byte_off_i32),
                            None,
                        )
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
                    if const_expr(accumulate):
                        row_byte_base = out_base_idx + t_idx * arith.constant(
                            model_dim * out_elem_bytes, index=True
                        )
                    else:
                        row_byte_base = out_base_idx + ts_idx * arith.constant(
                            model_dim * out_elem_bytes, index=True
                        )
                    row_byte_off_i32 = None
                    return ((fused2, row_byte_base, row_byte_off_i32), row_valid)

                def idx_to_llvm_ptr(idx_val, addr_space=1):
                    """Convert an index-typed byte address to !llvm.ptr<addr_space>."""
                    idx_v = idx_val._value if hasattr(idx_val, "_value") else idx_val
                    i64_v = arith.index_cast(T.i64, idx_v)
                    i64_raw = i64_v._value if hasattr(i64_v, "_value") else i64_v
                    ptr_ty = ir.Type.parse(f"!llvm.ptr<{addr_space}>")
                    return llvm.inttoptr(ptr_ty, i64_raw)

                def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                    fused, row_byte_base, row_byte_off_i32 = row_ctx
                    if const_expr(not bool(accumulate)):
                        col_idx = col_g0
                        byte_off_col = col_idx * arith.constant(
                            out_elem_bytes, index=True
                        )
                        ptr_addr_idx = row_byte_base + byte_off_col
                        out_ptr_v = idx_to_llvm_ptr(ptr_addr_idx)
                        frag_v = frag._value if hasattr(frag, "_value") else frag
                        llvm.StoreOp(
                            frag_v,
                            out_ptr_v,
                            alignment=e_vec * out_elem_bytes,
                            nontemporal=True,
                        )
                    elif const_expr(use_buf_atomic):
                        col_i32 = arith.index_cast(T.i32, col_g0)
                        col_byte_off_i32 = col_i32 * out_elem_bytes_i32
                        byte_off_i32 = row_byte_off_i32 + col_byte_off_i32
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            frag,
                            out_rsrc,
                            byte_off_i32,
                            zero_i32,
                            zero_i32,
                        )
                    else:
                        col_idx = col_g0
                        byte_off_col = col_idx * arith.constant(
                            out_elem_bytes, index=True
                        )
                        ptr_addr_idx = row_byte_base + byte_off_col
                        out_ptr_v = idx_to_llvm_ptr(ptr_addr_idx)
                        frag_v = frag._value if hasattr(frag, "_value") else frag
                        llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.fadd,
                            out_ptr_v,
                            frag_v,
                            llvm.AtomicOrdering.monotonic,
                            syncscope="agent",
                            alignment=e_vec * out_elem_bytes,
                        )

                e_vec = 2 if accumulate else min(tile_n // 32, 8)
                rocdl.s_setprio(3)
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
                    frag_elem_type=(
                        ir.BF16Type.get() if out_is_bf16 else ir.F16Type.get()
                    ),
                    write_row_to_lds=write_row_to_lds,
                    precompute_row=precompute_row,
                    store_pair=store_pair,
                )
                rocdl.s_setprio(0)

            all_valid = arith.andi(blk_valid, arith.andi(exp_valid, tile_has_tokens))

            if const_expr(persistent):
                cur_active = arith.andi(still_active, blk_valid)
                do_gemm = arith.andi(cur_active, arith.andi(exp_valid, tile_has_tokens))
                if_valid = scf.IfOp(do_gemm)
                with ir.InsertionPoint(if_valid.then_block):
                    moe_gemm2_then_body()
                    scf.YieldOp([])

                gpu.barrier()
                scf.YieldOp([cur_active])
            else:
                if_valid = scf.IfOp(all_valid)
                with ir.InsertionPoint(if_valid.then_block):
                    moe_gemm2_then_body()
                    scf.YieldOp([])

                gpu.barrier()
                scf.YieldOp([expert_i32, expert_b_base])
            for_ip.__exit__(None, None, None)

    cache_tag = (
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
        cu_num if persistent else 0,
        waves_per_eu,
        use_async_copy,
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
        _ = cache_tag
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        n_in = arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
        tile_n_idx = arith.constant(tile_n, index=True)
        model_dim_pad_idx = arith.constant(model_dim_pad, index=True)
        gx = (
            n_in - model_dim_pad_idx + tile_n_idx - arith.constant(1, index=True)
        ) / tile_n_idx
        if const_expr(persistent):
            gy = arith.constant(cu_num, index=True)
        else:
            c_pm_l = arith.constant(persist_m, index=True)
            gy = (
                arith.index_cast(ir.IndexType.get(), i32_size_expert_ids_in.ir_value())
                + c_pm_l
                - arith.constant(1, index=True)
            ) / c_pm_l

        launcher = moe_gemm2(
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
        )
        if const_expr(waves_per_eu is not None):
            wpe = int(waves_per_eu)
            for op in ctx.gpu_module_body.operations:
                if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, wpe)
        launcher.launch(
            grid=(gx, gy, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_mixed_moe_gemm2
