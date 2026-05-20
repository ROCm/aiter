# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE GEMM stage1/stage2 kernel implementations (FlyDSL MFMA FP8/FP16/FP4).

This module contains the **kernel builder code** for:
- `moe_gemm1` (stage1, with silu/swiglu activation)
- `moe_gemm2` (stage2)

It is extracted from `tests/kernels/test_moe_gemm.py` so that:
- `kernels/` holds the implementation
- `tests/` holds correctness/perf harnesses

Mixed-precision support (a_dtype x b_dtype):
- fp8 x fp8, fp8 x fp4 (A8W4 on gfx950), fp4 x fp4,
  fp16 x fp16, int8 x int4, ...

A8W4 path is selected by `a_dtype='fp8', b_dtype='fp4'` plus
`gate_mode=GateMode.INTERLEAVE` + `a_scale_one=True` in stage1.
"""

import os
import functools
from enum import Enum

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

try:
    from flydsl.runtime.device import supports_bf16_global_atomics
except ImportError:

    def supports_bf16_global_atomics(arch: str) -> bool:
        return str(arch).startswith(("gfx94", "gfx95", "gfx12"))


from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from flydsl._mlir import ir
from flydsl.expr.typing import T

from flydsl.expr import arith, gpu, buffer_ops, vector, rocdl, const_expr
from flydsl._mlir.dialects import llvm, scf, memref
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects.arith import CmpIPredicate

from .mfma_preshuffle_pipeline import (
    _buffer_load_vec,
    buffer_copy_gmem16_dwordx4,
    lds_store_16b_xor16,
    lds_store_8b_xor16,
    lds_store_4b_xor16,
    make_preshuffle_b_layout,
    make_preshuffle_scale_layout,
    load_b_raw_mxfp4_dwordx4,
    unpack_b_mxfp4_bf16,
    tile_chunk_coord_i32,
    swizzle_xor16,
)
from .mfma_epilogues import c_shuffle_epilog, default_epilog, mfma_epilog
from .layout_utils import crd2idx, idx2crd, get as layout_get
from .kernels_common import _if_then, validate_moe_dtypes


class GateMode(str, Enum):
    """Gate/Up computation strategy for stage1 GEMM.

    SEPARATED:      Two separate B-tile streams (gate + up), default mode.
    MOCK_GATE_ONLY: Single B-tile stream over full [0, 2*inter_dim), simulates
                    gate-only by doubling grid X on top of SEPARATED layout.
                    Requires split-K (k_batch>1).  NOT true gate-only.
    GATE_ONLY:      Reserved for future true gate-only implementation.
    INTERLEAVE:     Weight rows interleave gate/up (gate[0], up[0], gate[1], ...).
                    pack_N=2 routes even/odd N subtiles.  NOT tied to split-K.
    """

    SEPARATED = "separated"
    MOCK_GATE_ONLY = "mock_gate_only"
    GATE_ONLY = "gate_only"
    INTERLEAVE = "interleave"


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
    split_k_intra: int = 1,
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
    is_a16w4_stage1 = a_dtype == "bf16" and b_dtype in ("fp4", "mxfp4")

    # ---- Shared prelude (used by both A16W4 and generic stage1 paths) ----
    # Padding semantics: model_dim and inter_dim INCLUDE padding.
    #   model_dim = model_dim_true + model_dim_pad   (K direction)
    #   inter_dim = inter_dim_true + inter_dim_pad   (N direction)
    # The grid simply does not launch tiles for padding columns.
    gpu_arch = get_hip_arch()
    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")
    mock_gate_only = gate_mode is GateMode.MOCK_GATE_ONLY
    gate_up_interleave = gate_mode is GateMode.INTERLEAVE
    _inter_dim_valid = inter_dim - inter_dim_pad
    _is_splitk = k_batch > 1
    _use_lds128 = os.environ.get("FLIR_CK_LDS128", "1") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    pad_k = 0 if _use_lds128 else 8
    lds_stride = tile_k + pad_k

    # ==== Unified setup (merged A16W4 + Generic stage1) ====
    # All paths share the same setup variables; A16W4-specific blocks are
    # gated by `if is_a16w4_stage1:` and generic-only blocks by
    # `if not is_a16w4_stage1:`.  This mirrors the stage2 design.

    # A16W4 alias used throughout the body: gate_only ≡ mock_gate_only
    gate_only = mock_gate_only
    if gate_only and gate_up_interleave:
        raise ValueError(
            "gate_only / mock_gate_only and gate_up_interleave are mutually exclusive"
        )
    if gate_only and not _is_splitk:
        raise ValueError("gate_only / mock_gate_only requires k_batch > 1 (split-K)")

    _state = {}

    # ---- dtype probing (A16W4 hardcoded; generic dynamic) ----
    if not is_a16w4_stage1:
        validate_moe_dtypes(a_dtype, b_dtype)

    is_f16_a = is_a16w4_stage1 or (a_dtype == "fp16")
    is_f16_b = (not is_a16w4_stage1) and (b_dtype == "fp16")
    is_f8_a = (not is_a16w4_stage1) and (a_dtype == "fp8")
    is_f4_a = (not is_a16w4_stage1) and (a_dtype == "fp4")
    is_f4_b = (b_dtype in ("fp4", "mxfp4")) if is_a16w4_stage1 else (b_dtype == "fp4")
    is_int4 = (not is_a16w4_stage1) and (b_dtype == "int4")
    is_int8 = False

    # ---- wave / pack config ----
    sort_block_m = max(32, tile_m)
    if is_a16w4_stage1:
        num_waves = 4
        total_threads = 256
        pack_M = 1
        pack_N = 1
        pack_K = 2
        scale_mn_pack = 2
        a_elem_vec_pack = 1
        cbsz = 0
        blgp = 4
    else:
        num_waves = min(4, tile_n // 32)
        total_threads = num_waves * 64
        pack_M = 1 if tile_m < 32 else 2
        pack_N = min(2, (tile_n // num_waves) // 16)
        pack_K = 2
        scale_mn_pack = 2
        a_elem_vec_pack = 2 if is_f4_a else 1
        cbsz = 0 if is_f8_a else 4
        blgp = 4
    n_per_wave = tile_n // num_waves

    # ---- byte / element parameters ----
    elem_bytes = 2 if is_a16w4_stage1 else 1
    a_elem_bytes = 2 if is_f16_a else 1
    b_elem_bytes = 1
    tile_k_bytes = int(tile_k) * int(a_elem_bytes)
    if (tile_k_bytes % 64) != 0:
        raise ValueError(f"tile_k_bytes must be divisible by 64, got {tile_k_bytes}")

    # ---- output dtype parsing ----
    out_s = str(out_dtype).strip().lower()
    out_is_f32 = out_s in ("f32", "fp32", "float")
    out_is_bf16 = out_s in ("bf16", "bfloat16")

    # ---- A16W4-specific BF16 K32 MFMA helper ----
    mfma_f32_bf16_k32 = None
    if is_a16w4_stage1:
        if out_dtype not in ("f16", "bf16"):
            raise ValueError(f"out_dtype must be 'f16' or 'bf16', got {out_dtype!r}")
        _mfma_k32_raw = getattr(rocdl, "mfma_f32_16x16x32_bf16_", None)
        if _mfma_k32_raw is None:
            raise AttributeError(
                "BF16 K32 MFMA op not found: expected `rocdl.mfma_f32_16x16x32_bf16_`"
            )
        _split_mfma = rocdl._split_mfma_operands

        def mfma_f32_bf16_k32(result_type, operands, *, loc=None, ip=None):
            a, b, c, cbsz, abid, blgp = _split_mfma(operands, loc=loc)
            return _mfma_k32_raw(result_type, a, b, c, cbsz, abid, blgp, loc=loc, ip=ip)

    # ---- type helpers (A16W4 forces bf16 X / i8 W; generic dispatches on dtype flags) ----
    def out_mlir():
        return (lambda ty: ty() if callable(ty) else ty)(
            T.f16 if out_dtype == "f16" else T.bf16
        )

    def _x_elem_type():
        if is_a16w4_stage1:
            return T.bf16
        if is_f4_b:
            return T.f8 if is_f8_a else T.i8
        return T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)

    def _w_elem_type():
        if is_f4_b:
            return T.i8
        return T.f16 if is_f16_b else (T.i8 if is_int8 else T.f8)

    def out_elem():
        return T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)

    def x_lds_elem():
        if is_a16w4_stage1:
            return T.bf16
        return T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)

    # ---- K dimension & split-K validation ----
    if is_a16w4_stage1:
        if const_expr(_is_splitk):
            _k_per_batch = model_dim // k_batch
        else:
            _k_per_batch = model_dim
        _k_dim = _k_per_batch
    else:
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

    # A16W4 body uses _single_b to skip the up-projection in single-output paths
    _single_b = gate_only or gate_up_interleave

    # ---- Tile bytes / thread distribution ----
    bytes_x_per_tile = (
        int(tile_m)
        * int(tile_k)
        * int(a_elem_bytes if not is_a16w4_stage1 else elem_bytes)
    )
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads

    # ---- CShuffle epilog handling ----
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

    if is_a16w4_stage1:
        if out_dtype not in ("f16", "bf16") and _use_cshuffle_epilog:
            raise ValueError("stage1 cshuffle epilog supports only f16/bf16 output")
        _k_batch = split_k_intra
        if const_expr(_k_batch > 1):
            _use_cshuffle_epilog = False
            _waves_per_group = 4 // _k_batch
            _n_per_wave_check = tile_n // _waves_per_group
            if _n_per_wave_check < 16:
                raise ValueError(
                    f"split_k_intra={_k_batch} with tile_n={tile_n}: "
                    f"n_per_wave={_n_per_wave_check} < 16 (MFMA minimum)"
                )
        # GUI cross-wave fusion: needed when num_acc_n < 2 per wave
        # (standard pair fusion requires gate+up in same wave, i.e. num_acc_n >= 2)
        _n_per_wave_eff = tile_n // ((4 // _k_batch) if _k_batch > 1 else 4)
        _gui_xwave_fuse = (
            gate_up_interleave and not _is_splitk and (_n_per_wave_eff // 16) < 2
        )
        if const_expr(_gui_xwave_fuse):
            _use_cshuffle_epilog = False
        # Auto-disable cshuffle when tile_m doesn't meet CShuffleMLane constraint
        if const_expr(_use_cshuffle_epilog):
            _eff_out_n = (
                (tile_n // 2) if (gate_up_interleave and not _is_splitk) else tile_n
            )
            _cs_nlane_chk = min(32, _eff_out_n // 4)
            if _cs_nlane_chk > 0:
                _cs_mlane_chk = 256 // _cs_nlane_chk
                if tile_m % _cs_mlane_chk != 0:
                    _use_cshuffle_epilog = False
    else:
        _k_batch = 1  # not used by generic
        _need_fp4 = out_dtype == "fp4"
        _need_fp8 = out_dtype == "fp8"
        _need_quant = _need_fp4 or _need_fp8
        _need_sort = _need_quant
        if _need_quant:
            _use_cshuffle_epilog = True

    # ---- module_name (different format per path; cache key) ----
    if is_a16w4_stage1:
        _mode_tag = "gui" if gate_up_interleave else ("go" if gate_only else "sep")
        epilog_tag = "cshuffle" if _use_cshuffle_epilog else "direct"
        _wpe_tag = f"_wpe{waves_per_eu}" if waves_per_eu >= 1 else ""
        _ski_tag = f"_ski{_k_batch}" if _k_batch > 1 else ""
        _bias_tag = "_bias" if enable_bias else ""
        _act_tag = f"_{act}" if act != "silu" else ""
        _pad_tag = (
            f"_mp{model_dim_pad}_ip{inter_dim_pad}"
            if (model_dim_pad or inter_dim_pad)
            else ""
        )
        module_name = (
            f"mfma_a16w4_moe1_mxfp4_{out_dtype}_{_mode_tag}_{epilog_tag}"
            f"_t{tile_m}x{tile_n}x{tile_k}_kb{k_batch}"
            f"{_wpe_tag}{_ski_tag}{_bias_tag}{_act_tag}{_pad_tag}_abi1"
        ).replace("-", "_")
    else:
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

    # ---- LDS sizing ----
    if is_a16w4_stage1:
        kpack_bytes = 16  # MXFP4 preshuffle
        # For interleave+non-splitk cshuffle, the epilogue output tile_n is halved
        _gui_out_tile_n = (
            tile_n // 2 if (gate_up_interleave and not _is_splitk) else tile_n
        )
        _single_x_bytes = int(tile_m) * int(lds_stride) * int(elem_bytes)
        _single_x_elems = _single_x_bytes // int(elem_bytes)
        lds_out_bytes = (
            2 * int(tile_m) * int(_gui_out_tile_n) if _use_cshuffle_epilog else 0
        )
        # Ping-pong: pong holds max(input, output), ping holds input only
        _pong_buffer_bytes = max(_single_x_bytes, lds_out_bytes)
        _ping_buffer_bytes = _single_x_bytes
        lds_tid_bytes = 0
        _split_lds_out = False
    else:
        kpack_bytes = 8 if is_int4 else 16
        _cshuffle_elem_bytes = 4 if _need_quant else (4 if out_is_f32 else 2)
        _single_x_bytes = int(tile_m) * int(lds_stride) * int(a_elem_bytes)
        lds_out_bytes = (
            _cshuffle_elem_bytes * int(tile_m) * int(tile_n)
            if _use_cshuffle_epilog
            else 0
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

    # ---- LDS allocator pointers ----
    lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = lds_pong_offset + _pong_buffer_bytes
    if not is_a16w4_stage1:
        _lds_tid_offset_pong = allocator_pong._align(allocator_pong.ptr, 4)
        allocator_pong.ptr = _lds_tid_offset_pong + lds_tid_bytes
    lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = lds_ping_offset + _ping_buffer_bytes

    # ---- Generic-only postlude (pipeline schedule + quant block layout) ----
    if not is_a16w4_stage1:
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

        # Unified pipeline schedule (outside @flyc.kernel).  Each phase descriptor:
        #   mfma:      [(k_idx, mi_idx, ikxdl, imxdl, asv_idx), ...]
        #   a_reads:   [(k, mi), ...]            # A ds_read subtiles
        #   b_loads:   [('gate'/'up', ku, ni)]   # B VMEM loads
        #   has_scale: bool                       # A/B scale VMEM loads
        _pipe_m_repeat = tile_m // 16
        _pipe_k_unroll = tile_k_bytes // 128
        _pipe_k_unroll_packed = _pipe_k_unroll // pack_K
        _pipe_m_repeat_packed = _pipe_m_repeat // pack_M
        _pipe_num_acc_n = n_per_wave // 16

        # A ds_read groups: same mi, all k-values together
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

        # B VMEM loads: gate/up
        _pipe_b_loads = []
        for ku in range(_pipe_k_unroll):
            for ni in range(_pipe_num_acc_n):
                _pipe_b_loads.append(("gate", ku, ni))
                if not mock_gate_only and not gate_up_interleave:
                    _pipe_b_loads.append(("up", ku, ni))

        # MFMA order: B-major (fix B, cycle all A tiles before next B)
        _pipe_num_acc_n_packed = _pipe_num_acc_n // pack_N
        _pipe_all_mfma = []
        for _ku128 in range(_pipe_k_unroll_packed):
            for _ni_packed in range(_pipe_num_acc_n_packed):
                for _ikxdl in range(pack_K):
                    for _inxdl in range(pack_N):
                        _k_idx = _ku128 * pack_K + _ikxdl
                        _ni_idx = _ni_packed * pack_N + _inxdl
                        _pipe_all_mfma.append((_k_idx, _ni_idx, _ikxdl, _inxdl, _ku128))

        _pipe_mfma_per_phase = max(1, len(_pipe_all_mfma) // 4)
        _pipe_n_phases = len(_pipe_all_mfma) // _pipe_mfma_per_phase

        _a_groups_per_phase = (
            len(_pipe_a_groups) + _pipe_n_phases - 1
        ) // _pipe_n_phases
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

        _pp_mfma = [p["mfma"] for p in _pipe_phases]
        _pp_a_reads = [p["a_reads"] for p in _pipe_phases]
        _pp_b_loads = [p["b_loads"] for p in _pipe_phases]
        _pp_has_scale = [p["has_scale"] for p in _pipe_phases]

        fp4_ratio = 2 if a_dtype == "fp4" else 1
        gui_ratio = 1 if gate_up_interleave else 2
        _vmcnt_before_barrier = tile_m // 32 // fp4_ratio + tile_n // 32 * gui_ratio

    # ==== End unified setup ====
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
        if const_expr(is_a16w4_stage1):
            _ = arg_scale_x
            _ = arg_out_scale_sorted
            tokens_in = arith.index_cast(T.index, i32_tokens_in.ir_value())
            inter_in = arith.ArithValue(arith.index_cast(T.index, i32_n_in.ir_value()))
            k_in = arith.index_cast(T.index, i32_k_in.ir_value())
            size_expert_ids_in = arith.index_cast(
                T.index, i32_size_expert_ids_in.ir_value()
            )
            k_i32_v = i32_k_in.ir_value()

            x_elem = T.bf16
            w_elem = T.i8
            f32 = T.f32
            i32 = T.i32
            i64 = T.i64
            vec4_f32 = T.vec(4, f32)
            vec8_bf16 = T.vec(8, x_elem)
            vec16_x = T.vec(8, x_elem)
            vec2_i64 = T.vec(2, i64)

            def _silu_elem(g):
                neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                t = g * neg_log2e
                emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                one = arith.constant(1.0, type=f32)
                den = one + emu
                sig = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [den], [], [])
                return g * sig

            def _silu_mul_vec4(gate_v4, up_v4):
                result_elems = []
                for ei in range_constexpr(4):
                    g = vector.extract(
                        gate_v4, static_position=[ei], dynamic_position=[]
                    )
                    u = vector.extract(up_v4, static_position=[ei], dynamic_position=[])
                    result_elems.append(_silu_elem(g) * u)
                return vector.from_elements(vec4_f32, result_elems)

            def _swiglu_mul_vec4(gate_v4, up_v4):
                result_elems = []
                _alpha = arith.constant(1.702, type=f32)
                _limit = arith.constant(7.0, type=f32)
                _neg_limit = arith.constant(-7.0, type=f32)
                _one = arith.constant(1.0, type=f32)
                _neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                for ei in range_constexpr(4):
                    g = vector.extract(
                        gate_v4, static_position=[ei], dynamic_position=[]
                    )
                    u = vector.extract(up_v4, static_position=[ei], dynamic_position=[])
                    g = arith.minimumf(g, _limit)
                    u = arith.minimumf(u, _limit)
                    u = arith.maximumf(u, _neg_limit)
                    t = g * _alpha * _neg_log2e
                    emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                    den = _one + emu
                    sig = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [den], [], [])
                    result_elems.append(g * sig * (u + _one))
                return vector.from_elements(vec4_f32, result_elems)

            def _act_vec4(gate_v4, up_v4):
                if const_expr(act == "swiglu"):
                    return _swiglu_mul_vec4(gate_v4, up_v4)
                else:
                    return _silu_mul_vec4(gate_v4, up_v4)

            def _act_elem(g_e, u_e):
                if const_expr(act == "swiglu"):
                    _alpha = arith.constant(1.702, type=f32)
                    _limit = arith.constant(7.0, type=f32)
                    _neg_limit = arith.constant(-7.0, type=f32)
                    _one = arith.constant(1.0, type=f32)
                    _neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                    g_e = arith.minimumf(g_e, _limit)
                    u_e = arith.minimumf(u_e, _limit)
                    u_e = arith.maximumf(u_e, _neg_limit)
                    t = g_e * _alpha * _neg_log2e
                    emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                    den = _one + emu
                    sig = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [den], [], [])
                    return g_e * sig * (u_e + _one)
                else:
                    return _silu_elem(g_e) * u_e

            acc_init = arith.constant_vector(0.0, vec4_f32)

            layout_x = fx.make_layout(  # noqa: F841
                (arith.index_cast(i32, tokens_in), k_i32_v), stride=(k_i32_v, 1)
            )

            # Gate+up interleaved: N_total = experts * 2 * inter_dim
            c_n_total = arith.index(experts * (2 * inter_dim))
            c2 = arith.index(2)
            c_k_packed = k_in // c2
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=c_k_packed,
                kpack_bytes=kpack_bytes,
                elem_bytes=1,
            )
            layout_b = b_layout.layout_b

            layout_b_scale = make_preshuffle_scale_layout(
                arith,
                c_mn=c_n_total,
                c_k=k_in,
                mn_pack=2,
                k_pack=2,
                elem_bytes=4,
                scale_block_size=32,
            )

            shape_lds = fx.make_shape(tile_m, tile_k)
            stride_lds = fx.make_stride(lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")

            by = gpu.block_id("x")
            bx = gpu.block_id("y")

            bx_m = bx * arith.index(tile_m)
            numids_rsrc = buffer_ops.create_buffer_resource(
                arg_num_valid_ids,
                max_size=False,
                num_records_bytes=arith.constant(4, type=i32),
            )
            num_valid_i32 = buffer_ops.buffer_load(
                numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=i32
            )
            num_valid_idx = arith.index_cast(T.index, num_valid_i32)  # noqa: F841
            bx_m_i32 = arith.index_cast(i32, bx_m)
            blk_valid = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, num_valid_i32)

            k_blocks16 = arith.index(tile_k_bytes // 16)
            layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

            _if_blk = scf.IfOp(blk_valid)
            with _if_then(_if_blk):
                base_ptr_pong = allocator_pong.get_base()
                base_ptr_ping = allocator_ping.get_base()
                lds_x_pong = SmemPtr(
                    base_ptr_pong,
                    lds_pong_offset,
                    T.bf16,
                    shape=(_single_x_elems,),
                ).get()
                lds_x_ping = SmemPtr(
                    base_ptr_ping,
                    lds_ping_offset,
                    T.bf16,
                    shape=(_single_x_elems,),
                ).get()
                lds_out = (
                    SmemPtr(
                        base_ptr_pong,
                        lds_pong_offset,
                        out_mlir(),
                        shape=(tile_m * _gui_out_tile_n,),
                    ).get()
                    if _use_cshuffle_epilog
                    else None
                )

                c_topk = arith.index(topk)
                x_nbytes_idx = tokens_in * k_in * arith.index(int(elem_bytes))
                x_rsrc = buffer_ops.create_buffer_resource(
                    arg_x, max_size=False, num_records_bytes=x_nbytes_idx
                )
                w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)
                sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)

                # Split-K uses f32 atomics → out_elem_bytes = 4
                # Rename to avoid shadowing the enclosing-scope setup var.
                _a16_out_elem_bytes = 4 if _is_splitk else 2
                out_nbytes_idx = (
                    tokens_in * c_topk * inter_in * arith.index(_a16_out_elem_bytes)
                )
                out_rsrc = buffer_ops.create_buffer_resource(
                    arg_out, max_size=False, num_records_bytes=out_nbytes_idx
                )

                sorted_rsrc = buffer_ops.create_buffer_resource(
                    arg_sorted_token_ids, max_size=False
                )
                sorted_w_rsrc = buffer_ops.create_buffer_resource(
                    arg_sorted_weights, max_size=False
                )
                expert_rsrc = buffer_ops.create_buffer_resource(
                    arg_expert_ids,
                    max_size=False,
                    num_records_bytes=(size_expert_ids_in * arith.index(4)),
                )

                expert_i32 = buffer_ops.buffer_load(
                    expert_rsrc, bx, vec_width=1, dtype=i32
                )
                exp_valid = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    expert_i32,
                    arith.constant(experts, type=i32),
                )
                expert_idx = arith.index_cast(T.index, expert_i32)
                inter2_idx = arith.index(2 * inter_dim)
                expert_off_idx = expert_idx * inter2_idx

                bias_rsrc = (
                    buffer_ops.create_buffer_resource(arg_bias, max_size=False)
                    if enable_bias
                    else None
                )

                if const_expr(
                    bytes_per_thread_x >= 16 and bytes_per_thread_x % 16 == 0
                ):
                    x_load_bytes = 16
                elif const_expr(
                    bytes_per_thread_x >= 8 and bytes_per_thread_x % 8 == 0
                ):
                    x_load_bytes = 8
                elif const_expr(
                    bytes_per_thread_x >= 4 and bytes_per_thread_x % 4 == 0
                ):
                    x_load_bytes = 4
                else:
                    raise ValueError(
                        f"bytes_per_thread_x ({bytes_per_thread_x}) must be "
                        f"divisible by 4"
                    )
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4
                x_vec_elems = x_load_bytes // elem_bytes
                x_vec_x_ty = T.vec(x_vec_elems, x_elem)

                c_k_div4 = (k_in * arith.index(int(elem_bytes))) // arith.index(4)
                c_k_div4_i32 = arith.index_cast(i32, c_k_div4)  # noqa: F841
                tile_k_dwords = (int(tile_k) * int(elem_bytes)) // 4
                layout_x_tile_div4 = fx.make_layout(
                    (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                )
                c_chunk_i32 = arith.index(chunk_i32)
                tx_i32_base = tx * c_chunk_i32
                mask24 = arith.constant(0xFFFFFF, type=T.i32)
                tokens_i32 = arith.index_cast(i32, tokens_in)
                topk_i32 = arith.constant(topk, type=T.i32)

                def x_tile_chunk_coord_i32(i: int):
                    return tile_chunk_coord_i32(
                        arith,
                        tx_i32_base=tx_i32_base,
                        i=i,
                        total_threads=total_threads,
                        layout_tile_div4=layout_x_tile_div4,
                        chunk_i32=chunk_i32,
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
                        sorted_rsrc, sorted_row_i, vec_width=1, dtype=i32
                    )
                    t_i32 = fused_i & mask24
                    s_i32 = arith.shrui(fused_i, arith.constant(24, type=i32))
                    t_valid = arith.cmpi(arith.CmpIPredicate.ult, t_i32, tokens_i32)
                    s_valid = arith.cmpi(arith.CmpIPredicate.ult, s_i32, topk_i32)
                    ts_valid = arith.andi(t_valid, s_valid)
                    t_safe = arith.select(
                        ts_valid,
                        arith.index_cast(T.index, t_i32),
                        arith.index(0),
                    )
                    x_row_base_div4.append(t_safe * c_k_div4)

                # NOTE: load_x_tile is unused on the stage1 A16W4 path -- B-cycle
                # consumes X via dma_x_tile_to_lds + LDS reads. Removed.

                coord_wl = idx2crd(tx, layout_tx_wave_lane)
                wave_id = layout_get(coord_wl, 0)
                lane_id = layout_get(coord_wl, 1)

                _dma_bytes = 16
                _wave_size = 64

                def dma_x_tile_to_lds(base_k, lds_buffer):
                    """Async DMA: global -> LDS via buffer_load_lds, no VGPR."""
                    c4_idx = arith.index(4)
                    base_k_div4 = (
                        base_k * arith.index(int(elem_bytes))
                    ) // arith.index(4)

                    lds_ptr_i64 = None
                    for i in range_constexpr(num_x_loads):
                        row_local_i = x_row_local[i]
                        col_local_i32_i = x_col_local_i32[i]
                        col_local_sw = swizzle_xor16(
                            row_local_i, col_local_i32_i * c4_idx, k_blocks16
                        )
                        row_k_dw = x_row_base_div4[i] + base_k_div4
                        global_byte_idx = row_k_dw * c4_idx + col_local_sw
                        global_offset = arith.index_cast(i32, global_byte_idx)

                        if const_expr(i == 0):
                            lds_addr = memref.extract_aligned_pointer_as_index(
                                lds_buffer
                            ) + wave_id * arith.constant(
                                _wave_size * _dma_bytes, index=True
                            )
                            lds_ptr_i64 = rocdl.readfirstlane(
                                i64, arith.index_cast(i64, lds_addr)
                            )
                        else:
                            lds_ptr_i64 = lds_ptr_i64 + arith.constant(
                                total_threads * _dma_bytes, type=i64
                            )

                        lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                        lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64)

                        rocdl.raw_ptr_buffer_load_lds(
                            x_rsrc,
                            lds_ptr,
                            arith.constant(_dma_bytes, type=i32),
                            global_offset,
                            arith.constant(0, type=i32),
                            arith.constant(0, type=i32),
                            arith.constant(0, type=i32),
                        )

                def prefetch_x_to_lds(base_k, lds_buffer):
                    dma_x_tile_to_lds(base_k, lds_buffer)

                coord_l16 = idx2crd(lane_id, layout_lane16)
                lane_div_16 = layout_get(coord_l16, 0)
                lane_mod_16 = layout_get(coord_l16, 1)

                row_a_lds = lane_mod_16
                # CK-style A addressing: sub-lane L covers K[L*32..L*32+31]
                # within each k0 block (128 K elements).
                # col_offset_base = lane_div_16 * 32 bf16 = lane_div_16 * 64 bytes
                # stride per ku step = 8 bf16 = 16 bytes
                # For k0 boundaries: ku=4 wraps to next k0 block (+256 bytes)
                _a_sublane_stride = 64  # 32 bf16 * 2 bytes
                _a_ku_stride_bytes = 16  # 8 bf16 * 2 bytes
                col_offset_base_bytes = lane_div_16 * arith.index(_a_sublane_stride)

                by_n = by * arith.index(tile_n)
                _waves_per_group = 4 // _k_batch
                # NOTE: shadows enclosing-scope `n_per_wave`; rename to avoid
                # making it look like a local of moe_gemm1 (which would break
                # the generic body's enclosing-scope reference under FlyDSL).
                _a16_n_per_wave = tile_n // _waves_per_group
                num_acc_n = _a16_n_per_wave // 16
                c_n_per_wave = arith.index(_a16_n_per_wave)
                if const_expr(_k_batch > 1):
                    _wave_group = wave_id // arith.index(_waves_per_group)
                    _wave_in_group = wave_id % arith.index(_waves_per_group)
                    n_tile_base = _wave_in_group * c_n_per_wave
                    # A LDS: each wave group reads different K half
                    _k_half_bytes = (tile_k // _k_batch) * elem_bytes
                    col_offset_base_bytes = (
                        col_offset_base_bytes + _wave_group * arith.index(_k_half_bytes)
                    )
                else:
                    _wave_group = None
                    wave_mod_4 = wave_id % arith.index(4)
                    n_tile_base = wave_mod_4 * c_n_per_wave

                n_intra_gate = []
                n_blk_gate = []
                col_g_list = []
                inter_idx = arith.index(inter_dim)
                c_n0_static = experts * (2 * inter_dim) // 16
                layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))

                if const_expr(not _single_b):
                    # Separated mode: gate and up are distinct N regions.
                    n_intra_up = []
                    n_blk_up = []
                    for ni in range_constexpr(num_acc_n):
                        offset = arith.index(ni * 16)
                        col_g = by_n + n_tile_base + offset + lane_mod_16
                        col_g_list.append(col_g)

                        row_gate = expert_off_idx + col_g
                        row_up = row_gate + inter_idx

                        coord_gate = idx2crd(row_gate, layout_n_blk_intra)
                        n_blk_gate.append(layout_get(coord_gate, 0))
                        n_intra_gate.append(layout_get(coord_gate, 1))

                        coord_up = idx2crd(row_up, layout_n_blk_intra)
                        n_blk_up.append(layout_get(coord_up, 0))
                        n_intra_up.append(layout_get(coord_up, 1))
                else:
                    # gate_only / gate_up_interleave: single B stream
                    n_intra_up = None
                    n_blk_up = None
                    for ni in range_constexpr(num_acc_n):
                        offset = arith.index(ni * 16)
                        global_n = by_n + n_tile_base + offset + lane_mod_16
                        gate_row_w = expert_off_idx + global_n

                        coord_gate = idx2crd(gate_row_w, layout_n_blk_intra)
                        n_blk_gate.append(layout_get(coord_gate, 0))
                        n_intra_gate.append(layout_get(coord_gate, 1))

                    if const_expr(gate_up_interleave and not _is_splitk):
                        if const_expr(_gui_xwave_fuse):
                            if const_expr(_k_batch > 1):
                                # Split_k xwave: all waves share same output cols
                                _gui_col_g = by_n // arith.index(2) + lane_mod_16
                            else:
                                # 4-wave xwave: per-pair output cols
                                # pair (0,1)→cols[0:15], pair (2,3)→cols[16:31]
                                _xw_pair_off = (
                                    wave_id // arith.index(2)
                                ) * c_n_per_wave
                                _gui_col_g = (
                                    by_n // arith.index(2) + _xw_pair_off + lane_mod_16
                                )
                            col_g_list.append(_gui_col_g)
                        else:
                            # Standard pair fusion: pairs of N subtiles → one output col
                            # Renamed from `pack_N` to avoid shadowing the enclosing scope.
                            _a16_pack_N = 2
                            _gui_num_acc_n_out = num_acc_n // _a16_pack_N
                            for _gui_i in range_constexpr(_gui_num_acc_n_out):
                                _gui_offset = arith.index(_gui_i * 16)
                                _gui_col_g = (
                                    (by_n + n_tile_base) // arith.index(2)
                                    + _gui_offset
                                    + lane_mod_16
                                )
                                col_g_list.append(_gui_col_g)
                    else:
                        # gate_only or interleave+splitk: output covers full N
                        for ni in range_constexpr(num_acc_n):
                            offset = arith.index(ni * 16)
                            col_g = by_n + n_tile_base + offset + lane_mod_16
                            col_g_list.append(col_g)

                m_repeat = tile_m // 16
                k_unroll = (tile_k_bytes // 64) // _k_batch

                if const_expr(not _is_splitk and _k_batch == 1 and model_dim_pad > 0):
                    _pad_k_elems = model_dim_pad % tile_k
                else:
                    _pad_k_elems = 0
                _pad_ku_skip = _pad_k_elems // 32
                _tail_ku = k_unroll - _pad_ku_skip
                _tail_k0_count = (_tail_ku + 3) // 4 if _pad_ku_skip > 0 else None

                # Each dwordx4 covers 128 K elements; per-group count
                _k_per_dwordx4 = 128
                _k0_count = (tile_k // _k_per_dwordx4) // _k_batch

                # Scale mni for gate (and up if separated)
                scale_mni_gate = []
                scale_n_pack_gate = []
                for ni in range_constexpr(num_acc_n):
                    n_gate = expert_off_idx + by_n + n_tile_base + arith.index(ni * 16)
                    if const_expr(not _single_b):
                        n_gate_phys = n_gate
                    else:
                        n_gate_phys = n_gate
                    scale_mni_gate.append(n_gate_phys // arith.index(32))
                    scale_n_pack_gate.append(
                        (n_gate_phys // arith.index(16)) % arith.index(2)
                    )

                if const_expr(not _single_b):
                    scale_mni_up = []
                    scale_n_pack_up = []
                    for ni in range_constexpr(num_acc_n):
                        n_up = (
                            expert_off_idx
                            + by_n
                            + n_tile_base
                            + arith.index(ni * 16)
                            + inter_idx
                        )
                        scale_mni_up.append(n_up // arith.index(32))
                        scale_n_pack_up.append(
                            (n_up // arith.index(16)) % arith.index(2)
                        )
                else:
                    scale_mni_up = None
                    scale_n_pack_up = None

                def _load_scale_i32(scale_ku_idx, mni_val, scale_klane=None):
                    _klane = scale_klane if scale_klane is not None else lane_div_16
                    idx = (
                        mni_val * layout_b_scale.stride_n0
                        + scale_ku_idx * layout_b_scale.stride_k0
                        + _klane * layout_b_scale.stride_klane
                        + lane_mod_16
                    )
                    return buffer_ops.buffer_load(sw_rsrc, idx, vec_width=1, dtype=i32)

                def _extract_e8m0_f32_dynamic(packed_i32, byte_pos_idx):
                    """Extract E8M0 byte at runtime byte_pos and decode to f32."""
                    shift = arith.index_cast(i32, byte_pos_idx) * arith.constant(
                        8, type=i32
                    )
                    byte_i32 = arith.shrui(packed_i32, shift) & arith.constant(
                        0xFF, type=i32
                    )
                    scale_bits = arith.shli(byte_i32, arith.constant(23, type=i32))
                    return arith.bitcast(f32, scale_bits)

                def _get_scale_f32(base_k, ku, ni, mni_list, n_pack_list, scale_cache):
                    # CK addressing: adj_ku = base_k//32 + (ku//4)*4 + lane_div_16
                    # scale_klane = lane_div_16, k_pack_sub = (ku//4) % 2
                    _k0_blk = ku // 4
                    adj_ku = (
                        base_k // arith.index(32)
                        + arith.index(_k0_blk * 4)
                        + lane_div_16
                    )
                    scale_klane_rt = lane_div_16
                    k_pack_sub_rt = (adj_ku // arith.index(4)) % arith.index(2)
                    s_ku = adj_ku // arith.index(8)

                    cache_key = (_k0_blk, ni, id(mni_list))
                    if cache_key not in scale_cache:
                        scale_cache[cache_key] = _load_scale_i32(
                            s_ku, mni_list[ni], scale_klane=scale_klane_rt
                        )
                    packed = scale_cache[cache_key]
                    n_pack_sub_val = n_pack_list[ni]
                    byte_pos_even = k_pack_sub_rt * arith.index(2)
                    byte_pos_odd = byte_pos_even + arith.index(1)
                    scale_even = _extract_e8m0_f32_dynamic(packed, byte_pos_even)
                    scale_odd = _extract_e8m0_f32_dynamic(packed, byte_pos_odd)
                    n_pack_is_zero = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        arith.index_cast(i32, n_pack_sub_val),
                        arith.constant(0, type=i32),
                    )
                    return arith.select(n_pack_is_zero, scale_even, scale_odd)

                def load_b_raw(base_k, blk_list, intra_list, k0_limit=_k0_count):
                    """Load raw FP4 data via dwordx4. Returns raw_v4[k0_idx][ni] = vec4_i32."""
                    raw_all = []
                    for k0_idx in range_constexpr(k0_limit):
                        raw_k0 = []
                        k_off = base_k + arith.index(k0_idx * _k_per_dwordx4)
                        for ni in range_constexpr(num_acc_n):
                            v4 = load_b_raw_mxfp4_dwordx4(
                                buffer_ops,
                                arith,
                                vector,
                                arg_b=arg_w,
                                b_rsrc=w_rsrc,
                                layout_b=layout_b,
                                base_k=k_off,
                                n_blk=blk_list[ni],
                                n_intra=intra_list[ni],
                                lane_div_16=lane_div_16,
                                elem_type=w_elem,
                                kpack_bytes=kpack_bytes,
                                cache_modifier=2,
                            )
                            raw_k0.append(v4)
                        raw_all.append(raw_k0)
                    return raw_all

                def load_b_scale(base_k, mni_list, n_pack_list, ku_limit=k_unroll):
                    """Load scales for all ku × ni. Returns scales[ku][ni] = f32."""
                    scale_cache = {}
                    scales = []
                    for ku in range_constexpr(ku_limit):
                        scales_ku = []
                        for ni in range_constexpr(num_acc_n):
                            sf = _get_scale_f32(
                                base_k, ku, ni, mni_list, n_pack_list, scale_cache
                            )
                            scales_ku.append(sf)
                        scales.append(scales_ku)
                    return scales

                def load_all_b_raw(base_k, k0_limit=_k0_count, ku_limit=k_unroll):
                    """Load raw B (dwordx4) + scales for gate (and up if separated)."""
                    g_scales = load_b_scale(
                        base_k, scale_mni_gate, scale_n_pack_gate, ku_limit=ku_limit
                    )
                    u_scales = None
                    if const_expr(not _single_b):
                        u_scales = load_b_scale(
                            base_k, scale_mni_up, scale_n_pack_up, ku_limit=ku_limit
                        )

                    g_v4 = load_b_raw(
                        base_k, n_blk_gate, n_intra_gate, k0_limit=k0_limit
                    )
                    u_v4 = None
                    if const_expr(not _single_b):
                        u_v4 = load_b_raw(
                            base_k, n_blk_up, n_intra_up, k0_limit=k0_limit
                        )
                    return g_v4, g_scales, u_v4, u_scales

                def store_x_tile_to_lds(vec_x_in_parts, lds_buffer):
                    _lds_base_zero = arith.index(0)
                    for i in range_constexpr(num_x_loads):
                        row_local = x_row_local[i]
                        col_local_i32 = x_col_local_i32[i]
                        if const_expr(x_load_bytes >= 16):
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
                        else:
                            lds_store_8b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_buffer,
                                vec8_ty=x_vec_x_ty,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=_lds_base_zero,
                                vec_part_i32x2=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )

                def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_buffer):
                    col_base_swz_bytes = swizzle_xor16(
                        curr_row_a_lds, col_base_bytes, k_blocks16
                    )
                    col_base_swz = col_base_swz_bytes // arith.index(int(elem_bytes))
                    idx_a16 = crd2idx((curr_row_a_lds, col_base_swz), layout_lds)
                    loaded_a16 = vector.load_op(vec16_x, lds_buffer, [idx_a16])
                    a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                    a0 = vector.extract(
                        a_i64x2, static_position=[0], dynamic_position=[]
                    )
                    a1 = vector.extract(
                        a_i64x2, static_position=[1], dynamic_position=[]
                    )
                    return a0, a1

                def _a_col_bytes_for_ku(ku_val):
                    """CK-style A col address: L*64 + (ku%4)*16 + (ku//4)*256."""
                    _k0_blk = ku_val // 4
                    _ku_in = ku_val % 4
                    return col_offset_base_bytes + arith.index(
                        _ku_in * _a_ku_stride_bytes + _k0_blk * 256
                    )

                _can_full_preload = m_repeat == 1
                _m_preload = (
                    (k_unroll * m_repeat)
                    if _can_full_preload
                    else min(2, k_unroll * m_repeat)
                )
                _total_a_slots = k_unroll * m_repeat

                def preload_a_from_lds(lds_buffer):
                    """Load first _m_preload A tiles from LDS into VGPRs."""
                    a_tiles = [None] * _m_preload
                    for pl in range_constexpr(_m_preload):
                        _pl_mi = pl % m_repeat
                        _pl_ku = pl // m_repeat
                        col = _a_col_bytes_for_ku(_pl_ku)
                        row = row_a_lds + arith.index(_pl_mi * 16)
                        a_tiles[pl] = lds_load_packs_k64(row, col, lds_buffer)
                    return a_tiles

                def _mfma_k32(acc_in, a0, a1, b0, b1):
                    a_v2 = vector.from_elements(vec2_i64, [a0, a1])
                    a_v8 = vector.bitcast(vec8_bf16, a_v2)
                    b_v2 = vector.from_elements(vec2_i64, [b0, b1])
                    b_v8 = vector.bitcast(vec8_bf16, b_v2)
                    return mfma_f32_bf16_k32(vec4_f32, [a_v8, b_v8, acc_in, 0, 0, 0])

                def compute_tile(
                    acc_gate_in,
                    acc_up_in,
                    g_v4,
                    g_scales,
                    u_v4,
                    u_scales,
                    a_preloaded,
                    cur_lds_buffer,
                    next_lds_buffer,
                    ku_count=k_unroll,
                ):
                    """Compute GEMM tile with preloaded A.

                    Full preload (m_repeat=1): ni→ku, all A in VGPRs.
                    Partial preload (m_repeat>1): ku→mi→ni, m_preload=2
                    pipeline, reload remaining A from cur_lds_buffer.

                    ku_count: number of k_unroll iterations to execute
                    (< k_unroll for the last tile when K-padding is active).

                    Returns: (gate_list, up_list, a_tiles_next).
                    a_tiles_next has _m_preload tiles from next_lds_buffer.
                    """
                    gate_list = list(acc_gate_in)
                    up_list = list(acc_up_in) if acc_up_in is not None else None
                    a_tiles_next = [None] * _m_preload

                    if _can_full_preload:
                        # --- Full preload: ni → ku → mi ---
                        _is_last_ni = num_acc_n - 1

                        for ni in range_constexpr(num_acc_n):
                            for ku in range_constexpr(ku_count):
                                _k0_idx = ku // 4
                                _ku_in_k0 = ku % 4

                                g_raw_ku = vector.extract(
                                    g_v4[_k0_idx][ni],
                                    static_position=[_ku_in_k0],
                                    dynamic_position=[],
                                )
                                gb0, gb1 = unpack_b_mxfp4_bf16(
                                    g_raw_ku,
                                    arith,
                                    vector,
                                    scale_f32=g_scales[ku][ni],
                                )
                                if const_expr(up_list is not None):
                                    u_raw_ku = vector.extract(
                                        u_v4[_k0_idx][ni],
                                        static_position=[_ku_in_k0],
                                        dynamic_position=[],
                                    )
                                    ub0, ub1 = unpack_b_mxfp4_bf16(
                                        u_raw_ku,
                                        arith,
                                        vector,
                                        scale_f32=u_scales[ku][ni],
                                    )

                                for mi in range_constexpr(m_repeat):
                                    _flat = ku * m_repeat + mi
                                    a0, a1 = a_preloaded[_flat]
                                    acc_idx = mi * num_acc_n + ni
                                    gate_list[acc_idx] = _mfma_k32(
                                        gate_list[acc_idx],
                                        a0,
                                        a1,
                                        gb0,
                                        gb1,
                                    )
                                    if const_expr(up_list is not None):
                                        up_list[acc_idx] = _mfma_k32(
                                            up_list[acc_idx],
                                            a0,
                                            a1,
                                            ub0,
                                            ub1,
                                        )

                                if next_lds_buffer is not None and ni == _is_last_ni:
                                    for mi in range_constexpr(m_repeat):
                                        _flat = ku * m_repeat + mi
                                        _nxt_col = _a_col_bytes_for_ku(ku)
                                        _nxt_row = row_a_lds + arith.index(mi * 16)
                                        a_tiles_next[_flat] = lds_load_packs_k64(
                                            _nxt_row, _nxt_col, next_lds_buffer
                                        )
                    else:
                        # --- bm=32: full load from cur_lds inside compute ---
                        _local_a_slots = ku_count * m_repeat
                        all_a = [None] * _local_a_slots
                        for ku in range_constexpr(ku_count):
                            for mi in range_constexpr(m_repeat):
                                _col = _a_col_bytes_for_ku(ku)
                                _row = row_a_lds + arith.index(mi * 16)
                                all_a[ku * m_repeat + mi] = lds_load_packs_k64(
                                    _row, _col, cur_lds_buffer
                                )

                        for ni in range_constexpr(num_acc_n):
                            for ku in range_constexpr(ku_count):
                                _k0_idx = ku // 4
                                _ku_in_k0 = ku % 4

                                g_raw_ku = vector.extract(
                                    g_v4[_k0_idx][ni],
                                    static_position=[_ku_in_k0],
                                    dynamic_position=[],
                                )
                                gb0, gb1 = unpack_b_mxfp4_bf16(
                                    g_raw_ku,
                                    arith,
                                    vector,
                                    scale_f32=g_scales[ku][ni],
                                )
                                if const_expr(up_list is not None):
                                    u_raw_ku = vector.extract(
                                        u_v4[_k0_idx][ni],
                                        static_position=[_ku_in_k0],
                                        dynamic_position=[],
                                    )
                                    ub0, ub1 = unpack_b_mxfp4_bf16(
                                        u_raw_ku,
                                        arith,
                                        vector,
                                        scale_f32=u_scales[ku][ni],
                                    )

                                for mi in range_constexpr(m_repeat):
                                    _flat = ku * m_repeat + mi
                                    a0, a1 = all_a[_flat]
                                    acc_idx = mi * num_acc_n + ni
                                    gate_list[acc_idx] = _mfma_k32(
                                        gate_list[acc_idx],
                                        a0,
                                        a1,
                                        gb0,
                                        gb1,
                                    )
                                    if const_expr(up_list is not None):
                                        up_list[acc_idx] = _mfma_k32(
                                            up_list[acc_idx],
                                            a0,
                                            a1,
                                            ub0,
                                            ub1,
                                        )

                        # Load next round's preload from next_lds
                        if next_lds_buffer is not None:
                            for pl in range_constexpr(_m_preload):
                                _pl_mi = pl % m_repeat
                                _pl_ku = pl // m_repeat
                                _nxt_col = _a_col_bytes_for_ku(_pl_ku)
                                _nxt_row = row_a_lds + arith.index(_pl_mi * 16)
                                a_tiles_next[pl] = lds_load_packs_k64(
                                    _nxt_row, _nxt_col, next_lds_buffer
                                )

                    return gate_list, up_list, a_tiles_next

                rocdl.sched_barrier(0)

                _b_stream_mult = 1 if _single_b else 2

                def hot_loop_scheduler():
                    """CK-style scheduler: interleave MFMA, DS_READ, VMEM_READ."""
                    # Constants matching CK:
                    # dsread_per_wg = WG_M * WG_K * sizeof(bf16) / 64 / VectorLoadSize(16) = 16*32*2/64/16 = 1
                    _dsread_per_wg = 1
                    _mfma_per_wg = 1
                    _NIterPerWarp = num_acc_n * _b_stream_mult  # 2 or 4
                    _mfma_perM_perK = _NIterPerWarp * _mfma_per_wg

                    _HalfMIter = (m_repeat + 1) // 2

                    _Aload_num_perK = (
                        _dsread_per_wg * m_repeat
                    )  # num buffer_load_lds per K iter
                    _Aload_rep = max((_Aload_num_perK + m_repeat - 1) // m_repeat, 1)
                    _Bload_num_perK = num_acc_n * _b_stream_mult  # B loads per K iter
                    _Bload_rep = max(
                        (_Bload_num_perK + _HalfMIter - 1) // _HalfMIter, 1
                    )

                    for _ku in range_constexpr(k_unroll):
                        for _mi in range_constexpr(m_repeat):
                            _dsread_perM = _dsread_per_wg
                            _load_perM = 0

                            if const_expr(_mi < _HalfMIter):
                                _load_perM = (
                                    _Aload_rep
                                    if (
                                        _Aload_num_perK
                                        - (m_repeat - 1 - _mi) * _Aload_rep
                                    )
                                    > 0
                                    else 0
                                ) + (
                                    _Bload_rep
                                    if (
                                        _Bload_num_perK
                                        - (_HalfMIter - 1 - _mi) * _Bload_rep
                                    )
                                    > 0
                                    else 0
                                )
                            else:
                                _load_perM = (
                                    _Aload_rep
                                    if (
                                        _Aload_num_perK
                                        - (m_repeat - 1 - _mi) * _Aload_rep
                                    )
                                    > 0
                                    else 0
                                )

                            _sum_data = _dsread_perM + _load_perM
                            _round_data = max(
                                (_sum_data + _mfma_perM_perK - 1) // _mfma_perM_perK, 1
                            )

                            # Build instruction order: 2=VMEM, 3=DS_READ
                            _inst_order = []
                            _max_data = max(_load_perM, _dsread_perM)
                            for _j in range_constexpr(_max_data):
                                if const_expr(_load_perM > _j):
                                    _inst_order.append(2)
                                if const_expr(_dsread_perM > _j):
                                    _inst_order.append(3)
                            # Pad to mfma_perM_perK * round_data
                            _pad_len = _mfma_perM_perK * _round_data - len(_inst_order)
                            _inst_order.extend([0] * _pad_len)

                            for _nj in range_constexpr(_mfma_perM_perK):
                                if const_expr(_nj == 0):
                                    _inst_idx = 0
                                elif const_expr(_nj == 1):
                                    _inst_idx = (
                                        _mfma_perM_perK - 2
                                        if _mfma_perM_perK > 2
                                        else 1
                                    )
                                elif const_expr(_nj == 2):
                                    _inst_idx = _mfma_perM_perK - 1
                                else:
                                    _inst_idx = _mfma_perM_perK - _nj

                                rocdl.sched_mfma(1)

                                for _r in range_constexpr(_round_data):
                                    if const_expr(_r % 2 == 0):
                                        _oi = _inst_idx + _r * _mfma_perM_perK
                                    else:
                                        _oi = (_r + 1) * _mfma_perM_perK - 1 - _inst_idx
                                    if const_expr(_oi < len(_inst_order)):
                                        if const_expr(_inst_order[_oi] == 2):
                                            rocdl.sched_vmem(1)
                                        elif _inst_order[_oi] == 3:
                                            rocdl.sched_dsrd(1)

                    if const_expr(_Aload_num_perK == 0):
                        rocdl.sched_vmem(1)
                    rocdl.sched_barrier(0)

                def _s1_barrier(vmcnt=63, lgkmcnt=63):
                    """s_waitcnt + s_barrier via inline asm (bypasses LLVM)."""
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

                # ---- CK-style constants ----
                # NOTE: original A16W4 path computed `_vmcnt_before_barrier =
                # num_x_loads` here but never actually consumed it; the value
                # is dead code.  It was removed because (a) it shadows the
                # generic body's enclosing-scope binding, and (b) it would
                # otherwise leak as a closure local of `moe_gemm1`.

                # Split-K: base K offset
                if const_expr(_is_splitk):
                    bz = gpu.block_id("z")
                    k_base = bz * arith.index(_k_dim)
                else:
                    k_base = arith.index(0)

                # Intra-WG split-K: B offset per wave group
                if const_expr(_k_batch > 1):
                    _k_half = tile_k // _k_batch
                    _wg_k_off = _wave_group * arith.index(_k_half)
                else:
                    _wg_k_off = arith.index(0)

                # ---- CK-style pipeline: HEAD ----
                # DMA A[0] → pong (full tile_k, shared by all wave groups)
                k0 = k_base
                prefetch_x_to_lds(k0, lds_x_pong)
                rocdl.sched_barrier(0)

                # Load B[0] (raw + scale, per-group K range)
                g_raw_ping, g_sc_ping, u_raw_ping, u_sc_ping = load_all_b_raw(
                    k0 + _wg_k_off
                )

                rocdl.sched_barrier(0)

                # DMA A[1] → ping
                _k1 = k_base + arith.index(tile_k)

                prefetch_x_to_lds(_k1, lds_x_ping)
                rocdl.sched_barrier(0)

                # Init C
                acc_gate = [acc_init] * (num_acc_n * m_repeat)
                acc_up = [acc_init] * (num_acc_n * m_repeat) if not _single_b else None

                # Wait for all DMA + barrier to sync all threads
                rocdl.s_waitcnt(0)
                gpu.barrier()
                rocdl.sched_barrier(0)

                # Preload A[0] from pong LDS → VGPRs (safe: all threads' DMA done)
                a_cur = preload_a_from_lds(lds_x_pong)
                rocdl.sched_barrier(0)

                total_tiles = int(_k_dim) // int(tile_k)
                pair_iters = max((total_tiles - 2) // 2, 0)

                for pair_i in range_constexpr(pair_iters):
                    k_iv = k_base + arith.index(pair_i * (tile_k * 2))

                    # ---- Half 2i ----
                    _k_b1 = k_iv + arith.index(tile_k)
                    g_raw_pong, g_sc_pong, u_raw_pong, u_sc_pong = load_all_b_raw(
                        _k_b1 + _wg_k_off
                    )

                    acc_gate, acc_up, a_next = compute_tile(
                        acc_gate,
                        acc_up,
                        g_raw_ping,
                        g_sc_ping,
                        u_raw_ping,
                        u_sc_ping,
                        a_cur,
                        lds_x_pong,
                        lds_x_ping,
                    )

                    _k_a2 = k_iv + arith.index(tile_k * 2)
                    prefetch_x_to_lds(_k_a2, lds_x_pong)
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    rocdl.sched_barrier(0)
                    a_cur = a_next

                    # ---- Half 2i+1 ----
                    _k_b2 = k_iv + arith.index(tile_k * 2)
                    g_raw_ping, g_sc_ping, u_raw_ping, u_sc_ping = load_all_b_raw(
                        _k_b2 + _wg_k_off
                    )

                    acc_gate, acc_up, a_next = compute_tile(
                        acc_gate,
                        acc_up,
                        g_raw_pong,
                        g_sc_pong,
                        u_raw_pong,
                        u_sc_pong,
                        a_cur,
                        lds_x_ping,
                        lds_x_pong,
                    )

                    _k_a3 = k_iv + arith.index(tile_k * 3)
                    prefetch_x_to_lds(_k_a3, lds_x_ping)
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    rocdl.sched_barrier(0)
                    a_cur = a_next

                # ---- TAIL: last 2 tiles ----
                # Load B for tail-1 (partial if K-padding active)
                k_tail1 = k_base + arith.index(_k_dim) - arith.index(tile_k)
                if const_expr(_pad_ku_skip > 0):
                    g_raw_pong, g_sc_pong, u_raw_pong, u_sc_pong = load_all_b_raw(
                        k_tail1 + _wg_k_off,
                        k0_limit=_tail_k0_count,
                        ku_limit=_tail_ku,
                    )
                else:
                    g_raw_pong, g_sc_pong, u_raw_pong, u_sc_pong = load_all_b_raw(
                        k_tail1 + _wg_k_off
                    )

                # GEMM tail-0: use a_cur (from pong), load a_next from ping
                acc_gate, acc_up, a_next = compute_tile(
                    acc_gate,
                    acc_up,
                    g_raw_ping,
                    g_sc_ping,
                    u_raw_ping,
                    u_sc_ping,
                    a_cur,
                    lds_x_pong,
                    lds_x_ping,
                )
                hot_loop_scheduler()
                rocdl.s_waitcnt(0)
                # _s1_barrier()

                # GEMM tail-1: use a_next (from ping), no next round
                # When K-padding is active, only compute the valid portion
                acc_gate, acc_up, _ = compute_tile(
                    acc_gate,
                    acc_up,
                    g_raw_pong,
                    g_sc_pong,
                    u_raw_pong,
                    u_sc_pong,
                    a_next,
                    lds_x_ping,
                    None,
                    ku_count=_tail_ku if _pad_ku_skip > 0 else k_unroll,
                )

                # ---- Intra-WG split-K reduce via LDS ----
                if const_expr(_k_batch > 1):
                    _num_accs = num_acc_n * m_repeat
                    _has_up = not _single_b
                    _streams = 2 if _has_up else 1
                    # 4 f32 per vec4_f32 acc × _num_accs × _streams per thread
                    _f32_per_thread = 4 * _num_accs * _streams
                    _reduce_stride = arith.index(_f32_per_thread)
                    # 4 waves × 64 threads × _f32_per_thread
                    _reduce_f32_total = 4 * 64 * _f32_per_thread

                    reduce_lds = SmemPtr(
                        base_ptr_pong,
                        lds_pong_offset,
                        T.f32,
                        shape=(_reduce_f32_total,),
                    ).get()

                    tx_local = lane_id
                    _lds_base = (
                        wave_id * arith.index(64) * _reduce_stride
                        + tx_local * _reduce_stride
                    )

                    for _ai in range_constexpr(_num_accs):
                        _off = _lds_base + arith.index(_ai * 4)
                        vector.store(acc_gate[_ai], reduce_lds, [_off])
                    if _has_up:
                        for _ai in range_constexpr(_num_accs):
                            _off = _lds_base + arith.index(_num_accs * 4 + _ai * 4)
                            vector.store(acc_up[_ai], reduce_lds, [_off])

                    gpu.barrier()

                    _partner_wave = (
                        wave_id + arith.index(_waves_per_group)
                    ) % arith.index(4)
                    _partner_base = (
                        _partner_wave * arith.index(64) * _reduce_stride
                        + tx_local * _reduce_stride
                    )

                    for _ai in range_constexpr(_num_accs):
                        _off = _partner_base + arith.index(_ai * 4)
                        other_acc = vector.load_op(vec4_f32, reduce_lds, [_off])
                        acc_gate[_ai] = arith.addf(acc_gate[_ai], other_acc)
                    if _has_up:
                        for _ai in range_constexpr(_num_accs):
                            _off = _partner_base + arith.index(_num_accs * 4 + _ai * 4)
                            other_up = vector.load_op(vec4_f32, reduce_lds, [_off])
                            acc_up[_ai] = arith.addf(acc_up[_ai], other_up)

                    # Cross-wave gate-up fusion for GUI + split_k + num_acc_n=1
                    if const_expr(_gui_xwave_fuse):
                        gpu.barrier()
                        for _ai in range_constexpr(_num_accs):
                            _off = _lds_base + arith.index(_ai * 4)
                            vector.store(acc_gate[_ai], reduce_lds, [_off])
                        gpu.barrier()

                        # Read partner within same wave group (0↔1)
                        _xw_partner_in_grp = arith.index(1) - _wave_in_group
                        _xw_partner_wave = (
                            _wave_group * arith.index(_waves_per_group)
                            + _xw_partner_in_grp
                        )
                        _xw_partner_base = (
                            _xw_partner_wave * arith.index(64) * _reduce_stride
                            + tx_local * _reduce_stride
                        )

                        # silu(gate) * up — gate is wave_in_group=0, up is wave_in_group=1
                        _is_gate_wave = arith.cmpi(
                            arith.CmpIPredicate.eq,
                            arith.index_cast(i32, _wave_in_group),
                            arith.constant(0, type=T.i32),
                        )
                        for _ai in range_constexpr(_num_accs):
                            _xw_off = _xw_partner_base + arith.index(_ai * 4)
                            _xw_partner_acc = vector.load_op(
                                vec4_f32,
                                reduce_lds,
                                [_xw_off],
                            )
                            _fused_elems = [None] * 4
                            for _e in range_constexpr(4):
                                own_e = vector.extract(
                                    acc_gate[_ai],
                                    static_position=[_e],
                                    dynamic_position=[],
                                )
                                par_e = vector.extract(
                                    _xw_partner_acc,
                                    static_position=[_e],
                                    dynamic_position=[],
                                )
                                g_e = arith.select(_is_gate_wave, own_e, par_e)
                                u_e = arith.select(_is_gate_wave, par_e, own_e)
                                _fused_elems[_e] = _act_elem(g_e, u_e)
                            acc_gate[_ai] = vector.from_elements(vec4_f32, _fused_elems)

                # ---- Cross-wave gate-up fusion (4-wave, no split_k) ----
                if const_expr(_gui_xwave_fuse and _k_batch <= 1):
                    _xw_num_accs = num_acc_n * m_repeat
                    _xw_f32_per_thr = 4 * _xw_num_accs
                    _xw_stride = arith.index(_xw_f32_per_thr)
                    _xw_total = 4 * 64 * _xw_f32_per_thr

                    xw_lds = SmemPtr(
                        base_ptr_pong,
                        lds_pong_offset,
                        T.f32,
                        shape=(_xw_total,),
                    ).get()

                    _xw_tx = lane_id
                    _xw_base = (
                        wave_id * arith.index(64) * _xw_stride + _xw_tx * _xw_stride
                    )

                    for _ai in range_constexpr(_xw_num_accs):
                        _off = _xw_base + arith.index(_ai * 4)
                        vector.store(acc_gate[_ai], xw_lds, [_off])

                    gpu.barrier()

                    # Partner: wave_id XOR 1 → pairs (0,1) and (2,3)
                    _xw_wid_i32 = arith.index_cast(i32, wave_id)
                    _xw_pid_i32 = arith.xori(
                        _xw_wid_i32,
                        arith.constant(1, type=T.i32),
                    )
                    _xw_partner = arith.index_cast(T.index, _xw_pid_i32)
                    _xw_pbase = (
                        _xw_partner * arith.index(64) * _xw_stride + _xw_tx * _xw_stride
                    )

                    # even wave_id = gate, odd = up
                    _xw_is_gate = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        arith.andi(
                            _xw_wid_i32,
                            arith.constant(1, type=T.i32),
                        ),
                        arith.constant(0, type=T.i32),
                    )

                    for _ai in range_constexpr(_xw_num_accs):
                        _xw_off = _xw_pbase + arith.index(_ai * 4)
                        _xw_pacc = vector.load_op(
                            vec4_f32,
                            xw_lds,
                            [_xw_off],
                        )
                        _fused = [None] * 4
                        for _e in range_constexpr(4):
                            own_e = vector.extract(
                                acc_gate[_ai],
                                static_position=[_e],
                                dynamic_position=[],
                            )
                            par_e = vector.extract(
                                _xw_pacc,
                                static_position=[_e],
                                dynamic_position=[],
                            )
                            g_e = arith.select(_xw_is_gate, own_e, par_e)
                            u_e = arith.select(_xw_is_gate, par_e, own_e)
                            _fused[_e] = _act_elem(g_e, u_e)
                        acc_gate[_ai] = vector.from_elements(
                            vec4_f32,
                            _fused,
                        )

                # ---- Bias: add to raw accumulators before activation ----
                if enable_bias and not _is_splitk:
                    _bias_gate_vals = []
                    for _ni in range_constexpr(num_acc_n):
                        _bn = by_n + n_tile_base + arith.index(_ni * 16) + lane_mod_16
                        _bias_gate_vals.append(
                            buffer_ops.buffer_load(
                                bias_rsrc, expert_off_idx + _bn, vec_width=1, dtype=f32
                            )
                        )
                    for _mi in range_constexpr(m_repeat):
                        for _ni in range_constexpr(num_acc_n):
                            _aidx = _mi * num_acc_n + _ni
                            _bsplat = vector.splat(vec4_f32, _bias_gate_vals[_ni])
                            acc_gate[_aidx] = arith.addf(acc_gate[_aidx], _bsplat)

                    if not (gate_only or gate_up_interleave):
                        _bias_up_vals = []
                        for _ni in range_constexpr(num_acc_n):
                            _bn = (
                                by_n + n_tile_base + arith.index(_ni * 16) + lane_mod_16
                            )
                            _bias_up_vals.append(
                                buffer_ops.buffer_load(
                                    bias_rsrc,
                                    expert_off_idx + inter_idx + _bn,
                                    vec_width=1,
                                    dtype=f32,
                                )
                            )
                        for _mi in range_constexpr(m_repeat):
                            for _ni in range_constexpr(num_acc_n):
                                _aidx = _mi * num_acc_n + _ni
                                _bsplat = vector.splat(vec4_f32, _bias_up_vals[_ni])
                                acc_up[_aidx] = arith.addf(acc_up[_aidx], _bsplat)

                # ---- Epilogue ----
                tokens_i32_v = tokens_i32
                topk_i32_v = topk_i32
                inter_i32_v = arith.constant(inter_dim, type=T.i32)
                inter2_i32_v = arith.constant(inter_dim * 2, type=T.i32)
                mask24_i32 = arith.constant(0xFFFFFF, type=T.i32)

                # Fuse activation for non-split-K paths
                if const_expr(
                    gate_up_interleave and not _is_splitk and _gui_xwave_fuse
                ):
                    acc = acc_gate
                    _eff_num_acc_n = 1
                    _eff_tile_n = _gui_out_tile_n
                elif const_expr(gate_up_interleave and not _is_splitk):
                    pack_N_e = 2
                    _gui_out_n = num_acc_n // pack_N_e
                    acc = [None] * (_gui_out_n * m_repeat)
                    for _mi in range_constexpr(m_repeat):
                        for _ni in range_constexpr(_gui_out_n):
                            _g_idx = _mi * num_acc_n + _ni * pack_N_e
                            _u_idx = _g_idx + 1
                            _out_idx = _mi * _gui_out_n + _ni
                            acc[_out_idx] = _act_vec4(
                                acc_gate[_g_idx], acc_gate[_u_idx]
                            )
                    _eff_num_acc_n = _gui_out_n
                    _eff_tile_n = _gui_out_tile_n
                elif const_expr(not _is_splitk):
                    acc = [None] * (num_acc_n * m_repeat)
                    for _mi in range_constexpr(m_repeat):
                        for _ni in range_constexpr(num_acc_n):
                            _aidx = _mi * num_acc_n + _ni
                            acc[_aidx] = _act_vec4(acc_gate[_aidx], acc_up[_aidx])
                    _eff_num_acc_n = num_acc_n
                    _eff_tile_n = tile_n
                else:
                    acc = acc_gate
                    _eff_num_acc_n = num_acc_n
                    _eff_tile_n = tile_n

                col_i32_list = []
                for ni in range_constexpr(len(col_g_list)):
                    col_i32_list.append(arith.index_cast(i32, col_g_list[ni]))

                # Row stride for output indexing
                if const_expr(_is_splitk):
                    _out_row_stride_i32 = inter2_i32_v
                else:
                    _out_row_stride_i32 = inter_i32_v

                _sk_n_offset = [0]

                zero_i32_sk = arith.constant(0, type=T.i32)
                c4_i32_sk = arith.constant(4, type=T.i32)

                def _decode_fused2_at_row(row):
                    """Load fused2 from sorted_rsrc[row]; decode t/s and validity flags."""
                    fused2 = buffer_ops.buffer_load(
                        sorted_rsrc, row, vec_width=1, dtype=i32
                    )
                    t2 = fused2 & mask24_i32
                    s2 = fused2 >> 24
                    row_i32 = arith.index_cast(i32, row)
                    row_ok = arith.cmpi(arith.CmpIPredicate.ult, row_i32, num_valid_i32)
                    t_ok = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                    s_ok = arith.cmpi(arith.CmpIPredicate.ult, s2, topk_i32_v)
                    return fused2, t2, s2, row_ok, t_ok, s_ok

                if const_expr(_is_splitk):
                    if const_expr(gate_only or gate_up_interleave):
                        # Single-pass atomic: no activation fusion
                        def _splitk_store_row(*, mi: int, ii: int, row_in_tile, row):
                            _, t2, s2, row_ok, t_ok, s_ok = _decode_fused2_at_row(row)
                            all_ok = row_ok & t_ok & s_ok
                            sx = arith.select(
                                all_ok,
                                arith.constant(1.0, type=T.f32),
                                arith.constant(0.0, type=T.f32),
                            )
                            t2_safe = arith.select(
                                all_ok, t2, arith.constant(0, type=T.i32)
                            )
                            s2_safe = arith.select(
                                all_ok, s2, arith.constant(0, type=T.i32)
                            )
                            idx0 = (
                                t2_safe * topk_i32_v + s2_safe
                            ) * _out_row_stride_i32

                            for ni in range_constexpr(_eff_num_acc_n):
                                col_i32 = col_i32_list[ni]
                                acc_idx = mi * _eff_num_acc_n + ni
                                val = vector.extract(
                                    acc[acc_idx],
                                    static_position=[ii],
                                    dynamic_position=[],
                                )
                                val = val * sx
                                idx_elem = idx0 + col_i32
                                byte_off = idx_elem * c4_i32_sk
                                rocdl.raw_ptr_buffer_atomic_fadd(
                                    val,
                                    out_rsrc,
                                    byte_off,
                                    zero_i32_sk,
                                    zero_i32_sk,
                                )

                        mfma_epilog(
                            use_cshuffle=False,
                            arith=arith,
                            range_constexpr=range_constexpr,
                            m_repeat=m_repeat,
                            lane_div_16=lane_div_16,
                            bx_m=bx_m,
                            body_row=_splitk_store_row,
                        )
                    else:
                        # Separated split-K: two-pass atomic (gate then up)
                        def _splitk_sep_store_row(
                            *, mi: int, ii: int, row_in_tile, row
                        ):
                            _, t2, s2, row_ok, t_ok, s_ok = _decode_fused2_at_row(row)
                            all_ok = row_ok & t_ok & s_ok
                            sx = arith.select(
                                all_ok,
                                arith.constant(1.0, type=T.f32),
                                arith.constant(0.0, type=T.f32),
                            )
                            t2_safe = arith.select(
                                all_ok, t2, arith.constant(0, type=T.i32)
                            )
                            s2_safe = arith.select(
                                all_ok, s2, arith.constant(0, type=T.i32)
                            )
                            idx0 = (
                                t2_safe * topk_i32_v + s2_safe
                            ) * _out_row_stride_i32 + arith.constant(
                                _sk_n_offset[0], type=T.i32
                            )

                            for ni in range_constexpr(num_acc_n):
                                col_i32 = col_i32_list[ni]
                                acc_idx = mi * num_acc_n + ni
                                val = vector.extract(
                                    acc[acc_idx],
                                    static_position=[ii],
                                    dynamic_position=[],
                                )
                                val = val * sx
                                idx_elem = idx0 + col_i32
                                byte_off = idx_elem * c4_i32_sk
                                rocdl.raw_ptr_buffer_atomic_fadd(
                                    val,
                                    out_rsrc,
                                    byte_off,
                                    zero_i32_sk,
                                    zero_i32_sk,
                                )

                        # Pass 1: gate (offset 0)
                        acc = acc_gate
                        _sk_n_offset[0] = 0
                        mfma_epilog(
                            use_cshuffle=False,
                            arith=arith,
                            range_constexpr=range_constexpr,
                            m_repeat=m_repeat,
                            lane_div_16=lane_div_16,
                            bx_m=bx_m,
                            body_row=_splitk_sep_store_row,
                        )
                        gpu.barrier()
                        # Pass 2: up (offset inter_dim)
                        acc = acc_up
                        _sk_n_offset[0] = inter_dim
                        mfma_epilog(
                            use_cshuffle=False,
                            arith=arith,
                            range_constexpr=range_constexpr,
                            m_repeat=m_repeat,
                            lane_div_16=lane_div_16,
                            bx_m=bx_m,
                            body_row=_splitk_sep_store_row,
                        )
                elif const_expr(_use_cshuffle_epilog):
                    if const_expr(lds_out is None):
                        raise RuntimeError("CShuffle requires lds_out.")

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
                        _, _, _, _, t_valid, _ = _decode_fused2_at_row(row)

                        if const_expr(doweight_stage1):
                            tw = buffer_ops.buffer_load(
                                sorted_w_rsrc, row, vec_width=1, dtype=f32
                            )

                        _out_ty = out_mlir()
                        _vec1_out = T.vec(1, _out_ty)
                        for ni in range_constexpr(_eff_num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            acc_idx = mi * _eff_num_acc_n + ni
                            y = vector.extract(
                                acc[acc_idx],
                                static_position=[ii],
                                dynamic_position=[],
                            )
                            if const_expr(doweight_stage1):
                                y = y * tw
                            y16 = arith.trunc_f(_out_ty, y)
                            lds_idx = row_base_lds + col_local
                            v1 = vector.from_elements(_vec1_out, [y16])
                            vector.store(v1, lds_out, [lds_idx], alignment=2)

                    def precompute_row(*, row_local, row):
                        _, t2, s2, row_ok, t_ok, s_ok = _decode_fused2_at_row(row)
                        row_valid = row_ok & t_ok & s_ok
                        row_byte_base = (t2 * topk_i32_v + s2) * inter_i32_v
                        return (row_byte_base, row_valid)

                    def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                        _, _, _, _, t_valid, _ = _decode_fused2_at_row(row)
                        _if_valid = scf.IfOp(t_valid)
                        with _if_then(_if_valid):
                            idx0 = row_ctx
                            col_i32 = arith.index_cast(i32, col_g0)
                            idx_out = idx0 + col_i32
                            buffer_ops.buffer_store(frag, out_rsrc, idx_out)

                    _cs_by_n = by_n
                    _cs_n_tile_base = n_tile_base
                    if const_expr(gate_up_interleave):
                        _cs_by_n = by_n // arith.index(2)
                        _cs_n_tile_base = n_tile_base // arith.index(2)

                    _cs_nlane = min(32, _eff_tile_n // 4)
                    mfma_epilog(
                        use_cshuffle=True,
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=_eff_tile_n,
                        e_vec=4,
                        cshuffle_nlane=_cs_nlane,
                        m_repeat=m_repeat,
                        num_acc_n=_eff_num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=_cs_by_n,
                        n_tile_base=_cs_n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=out_mlir(),
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                    )
                else:
                    # Direct epilogue (non-split-K, non-cshuffle)
                    def _stage1_store_row(*, mi: int, ii: int, row_in_tile, row):
                        _, t2, s2, row_ok, t_ok, s_ok = _decode_fused2_at_row(row)
                        row_valid = row_ok & t_ok & s_ok

                        idx0 = (t2 * topk_i32_v + s2) * inter_i32_v

                        if const_expr(doweight_stage1):
                            tw = buffer_ops.buffer_load(
                                sorted_w_rsrc, row, vec_width=1, dtype=f32
                            )

                        _if_valid = scf.IfOp(row_valid)
                        with _if_then(_if_valid):
                            for ni in range_constexpr(_eff_num_acc_n):
                                col_i32 = col_i32_list[ni]
                                acc_idx = mi * _eff_num_acc_n + ni
                                y = vector.extract(
                                    acc[acc_idx],
                                    static_position=[ii],
                                    dynamic_position=[],
                                )
                                if const_expr(doweight_stage1):
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

            return

        tokens_in = arith.index_cast(ir.IndexType.get(), i32_tokens_in.ir_value())
        n_in = arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
        k_in = arith.index_cast(ir.IndexType.get(), i32_k_in.ir_value())
        size_expert_ids_in = arith.index_cast(
            ir.IndexType.get(), i32_size_expert_ids_in.ir_value()
        )

        x_elem = T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)
        f32 = T.f32
        i32 = T.i32
        i64 = T.i64
        vec4_f32 = T.vec(4, f32)
        vec16_elems = 16 if a_elem_bytes == 1 else 8
        vec16_x = T.vec(vec16_elems, x_elem)
        vec2_i64 = T.vec(2, i64)

        acc_init = arith.constant_vector(0.0, vec4_f32)

        # --- Stage1 dimension mapping ---
        # X: [tokens, model_dim] -- M = sorted tokens, K = model_dim
        # W: [E*2*inter_dim, model_dim] gate portion -- N = inter_dim
        # Out: [tokens*topk, inter_dim]

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
            _c1_sw = arith.constant(1, index=True)
            _c_tn_sw = arith.constant(tile_n, index=True)
            _c_idp_sw = arith.constant(2 * inter_dim_pad, index=True)
            if const_expr(mock_gate_only or gate_up_interleave):
                _gx = (n_in - _c_idp_sw + _c_tn_sw - _c1_sw) / _c_tn_sw
            else:
                _c2_sw = arith.constant(2, index=True)
                _gx = (
                    (n_in - _c_idp_sw + _c2_sw * _c_tn_sw - _c1_sw) / _c_tn_sw / _c2_sw
                )
            _c_pm_sw = arith.constant(persist_m, index=True)
            _gy = (size_expert_ids_in + _c_pm_sw - _c1_sw) / _c_pm_sw

            _linear_id = bx_persist * _gx + by
            _num_wgs = _gx * _gy

            _c_xcds = arith.constant(_NUM_XCDS_S1, index=True)
            _wgs_per_xcd = _num_wgs / _c_xcds
            _wgid = (_linear_id % _c_xcds) * _wgs_per_xcd + (_linear_id / _c_xcds)

            _WGM_S1 = xcd_swizzle
            _c_wgm = arith.constant(_WGM_S1, index=True)
            _num_wgid_in_group = _c_wgm * _gx
            _group_id = _wgid / _num_wgid_in_group
            _first_pid_m = _group_id * _c_wgm
            _remaining_m = _gy - _first_pid_m
            _cmp_m = arith.cmpi(CmpIPredicate.ult, _remaining_m, _c_wgm)
            _group_size_m = arith.select(_cmp_m, _remaining_m, _c_wgm)

            _wgid_in_group = _wgid % _num_wgid_in_group
            bx_persist = _first_pid_m + (_wgid_in_group % _group_size_m)
            by = _wgid_in_group / _group_size_m

        by_n = by * arith.constant(tile_n, index=True)

        k_base_idx = arith.index(0)
        if const_expr(_is_splitk):
            bz = gpu.block_id("z")  # K-batch id
            k_base_idx = bz * arith.constant(_k_dim, index=True)

        k_blocks16 = arith.constant(_eff_tile_k_bytes // 16, index=True)
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
        numids_rsrc = buffer_ops.create_buffer_resource(
            arg_num_valid_ids,
            max_size=False,
            num_records_bytes=arith.constant(4, type=T.i32),
        )
        num_valid_i32 = buffer_ops.buffer_load(
            numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=T.i32
        )

        sx_rsrc = 1
        sw_rsrc = 1
        if const_expr(not (is_f16_a or a_scale_one)):
            # A scale: [sorted_size, model_dim/32] pre-scattered by caller
            c32 = arith.constant(32, index=True)
            kblk = k_in / c32
            sx_nbytes_idx = sorted_m * kblk
            sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
            sx_rsrc = buffer_ops.create_buffer_resource(
                arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_i32
            )

        if const_expr(not is_f16_b):
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
        bias_rsrc = (
            buffer_ops.create_buffer_resource(arg_bias, max_size=False)
            if enable_bias
            else None
        )

        # Sorted-scale buffer resource for fused mxfp4 quantization
        _sorted_scale_cols = inter_dim // 32
        _sorted_scale_cols_i32 = arith.constant(_sorted_scale_cols, type=T.i32)
        sorted_scale_rsrc = None
        if const_expr(_need_sort):
            sorted_scale_rsrc = buffer_ops.create_buffer_resource(
                arg_out_scale_sorted, max_size=False
            )

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
        expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=T.i32)
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
                idx_elem = idx_i32 if a_elem_bytes == 1 else (idx_i32 * arith.index(2))
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
                    (base_k / c_a_pack) * arith.constant(int(a_elem_bytes), index=True)
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
                    _gui_c_offset = arith.constant(_gui_offset, index=True)
                    _gui_col_g = (
                        (by_n + n_tile_base) // arith.constant(2, index=True)
                        + _gui_c_offset
                        + lane_mod_16
                    )
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
                c64 = arith.constant(64, index=True)
                base_k_bytes = base_k * arith.constant(int(b_elem_bytes), index=True)
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
                b0 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
                b1 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])
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
                _col_base = (
                    by_n + n_tile_base + arith.constant(_ni * 16 * pack_N, index=True)
                )
                _gate_mni = (expert_off_idx + _col_base) // arith.constant(
                    32, index=True
                )
                _gate_scale_bases.append(
                    _gate_mni * layout_b_scale.stride_n0 + _scale_lane_elem
                )
                if const_expr(not mock_gate_only and not gate_up_interleave):
                    _up_mni = (
                        expert_off_idx + inter_idx + _col_base
                    ) // arith.constant(32, index=True)
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

            _c16_idx = arith.constant(16, index=True)
            _c2_idx = arith.constant(2, index=True)
            _scale_mask_lo = arith.constant(0xFF, type=T.i32)

            _m_half_idx = arith.constant(0, type=T.i32)
            _m_half_i32 = arith.constant(0, type=T.i32)
            _scale_shift = arith.constant(0, type=T.i32)
            _scale_shift_hi = arith.constant(0, type=T.i32)
            _n_half_idx = arith.constant(0, type=T.i32)
            _n_half_i32 = arith.constant(0, type=T.i32)
            _bscale_shift = arith.constant(0, type=T.i32)
            _bscale_shift_hi = arith.constant(0, type=T.i32)
            if const_expr(pack_M < scale_mn_pack):
                _m_half_idx = (bx_m // _c16_idx) % _c2_idx
                _m_half_i32 = arith.index_cast(T.i32, _m_half_idx)
                _scale_shift = _m_half_i32 * arith.constant(8, type=T.i32)
                _scale_shift_hi = _scale_shift + arith.constant(16, type=T.i32)

            if const_expr(pack_N < scale_mn_pack):
                _n_half_idx = (n_tile_base // _c16_idx) % _c2_idx
                _n_half_i32 = arith.index_cast(T.i32, _n_half_idx)
                _bscale_shift = _n_half_i32 * arith.constant(8, type=T.i32)
                _bscale_shift_hi = _bscale_shift + arith.constant(16, type=T.i32)

            def _rearrange_a_scale(raw_i32):
                """Rearrange scale bytes for pack_M=1: extract m_half's k0,k1 bytes."""
                if const_expr(pack_M >= scale_mn_pack):
                    return raw_i32
                b_k0 = arith.andi(arith.shrui(raw_i32, _scale_shift), _scale_mask_lo)
                b_k1 = arith.andi(arith.shrui(raw_i32, _scale_shift_hi), _scale_mask_lo)
                return arith.ori(b_k0, arith.shli(b_k1, arith.constant(8, type=T.i32)))

            def _rearrange_b_scale(raw_i32):
                """Rearrange scale bytes for pack_N=1: extract n_half's k0,k1 bytes."""
                if const_expr(pack_N >= scale_mn_pack):
                    return raw_i32
                b_k0 = arith.andi(arith.shrui(raw_i32, _bscale_shift), _scale_mask_lo)
                b_k1 = arith.andi(
                    arith.shrui(raw_i32, _bscale_shift_hi), _scale_mask_lo
                )
                return arith.ori(b_k0, arith.shli(b_k1, arith.constant(8, type=T.i32)))

            if const_expr(a_scale_one):
                _as1_const = arith.constant(0x7F7F7F7F, type=T.i32)
                _as1_vec = vector.from_elements(T.vec(1, T.i32), [_as1_const])

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
                            a_scale_tile.append(
                                vector.from_elements(T.vec(1, T.i32), [s])
                            )
                    for ni in range_constexpr(num_acc_n_packed):
                        gs = buffer_ops.buffer_load(
                            sw_rsrc,
                            _gate_scale_bases[ni] + k_off,
                            vec_width=1,
                            dtype=T.i32,
                            cache_modifier=0,
                        )
                        gs = _rearrange_b_scale(gs)
                        gate_b_scale.append(vector.from_elements(T.vec(1, T.i32), [gs]))
                        if const_expr(not mock_gate_only and not gate_up_interleave):
                            us = buffer_ops.buffer_load(
                                sw_rsrc,
                                _up_scale_bases[ni] + k_off,
                                vec_width=1,
                                dtype=T.i32,
                                cache_modifier=0,
                            )
                            us = _rearrange_b_scale(us)
                            up_b_scale.append(
                                vector.from_elements(T.vec(1, T.i32), [us])
                            )
                return [a_scale_tile, gate_b_scale, up_b_scale]

            _lds_base_zero = arith.index(0)

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
                    c4_idx = arith.index(4)
                    base_k_div4 = (
                        (base_k / c_a_pack)
                        * arith.constant(int(elem_bytes), index=True)
                    ) / arith.index(4)

                    lds_ptr_i64 = None
                    for i in range_constexpr(_num_dma_loads):
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
                            ) + wave_id * arith.constant(
                                _wave_size * _dma_bytes, index=True
                            )
                            lds_ptr_i64 = rocdl.readfirstlane(
                                T.i64, arith.index_cast(T.i64, lds_addr)
                            )
                        else:
                            lds_ptr_i64 = lds_ptr_i64 + arith.constant(
                                total_threads * _dma_bytes, type=T.i64
                            )

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
                col_base_swz_bytes = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
                col_base_swz = (
                    col_base_swz_bytes
                    if elem_bytes == 1
                    else (col_base_swz_bytes / arith.index(2))
                )
                idx_a16 = crd2idx([curr_row_a_lds, col_base_swz], layout_lds)
                loaded_a16 = vector.load_op(vec16_x, lds_buffer, [idx_a16])
                a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
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
                        bias_pf = []
                        for ni in range_constexpr(num_acc_n):
                            if const_expr(gate_up_interleave):
                                _logical_col = (
                                    (by_n + n_tile_base)
                                    // arith.constant(2, index=True)
                                    + arith.constant((ni // 2) * 16, index=True)
                                    + lane_mod_16
                                )
                                _up_off = (
                                    inter_idx
                                    if (ni % 2 == 1)
                                    else arith.constant(0, index=True)
                                )
                                bias_offset = expert_off_idx + _up_off + _logical_col
                            else:
                                global_n = (
                                    by_n
                                    + n_tile_base
                                    + arith.constant(ni * 16, index=True)
                                    + lane_mod_16
                                )
                                bias_offset = expert_off_idx + global_n
                            bias_pf.append(
                                buffer_ops.buffer_load(
                                    bias_rsrc, bias_offset, vec_width=1, dtype=f32
                                )
                            )
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

                c0_i64 = arith.constant(0, type=T.i64)
                vec4_i64 = T.vec(4, T.i64)
                vec8_i32 = T.vec(8, T.i32)

                def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                    v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                    return vector.bitcast(vec8_i32, v4)

                _eff_packed = (ku_count + pack_K - 1) // pack_K
                # B-major: fix B (ni), cycle A (mi) -- B from VMEM stays
                # in registers while A from LDS is repacked per mi.
                for ku128 in range_constexpr(_eff_packed):
                    for ni in range_constexpr(num_acc_n_packed):
                        gate_bs_i32 = gate_b_scale[ku128 * num_acc_n_packed + ni]
                        gate_bs_val = vector.extract(
                            gate_bs_i32,
                            static_position=[0],
                            dynamic_position=[],
                        )
                        if const_expr(not _single_b):
                            up_bs_i32 = up_b_scale[ku128 * num_acc_n_packed + ni]
                            up_bs_val = vector.extract(
                                up_bs_i32, static_position=[0], dynamic_position=[]
                            )
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
                                        a_scale_val = vector.extract(
                                            a_scale_i32,
                                            static_position=[0],
                                            dynamic_position=[],
                                        )
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
                mi_val = arith.constant(mi_idx * 16, index=True)
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
                c0_i64 = arith.constant(0, type=T.i64)
                vec4_i64 = T.vec(4, T.i64)
                vec8_i32 = T.vec(8, T.i32)

                def _pack(x0, x1, x2, x3):
                    v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                    return vector.bitcast(vec8_i32, v4)

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
                _abs_k = k_base_idx + arith.constant(next_k_load, index=True)
                _bk = _abs_k // arith.constant(2, index=True)
                _sk = _abs_k // arith.constant(pack_K * 128, index=True)
                _k_off = _sk * layout_b_scale.stride_k0

                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(_vmcnt_before_barrier)
                _barrier()
                rocdl.sched_barrier(0)

                # DMA A to OTHER buffer (for next half), non-blocking
                _abs_k_dma = k_base_idx + arith.constant(next_k_dma_py, index=True)
                if const_expr(use_async_copy and next_k_dma_py < int(_k_dim)):
                    prefetch_x_to_lds(_abs_k_dma, lds_write)
                if const_expr(not use_async_copy):
                    _x_regs = load_x_tile(_abs_k_dma)

                # ---- Extract previous scale values ----
                _prev_asvs = []
                for _mi_p in range_constexpr(m_repeat_packed):
                    _prev_asvs.append(
                        vector.extract(
                            prev_a_scale[_mi_p],
                            static_position=[0],
                            dynamic_position=[],
                        )
                    )
                _prev_gsv_list = []
                for _gs_ni in range_constexpr(num_acc_n_packed):
                    _prev_gsv_list.append(
                        vector.extract(
                            prev_gate_bs[_gs_ni],
                            static_position=[0],
                            dynamic_position=[],
                        )
                    )
                if const_expr(not _single_b_pipe):
                    _prev_usv_list = []
                    for _us_ni in range_constexpr(num_acc_n_packed):
                        _prev_usv_list.append(
                            vector.extract(
                                prev_up_bs[_us_ni],
                                static_position=[0],
                                dynamic_position=[],
                            )
                        )

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
                        vector.from_elements(
                            T.vec(1, T.i32),
                            [_new_as_list[_mi_p]],
                        )
                    )
                cur_gate_bs = []
                for _gs_ni in range_constexpr(num_acc_n_packed):
                    cur_gate_bs.append(
                        vector.from_elements(T.vec(1, T.i32), [_new_gs_list[_gs_ni]])
                    )
                if const_expr(not _single_b_pipe):
                    cur_up_bs = []
                    for _us_ni in range_constexpr(num_acc_n_packed):
                        cur_up_bs.append(
                            vector.from_elements(
                                T.vec(1, T.i32), [_new_us_list[_us_ni]]
                            )
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
            _k0_scale = k_base_idx // arith.constant(pack_K * 128, index=True)
            a_scale_pong, gate_bs_pong, up_bs_pong = prefetch_ab_scale_tile(_k0_scale)
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
            acc_up = [acc_init] * num_acc_n * m_repeat if not _single_b_pipe else None

            _k1 = k_base_idx + arith.constant(tile_k, index=True)
            rocdl.sched_barrier(0)
            if const_expr(use_async_copy):
                prefetch_x_to_lds(_k1, lds_x_ping)
            else:
                _x_regs_prime = load_x_tile(_k1)
                store_x_tile_to_lds(_x_regs_prime, lds_x_ping)

            _k0_b = k_base_idx // arith.constant(2, index=True)
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

            # _wave_mod2_b = wave_id % arith.constant(2, index=True)
            # _wave_odd = arith.cmpi(
            #     CmpIPredicate.eq, _wave_mod2_b, arith.constant(1, index=True)
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
                _k_tail_rel = arith.constant(_k_dim - tile_k, index=True)
                k_tail1 = k_base_idx + _k_tail_rel
                x_regs_ping = []
                if const_expr(use_async_copy):
                    prefetch_x_to_lds(k_tail1, lds_x_ping)
                else:
                    x_regs_ping = load_x_tile(k_tail1)
                if const_expr(_pad_ku_skip > 0):
                    gate_w_ping, up_w_ping = load_b_tile(
                        k_tail1 // arith.constant(2, index=True),
                        ku_limit=_tail_ku,
                    )
                    a_scale_ping, gate_bs_ping, up_bs_ping = prefetch_ab_scale_tile(
                        k_tail1 // arith.constant(pack_K * 128, index=True),
                        ku_packed_limit=_tail_ku_packed,
                    )
                else:
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

            # Activation helpers (f32 element-wise on vec4_f32)
            def _silu_elem(g):
                """silu(x) = x * sigmoid(x); HW fast path: exp2, rcp"""
                neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                t = g * neg_log2e
                emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                one = arith.constant(1.0, type=f32)
                den = one + emu
                sig = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [den], [], [])
                return g * sig

            def _silu_mul_vec4(gate_v4, up_v4):
                """Element-wise silu(gate) * up on vec4_f32."""
                result_elems = []
                for ei in range_constexpr(4):
                    g = vector.extract(
                        gate_v4, static_position=[ei], dynamic_position=[]
                    )
                    u = vector.extract(up_v4, static_position=[ei], dynamic_position=[])
                    result_elems.append(_silu_elem(g) * u)
                return vector.from_elements(vec4_f32, result_elems)

            def _swiglu_mul_vec4(gate_v4, up_v4):
                """Element-wise swiglu(gate, up) on vec4_f32.
                swiglu(g, u) = g * sigmoid(alpha * g) * (u + 1)
                with clamping: gate <= limit, -limit <= up <= limit.
                """
                result_elems = []
                _alpha = arith.constant(1.702, type=f32)
                _limit = arith.constant(7.0, type=f32)
                _neg_limit = arith.constant(-7.0, type=f32)
                _one = arith.constant(1.0, type=f32)
                _neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                for ei in range_constexpr(4):
                    g = vector.extract(
                        gate_v4, static_position=[ei], dynamic_position=[]
                    )
                    u = vector.extract(up_v4, static_position=[ei], dynamic_position=[])
                    g = arith.minimumf(g, _limit)
                    u = arith.minimumf(u, _limit)
                    u = arith.maximumf(u, _neg_limit)
                    t = g * _alpha * _neg_log2e
                    emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                    den = _one + emu
                    sig = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [den], [], [])
                    result_elems.append(g * sig * (u + _one))
                return vector.from_elements(vec4_f32, result_elems)

            def _act_vec4(gate_v4, up_v4):
                """Dispatch activation based on `act` parameter."""
                if const_expr(act == "swiglu"):
                    return _swiglu_mul_vec4(gate_v4, up_v4)
                else:
                    return _silu_mul_vec4(gate_v4, up_v4)

            # Add bias to raw GEMM accumulators before activation.
            # bias layout: [E, 2*inter_dim] flat f32 (non-interleaved: gate then up).
            # For gate_up_interleave, map physical column to logical bias offset.
            if const_expr(enable_bias and not _is_splitk):
                if const_expr(bias_pf is not None):
                    _bias_gate_vals = bias_pf
                else:
                    _bias_gate_vals = []
                    for _ni in range_constexpr(num_acc_n):
                        if const_expr(gate_up_interleave):
                            _logical_col = (
                                (by_n + n_tile_base) // arith.constant(2, index=True)
                                + arith.constant((_ni // 2) * 16, index=True)
                                + lane_mod_16
                            )
                            _up_off = (
                                inter_idx
                                if (_ni % 2 == 1)
                                else arith.constant(0, index=True)
                            )
                            _bias_off = expert_off_idx + _up_off + _logical_col
                        else:
                            _bn = (
                                by_n
                                + n_tile_base
                                + arith.constant(_ni * 16, index=True)
                                + lane_mod_16
                            )
                            _bias_off = expert_off_idx + _bn
                        _bias_gate_vals.append(
                            buffer_ops.buffer_load(
                                bias_rsrc, _bias_off, vec_width=1, dtype=f32
                            )
                        )
                for _mi in range_constexpr(m_repeat):
                    for _ni in range_constexpr(num_acc_n):
                        _aidx = _mi * num_acc_n + _ni
                        _bsplat = vector.from_elements(
                            vec4_f32, [_bias_gate_vals[_ni]] * 4
                        )
                        acc_gate[_aidx] = arith.addf(acc_gate[_aidx], _bsplat)

                if const_expr(not (mock_gate_only or gate_up_interleave)):
                    _bias_up_vals = []
                    for _ni in range_constexpr(num_acc_n):
                        _bn = (
                            by_n
                            + n_tile_base
                            + arith.constant(_ni * 16, index=True)
                            + lane_mod_16
                        )
                        _bias_up_vals.append(
                            buffer_ops.buffer_load(
                                bias_rsrc,
                                expert_off_idx + inter_idx + _bn,
                                vec_width=1,
                                dtype=f32,
                            )
                        )
                    for _mi in range_constexpr(m_repeat):
                        for _ni in range_constexpr(num_acc_n):
                            _aidx = _mi * num_acc_n + _ni
                            _bsplat = vector.from_elements(
                                vec4_f32, [_bias_up_vals[_ni]] * 4
                            )
                            acc_up[_aidx] = arith.addf(acc_up[_aidx], _bsplat)

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
                        acc[_aidx] = _silu_mul_vec4(acc_gate[_aidx], acc_up[_aidx])

            # ---- Epilogue: CShuffle + direct store (accumulate=False) ----
            # Output: out[(t*topk+s) * inter_dim + col] = silu(gate) * up
            # For split-K: skip silu, output gate/up separately with atomic add
            tw_pf = None
            bias_pf = None
            if const_expr(epilogue_pf is not None):
                _, tw_pf, bias_pf = epilogue_pf

            mask24_i32 = arith.constant(0xFFFFFF)
            topk_i32_v = topk_i32
            tokens_i32_v = tokens_i32

            from flydsl._mlir.dialects import fly as _fly

            _llvm_ptr_ty = ir.Type.parse("!llvm.ptr")
            out_base_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty, arg_out)
            out_base_i64 = llvm.ptrtoint(T.i64, out_base_ptr)
            out_base_idx = arith.index_cast(ir.IndexType.get(), out_base_i64)

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
                    v = vector.extract(
                        acc[acc_idx], static_position=[ii], dynamic_position=[]
                    )
                    if const_expr(_apply_weight):
                        v = v * tw
                    if const_expr(_need_quant):
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
                inter_dim * 2 * out_elem_bytes
                if _is_splitk
                else (
                    inter_dim // 2
                    if _need_fp4
                    else (inter_dim if _need_fp8 else inter_dim * out_elem_bytes)
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

            _c8_i32 = arith.constant(8, type=T.i32)
            _fp_headroom = 2 if _need_fp4 else (8 if _need_fp8 else 0)
            _c_headroom_i32 = arith.constant(_fp_headroom, type=T.i32)

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

            if const_expr(_need_sort):
                _n32_sort = _sorted_scale_cols_i32 * _c32_i32

            # Mutable slot for split-K N-offset (gate=0, up=inter_dim)
            _sk_n_offset = [0]

            def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                fused, row_byte_base = row_ctx
                if const_expr(_need_quant and not _is_splitk):
                    frag_vals = []
                    for i in range_constexpr(_e_vec):
                        frag_vals.append(
                            vector.extract(
                                frag, static_position=[i], dynamic_position=[]
                            )
                        )

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
                    e8m0_biased = arith.maxsi(exp_field - _c_headroom_i32, _c0_i32)

                    quant_exp = _c254_i32 - e8m0_biased
                    quant_scale = (quant_exp << _c23_i32).bitcast(T.f32)

                    if const_expr(_need_fp4):
                        fp4_vals = []
                        for i in range_constexpr(_e_vec):
                            scaled_v = frag_vals[i] * quant_scale
                            fp4_vals.append(_f32_to_e2m1(scaled_v))

                        packed_i32 = fp4_vals[0] | (fp4_vals[1] << _c4_i32)
                        for k in range_constexpr(1, _e_vec // 2):
                            byte_k = fp4_vals[2 * k] | (fp4_vals[2 * k + 1] << _c4_i32)
                            packed_i32 = packed_i32 | (
                                byte_k << arith.constant(k * 8, type=T.i32)
                            )

                        ptr_addr_idx = row_byte_base + col_g0 / arith.constant(
                            2, index=True
                        )
                        out_ptr_v = _idx_to_llvm_ptr(ptr_addr_idx)
                        _pack_bytes = _e_vec // 2
                        if const_expr(_pack_bytes == 1):
                            store_val = arith.TruncIOp(T.i8, packed_i32)
                            store_raw = (
                                store_val._value
                                if hasattr(store_val, "_value")
                                else store_val
                            )
                            llvm.StoreOp(
                                store_raw, out_ptr_v, alignment=1, nontemporal=True
                            )
                        elif const_expr(_pack_bytes == 2):
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

                    elif const_expr(_need_fp8):
                        scaled_vals = []
                        for i in range_constexpr(_e_vec):
                            scaled_vals.append(frag_vals[i] * quant_scale)

                        ptr_addr_idx = row_byte_base + col_g0
                        if const_expr(_e_vec <= 4):
                            packed_i32 = _c0_i32
                            for _w in range_constexpr(_e_vec // 2):
                                packed_i32 = rocdl.cvt_pk_fp8_f32(
                                    T.i32,
                                    scaled_vals[2 * _w],
                                    scaled_vals[2 * _w + 1],
                                    packed_i32,
                                    _w,
                                )
                            out_ptr_v = _idx_to_llvm_ptr(ptr_addr_idx)
                            if const_expr(_e_vec == 2):
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
                            for _wg in range_constexpr(_e_vec // 4):
                                _b = _wg * 4
                                packed_w = _c0_i32
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
                                word_ptr = ptr_addr_idx + arith.constant(
                                    _wg * 4, index=True
                                )
                                out_ptr_v = _idx_to_llvm_ptr(word_ptr)
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
                        is_scale_writer = arith.cmpi(
                            CmpIPredicate.eq, col_g0_i32 & _c31_i32, _c0_i32
                        )
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
                            scf.YieldOp([])
                elif const_expr(_is_splitk):
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

            _frag_elem = (
                ir.F32Type.get()
                if _need_quant
                else (ir.BF16Type.get() if out_is_bf16 else ir.F16Type.get())
            )

            if const_expr(gate_up_interleave and not _is_splitk):
                # gui without splitk: acc has activation applied, halved N
                _gui_eff_n = _gui_out_n
                _gui_tile_n = tile_n // 2
                _gui_cshuffle_nlane = min(32, _gui_tile_n // _e_vec)
                _gui_by_n = by_n / arith.constant(2, index=True)
                _gui_n_tile_base = n_tile_base / arith.constant(2, index=True)
                c_shuffle_epilog(
                    arith=arith,
                    vector=vector,
                    gpu=gpu,
                    scf=scf,
                    range_constexpr=range_constexpr,
                    tile_m=tile_m,
                    tile_n=_gui_tile_n,
                    e_vec=_e_vec,
                    cshuffle_nlane=_gui_cshuffle_nlane,
                    block_size=total_threads,
                    m_repeat=m_repeat,
                    num_acc_n=_gui_eff_n,
                    tx=tx,
                    lane_div_16=lane_div_16,
                    lane_mod_16=lane_mod_16,
                    bx_m=bx_m,
                    by_n=_gui_by_n,
                    n_tile_base=_gui_n_tile_base,
                    lds_out=lds_out,
                    frag_elem_type=_frag_elem,
                    write_row_to_lds=write_row_to_lds,
                    precompute_row=precompute_row,
                    store_pair=store_pair,
                )
            elif const_expr(mock_gate_only or (gate_up_interleave and _is_splitk)):
                # mock_gate_only: single pass, by_n covers full [0, 2*inter_dim)
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
                    lds_out_split=lds_out_B,
                )
            elif const_expr(_is_splitk):
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
                    lds_out_split=lds_out_B,
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
                    lds_out_split=lds_out_B,
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
    # Unified cache key: gate_mode encodes gate_only / gate_up_interleave,
    # so a single tuple covers both A16W4 and generic stage1 paths.  The
    # tag includes A16W4-only fields (`_k_batch` = split_k_intra, `_use_cshuffle_epilog`)
    # which are no-ops for the generic path (constant 1 / unchanged).
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
        _use_cshuffle_epilog,
        persist_m,
        use_async_copy,
        waves_per_eu,
        k_batch,
        gate_mode,
        a_scale_one,
        xcd_swizzle,
        _k_batch,
    )

    # ---- Shared postlude (used by both A16W4 and generic stage1 paths) ----
    # waves_per_eu LDS-clamp: pad LDS occupancy so that exactly waves_per_eu
    # waves fit on each EU (the rest is reserved by inflating ping ptr).
    if waves_per_eu is not None and waves_per_eu >= 1:
        _total_cu_lds = 160 * 1024
        _min_lds = _total_cu_lds // (waves_per_eu + 1) + 1
        _pong_sz = allocator_pong._align(allocator_pong.ptr, 128)
        _ping_sz = allocator_ping._align(allocator_ping.ptr, 128)
        _cur_lds = _pong_sz + _ping_sz
        if _cur_lds < _min_lds:
            allocator_ping.ptr += _min_lds - _cur_lds

    # Shared host launcher (placed outside the if/else so both paths reuse it).
    # `moe_gemm1`, `_cache_tag`, and `total_threads` are bound in either branch
    # above; `allocator_pong/ping`, `mock_gate_only`, and `gate_up_interleave`
    # are bound in the shared prelude.
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
        tile_n_index = arith.constant(tile_n, index=True)
        inter_dim_pad_total = arith.constant(2 * inter_dim_pad, index=True)
        if const_expr(mock_gate_only or gate_up_interleave):
            gx = (inter_in - inter_dim_pad_total + tile_n_index - 1) / tile_n_index
        else:
            gx = (
                (inter_in - inter_dim_pad_total + 2 * tile_n_index - 1)
                / tile_n_index
                / arith.constant(2, index=True)
            )
        if const_expr(is_a16w4_stage1):
            # A16W4 stage1 has no persistent-M scheduling; gy = size_expert_ids.
            gy = arith.index_cast(ir.IndexType.get(), i32_size_expert_ids_in.ir_value())
        else:
            _c_pm_l = arith.constant(persist_m, index=True)
            gy = (
                arith.index_cast(ir.IndexType.get(), i32_size_expert_ids_in.ir_value())
                + _c_pm_l
                - arith.constant(1, index=True)
            ) / _c_pm_l

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
    waves_per_eu: int = 0,
    k_batch: int = 1,
):
    """Compile stage2 kernel (`moe_gemm2`) and return the compiled executable.

    persist_m:
      - > 0: legacy mode -- each CTA processes exactly persist_m consecutive M tiles.
      - <= 0: **persistent mode** -- grid_y = cu_num (auto-detected), each CTA
        round-robins over M tiles with stride cu_num.

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

    `k_batch` is the A16W4 stage2 inter-block split-K factor (grid Z dimension).
    Splits `inter_dim` across `k_batch` CTAs; each CTA atomically accumulates
    its partial result.  Equivalent to `k_batch` in aiter's stage2 kernel.
    """
    is_a16w4 = b_dtype == "fp4" and a_dtype in ("bf16", "fp16")
    # ---- Unified setup (A16W4 + Generic) ----
    gpu_arch = get_hip_arch()
    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")

    _sort_block_m = tile_m if (is_a16w4 or sort_block_m <= 0) else sort_block_m
    if _sort_block_m != tile_m and _sort_block_m % tile_m != 0:
        raise ValueError(
            f"sort_block_m ({_sort_block_m}) must be a multiple of tile_m ({tile_m})"
        )

    if not is_a16w4:
        validate_moe_dtypes(a_dtype, b_dtype)

    is_f16_a = is_a16w4 or (a_dtype == "fp16")
    is_f16_b = (not is_a16w4) and (b_dtype == "fp16")
    is_f8_a = (not is_a16w4) and (a_dtype == "fp8")
    is_f4_a = (not is_a16w4) and (a_dtype == "fp4")
    is_f4_b = b_dtype == "fp4"
    is_int4 = (not is_a16w4) and (b_dtype == "int4")
    is_int8 = False

    _scale_pack_m = 2
    _scale_pack_n = 2
    _scale_pack_k = 2
    pack_M = min(_scale_pack_m, tile_m // 16)
    pack_N = min(_scale_pack_n, tile_n // 64)
    _k_unroll_raw = (int(tile_k) * (2 if is_f16_a else 1)) // 128
    pack_K = min(_scale_pack_k, _k_unroll_raw)

    elem_bytes = 2 if is_a16w4 else 1
    a_elem_bytes = 2 if is_f16_a else 1
    b_elem_bytes = 1
    tile_k_bytes = int(tile_k) * int(a_elem_bytes)

    a_elem_vec_pack = 1 if is_a16w4 else (2 if is_f4_a else 1)
    cbsz = 0 if is_f8_a else 4
    blgp = 4

    # ---- Static B preshuffle strides (compile-time) ----
    _b_kpack_bytes_s = 8 if is_int4 else 16
    _b_kpack_elems_s = _b_kpack_bytes_s // b_elem_bytes
    _b_c_k_s = inter_dim // _scale_pack_k
    _b_c_k0_s = (_b_c_k_s * b_elem_bytes) // 64
    _b_stride_nlane = _b_kpack_elems_s
    _b_stride_klane = 16 * _b_stride_nlane
    _b_stride_k0 = 4 * _b_stride_klane
    _b_stride_n0 = _b_c_k0_s * _b_stride_k0
    assert model_dim % 16 == 0, "model_dim must be divisible by 16"
    _expert_b_stride = (model_dim // 16) * _b_stride_n0

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

    # A16W4-specific: BF16 K32 MFMA
    mfma_f32_bf16_k32 = None
    kpack_bytes = 16  # MXFP4 preshuffle (used by A16W4 body)
    if is_a16w4:
        _mfma_k32_raw = getattr(rocdl, "mfma_f32_16x16x32_bf16_", None)
        if _mfma_k32_raw is None:
            raise AttributeError(
                "BF16 K32 MFMA op not found: expected `rocdl.mfma_f32_16x16x32_bf16_`"
            )
        _split_mfma = rocdl._split_mfma_operands

        def mfma_f32_bf16_k32(result_type, operands, *, loc=None, ip=None):
            a, b, c, cbsz, abid, blgp = _split_mfma(operands, loc=loc)
            return _mfma_k32_raw(result_type, a, b, c, cbsz, abid, blgp, loc=loc, ip=ip)

    def _x_elem_type():
        if is_a16w4:
            return T.bf16
        if is_f4_b:
            return T.f8 if is_f8_a else T.i8
        return T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)

    def _w_elem_type():
        if is_f4_b:
            return T.i8
        return T.f16 if is_f16_b else (T.i8 if is_int8 else T.f8)

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

    if a_elem_vec_pack > 1:
        _eff_lds_stride = lds_stride // a_elem_vec_pack
        _eff_tile_k_bytes = tile_k_bytes // a_elem_vec_pack
    else:
        _eff_lds_stride = lds_stride
        _eff_tile_k_bytes = tile_k_bytes

    if out_is_f32:
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

    if out_is_bf16:
        if not supports_bf16_global_atomics(gpu_arch):
            raise ValueError(
                f"out_dtype='bf16' requires bf16 global atomics, got arch={gpu_arch!r}"
            )

    def out_elem():
        return T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)

    def x_lds_elem():
        if is_a16w4:
            return T.bf16
        return T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)

    # ---- Persist / module name ----
    if is_a16w4:
        _persistent = False
        _cu_num = 0
        persist_m = 1  # A16W4: single M-tile per WG
    else:
        _persistent = persist_m <= 0
        if _persistent:
            from aiter.jit.utils.chip_info import get_cu_num

            _cu_num = get_cu_num()
        else:
            _cu_num = 0

    # Unified module name (cache key). Optional tags are appended only when
    # they deviate from defaults so A8W4/A4W4 keys remain backward-compatible
    # with previously cached binaries (kb/wpe/bias/pad default off).
    _kb_tag = f"_kb{int(k_batch)}" if int(k_batch) > 1 else ""
    _wpe_tag = f"_wpe{waves_per_eu}" if waves_per_eu >= 1 else ""
    _bias_tag = "_bias" if enable_bias else ""
    _pad_tag = (
        f"_mp{model_dim_pad}_ip{inter_dim_pad}"
        if (model_dim_pad or inter_dim_pad)
        else ""
    )
    _sbm_tag = "" if _sort_block_m == tile_m else f"_sbm{_sort_block_m}"
    _pm_tag = f"_persist_cu{_cu_num}" if _persistent else f"_pm{persist_m}"
    _xcd_tag = f"_xcd{xcd_swizzle}" if xcd_swizzle > 0 else ""
    module_name = (
        f"mfma_moe2_a{a_dtype}_w{b_dtype}_{out_s}_cshuffle"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_vscale_fix3"
        f"{_pm_tag}{_sbm_tag}{_xcd_tag}{_kb_tag}{_wpe_tag}{_bias_tag}{_pad_tag}"
    ).replace("-", "_")

    # ---- A16W4 k_batch validation ----
    _k_batch = int(k_batch)
    if is_a16w4 and _k_batch > 1:
        if inter_dim % (_k_batch * tile_k) != 0:
            raise ValueError(
                f"inter_dim={inter_dim} must be divisible by "
                f"k_batch*tile_k={_k_batch * tile_k}"
            )
        _k_dim = inter_dim // _k_batch
        _total_tiles_check = _k_dim // tile_k
        if _total_tiles_check < 2 or _total_tiles_check % 2 != 0:
            raise ValueError(
                f"k_batch={_k_batch}: "
                f"_k_dim/tile_k={_total_tiles_check} must be even and >= 2 "
                f"for the ping-pong pipeline"
            )
    elif is_a16w4:
        _k_dim = inter_dim
    else:
        _k_dim = inter_dim  # generic: unused but defined for uniformity

    # ---- LDS sizing ----
    _single_x_bytes = int(tile_m) * int(_eff_lds_stride) * int(a_elem_bytes)
    _cshuffle_elem_bytes_s2 = 2
    lds_out_bytes = (
        _cshuffle_elem_bytes_s2 * int(tile_m) * int(tile_n)
        if _use_cshuffle_epilog
        else 0
    )
    lds_tid_bytes = int(tile_m) * 4
    _input_elems = _single_x_bytes if a_elem_bytes == 1 else (_single_x_bytes // 2)
    _single_x_elems = _single_x_bytes // int(a_elem_bytes) if is_a16w4 else _input_elems

    _pong_buffer_bytes = max(_single_x_bytes, lds_out_bytes)
    _ping_buffer_bytes = _single_x_bytes

    lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = lds_pong_offset + _pong_buffer_bytes

    # Unified sorted-token cache slot in LDS. tile_m i32 values (== lds_tid_bytes).
    # A16W4 epilogue calls this `lds_sorted_info_offset`; generic uses `_lds_tid_offset_pong`.
    _lds_tid_offset_pong = allocator_pong._align(allocator_pong.ptr, 4)
    allocator_pong.ptr = _lds_tid_offset_pong + lds_tid_bytes

    lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = lds_ping_offset + _ping_buffer_bytes

    if waves_per_eu >= 1:
        _total_cu_lds = 160 * 1024
        _min_lds = _total_cu_lds // (waves_per_eu + 1) + 1
        _pong_sz = allocator_pong._align(allocator_pong.ptr, 128)
        _ping_sz = allocator_ping._align(allocator_ping.ptr, 128)
        _cur_lds = _pong_sz + _ping_sz
        if _cur_lds < _min_lds:
            allocator_ping.ptr += _min_lds - _cur_lds

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
            tokens_in = arith.index_cast(ir.IndexType.get(), i32_tokens_in.ir_value())
            n_in = arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
            k_in = arith.index_cast(ir.IndexType.get(), i32_k_in.ir_value())
            size_expert_ids_in = arith.index_cast(
                ir.IndexType.get(), i32_size_expert_ids_in.ir_value()
            )
            x_elem = T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)
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

            acc_init = (
                arith.constant_vector(0, vec4_i32)
                if is_int8
                else arith.constant_vector(0.0, vec4_f32)
            )

            # A2 layout (flatten token-slot -> M; use i32 for fly.make_shape).
            topk_idx = arith.constant(topk, index=True)
            m_in = tokens_in * topk_idx

            # B preshuffle layout: [experts*model_dim, inter_dim]
            c_n_total = arith.constant(experts * model_dim, index=True)
            mixed_kpack_bytes = 8 if is_int4 else 16
            from .layout_utils import _div_pow2, _mod_pow2

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

            shape_lds = fx.make_shape(tile_m, _eff_lds_stride)
            stride_lds = fx.make_stride(_eff_lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            by = gpu.block_id("x")  # tile along model_dim (N-dim)
            bx_persist = gpu.block_id("y")  # persistent WG index (M-dim)

            if const_expr(xcd_swizzle > 0):
                _NUM_XCDS_S = 8
                _c1_sw = arith.constant(1, index=True)
                _c_tn_sw = arith.constant(tile_n, index=True)
                _c_mdp_sw = arith.constant(model_dim_pad, index=True)
                _gx = (n_in - _c_mdp_sw + _c_tn_sw - _c1_sw) / _c_tn_sw
                if const_expr(_persistent):
                    _gy = arith.constant(_cu_num, index=True)
                else:
                    _c_pm_sw = arith.constant(persist_m, index=True)
                    _gy = (size_expert_ids_in + _c_pm_sw - _c1_sw) / _c_pm_sw

                _linear_id = bx_persist * _gx + by
                _num_wgs = _gx * _gy

                _c_xcds = arith.constant(_NUM_XCDS_S, index=True)
                _wgs_per_xcd = _num_wgs / _c_xcds
                _wgid = (_linear_id % _c_xcds) * _wgs_per_xcd + (_linear_id / _c_xcds)

                _WGM_S = xcd_swizzle
                _c_wgm = arith.constant(_WGM_S, index=True)
                _num_wgid_in_group = _c_wgm * _gx
                _group_id = _wgid / _num_wgid_in_group
                _first_pid_m = _group_id * _c_wgm
                _remaining_m = _gy - _first_pid_m
                _cmp_m = arith.cmpi(CmpIPredicate.ult, _remaining_m, _c_wgm)
                _group_size_m = arith.select(_cmp_m, _remaining_m, _c_wgm)

                _wgid_in_group = _wgid % _num_wgid_in_group
                bx_persist = _first_pid_m + (_wgid_in_group % _group_size_m)
                by = _wgid_in_group / _group_size_m

            # XOR16 swizzle parameter (in bytes; constant, power-of-two in our configs).
            k_blocks16 = arith.constant(_eff_tile_k_bytes // 16, index=True)
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
            c_topk = arith.constant(topk, index=True)

            # X(A2): buffer size in bytes, accounting for FP4 packing (2 elements per byte).
            # fp8/int8: 1 byte per element  -> bytes = tokens*topk * K
            # fp4:      2 elements per byte -> bytes = tokens*topk * K / 2
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
            if const_expr(not bool(accumulate)):
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
            sx_rsrc = 1
            sw_rsrc = 1
            # A activation microscale (e8m0). Only present when activations are
            # FP8 or FP4; A16W4 (bf16/fp16 activations) keeps sx_rsrc=1 sentinel.
            # Scale layout is [sorted_size, K/32] -- caller pre-scatters via
            # moe_mxfp4_sort.
            if const_expr(is_f4_a or is_f8_a):
                kblk = _div_pow2(k_in, 32)
                sx_nbytes_idx = num_valid_idx * kblk
                sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
                sx_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_i32
                )

            # Weight microscale buffer (packed i32 holding e8m0 bytes). All
            # supported configurations (a16w4 / a8w4 / a4w4) have FP4 weights,
            # so this descriptor is always created.
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

            # ---- persist loop ----
            _c0_p = arith.constant(0, index=True)
            _c1_p = arith.constant(1, index=True)

            if const_expr(_persistent):
                # Expert-phase scheduling: contiguous M-tile dispatch.
                # grid_y = cu_num, each CTA handles a contiguous chunk of M-tiles:
                #   [bx_persist * tiles_per_block, ..., (bx_persist+1) * tiles_per_block - 1]
                # Adjacent blocks process adjacent M-tiles -> same expert -> B weight L2 reuse.
                _c_cu = arith.constant(_cu_num, index=True)
                _c_tm_p = arith.constant(tile_m, index=True)
                _num_valid_idx = arith.index_cast(ir.IndexType.get(), num_valid_i32)
                _total_m_tiles = (_num_valid_idx + _c_tm_p - _c1_p) / _c_tm_p
                _tiles_per_block = (_total_m_tiles + _c_cu - _c1_p) / _c_cu
                _i1 = ir.IntegerType.get_signless(1)
                _init_active = arith.constant(1, type=_i1)
                _for_persist = scf.ForOp(_c0_p, _tiles_per_block, _c1_p, [_init_active])
            else:
                # Legacy mode: fixed persist_m consecutive tiles.
                _c_pm = arith.constant(persist_m, index=True)
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

            if const_expr(_persistent):
                _still_active = _for_persist.inner_iter_args[0]
                bx = bx_persist * _tiles_per_block + _mi_p
            else:
                _prev_expert_i32 = _for_persist.inner_iter_args[0]
                _prev_expert_b_base = _for_persist.inner_iter_args[1]
                bx = bx_persist * arith.constant(persist_m, index=True) + _mi_p

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

            if const_expr(_persistent):
                # Absolute B-base: no cross-iteration state needed.
                _expert_b_base = expert_idx * arith.constant(
                    _expert_b_stride, index=True
                )
            else:
                # Legacy incremental B-base: delta = (cur - prev) * stride
                _delta_expert = arith.subi(expert_i32, _prev_expert_i32)
                _delta_expert_idx = arith.index_cast(ir.IndexType.get(), _delta_expert)
                _delta_b = _delta_expert_idx * arith.constant(
                    _expert_b_stride, index=True
                )
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
            if const_expr(pack_M < _scale_pack_m):
                _m_off = _mod_pow2(_div_pow2(bx_m, 16), _scale_pack_m)
                _m_scale_shift_i32 = arith.index_cast(
                    T.i32, _m_off * arith.constant(8, index=True)
                )
            else:
                _m_scale_shift_i32 = None

            def _moe_gemm2_then_body():
                # ===== Shared preamble (consumed by both A16W4 and generic paths) =====
                # Expert offset (in N elements) and tx -> wave/lane decomposition.
                n_idx = arith.constant(model_dim, index=True)
                expert_off_idx = expert_idx * n_idx  # index

                coord_wl = idx2crd(tx, layout_tx_wave_lane)
                wave_id = layout_get(coord_wl, 0)
                lane_id = layout_get(coord_wl, 1)
                coord_l16 = idx2crd(lane_id, layout_lane16)
                lane_div_16 = layout_get(coord_l16, 0)
                lane_mod_16 = layout_get(coord_l16, 1)

                if const_expr(is_a16w4):
                    # ===== A16W4-only type aliases =====
                    # i32 / f32 / i64 / vec4_f32 / vec2_i64 / acc_init come from
                    # the moe_gemm2 setup-level closure (already defined for the
                    # generic path and equivalent for A16W4 since is_int8=False).
                    _a16_x_elem = T.bf16
                    w_elem = T.i8
                    vec1_i32 = T.vec(1, T.i32)
                    vec8_bf16 = T.vec(8, _a16_x_elem)

                    # ===== A16W4 M / X layout (uses runtime k_in) =====
                    topk_idx = arith.index(topk)
                    m_in = tokens_in * topk_idx
                    m_i32_v = arith.index_cast(i32, m_in)
                    k_i32_v = i32_k_in.ir_value()
                    layout_x = fx.make_layout(  # noqa: F841
                        (m_i32_v, k_i32_v), stride=(k_i32_v, 1)
                    )

                    # ===== A16W4 B layout (MXFP4 kpack=16, c_k = k_in // 2) =====
                    c2 = arith.index(2)
                    c_k_packed = k_in // c2
                    b_layout = make_preshuffle_b_layout(
                        arith,
                        c_n=c_n_total,
                        c_k=c_k_packed,
                        kpack_bytes=kpack_bytes,
                        elem_bytes=1,
                    )
                    layout_b = b_layout.layout_b

                    # ===== A16W4 B scale layout (mn_pack=2, k_pack=2, scale_block_size=32) =====
                    _a16_layout_b_scale = make_preshuffle_scale_layout(
                        arith,
                        c_mn=c_n_total,
                        c_k=k_in,
                        mn_pack=2,
                        k_pack=2,
                        elem_bytes=4,
                        scale_block_size=32,
                    )

                    # ===== A16W4 LDS aliases =====
                    # Generic upper preamble preloads sorted_idx into lds_tid; A16W4 path
                    # also writes to it during X-load decode (same offset, same data).
                    lds_sorted_cache = lds_tid

                    # n_idx / expert_off_idx come from the shared preamble above.
                    # For A16W4, _sort_block_m == tile_m so generic's sort_blk = bx
                    # yields the same expert_idx as the original A16W4 path.

                    if const_expr(
                        bytes_per_thread_x >= 16 and bytes_per_thread_x % 16 == 0
                    ):
                        x_load_bytes = 16
                    elif const_expr(
                        bytes_per_thread_x >= 8 and bytes_per_thread_x % 8 == 0
                    ):
                        x_load_bytes = 8
                    elif const_expr(
                        bytes_per_thread_x >= 4 and bytes_per_thread_x % 4 == 0
                    ):
                        x_load_bytes = 4
                    else:
                        raise ValueError(
                            f"bytes_per_thread_x ({bytes_per_thread_x}) must be "
                            f"divisible by 4"
                        )
                    num_x_loads = bytes_per_thread_x // x_load_bytes
                    chunk_i32 = x_load_bytes // 4
                    _a16_vec16_x = T.vec(8, _a16_x_elem)

                    c_k_div4 = (k_in * arith.index(int(elem_bytes))) // arith.index(4)
                    c_k_div4_i32 = arith.index_cast(i32, c_k_div4)
                    layout_x_div4 = fx.make_layout(  # noqa: F841
                        (m_i32_v, c_k_div4_i32), stride=(c_k_div4_i32, 1)
                    )
                    tile_k_dwords = (int(tile_k) * int(elem_bytes)) // 4
                    layout_x_tile_div4 = fx.make_layout(
                        (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                    )
                    c_chunk_i32 = arith.index(chunk_i32)
                    tx_i32_base = tx * c_chunk_i32

                    topk_i32 = arith.constant(topk, type=T.i32)
                    mask24 = arith.constant(0xFFFFFF, type=T.i32)
                    tokens_i32 = arith.index_cast(i32, tokens_in)

                    def x_tile_chunk_coord_i32(i: int):
                        return tile_chunk_coord_i32(
                            arith,
                            tx_i32_base=tx_i32_base,
                            i=i,
                            total_threads=total_threads,
                            layout_tile_div4=layout_x_tile_div4,
                            chunk_i32=chunk_i32,
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
                            sorted_rsrc, sorted_row_i, vec_width=1, dtype=i32
                        )
                        _fused_v1 = vector.from_elements(vec1_i32, [fused_i])
                        vector.store(_fused_v1, lds_sorted_cache, [row_local])
                        t_i32 = fused_i & mask24
                        s_i32 = arith.shrui(fused_i, arith.constant(24, type=T.i32))
                        t_valid = arith.cmpi(arith.CmpIPredicate.ult, t_i32, tokens_i32)
                        s_valid = arith.cmpi(arith.CmpIPredicate.ult, s_i32, topk_i32)
                        ts_valid = t_valid & s_valid
                        t_safe = arith.select(
                            ts_valid, t_i32, arith.constant(0, type=T.i32)
                        )
                        s_safe = arith.select(
                            ts_valid, s_i32, arith.constant(0, type=T.i32)
                        )
                        row_ts_i32 = t_safe * topk_i32 + s_safe
                        row_ts_idx = arith.index_cast(T.index, row_ts_i32)
                        x_row_base_div4.append(row_ts_idx * c_k_div4)

                    # NOTE: load_x_tile is unused on the A16W4 path -- B-cycle
                    # consumes X via dma_x_tile_to_lds + LDS reads. Removed.
                    # wave_id / lane_id / lane_div_16 / lane_mod_16 come from the
                    # shared preamble above.

                    _dma_bytes = 16
                    _wave_size = 64

                    def dma_x_tile_to_lds(base_k, lds_buffer):
                        """Async DMA: global -> LDS via buffer_load_lds, no VGPR."""
                        c4_idx = arith.index(4)
                        base_k_div4 = (
                            base_k * arith.index(int(elem_bytes))
                        ) // arith.index(4)

                        lds_ptr_i64 = None
                        for i in range_constexpr(num_x_loads):
                            row_local_i = x_row_local[i]
                            col_local_i32_i = x_col_local_i32[i]
                            col_local_sw = swizzle_xor16(
                                row_local_i, col_local_i32_i * c4_idx, k_blocks16
                            )
                            row_k_dw = x_row_base_div4[i] + base_k_div4
                            global_byte_idx = row_k_dw * c4_idx + col_local_sw
                            global_offset = arith.index_cast(i32, global_byte_idx)

                            if const_expr(i == 0):
                                lds_addr = memref.extract_aligned_pointer_as_index(
                                    lds_buffer
                                ) + wave_id * arith.constant(
                                    _wave_size * _dma_bytes, index=True
                                )
                                lds_ptr_i64 = rocdl.readfirstlane(
                                    i64, arith.index_cast(i64, lds_addr)
                                )
                            else:
                                lds_ptr_i64 = lds_ptr_i64 + arith.constant(
                                    total_threads * _dma_bytes, type=i64
                                )

                            lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                            lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64)

                            rocdl.raw_ptr_buffer_load_lds(
                                x_rsrc,
                                lds_ptr,
                                arith.constant(_dma_bytes, type=i32),
                                global_offset,
                                arith.constant(0, type=i32),
                                arith.constant(0, type=i32),
                                arith.constant(0, type=i32),
                            )

                    def prefetch_x_to_lds(base_k, lds_buffer):
                        dma_x_tile_to_lds(base_k, lds_buffer)

                    row_a_lds = lane_mod_16
                    _a_sublane_stride = 64  # 32 bf16 * 2 bytes
                    _a_ku_stride_bytes = 16  # 8 bf16 * 2 bytes
                    col_offset_base_bytes = lane_div_16 * arith.index(_a_sublane_stride)

                    by_n = by * arith.index(tile_n)
                    num_waves = 4
                    n_per_wave = tile_n // num_waves
                    num_acc_n = n_per_wave // 16
                    c_n_per_wave = arith.index(n_per_wave)
                    wave_mod_4 = wave_id % arith.index(4)
                    n_tile_base = wave_mod_4 * c_n_per_wave

                    n_intra_list = []
                    n_blk_list = []
                    col_g_list = []
                    c_n0_static = experts * model_dim // 16
                    layout_n_blk_intra = fx.make_layout(
                        (c_n0_static, 16), stride=(16, 1)
                    )
                    for ni in range_constexpr(num_acc_n):
                        offset = arith.index(ni * 16)
                        col_g = by_n + n_tile_base + offset + lane_mod_16
                        col_g_list.append(col_g)

                        row_w = expert_off_idx + col_g
                        coord_w = fx.idx2crd(row_w, layout_n_blk_intra)
                        n_blk_list.append(fx.get(coord_w, 0))
                        n_intra_list.append(fx.get(coord_w, 1))

                    m_repeat = tile_m // 16
                    k_unroll = tile_k_bytes // 64

                    _pad_k_elems = (
                        (inter_dim_pad % tile_k)
                        if (_k_batch == 1 and inter_dim_pad > 0)
                        else 0
                    )
                    _pad_ku_skip = _pad_k_elems // 32
                    _tail_ku = k_unroll - _pad_ku_skip
                    _tail_k0_count = (_tail_ku + 3) // 4 if _pad_ku_skip > 0 else None

                    # ---- Scale index helpers ----
                    # mni for each ni: (expert_off + by_n + n_tile_base + ni*16) // 32
                    scale_mni_list = []
                    scale_n_pack_list = []
                    for ni in range_constexpr(num_acc_n):
                        n_global = (
                            expert_off_idx + by_n + n_tile_base + arith.index(ni * 16)
                        )
                        scale_mni_list.append(n_global // arith.index(32))
                        n_block_16 = n_global // arith.index(16)
                        scale_n_pack_list.append(n_block_16 % arith.index(2))

                    def _load_scale_i32(scale_ku_idx, ni, scale_klane=None):
                        """Load one packed i32 from the scale buffer."""
                        _klane = scale_klane if scale_klane is not None else lane_div_16
                        idx = (
                            scale_mni_list[ni] * _a16_layout_b_scale.stride_n0
                            + scale_ku_idx * _a16_layout_b_scale.stride_k0
                            + _klane * _a16_layout_b_scale.stride_klane
                            + lane_mod_16
                        )
                        return buffer_ops.buffer_load(
                            sw_rsrc, idx, vec_width=1, dtype=i32
                        )

                    def _extract_e8m0_f32_dynamic(packed_i32, byte_pos_idx):
                        """Extract E8M0 byte at runtime byte_pos and decode to f32."""
                        shift = arith.index_cast(i32, byte_pos_idx) * arith.constant(
                            8, type=i32
                        )
                        byte_i32 = arith.shrui(packed_i32, shift) & arith.constant(
                            0xFF, type=i32
                        )
                        scale_bits = arith.shli(byte_i32, arith.constant(23, type=i32))
                        return arith.bitcast(f32, scale_bits)

                    # ---- B Load (dwordx4) + Scale for MXFP4 ----
                    def _get_scale_f32(base_k, ku, ni, scale_cache):
                        """CK addressing for scale: adj_ku = base_k//32 + (ku//4)*4 + lane_div_16."""
                        _k0_blk = ku // 4
                        adj_ku = (
                            base_k // arith.index(32)
                            + arith.index(_k0_blk * 4)
                            + lane_div_16
                        )
                        scale_klane_rt = lane_div_16
                        k_pack_sub_rt = (adj_ku // arith.index(4)) % arith.index(2)
                        s_ku = adj_ku // arith.index(8)

                        cache_key = (_k0_blk, ni)
                        if cache_key not in scale_cache:
                            scale_cache[cache_key] = _load_scale_i32(
                                s_ku, ni, scale_klane=scale_klane_rt
                            )
                        packed = scale_cache[cache_key]
                        n_pack_sub_val = scale_n_pack_list[ni]
                        byte_pos_even = k_pack_sub_rt * arith.index(2)
                        byte_pos_odd = byte_pos_even + arith.index(1)
                        scale_even = _extract_e8m0_f32_dynamic(packed, byte_pos_even)
                        scale_odd = _extract_e8m0_f32_dynamic(packed, byte_pos_odd)
                        n_pack_is_zero = arith.cmpi(
                            arith.CmpIPredicate.eq,
                            arith.index_cast(i32, n_pack_sub_val),
                            arith.constant(0, type=i32),
                        )
                        return arith.select(n_pack_is_zero, scale_even, scale_odd)

                    _k_per_dwordx4 = 128
                    _k0_count = tile_k // _k_per_dwordx4

                    def load_b_raw(base_k, k0_limit=_k0_count):
                        """Load raw FP4 data via dwordx4. Returns raw_v4[k0_idx][ni]."""
                        raw_all = []
                        for k0_idx in range_constexpr(k0_limit):
                            raw_k0 = []
                            k_off = base_k + arith.index(k0_idx * _k_per_dwordx4)
                            for ni in range_constexpr(num_acc_n):
                                v4 = load_b_raw_mxfp4_dwordx4(
                                    buffer_ops,
                                    arith,
                                    vector,
                                    arg_b=arg_w,
                                    b_rsrc=w_rsrc,
                                    layout_b=layout_b,
                                    base_k=k_off,
                                    n_blk=n_blk_list[ni],
                                    n_intra=n_intra_list[ni],
                                    lane_div_16=lane_div_16,
                                    elem_type=w_elem,
                                    kpack_bytes=kpack_bytes,
                                    cache_modifier=2,
                                )
                                raw_k0.append(v4)
                            raw_all.append(raw_k0)
                        return raw_all

                    def load_b_scale_raw(base_k, k0_limit=_k0_count):
                        """Issue scale buffer_loads only (no extraction).
                        Returns (packed_dict, kps_dict):
                          packed_dict: {(k0_blk, ni): packed_i32}
                          kps_dict: {k0_blk: k_pack_sub_rt}
                        """
                        packed_dict = {}
                        kps_dict = {}
                        for k0_blk in range_constexpr(k0_limit):
                            adj_ku = (
                                base_k // arith.index(32)
                                + arith.index(k0_blk * 4)
                                + lane_div_16
                            )
                            scale_klane_rt = lane_div_16
                            kps_dict[k0_blk] = (adj_ku // arith.index(4)) % arith.index(
                                2
                            )
                            s_ku = adj_ku // arith.index(8)
                            for ni in range_constexpr(num_acc_n):
                                packed_dict[(k0_blk, ni)] = _load_scale_i32(
                                    s_ku, ni, scale_klane=scale_klane_rt
                                )
                        return packed_dict, kps_dict

                    def extract_b_scales(packed_dict, kps_dict, ku_limit=k_unroll):
                        """Extract f32 scales from pre-loaded packed i32.
                        Returns scales[ku][ni] = f32.
                        """
                        scales = []
                        for ku in range_constexpr(ku_limit):
                            scales_ku = []
                            _k0_blk = ku // 4
                            k_pack_sub_rt = kps_dict[_k0_blk]
                            for ni in range_constexpr(num_acc_n):
                                packed = packed_dict[(_k0_blk, ni)]
                                n_pack_sub_val = scale_n_pack_list[ni]
                                byte_pos_even = k_pack_sub_rt * arith.index(2)
                                byte_pos_odd = byte_pos_even + arith.index(1)
                                scale_even = _extract_e8m0_f32_dynamic(
                                    packed, byte_pos_even
                                )
                                scale_odd = _extract_e8m0_f32_dynamic(
                                    packed, byte_pos_odd
                                )
                                n_pack_is_zero = arith.cmpi(
                                    arith.CmpIPredicate.eq,
                                    arith.index_cast(i32, n_pack_sub_val),
                                    arith.constant(0, type=i32),
                                )
                                sf = arith.select(n_pack_is_zero, scale_even, scale_odd)
                                scales_ku.append(sf)
                            scales.append(scales_ku)
                        return scales

                    def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_buffer):
                        col_base_swz_bytes = swizzle_xor16(
                            curr_row_a_lds, col_base_bytes, k_blocks16
                        )
                        col_base_swz = col_base_swz_bytes // arith.index(
                            int(elem_bytes)
                        )
                        idx_a16 = crd2idx((curr_row_a_lds, col_base_swz), layout_lds)
                        loaded_a16 = vector.load_op(_a16_vec16_x, lds_buffer, [idx_a16])
                        a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                        a0 = vector.extract(
                            a_i64x2, static_position=[0], dynamic_position=[]
                        )
                        a1 = vector.extract(
                            a_i64x2, static_position=[1], dynamic_position=[]
                        )
                        return a0, a1

                    def _a_col_bytes_for_ku(ku_val):
                        """CK-style A col address: L*64 + (ku%4)*16 + (ku//4)*256."""
                        _k0_blk = ku_val // 4
                        _ku_in = ku_val % 4
                        return col_offset_base_bytes + arith.index(
                            _ku_in * _a_ku_stride_bytes + _k0_blk * 256
                        )

                    _total_a_slots = k_unroll * m_repeat

                    def preload_a_from_lds(lds_buffer, ku_limit=k_unroll):
                        """Load all A tiles for ku_limit × m_repeat from LDS into VGPRs."""
                        a_tiles = [None] * (ku_limit * m_repeat)
                        for ku in range_constexpr(ku_limit):
                            for mi in range_constexpr(m_repeat):
                                col = _a_col_bytes_for_ku(ku)
                                row = row_a_lds + arith.index(mi * 16)
                                a_tiles[ku * m_repeat + mi] = lds_load_packs_k64(
                                    row, col, lds_buffer
                                )
                        return a_tiles

                    def _mfma_k32(acc_in, a0, a1, b0, b1):
                        a_v2 = vector.from_elements(vec2_i64, [a0, a1])
                        a_v8 = vector.bitcast(vec8_bf16, a_v2)
                        b_v2 = vector.from_elements(vec2_i64, [b0, b1])
                        b_v8 = vector.bitcast(vec8_bf16, b_v2)
                        return mfma_f32_bf16_k32(
                            vec4_f32, [a_v8, b_v8, acc_in, 0, 0, 0]
                        )

                    def compute_tile(
                        acc_in,
                        b_v4,
                        b_scales,
                        a_tiles_cur,
                        *,
                        ku_count=k_unroll,
                        prefetch_epilogue: bool = False,
                    ):
                        """Compute GEMM tile with preloaded A (pure compute, no ds_read).

                        Returns: (acc_list, epilogue_pf).
                        """
                        acc_list = list(acc_in)

                        epilogue_pf = None
                        if const_expr(prefetch_epilogue):
                            tw_pf = None
                            if const_expr(doweight_stage2):
                                tw_pf = []
                                lane_div_16_mul4_pf = lane_div_16 * arith.index(4)
                                ii_idx_list_pf = [arith.index(ii) for ii in range(4)]
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
                                                dtype=f32,
                                            )
                                        )
                            epilogue_pf = (None, tw_pf)

                        for ni in range_constexpr(num_acc_n):
                            for ku in range_constexpr(ku_count):
                                _k0_idx = ku // 4
                                _ku_in_k0 = ku % 4

                                b_raw_ku = vector.extract(
                                    b_v4[_k0_idx][ni],
                                    static_position=[_ku_in_k0],
                                    dynamic_position=[],
                                )
                                bb0, bb1 = unpack_b_mxfp4_bf16(
                                    b_raw_ku,
                                    arith,
                                    vector,
                                    scale_f32=b_scales[ku][ni],
                                )

                                for mi in range_constexpr(m_repeat):
                                    _flat = ku * m_repeat + mi
                                    a0, a1 = a_tiles_cur[_flat]

                                    acc_idx = mi * num_acc_n + ni
                                    acc_list[acc_idx] = _mfma_k32(
                                        acc_list[acc_idx],
                                        a0,
                                        a1,
                                        bb0,
                                        bb1,
                                    )

                        return acc_list, epilogue_pf

                    rocdl.sched_barrier(0)

                    def hot_loop_scheduler():
                        """CK-style scheduler: interleave MFMA, DS_READ, VMEM_READ."""
                        _dsread_per_wg = 1
                        _mfma_per_wg = 1
                        _NIterPerWarp = num_acc_n
                        _mfma_perM_perK = _NIterPerWarp * _mfma_per_wg

                        _HalfMIter = (m_repeat + 1) // 2

                        _Aload_num_perK = _dsread_per_wg * m_repeat
                        _Aload_rep = max(
                            (_Aload_num_perK + m_repeat - 1) // m_repeat, 1
                        )
                        _Bload_num_perK = num_acc_n
                        _Bload_rep = max(
                            (_Bload_num_perK + _HalfMIter - 1) // _HalfMIter, 1
                        )

                        for _ku in range_constexpr(k_unroll):
                            for _mi in range_constexpr(m_repeat):
                                _dsread_perM = _dsread_per_wg
                                _load_perM = 0

                                if const_expr(_mi < _HalfMIter):
                                    _load_perM = (
                                        _Aload_rep
                                        if (
                                            _Aload_num_perK
                                            - (m_repeat - 1 - _mi) * _Aload_rep
                                        )
                                        > 0
                                        else 0
                                    ) + (
                                        _Bload_rep
                                        if (
                                            _Bload_num_perK
                                            - (_HalfMIter - 1 - _mi) * _Bload_rep
                                        )
                                        > 0
                                        else 0
                                    )
                                else:
                                    _load_perM = (
                                        _Aload_rep
                                        if (
                                            _Aload_num_perK
                                            - (m_repeat - 1 - _mi) * _Aload_rep
                                        )
                                        > 0
                                        else 0
                                    )

                                _sum_data = _dsread_perM + _load_perM
                                _round_data = max(
                                    (_sum_data + _mfma_perM_perK - 1)
                                    // _mfma_perM_perK,
                                    1,
                                )

                                _inst_order = []
                                _max_data = max(_load_perM, _dsread_perM)
                                for _j in range_constexpr(_max_data):
                                    if const_expr(_load_perM > _j):
                                        _inst_order.append(2)
                                    if const_expr(_dsread_perM > _j):
                                        _inst_order.append(3)
                                _pad_len = _mfma_perM_perK * _round_data - len(
                                    _inst_order
                                )
                                _inst_order.extend([0] * _pad_len)

                                for _nj in range_constexpr(_mfma_perM_perK):
                                    if const_expr(_nj == 0):
                                        _inst_idx = 0
                                    elif const_expr(_nj == 1):
                                        _inst_idx = (
                                            _mfma_perM_perK - 2
                                            if _mfma_perM_perK > 2
                                            else 1
                                        )
                                    elif const_expr(_nj == 2):
                                        _inst_idx = _mfma_perM_perK - 1
                                    else:
                                        _inst_idx = _mfma_perM_perK - _nj

                                    rocdl.sched_mfma(1)

                                    for _r in range_constexpr(_round_data):
                                        if const_expr(_r % 2 == 0):
                                            _oi = _inst_idx + _r * _mfma_perM_perK
                                        else:
                                            _oi = (
                                                (_r + 1) * _mfma_perM_perK
                                                - 1
                                                - _inst_idx
                                            )
                                        if const_expr(_oi < len(_inst_order)):
                                            if const_expr(_inst_order[_oi] == 2):
                                                rocdl.sched_vmem(1)
                                            elif const_expr(_inst_order[_oi] == 3):
                                                rocdl.sched_dsrd(1)

                        if const_expr(_Aload_num_perK == 0):
                            rocdl.sched_vmem(1)
                        rocdl.sched_barrier(0)

                    # ---- intra-block split-K offset ----
                    if const_expr(_k_batch > 1):
                        bz = gpu.block_id("z")
                        k_base = bz * arith.index(_k_dim)
                    else:
                        k_base = arith.index(0)

                    # ---- CK-style pipeline: HEAD (scale prefetch) ----
                    k0 = k_base
                    prefetch_x_to_lds(k0, lds_x_pong)
                    rocdl.sched_barrier(0)

                    sc_raw_cur, kps_cur = load_b_scale_raw(k0)
                    b_v4_cur = load_b_raw(k0)
                    rocdl.sched_barrier(0)

                    _k1 = k_base + arith.index(tile_k)
                    prefetch_x_to_lds(_k1, lds_x_ping)
                    rocdl.sched_barrier(0)

                    acc = [acc_init] * (num_acc_n * m_repeat)

                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    rocdl.sched_barrier(0)
                    a_cur = preload_a_from_lds(lds_x_pong)
                    b_sc_cur = extract_b_scales(sc_raw_cur, kps_cur)
                    gpu.barrier()
                    rocdl.sched_barrier(0)

                    total_tiles = int(_k_dim) // int(tile_k)
                    pair_iters = max((total_tiles - 2) // 2, 0)

                    for pair_i in range_constexpr(pair_iters):
                        k_iv = k_base + arith.index(pair_i * (tile_k * 2))

                        # ---- Half 2i: scale prefetch -> B_raw -> compute -> extract -> barrier ----
                        rocdl.sched_barrier(0)
                        _k_a2 = k_iv + arith.index(tile_k * 2)
                        prefetch_x_to_lds(_k_a2, lds_x_pong)
                        rocdl.sched_barrier(0)
                        _k_b1 = k_iv + arith.index(tile_k)
                        sc_raw_nxt, kps_nxt = load_b_scale_raw(_k_b1)
                        rocdl.sched_barrier(0)

                        b_v4_nxt = load_b_raw(_k_b1)

                        rocdl.sched_barrier(0)
                        acc, _ = compute_tile(
                            acc,
                            b_v4_cur,
                            b_sc_cur,
                            a_cur,
                        )
                        a_next = preload_a_from_lds(lds_x_ping)
                        rocdl.sched_barrier(0)
                        b_sc_nxt = extract_b_scales(sc_raw_nxt, kps_nxt)

                        rocdl.sched_barrier(0)
                        _barrier(lgkmcnt=2)
                        rocdl.sched_barrier(0)
                        a_cur = a_next

                        # ---- Half 2i+1: scale prefetch -> B_raw -> compute -> extract -> barrier ----
                        _k_a3 = k_iv + arith.index(tile_k * 3)
                        prefetch_x_to_lds(_k_a3, lds_x_ping)
                        rocdl.sched_barrier(0)

                        _k_b2 = k_iv + arith.index(tile_k * 2)
                        sc_raw_cur2, kps_cur2 = load_b_scale_raw(_k_b2)
                        b_v4_cur2 = load_b_raw(_k_b2)

                        rocdl.sched_barrier(0)
                        acc, _ = compute_tile(
                            acc,
                            b_v4_nxt,
                            b_sc_nxt,
                            a_cur,
                        )
                        a_next = preload_a_from_lds(lds_x_pong)
                        b_sc_cur2 = extract_b_scales(sc_raw_cur2, kps_cur2)

                        rocdl.sched_barrier(0)
                        _barrier(lgkmcnt=2)
                        rocdl.sched_barrier(0)
                        b_v4_cur, b_sc_cur = b_v4_cur2, b_sc_cur2
                        a_cur = a_next

                    # ---- TAIL: last 2 tiles (scale prefetch) ----
                    k_tail1 = k_base + arith.index(_k_dim) - arith.index(tile_k)
                    if const_expr(_pad_ku_skip > 0):
                        sc_raw_tail, kps_tail = load_b_scale_raw(
                            k_tail1, k0_limit=_tail_k0_count
                        )
                        b_v4_tail = load_b_raw(k_tail1, k0_limit=_tail_k0_count)
                    else:
                        sc_raw_tail, kps_tail = load_b_scale_raw(k_tail1)
                        b_v4_tail = load_b_raw(k_tail1)

                    acc, _ = compute_tile(
                        acc,
                        b_v4_cur,
                        b_sc_cur,
                        a_cur,
                    )
                    if const_expr(_pad_ku_skip > 0):
                        a_next = preload_a_from_lds(lds_x_ping, ku_limit=_tail_ku)
                        b_sc_tail = extract_b_scales(
                            sc_raw_tail, kps_tail, ku_limit=_tail_ku
                        )
                    else:
                        a_next = preload_a_from_lds(lds_x_ping)
                        b_sc_tail = extract_b_scales(sc_raw_tail, kps_tail)

                    hot_loop_scheduler()
                    rocdl.s_waitcnt(0)
                    a_cur = a_next

                    acc, epilogue_pf = compute_tile(
                        acc,
                        b_v4_tail,
                        b_sc_tail,
                        a_cur,
                        ku_count=_tail_ku if _pad_ku_skip > 0 else k_unroll,
                        prefetch_epilogue=True,
                    )

                    # ---- Bias: add to raw accumulators ----
                    if const_expr(enable_bias):
                        _bias_vals = []
                        for _ni in range_constexpr(num_acc_n):
                            _bn = (
                                by_n + n_tile_base + arith.index(_ni * 16) + lane_mod_16
                            )
                            _bias_vals.append(
                                buffer_ops.buffer_load(
                                    bias_rsrc,
                                    expert_off_idx + _bn,
                                    vec_width=1,
                                    dtype=f32,
                                )
                            )
                        for _mi in range_constexpr(m_repeat):
                            for _ni in range_constexpr(num_acc_n):
                                _aidx = _mi * num_acc_n + _ni
                                _bsplat = vector.from_elements(
                                    vec4_f32,
                                    [
                                        _bias_vals[_ni],
                                        _bias_vals[_ni],
                                        _bias_vals[_ni],
                                        _bias_vals[_ni],
                                    ],
                                )
                                acc[_aidx] = arith.addf(acc[_aidx], _bsplat)

                    # ---- Epilogue ----
                    mask24_i32 = arith.constant(0xFFFFFF, type=T.i32)
                    model_i32 = arith.constant(model_dim, type=T.i32)
                    topk_i32_v = topk_i32

                    zero_i32 = arith.constant(0, type=T.i32)
                    c2_i32 = arith.constant(2, type=T.i32)
                    mask_even_i32 = arith.constant(0xFFFFFFFE, type=T.i32)
                    e_vec = _e_vec

                    sw_pf = None
                    tw_pf = None
                    if const_expr(epilogue_pf is not None):
                        sw_pf, tw_pf = epilogue_pf

                    # No per-channel weight scale for MXFP4 (scales already applied in dequant).
                    sw_vals = [  # noqa: F841
                        arith.constant(1.0, type=T.f32)
                    ] * num_acc_n

                    def _a16_decode_lds_fused(idx):
                        """Load fused2 from lds_sorted_cache[idx]; decode (t, s, ts_ok)."""
                        _fv1 = vector.load_op(vec1_i32, lds_sorted_cache, [idx])
                        fused2 = vector.extract(
                            _fv1, static_position=[0], dynamic_position=[]
                        )
                        t = fused2 & mask24_i32
                        s = fused2 >> 24
                        t_ok = arith.cmpi(arith.CmpIPredicate.ult, t, tokens_i32)
                        s_ok = arith.cmpi(arith.CmpIPredicate.ult, s, topk_i32_v)
                        return fused2, t, s, t_ok & s_ok

                    if const_expr(out_is_f32):
                        c4_i32 = arith.constant(4, type=T.i32)

                        def _stage2_row_atomic(*, mi: int, ii: int, row_in_tile, row):
                            fused2, t2, s2, ts_ok = _a16_decode_lds_fused(row_in_tile)
                            t2_safe = arith.select(
                                ts_ok, t2, arith.constant(0, type=T.i32)
                            )
                            sx = arith.select(
                                ts_ok,
                                arith.constant(1.0, type=T.f32),
                                arith.constant(0.0, type=T.f32),
                            )
                            if const_expr(doweight_stage2):
                                tw_idx = (mi * 4) + ii
                                if const_expr(tw_pf is not None):
                                    tw = arith.select(
                                        ts_ok,
                                        tw_pf[tw_idx],
                                        arith.constant(0.0, type=T.f32),
                                    )
                                else:
                                    tw = arith.select(
                                        ts_ok,
                                        buffer_ops.buffer_load(
                                            sorted_w_rsrc, row, vec_width=1, dtype=f32
                                        ),
                                        arith.constant(0.0, type=T.f32),
                                    )
                            idx0 = t2_safe * model_i32

                            for ni in range_constexpr(num_acc_n):
                                col_g = col_g_list[ni]
                                acc_idx = mi * num_acc_n + ni
                                v = vector.extract(
                                    acc[acc_idx],
                                    static_position=[ii],
                                    dynamic_position=[],
                                )
                                v = v * sx
                                if const_expr(doweight_stage2):
                                    v = v * tw
                                col_i32 = arith.index_cast(i32, col_g)
                                idx_elem = idx0 + col_i32
                                byte_off = idx_elem * c4_i32
                                rocdl.raw_ptr_buffer_atomic_fadd(
                                    v,
                                    out_rsrc,
                                    byte_off,
                                    zero_i32,
                                    zero_i32,
                                )

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
                            raise RuntimeError("CShuffle epilogue requires lds_out.")

                        out_base_idx = None
                        if const_expr(out_is_bf16):
                            _llvm_ptr_ty = ir.Type.parse("!llvm.ptr")
                            _out_base_ptr = _fly.extract_aligned_pointer_as_index(
                                _llvm_ptr_ty, arg_out
                            )
                            out_base_idx = arith.index_cast(
                                T.index, llvm.ptrtoint(T.i64, _out_base_ptr)
                            )

                        def _row_valid_for(row, ts_ok):
                            row_i32 = arith.index_cast(i32, row)
                            return (
                                arith.cmpi(
                                    arith.CmpIPredicate.ult, row_i32, num_valid_i32
                                )
                                & ts_ok
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
                            _, _, _, ts_ok = _a16_decode_lds_fused(row_in_tile)
                            row_valid = _row_valid_for(row, ts_ok)  # noqa: F841

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
                                    acc[acc_idx],
                                    static_position=[ii],
                                    dynamic_position=[],
                                )
                                if const_expr(doweight_stage2):
                                    v = v * tw
                                v_out = arith.trunc_f(out_elem(), v)
                                lds_idx = row_base_lds + col_local
                                vec1_out = T.vec(1, out_elem())
                                v1 = vector.from_elements(vec1_out, [v_out])
                                vector.store(v1, lds_out, [lds_idx], alignment=2)

                        def precompute_row(*, row_local, row):
                            fused2, _, _, ts_ok = _a16_decode_lds_fused(row_local)
                            return (fused2, _row_valid_for(row, ts_ok))

                        def store_pair(
                            *, row_local, row, row_ctx, col_pair0, col_g0, frag
                        ):
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
                            if const_expr(out_is_bf16):
                                if const_expr(bool(accumulate)):
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
                                    buffer_ops.buffer_store(
                                        frag, out_rsrc, idx_elem_even
                                    )
                            else:
                                byte_off = idx_elem_even * c2_i32
                                if const_expr(bool(accumulate)):
                                    rocdl.raw_ptr_buffer_atomic_fadd(
                                        frag,
                                        out_rsrc,
                                        byte_off,
                                        zero_i32,
                                        zero_i32,
                                    )
                                else:
                                    buffer_ops.buffer_store(
                                        frag, out_rsrc, idx_elem_even
                                    )

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
                    return

                # n_idx / expert_off_idx already computed in the shared preamble.

                # ---- X gmem->reg prefetch (match preshuffle GEMM mapping) ----
                # Prefer 16B buffer-load (dwordx4). If the per-thread byte count isn't divisible by
                # 16, fall back to 8B (dwordx2) or 4B (dword) loads. For fp16 we require 16B.
                if const_expr(is_f16_a):
                    if const_expr(bytes_per_thread_x % 16 != 0):
                        raise ValueError(
                            f"[fp16] bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 16"
                        )
                    x_load_bytes = 16
                else:
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

                        if const_expr(x_load_bytes == 16):
                            parts.append(vector.bitcast(vec4_i32, x_vec))
                        elif const_expr(x_load_bytes == 8):
                            parts.append(vector.bitcast(vec2_i32, x_vec))
                        else:
                            parts.append(vector.bitcast(vec1_i32, x_vec))
                    return parts

                # tx -> wave/lane decomposition computed in the shared preamble.
                row_a_lds = lane_mod_16

                col_offset_base = lane_div_16 * arith.constant(16, index=True)

                # Dynamic N tiling within block.
                num_waves = 4
                n_per_wave = tile_n // num_waves
                num_acc_n = n_per_wave // 16
                c_n_per_wave = arith.constant(n_per_wave, index=True)
                wave_mod_4 = _mod_pow2(wave_id, 4)
                n_tile_base = wave_mod_4 * c_n_per_wave

                by_n = by * arith.constant(tile_n, index=True)

                if const_expr(pack_N < _scale_pack_n):
                    _global_n_base = expert_off_idx + by_n + n_tile_base
                    _n_off = _mod_pow2(_div_pow2(_global_n_base, 16), _scale_pack_n)
                    _n_scale_shift_i32 = arith.index_cast(
                        T.i32, _n_off * arith.constant(8, index=True)
                    )
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
                    c_offset = arith.constant(offset, index=True)
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
                    (_tail_ku_s2 + pack_K - 1) // pack_K
                    if _pad_ku_skip_s2 > 0
                    else None
                )

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

                    vec_elems = mixed_kpack_bytes // int(b_elem_bytes)
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

                def load_b_tile(base_k, ku_limit=k_unroll):
                    b_tile = []
                    for ku in range_constexpr(ku_limit):
                        packs0 = []
                        packs1 = []
                        for ni in range_constexpr(num_acc_n):
                            b0, b1 = load_b_packs_k64(base_k, ku, ni)
                            packs0.append(b0)
                            packs1.append(b1)
                        b_tile.append((packs0, packs1))
                    return b_tile

                _b_split_enabled = k_unroll >= 2
                _b_split_ku = k_unroll // 2 if _b_split_enabled else k_unroll

                def load_b_tile_lo(base_k):
                    """Load first half of B tile (ku < _b_split_ku)."""
                    b_tile = []
                    for ku in range_constexpr(_b_split_ku):
                        packs0 = []
                        packs1 = []
                        for ni in range_constexpr(num_acc_n):
                            b0, b1 = load_b_packs_k64(base_k, ku, ni)
                            packs0.append(b0)
                            packs1.append(b1)
                        b_tile.append((packs0, packs1))
                    return b_tile

                def load_b_tile_hi(base_k):
                    """Load second half of B tile (ku >= _b_split_ku)."""
                    b_tile = []
                    for ku in range_constexpr(_b_split_ku, k_unroll):
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
                    if const_expr(k_shift_bits > 0):
                        val = vector.extract(
                            scale_vec, static_position=[0], dynamic_position=[]
                        )
                        val = arith.shrui(val, arith.constant(k_shift_bits, type=T.i32))
                        return vector.from_elements(T.vec(1, T.i32), [val])
                    return scale_vec

                def load_b_scale_tile(
                    base_k, k_shift_bits=0, ku_packed_limit=k_unroll_packed
                ):
                    b_scale_tile = []
                    for ku in range_constexpr(ku_packed_limit):
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

                def load_a_scale_tile(
                    base_k, k_shift_bits=0, ku_packed_limit=k_unroll_packed
                ):
                    a_scale_tile = []
                    for ku in range_constexpr(ku_packed_limit):
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

                # ---- Pipeline helpers: store X tile to LDS (unused in DMA path) ----
                _lds_base_zero = arith.index(0)

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
                                lds_base=_lds_base_zero,
                                vec_part_i32x4=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )
                        elif const_expr(x_load_bytes == 8):
                            lds_store_8b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_buffer,
                                vec8_ty=vec8_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=_lds_base_zero,
                                vec_part_i32x2=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )
                        else:  # x_load_bytes == 4
                            lds_store_4b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_buffer,
                                vec4_ty=vec4_x_lds,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=_lds_base_zero,
                                vec_part_i32x1=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )

                # --- A LDS load helper for K64 (load 16B once, extract 2x i64 halves) ---
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
                    mfma_res_ty = vec4_i32 if is_int8 else vec4_f32

                    epilogue_pf = None
                    bias = None
                    if const_expr(prefetch_epilogue):
                        if const_expr(enable_bias):
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
                        if const_expr(doweight_stage2):
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
                            a_scale_val = vector.extract(
                                a_scale_i32, static_position=[0], dynamic_position=[]
                            )
                            if const_expr(_m_scale_shift_i32 is not None):
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
                                if const_expr(_n_scale_shift_i32 is not None):
                                    b_scale_val = arith.shrui(
                                        b_scale_val, _n_scale_shift_i32
                                    )

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
                                            curr_row_a_lds, col_base0, lds_buffer
                                        )

                                    if const_expr(is_f8_a):
                                        col_base1 = col_base + 64
                                        a2, a3 = lds_load_packs_k64(
                                            curr_row_a_lds, col_base1, lds_buffer
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
                    c4_idx = arith.index(4)
                    base_k_div4 = _div_pow2(
                        _div_pow2(base_k, int(a_elem_vec_pack))
                        * arith.constant(int(a_elem_bytes), index=True),
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
                        global_offset = arith.index_cast(T.i32, global_byte_idx)

                        if const_expr(i == 0):
                            lds_addr = memref.extract_aligned_pointer_as_index(
                                lds_buffer
                            ) + wave_id * arith.constant(
                                _wave_size * _dma_bytes, index=True
                            )
                            lds_ptr_i64 = rocdl.readfirstlane(
                                T.i64, arith.index_cast(T.i64, lds_addr)
                            )
                        else:
                            lds_ptr_i64 = lds_ptr_i64 + arith.constant(
                                total_threads * _dma_bytes, type=T.i64
                            )

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

                # Prologue -- B-first + async DMA X(0) -> pong.
                k0 = arith.index(0)
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

                c2_tile_k = arith.constant(tile_k * 2, index=True)
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
                        k_iv = arith.index(k_iv_py)
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
                                _make_b_hi_loader(k0_pong_bk)
                                if _b_split_enabled
                                else None
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
                                _make_b_hi_loader(next_k1_bk)
                                if _b_split_enabled
                                else None
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
                    k_tail1_py = (
                        int(inter_dim) + tile_k - 1
                    ) // tile_k * tile_k - tile_k
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
                                _make_b_hi_loader(k_tail1_bk)
                                if _b_split_enabled
                                else None
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
                if const_expr(lds_out is None):
                    raise RuntimeError(
                        "FLIR_MOE_STAGE2_CSHUFFLE=1 but lds_out is not allocated/aliased."
                    )

                # Precompute the output base address (i64 index) for ALL paths.
                # Both accumulate=True (global atomic) and accumulate=False (global store)
                # need 64-bit addressing to avoid i32 offset overflow when
                # tokens * model_dim * elem_bytes > INT32_MAX (~150K tokens for model_dim=7168).
                # NOTE: `_fly` is already imported at module level; do not re-import here
                # because that would shadow the closure binding inside this function and
                # cause UnboundLocalError when the A16W4 early-return branch reads `_fly`.
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
                        if const_expr(is_int8):
                            v = arith.sitofp(f32, v)
                        if const_expr(enable_bias):
                            v = v + bias_pf[ni]

                        if const_expr(doweight_stage2):
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
                    if const_expr(accumulate):
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
                    if const_expr(not bool(accumulate)):
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

                # NOTE: rename to avoid Python closure conflict with the A16W4 branch
                # that reads `_e_vec` from the outer setup scope.
                _e_vec_cs = 2 if accumulate else min(tile_n // 32, 8)
                c_shuffle_epilog(
                    arith=arith,
                    vector=vector,
                    gpu=gpu,
                    scf=scf,
                    range_constexpr=range_constexpr,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    e_vec=_e_vec_cs,
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

            if const_expr(_persistent):
                # Short-circuit: contiguous tiles are monotonically increasing,
                # so once bx_m >= num_valid_ids all remaining tiles are invalid.
                _cur_active = arith.andi(_still_active, blk_valid)
                _do_gemm = arith.andi(
                    _cur_active, arith.andi(exp_valid, tile_has_tokens)
                )
                _if_valid = scf.IfOp(_do_gemm)
                with ir.InsertionPoint(_if_valid.then_block):
                    _moe_gemm2_then_body()
                    scf.YieldOp([])

                gpu.barrier()
                scf.YieldOp([_cur_active])
            else:
                _if_valid = scf.IfOp(_all_valid)
                with ir.InsertionPoint(_if_valid.then_block):
                    _moe_gemm2_then_body()
                    scf.YieldOp([])

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
        sort_block_m,
        b_nt,
        xcd_swizzle,
        waves_per_eu,
        k_batch,
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
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()

        n_in = arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
        _tile_n_idx = arith.constant(tile_n, index=True)
        _model_dim_pad_idx = arith.constant(model_dim_pad, index=True)
        gx = (
            n_in - _model_dim_pad_idx + _tile_n_idx - arith.constant(1, index=True)
        ) / _tile_n_idx
        if const_expr(_persistent):
            gy = arith.constant(_cu_num, index=True)
        else:
            # For A16W4: persist_m=1 (set in setup), so gy reduces to size_expert_ids_in.
            _c_pm_l = arith.constant(persist_m, index=True)
            gy = (
                arith.index_cast(ir.IndexType.get(), i32_size_expert_ids_in.ir_value())
                + _c_pm_l
                - arith.constant(1, index=True)
            ) / _c_pm_l

        gz = _k_batch if is_a16w4 else 1

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
            grid=(gx, gy, gz),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_mixed_moe_gemm2


# ---------------------------------------------------------------------------
# Stage 2: A16W4 MXFP4 kernel (BF16 activations x FP4 E2M1 weights)
# ---------------------------------------------------------------------------


def _decode_e8m0_byte_to_f32(byte_i8, arith_mod):
    """Convert a single E8M0 byte (i8) to f32 = 2^(e - 127)."""
    c23 = arith_mod.constant(23, type=T.i32)
    byte_u32 = arith_mod.extui(T.i32, byte_i8)
    scale_bits = arith_mod.shli(byte_u32, c23)
    return arith_mod.bitcast(T.f32, scale_bits)


def compile_a16w4_moe_gemm2(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    out_dtype: str = "bf16",
    use_cshuffle_epilog: bool | None = None,
    enable_bias: bool = False,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    accumulate: bool = True,
    waves_per_eu: int = 0,
    k_batch: int = 1,
):
    """Compatibility wrapper for callers that import the legacy A16W4 entry."""
    return compile_mixed_moe_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=doweight_stage2,
        a_dtype="bf16",
        b_dtype="fp4",
        out_dtype=out_dtype,
        use_cshuffle_epilog=use_cshuffle_epilog,
        enable_bias=enable_bias,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        accumulate=accumulate,
        waves_per_eu=waves_per_eu,
        k_batch=k_batch,
    )
