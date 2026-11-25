# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE stage 1 -> stage 2 in one launch, built from AITER's own FlyDSL bodies.

For DeepSeek-V4.1-Flash TP4 A8W4 verify steps above 256 rows, on gfx950.
Stage 1 is AITER's stage-1 GEMM with its FP8 epilogue (clamp-SwiGLU, MXFP8 codes,
e8m0 scales in the sorted layout stage 2 reads; `fp8_exact` gives the quant rule of
the two-kernel path). Stage 2 is one of:
  * AITER's mixed stage-2 GEMM (`flydsl_moe2_afp8_wfp4_*`; FP8 intermediate in
    [token, slot] rows), kernel `fused`;
  * AITER's v2-layout stage-2 tile (`flydsl_moe2_layout_*`, emitted through
    `compile_gemm2_a4w4_port(_composition=)`; stage 1 writes the intermediate in
    sorted rows), kernel `fused_v2`.
Uses the trace-time hooks `_FUSED_S12_HOOKS`, `_FUSED_S12_BUILT` of `mixed_moe_gemm_2stage_common.py`.

Grid (1-D): workgroups [0, N1) are stage-1 tiles in AITER's order (sort block =
wg // gx1, N tile = wg % gx1); the rest are stage-2 tiles in the stage-2 kernel's
own order. A stage-1 tile stores its rows write-through (sc1), waits for its
stores, and thread 0 adds 1 to its block's ready counter (device-scope atomic).
A stage-2 tile's thread 0 waits until that counter reaches gx1 before the tile
loads anything; after the tile it counts itself, and the last of the block's
stage-2 tiles sets both words back to 0 for the next call.

No co-residency is assumed: a stage-2 tile waits only for stage-1 tiles, which have
lower ids; each XCD dispatches its workgroups in id order, so once any later
workgroup runs on an XCD, all its stage-1 tiles are resident or done, and stage-1
tiles never wait. No acquire fence: no load of this launch touches the intermediate
of a block before the block's counter is complete, and kernel start leaves no stale
copy in L2. A wait gives up after SPIN_MAX polls and counts the event in the
workspace error word (`errors()`); that would be a bug, never expected.

Use: `with moe_fused_s12.enabled(): aiter.fused_moe(...)` (op tests), or
`install()` for a whole process (serving, env AITER_MOE_FUSED_S12=1), with `_fp8`
stage-1 rows (mixed stage 2) or v2-layout stage-2 rows. The stage-1 launch of a
fused_moe call is held back and launched together with the stage-2 launch of the
same call; the reduction kernel after stage 2 is unchanged. Calls on one stream
only (one workspace per device).
"""

import contextlib
import functools
import hashlib
import os
import sys
import threading

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, gpu, ptrtoint, rocdl
from flydsl.expr.typing import T

import aiter.ops.flydsl.moe_kernels as mk
from aiter.ops.flydsl.kernels import mixed_moe_gemm_2stage_common as C
from aiter.ops.flydsl.kernels import mxmoe_dispatcher as mxd
from aiter.ops.flydsl.kernels.kernels_common import create_llvm_ptr
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import global_typed_ptr
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled as _dispatch
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw, ptr_arg

VERSION = "v2"
# bytes per sort block in the workspace: ready count at +0, consumer count at +4
# (own cache line)
STRIDE = 128
MAX_BLOCKS = 4096
ERR_OFF = MAX_BLOCKS * STRIDE  # byte offset of the error word
WS_INTS = (ERR_OFF + STRIDE) // 4
SPIN_MAX = 1 << 22
# workgroups per CU the fused kernel's LDS allows
OCC = int(os.environ.get("AITER_MOE_FUSED_S12_OCC", "3"))
# fuse calls of at most this many tokens
MAX_ROWS = int(os.environ.get("AITER_MOE_FUSED_S12_MAX_ROWS", "512"))
G2_KEY = (
    "BM",
    "BN",
    "BK",
    "use_nt",
    "HIDDEN_MAX",
    "epilog",
    "INTER_MAX",
    "a_dtype",
    "b_dtype",
    "topk",
    "SBM",
    "persist",
    "cu_num",
    "g2_bhoist",
    "g2_ascale_pf",
    "g2_spart",
    "g2_bf16_lds",
    "out_dtype",
    "enable_bias",
    "g2_prefetch_ids",
    "use_valid_token_count",
    "g2_scatter_lds_pairs",
    "K_VALID",
)  # mxmoe_dispatcher.get_g2's cache key, in order

_orig_mk_run = mk._run_compiled
_orig_mxd_run = mxd.run_compiled
_tls = threading.local()
_ws = {}


def workspace(device=None):
    dev = (
        torch.device("cuda", torch.cuda.current_device())
        if device is None
        else torch.device(device)
    )
    if dev not in _ws:
        _ws[dev] = torch.zeros(WS_INTS, dtype=torch.int32, device=dev)
    return _ws[dev]


def errors(device=None):
    return int(workspace(device)[ERR_OFF // 4].item())


def counters_clean(device=None):
    return int(workspace(device)[: ERR_OFF // 4].abs().sum().item()) == 0


def _atomic(op, addr, val):
    return fx.Int32(
        llvm.AtomicRMWOp(
            op,
            create_llvm_ptr(addr),
            _to_raw(fx.Int32(val)),
            llvm.AtomicOrdering.monotonic,
            syncscope="agent",
            alignment=4,
        ).result
    )


def _counter_hooks(ws, tid, producers, consumers):
    """Trace-time helpers on the workspace at `ws` (i64).

    producers / consumers are the i32 tile counts per sort block.
    """

    def s1_done(blk, blk_valid, exp_valid):
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        ok = fx.Boolean(blk_valid) & fx.Boolean(exp_valid) & (tid == fx.Int32(0))
        addr = ws + fx.Int64(fx.Int32(blk)) * fx.Int64(STRIDE)

        @flyc.jit
        def _sig():
            if ok:
                _atomic(llvm.AtomicBinOp.add, addr, 1)

        _sig()

    def s2_wait(blk, blk_valid, exp_valid):
        ok = fx.Boolean(blk_valid) & fx.Boolean(exp_valid) & (tid == fx.Int32(0))
        addr = ws + fx.Int64(fx.Int32(blk)) * fx.Int64(STRIDE)

        @flyc.jit
        def _wait():
            if ok:
                v = _atomic(llvm.AtomicBinOp.add, addr, 0)
                n = fx.Int32(0)
                while (v < producers) & (n < fx.Int32(SPIN_MAX)):
                    rocdl.s_sleep(1)
                    v = _atomic(llvm.AtomicBinOp.add, addr, 0)
                    n = n + fx.Int32(1)
                if v < producers:
                    _atomic(llvm.AtomicBinOp.add, ws + fx.Int64(ERR_OFF), 1)

        _wait()
        gpu.barrier()

    def s2_done(blk, blk_valid, exp_valid):
        ok = fx.Boolean(blk_valid) & fx.Boolean(exp_valid) & (tid == fx.Int32(0))
        addr = ws + fx.Int64(fx.Int32(blk)) * fx.Int64(STRIDE)

        @flyc.jit
        def _retire():
            if ok:
                old = _atomic(llvm.AtomicBinOp.add, addr + fx.Int64(4), 1)
                if old == consumers - fx.Int32(1):
                    _atomic(llvm.AtomicBinOp.xchg, addr, 0)
                    _atomic(llvm.AtomicBinOp.xchg, addr + fx.Int64(4), 0)

        _retire()

    return s1_done, s2_wait, s2_done


def _lds_rewind(alloc, hi, final, floor=0):
    """Trace-time: note the LDS high-water mark, then restart the allocator at 0.

    So the next body's LDS aliases this one's (the two branches never run in one
    workgroup). After the last body, set the kernel's LDS size to max(mark, floor).
    """
    hi[0] = max(hi[0], alloc._offset)
    alloc._offset = max(hi[0], floor) if final else 0


def _allocate(alloc, storage):
    # a call, not a method call in the kernel body: the AST rewriter would treat
    # `alloc` as branch state
    return alloc.allocate(storage).peek()


def _s1_candidate(m1):
    """A stage-1 launch whose output goes straight to stage 2.

    FP8 epilogue: no quant kernel in between.
    """
    return (
        m1 is not None
        and "grid" in m1
        and m1["need_fp8"]
        and m1["k_batch"] == 1
        and m1["persist_m"] == 1
        and not m1["heterogeneous_b"]
        and m1["total_threads"] == 256
    )


def fusable(exe1, exe2):
    m1, m2 = C._FUSED_S12_BUILT.get(id(exe1)), C._FUSED_S12_BUILT.get(id(exe2))
    return (
        _s1_candidate(m1)
        and not m1["v2_output_layout"]
        and m2 is not None
        and "tile_n" in m2
        and not m2["heterogeneous_b"]
        and not m2["persistent"]
        and m2["persist_m"] == 1
        and not m2["accumulate"]
        and m1["sort_block_m"] == m2["sort_block_m"] == m2["tile_m"]
    )


def _g2_key(launch):
    for k, v in mxd.G2_CACHE.items():
        if v is launch:
            return k
    return None


def fusable_v2(exe1, launch2):
    m1, key = C._FUSED_S12_BUILT.get(id(exe1)), _g2_key(launch2)
    if not (_s1_candidate(m1) and m1["v2_output_layout"] and key is not None):
        return False
    k = dict(zip(G2_KEY, key))
    return (
        k["epilog"] == "reduce"
        and k["a_dtype"] == "fp8"
        and k["out_dtype"] != "fp4"
        and not k["persist"]
        and not k["enable_bias"]
        and not k["use_valid_token_count"]
        and k["BM"] == k["SBM"] == m1["sort_block_m"]
    )


def _stage1_body(m1, occ):
    """The stage-1 builder rebuilt without its own LDS occupancy pad.

    Also returns the LDS size that allows `occ` workgroups per CU.
    """
    exe1 = C.compile_mixed_moe_gemm1_common(**{**m1["kwargs"], "waves_per_eu": None})
    return C._FUSED_S12_BUILT[id(exe1)], (
        160 * 1024 // (occ + 1) + 16 if occ > 0 else 0
    )


@functools.cache
def _build(id1, id2, occ):
    m1, lds_target = _stage1_body(C._FUSED_S12_BUILT[id1], occ)
    m2 = C._FUSED_S12_BUILT[id2]
    emit1, grid1, emit2 = m1["emit"], m1["grid"], m2["emit"]
    tile_n2, pad2 = m2["tile_n"], m2["model_dim_pad"]
    # module names omit some build knobs (e.g. b_nt), so the tag carries the
    # stage-1 build arguments
    cache_tag = (
        VERSION,
        occ,
        m1["module_name"],
        repr(sorted(m1["kwargs"].items())),
        m2["module_name"],
        m2["lds_bytes"],
    )
    name = "moe_fused_s12_" + hashlib.md5(repr(cache_tag).encode()).hexdigest()[:12]

    def grids(i1_inter, i1_nblk, i2_n, i2_nblk):
        gx1, gy1 = (fx.Index(g) for g in grid1(i1_inter, i1_nblk))
        c1 = arith.constant(1, index=True)
        gx2 = (
            fx.Index(i2_n)
            - arith.constant(pad2, index=True)
            + arith.constant(tile_n2, index=True)
            - c1
        ) // arith.constant(tile_n2, index=True)
        return gx1, gy1, gx2, fx.Index(i2_nblk)

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def fused(
        a1_out: fx.Pointer,
        a1_x: fx.Pointer,
        a1_w: fx.Pointer,
        a1_sx: fx.Pointer,
        a1_sw: fx.Pointer,
        a1_sid: fx.Pointer,
        a1_eid: fx.Pointer,
        a1_swt: fx.Pointer,
        a1_nv: fx.Pointer,
        a1_bias: fx.Pointer,
        a1_osc: fx.Pointer,
        i1_tokens: fx.Int32,
        i1_inter: fx.Int32,
        i1_k: fx.Int32,
        i1_nblk: fx.Int32,
        f1_b: fx.Float32,
        f1_br: fx.Float32,
        f1_lb: fx.Float32,
        f1_lbr: fx.Float32,
        f1_lim: fx.Float32,
        a2_out: fx.Pointer,
        a2_x: fx.Pointer,
        a2_w: fx.Pointer,
        a2_sx: fx.Pointer,
        a2_sw: fx.Pointer,
        a2_sid: fx.Pointer,
        a2_eid: fx.Pointer,
        a2_swt: fx.Pointer,
        a2_nv: fx.Pointer,
        a2_bias: fx.Pointer,
        i2_tokens: fx.Int32,
        i2_xrows: fx.Int32,
        i2_n: fx.Int32,
        i2_k: fx.Int32,
        i2_nblk: fx.Int32,
        a_ws: fx.Pointer,
    ):
        _ = cache_tag
        gx1, gy1, gx2, _gy2 = grids(i1_inter, i1_nblk, i2_n, i2_nblk)
        n1 = gx1 * gy1
        tid = fx.Int32(gpu.thread_id("x"))
        s1_done, s2_wait, s2_done = _counter_hooks(
            fx.Int64(ptrtoint(a_ws)), tid, fx.Int32(gx1), fx.Int32(gx2)
        )
        bid = gpu.block_id("x")
        alloc = fx.SharedAllocator(static=False)
        lds_hi = [0]
        if bid < n1:
            C._FUSED_S12_HOOKS.clear()
            C._FUSED_S12_HOOKS.update(
                bid={"x": bid % gx1, "y": bid // gx1},
                grid={"x": gx1, "y": gy1},
                wt=True,
                s1_done=s1_done,
                alloc=alloc,
            )
            emit1(
                a1_out,
                a1_x,
                a1_w,
                a1_sx,
                a1_sw,
                a1_w,
                a1_sw,
                a1_sid,
                a1_eid,
                a1_swt,
                a1_nv,
                a1_bias,
                a1_osc,
                i1_tokens,
                i1_inter,
                i1_k,
                i1_nblk,
                f1_b,
                f1_br,
                f1_lb,
                f1_lbr,
                f1_lim,
            )
            C._FUSED_S12_HOOKS.clear()
            _lds_rewind(alloc, lds_hi, False)
        else:
            r = bid - n1
            C._FUSED_S12_HOOKS.clear()
            C._FUSED_S12_HOOKS.update(
                bid={"x": r % gx2, "y": r // gx2},
                s2_wait=s2_wait,
                s2_done=s2_done,
                alloc=alloc,
            )
            emit2(
                a2_out,
                a2_x,
                a2_w,
                a2_sx,
                a2_sw,
                a2_w,
                a2_sw,
                a2_sid,
                a2_eid,
                a2_swt,
                a2_nv,
                a2_bias,
                i2_tokens,
                i2_xrows,
                i2_n,
                i2_k,
                i2_nblk,
            )
            C._FUSED_S12_HOOKS.clear()
            _lds_rewind(alloc, lds_hi, True, lds_target)

    @flyc.jit
    def launch(
        a1_out: fx.Pointer,
        a1_x: fx.Pointer,
        a1_w: fx.Pointer,
        a1_sx: fx.Pointer,
        a1_sw: fx.Pointer,
        a1_sid: fx.Pointer,
        a1_eid: fx.Pointer,
        a1_swt: fx.Pointer,
        a1_nv: fx.Pointer,
        a1_bias: fx.Pointer,
        a1_osc: fx.Pointer,
        i1_tokens: fx.Int32,
        i1_inter: fx.Int32,
        i1_k: fx.Int32,
        i1_nblk: fx.Int32,
        f1_b: fx.Float32,
        f1_br: fx.Float32,
        f1_lb: fx.Float32,
        f1_lbr: fx.Float32,
        f1_lim: fx.Float32,
        a2_out: fx.Pointer,
        a2_x: fx.Pointer,
        a2_w: fx.Pointer,
        a2_sx: fx.Pointer,
        a2_sw: fx.Pointer,
        a2_sid: fx.Pointer,
        a2_eid: fx.Pointer,
        a2_swt: fx.Pointer,
        a2_nv: fx.Pointer,
        a2_bias: fx.Pointer,
        i2_tokens: fx.Int32,
        i2_xrows: fx.Int32,
        i2_n: fx.Int32,
        i2_k: fx.Int32,
        i2_nblk: fx.Int32,
        a_ws: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = cache_tag
        gx1, gy1, gx2, gy2 = grids(i1_inter, i1_nblk, i2_n, i2_nblk)
        total = gx1 * gy1 + gx2 * gy2
        fused(
            a1_out,
            a1_x,
            a1_w,
            a1_sx,
            a1_sw,
            a1_sid,
            a1_eid,
            a1_swt,
            a1_nv,
            a1_bias,
            a1_osc,
            i1_tokens,
            i1_inter,
            i1_k,
            i1_nblk,
            f1_b,
            f1_br,
            f1_lb,
            f1_lbr,
            f1_lim,
            a2_out,
            a2_x,
            a2_w,
            a2_sx,
            a2_sw,
            a2_sid,
            a2_eid,
            a2_swt,
            a2_nv,
            a2_bias,
            i2_tokens,
            i2_xrows,
            i2_n,
            i2_k,
            i2_nblk,
            a_ws,
        ).launch(grid=(total, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


class _Composed:
    """What compile_gemm2_a4w4_port hands a composition.

    The stage-2 tile emitter and its LDS struct.
    """

    def __init__(self, module_name, emit_gemm2_tile, shared_storage):
        self.module_name, self.tile, self.storage, self.compile_hints = (
            module_name,
            emit_gemm2_tile,
            shared_storage,
            {},
        )


@functools.cache
def _build_v2(id1, key2, occ):
    m1, lds_target = _stage1_body(C._FUSED_S12_BUILT[id1], occ)
    k = dict(zip(G2_KEY, key2))
    comp = mxd.compile_gemm2_a4w4_port(
        BM=k["BM"],
        BN=k["BN"],
        BK=k["BK"],
        use_nt=k["use_nt"],
        HIDDEN_MAX=k["HIDDEN_MAX"],
        epilog=k["epilog"],
        INTER_MAX=k["INTER_MAX"],
        a_dtype=k["a_dtype"],
        b_dtype=k["b_dtype"],
        topk=k["topk"],
        SBM=k["SBM"],
        persist=k["persist"],
        cu_num=k["cu_num"],
        g2_bhoist=k["g2_bhoist"],
        g2_ascale_pf=k["g2_ascale_pf"],
        g2_spart=k["g2_spart"],
        g2_bf16_lds=k["g2_bf16_lds"],
        out_dtype=k["out_dtype"],
        enable_bias=k["enable_bias"],
        g2_prefetch_ids=k["g2_prefetch_ids"],
        use_valid_token_count=k["use_valid_token_count"],
        g2_scatter_lds_pairs=k["g2_scatter_lds_pairs"],
        K_VALID=k["K_VALID"],
        _composition=_Composed,
    )
    emit1, grid1, tile2, storage2 = m1["emit"], m1["grid"], comp.tile, comp.storage
    BM, BN, E = k["BM"], k["BN"], m1["kwargs"]["experts"]
    gnum, m01 = (
        (k["g2_spart"] // 100, k["g2_spart"] % 100) if k["g2_spart"] > 0 else (0, 0)
    )
    cache_tag = (
        VERSION,
        occ,
        m1["module_name"],
        repr(sorted(m1["kwargs"].items())),
        comp.module_name,
    )
    name = "moe_fused_s12v2_" + hashlib.md5(repr(cache_tag).encode()).hexdigest()[:12]

    def tile_of(r, total_m, num_n):
        if gnum > 0:  # the standalone kernel's spatially-local tile order
            return mxd._spart_output_tile_index(r, total_m, num_n, gnum, m01)
        m = mxd._udiv(r, num_n)
        return m, r - m * num_n

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def fused_v2(
        a1_out: fx.Pointer,
        a1_x: fx.Pointer,
        a1_w: fx.Pointer,
        a1_sx: fx.Pointer,
        a1_sw: fx.Pointer,
        a1_sid: fx.Pointer,
        a1_eid: fx.Pointer,
        a1_swt: fx.Pointer,
        a1_nv: fx.Pointer,
        a1_bias: fx.Pointer,
        a1_osc: fx.Pointer,
        i1_tokens: fx.Int32,
        i1_inter: fx.Int32,
        i1_k: fx.Int32,
        i1_nblk: fx.Int32,
        f1_b: fx.Float32,
        f1_br: fx.Float32,
        f1_lb: fx.Float32,
        f1_lbr: fx.Float32,
        f1_lim: fx.Float32,
        q_aq: fx.Int64,
        q_asc: fx.Int64,
        q_bq: fx.Int64,
        q_bsc: fx.Int64,
        q_eids: fx.Int64,
        q_cum: fx.Int64,
        q_stids: fx.Int64,
        q_sw: fx.Int64,
        q_bias: fx.Int64,
        j_m: fx.Int32,
        j_mmb: fx.Int32,
        j_inter: fx.Int32,
        j_hidden: fx.Int32,
        q_out: fx.Int64,
        a_ws: fx.Pointer,
    ):
        _ = cache_tag
        gx1, gy1 = (fx.Index(g) for g in grid1(i1_inter, i1_nblk))
        n1 = gx1 * gy1
        num_n = mxd._udiv(j_hidden, BN)
        tid = fx.Int32(gpu.thread_id("x"))
        s1_done, s2_wait, s2_done = _counter_hooks(
            fx.Int64(ptrtoint(a_ws)), tid, fx.Int32(gx1), num_n
        )
        bid = gpu.block_id("x")
        alloc = fx.SharedAllocator(static=False)
        lds_hi = [0]
        if bid < n1:
            C._FUSED_S12_HOOKS.clear()
            C._FUSED_S12_HOOKS.update(
                bid={"x": bid % gx1, "y": bid // gx1},
                grid={"x": gx1, "y": gy1},
                wt=True,
                s1_done=s1_done,
                alloc=alloc,
            )
            emit1(
                a1_out,
                a1_x,
                a1_w,
                a1_sx,
                a1_sw,
                a1_w,
                a1_sw,
                a1_sid,
                a1_eid,
                a1_swt,
                a1_nv,
                a1_bias,
                a1_osc,
                i1_tokens,
                i1_inter,
                i1_k,
                i1_nblk,
                f1_b,
                f1_br,
                f1_lb,
                f1_lbr,
                f1_lim,
            )
            C._FUSED_S12_HOOKS.clear()
            _lds_rewind(alloc, lds_hi, False)
        else:
            lds = _allocate(alloc, storage2)
            r = fx.Int32(bid - n1)
            lane = tid % fx.Int32(64)
            wave = rocdl.readfirstlane(T.i32, _to_raw(tid // fx.Int32(64)))
            total_m = mxd._udiv(fx.Int32(global_typed_ptr(q_cum, T.i32)[0]), BM)
            if r < total_m * num_n:
                m, n = tile_of(r, total_m, num_n)
                e_ok = fx.Uint32(
                    fx.Int32(global_typed_ptr(q_eids, T.i32)[m])
                ) < fx.Uint32(E)
                s2_wait(m, e_ok, e_ok)
                tile2(
                    q_aq,
                    q_asc,
                    q_bq,
                    q_bsc,
                    q_eids,
                    q_stids,
                    q_sw,
                    q_bias,
                    q_out,
                    m,
                    n,
                    lane,
                    wave,
                    j_m,
                    j_mmb,
                    j_inter,
                    j_hidden,
                    lds,
                )
                gpu.barrier()
                s2_done(m, e_ok, e_ok)
            _lds_rewind(alloc, lds_hi, True, lds_target)

    @flyc.jit
    def launch(
        a1_out: fx.Pointer,
        a1_x: fx.Pointer,
        a1_w: fx.Pointer,
        a1_sx: fx.Pointer,
        a1_sw: fx.Pointer,
        a1_sid: fx.Pointer,
        a1_eid: fx.Pointer,
        a1_swt: fx.Pointer,
        a1_nv: fx.Pointer,
        a1_bias: fx.Pointer,
        a1_osc: fx.Pointer,
        i1_tokens: fx.Int32,
        i1_inter: fx.Int32,
        i1_k: fx.Int32,
        i1_nblk: fx.Int32,
        f1_b: fx.Float32,
        f1_br: fx.Float32,
        f1_lb: fx.Float32,
        f1_lbr: fx.Float32,
        f1_lim: fx.Float32,
        q_aq: fx.Int64,
        q_asc: fx.Int64,
        q_bq: fx.Int64,
        q_bsc: fx.Int64,
        q_eids: fx.Int64,
        q_cum: fx.Int64,
        q_stids: fx.Int64,
        q_sw: fx.Int64,
        q_bias: fx.Int64,
        j_m: fx.Int32,
        j_mmb: fx.Int32,
        j_gb: fx.Int32,
        j_inter: fx.Int32,
        j_hidden: fx.Int32,
        q_out: fx.Int64,
        q_osc: fx.Int64,
        a_ws: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = cache_tag
        gx1, gy1 = (fx.Index(g) for g in grid1(i1_inter, i1_nblk))
        total = gx1 * gy1 + fx.Index(
            j_gb * fx.Int32(fx.Uint32(j_hidden) // fx.Uint32(BN))
        )
        fused_v2(
            a1_out,
            a1_x,
            a1_w,
            a1_sx,
            a1_sw,
            a1_sid,
            a1_eid,
            a1_swt,
            a1_nv,
            a1_bias,
            a1_osc,
            i1_tokens,
            i1_inter,
            i1_k,
            i1_nblk,
            f1_b,
            f1_br,
            f1_lb,
            f1_lbr,
            f1_lim,
            q_aq,
            q_asc,
            q_bq,
            q_bsc,
            q_eids,
            q_cum,
            q_stids,
            q_sw,
            q_bias,
            j_m,
            j_mmb,
            j_inter,
            j_hidden,
            q_out,
            a_ws,
        ).launch(grid=(total, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


def build(exe1, exe2, occ=None):
    assert fusable(exe1, exe2), "not a fusable stage-1 / stage-2 pair"
    return _build(id(exe1), id(exe2), OCC if occ is None else occ)


def build_v2(exe1, launch2, occ=None):
    assert fusable_v2(exe1, launch2), "not a fusable stage-1 / v2-layout stage-2 pair"
    return _build_v2(id(exe1), _g2_key(launch2), OCC if occ is None else occ)


def _ready_workspace():
    """The workspace, or None if it does not exist yet during a CUDA graph capture.

    None means: no fusion for this call.
    """
    dev = torch.device("cuda", torch.cuda.current_device())
    if dev not in _ws and torch.cuda.is_current_stream_capturing():
        return None
    return workspace(dev)


_announced = set()


def _count(key, what=None):
    setattr(_tls, key, getattr(_tls, key, 0) + 1)
    # one line per kernel and process: proof of use in server logs
    if what is not None and what not in _announced:
        _announced.add(what)
        print(f"[moe_fused_s12] {key}: {what}", file=sys.stderr, flush=True)


def _flush():
    pend = getattr(_tls, "pend", None)
    _tls.pend = None
    if pend is not None:
        _orig_mk_run(*pend)
        _count("unfused")


def _mk_hook(exe, args):
    """moe_kernels._run_compiled: hold a stage-1 launch.

    Fuse it with the next mixed stage-2 launch.
    """
    pend = getattr(_tls, "pend", None)
    # args[11]: tokens
    if (
        pend is None
        and _s1_candidate(C._FUSED_S12_BUILT.get(id(exe)))
        and args[11] <= MAX_ROWS
    ):
        _tls.pend = (exe, args)
        return None
    if pend is not None and fusable(pend[0], exe):
        ws = _ready_workspace()
        if ws is not None:
            _tls.pend = None
            _dispatch(
                build(pend[0], exe), *pend[1][:-1], *args[:-1], ptr_arg(ws), args[-1]
            )
            _count(
                "fused",
                f"{C._FUSED_S12_BUILT[id(pend[0])]['module_name']} + {C._FUSED_S12_BUILT[id(exe)]['module_name']}"
                f" occ {OCC}",
            )
            return None
    _flush()
    return _orig_mk_run(exe, args)


def _mxd_hook(launch, *args, **kw):
    """mxmoe_dispatcher.run_compiled: fuse a held v2-layout stage-1 launch.

    The held launch runs together with this v2-layout stage-2 launch.
    """
    pend = getattr(_tls, "pend", None)
    if pend is not None and not kw and fusable_v2(pend[0], launch):
        ws = _ready_workspace()
        if ws is not None:
            _tls.pend = None
            _dispatch(
                build_v2(pend[0], launch),
                *pend[1][:-1],
                *args[:-1],
                ptr_arg(ws),
                args[-1],
            )
            _count(
                "fused_v2",
                f"{C._FUSED_S12_BUILT[id(pend[0])]['module_name']} + v2 {_g2_key(launch)} occ {OCC}",
            )
            return None
    _flush()
    return _orig_mxd_run(launch, *args, **kw)


def install():
    """For serving (env AITER_MOE_FUSED_S12=1): fuse for the whole process."""
    first = mk._run_compiled is not _mk_hook
    mk._run_compiled = _mk_hook
    mxd.run_compiled = _mxd_hook
    if (
        torch.cuda.is_available()
        and torch.cuda.is_initialized()
        and not torch.cuda.is_current_stream_capturing()
    ):
        # else the first eager fused_moe call allocates it; calls inside a capture
        # before that run unfused
        workspace()
    if first:
        print(
            f"[moe_fused_s12] installed: occ {OCC}, max rows {MAX_ROWS}",
            file=sys.stderr,
            flush=True,
        )


@contextlib.contextmanager
def enabled():
    """Inside: each fused_moe call's stage-1 + stage-2 launches run as one launch.

    Only fusable pairs are fused.
    """
    workspace()
    prev = mk._run_compiled, mxd.run_compiled
    install()
    try:
        yield
    finally:
        mk._run_compiled, mxd.run_compiled = prev
        _flush()


def stats():
    return {k: getattr(_tls, k, 0) for k in ("fused", "fused_v2", "unfused")}


if os.environ.get("AITER_MOE_FUSED_S12", "0") == "1":
    install()
