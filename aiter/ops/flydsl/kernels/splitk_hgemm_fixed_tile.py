# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# =============================================================================
# splitk_hgemm_fixed_tile (flydsl) — a fixed-tile (4-wave / 256-thread) split-K bf16
#   GEMM, faithful FlyDSL port of the hand-written HIP cuh
#   csrc/kernels/prezero_gemm/splitk_gemm_a16w16.cuh.
#
#   C[M,N] += A[M,K] @ B[N,K]^T   (TN, bf16 in / fp32 accumulate / packed-bf16 atomic out).
#   C is zeroed in-kernel: ksplit==0 zeros its tile + a per-tile
#   signal/sema handshake before atomic-adds — no external memset.
#
# Same design as the cuh (see its header for the full rationale):
#   * splitK grid = (N/BN, SPLITK, ceil(M/BM)); one block per (N-tile, K-slice, M-tile).
#   * MFMA v_mfma_f32_16x16x32_bf16 (K=32): each lane feeds 8 contiguous bf16 (one b128).
#   * A/B staged to LDS as plain [rows][BK] bf16 tiles, double-buffered, 1 tile prefetch.
#   * 8-block XOR swizzle phys()/rot021 -> 0 bank-conflict for both b128 LDS read & write.
#   * 4 waves (256 threads), BMxBN output split 2(M) x 2(N): nchunk / mbase as in the cuh.
#   * Epilogue cshuffles the fp32 acc through an LDS image overlaying s_A, then packs
#     2 adjacent columns into one global_atomic_pk_add_bf16 (== the cuh is_out_b16 path).
# =============================================================================
from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP, SmemAllocator, SmemPtr

from .tensor_shim import GTensor, STensor, get_dtype_in_kernel

# MFMA-fixed constants (independent of shape/tile), mirroring the cuh.
MFMA_M, MFMA_N, MFMA_K = 16, 16, 32
KOCT = 8  # bf16 per lane per MFMA operand (= one b128)
ACC_ROWS = 4  # C rows held per lane in the fp32 accumulator (floatx4)
WARP_SIZE = 64
DTYPE_BYTES = 2


# Tuner search-space bounds. The kernel's own asserts (in `compile_splitk_hgemm_fixed_tile`)
# are the single source of truth for validity; `iter_fixed_tile_configs` just walks a
# bounded grid and keeps the (BM, BN, BK, SPLITK) points the kernel accepts.
FIXED_TILE_M_OPTIONS = (16, 32, 64, 128)
FIXED_TILE_N_OPTIONS = (32, 64)
FIXED_TILE_K = 128
FIXED_TILE_MAX_SPLIT_K = 16
# Persistent per-device split-K signal/semaphore buffer length; one slot is
# needed per output tile, so configs whose tile count exceeds this are invalid.
SIG_SEM_SLOTS = 1 << 16


def _fixed_tile_ok(N, K, BN, BM, BK, SPLITK):
    if N % BN or K % SPLITK:
        return False
    if (K // SPLITK) % BK:
        return False
    NCH = BN // MFMA_N
    if BN % MFMA_N or 4 % NCH:
        return False
    MGROUPS = 4 // NCH
    if BM < MGROUPS * MFMA_M or BM % (MGROUPS * MFMA_M):
        return False
    if BM * BK // KOCT < 256 or (BM * BK // KOCT) % 256:
        return False
    if (BM * BN // 2) % 256:
        return False
    # Double-buffered A/B LDS tiles must fit, and the fp32 epilogue overlays s_A.
    AS_BYTES = 2 * BM * BK * DTYPE_BYTES
    BS_BYTES = 2 * BN * BK * DTYPE_BYTES
    if BM * BN * 4 > AS_BYTES:
        return False
    return AS_BYTES + BS_BYTES <= SMEM_CAPACITY_MAP[get_rocm_arch()]


def iter_fixed_tile_configs(M, N, K):
    """Yield (BM, BN, BK, SPLITK) tuples the fixed-4-wave kernel accepts for (M, N, K)."""
    for BN in FIXED_TILE_N_OPTIONS:
        for BM in FIXED_TILE_M_OPTIONS:
            if (N // BN) * ((M + BM - 1) // BM) > SIG_SEM_SLOTS:
                continue
            for SPLITK in range(1, FIXED_TILE_MAX_SPLIT_K + 1):
                if _fixed_tile_ok(N, K, BN, BM, FIXED_TILE_K, SPLITK):
                    yield (BM, BN, FIXED_TILE_K, SPLITK)


def _rot021(e):
    # tile-independent 8-block XOR swizzle helper (cuh rot021).
    return ((e & 6) >> 1) | ((e & 1) << 2) | (e & 8)


def _phys(row, k, BK):
    # LDS element offset of [row][k] in a [rows][BK] bf16 tile. The in-block offset (k&7)
    # stays contiguous (b128 read/write); the block index (k>>3) is XOR-scattered by
    # rot021(row&15) so the 16 MFMA rows hit distinct banks. Bijective -> always correct
    # as long as store & load use the same _phys; only bank-conflict (perf) depends on it.
    return row * BK + (((k >> 3) ^ _rot021(row & 15)) << 3) + (k & 7)


@functools.lru_cache(maxsize=4096)
def compile_splitk_hgemm_fixed_tile(
    N: int,
    K: int,
    BN: int,
    SPLITK: int,
    BM: int,
    BK: int = 128,
    dtype: str = "bf16",
):
    assert dtype == "bf16", "cuh port is bf16-only"
    KSLICE = K // SPLITK
    N_TILES = N // BN
    assert K % SPLITK == 0 and KSLICE % BK == 0, "bad splitK"
    NCH = BN // MFMA_N  # n-chunks per tile
    assert BN % MFMA_N == 0 and 4 % NCH == 0, "BN must give 1/2/4 n-chunks (<=4 waves)"
    MGROUPS = 4 // NCH  # wave M-groups
    MCH_W = BM // (MGROUPS * MFMA_M)  # m-chunks per wave
    assert BM % (MGROUPS * MFMA_M) == 0 and MCH_W >= 1
    KK = BK // MFMA_K  # MFMA K-steps per tile
    A_CHUNKS = BM * BK // KOCT // 256  # b128 A chunks loaded per thread
    B_TOTAL = BN * BK // KOCT  # b128 B chunks total
    B_CHUNKS = (B_TOTAL + 255) // 256  # b128 B chunks loaded per thread
    assert A_CHUNKS >= 1, "BM*BK too small for 256 threads (need >=1 b128/thread)"
    assert (BM * BK // KOCT) % 256 == 0, "BM*BK//KOCT must be a multiple of 256"
    ROW_KOCT = BK // KOCT  # b128 columns per LDS row (= 16 for BK=128)
    NTILES = KSLICE // BK  # K-tiles per slice
    PAIRS = (BM * BN // 2) // 256  # column-pairs per thread in the epilogue
    assert (BM * BN // 2) % 256 == 0, "BM*BN/2 must be a multiple of 256"

    GPU_ARCH = get_rocm_arch()
    AS_BYTES = 2 * BM * BK * DTYPE_BYTES  # double-buffered A tile
    BS_BYTES = 2 * BN * BK * DTYPE_BYTES  # double-buffered B tile
    assert BM * BN * 4 <= AS_BYTES, "fp32 epilogue image must fit inside s_A"
    SMEM_USE = AS_BYTES + BS_BYTES
    assert (
        SMEM_USE <= SMEM_CAPACITY_MAP[GPU_ARCH]
    ), f"LDS {SMEM_USE} > cap {SMEM_CAPACITY_MAP[GPU_ARCH]}"

    allocator = SmemAllocator(None, arch=GPU_ARCH, global_sym_name="smem")
    smem_a_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_a_offset + AS_BYTES
    smem_b_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_b_offset + BS_BYTES

    KERNEL_NAME = f"splitk_hgemm_fixed_tile_{dtype}_M{BM}xN{BN}xK{BK}_SPK{SPLITK}_{GPU_ARCH}"

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kern(
        C: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        m: fx.Int32,
        semaphore: fx.Pointer,
        signal: fx.Pointer,
    ):
        dt = get_dtype_in_kernel(dtype)
        acc_zero = arith.constant_vector(0.0, T.vec(ACC_ROWS, T.f32))

        A_ = GTensor(A, dtype=dt, shape=(-1, K))
        B_ = GTensor(B, dtype=dt, shape=(N, K))
        C_ = GTensor(C, dtype=dt, shape=(-1, N))
        semaphore_ = GTensor(semaphore, dtype=T.i32, shape=(-1,))
        signal_ = GTensor(signal, dtype=T.i32, shape=(-1,))

        base_ptr = allocator.get_base()
        as_ = STensor(
            SmemPtr(base_ptr, smem_a_offset, dt, shape=(2 * BM * BK,)),
            dt,
            shape=(2, BM * BK),
        )
        bs_ = STensor(
            SmemPtr(base_ptr, smem_b_offset, dt, shape=(2 * BN * BK,)),
            dt,
            shape=(2, BN * BK),
        )
        fs_ = STensor(  # fp32 cshuffle image overlaying s_A (disjoint lifetime)
            SmemPtr(base_ptr, smem_a_offset, T.f32, shape=(BM * BN,)),
            T.f32,
            shape=(BM, BN),
        )

        tid = fx.Index(fx.thread_idx.x)  # everything below is index-typed
        wave = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        g = lane // MFMA_M
        e = lane % MFMA_M
        nchunk = wave % NCH
        mbase = (wave // NCH) * MCH_W

        n_tile = fx.Index(fx.block_idx.x)
        ksplit = fx.Index(fx.block_idx.y)
        mtile = fx.Index(fx.block_idx.z)
        mrow0 = mtile * BM
        n0 = n_tile * BN
        kbeg = ksplit * KSLICE
        m_idx = fx.Index(m)
        idx0 = fx.Index(0)

        # *16: one cache line per tile (avoid semaphore false sharing)
        signal_idx = (mtile * N_TILES + n_tile) * 16

        def _gptr(ptr, off_elems, elem_bytes):
            base = arith.index_cast(T.i64, fx.ptrtoint(ptr))
            boff = arith.index_cast(
                T.i64, fx.Index(off_elems) * fx.Index(elem_bytes)
            )
            return llvm.IntToPtrOp(
                ir.Type.parse("!llvm.ptr<1>"),
                llvm.AddOp(base, boff, llvm.IntegerOverflowFlags(0)).result,
            ).result

        ZVEC = 8
        ZG_TOTAL = BM * BN // ZVEC
        ZG_PER_T = (ZG_TOTAL + 255) // 256
        ZG_GUARD = ZG_TOTAL % 256 != 0

        def zero_c_tile():
            cond_ks0 = arith.cmpi(arith.CmpIPredicate.eq, ksplit, idx0)
            ks0_if = scf.IfOp(cond_ks0, results_=[], has_else=False)
            with ir.InsertionPoint(ks0_if.then_block):
                zvec = vector.broadcast(
                    T.vec(ZVEC, dt), arith.constant(0.0, type=dt)
                )
                zvec_v = (
                    zvec._value if const_expr(hasattr(zvec, "_value")) else zvec
                )
                for c in range_constexpr(ZG_PER_T):
                    gidx = tid + c * 256
                    el = gidx * ZVEC
                    row = el // BN
                    col = el % BN
                    gm = mrow0 + row
                    cond = arith.cmpi(arith.CmpIPredicate.ult, gm, m_idx)
                    if const_expr(ZG_GUARD):
                        cond = arith.andi(
                            cond,
                            arith.cmpi(
                                arith.CmpIPredicate.ult, gidx, fx.Index(ZG_TOTAL)
                            ),
                        )
                    cif = scf.IfOp(cond, results_=[], has_else=False)
                    with ir.InsertionPoint(cif.then_block):
                        lin = arith.index_cast(
                            T.i32, C_.linear_offset((gm, n0 + col))
                        )
                        cptr = _gptr(C, fx.Index(lin), DTYPE_BYTES)
                        llvm.InlineAsmOp(
                            None,
                            [cptr, zvec_v],
                            "global_store_dwordx4 $0, $1, off sc0 sc1",
                            "v,v",
                            has_side_effects=True,
                        )
                        scf.YieldOp([])
                gpu.barrier()
                is_t0 = arith.cmpi(arith.CmpIPredicate.eq, tid, idx0)
                t0_if = scf.IfOp(is_t0, results_=[], has_else=False)
                with ir.InsertionPoint(t0_if.then_block):
                    sptr = _gptr(signal, signal_idx, 4)
                    llvm.InlineAsmOp(
                        None,
                        [sptr, arith.constant(1, type=T.i32)],
                        "global_store_dword $0, $1, off sc0 sc1",
                        "v,v",
                        has_side_effects=True,
                    )
                    scf.YieldOp([])
                gpu.barrier()
                scf.YieldOp([])

        def split_k_barrier():
            is_t0 = arith.cmpi(arith.CmpIPredicate.eq, tid, idx0)
            t0_if = scf.IfOp(is_t0, results_=[], has_else=False)
            with ir.InsertionPoint(t0_if.then_block):
                init_cur = arith.constant(0, type=T.i32)
                w = scf.WhileOp([T.i32], [init_cur])
                before = ir.Block.create_at_start(w.before, [T.i32])
                after = ir.Block.create_at_start(w.after, [T.i32])
                with ir.InsertionPoint(before):
                    cur = before.arguments[0]
                    need = arith.CmpIOp(
                        arith.CmpIPredicate.eq, cur, arith.constant(0, type=T.i32)
                    ).result
                    scf.ConditionOp(need, [cur])
                with ir.InsertionPoint(after):
                    sptr = _gptr(signal, signal_idx, 4)
                    data = llvm.InlineAsmOp(
                        T.i32,
                        [sptr],
                        "global_load_dword $0, $1, off sc1",
                        "=v,v",
                        has_side_effects=True,
                    ).result
                    rocdl.s_waitcnt(0)
                    scf.YieldOp([data])
                scf.YieldOp([])
            rocdl.sched_barrier(0)
            gpu.barrier()
            t0_if2 = scf.IfOp(is_t0, results_=[], has_else=False)
            with ir.InsertionPoint(t0_if2.then_block):
                semptr = _gptr(semaphore, signal_idx, 4)
                arrive = llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    semptr,
                    arith.constant(1, type=T.i32),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                ).result
                cond_last = arith.cmpi(
                    arith.CmpIPredicate.eq, fx.Index(arrive), fx.Index(SPLITK - 1)
                )
                last_if = scf.IfOp(cond_last, results_=[], has_else=False)
                with ir.InsertionPoint(last_if.then_block):
                    semaphore_[signal_idx] = arith.constant(0, type=T.i32)
                    signal_[signal_idx] = arith.constant(0, type=T.i32)
                    scf.YieldOp([])
                scf.YieldOp([])
            gpu.barrier()

        # Per-thread global<->LDS coordinates (tile-local row/k of each b128 chunk).
        a_row = [0] * A_CHUNKS
        a_k = [0] * A_CHUNKS
        for c in range_constexpr(A_CHUNKS):
            j = tid + c * 256
            a_row[c] = j // ROW_KOCT
            a_k[c] = (j % ROW_KOCT) * KOCT
        b_row = [0] * B_CHUNKS
        b_k = [0] * B_CHUNKS
        b_valid = [None] * B_CHUNKS
        for c in range_constexpr(B_CHUNKS):
            j = tid + c * 256
            b_valid[c] = arith.cmpi(arith.CmpIPredicate.ult, j, fx.Index(B_TOTAL))
            b_row[c] = j // ROW_KOCT
            b_k[c] = (j % ROW_KOCT) * KOCT

        def loadA(kabs):
            r = []
            for c in range_constexpr(A_CHUNKS):
                row = mrow0 + a_row[c]
                safe = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, row, m_idx), row, idx0
                )
                r.append(A_.vec_load((safe, kabs + a_k[c]), KOCT))
            return r

        def loadB(kabs):
            r = []
            for c in range_constexpr(B_CHUNKS):
                # B rows are always in-range (N % BN == 0); guard the OOB tail chunk only.
                row = arith.select(b_valid[c], n0 + b_row[c], idx0)
                r.append(B_.vec_load((row, kabs + b_k[c]), KOCT))
            return r

        def storeA(buf, vecs):
            for c in range_constexpr(A_CHUNKS):
                as_.vec_store((buf, _phys(a_row[c], a_k[c], BK)), vecs[c], KOCT)

        def storeB(buf, vecs):
            for c in range_constexpr(B_CHUNKS):
                off = _phys(b_row[c], b_k[c], BK)
                # only the valid lanes write; clamp invalid to slot 0 (harmless overwrite)
                off = arith.select(b_valid[c], off, idx0)
                bs_.vec_store((buf, off), vecs[c], KOCT)

        def compute(buf, acc):
            acc_new = list(acc)
            for kk in range_constexpr(KK):
                k = kk * MFMA_K + g * KOCT
                b = bs_.vec_load((buf, _phys(nchunk * MFMA_N + e, k, BK)), KOCT)
                for mc in range_constexpr(MCH_W):
                    a = as_.vec_load(
                        (buf, _phys((mbase + mc) * MFMA_M + e, k, BK)), KOCT
                    )
                    acc_new[mc] = rocdl.mfma_f32_16x16x32_bf16(
                        T.vec(ACC_ROWS, T.f32), [a, b, acc_new[mc], 0, 0, 0]
                    )
            return acc_new

        zero_c_tile()

        # ---- main loop: double-buffered LDS, one K-tile prefetch ahead of compute ----
        storeA(idx0, loadA(kbeg))
        storeB(idx0, loadB(kbeg))
        rocdl.s_waitcnt(0)
        gpu.barrier()

        acc = [acc_zero] * MCH_W
        if const_expr(NTILES == 1):
            acc = compute(idx0, acc)
        else:
            init_state = [kbeg + BK, arith.constant(0, index=True)] + acc
            for _, state in range(1, NTILES, init=init_state):
                k_next = fx.Index(state[0])
                stage = fx.Index(state[1])
                acc_cur = state[2:]
                next_stage = 1 - stage
                a_next = loadA(k_next)
                b_next = loadB(k_next)
                acc_new = compute(stage, acc_cur)
                storeA(next_stage, a_next)
                storeB(next_stage, b_next)
                gpu.barrier()
                results = yield [k_next + BK, next_stage] + acc_new
            stage = fx.Index(results[1])
            acc = compute(stage, results[2:])

        # ---- epilogue: cshuffle fp32 acc through LDS image, packed bf16 atomic into C ----
        gpu.barrier()  # all lanes done reading s_A in compute()
        for mc in range_constexpr(MCH_W):
            for v in range_constexpr(ACC_ROWS):
                c_row = (mbase + mc) * MFMA_M + g * ACC_ROWS + v
                c_col = nchunk * MFMA_N + e
                fs_[c_row, c_col] = vector.extract(
                    acc[mc], static_position=[v], dynamic_position=[]
                )
        gpu.barrier()

        split_k_barrier()

        vec2_ty = T.vec(2, dt)
        for p4 in range_constexpr(PAIRS):
            p = tid + p4 * 256
            row = p // (BN // 2)
            c0 = (p % (BN // 2)) * 2
            e0 = fs_[row, c0].truncf(dt)
            e1 = fs_[row, c0 + 1].truncf(dt)
            pair = vector.from_elements(vec2_ty, [e0, e1])
            pair_v = pair._value if const_expr(hasattr(pair, "_value")) else pair
            gm = mrow0 + row
            cond = arith.cmpi(arith.CmpIPredicate.ult, gm, m_idx)
            cond_if = scf.IfOp(cond, results_=[], has_else=False)
            with ir.InsertionPoint(cond_if.then_block):
                lin_i32 = arith.index_cast(T.i32, C_.linear_offset((gm, n0 + c0)))
                base_i64 = arith.index_cast(T.i64, fx.ptrtoint(C))
                byte_off = arith.index_cast(
                    T.i64, fx.Index(lin_i32) * fx.Index(DTYPE_BYTES)
                )
                ptr = llvm.IntToPtrOp(
                    ir.Type.parse("!llvm.ptr<1>"),
                    llvm.AddOp(base_i64, byte_off, llvm.IntegerOverflowFlags(0)).result,
                ).result
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    ptr,
                    pair_v,
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )
                scf.YieldOp([])
        return

    @flyc.jit
    def launch(
        C: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        m: fx.Int32,
        semaphore: fx.Pointer,
        signal: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        bm = (m + BM - 1) // BM
        kern._func.__name__ = KERNEL_NAME
        kern(C, A, B, m, semaphore, signal).launch(
            grid=(N // BN, SPLITK, bm),
            block=(256, 1, 1),
            stream=stream,
            value_attrs={"rocdl.waves_per_eu": 2},
        )

    return launch


_SIG_SEM_CACHE = {}


def _get_fixed_tile_sig_sem(device):
    """Persistent per-device signal/semaphore buffers, reset in-kernel."""
    import torch

    key = (device.type, device.index)
    bufs = _SIG_SEM_CACHE.get(key)
    if bufs is None:
        sema = torch.zeros(SIG_SEM_SLOTS, dtype=torch.int32, device=device)
        sig = torch.zeros(SIG_SEM_SLOTS, dtype=torch.int32, device=device)
        _SIG_SEM_CACHE[key] = bufs = (sema, sig)
    return bufs


def splitk_hgemm_fixed_tile(C, A, B, BN, SPLITK, BM, BK=128, stream=None):
    """C[M,N] += A@B^T (bf16, TN); C is zeroed in-kernel (may be uninitialized)."""
    import torch
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from .tensor_shim import _run_compiled

    N, K, m = B.shape[0], A.shape[1], int(A.shape[0])
    launch = compile_splitk_hgemm_fixed_tile(N, K, BN, SPLITK, BM, BK=BK, dtype="bf16")
    sema, sig = _get_fixed_tile_sig_sem(C.device)
    s = torch.cuda.current_stream() if stream is None else stream

    def _pv(t):
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    _run_compiled(launch, _pv(C), _pv(A), _pv(B), m, _pv(sema), _pv(sig), fx.Stream(s))
    return C
