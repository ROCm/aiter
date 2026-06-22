# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correct (un-tuned) fp8 E4M3 flash-attention forward for CDNA4 (gfx950).

This is the *correctness* skeleton requested as a precursor to a tuned,
ping-pong fp8 FMHA kernel.  It deliberately keeps things simple:

- All 8 waves run the identical ``QK -> softmax -> PV`` sequence, each on
  its own 32 query rows (no wave-role ping-pong).
- One CDNA4 ``mfma_scale_f32_32x32x64_f8f6f4`` atom (fp8 E4M3, f32 accum),
  reused from :mod:`aiter.ops.flydsl.rocdl_mfma_fp8`.
- SageAttention per-tensor recipe: exact softmax in the log2 domain
  (``exp2``), online over KV tiles, with scalar Q/K/V descales.
- ``P`` is register-resident (Step 1): the GEMM1 C-layout is reshaped to
  the GEMM2 B-operand via an intra-wave hi-peer ``shuffle_xor`` (no LDS).
- ``V`` is stored row-major in 16-wide d-blocks and read with HW transpose
  ``ds_read_tr8_b64`` (Step 2) for the PV A-operand (no scatter store).

Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq, heads, head_dim).
Grid:   (num_q_tiles * batch * num_heads,)  with num_q_tiles = seq/BLOCK_M.
Block:  (512,) == 8 waves of 64 lanes.

Config (v1): HEAD_DIM=128, BLOCK_M=256 (8 waves x 32 q-rows), BLOCK_N=64,
non-causal, no GQA, fp8 E4M3 in, bf16 out.

The MFMA fragment layouts (lane L, lo=L%32, hi=L//32, verified in
``aiter/ops/flydsl/rocdl_mfma_fp8.py``):

- A (M=32 x K=64): ``A_frag[L][v] = A[row=lo, col=hi*32+v]``      v in [0,32)
- B (K=64 x N=32): ``B_frag[L][v] = B[row(K)=hi*32+v, col(N)=lo]``  v in [0,32)
- C (M=32 x N=32): ``C_frag[L][v] = C[row=hi*4+(v%4)+8*(v//4), col=lo]`` v in [0,16)

GEMM1 computes ``S = K @ Q^T`` so the C accumulator has M=kv (in the
value index) and N=q (in the lane), matching the bf16 reference's online
softmax which reduces over the value index (cheap, no cross-lane max).

GEMM2 computes ``O = V^T @ P`` so the O accumulator has M=d (value) and
N=q (lane) -- the store layout used by the bf16 reference.
"""

import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from aiter.ops.flydsl.rocdl_mfma_fp8 import Mfma32x32x64

_LOG2E = host_math.log2(host_math.e)

# Atom geometry (32x32x64 fp8).
MFMA_M = 32
MFMA_N = 32
MFMA_K = 64
WARP_SIZE = 64
A_FP8_PER_LANE = 32  # vec<8xi32>
C_F32_PER_LANE = 16  # vec<16xf32>


def _llvm_value(value):
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def _extract_aligned_pointer(tensor, address_space=None) -> ir.Value:
    ptr_type = ir.Type.parse(
        "!llvm.ptr" if address_space is None else f"!llvm.ptr<{address_space}>"
    )
    return _fly.extract_aligned_pointer_as_index(ptr_type, _llvm_value(tensor))


def _pointer_load(result_type: ir.Type, ptr: ir.Value) -> ir.Value:
    return llvm.LoadOp(result_type, _llvm_value(ptr)).result


def _pointer_store(value: ir.Value, ptr: ir.Value):
    return llvm.StoreOp(_llvm_value(value), _llvm_value(ptr))


def build_flash_attn_fp8_module(
    num_heads,
    head_dim=128,
    softmax_scale=None,
    waves_per_eu=2,
):
    """Build the fp8 flash-attention launcher (correctness variant).

    Returns a callable ``launch(Q, K, V, O, q_descale, k_descale,
    v_descale, batch, seq_len)`` (plus ``.compile`` for explicit AOT
    compilation).  Q/K/V are flat fp8 (viewed as int8), O flat bf16.
    """
    gpu_arch = get_hip_arch()
    assert gpu_arch.startswith(
        "gfx95"
    ), f"fp8 32x32x64 MFMA requires CDNA4, got {gpu_arch}"
    assert head_dim == 128, "v1 only supports head_dim=128"

    BLOCK_M = 256
    BLOCK_N = 128
    HEAD_DIM = head_dim
    NUM_HEADS = num_heads
    NUM_WAVES = 8
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE  # 512
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES  # 32
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    K_STEPS = HEAD_DIM // MFMA_K  # 2 head-dim chunks of 64 (QK contraction)
    N_KV_TILES = BLOCK_N // MFMA_N  # 4 kv sub-tiles of 32 (GEMM1 N)
    D_TILES = HEAD_DIM // MFMA_N  # 4 output d sub-tiles of 32
    PV_K_STEPS = BLOCK_N // MFMA_K  # 2 kv K-steps of 64 (PV contraction)
    P_PACK_WORDS = PV_K_STEPS * (A_FP8_PER_LANE // 4)  # 16 fp8 words for P B-operand

    if softmax_scale is None:
        softmax_scale = 1.0 / host_math.sqrt(head_dim)

    # ---- LDS layout (fp8 element type) ----
    # K tile : [BLOCK_N kv][HEAD_DIM d]          row-major
    # V tile : [d_block(8)][BLOCK_N kv][d_in(16)] row-major (Step 2)
    #
    # P is NOT in LDS (Step 1, register-resident): the GEMM1 C-layout P is
    # packed to fp8 in registers and reshaped into the GEMM2 B-operand via an
    # intra-wave cross-lane shuffle (hi-peer exchange), so no LDS round-trip.
    #
    # V layout (Step 2, HW-transpose read): V is stored row-major in 16-wide
    # d blocks so the cooperative load is a clean 16-byte vector store (no
    # scatter).  The GEMM2 A-operand
    #   A_frag[L][8*kc + r] = V[kv = hi*32 + 8*kc + r, d = lo + dt*32]
    # is produced by ds_read_tr8_b64 (HW transpose-on-read): for a 16-lane
    # group the atom returns result[lane][r] = LDS[group_base + lane%16 + 16*r]
    # (an 8x16 byte transpose).  With the d-block layout the per-(dt,kc) group
    # base = d_block*(BLOCK_N*16) + (hi*32+8*kc)*16 and d_block = lo//16 + 2*dt,
    # giving exactly V[kv, d] for the A fragment.  4 reads (kc=0..3) per d-tile.
    K_STRIDE = HEAD_DIM
    V_KV_STRIDE = 16  # bytes per kv within a 16-wide d block
    V_DBLOCK_STRIDE = BLOCK_N * V_KV_STRIDE
    N_DBLOCKS = HEAD_DIM // 16  # 8
    # DMA global->LDS pipeline (spec 8wave_fp8_zhuo): K^i is read only within
    # iteration i, so K double-buffers (i%2); V^i is read in iteration i+1
    # (deferred PV), so its buffer must survive V^{i-1} (read this iter) + V^i +
    # V^{i+1} (DMA'd this iter) -> V triple-buffers (i%3).
    NUM_BUF_K = 2
    NUM_BUF_V = 3
    LDS_K_TILE = BLOCK_N * K_STRIDE
    LDS_V_TILE = N_DBLOCKS * V_DBLOCK_STRIDE  # == HEAD_DIM * BLOCK_N
    LDS_K_SIZE = NUM_BUF_K * LDS_K_TILE
    LDS_V_SIZE = NUM_BUF_V * LDS_V_TILE
    LDS_K_OFF = 0
    LDS_V_OFF = LDS_K_OFF + LDS_K_SIZE
    LDS_TOTAL = LDS_V_OFF + LDS_V_SIZE

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name="flash_attn_fp8_smem",
    )
    lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_offset + LDS_TOTAL

    bf16_dtype = fx.BFloat16

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fp8_attn_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        q_descale: fx.Float32,
        k_descale: fx.Float32,
        v_descale: fx.Float32,
        seq_len: fx.Int32,
    ):
        # IR types must be materialized inside the kernel (MLIR context).
        #
        # NOTE: the raw LLVM dialect load/store path does not lower fp8
        # vector types ("unknown LLVM dialect type" -> crash), so all
        # global/LDS *storage* is done as int8 (1 byte, same as fp8) and we
        # bitcast to fp8 / i32 only at the compute boundary.  The MFMA atom
        # consumes vec<8xi32> operands, which we obtain by bitcasting i8x32.
        i8_dtype = fx.Int8
        i8_type = i8_dtype.ir_type
        bf16_type = bf16_dtype.ir_type
        v_i8x16 = Vec.make_type(16, i8_dtype)  # 16 bytes for coop load
        v_i8x32 = Vec.make_type(A_FP8_PER_LANE, i8_dtype)  # MFMA operand bytes

        fm_fast = fx.arith.FastMathFlags.fast

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmax(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm_fast).result

        def _f32_to_fp8_byte(f):
            # arith.truncf cannot lower f32 -> fp8 on this target; use the
            # rocdl pack intrinsic and keep the low fp8 byte.  Returns an i8.
            packed = rocdl.cvt_pk_fp8_f32(
                T.i32, _raw(f), _raw(c_zero_f), fx.Int32(0), False
            )
            return arith.trunci(T.i8, _raw(packed))

        def _f32x4_to_fp8_word(f0, f1, f2, f3):
            # Pack 4 f32 -> one i32 (4 contiguous fp8 bytes [f0,f1,f2,f3]) using
            # two cvt_pk_fp8_f32: low word = (f0,f1), high word = (f2,f3).
            # Halves both the cvt count and the LDS store count vs per-byte.
            w0 = rocdl.cvt_pk_fp8_f32(T.i32, _raw(f0), _raw(f1), fx.Int32(0), False)
            w1 = rocdl.cvt_pk_fp8_f32(T.i32, _raw(f2), _raw(f3), _raw(w0), True)
            return w1

        mfma = Mfma32x32x64()

        q_ptr = _extract_aligned_pointer(Q)
        k_ptr = _extract_aligned_pointer(K)
        v_ptr = _extract_aligned_pointer(V)
        o_ptr = _extract_aligned_pointer(O)

        seq_len_v = fx.Index(seq_len)

        # ---- LDS pointers (int8 storage) ----
        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_offset, i8_type, shape=(LDS_TOTAL,)).get()

        # ---- Thread / block indices ----
        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lo = lane % 32
        hi = lane // 32

        wave_q_offset = wave_id * ROWS_PER_WAVE

        # ---- Decompose block_id -> (head, batch, q_tile) ----
        head_idx = block_id % NUM_HEADS
        batch_q_tile_id = block_id // NUM_HEADS
        num_q_tiles = (seq_len_v + BLOCK_M - 1) // BLOCK_M
        q_tile_idx = batch_q_tile_id % num_q_tiles
        batch_idx = batch_q_tile_id // num_q_tiles
        q_start = q_tile_idx * BLOCK_M

        def global_idx(token_idx, col):
            token = batch_idx * seq_len_v + token_idx
            return token * STRIDE_TOKEN + head_idx * HEAD_DIM + col

        # ---- Scales (log2 domain) ----
        # qk_scale = q_descale * k_descale * softmax_scale
        c_log2e = fx.Float32(_LOG2E)
        qk_scale = _fmul(_fmul(q_descale, k_descale), fx.Float32(softmax_scale))
        scale_log2e = _fmul(qk_scale, c_log2e)
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)

        # ===================================================================
        # DMA global->LDS (raw_ptr_buffer_load_lds): each lane streams a 16-byte
        # (16 fp8) chunk directly from global into LDS, lane-contiguous from a
        # wave-uniform base (no register round-trip).
        #
        # In the ping-pong each DMA is issued by ONE 4-wave group only (G1 loads
        # K^{i+1} in the mfma phase; G0 loads V^{i+1} in the softmax phase), so
        # the cooperative load is sized for 256 lanes, not 512: 256 lanes x 16B
        # = 4096 B/pass; a 128x128 fp8 tile is 16384 B -> DMA_PASSES = 4.
        # ===================================================================
        DMA_BYTES = 16
        DMA_LANES = (NUM_WAVES // 2) * WARP_SIZE  # 256 (one 4-wave group)
        DMA_PASSES = LDS_K_TILE // (DMA_LANES * DMA_BYTES)  # 4
        WAVE_DMA_STRIDE = WARP_SIZE * DMA_BYTES  # 1024: per-wave LDS span/pass
        PASS_DMA_STRIDE = DMA_LANES * DMA_BYTES  # 4096: per-pass LDS span

        # Per-(batch,head) buffer resources based at token 0, so the per-lane
        # voffset is just kv_abs*STRIDE_TOKEN + col (bytes; fp8 == 1 B/elem).
        head_base_elem = batch_idx * seq_len_v * fx.Index(
            STRIDE_TOKEN
        ) + head_idx * fx.Index(HEAD_DIM)

        def _rsrc(ptr):
            base_i64 = llvm.PtrToIntOp(T.i64, ptr).result
            off_i64 = arith.index_cast(T.i64, _raw(head_base_elem))
            addr_i64 = arith.addi(base_i64, off_i64)
            return fx.buffer_ops.create_buffer_resource_from_addr(addr_i64)

        k_rsrc = _rsrc(k_ptr)
        v_rsrc = _rsrc(v_ptr)

        _dma_size = arith.constant(DMA_BYTES, type=T.i32)
        _dma_zero = arith.constant(0, type=T.i32)
        _dma_aux = arith.constant(1, type=T.i32)

        _lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")

        def _dma_issue(rsrc, lds_byte_off, voffset_idx):
            voff_i32 = arith.index_cast(T.i32, voffset_idx)
            # raw_ptr_buffer_load_lds writes via m0 -> the LDS base must be a
            # wave-uniform scalar; force it into an SGPR with readfirstlane.
            lds_addr = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, lds_byte_off))
            lds_ptr = llvm.inttoptr(_lds_ptr_ty, lds_addr)
            rocdl.raw_ptr_buffer_load_lds(
                rsrc, lds_ptr, _dma_size, voff_i32, _dma_zero, _dma_zero, _dma_aux
            )

        def dma_k(buf, kv_start):
            # K row-major into LDSK[buf], cooperatively by G1 (waves 4-7).
            # group-local cell c = p*256 + ltid maps to kv = c//8, d = (c%8)*16.
            # The HW writes to lds_ptr + lane*16 (lane = ltid%64), so the
            # wave-uniform lds_ptr carries only p and the group-local wave index.
            k_buf_off = fx.Index(LDS_K_OFF) + buf * fx.Index(LDS_K_TILE)
            ltid = tid - fx.Index(DMA_LANES)  # 0..255 within G1
            lwave = wave_id - fx.Index(NUM_WAVES // 2)  # 0..3 within G1
            for p in range_constexpr(DMA_PASSES):
                c = fx.Index(p * DMA_LANES) + ltid
                kv = c // fx.Index(8)
                d = (c % fx.Index(8)) * fx.Index(16)
                kv_abs = kv_start + kv
                in_b = kv_abs < seq_len_v
                kv_safe = fx.Index(ArithValue(in_b).select(kv_abs, fx.Index(0)))
                voff = kv_safe * fx.Index(STRIDE_TOKEN) + d
                lds_byte = (
                    fx.Index(lds_offset)
                    + k_buf_off
                    + fx.Index(p * PASS_DMA_STRIDE)
                    + lwave * fx.Index(WAVE_DMA_STRIDE)
                )
                _dma_issue(k_rsrc, lds_byte, voff)

        def dma_v(buf, kv_start):
            # V into the canonical d-block layout in LDSV[buf], cooperatively by
            # G0 (waves 0-3).  group-local cell c = p*256 + tid maps to
            # d_block = c//128, kv = c%128; LDS byte = c*16 = d_block*2048+kv*16.
            v_buf_off = fx.Index(LDS_V_OFF) + buf * fx.Index(LDS_V_TILE)
            for p in range_constexpr(DMA_PASSES):
                c = fx.Index(p * DMA_LANES) + tid
                d_block = c // fx.Index(BLOCK_N)
                kv = c % fx.Index(BLOCK_N)
                kv_abs = kv_start + kv
                in_b = kv_abs < seq_len_v
                kv_safe = fx.Index(ArithValue(in_b).select(kv_abs, fx.Index(0)))
                voff = kv_safe * fx.Index(STRIDE_TOKEN) + d_block * fx.Index(16)
                lds_byte = (
                    fx.Index(lds_offset)
                    + v_buf_off
                    + fx.Index(p * PASS_DMA_STRIDE)
                    + wave_id * fx.Index(WAVE_DMA_STRIDE)
                )
                _dma_issue(v_rsrc, lds_byte, voff)

        def waitcnt_barrier():
            # Publish the DMA-issuing wave's global->LDS writes (vmcnt) before the
            # phase barrier so the consuming (mfma-role) waves see them.
            llvm.InlineAsmOp(
                None, [], "s_waitcnt vmcnt(0)\n\ts_barrier", "", has_side_effects=True
            )

        # ---- Preload Q packs (register resident) for this wave ----
        # A-operand for GEMM1 is K; B-operand is Q (Q^T).  But we keep Q in
        # registers as the B-operand pack:
        #   B_frag[L][v] = Q[q = lo, d = ks*64 + hi*32 + v]
        # The wave owns q-rows [q_start + wave_q_offset, +32).  In the B
        # fragment the q index is the lane lo, so each lane reads its own
        # q-row = q_start + wave_q_offset + lo.
        q_row = q_start + wave_q_offset + lo
        q_in_bounds = q_row < seq_len_v
        q_row_safe = fx.Index(ArithValue(q_in_bounds).select(q_row, fx.Index(0)))
        zero_qpack = Vec.filled(A_FP8_PER_LANE, 0, i8_dtype)

        q_packs = []
        for ks in range_constexpr(K_STEPS):
            d_col = fx.Index(ks * MFMA_K) + hi * 32
            g_idx = global_idx(q_row_safe, d_col)
            # 32 contiguous fp8 bytes along d (hi*32 + v, v in [0,32)).
            gep = fx.buffer_ops.get_element_ptr(
                q_ptr, fx.Int64(g_idx), elem_type=i8_type
            )
            raw = _pointer_load(v_i8x32, gep)
            raw = ArithValue(q_in_bounds).select(raw, zero_qpack.ir_value())
            q_packs.append(Vec(raw).bitcast(fx.Int32))  # vec<8xi32>

        # ===================================================================
        # Online-softmax loop carried state, per wave (one 32-wide tile of q).
        # The C-layout puts q in the lane (lo) and kv in the value index, so
        # each lane independently owns the running stats for q = lo.  We keep
        # m / l as scalars per lane and O as 4 vec<16xf32> (one per d-tile).
        # ===================================================================
        # L[q] = sum_kv P[kv,q] is a VALU row-sum (computed in do_softmax over the
        # 64 exact f32 exp values this lane holds, then combined with the hi-peer
        # half via shuffle_xor(32) -> full 128-kv sum).  Carried as a scalar f32
        # per lane (q = lo) and rescaled by corr each online-softmax step.

        # ---- V HW-transpose read + PV accumulate helper ----
        # Factored out so both the deferred in-loop PV (tile i-1) and the
        # epilogue PV (tile N-1) share one code path.
        v_tr8_ty = Vec.make_type(2, fx.Int32)
        lo_in_grp = lo % fx.Index(16)

        def read_v_pack(v_off, dt, ks):
            d_block = (lo // fx.Index(16)) + fx.Index(2 * dt)
            grp_db = fx.Index(lds_offset) + v_off + d_block * fx.Index(V_DBLOCK_STRIDE)
            reads = []
            for kc in range_constexpr(4):
                kv0 = hi * fx.Index(32) + fx.Index(ks * 64 + 8 * kc)
                byte_off = (
                    grp_db + kv0 * fx.Index(V_KV_STRIDE) + lo_in_grp * fx.Index(8)
                )
                ptr = fx.buffer_ops.create_llvm_ptr(fx.Int64(byte_off), address_space=3)
                reads.append(Vec(rocdl.ds_read_tr8_b64(v_tr8_ty, ptr).result))
            ab = reads[0].shuffle(reads[1], list(range(4)))
            cd = reads[2].shuffle(reads[3], list(range(4)))
            return ab.shuffle(cd, list(range(8)))

        def apply_pv(o_accs, l_acc, p_pack, p_rowsum, corr, v_off):
            # O *= corr ; then O += V^T@P over the PV_K_STEPS 64-kv K-steps that
            # tile BLOCK_N.  L is a scalar VALU online update L = L*corr + rowsum
            # (rowsum computed in do_softmax).  The PV MFMAs (matrix pipe) overlap
            # the following tile's softmax exp2 (transcendental pipe) -- deferred.
            corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(
                C_F32_PER_LANE
            )
            l2 = _fadd(_fmul(l_acc, corr), p_rowsum)
            # Scale O by corr in place on the first K-step (fold into the slot the
            # MFMA accumulates into) so the rescaled and pre-rescale copies never
            # co-exist as two full 64-VGPR lists.
            o = list(o_accs)
            for ks in range_constexpr(PV_K_STEPS):
                p_ks = Vec(p_pack).shuffle(Vec(p_pack), list(range(ks * 8, ks * 8 + 8)))
                for dt in range_constexpr(D_TILES):
                    v_pack = read_v_pack(v_off, dt, ks)
                    if const_expr(ks == 0):
                        o[dt] = _fmul(Vec(o[dt]), corr_vec)
                    o[dt] = mfma.call(v_pack, p_ks, o[dt])
            return o, l2

        def do_qk(k_buf_off):
            # GEMM1: S[kv,q] = K @ Q^T over N_KV_TILES kv subtiles.  Returns the
            # list of N_KV_TILES accumulators (each vec<16xf32>).
            s_accs = []
            for nt in range_constexpr(N_KV_TILES):
                s_acc = mfma.zero_value
                for ks in range_constexpr(K_STEPS):
                    kv_row = lo + fx.Index(nt * 32)
                    d_base = fx.Index(ks * MFMA_K) + hi * 32
                    blk_lo = Vec(
                        Vec.load(
                            v_i8x16,
                            lds,
                            [k_buf_off + kv_row * K_STRIDE + d_base],
                        )
                    ).bitcast(fx.Int32)
                    blk_hi = Vec(
                        Vec.load(
                            v_i8x16,
                            lds,
                            [k_buf_off + kv_row * K_STRIDE + d_base + fx.Index(16)],
                        )
                    ).bitcast(fx.Int32)
                    k_pack = blk_lo.shuffle(blk_hi, list(range(8)))
                    s_acc = mfma.call(k_pack, q_packs[ks], s_acc)
                s_accs.append(s_acc)
            return s_accs

        def do_softmax(s_accs, m_running):
            # Online softmax (VALU/transcendental only): returns (m_new, corr,
            # p_pack) where p_pack is the PV B-operand (P_PACK_WORDS fp8 words).
            # Pass 1: row max read directly off the carried S accumulators -- no
            # 64-element s_raw list, so the extracted S scalars never co-exist
            # with the 64 exp2 outputs (that overlap was the spill driver).
            local_max = Vec(s_accs[0])[0]
            for nt in range_constexpr(N_KV_TILES):
                for r in range_constexpr(C_F32_PER_LANE):
                    if const_expr(nt == 0 and r == 0):
                        continue
                    local_max = _fmax(local_max, Vec(s_accs[nt])[r])
            peer_max = fx.Float32(local_max).shuffle_xor(
                fx.Int32(32), fx.Int32(WARP_SIZE)
            )
            row_max_int = _fmax(local_max, peer_max)
            m_new = _fmax(m_running, row_max_int)
            corr = ArithValue(_fmul(_fsub(m_running, m_new), scale_log2e)).exp2(
                fastmath=fm_fast
            )
            neg_scaled_m_new = _fsub(c_zero_f, _fmul(scale_log2e, m_new))
            n_groups = C_F32_PER_LANE // 4

            def _p(nt, r):
                e = _fadd(_fmul(Vec(s_accs[nt])[r], scale_log2e), neg_scaled_m_new)
                return ArithValue(e).exp2(fastmath=fm_fast)

            # Pass 2: exp2 -> P packed 4-at-a-time (at most 4 exp outputs live
            # before they collapse into one fp8 i32 word), accumulating the exact
            # f32 row sum for the VALU L normalizer as we go.
            row_sum = c_zero_f
            myword = []
            for nt in range_constexpr(N_KV_TILES):
                words = []
                for rg in range_constexpr(n_groups):
                    p0 = _p(nt, rg * 4 + 0)
                    p1 = _p(nt, rg * 4 + 1)
                    p2 = _p(nt, rg * 4 + 2)
                    p3 = _p(nt, rg * 4 + 3)
                    row_sum = _fadd(row_sum, _fadd(_fadd(p0, p1), _fadd(p2, p3)))
                    words.append(_f32x4_to_fp8_word(p0, p1, p2, p3))
                myword.append(words)
            # This lane holds 64 of the 128 kv for q = lo; the hi-peer holds the
            # other 64 -> combine via shuffle_xor(32) for the full row sum.
            peer_sum = fx.Float32(row_sum).shuffle_xor(
                fx.Int32(32), fx.Int32(WARP_SIZE)
            )
            p_rowsum = _fadd(row_sum, peer_sum)

            # Reshape P from the GEMM1 C-layout to the GEMM2 B-operand: a 32-kv
            # subtile's rows are split across the hi peers, so lane hi=0 needs the
            # 16 rows physically held by its hi=1 peer (and vice versa) ->
            # shuffle_xor(32).  p_pack K-step ks reads dwords [ks*8, ks*8+8).
            peerword = [
                [
                    fx.Int32(myword[nt][rg]).shuffle_xor(
                        fx.Int32(32), fx.Int32(WARP_SIZE)
                    )
                    for rg in range_constexpr(n_groups)
                ]
                for nt in range_constexpr(N_KV_TILES)
            ]
            hi_is0 = hi < fx.Index(1)
            p_words = []
            for ks in range_constexpr(PV_K_STEPS):
                for w in range_constexpr(8):
                    rg = w // 2
                    if const_expr(w % 2 == 0):
                        sel = ArithValue(hi_is0).select(
                            _raw(myword[2 * ks + 0][rg]), _raw(peerword[2 * ks + 1][rg])
                        )
                    else:
                        sel = ArithValue(hi_is0).select(
                            _raw(peerword[2 * ks + 0][rg]), _raw(myword[2 * ks + 1][rg])
                        )
                    p_words.append(fx.Int32(sel))
            return m_new, corr, Vec.from_elements(p_words, fx.Int32), p_rowsum

        # ---- Shared init values (computed by all 512 lanes before the split) --
        m_init = c_neg_inf
        l_init = c_zero_f  # L is a scalar VALU accumulator now
        o_init = [
            Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
            for _ in range_constexpr(D_TILES)
        ]
        p_init = Vec.filled(P_PACK_WORDS, 0, fx.Int32)  # p_carry=0
        prowsum_init = c_zero_f
        corr_init = fx.Float32(1.0)  # no-op rescale at i=0
        s_init = [
            Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
            for _ in range_constexpr(N_KV_TILES)
        ]
        zero_p = Vec.filled(P_PACK_WORDS, 0, fx.Int32)

        def _bufs(kv_start):
            # Derive the K (double) / V (triple) buffer offsets from the iteration
            # index i = kv_start / BLOCK_N.  K^i in LDSK[i%2]; V^{i-1} (deferred PV
            # read) in LDSV[(i-1)%3]; next-tile DMA targets LDSK[(i+1)%2] /
            # LDSV[(i+1)%3].  At i=0 there is no V^{i-1}: point the prev read at
            # the prologue-loaded V^0 buffer (P_prev=0 nullifies it, but the PV
            # MFMA still issues a V read -- 0*NaN=NaN if it hit uninit LDS).
            i = kv_start // fx.Index(BLOCK_N)
            is_first = i < fx.Index(1)
            k_cur = i % fx.Index(NUM_BUF_K)
            k_buf_off = fx.Index(LDS_K_OFF) + k_cur * fx.Index(LDS_K_TILE)
            v_cur = i % fx.Index(NUM_BUF_V)
            v_prev = (i + fx.Index(NUM_BUF_V - 1)) % fx.Index(NUM_BUF_V)
            v_prev_sel = fx.Index(ArithValue(is_first).select(v_cur, v_prev))
            v_prev_off = fx.Index(LDS_V_OFF) + v_prev_sel * fx.Index(LDS_V_TILE)
            k_next = (i + fx.Index(1)) % fx.Index(NUM_BUF_K)
            v_next = (i + fx.Index(1)) % fx.Index(NUM_BUF_V)
            return is_first, k_buf_off, v_prev_off, k_next, v_next

        # Prologue: DMA K^0 -> LDSK0 (by G1, waves 4-7) and V^0 -> LDSV0 (by G0,
        # waves 0-3), each a full 4-wave cooperative tile load, then publish.
        is_g0 = wave_id < fx.Index(NUM_WAVES // 2)
        if is_g0:
            dma_v(fx.Index(0), fx.Index(0))
        else:
            dma_k(fx.Index(0), fx.Index(0))
        waitcnt_barrier()

        # Cross-wave role ping-pong, hoisted into TWO specialized loop bodies.
        # The role split lives OUTSIDE the loop: G0 and G1 each run their own
        # scf.for (via scf_for_dispatch).  Each iteration has exactly two phase
        # barriers (mfma phase, softmax phase); in each phase the softmax-role
        # group overlaps the next-tile global->LDS DMA with its softmax VALU:
        #   mfma phase   : G0 = matrix (PV+QK),  G1 = softmax + DMA K^{i+1}
        #   softmax phase: G0 = softmax + DMA V^{i+1},  G1 = matrix (PV+QK)
        # G0 carries P (deferred PV); G1 carries S (its phase-2 QK feeds the next
        # iter's phase-1 softmax).  Both bodies hit 2 s_barriers/iter so all 8
        # waves stay barrier-lockstep.
        loop_lo = fx.Int32(0)
        loop_step = fx.Int32(BLOCK_N)
        num_iters = (seq_len_v + fx.Index(BLOCK_N - 1)) // fx.Index(BLOCK_N)
        last_i = num_iters - fx.Index(1)
        v_last_buf = last_i % fx.Index(NUM_BUF_V)
        v_last_off = fx.Index(LDS_V_OFF) + v_last_buf * fx.Index(LDS_V_TILE)
        of0 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        of1 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        of2 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        of3 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        lf = c_zero_f

        if is_g0:
            g0_names = [
                "m_r",
                "l_a",
                "oo0",
                "oo1",
                "oo2",
                "oo3",
                "p_c",
                "prowsum_c",
                "corr_c",
            ]
            g0_init = [
                m_init,
                l_init,
                o_init[0],
                o_init[1],
                o_init[2],
                o_init[3],
                p_init,
                prowsum_init,
                corr_init,
            ]

            def g0_body(
                iv, _names, m_r, l_a, oo0, oo1, oo2, oo3, p_c, prowsum_c, corr_c
            ):
                kv_start = fx.Index(iv)
                _isf, k_buf_off, v_prev_off, _kn, v_next = _bufs(kv_start)
                next_kv = kv_start + fx.Index(BLOCK_N)
                # PHASE 1 (mfma phase): matrix -- deferred PV(i-1) then QK(i).
                oA, lA = apply_pv(
                    [oo0, oo1, oo2, oo3], l_a, p_c, prowsum_c, corr_c, v_prev_off
                )
                sA = do_qk(k_buf_off)
                gpu.barrier()
                # PHASE 2 (softmax phase): softmax(i) | DMA V^{i+1}.
                m_new, corr_new, p_new, prowsum_new = do_softmax(
                    [sA[0], sA[1], sA[2], sA[3]], m_r
                )
                dma_v(v_next, next_kv)
                waitcnt_barrier()
                return [
                    m_new,
                    lA,
                    oA[0],
                    oA[1],
                    oA[2],
                    oA[3],
                    p_new,
                    prowsum_new,
                    corr_new,
                ]

            res = scf_for_dispatch(  # noqa: F821
                loop_lo,
                seq_len,
                loop_step,
                g0_body,
                result_names=g0_names,
                result_values=g0_init,
            )
            # G0 epilogue: still owes PV(N-1) (softmax(N-1) ran in last phase 2).
            m_e = res[0]  # noqa: F841
            l_e = res[1]
            oe0 = res[2]
            oe1 = res[3]
            oe2 = res[4]
            oe3 = res[5]
            p_e = res[6]
            prowsum_e = res[7]
            corr_e = res[8]
            o_fin, l_fin_g = apply_pv(
                [oe0, oe1, oe2, oe3], l_e, p_e, prowsum_e, corr_e, v_last_off
            )
            of0 = o_fin[0]
            of1 = o_fin[1]
            of2 = o_fin[2]
            of3 = o_fin[3]
            lf = l_fin_g
        else:
            g1_names = [
                "m_r",
                "l_a",
                "oo0",
                "oo1",
                "oo2",
                "oo3",
                "ss0",
                "ss1",
                "ss2",
                "ss3",
            ]
            g1_init = [
                m_init,
                l_init,
                o_init[0],
                o_init[1],
                o_init[2],
                o_init[3],
                s_init[0],
                s_init[1],
                s_init[2],
                s_init[3],
            ]

            def g1_body(iv, _names, m_r, l_a, oo0, oo1, oo2, oo3, ss0, ss1, ss2, ss3):
                kv_start = fx.Index(iv)
                is_first, k_buf_off, v_prev_off, k_next, _vn = _bufs(kv_start)
                next_kv = kv_start + fx.Index(BLOCK_N)
                # PHASE 1 (mfma phase): softmax(i-1) | DMA K^{i+1}.
                m_sm, corr_sm, p_sm, prowsum_sm = do_softmax([ss0, ss1, ss2, ss3], m_r)
                mA = ArithValue(is_first).select(m_r, m_sm)
                corrA = ArithValue(is_first).select(fx.Float32(1.0), corr_sm)
                pA = ArithValue(is_first).select(zero_p.ir_value(), _raw(p_sm))
                prowsumA = ArithValue(is_first).select(_raw(c_zero_f), prowsum_sm)
                dma_k(k_next, next_kv)
                waitcnt_barrier()
                # PHASE 2 (softmax phase): matrix -- deferred PV(i-1), QK(i)->S.
                oB, lB = apply_pv(
                    [oo0, oo1, oo2, oo3], l_a, pA, prowsumA, corrA, v_prev_off
                )
                sB = do_qk(k_buf_off)
                gpu.barrier()
                return [
                    mA,
                    lB,
                    oB[0],
                    oB[1],
                    oB[2],
                    oB[3],
                    sB[0],
                    sB[1],
                    sB[2],
                    sB[3],
                ]

            res = scf_for_dispatch(  # noqa: F821
                loop_lo,
                seq_len,
                loop_step,
                g1_body,
                result_names=g1_names,
                result_values=g1_init,
            )
            # G1 epilogue: owes BOTH softmax(N-1) (carried S) and PV(N-1).
            m_e = res[0]
            l_e = res[1]
            oe0 = res[2]
            oe1 = res[3]
            oe2 = res[4]
            oe3 = res[5]
            sce0 = res[6]
            sce1 = res[7]
            sce2 = res[8]
            sce3 = res[9]
            _m_e, corrf, pf, prowsumf = do_softmax([sce0, sce1, sce2, sce3], m_e)
            o_fin, l_fin_g = apply_pv(
                [oe0, oe1, oe2, oe3], l_e, pf, prowsumf, corrf, v_last_off
            )
            of0 = o_fin[0]
            of1 = o_fin[1]
            of2 = o_fin[2]
            of3 = o_fin[3]
            lf = l_fin_g

        o_finals = [of0, of1, of2, of3]
        # L is the scalar VALU row-sum normalizer for this lane (q = lo).
        l_final = lf

        inv_l = rocdl.rcp(T.f32, l_final)
        inv_l_v = _fmul(inv_l, v_descale)
        inv_l_vec = Vec.from_elements([inv_l_v], fx.Float32).broadcast_to(
            C_F32_PER_LANE
        )

        if q_in_bounds:
            for dt in range_constexpr(D_TILES):
                o_norm = Vec(o_finals[dt]) * inv_l_vec
                for r in range_constexpr(C_F32_PER_LANE):
                    o_val = Vec(o_norm)[r]
                    o_bf16 = fx.Float32(o_val).to(bf16_dtype)
                    # O C-layout: row = d = hi*4 + (r%4) + 8*(r//4) (+ dt*32);
                    # col = q = lo.  But this wave's q = q_row (= lo-based).
                    d_row = hi * 4 + (r % 4) + 8 * (r // 4) + dt * 32
                    o_global = global_idx(q_row, fx.Index(d_row))
                    gep = fx.buffer_ops.get_element_ptr(
                        o_ptr, fx.Int64(o_global), elem_type=bf16_type
                    )
                    _pointer_store(o_bf16, gep)

    @flyc.jit
    def launch_fp8_attn(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        q_descale: fx.Float32,
        k_descale: fx.Float32,
        v_descale: fx.Float32,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
        num_q_tiles = (sl_idx + BLOCK_M - 1) // BLOCK_M
        grid_x = bs_idx * num_q_tiles * NUM_HEADS

        fp8_attn_kernel(
            Q,
            K,
            V,
            O,
            q_descale,
            k_descale,
            v_descale,
            seq_len,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": f"{BLOCK_SIZE},{BLOCK_SIZE}",
            },
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    def _compile(  # noqa: E741
        Q,
        K,
        V,
        O,  # noqa: E741
        q_descale,
        k_descale,
        v_descale,
        batch_size,
        seq_len,
        stream=None,
    ):
        return flyc.compile(
            launch_fp8_attn,
            Q,
            K,
            V,
            O,
            q_descale,
            k_descale,
            v_descale,
            batch_size,
            seq_len,
            fx.Stream(stream),
        )

    launch_fp8_attn.compile = _compile
    return launch_fp8_attn


compile_flash_attn_fp8 = build_flash_attn_fp8_module
