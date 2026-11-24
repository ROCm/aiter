# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MHA backward, head dim 128, bf16, gfx1250 (MI400 / mi450) -- FlyDSL device code.

A FlyDSL port of the gfx1250 asm backward (``hsa/gfx1250/fmha_v3_bwd/bwd_hd128_bf16[_causal_br]_a32_pssk.co``):
same three kernels, same tiling, same dataflow.

``k_odo``     D = rowsum(dO * O) in fp32, and zero-fills the fp32 dQ accumulator.
``k_dqdkdv``  One workgroup = 4 waves (wave32) owning a 128-key block, 32 keys per wave.
              K, V and K^T stay in VGPRs; dK^T / dV^T accumulate in VGPRs across every
              32-row Q tile. Per tile and wave, five 16-WMMA GEMMs
              (``v_wmma_f32_16x16x32_bf16``):

                gemm0  S     = Q . K^T        A = Q (LDS)           B = K (VGPR)
                gemm1  dP-D  = dO . V^T - D   A = dO (LDS)          B = V (VGPR), -D via neg C
                gemm2  dV^T += dO^T . P       A = dO^T (LDS, tr16)  B = P (S accumulator layout)
                gemm3  dK^T += Q^T . dS       A = Q^T (LDS, tr16)   B = dS
                gemm4  dQ    = dS . K         A = dS of all 4 waves (LDS, tr16)  B = K^T (VGPR)

              dQ goes to the accumulator with fp32 structured-buffer atomics.
``k_cvt``     dQ = bf16(dQ_acc * softmax_scale).

Pipeline of ``k_dqdkdv``:

* Persistent workgroups (one per CU) walk the work items in XCD-swizzled order (runs of 64 consecutive items per
  XCD, so the KV blocks of a head share an L2); the K / V and first Q tiles of the next block are fetched by TDM
  while the current block drains.
* Q / dO / LSE / D stream through a 6-slot LDS ring filled by TDM: per wave and tile, 16 full rows of Q (waves 0/1) or
  dO (waves 2/3) into 272 B padded rows, and 16 values of LSE or D. Q row half h lives in LDS segment h, dO in segment
  2+h; K / V / dS of wave w in segment w.
* One split barrier per Q tile. Iteration t:
    A  gemm0/1 of tile t+1   | dQ atomics of tile t-1, Q^T / dO^T reads of tile t, softmax of t+1 (half)
    B  gemm2/3 of tile t     | softmax of tile t+1 (rest), dS^T reads of tile t, TDM of tile t+5, more atomics,
                               part of the Q / dO reads of tile t+2
    C  gemm4 of tile t       | the other Q / dO / LSE / D reads of tile t+2, dS(t+1) store, more atomics
  so P / dS of t+1, dQ of t and the operands of t+2 are loop-carried.
* Every 8-cycle WMMA window gets an even share of the side work (its order pinned with sched_barrier fences):
  WMMA first, then its memory ops, then at most two VALU ops of the softmax. The SALU setup of a TDM is spread over
  windows of phase A that hold a single memory op.
* The LDS lane bases and the dQ atomic index live in VGPR segment 0, the segment of the WMMA / VALU sources, so
  no s_set_vgpr_msb follows a memory op (it would not co-issue there); the dQ element offsets sit in the offset
  field of the structured-buffer atomics.
* The unmasked tiles run in a loop unrolled by the ring size, like the asm: ring slots and dS buffers are then
  trace-time constants and each LDS address is a lane base plus an immediate. Its trip counter is i32.
* Causal (bottom-right): a workgroup takes KV blocks bx and n_kb-1-bx (the asm kb fold);
  only Q tiles that cross the diagonal run the masked body.
* ``-amdgpu-anti-hints-for-va-vdst`` (llvm/llvm-project#226397) keeps the register allocator from handing a load
  the VGPRs that a WMMA or VALU op still reads; without it the loads of the co-executing schedule wait on those WAR
  hazards (``s_wait_alu depctr_va_vdst(0)``). ``k_dqdkdv`` requests it when FlyDSL's LLVM registers the option
  (FlyDSL 0.5.3's does not); the kernels are correct either way.
"""

import functools
import warnings

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import scf
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _ir

HDIM = 128
SUB_QO = 32  # Q rows per tile
SUB_KV = 128  # keys per workgroup
WAVES = 4
WV_KV = SUB_KV // WAVES  # keys per wave
BLOCK = WAVES * 32
LOG2E = 1.4426950408889634

BF16 = fx.BFloat16
F32 = fx.Float32

# ---- LDS: 4 x 64 KB segments, each laid out the same way ----
PF = 6  # ring slots and unroll factor (even: 2 dS buffers); tile t+2 is resident at iteration t's barrier
SEG = 64 * 1024
ROW_B = (HDIM + 8) * 2  # 272 B padded Q / dO / K / V row
HALF_B = 16 * ROW_B  # one 16-row half of a Q or dO ring slot
KV_OFF = PF * HALF_B
V_SUB = WV_KV * ROW_B
DS_ROW_B = 176  # dS^T row: 8-q group g at (g % 2) * 128 + (g // 2) * 16
DS_BUF_B = WV_KV * DS_ROW_B
DS_OFF = KV_OFF + 2 * V_SUB
LSED_OFF = DS_OFF + 2 * DS_BUF_B
SEG_USED = LSED_OFF + PF * 64
LDS_BYTES = 3 * SEG + SEG_USED
# per wave, its dK and dV as bf16 rows in the K / V row layout, staged for the epilogue's tensor stores
STG_OFF = LDS_BYTES
STG_W = 2 * WV_KV * ROW_B
LDS_BYTES = STG_OFF + 4 * STG_W
assert LDS_BYTES <= 320 * 1024
assert SEG_USED <= SEG and PF % 2 == 0

# TDM padding (data size 1 B): interval = 2^(enc+1) DW, amount = enc+1 DW
SWZ_CHUNK_LOG2 = 6  # 64 work items per XCD run, as in the asm
SWZ_CHUNK = 1 << SWZ_CHUNK_LOG2
# one persistent dqdkdv workgroup per CU (1 wave per SIMD: a workgroup fills its CU)
N_CU = (
    torch.cuda.get_device_properties(0).multi_processor_count
    if torch.cuda.is_available()
    else 256
)
KV_PAD = (1 << 20) | (5 << 22) | (3 << 25)  # 256 B data + 16 B pad
TDM_PER_TILE = (
    2  # tensor loads per wave and Q tile: 16 full rows of Q or dO, then LSE or D
)
DQ_AUX = 16  # dQ atomics at device scope

ADIST = (6, 18, 8)  # dQ atomics of tile t-1 issued during phases A / B / C
ODO_ROWS = 32
CVT_ROWS = 32
ANTI_HINTS = "amdgpu-anti-hints-for-va-vdst"


# ---------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------


@functools.cache
def _llvm_has_option(name):
    """True when FlyDSL's LLVM registers the cl::opt ``name``: setting an unknown option raises in FlyDSL 0.5.3
    and warns in later versions."""
    from flydsl.compiler.llvm_options import llvm_options

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with llvm_options({name: False}):
                pass
    except (RuntimeError, RuntimeWarning):
        return False
    return True


def _lane_id():
    return fx.Int32(
        rocdl.mbcnt_lo(T.i32, fx.Int32(-1).ir_value(), fx.Int32(0).ir_value())
    )


def _wave_id():
    return fx.Int32(rocdl.wave_id())


def _base_i64(t):
    return fx.Int64(fx.ptrtoint(fx.get_iter(t)))


def _gptr(addr_i64, elem=fx.Int8, align=16):
    ty = fx.PointerType.get(elem.ir_type, fx.AddressSpace.Global, alignment=align)
    return fx.inttoptr(ty, addr_i64)


def _lptr(addr_i32, elem=fx.Int8, align=16):
    ty = fx.PointerType.get(elem.ir_type, fx.AddressSpace.Shared, alignment=align)
    return fx.inttoptr(ty, fx.Int32(addr_i32))


def _v8bf16():
    return fx.Vector.make_type(8, BF16)


def _lds_v8(addr):
    return fx.ptr_load(_lptr(addr), result_type=_v8bf16())


def _lds_tr_v8(addr):
    return fx.Vector(rocdl.ds_load_tr16_b128(_v8bf16(), fx.to_llvm_ptr(_lptr(addr))))


def _cat(a, b):
    return a.shuffle(b, list(range(2 * a.numel)))


def _wmma(a, b, c):
    v8f32 = fx.Vector.make_type(8, F32)
    return fx.Vector(
        rocdl.wmma_f32_16x16x32_bf16(
            v8f32, _ir(a), _ir(b), _ir(c), reuseA=False, reuseB=False
        ).result
    )


@functools.cache
def _neg_c_text():
    """The WMMA C-negate attribute in the ROCDL syntax of FlyDSL's MLIR (it changed with the LLVM e2ce52c885 pin)."""
    for text in ("#rocdl.wmma_c_modifier<neg>", "#rocdl<wmma_c_modifier neg>"):
        try:
            ir.Attribute.parse(text)
            return text
        except ir.MLIRError:
            pass
    raise RuntimeError("no known syntax of the ROCDL WMMA C modifier parses")


def _wmma_negc(a, b, c):
    """a . b - c (C negate modifier): folds -D into the first dP WMMA."""
    v8f32 = fx.Vector.make_type(8, F32)
    neg = ir.Attribute.parse(_neg_c_text())
    return fx.Vector(
        rocdl.wmma_f32_16x16x32_bf16(
            v8f32, _ir(a), _ir(b), _ir(c), modC=neg, reuseA=False, reuseB=False
        ).result
    )


def _fence():
    rocdl.sched_barrier(0)


def _keepalive(vals):
    """Empty side-effecting asm that uses vals, so their VGPRs are not handed to the next
    load while the WMMA that read them may still be in flight (a WAR stall otherwise).
    """
    ops = [_ir(v) for v in vals]
    if ops:
        llvm_dialect.inline_asm(
            None, ops, "", ",".join("v" for _ in ops), has_side_effects=True
        )


def _pin_v(val, reg):
    """val moved into VGPR v<reg>: the register allocator keeps the value there (copy hint), which puts the
    address operand of every ds op using it into VGPR segment 0, the segment of the WMMA / VALU sources around
    it, so no s_set_vgpr_msb is needed between them. The asm is empty: the move is a COPY of LLVM's own, which
    its expert-scheduling waits know about (a VALU write inside the asm text would not get the s_wait_alu
    a ds op reading the register right behind it needs)."""
    return fx.Int32(
        llvm_dialect.inline_asm(
            T.i32, [_ir(val)], "", f"={{v{reg}}},0", has_side_effects=False
        )
    )


def _rsrc_records4(addr_i64):
    """Buffer resource at addr_i64 with 4-byte records (stride 4) and no bound."""
    ptr_type = fx.PointerType.get(T.i8, fx.AddressSpace.Generic, 1)
    base_ptr = fx.to_llvm_ptr(fx.inttoptr(ptr_type, _ir(addr_i64)))
    return fx.rocdl.make_buffer_rsrc(
        ir.Type.parse("!llvm.ptr<8>"),
        base_ptr,
        fx.Int16(4).ir_value(),
        fx.Int64(0xFFFFFFFF).ir_value(),
        fx.Int32(buffer_ops._get_buffer_flags()).ir_value(),
    )


def _atomic_fadd_idx(v, rsrc, vidx, off, soff):
    """buffer_atomic_add_f32 v at record vidx + byte offset off (a constant: the offset field) + soff (SGPR)."""
    llvm_dialect.call_intrinsic(
        T.f32,
        "llvm.amdgcn.struct.ptr.buffer.atomic.fadd.f32",
        [_ir(v), rsrc, _ir(vidx), _ir(fx.Int32(off)), _ir(soff), _ir(fx.Int32(DQ_AUX))],
        [],
        [],
    )


def _pin_v16(vals, reg):
    """16 values moved together into VGPRs v<reg> .. v<reg + 15> as one tuple: the register allocator places a
    16-VGPR tuple ahead of the 8-VGPR operand tuples, so the copy hint holds where single VGPRs lose it.
    """
    vec = fx.Vector.from_elements(vals, fx.Int32)
    ty = fx.Vector.make_type(16, fx.Int32)
    out = fx.Vector(
        llvm_dialect.inline_asm(
            ty, [_ir(vec)], "", f"={{v[{reg}:{reg + 15}]}},0", has_side_effects=False
        )
    )
    return [out[x] for x in range(16)]


def _tdm_load(lds_addr, gaddr, g1, store=False):
    lo = fx.Int32(gaddr)
    hi = fx.Int32(gaddr >> fx.Int64(32))
    g0 = fx.Vector.from_elements(
        [
            fx.Int32(1),
            fx.Int32(lds_addr),
            lo,
            (hi & fx.Int32(0x01FFFFFF)) | fx.Int32(-(1 << 31)),
        ],
        fx.Int32,
    )
    g1v = fx.Vector.from_elements([fx.Int32(x) for x in g1], fx.Int32)
    z4 = fx.Vector.from_elements([fx.Int32(0)] * 4, fx.Int32)
    z8 = fx.Vector.from_elements([fx.Int32(0)] * 8, fx.Int32)
    (rocdl.tensor_store_from_lds if store else rocdl.tensor_load_to_lds)(
        _ir(g0), _ir(g1v), _ir(z4), _ir(z4), _ir(z8)
    )


def _tdm_desc(lds_addr, gaddr, g1):
    """D# groups 0 / 1 of a tensor load as SGPR vectors, built ahead of the load (its SALU work then lands in
    WMMA windows with spare issue cycles, not in the window of the load)."""
    lo = fx.Int32(gaddr)
    hi = fx.Int32(gaddr >> fx.Int64(32))
    g0 = fx.Vector.from_elements(
        [
            fx.Int32(1),
            fx.Int32(lds_addr),
            lo,
            (hi & fx.Int32(0x01FFFFFF)) | fx.Int32(-(1 << 31)),
        ],
        fx.Int32,
    )
    return g0, fx.Vector.from_elements([fx.Int32(x) for x in g1], fx.Int32)


def _tdm_issue(desc):
    z4 = fx.Vector.from_elements([fx.Int32(0)] * 4, fx.Int32)
    z8 = fx.Vector.from_elements([fx.Int32(0)] * 8, fx.Int32)
    rocdl.tensor_load_to_lds(_ir(desc[0]), _ir(desc[1]), _ir(z4), _ir(z4), _ir(z8))


def _tdm_rows_g1(row_bytes, rows_valid, tile_rows, stride_bytes, pad_cfg):
    dim1 = fx.max(rows_valid, fx.Int32(0))
    return [
        pad_cfg,
        (row_bytes & 0xFFFF) << 16,
        ((row_bytes >> 16) & 0xFFFF) | (dim1 << fx.Int32(16)),
        fx.Int32(row_bytes << 16) | (dim1 >> fx.Int32(16)),
        tile_rows,
        stride_bytes,
        0,
        0,
    ]


def _tdm_vec_g1(n_valid, n):
    dim0 = fx.max(n_valid, fx.Int32(0)) * 4
    return [
        0,
        (dim0 & fx.Int32(0xFFFF)) << fx.Int32(16),
        ((dim0 >> fx.Int32(16)) & fx.Int32(0xFFFF)) | fx.Int32(1 << 16),
        ((n * 4) << 16),
        1,
        n * 4,
        0,
        0,
    ]


def _tdm_rows(
    lds_addr,
    gaddr,
    row_bytes,
    rows_valid,
    tile_rows,
    stride_bytes,
    pad_cfg,
    store=False,
):
    """tile_rows x row_bytes, rows from rows_valid on are zero-filled by the TDM bound (dropped, for a store).
    The bound is only clamped at 0 (no min with the tile): a clamp pair becomes v_med3 and
    drags the whole descriptor into VGPRs."""
    dim1 = fx.max(rows_valid, fx.Int32(0))
    g1 = [
        pad_cfg,
        (row_bytes & 0xFFFF) << 16,
        ((row_bytes >> 16) & 0xFFFF) | (dim1 << fx.Int32(16)),
        fx.Int32(row_bytes << 16) | (dim1 >> fx.Int32(16)),
        tile_rows,
        stride_bytes,
        0,
        0,
    ]
    _tdm_load(lds_addr, gaddr, g1, store)


def _tdm_store_rows(
    lds_addr, gaddr, row_bytes, lds_row_bytes, rows_valid, tile_rows, stride_bytes
):
    """Store tile_rows rows from LDS rows lds_row_bytes apart: a tensor store ignores the LDS padding, so the
    tile row spans the padded LDS row and its tail past the tensor row (row_bytes) is dropped by the bound,
    as are the rows from rows_valid on."""
    dim1 = fx.max(rows_valid, fx.Int32(0))
    g1 = [
        0,
        (row_bytes & 0xFFFF) << 16,
        ((row_bytes >> 16) & 0xFFFF) | (dim1 << fx.Int32(16)),
        fx.Int32(lds_row_bytes << 16) | (dim1 >> fx.Int32(16)),
        tile_rows,
        stride_bytes,
        0,
        0,
    ]
    _tdm_load(lds_addr, gaddr, g1, store=True)


def _tdm_vec(lds_addr, gaddr, n_valid, n):
    """n contiguous fp32, zero-filled from n_valid on."""
    dim0 = fx.max(n_valid, fx.Int32(0)) * 4
    g1 = [
        0,
        (dim0 & fx.Int32(0xFFFF)) << fx.Int32(16),
        ((dim0 >> fx.Int32(16)) & fx.Int32(0xFFFF)) | fx.Int32(1 << 16),
        ((n * 4) << 16),
        1,
        n * 4,
        0,
        0,
    ]
    _tdm_load(lds_addr, gaddr, g1)


def _spread(thunks, n_steps):
    """Split thunks into n_steps nearly equal consecutive groups."""
    out, k = [], 0
    for i in range(n_steps):
        n = (len(thunks) - k) // (n_steps - i)
        out.append(thunks[k : k + n])
        k += n
    return out


# ---------------------------------------------------------------------------------
# odo: D[b, h, i] = sum_d dO * O, and dq_acc[b, h, i, :] = 0
# ---------------------------------------------------------------------------------


@functools.cache
def build_odo():
    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def k_fmha_bwd_odo_hd128(
        o: fx.Tensor,
        do: fx.Tensor,
        d_out: fx.Tensor,
        dq_acc: fx.Tensor,
        o_sb: fx.Int32,
        o_ss: fx.Int32,
        o_sh: fx.Int32,
        do_sb: fx.Int32,
        do_ss: fx.Int32,
        do_sh: fx.Int32,
        d_sb: fx.Int32,
        d_sh: fx.Int32,
        seqlen_q: fx.Int32,
        sq_pad: fx.Int32,
        nhead: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        h = fx.Int32(gpu.block_id("y"))
        b = fx.Int32(gpu.block_id("z"))
        o_base = (
            _base_i64(o)
            + (fx.Int64(b) * fx.Int64(o_sb) + fx.Int64(h) * fx.Int64(o_sh)) * 2
        )
        do_base = (
            _base_i64(do)
            + (fx.Int64(b) * fx.Int64(do_sb) + fx.Int64(h) * fx.Int64(do_sh)) * 2
        )
        d_base = (
            _base_i64(d_out)
            + (fx.Int64(b) * fx.Int64(d_sb) + fx.Int64(h) * fx.Int64(d_sh)) * 4
        )
        acc_base = _base_i64(dq_acc) + (
            (fx.Int64(b) * fx.Int64(nhead) + fx.Int64(h)) * fx.Int64(sq_pad)
        ) * (HDIM * 4)
        row_in = tid // 16  # 8 rows per pass, 16 lanes x 8 bf16 per row
        chunk = tid % 16
        zero_v8 = fx.full(8, 0.0, BF16)
        for p in range_constexpr(ODO_ROWS // 8):
            row = bx * ODO_ROWS + p * 8 + row_in
            valid = row < seqlen_q
            row_c = valid.select(row, fx.Int32(0))
            vo = fx.ptr_load(
                _gptr(
                    o_base
                    + (fx.Int64(row_c) * fx.Int64(o_ss) + fx.Int64(chunk * 8)) * 2
                ),
                result_type=_v8bf16(),
            )
            vd = fx.ptr_load(
                _gptr(
                    do_base
                    + (fx.Int64(row_c) * fx.Int64(do_ss) + fx.Int64(chunk * 8)) * 2
                ),
                result_type=_v8bf16(),
            )
            vo = valid.select(vo, zero_v8)
            vd = valid.select(vd, zero_v8)
            pf = vo.to(F32) * vd.to(F32)
            s = pf[0]
            for e in range_constexpr(1, 8):
                s = s + pf[e]
            for sh in (8, 4, 2, 1):
                s = s + s.shuffle_xor(fx.Int32(sh), fx.Int32(32))
            if (chunk == fx.Int32(0)) & valid:
                fx.ptr_store(s, _gptr(d_base + fx.Int64(row) * 4, elem=F32, align=4))
        zero_f4 = fx.full(4, 0.0, F32)
        for p in range_constexpr(ODO_ROWS * HDIM // 4 // BLOCK):
            idx = p * BLOCK + tid
            row = bx * ODO_ROWS + idx // (HDIM // 4)
            col4 = idx % (HDIM // 4)
            if row < sq_pad:
                fx.ptr_store(
                    zero_f4,
                    _gptr(
                        acc_base + (fx.Int64(row) * HDIM + fx.Int64(col4 * 4)) * 4,
                        elem=F32,
                        align=16,
                    ),
                )

    @flyc.jit
    def launch_odo(
        o: fx.Tensor,
        do: fx.Tensor,
        d_out: fx.Tensor,
        dq_acc: fx.Tensor,
        o_sb: fx.Int32,
        o_ss: fx.Int32,
        o_sh: fx.Int32,
        do_sb: fx.Int32,
        do_ss: fx.Int32,
        do_sh: fx.Int32,
        d_sb: fx.Int32,
        d_sh: fx.Int32,
        seqlen_q: fx.Int32,
        sq_pad: fx.Int32,
        nhead: fx.Int32,
        batch: fx.Int32,
        stream: fx.Stream,
    ):
        gx = (sq_pad + ODO_ROWS - 1) // ODO_ROWS
        k_fmha_bwd_odo_hd128(
            o,
            do,
            d_out,
            dq_acc,
            o_sb,
            o_ss,
            o_sh,
            do_sb,
            do_ss,
            do_sh,
            d_sb,
            d_sh,
            seqlen_q,
            sq_pad,
            nhead,
        ).launch(grid=(gx, nhead, batch), block=(BLOCK, 1, 1), stream=stream)

    return launch_odo


# ---------------------------------------------------------------------------------
# convert: dq[b, i, h, :] = bf16(dq_acc[b, h, i, :] * scale)
# ---------------------------------------------------------------------------------


@functools.cache
def build_convert():
    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def k_fmha_bwd_dq_convert_hd128(
        dq_acc: fx.Tensor,
        dq: fx.Tensor,
        dq_sb: fx.Int32,
        dq_ss: fx.Int32,
        dq_sh: fx.Int32,
        seqlen_q: fx.Int32,
        sq_pad: fx.Int32,
        nhead: fx.Int32,
        scale: fx.Float32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        h = fx.Int32(gpu.block_id("y"))
        b = fx.Int32(gpu.block_id("z"))
        acc_base = _base_i64(dq_acc) + (
            (fx.Int64(b) * fx.Int64(nhead) + fx.Int64(h)) * fx.Int64(sq_pad)
        ) * (HDIM * 4)
        dq_base = (
            _base_i64(dq)
            + (fx.Int64(b) * fx.Int64(dq_sb) + fx.Int64(h) * fx.Int64(dq_sh)) * 2
        )
        row_in = tid // 16
        chunk = tid % 16
        for p in range_constexpr(CVT_ROWS // 8):
            row = bx * CVT_ROWS + p * 8 + row_in
            if row < seqlen_q:
                # dQ_acc 16-row block layout of the dqdkdv atomics: [row % 8][16-column group][row // 8 % 2][16 columns]
                sw = (row // 16) * (16 * HDIM) + (
                    ((row % 8) * (HDIM // 16) + chunk // 2) * 2 + (row // 8) % 2
                ) * 16
                a = acc_base + fx.Int64(sw + (chunk % 2) * 8) * 4
                lo = fx.ptr_load(_gptr(a), result_type=fx.Vector.make_type(4, F32))
                hi = fx.ptr_load(_gptr(a + 16), result_type=fx.Vector.make_type(4, F32))
                v = _cat(lo, hi) * scale
                fx.ptr_store(
                    v.to(BF16),
                    _gptr(
                        dq_base
                        + (fx.Int64(row) * fx.Int64(dq_ss) + fx.Int64(chunk * 8)) * 2
                    ),
                )

    @flyc.jit
    def launch_cvt(
        dq_acc: fx.Tensor,
        dq: fx.Tensor,
        dq_sb: fx.Int32,
        dq_ss: fx.Int32,
        dq_sh: fx.Int32,
        seqlen_q: fx.Int32,
        sq_pad: fx.Int32,
        nhead: fx.Int32,
        batch: fx.Int32,
        scale: fx.Float32,
        stream: fx.Stream,
    ):
        gx = (seqlen_q + CVT_ROWS - 1) // CVT_ROWS
        k_fmha_bwd_dq_convert_hd128(
            dq_acc, dq, dq_sb, dq_ss, dq_sh, seqlen_q, sq_pad, nhead, scale
        ).launch(grid=(gx, nhead, batch), block=(BLOCK, 1, 1), stream=stream)

    return launch_cvt


# ---------------------------------------------------------------------------------
# dqdkdv
# ---------------------------------------------------------------------------------


@functools.cache
def build_dqdkdv(causal: bool):
    CAUSAL = bool(causal)

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def k_fmha_bwd_dqdkdv_hd128(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        do: fx.Tensor,
        lse: fx.Tensor,
        dd: fx.Tensor,
        dq_acc: fx.Tensor,
        dk: fx.Tensor,
        dv: fx.Tensor,
        q_sb: fx.Int32,
        q_ss: fx.Int32,
        q_sh: fx.Int32,
        k_sb: fx.Int32,
        k_ss: fx.Int32,
        k_sh: fx.Int32,
        v_sb: fx.Int32,
        v_ss: fx.Int32,
        v_sh: fx.Int32,
        do_sb: fx.Int32,
        do_ss: fx.Int32,
        do_sh: fx.Int32,
        dk_sb: fx.Int32,
        dk_ss: fx.Int32,
        dk_sh: fx.Int32,
        dv_sb: fx.Int32,
        dv_ss: fx.Int32,
        dv_sh: fx.Int32,
        lse_sb: fx.Int32,
        lse_sh: fx.Int32,
        d_sb: fx.Int32,
        d_sh: fx.Int32,
        seqlen_q: fx.Int32,
        seqlen_k: fx.Int32,
        sq_pad: fx.Int32,
        nhead: fx.Int32,
        ratio: fx.Int32,
        scale_log2e: fx.Float32,
        scale: fx.Float32,
        seg: fx.Int32,  # SEG as a runtime value: keeps the 64 KB segment offsets out of the ds immediates
        batch: fx.Int32,
    ):
        lane = _lane_id()
        wave = _wave_id()
        wg = fx.Int32(gpu.block_id("x"))
        n_wgs = fx.Int32(gpu.grid_dim.x)

        smem = fx.SharedAllocator().allocate(LDS_BYTES)
        lds = fx.Int32(fx.ptrtoint(smem.peek().ptr))

        w_half = wave % 2  # Q / dO row half this wave copies
        seg_base = [lds + seg * s_ for s_ in range(4)]
        seg_w = lds + wave * seg
        lane16 = lane % 16
        lane_hi = lane // 16
        q_lane = lane_hi * 8
        # A/B operand with K along an LDS row: row lane16, K elements 8 lane_hi + [0,8) and +16
        rm_off = lane16 * ROW_B + lane_hi * 16
        # tr16 operand from a [K][M] block: K row 8 lane_hi + lane%8, M column 8 ((lane//8)%2)
        tr_off = (lane_hi * 8 + lane % 8) * ROW_B + ((lane // 8) % 2) * 16
        # Q^T / dO^T: the second 8-lane group reads 64 columns over, which keeps the transposed
        # read of a 272 B row conflict-free; d-tile e is then d = 8e + [0,8) and 64 + 8e + [0,8),
        # so dK^T / dV^T lane l holds d = 64 (l // 16) + 8e + n.
        trq_off = (lane_hi * 8 + lane % 8) * ROW_B + ((lane // 8) % 2) * 128
        ds_tr_off = (lane_hi * 8 + lane % 8) * DS_ROW_B + ((lane // 8) % 2) * 128
        ds_st_off = lane16 * DS_ROW_B + lane_hi * 128
        # dQ element (i, e, n): row 16 i + 8 lane_hi + n, column 32 wave + 16 e + lane16. dQ_acc keeps each
        # 16-row block as [row % 8][16-column group][row // 8 % 2][16 columns], so the element sits at
        # 16 i HDIM + 32 (8 n + e) + 64 wave + lane and the 32 lanes of an atomic fill one 128 B line
        # (rows r and r + 8 of one 16-column group) instead of half of two.
        dq_vidx = lane + 64 * wave  # dQ_acc record of this lane (4 B records)
        q_ss_b = q_ss * 2
        do_ss_b = do_ss * 2
        causal_off = seqlen_k - seqlen_q
        zero_acc = fx.full(8, 0.0, F32)
        n_kb = (seqlen_k + SUB_KV - 1) // SUB_KV

        # Loop-invariant lane bases of the ring / exchange reads: with a slot or buffer index known
        # at trace time the offset folds into the ds immediate, so the loop computes no address.
        trq_base = [seg_base[s_] + trq_off for s_ in range(4)]
        rm_base = [seg_base[s_] + rm_off for s_ in range(4)]
        lsed_base = [seg_base[s_] + LSED_OFF + q_lane * 4 for s_ in range(4)]
        dstr_base = [seg_base[c] + DS_OFF + ds_tr_off for c in range(4)]
        dsst_base = seg_w + DS_OFF + ds_st_off
        # The lane bases of the ds ops and dQ atomics in VGPR segment 0, the segment of the WMMA / VALU sources
        # around them, so no s_set_vgpr_msb (which cannot co-issue there) follows a memory op. The 16 LDS bases go
        # as one 16-VGPR tuple: single pinned VGPRs lose their place to the 8-VGPR operand tuples.
        lbv = _pin_v16(trq_base + rm_base + lsed_base + dstr_base, 240)
        lb = {
            "trq": lbv[0:4],
            "rm": lbv[4:8],
            "lsed": lbv[8:12],
            "dstr": lbv[12:16],
            "dsst": _pin_v(dsst_base, 238),
            "vidx": _pin_v(dq_vidx, 239),
        }

        def issue_tile(m, q_lo, slot, bs):
            """TDM this wave's share of Q tile m (Q, dO, and LSE or D) of head bases bs into ring slot `slot`."""
            for x in issue_tile_thunks(m, q_lo, slot, bs):
                x()

        def issue_tile_thunks(m, q_lo, slot, bs):
            """issue_tile as its TDM_PER_TILE tensor loads, so a phase can overlap their SALU setup with WMMAs."""
            return [
                lambda: _tdm_qdo(m, q_lo, slot, bs),
                lambda: _tdm_lsed(m, q_lo, slot, bs),
            ]

        def tdm_prep_thunks(m, q_lo, slot, bs, desc):
            """The SALU setup of issue_tile_thunks as seven small thunks, in this order; desc[0] / desc[1] get
            the D# of the Q-or-dO and the LSE-or-D load, for _tdm_issue."""
            st = {}

            def rows():
                st["r0"] = q_lo + m * SUB_QO + w_half * 16
                st["rows"] = seqlen_q - st["r0"]

            def lsed_addr():
                st["vsrc"] = (wave >= 2).select(bs["d"], bs["lse"]) + fx.Int64(
                    st["r0"]
                ) * 4

            def lsed_g1():
                st["g1v"] = _tdm_vec_g1(st["rows"], 16)

            def lsed_desc():
                desc[1] = _tdm_desc(seg_w + LSED_OFF + slot * 64, st["vsrc"], st["g1v"])

            def qdo_addr():
                st["ss_b"] = (wave >= 2).select(do_ss_b, q_ss_b)
                st["src"] = (wave >= 2).select(bs["do"], bs["q"]) + fx.Int64(
                    st["r0"]
                ) * fx.Int64(st["ss_b"])

            def qdo_g1():
                st["g1q"] = _tdm_rows_g1(256, st["rows"], 16, st["ss_b"], KV_PAD)

            def qdo_desc():
                desc[0] = _tdm_desc(seg_w + slot * HALF_B, st["src"], st["g1q"])

            return [rows, lsed_addr, lsed_g1, lsed_desc, qdo_addr, qdo_g1, qdo_desc]

        def _tdm_qdo(m, q_lo, slot, bs):
            """16 full rows into this wave's segment: Q row half w (waves 0/1) or dO row half w - 2 (waves 2/3),
            the same padded 272 B rows two quarter-row loads per wave pair used to build.
            """
            r0 = q_lo + m * SUB_QO + w_half * 16
            rows = seqlen_q - r0
            is_do = wave >= 2
            ss_b = is_do.select(do_ss_b, q_ss_b)
            src = is_do.select(bs["do"], bs["q"]) + fx.Int64(r0) * fx.Int64(ss_b)
            _tdm_rows(seg_w + slot * HALF_B, src, 256, rows, 16, ss_b, KV_PAD)

        def _tdm_lsed(m, q_lo, slot, bs):
            r0 = q_lo + m * SUB_QO + w_half * 16
            rows = seqlen_q - r0
            is_d = wave >= 2
            src = is_d.select(bs["d"], bs["lse"]) + fx.Int64(r0) * 4
            _tdm_vec(seg_w + LSED_OFF + slot * 64, src, rows, 16)

        # Operand loads are built as lists of thunks so a phase can spread them between its WMMAs.
        def qdo_loads(slot):
            """40 ds_load_b128 of the tile in ring slot `slot`: Q / dO A-fragments and LSE / D.
            Run the thunks, then assemble()."""
            vals = {}
            thunks = []
            v4 = fx.Vector.make_type(4, F32)
            for i in range_constexpr(2):
                qa = lb["rm"][i] + slot * HALF_B
                da = lb["rm"][2 + i] + slot * HALF_B
                for c in range_constexpr(4):
                    for h in range_constexpr(2):
                        thunks.append(
                            lambda key=(
                                "q",
                                i,
                                c,
                                h,
                            ), a=qa + 64 * c + 32 * h: vals.__setitem__(key, _lds_v8(a))
                        )
                        thunks.append(
                            lambda key=(
                                "o",
                                i,
                                c,
                                h,
                            ), a=da + 64 * c + 32 * h: vals.__setitem__(key, _lds_v8(a))
                        )
                la = lb["lsed"][i] + slot * 64
                dl = lb["lsed"][2 + i] + slot * 64
                for h in range_constexpr(2):
                    thunks.append(
                        lambda key=("l", i, h), a=la + 16 * h: vals.__setitem__(
                            key, fx.ptr_load(_lptr(a), result_type=v4)
                        )
                    )
                    thunks.append(
                        lambda key=("d", i, h), a=dl + 16 * h: vals.__setitem__(
                            key, fx.ptr_load(_lptr(a), result_type=v4)
                        )
                    )

            def assemble():
                Qf = [
                    [_cat(vals[("q", i, c, 0)], vals[("q", i, c, 1)]) for c in range(4)]
                    for i in range(2)
                ]
                dOf = [
                    [_cat(vals[("o", i, c, 0)], vals[("o", i, c, 1)]) for c in range(4)]
                    for i in range(2)
                ]
                lv = [_cat(vals[("l", i, 0)], vals[("l", i, 1)]) for i in range(2)]
                dv_ = [_cat(vals[("d", i, 0)], vals[("d", i, 1)]) for i in range(2)]
                return Qf, dOf, lv, dv_

            return thunks, assemble

        def tr_loads(slot):
            """32 ds_load_tr16_b128 of the tile in ring slot `slot`: the dO^T / Q^T A-fragments of
            gemm2 / gemm3."""
            vals = {}
            thunks = []
            qa = [lb["trq"][h] + slot * HALF_B for h in range(2)]
            da = [lb["trq"][2 + h] + slot * HALF_B for h in range(2)]
            for e in range_constexpr(8):
                for h in range_constexpr(2):
                    thunks.append(
                        lambda key=("o", e, h), a=da[h] + 16 * e: vals.__setitem__(
                            key, _lds_tr_v8(a)
                        )
                    )
                    thunks.append(
                        lambda key=("q", e, h), a=qa[h] + 16 * e: vals.__setitem__(
                            key, _lds_tr_v8(a)
                        )
                    )

            def get(e):
                return _cat(vals[("o", e, 0)], vals[("o", e, 1)]), _cat(
                    vals[("q", e, 0)], vals[("q", e, 1)]
                )

            return thunks, get

        def ds_loads(buf):
            """16 ds_load_tr16_b128 from exchange buffer `buf`: dS A-fragments of gemm4 (q tile i,
            key chunk c = producer wave c)."""
            vals = {}
            thunks = []
            ab = [lb["dstr"][c] + buf * DS_BUF_B for c in range(4)]
            for i in range_constexpr(2):
                for c in range_constexpr(4):
                    a = ab[c] + 16 * i
                    for h in range_constexpr(2):
                        thunks.append(
                            lambda key=(
                                i,
                                c,
                                h,
                            ), a_=a + 16 * DS_ROW_B * h: vals.__setitem__(
                                key, _lds_tr_v8(a_)
                            )
                        )

            def get(i, c):
                return _cat(vals[(i, c, 0)], vals[(i, c, 1)])

            return thunks, get

        def atomic_thunks(dQ, q0, bs):
            """The 32 dQ atomics of a tile as structured-buffer atomics (4 B records, record = lane + 64 wave): the
            element's byte offset is a plain constant voffset, which goes whole into the offset field, so every
            atomic addresses through the one VGPR dq_vidx (an offset VGPR + constant hoisted out of the loop
            would be one VGPR per element)."""
            soff = q0 * (HDIM * 4)
            thunks = []
            for i in range_constexpr(2):
                for e in range_constexpr(2):
                    for n in range_constexpr(8):
                        c = (16 * i * HDIM + 32 * (HDIM // 16) * n + 32 * e) * 4
                        thunks.append(
                            lambda v=dQ[i * 2 + e], n_=n, c_=c: _atomic_fadd_idx(
                                v[n_], bs["dq_rsrc"], lb["vidx"], c_, soff
                            )
                        )
            return thunks

        def softmax_tile(q0, i, j, S, dP, lv, masked, kv0):
            """bf16 P and dS of the 16x16 sub-tile (i, j); masked elements become exp2(-inf) = 0
            (applied after the lse subtraction, so a fully masked row with lse = -inf gives 0).
            """
            x = S[i][j] * scale_log2e + lv[i] * (-LOG2E)
            if const_expr(masked):
                key = kv0 + wave * WV_KV + 16 * j + lane16
                thr = key - (q0 + 16 * i + q_lane)
                xs = []
                for n in range_constexpr(8):
                    if const_expr(CAUSAL):
                        dead = thr > (
                            causal_off + n
                        )  # key past the bottom-right diagonal of row n
                    else:
                        dead = key >= seqlen_k  # zero-filled key of a partial block
                    xs.append(dead.select(F32(float("-inf")), x[n]))
                x = fx.Vector.from_elements(xs, F32)
            p = fx.Vector.from_elements(
                [fx.exp2(x[n], fastmath="afn") for n in range(8)], F32
            )
            return p.to(BF16), (p * dP[i][j]).to(BF16)

        scale2 = fx.Vector.from_elements([scale_log2e] * 2, F32)

        def softmax_chunks(q0, i, j, S, dP, nl, masked, kv0, out):
            """The softmax of sub-tile (i, j) as 12 thunks of 2 VALU ops (at most 1 transcendental), one per WMMA
            co-execution window, each using only results of earlier thunks: out[(i, j)] = (bf16 P, bf16 dS).
            nl[i] = -lse log2(e)."""
            st = {}

            def pair(v, m):
                return v.shuffle(v, [2 * m, 2 * m + 1])

            def fma(m):
                if const_expr(masked and m == 0):
                    st["key"] = kv0 + wave * WV_KV + 16 * j + lane16
                    st["thr"] = st["key"] - (q0 + 16 * i + q_lane)
                st[("x", m)] = fx.fma(pair(S[i][j], m), scale2, pair(nl[i], m))

            def exp(n):
                x = st[("x", n // 2)][n % 2]
                if const_expr(masked):
                    if const_expr(CAUSAL):
                        dead = st["thr"] > (causal_off + n)
                    else:
                        dead = st["key"] >= seqlen_k
                    x = dead.select(F32(float("-inf")), x)
                st[("e", n)] = fx.exp2(x, fastmath="afn")

            def p_of(m):
                return fx.Vector.from_elements(
                    [st[("e", 2 * m)], st[("e", 2 * m + 1)]], F32
                )

            def mul(m):
                st[("ds", m)] = p_of(m) * pair(dP[i][j], m)

            def cvt_p(m):
                st[("P", m)] = p_of(m).to(BF16)

            def cvt_d(m):
                st[("D", m)] = st[("ds", m)].to(BF16)

            def done():
                cat4 = lambda k: _cat(
                    _cat(st[(k, 0)], st[(k, 1)]), _cat(st[(k, 2)], st[(k, 3)])
                )
                out[(i, j)] = (cat4("P"), cat4("D"))

            plan = [
                [lambda: fma(0), lambda: fma(1)],
                [lambda: fma(2), lambda: exp(0)],
                [lambda: fma(3), lambda: exp(1)],
                [lambda: exp(2), lambda: mul(0)],
                [lambda: exp(3), lambda: cvt_p(0)],
                [lambda: exp(4), lambda: mul(1)],
                [lambda: exp(5), lambda: cvt_p(1)],
                [lambda: exp(6), lambda: cvt_d(0)],
                [lambda: exp(7), lambda: mul(2)],
                [lambda: cvt_p(2), lambda: cvt_d(1)],
                [lambda: mul(3), lambda: cvt_d(2)],
                [lambda: cvt_p(3), lambda: cvt_d(3), done],
            ]
            return [lambda ops=ops: [f() for f in ops] for ops in plan]

        def pack_pds(P16, dS16):
            return [_cat(P16[0][j], P16[1][j]) for j in range(2)], [
                _cat(dS16[0][j], dS16[1][j]) for j in range(2)
            ]

        def gemm01(
            Qf, dOf, dv_, Kf, Vf, extra=None, keep=False, post=None, mid=None, acc=None
        ):
            """S and dP - D in (i, j, c) order; extra[k] = memory thunks of WMMA pair k, the first half issued
            behind its S WMMA, the rest behind its dP WMMA, so both 8-cycle windows of the pair carry memory
            work; mid[k] / post[k] = VALU work co-executing with the S / dP WMMA of pair k. acc = (S, dP)
            lists updated in place."""
            if const_expr(acc is None):
                S = [[zero_acc, zero_acc], [zero_acc, zero_acc]]
                dP = [[dv_[0], dv_[0]], [dv_[1], dv_[1]]]
            else:
                S, dP = acc
            pend = {}
            for kk in range_constexpr(16):
                if (kk // 4) % 2 == 1:  # last use of Qf / dOf[kk // 8][kk % 4]
                    pend.setdefault(min(kk + 1, 16), []).extend(
                        [Qf[kk // 8][kk % 4], dOf[kk // 8][kk % 4]]
                    )
            k_ = 0
            for i in range_constexpr(2):
                for j in range_constexpr(2):
                    for c in range_constexpr(4):
                        # WMMA first, then the memory ops: they issue in its execute cycles (no VALU
                        # co-issue there) instead of holding back the next WMMA
                        ex = extra[k_] if const_expr(extra is not None) else []
                        ex_s, ex_p = ex[: (len(ex) + 1) // 2], ex[(len(ex) + 1) // 2 :]
                        _fence()
                        S[i][j] = _wmma(Qf[i][c], Kf[j][c], S[i][j])
                        if const_expr(len(ex_s) > 0):
                            _fence()
                            for x in ex_s:
                                x()
                            _fence()
                        if const_expr(mid is not None and k_ in mid):
                            _fence()
                            mid[k_]()
                            _fence()
                        if c == 0:
                            dP[i][j] = _wmma_negc(dOf[i][c], Vf[j][c], dP[i][j])
                        else:
                            dP[i][j] = _wmma(dOf[i][c], Vf[j][c], dP[i][j])
                        if const_expr(len(ex_p) > 0):
                            _fence()
                            for x in ex_p:
                                x()
                        if const_expr(keep and k_ in pend):
                            _keepalive(pend[k_])
                        if const_expr(len(ex_p) > 0):
                            _fence()
                        if const_expr(post is not None and k_ in post):
                            _fence()
                            post[k_]()
                        k_ += 1
            _fence()
            if keep and 16 in pend:
                _keepalive(pend[16])
            return S, dP

        def store_ds(buf, dSf):
            """This wave's dS^T ([32 keys][32 q], bf16) into exchange buffer `buf`."""
            a = lb["dsst"] + buf * DS_BUF_B
            for i in range_constexpr(2):
                for j in range_constexpr(2):
                    half = dSf[j].shuffle(dSf[j], list(range(8 * i, 8 * i + 8)))
                    fx.ptr_store(half, _lptr(a + 16 * j * DS_ROW_B + 16 * i))

        N_STATE = 60

        def unpack(state):
            st = [fx.Vector(x) for x in state[:N_STATE]]
            return {
                "dKt": st[0:16],
                "dVt": st[16:32],
                "Pf": st[32:34],
                "dSf": st[34:36],
                "dQ": st[36:40],
                "Qf": [st[40:44], st[44:48]],
                "dOf": [st[48:52], st[52:56]],
                "lv": st[56:58],
                "dv": st[58:60],
            }

        def pack(dKt, dVt, Pf, dSf, dQ, Qf, dOf, lv, dv_):
            vals = (
                list(dKt)
                + list(dVt)
                + list(Pf)
                + list(dSf)
                + list(dQ)
                + Qf[0]
                + Qf[1]
                + dOf[0]
                + dOf[1]
                + list(lv)
                + list(dv_)
            )
            return [_ir(x) for x in vals]

        # Phase C spreads the 40 Q / dO / LSE / D loads over its 16 WMMA steps and stores dS(t+1)
        # after the loads of step STORE_SLOT. The barrier only needs that store done, not the loads behind it
        # (they feed the next phase A), and ds ops complete in order.
        STORE_SLOT = 3  # three WMMAs after the last dS conversion: its RAW wait does not drain the XDL pipe
        A_SUBS = (
            2  # softmax sub-tiles co-executing with phase A; the rest go to phase B
        )
        QDO_IN_B = 8  # Q / dO loads of t+2 issued in the atomic windows of phase B (slots 16-31), the rest in C
        DS_AFTER_STORE = sum(
            len(g) for g in _spread(list(range(40 - QDO_IN_B)), 16)[STORE_SLOT + 1 :]
        )

        def body(t, state, masked, kv0, q_lo, Kf, Vf, Ktf, rot, rot2, k=None, bs=None):
            """Iteration t. Tile m sits in ring slot (m + rot) % PF and its dS in exchange buffer
            (m + rot2) % 2; the unrolled loop passes k = (t + rot) % PF as a trace-time int.
            """
            s_ = unpack(state)
            static = k is not None

            def sl(d):
                return (k + d) % PF if static else (t + d + rot) % PF

            def bf(d):
                return (k + d) % 2 if static else (t + d + rot2) % 2

            # dS(t) is already in LDS (stored in phase C of t-1 or by the prologue). Signal now and
            # wait before phase B: phase A only touches registers, tile t (published by the
            # previous barrier) and global memory.
            rocdl.s_wait_tensorcnt(TDM_PER_TILE * (PF - 4))
            _fence()
            rocdl.s_wait_dscnt(DS_AFTER_STORE)
            rocdl.s_barrier_signal(-1)
            _fence()

            # A: gemm0/1 of t+1 | atomics of t-1, Q^T / dO^T of t
            q_prev = q_lo + fx.max(t - 1, fx.Int32(0)) * SUB_QO
            at = atomic_thunks(s_["dQ"], q_prev, bs)
            at_a, at_b, at_c = (
                at[: ADIST[0]],
                at[ADIST[0] : ADIST[0] + ADIST[1]],
                at[ADIST[0] + ADIST[1] :],
            )
            tr_th, tr_get = tr_loads(sl(0))
            tr_s = _spread(tr_th, 12) + [[]] * 4  # done 4 WMMA pairs before phase B
            at_s = _spread(at_a, 16)
            nl = [None, None]

            def mk_nl(i):
                def f():
                    nl[i] = s_["lv"][i] * (-LOG2E)

                return f

            # softmax of t+1 in 12 slices per sub-tile: sub-tile s is complete after WMMA pair 4s+3, so
            # sub-tiles 0 .. A_SUBS-1 run in the following pairs of phase A, the rest in phase B
            q1 = q_lo + (t + 1) * SUB_QO
            S = [[zero_acc, zero_acc], [zero_acc, zero_acc]]
            dP = [[s_["dv"][0], s_["dv"][0]], [s_["dv"][1], s_["dv"][1]]]
            sm_out = {}
            sub_ch = [
                softmax_chunks(q1, s // 2, s % 2, S, dP, nl, masked, kv0, sm_out)
                for s in range(4)
            ]
            a_ch = [
                c for s in range(A_SUBS) for c in sub_ch[s]
            ]  # sub-tile s starts at pair 6s+4 >= 4s+4
            mid = {4 + p: a_ch[2 * p] for p in range(12)}
            post = {4 + p: a_ch[2 * p + 1] for p in range(12)}
            post[2] = mk_nl(0)
            post[3] = mk_nl(1)
            # the SALU setup of the two TDMs of phase B goes into windows of phase A that hold a single memory op
            # (pairs 1-3) or one memory op and one VALU op (pairs 12-13); phase B then issues just the tensor loads
            tdm_desc = [None, None]
            prep = tdm_prep_thunks(t + (PF - 1), q_lo, sl(PF - 1), bs, tdm_desc)
            for x, (kk, d) in enumerate(
                [
                    (1, mid),
                    (1, post),
                    (2, mid),
                    (3, mid),
                    (12, mid),
                    (12, post),
                    (13, mid),
                ]
            ):
                d[kk] = (lambda a=prep[x], b=d[kk]: (a(), b())) if kk in d else prep[x]
            gemm01(
                s_["Qf"],
                s_["dOf"],
                s_["dv"],
                Kf,
                Vf,
                extra=[tr_s[x] + at_s[x] for x in range(16)],
                keep=True,
                post=post,
                mid=mid,
                acc=(S, dP),
            )
            _fence()
            rocdl.s_barrier_wait(-1)
            _fence()
            # one wait for all Q^T / dO^T fragments (phase A issues no other ds op) instead of one in
            # front of each WMMA reading one: s_wait_* never co-issues
            rocdl.s_wait_dscnt(0)

            # B: gemm2/3 of t, WMMA by WMMA, the remaining softmax slices of t+1 spread over them (a bf16
            # WMMA leaves 4 VALU issue cycles, 2 of them for transcendentals). Ahead of each WMMA: the TDM
            # of tile t+PF-1 (slots 1-3), the dS^T loads of phase C (slots 0-15), the atomics (slots 16-31).
            ds_th, ds_get = ds_loads(bf(0))
            dKt = list(s_["dKt"])
            dVt = list(s_["dVt"])
            rest = [c for s in range(A_SUBS, 4) for c in sub_ch[s]]
            chunks = [None] * 32
            for x in range_constexpr(len(rest)):
                chunks[x * 32 // len(rest)] = rest[x]
            # tile t+2 is in LDS since the barrier above (s_wait_tensorcnt at the top of t + barrier)
            ld_th, ld_asm = qdo_loads(sl(2))
            mem = [[x] for x in ds_th] + [
                g + [ld_th[x]] for x, g in enumerate(_spread(at_b, 16))
            ]
            for x in range_constexpr(TDM_PER_TILE):
                mem[1 + x] = mem[1 + x] + [lambda x_=x: _tdm_issue(tdm_desc[x_])]
            k_ = 0
            for e in range_constexpr(8):
                dOt, Qt = tr_get(e)
                for j in range_constexpr(2):
                    for g in range_constexpr(2):
                        _fence()
                        if g == 0:
                            dVt[e * 2 + j] = _wmma(dOt, s_["Pf"][j], dVt[e * 2 + j])
                        else:
                            dKt[e * 2 + j] = _wmma(Qt, s_["dSf"][j], dKt[e * 2 + j])
                        if const_expr(len(mem[k_]) > 0):
                            _fence()
                            for x in mem[k_]:
                                x()
                        if const_expr(chunks[k_] is not None):
                            _fence()
                            chunks[k_]()
                        k_ += 1
            _fence()
            nPf, ndSf = pack_pds(
                [[sm_out[(i, j)][0] for j in range(2)] for i in range(2)],
                [[sm_out[(i, j)][1] for j in range(2)] for i in range(2)],
            )
            # every dS^T fragment was issued in the first half of phase B, before the QDO_IN_B Q / dO loads (ds
            # ops complete in order): one wait instead of one per use
            rocdl.s_wait_dscnt(QDO_IN_B)

            # C: gemm4 of t | the rest of Q / dO / LSE / D of t+2, dS(t+1) store
            ld_s = _spread(ld_th[QDO_IN_B:], 16)
            assert sum(len(g) for g in ld_s[STORE_SLOT + 1 :]) == DS_AFTER_STORE
            atc_s = _spread(at_c, 16)
            ld_s[STORE_SLOT] = ld_s[STORE_SLOT] + [lambda: store_ds(bf(1), ndSf)]
            dQ = [None] * 4
            k_ = 0
            for i in range_constexpr(2):
                for e in range_constexpr(2):
                    acc = zero_acc
                    for c in range_constexpr(4):
                        _fence()
                        acc = _wmma(ds_get(i, c), Ktf[e][c], acc)
                        ex = ld_s[k_] + atc_s[k_]
                        if const_expr(len(ex) > 0):
                            _fence()
                            for x in ex:
                                x()
                        k_ += 1
                    dQ[i * 2 + e] = acc
            _fence()
            _keepalive(list(s_["dQ"]))
            nQf, ndOf, nlv, ndv = ld_asm()
            return pack(dKt, dVt, nPf, ndSf, dQ, nQf, ndOf, nlv, ndv)

        # Persistent workgroups: one per CU, walking work items wg, wg + G, ... (G = the grid, a multiple
        # of 8 so a workgroup's items stay on its XCD under the swizzle). Work item -> (KV block column,
        # head, batch) through the XCD swizzle of the asm (SWIZZLE_XCD): every whole run of 8 SWZ_CHUNK
        # items is dealt out as SWZ_CHUNK consecutive ones (KV block fastest, then head, then batch) per
        # XCD, so the KV blocks of a head share one XCD's L2 for their Q / dO reads and dQ atomics.
        gxi = (n_kb + 1) // 2 if const_expr(CAUSAL) else n_kb
        n_items = gxi * nhead * batch
        n_my = (n_items - wg + n_wgs - 1) // n_wgs

        def item(kq):
            L = wg + kq * n_wgs
            swz = (
                ((L & 7) << SWZ_CHUNK_LOG2)
                | ((L >> 3) & (SWZ_CHUNK - 1))
                | (L & -(8 * SWZ_CHUNK))
            )
            w = (L < (n_items & -(8 * SWZ_CHUNK))).select(swz, L)
            return w % gxi, (w // gxi) % nhead, w // gxi // nhead

        def bases(hq, b):
            hk = hq // ratio
            acc = _base_i64(dq_acc) + (
                (fx.Int64(b) * fx.Int64(nhead) + fx.Int64(hq)) * fx.Int64(sq_pad)
            ) * (HDIM * 4)
            return {
                "q": _base_i64(q)
                + (fx.Int64(b) * fx.Int64(q_sb) + fx.Int64(hq) * fx.Int64(q_sh)) * 2,
                "do": _base_i64(do)
                + (fx.Int64(b) * fx.Int64(do_sb) + fx.Int64(hq) * fx.Int64(do_sh)) * 2,
                "k": _base_i64(k)
                + (fx.Int64(b) * fx.Int64(k_sb) + fx.Int64(hk) * fx.Int64(k_sh)) * 2,
                "v": _base_i64(v)
                + (fx.Int64(b) * fx.Int64(v_sb) + fx.Int64(hk) * fx.Int64(v_sh)) * 2,
                "dk": _base_i64(dk)
                + (fx.Int64(b) * fx.Int64(dk_sb) + fx.Int64(hq) * fx.Int64(dk_sh)) * 2,
                "dv": _base_i64(dv)
                + (fx.Int64(b) * fx.Int64(dv_sb) + fx.Int64(hq) * fx.Int64(dv_sh)) * 2,
                "lse": _base_i64(lse)
                + (fx.Int64(b) * fx.Int64(lse_sb) + fx.Int64(hq) * fx.Int64(lse_sh))
                * 4,
                "d": _base_i64(dd)
                + (fx.Int64(b) * fx.Int64(d_sb) + fx.Int64(hq) * fx.Int64(d_sh)) * 4,
                "dq_rsrc": _rsrc_records4(acc),
            }

        def n_blocks(bx):
            if const_expr(CAUSAL):
                return (n_kb - 1 - bx > bx).select(fx.Int32(2), fx.Int32(1))
            return fx.Int32(1)

        def block_of(bx, r):
            """KV block r of work item column bx (a causal item folds blocks bx and n_kb - 1 - bx)."""
            kb = (
                (r == fx.Int32(0)).select(bx, n_kb - 1 - bx)
                if const_expr(CAUSAL)
                else bx
            )
            kv0 = kb * SUB_KV
            if const_expr(CAUSAL):
                q_lo0 = fx.max(kv0 - causal_off, fx.Int32(0))
                q_lo = (
                    q_lo0 - q_lo0 % SUB_QO
                )  # first tile on the dQ_acc grid that sees this block
                thr_q = (
                    kv0 + SUB_KV - 1 - causal_off
                )  # first q0 that sees the whole block
                n_tiles = (seqlen_q - q_lo + SUB_QO - 1) // SUB_QO
                n_mask = fx.min(
                    fx.max((thr_q - q_lo + SUB_QO - 1) // SUB_QO, fx.Int32(0)), n_tiles
                )
            else:
                q_lo = fx.Int32(0)
                n_tiles = (seqlen_q + SUB_QO - 1) // SUB_QO
                n_mask = (kv0 + SUB_KV > seqlen_k).select(n_tiles, fx.Int32(0))
            # tiles whose next tile still crosses the diagonal run the masked body; the ring is
            # rotated so that the unmasked tiles from n_a on start at slot 0 and buffer 0
            n_a = fx.max(n_mask - 1, fx.Int32(0))
            return {
                "kv0": kv0,
                "q_lo": q_lo,
                "n_tiles": n_tiles,
                "n_a": n_a,
                "rot": (PF - n_a % PF) % PF,
                "rot2": n_a % 2,
            }

        def issue_block(bk, bs):
            """TDM the K / V of this wave's 32 keys of block bk, then its Q tiles 0 .. PF-3: ten tensor ops, the
            most that issue back to back (LLVM keeps TENSORcnt <= 10 in front of each further one). Tile PF-2 goes
            out in the block's prologue, once K / V have landed."""
            kr0 = bk["kv0"] + wave * WV_KV
            krows = seqlen_k - kr0
            _tdm_rows(
                seg_w + KV_OFF,
                bs["k"] + fx.Int64(kr0) * fx.Int64(k_ss * 2),
                256,
                krows,
                WV_KV,
                k_ss * 2,
                KV_PAD,
            )
            _tdm_rows(
                seg_w + KV_OFF + V_SUB,
                bs["v"] + fx.Int64(kr0) * fx.Int64(v_ss * 2),
                256,
                krows,
                WV_KV,
                v_ss * 2,
                KV_PAD,
            )
            for m in range_constexpr(PF - 2):
                issue_tile(fx.Int32(m), bk["q_lo"], (bk["rot"] + m) % PF, bs)

        bx0, hq0, b0 = item(fx.Int32(0))
        issue_block(block_of(bx0, fx.Int32(0)), bases(hq0, b0))
        for kq in range(fx.Int32(0), n_my, fx.Int32(1)):
            bx, hq, b = item(fx.Int32(kq))
            bs = bases(hq, b)
            nb = n_blocks(bx)
            for r in range(fx.Int32(0), nb, fx.Int32(1)):
                bk = block_of(bx, fx.Int32(r))
                kv0, q_lo, n_tiles, n_a, rot, rot2 = (
                    bk[x] for x in ("kv0", "q_lo", "n_tiles", "n_a", "rot", "rot2")
                )
                # K / V and Q tiles 0 .. PF-3 of this block were issued last, then (but for the first block) the last
                # block's two dK / dV stores: K / V done. A plain barrier, not gpu.barrier(): its fence would also wait
                # for the last block's dK / dV stores
                rocdl.s_wait_tensorcnt(TDM_PER_TILE * (PF - 2))
                rocdl.s_barrier_signal(-1)
                rocdl.s_barrier_wait(-1)
                kv_own = seg_w + KV_OFF
                Kf = [[None] * 4, [None] * 4]
                Vf = [[None] * 4, [None] * 4]
                Ktf = [[None] * 4, [None] * 4]
                for j in range_constexpr(2):
                    for c in range_constexpr(4):
                        ka = kv_own + rm_off + 16 * j * ROW_B + 64 * c
                        Kf[j][c] = _cat(_lds_v8(ka), _lds_v8(ka + 32))
                        Vf[j][c] = _cat(_lds_v8(ka + V_SUB), _lds_v8(ka + V_SUB + 32))
                for e in range_constexpr(2):
                    for c in range_constexpr(4):
                        ta = seg_base[c] + KV_OFF + tr_off + (wave * WV_KV + 16 * e) * 2
                        Ktf[e][c] = _cat(_lds_tr_v8(ta), _lds_tr_v8(ta + 16 * ROW_B))
                issue_tile(fx.Int32(PF - 2), q_lo, (rot + PF - 2) % PF, bs)
                # tiles 0 and 1 done: behind them at most tiles 2, 3 (and the dK / dV stores), and tile PF-2
                rocdl.s_wait_tensorcnt(TDM_PER_TILE * (PF - 3))
                _fence()
                rocdl.s_wait_dscnt(0)
                rocdl.s_barrier_signal(-1)
                rocdl.s_barrier_wait(-1)
                _fence()

                # prologue: first half of tile 0, operands of tile 1
                th0, asm0 = qdo_loads(rot)
                for x in th0:
                    x()
                Qf0, dOf0, lv0, dv0 = asm0()
                S0, dP0 = gemm01(Qf0, dOf0, dv0, Kf, Vf)
                P16 = [[None, None], [None, None]]
                dS16 = [[None, None], [None, None]]
                for i in range_constexpr(2):
                    for j in range_constexpr(2):
                        P16[i][j], dS16[i][j] = softmax_tile(
                            q_lo, i, j, S0, dP0, lv0, True, kv0
                        )
                Pf0, dSf0 = pack_pds(P16, dS16)
                store_ds(rot2, dSf0)
                th1, asm1 = qdo_loads((rot + 1) % PF)
                for x in th1:
                    x()
                Qf1, dOf1, lv1, dv1 = asm1()
                init = pack(
                    [zero_acc] * 16,
                    [zero_acc] * 16,
                    Pf0,
                    dSf0,
                    [zero_acc] * 4,
                    Qf1,
                    dOf1,
                    lv1,
                    dv1,
                )

                ring = (rot, rot2)
                for t, state in range(fx.Int32(0), n_a, fx.Int32(1), init=init):
                    res_a = yield body(
                        fx.Int32(t), state, True, kv0, q_lo, Kf, Vf, Ktf, *ring, bs=bs
                    )
                # unmasked tiles PF per trip, so ring slots and dS buffers are trace-time constants
                n_u = (n_tiles - n_a) // PF
                # an i32 trip counter (range() would count in index = i64): the exit test is then an s_cmp on SCC,
                # not a 64-bit v_cmp into an SGPR, which can land on the soffset of the last, still untranslated
                # dQ atomics and cost an s_wait_xcnt
                loop_u = scf.ForOp(
                    _ir(fx.Int32(0)), _ir(n_u), _ir(fx.Int32(1)), list(res_a)
                )
                with ir.InsertionPoint(loop_u.body):
                    t0 = n_a + fx.Int32(loop_u.induction_variable) * PF
                    s_u = list(loop_u.inner_iter_args)
                    for kk in range_constexpr(PF):
                        s_u = body(
                            t0 + kk,
                            s_u,
                            False,
                            kv0,
                            q_lo,
                            Kf,
                            Vf,
                            Ktf,
                            *ring,
                            k=kk,
                            bs=bs,
                        )
                    scf.YieldOp(s_u)
                res_u = list(loop_u.results)
                for t2, state2 in range(
                    n_a + n_u * PF, n_tiles, fx.Int32(1), init=res_u
                ):
                    res_b = yield body(
                        fx.Int32(t2),
                        state2,
                        False,
                        kv0,
                        q_lo,
                        Kf,
                        Vf,
                        Ktf,
                        *ring,
                        bs=bs,
                    )

                fin = unpack(res_b)
                # every wave is done with the ring, the dS exchange and K / V: the next block's loads go
                # out now and land while this block's last dQ atomics and dK / dV stores drain
                rocdl.s_wait_dscnt(0)
                rocdl.s_barrier_signal(-1)
                rocdl.s_barrier_wait(-1)
                # the ring loads past this block's last tile write zeros into slots the next block's first
                # tiles reuse; tensor ops complete in order but their data may land out of order
                rocdl.s_wait_tensorcnt(0)
                more = fx.Int32(r) + 1 < nb
                nxt = more.select(
                    fx.Int32(1),
                    (fx.Int32(kq) + 1 < n_my).select(fx.Int32(1), fx.Int32(0)),
                )
                if nxt > fx.Int32(0):
                    nbx, nhq, nbb = item(more.select(fx.Int32(kq), fx.Int32(kq) + 1))
                    issue_block(
                        block_of(nbx, more.select(fx.Int32(r) + 1, fx.Int32(0))),
                        bases(nhq, nbb),
                    )
                for x in atomic_thunks(fin["dQ"], q_lo + (n_tiles - 1) * SUB_QO, bs):
                    x()
                # dK / dV leave through the LDS: bf16 rows into this wave's staging area, then two tensor
                # stores (rows past seqlen_k dropped by the TDM bound). Nothing waits on the global writes, and
                # they go out behind the next block's loads, so the waits for those do not cover them. The
                # last block's stores are done reading the staging area: every tile waits for all but the
                # newest 6 tensor ops. Lane (key 16 j + lane16, d = 64 lane_hi + 8 e) of tile (e, j).
                kts = [(fin["dKt"][x] * scale).to(BF16) for x in range(16)]
                vts = [fin["dVt"][x].to(BF16) for x in range(16)]
                stg = lds + STG_OFF + wave * STG_W
                _fence()
                for e in range_constexpr(8):
                    for j in range_constexpr(2):
                        a = stg + (16 * j + lane16) * ROW_B + (64 * lane_hi + 8 * e) * 2
                        fx.ptr_store(kts[e * 2 + j], _lptr(a))
                        fx.ptr_store(vts[e * 2 + j], _lptr(a + WV_KV * ROW_B))
                rocdl.s_wait_dscnt(0)
                kr0 = kv0 + wave * WV_KV
                for x, base, ss in ((0, bs["dk"], dk_ss), (1, bs["dv"], dv_ss)):
                    _tdm_store_rows(
                        stg + x * WV_KV * ROW_B,
                        base + fx.Int64(kr0) * fx.Int64(ss * 2),
                        256,
                        ROW_B,
                        seqlen_k - kr0,
                        WV_KV,
                        ss * 2,
                    )
        rocdl.s_wait_tensorcnt(
            0
        )  # the ring loads past the last tile still write the LDS

    @flyc.jit
    def launch_dqdkdv(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        do: fx.Tensor,
        lse: fx.Tensor,
        dd: fx.Tensor,
        dq_acc: fx.Tensor,
        dk: fx.Tensor,
        dv: fx.Tensor,
        q_sb: fx.Int32,
        q_ss: fx.Int32,
        q_sh: fx.Int32,
        k_sb: fx.Int32,
        k_ss: fx.Int32,
        k_sh: fx.Int32,
        v_sb: fx.Int32,
        v_ss: fx.Int32,
        v_sh: fx.Int32,
        do_sb: fx.Int32,
        do_ss: fx.Int32,
        do_sh: fx.Int32,
        dk_sb: fx.Int32,
        dk_ss: fx.Int32,
        dk_sh: fx.Int32,
        dv_sb: fx.Int32,
        dv_ss: fx.Int32,
        dv_sh: fx.Int32,
        lse_sb: fx.Int32,
        lse_sh: fx.Int32,
        d_sb: fx.Int32,
        d_sh: fx.Int32,
        seqlen_q: fx.Int32,
        seqlen_k: fx.Int32,
        sq_pad: fx.Int32,
        nhead: fx.Int32,
        ratio: fx.Int32,
        scale_log2e: fx.Float32,
        scale: fx.Float32,
        seg: fx.Int32,
        batch: fx.Int32,
        stream: fx.Stream,
    ):
        n_kb = (seqlen_k + SUB_KV - 1) // SUB_KV
        if const_expr(CAUSAL):
            gx = (n_kb + 1) // 2
        else:
            gx = n_kb
        k_fmha_bwd_dqdkdv_hd128(
            q, k, v, do, lse, dd, dq_acc, dk, dv,
            q_sb, q_ss, q_sh, k_sb, k_ss, k_sh, v_sb, v_ss, v_sh, do_sb, do_ss, do_sh,
            dk_sb, dk_ss, dk_sh, dv_sb, dv_ss, dv_sh, lse_sb, lse_sh, d_sb, d_sh,
            seqlen_q, seqlen_k, sq_pad, nhead, ratio, scale_log2e, scale, seg, batch,
        ).launch(grid=(fx.min(gx * nhead * batch, fx.Int32(N_CU)), 1, 1), block=(BLOCK, 1, 1), stream=stream)  # fmt: skip

    # 1 wave per SIMD: the kernel keeps ~850 VGPRs live, like the asm
    launch_dqdkdv.compile_hints["waves_per_eu"] = 1
    launch_dqdkdv.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": True
    }
    if _llvm_has_option(ANTI_HINTS):
        launch_dqdkdv.compile_hints["llvm_options"][ANTI_HINTS] = True
    return launch_dqdkdv
