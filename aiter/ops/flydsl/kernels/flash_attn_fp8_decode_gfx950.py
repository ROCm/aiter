# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 decode-specialized fp8 flash attention.

A single-pass, single-wave-per-(batch, kv-head) decode kernel. The GQA group is
the MMA M-dim: BLOCK_M=16 (16 q-heads ride in M for one kv-head), so one wave
computes one 16x16 tile with zero wasted rows -- the opposite of the prefill
body's BLOCK_M=256 packing. head_dim=128, fp8 e4m3 QKV, paged KV (page/block
size 64), query_len=1, causal (decode => attend all kv < kv_len), varlen, and
bf16/fp16 output.

This is a NEW body, not a trait flip of ``flash_attn_fp8_gfx950.py``: the 16-row
wave mapping and the register-side QK->PV repack cannot be expressed as
const_expr branches of the prefill kernel. It is deliberately decode-local and
does NOT call the welded dualwave loaders (which are hardwired to
num_waves=8 / rows_per_wave=32); it carries its own small decode traits.

Proven cores, measured empirically:

- QK = mma(A=K, B=Q) -> S^T[n_kv, m_q]. C-output packing: value v in {0..3} of
  lane = C[m=(lane//16)*4+v, n=lane%16]; here M=n_kv, n=m_q. Softmax reduces
  over n_kv (the M-axis): within-lane over the 4 v-slots x the two KV-tile
  fragments, then cross-lane-group XOR at shifts {16, 32} (keeping m_q=lane%16
  distinct).
- QK C-output -> PV B-operand handoff is a register-side ds_bpermute gather (the
  4-per-group vs 8-per-group packing mismatch), not a free relabel and not an
  LDS round-trip. See ``repack_p``.
- PV = mma(A=V^T, B=P) -> O^T[d, m_q]. C-output m=d, n=m_q=lane%16.

V mechanism: bespoke single-wave LDS transpose-scatter. V is
stored [n_kv, d] (d contiguous); we scatter it into LDS as [d, n_kv] (n_kv
contiguous) so a plain 8-fp8 register read lands n_kv on the MMA K axis. This is
NOT ds_read_b64_tr_b8 (linear-welded) and NOT the welded coalesced V loaders.

Cache layout: ``kv_cache_layout`` selects the linear 4D paged pool ("linear")
or the 5D shuffled vLLM-v1 layout ("vectorized"). The K and V *LDS* tile
layouts -- and the whole compute core -- are cache-layout-independent; ONLY the
global->LDS load addressing differs, which makes the shuffled path a purely
additive change over the linear one. Perf: the shuffled V is pre-transposed and
read straight from global (no LDS V tile), while the linear path
transpose-scatters V into LDS per page (slower). Shuffled is the
production/performance layout; linear is the correctness/reference layout --
vectorizing its V transpose is deferred unless production is confirmed to serve
a linear KV cache at scale.
"""

import math as _math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.flash_attn_dualwave_common import (
    _make_page_view,
    waitcnt_vm_n,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, _to_raw, ptr_arg

FP8 = fx.Float8E4M3FN
WARP = 64
PAGE = 64  # block/page size, structural (matches the 5D shuffled cache)
HEAD_DIM = 128
BLOCK_M = 16  # one GQA group = MMA M-dim
VEC = 16  # 5D-shuffled vectorization width
_LOG2E = _math.log2(_math.e)
GFX950_CUS = 256  # MI355X compute units


def plan_num_kv_splits(batch, num_kv_heads, npages, target_wg=GFX950_CUS, s_max=8):
    """Fill-aware host plan for the inter-workgroup KV split S.

    The base grid is ``batch * num_kv_heads`` workgroups; at low batch this
    under-fills the 256 CUs. Split so the total grid reaches ``target_wg``
    resident workgroups. ``target_wg=256`` (1 WG/CU) is the measured optimum:
    although freeing V's LDS opened 2 WG/CU, over-splitting past one-WG/CU fill
    regresses (the per-split combine overhead and smaller tiles outweigh the
    extra occupancy). This gives S = ceil(256/(b*H_kv)), i.e. b=8->8, b=16->4,
    b=32->2, b=64->1 -- matching the per-cell S sweep and never regressing b=64
    (base=256 -> S=1). S is capped at ``s_max`` and at ``npages`` (>= 1 page per
    split; the -1e30 mask makes an over-split correct anyway, just wasteful).
    """
    base = max(1, batch * num_kv_heads)
    s = (target_wg + base - 1) // base  # ceil(target_wg / base)
    s = max(1, min(s, s_max, max(1, int(npages))))
    return s


def build_flash_attn_fp8_decode_module(
    num_heads,
    head_dim,
    num_kv_heads,
    causal=True,
    dtype_str="fp8",
    out_dtype_str="bf16",
    varlen=False,
    paged=True,
    kv_cache_layout="linear",
    num_waves=8,  # tuned for full-machine b=64 (8 waves/CU, matches the asm ref)
    num_kv_splits=1,  # inter-workgroup KV split S; host-computed per shape
):
    """Build the gfx950 decode-specialized fp8 flash-attention launcher.

    Returns a Python callable ``mod(q, k, v, o, batch, ...)`` that wraps torch
    tensors and launches the kernel. This kernel requires ``paged=True`` and
    ``head_dim==128``; ``num_heads // num_kv_heads`` must be ``BLOCK_M`` (16).
    ``kv_cache_layout`` is ``"linear"`` (4D pool) or ``"vectorized"`` (5D
    shuffled). Output ``out_dtype_str`` is ``"bf16"`` or ``"f16"``.
    """
    if head_dim != HEAD_DIM:
        raise ValueError(f"decode kernel is D=128 only, got {head_dim}")
    if dtype_str != "fp8":
        raise ValueError(f"decode kernel builds fp8 QKV only, got {dtype_str}")
    if not paged:
        raise ValueError("decode kernel is paged-only")
    if out_dtype_str not in ("bf16", "f16"):
        raise ValueError(f"decode output supports bf16/f16 only, got {out_dtype_str}")
    if kv_cache_layout not in ("linear", "vectorized"):
        raise ValueError(f"unknown kv_cache_layout {kv_cache_layout}")

    H = num_heads
    HKV = num_kv_heads
    D = HEAD_DIM
    GQA = H // HKV
    if GQA != BLOCK_M:
        raise ValueError(f"decode kernel assumes GQA==BLOCK_M==16, got {H}/{HKV}={GQA}")

    OUT = fx.BFloat16 if out_dtype_str == "bf16" else fx.Float16
    SHUF = kv_cache_layout == "vectorized"
    VARLEN = bool(varlen)
    # Multi-wave cooperative workgroup (occupancy fix): NW waves per
    # (batch, kv-head) share the 16 query-heads and split the KV page loop
    # (wave w streams pages w, w+NW, ...). Each produces a partial flash-decoding
    # softmax (m_w, l_w, O_w); the NW partials are LSE-merged on-chip via LDS.
    # This lifts waves/CU from 1 (single-wave) toward the CU's capacity -- the
    # single-wave grid (b*HKV workgroups) was the throughput ceiling. One KV
    # buffer PER WAVE (no per-wave double-buffer: inter-wave parallelism hides
    # the DMA latency instead).
    NW = int(num_waves)
    BUFSZ = PAGE * HEAD_DIM  # one K (or V) tile in bytes/elements, per wave
    OPART = D * BLOCK_M  # one wave's O^T partial (128*16 f32)
    # Inter-workgroup KV split. S=1 => direct O store, no workspace. S>1 =>
    # each (batch, kv-head, split) workgroup writes an UNNORMALIZED partial
    # (m, l, O) for its contiguous KV sub-range to a global workspace; a
    # separate combine kernel LSE-merges the S partials.
    SPLK = int(num_kv_splits)
    SPLITK = SPLK > 1
    # per (b,h,s) workspace row: O[D*BLOCK_M] + m[BLOCK_M] + l[BLOCK_M].
    WS_PER = D * BLOCK_M + 2 * BLOCK_M
    # JIT-cache disambiguator. Builds that differ ONLY in out_dtype (bf16 vs
    # f16), cache layout, or wave count produce near-identical IR; without a
    # distinct closure value the flydsl disk cache aliases them and serves the
    # wrong binary. Referenced inside launch so it is part of the cache key.
    # causal is effectively constant for query_len=1 decode; it rides in the
    # tag only for uniformity with the prefill kernel's cache key shape.
    _cache_tag = (
        f"flash_attn_fp8_decode|out={out_dtype_str}|layout={kv_cache_layout}"
        f"|varlen={int(VARLEN)}|causal={int(bool(causal))}|H={H}|HKV={HKV}"
        f"|nw={NW}|splk={SPLK}"
    )
    # rsqrt(d)*log2e folds the softmax scale and the exp->exp2 change of base;
    # q/k per-tensor descale multiply it in-kernel (see c_logit_scale below).
    CONST_SCALE = (1.0 / _math.sqrt(D)) * _LOG2E
    NEG = -1.0e30  # masked-logit sentinel; exp2(NEG - m) underflows to 0

    # Cache-region sizes (elements). Both layouts pack one page-region as
    # HKV*D*PAGE with page stride = HKV*D*PAGE.
    PAGE_REGION = HKV * D * PAGE  # 32768 for H_kv=4
    # Linear 4D pool [n_pages, PAGE, HKV, D]: strides (PAGE*HKV*D, HKV*D, D, 1).
    LIN_T_STRIDE = HKV * D  # 512
    # 5D shuffled: K [nb, HKV, D//16, PAGE, 16]; V [nb, HKV, PAGE//16, D, 16].
    SHUF_HEAD_STRIDE = D * PAGE  # 8192
    SHUF_K_DCHUNK_STRIDE = PAGE * VEC  # 1024
    SHUF_V_TCHUNK_STRIDE = D * VEC  # 2048

    # Single K buffer per wave (no double-buffer: inter-wave parallelism hides
    # the DMA latency instead). Linear keeps a single K buffer + an LDS V tile
    # (its V cannot be read straight from global). Byte-typed (Int8): fp8
    # lowers cleanly through i8 LDS and is bitcast to fp8 only when building
    # the MMA operand.
    if const_expr(SHUF):

        @fx.struct
        class SharedStorage:
            # K [n_kv,d] (d contiguous), one buffer per wave. V is NOT in LDS:
            # the 5D-shuffled V is already in MMA-operand order, so the V^T
            # fragment is 8 contiguous global bytes (a direct b64 load) -- no
            # transpose, no LDS tile. The K region is reused post-loop as the
            # O-partial combine scratch.
            k: fx.Array[fx.Int8, NW * PAGE * D]
            m_part: fx.Array[fx.Float32, NW * BLOCK_M]
            l_part: fx.Array[fx.Float32, NW * BLOCK_M]

    else:

        @fx.struct
        class SharedStorage:
            # Linear: K [n_kv,d] + V transposed [d,n_kv], one buffer per wave.
            k: fx.Array[fx.Int8, NW * PAGE * D]
            v: fx.Array[fx.Int8, NW * PAGE * D]
            m_part: fx.Array[fx.Float32, NW * BLOCK_M]
            l_part: fx.Array[fx.Float32, NW * BLOCK_M]

    def _exp2(x):
        return fx.Float32(rocdl.exp2(T.f32, _to_raw(x)))

    def _rmax_q(v):
        # Reduce max over the 4 lane-groups, keeping m_q=lane%16 distinct.
        # Not shared with the prefill body's _reduction_pair: that reduces 32
        # in-register slots then one permlane32_swap across the 32-lane halves,
        # while decode reduces a per-lane scalar with shuffle_xor at {16,32}
        # across four lane-groups -- different input shape, primitive, and
        # depth, so no rows_per_wave parameterization unifies them.
        for sh in (16, 32):
            v = v.maximumf(v.shuffle_xor(fx.Int32(sh), fx.Int32(64)))
        return v

    def _rsum_q(v):
        for sh in (16, 32):
            v = v + v.shuffle_xor(fx.Int32(sh), fx.Int32(64))
        return v

    def _pk2(a, b):
        w = rocdl.cvt_pk_fp8_f32(
            T.i32, _to_raw(a), _to_raw(b), fx.Int32(0).ir_value(), 0
        )
        return fx.Int32(w).to(fx.Int16)

    @flyc.kernel(known_block_size=(NW * WARP, 1, 1))
    def decode_kernel(
        q_ptr: fx.Pointer,
        k_ptr: fx.Pointer,  # per-page buffer resource rebased from block table
        v_ptr: fx.Pointer,  # per-page plain view rebased from block table
        o_ptr: fx.Pointer,
        bt_ptr: fx.Pointer,
        cuq_ptr: fx.Pointer,
        cukv_ptr: fx.Pointer,
        qd_ptr: fx.Pointer,
        kd_ptr: fx.Pointer,
        vd_ptr: fx.Pointer,
        ws_ptr: fx.Pointer,  # split-K partial workspace (unused when SPLITK is off)
        seq_len_kv: fx.Int32,
        bt_stride: fx.Int32,
    ):
        AS_G = fx.AddressSpace.Global
        tid = fx.thread_idx.x
        wave = tid // WARP  # cooperating wave in [0, NW)
        lane = tid % WARP
        h = fx.block_idx.x  # kv-head in [0, HKV)
        b = fx.block_idx.y  # batch index
        split = fx.block_idx.z  # KV split in [0, SPLK); 0 when SPLITK is off
        lg = lane // 16
        n = lane % 16

        def gviewi8_2d(ptr):
            # [N, 8] i8 view (8-byte rows) for coalesced b64 operand loads.
            pt = fx.PointerType.get(fx.Int8.ir_type, address_space=AS_G, alignment=1)
            return fx.make_view(
                fx.inttoptr(pt, fx.Int64(fx.ptrtoint(ptr))),
                fx.make_layout((1 << 27, 8), (8, 1)),
            )

        def gviewf32(ptr):
            pt = fx.PointerType.get(fx.Float32.ir_type, address_space=AS_G, alignment=4)
            return fx.make_view(
                fx.inttoptr(pt, fx.Int64(fx.ptrtoint(ptr))),
                fx.make_layout((1 << 20,), (1,)),
            )

        def gviewi32(ptr):
            pt = fx.PointerType.get(fx.Int32.ir_type, address_space=AS_G, alignment=4)
            return fx.make_view(
                fx.inttoptr(pt, fx.Int64(fx.ptrtoint(ptr))),
                fx.make_layout((1 << 20,), (1,)),
            )

        def gviewout(ptr):
            pt = fx.PointerType.get(OUT.ir_type, address_space=AS_G, alignment=2)
            return fx.make_view(
                fx.inttoptr(pt, fx.Int64(fx.ptrtoint(ptr))),
                fx.make_layout((1 << 30,), (1,)),
            )

        gq2 = gviewi8_2d(q_ptr)
        go = gviewout(o_ptr)
        gbt = gviewi32(bt_ptr)
        if const_expr(SPLITK):
            gws = fx.make_view(
                fx.inttoptr(
                    fx.PointerType.get(
                        fx.Float32.ir_type, address_space=AS_G, alignment=4
                    ),
                    fx.Int64(fx.ptrtoint(ws_ptr)),
                ),
                fx.make_layout((1 << 28,), (1,)),
            )

        # Per-page buffer rebasing (production-safe). A >= 2**31-element KV pool
        # cannot be one int32-shaped memref, and the buffer voffset is 32-bit, so
        # instead of one whole-pool resource + flat page_id*PAGE_REGION offset we
        # anchor a BufferDesc at each page's base (from the block table) and index
        # only WITHIN the page (< PAGE_REGION, int32-safe). Same pattern as the
        # prefill kernel's kv_page_div (_make_page_view, reused).
        dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        lds_ptr_ty = fx.PointerType.get(fx.Int8.ir_type, 2, 16)
        _page_i8_ty = fx.PointerType.get(
            fx.Int8.ir_type, address_space=AS_G, alignment=1
        )
        _page_nrec = fx.Int64(PAGE_REGION)
        _page_layout = fx.make_layout(fx.Int32(PAGE_REGION), fx.Int32(1))
        _buf_flags = fx.Int32(buffer_ops._get_buffer_flags())
        v_base_addr = fx.Int64(fx.ptrtoint(v_ptr))

        def k_page_div(page_id):
            # BufferDesc covering exactly page `page_id` (num_records = one page).
            return _make_page_view(
                k_ptr,
                _page_i8_ty,
                1,
                fx.Int64(page_id),
                PAGE_REGION,
                _page_nrec,
                _page_layout,
                fx.Int8.ir_type,
                _buf_flags,
            )

        def _v_page_addr(page_id):
            return v_base_addr + fx.Int64(page_id) * fx.Int64(PAGE_REGION)

        def v_page_view2d(page_id):
            # Per-page [PAGE_REGION//8, 8] i8 view for coalesced b64 V^T operand
            # loads (shuffled V is already in operand order within the page).
            return fx.make_view(
                fx.inttoptr(_page_i8_ty, _v_page_addr(page_id)),
                fx.make_layout((PAGE_REGION // 8, 8), (8, 1)),
            )

        def v_page_flat(page_id):
            # Per-page flat i8 view for the linear scatter-transpose scalar reads.
            return fx.make_view(
                fx.inttoptr(_page_i8_ty, _v_page_addr(page_id)),
                fx.make_layout((PAGE_REGION,), (1,)),
            )

        # Per-tensor descale scalars. c_logit_scale multiplies the fp32 QK
        # logits (fp8 Q/K feed the MFMA raw); vd folds into the epilogue 1/l.
        qd = gviewf32(qd_ptr)[0]
        kd = gviewf32(kd_ptr)[0]
        vd = gviewf32(vd_ptr)[0]
        # GENERALIZATION CANDIDATE: DualwaveFp8KernelContext.init_descale --
        # unify via a decode-safe (no-sink, no-dequant-buffer) descale helper;
        # math here is the same rsqrt(d)*log2e*qd*kd fold.
        c_logit_scale = (fx.Float32(CONST_SCALE) * qd) * kd

        # Sequence-local token index (Q/O) and kv length.
        if const_expr(VARLEN):
            gcuq = gviewi32(cuq_ptr)
            gcukv = gviewi32(cukv_ptr)
            token = gcuq[b]
            kv_len = gcukv[b + 1] - gcukv[b]
        else:
            token = b  # decode: sq==1, so token index == batch index
            kv_len = seq_len_kv

        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, FP8))

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # 2D [N, 8] views for coalesced b64 operand reads (all operand bases are
        # 8-aligned by construction). Each wave owns a BUFSZ K slice per buffer.
        lk2 = lds.k.view(fx.make_layout((NW * PAGE * D // 8, 8), (8, 1)))
        lk_base = fx.Int32(fx.ptrtoint(lds.k.ptr))  # DMA m0 base
        wave_kv = wave * fx.Int32(BUFSZ)  # this wave's K region byte offset
        if const_expr(not SHUF):
            lv2 = lds.v.view(fx.make_layout((NW * PAGE * D // 8, 8), (8, 1)))
            lvf = lds.v.view(fx.make_layout((NW * PAGE * D,), (1,)))  # LDS_V[d,t]
            wave_v = wave * fx.Int32(BUFSZ)
        # Cross-wave combine scratch. o_part overlays the K region (free after the
        # page loop) as f32; a barrier gates the overlay write behind all waves'
        # last K read (see epilogue).
        o_part = fx.make_view(
            fx.inttoptr(
                fx.PointerType.get(fx.Float32.ir_type, 2, 4),
                fx.Int32(fx.ptrtoint(lds.k.ptr)),
            ),
            fx.make_layout((NW * D * BLOCK_M,), (1,)),
        )
        m_part = lds.m_part.view(fx.make_layout((NW * BLOCK_M,), (1,)))
        l_part = lds.l_part.view(fx.make_layout((NW * BLOCK_M,), (1,)))

        def dma128(src_div, lds_addr, src_elem):
            # 64 lanes x 16 bytes -> LDS[lds_addr + lane*16]; src_elem is this
            # lane's element (byte) voffset into the whole-pool buffer tensor.
            lp = fx.inttoptr(lds_ptr_ty, lds_addr)
            dst = fx.make_view(lp, fx.make_layout(1, 1))
            src = fx.slice(src_div, (None, fx.Int32(src_elem)))
            fx.copy(dma_atom, src, dst, soffset=fx.Int32(0))

        def dma_page_k_shuf(page_id, kbufoff):
            # Issue this wave's 8 K DMAs for a shuffled-cache page into K buffer at
            # byte offset kbufoff. The BufferDesc is rebased to this page, so k_src
            # is the WITHIN-page offset (kv-head + layout), int32-safe.
            kdv = k_page_div(page_id)
            hoff32 = fx.Int32(h) * fx.Int32(SHUF_HEAD_STRIDE)
            for c in range_constexpr(8):
                flat0 = fx.Int32(c * 1024) + fx.Int32(lane) * 16
                n_kv = flat0 // D
                d0 = flat0 % D
                k_src = (
                    hoff32 + (d0 // VEC) * fx.Int32(SHUF_K_DCHUNK_STRIDE) + n_kv * VEC
                )
                dma128(kdv, lk_base + kbufoff + fx.Int32(c * 1024), k_src)

        cp64 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int8)

        def load_frag8(view2d, base):
            # operand register r of lane holds K-index (lane//16)*8+r (measured
            # empirically); `base` is the 8-aligned flat i8 offset of
            # r==0, so row = base//8 is the [N,8] row of the 8 contiguous bytes.
            row = fx.Int32(base // 8)
            fr8 = fx.make_rmem_tensor(8, fx.Int8)
            fx.copy(cp64, fx.slice(view2d, (row, None)), fr8)
            fr = fx.make_rmem_tensor(8, FP8)
            fr.store(fx.Vector(fr8.load()).bitcast(FP8))
            return fr

        # ---- Q operand fragments (B-operand: N=m_q=lane%16, K=d) ----
        # Loaded once; Q is [token, q_head=h*16+m_q, d], d contiguous.
        q_row_base = fx.Int64(token) * (H * D) + (fx.Int64(h) * 16 + fx.Int64(n)) * D
        q_frags = []
        for c in range_constexpr(4):
            base = q_row_base + fx.Int64(c * 32) + fx.Int64(lg) * 8
            q_frags.append(load_frag8(gq2, base))

        def repack_p(p):
            # QK C-output -> PV B-operand ds_bpermute gather.
            # p = [d1a[0..3], d1b[0..3]] are the two KV-tile P fragments.
            vals8 = []
            for r in range_constexpr(8):
                sg = (lg % fx.Int32(2)) * 2 + fx.Int32(r // 4)
                sl = sg * 16 + n
                lo = fx.shuffle_idx(p[r % 4], sl, fx.Int32(64))
                hi = fx.shuffle_idx(p[4 + r % 4], sl, fx.Int32(64))
                vals8.append((lg >= fx.Int32(2)).select(hi, lo))
            halfs = [
                _pk2(vals8[2 * hh], vals8[2 * hh + 1]) for hh in range_constexpr(4)
            ]
            b2 = fx.Vector.from_elements(halfs, dtype=fx.Int16).bitcast(fx.Int8)
            fr = fx.make_rmem_tensor(8, FP8)
            fr.store(b2.bitcast(FP8))
            return fr

        npages = (kv_len + fx.Int32(PAGE - 1)) // fx.Int32(PAGE)

        # Inter-workgroup KV split: this workgroup handles split `split`'s
        # contiguous page sub-range [sp0, sp1). Host caps SPLK <= npages so
        # sp0 < sp1 for every launched split (wave 0 always gets sp0).
        if const_expr(SPLITK):
            pages_per_split = (npages + fx.Int32(SPLK - 1)) // fx.Int32(SPLK)
            sp0 = split * pages_per_split
            sp1 = sp0 + pages_per_split
            sp1 = (sp1 < npages).select(sp1, npages)
        else:
            sp0 = fx.Int32(0)
            sp1 = npages

        m_init = fx.Float32(NEG)
        l_init = fx.Float32(0.0)
        fo_init = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(8)]
        init_state = [m_init, l_init] + fo_init

        # Wave-strided KV split within this split's range: wave w streams pages
        # sp0+w, sp0+w+NW, ...  Each wave keeps its own register-resident partial
        # (m_c, l_c, fo_c); the NW partials are LSE-merged in the epilogue.
        # NO s_barrier inside this loop: waves have unequal trip counts, so a
        # workgroup-wide barrier here would deadlock. Ordering within a wave (DMA
        # -> LDS read, and read -> next-page overwrite) is a wave-local s_waitcnt.
        p_start = fx.Int64(sp0 + wave)
        p_stop = fx.Int64(sp1)
        p_step = fx.Int64(NW)
        for pv, state in range(p_start, p_stop, p_step, init=init_state):
            m_c = state[0]
            l_c = state[1]
            fo_c = list(state[2:])
            pvi = fx.Int32(pv)

            # ---- global -> LDS staging (the ONLY cache-layout-dependent code) ----
            # GENERALIZATION CANDIDATE: DualwaveFp8KvGmemToLdsLoader addressing
            # -- unify via a num_waves/rows_per_wave param (decode uses
            # rows_per_wave=16 and a transposed [d,n_kv] V LDS tile).
            if const_expr(SHUF):
                page_id = gbt[b * bt_stride + pvi]
                dma_page_k_shuf(page_id, wave_kv)  # K -> LDS (async)
                gv2p = v_page_view2d(page_id)  # this page's V view
                hoff_v = fx.Int32(h) * fx.Int32(SHUF_HEAD_STRIDE)  # within-page
                # Prefetch every V^T operand fragment (8 chunks x 2 steps) as a b64
                # global load into registers NOW, so V's global latency overlaps
                # the K DMA (and each other) instead of stalling the PV loop.
                v_frags = []
                for step in range_constexpr(2):
                    vf_step = []
                    for c in range_constexpr(8):
                        d_op = fx.Int32(c * 16) + n
                        voff = (
                            hoff_v
                            + (fx.Int32((step * 32) // 16) + lg // 2)
                            * fx.Int32(SHUF_V_TCHUNK_STRIDE)
                            + d_op * VEC
                            + (lg % 2) * 8
                        )
                        vf_step.append(load_frag8(gv2p, voff))
                    v_frags.append(vf_step)
                waitcnt_vm_n(0)  # K DMA + all V loads complete (vmcnt)
            else:
                # Linear (correctness reference): K async DMA into canonical
                # LDS_K[n_kv, d]; V scatter-transposed per element (its contiguous
                # axis is d, but the transposed LDS tile needs n_kv contiguous, so
                # it cannot be DMA-transposed).
                page_id = gbt[b * bt_stride + pvi]
                kdv = k_page_div(page_id)  # BufferDesc rebased to this page
                gvp = v_page_flat(page_id)
                for c in range_constexpr(8):
                    flat0 = fx.Int32(c * 1024) + fx.Int32(lane) * 16
                    n_kv = flat0 // D
                    d0 = flat0 % D
                    # within-page linear offset (page anchored in the BufferDesc)
                    k_src = n_kv * fx.Int32(LIN_T_STRIDE) + fx.Int32(h) * D + d0
                    dma128(kdv, lk_base + wave_kv + fx.Int32(c * 1024), k_src)
                base_lin = fx.Int32(lane) * fx.Int32(LIN_T_STRIDE) + fx.Int32(h) * D
                for d in range_constexpr(D):
                    lvf[wave_v + fx.Int32(d * PAGE) + fx.Int32(lane)] = gvp[
                        base_lin + d
                    ]
                fx.rocdl.s_waitcnt(0)  # linear: DMA (vmcnt) + scalar V write (lgkmcnt)

            # ---- two 32-token flash steps per 64-token page ----
            for step in range_constexpr(2):
                kv_local = step * 32  # token base within page for this step

                # QK: two 16-token sub-tiles -> S^T[n_kv, m_q] C-outputs.
                sub = []
                for st in range_constexpr(2):
                    fc = fx.make_rmem_tensor(4, fx.Float32)
                    fc.store(fx.Vector.filled(4, 0.0, fx.Float32))
                    row0 = kv_local + st * 16
                    for c in range_constexpr(4):
                        kbase = (
                            fx.Int32(row0 + n) * D + fx.Int32(c * 32) + lg * 8 + wave_kv
                        )
                        kfrag = load_frag8(lk2, kbase)
                        fx.gemm(mma, fc, kfrag, q_frags[c], fc)
                    sub.append(fx.Vector(fc.load()))

                # scaled logits + causal/range mask (n_kv >= kv_len -> masked)
                s = []
                for i in range_constexpr(8):
                    st = i // 4
                    vv = i % 4
                    sl = sub[st][vv] * c_logit_scale
                    nkv = (
                        pvi * PAGE
                        + fx.Int32(kv_local + st * 16)
                        + lg * 4
                        + fx.Int32(vv)
                    )
                    valid = nkv < kv_len
                    s.append(valid.select(sl, fx.Float32(NEG)))

                m_loc = s[0]
                for i in range_constexpr(7):
                    m_loc = m_loc.maximumf(s[i + 1])
                m_step = _rmax_q(m_loc)
                m_new = m_c.maximumf(m_step)
                corr = _exp2(m_c - m_new)

                p = [_exp2(s[i] - m_new) for i in range_constexpr(8)]

                sl_loc = p[0]
                for i in range_constexpr(7):
                    sl_loc = sl_loc + p[i + 1]
                step_sum = _rsum_q(sl_loc)

                l_c = l_c * corr + step_sum
                m_c = m_new
                corr_vec = fx.Vector.from_elements(
                    [corr, corr, corr, corr], dtype=fx.Float32
                )

                p_frag = repack_p(p)

                # PV: O^T[d, m_q] += V^T_c @ P, one 16x16x32 issue per d-chunk.
                for c in range_constexpr(8):
                    if const_expr(SHUF):
                        vfrag = v_frags[step][c]  # prefetched register fragment
                    else:
                        vbase = (
                            fx.Int32(c * 16 + n) * PAGE
                            + fx.Int32(kv_local)
                            + lg * 8
                            + wave_v
                        )
                        vfrag = load_frag8(lv2, vbase)
                    pv_acc = fx.make_rmem_tensor(4, fx.Float32)
                    pv_acc.store(fx.Vector.filled(4, 0.0, fx.Float32))
                    fx.gemm(mma, pv_acc, vfrag, p_frag, pv_acc)
                    delta = fx.Vector(pv_acc.load())
                    fo_c[c] = fo_c[c] * corr_vec + delta

            results = yield [m_c, l_c] + fo_c

        # ---- per-wave partial (raw accumulator at this wave's running max m_c) ----
        m_w = results[0]
        l_w = results[1]
        fo_w = results[2:]
        # o_part overlays the K LDS region -- gate the overlay write behind every
        # wave's last K read (waves have unequal trip counts, so a straggler may
        # still be reading K while a finished wave would overwrite it).
        fx.rocdl.s_barrier()
        # Write this wave's partial to LDS combine scratch. o_part[wave, d, m_q]
        # is UNNORMALIZED (accumulator at m_w, not yet divided by l_w).
        wo = wave * fx.Int32(OPART)
        for c in range_constexpr(8):
            fov = fo_w[c]
            for v in range_constexpr(4):
                d_idx = fx.Int32(c * 16) + lg * 4 + fx.Int32(v)
                o_part[wo + d_idx * BLOCK_M + n] = fov[v]
        # m_w / l_w are per query-head n (broadcast across lg); the 4 lg lanes
        # write the same value to the same slot.
        m_part[wave * BLOCK_M + n] = m_w
        l_part[wave * BLOCK_M + n] = l_w
        fx.rocdl.s_waitcnt(0)
        fx.rocdl.s_barrier()  # all NW partials visible before the cross-wave merge

        # ---- flash-decoding LSE merge across the NW waves (on-chip) ----
        # Every wave recomputes the merge for its lanes' (d, m_q) and writes O;
        # all NW waves write identical values to the same global slots
        # (idempotent), which avoids a runtime wave==0 store guard.
        mg = m_part[n]
        for w in range_constexpr(NW - 1):
            mg = mg.maximumf(m_part[fx.Int32((w + 1) * BLOCK_M) + n])
        sc = []
        den = fx.Float32(0.0)
        for w in range_constexpr(NW):
            s_w = _exp2(m_part[fx.Int32(w * BLOCK_M) + n] - mg)
            sc.append(s_w)
            den = den + l_part[fx.Int32(w * BLOCK_M) + n] * s_w

        if const_expr(SPLITK):
            # Write this split's UNNORMALIZED partial (O at running max mg, plus
            # mg and combined l=den) to the global workspace. The combine kernel
            # LSE-merges the SPLK splits and applies vd + 1/l. Layout per
            # (b,h,split): O[D*BLOCK_M] + m[BLOCK_M] + l[BLOCK_M].
            ws_base = fx.Int64(
                (b * fx.Int32(HKV) + h) * fx.Int32(SPLK) + split
            ) * fx.Int64(WS_PER)
            for c in range_constexpr(8):
                for v in range_constexpr(4):
                    d_idx = fx.Int32(c * 16) + lg * 4 + fx.Int32(v)
                    acc = fx.Float32(0.0)
                    for w in range_constexpr(NW):
                        acc = (
                            acc
                            + o_part[fx.Int32(w * OPART) + d_idx * BLOCK_M + n] * sc[w]
                        )
                    gws[ws_base + fx.Int64(d_idx * BLOCK_M) + fx.Int64(n)] = acc
            gws[ws_base + fx.Int64(D * BLOCK_M) + fx.Int64(n)] = mg
            gws[ws_base + fx.Int64(D * BLOCK_M + BLOCK_M) + fx.Int64(n)] = den
        else:
            inv = vd / den  # vd folds the V descale into the combined 1/l
            o_row_base = (
                fx.Int64(token) * (H * D) + (fx.Int64(h) * 16 + fx.Int64(n)) * D
            )
            for c in range_constexpr(8):
                for v in range_constexpr(4):
                    d_idx = fx.Int32(c * 16) + lg * 4 + fx.Int32(v)
                    acc = fx.Float32(0.0)
                    for w in range_constexpr(NW):
                        acc = (
                            acc
                            + o_part[fx.Int32(w * OPART) + d_idx * BLOCK_M + n] * sc[w]
                        )
                    ooff = o_row_base + fx.Int64(d_idx)
                    go[ooff] = (acc * inv).to(OUT)

    @flyc.jit
    def launch(
        q_ptr: fx.Pointer,
        k_ptr: fx.Pointer,
        v_ptr: fx.Pointer,
        o_ptr: fx.Pointer,
        bt_ptr: fx.Pointer,
        cuq_ptr: fx.Pointer,
        cukv_ptr: fx.Pointer,
        qd_ptr: fx.Pointer,
        kd_ptr: fx.Pointer,
        vd_ptr: fx.Pointer,
        ws_ptr: fx.Pointer,
        batch_size: fx.Int32,
        seq_len_kv: fx.Int32,
        bt_stride: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = _cache_tag  # bind into the JIT cache key (see _cache_tag comment)
        decode_kernel(
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            bt_ptr,
            cuq_ptr,
            cukv_ptr,
            qd_ptr,
            kd_ptr,
            vd_ptr,
            ws_ptr,
            seq_len_kv,
            bt_stride,
        ).launch(
            grid=(HKV, batch_size, SPLK),
            block=(NW * WARP, 1, 1),
            stream=stream,
        )

    # GENERALIZATION CANDIDATE: DualwaveSplitKCombineContext /
    # dualwave_splitk_workspace_elems -- reuse the split-K combine scaffolding
    # once it is de-welded from the prefill traits (num_waves=8/rows_per_wave=32)
    # and the CDNA4 no-cluster (plain global reduction) reality. This decode
    # combine is a self-contained global-reduction LSE merge; one wave per
    # (batch, kv-head) merges the SPLK partials.
    @flyc.kernel(known_block_size=(WARP, 1, 1))
    def combine_kernel(
        o_ptr: fx.Pointer,
        ws_ptr: fx.Pointer,
        vd_ptr: fx.Pointer,
        cuq_ptr: fx.Pointer,
    ):
        AS_G = fx.AddressSpace.Global
        lane = fx.thread_idx.x
        h = fx.block_idx.x
        b = fx.block_idx.y
        lg = lane // 16
        n = lane % 16

        def gviewf32c(ptr):
            pt = fx.PointerType.get(fx.Float32.ir_type, AS_G, alignment=4)
            return fx.make_view(
                fx.inttoptr(pt, fx.Int64(fx.ptrtoint(ptr))),
                fx.make_layout((1 << 28,), (1,)),
            )

        def gviewoutc(ptr):
            pt = fx.PointerType.get(OUT.ir_type, AS_G, alignment=2)
            return fx.make_view(
                fx.inttoptr(pt, fx.Int64(fx.ptrtoint(ptr))),
                fx.make_layout((1 << 30,), (1,)),
            )

        gws = gviewf32c(ws_ptr)
        go = gviewoutc(o_ptr)
        vd = gviewf32c(vd_ptr)[0]
        if const_expr(VARLEN):
            gcuq = fx.make_view(
                fx.inttoptr(
                    fx.PointerType.get(fx.Int32.ir_type, AS_G, alignment=4),
                    fx.Int64(fx.ptrtoint(cuq_ptr)),
                ),
                fx.make_layout((1 << 20,), (1,)),
            )
            token = gcuq[b]
        else:
            token = b

        # ws_base(s) = ((b*HKV + h)*SPLK + s) * WS_PER
        bh = (b * fx.Int32(HKV) + h) * fx.Int32(SPLK)

        def m_of(s):
            base = fx.Int64(bh + fx.Int32(s)) * fx.Int64(WS_PER)
            return gws[base + fx.Int64(D * BLOCK_M) + fx.Int64(n)]

        def l_of(s):
            base = fx.Int64(bh + fx.Int32(s)) * fx.Int64(WS_PER)
            return gws[base + fx.Int64(D * BLOCK_M + BLOCK_M) + fx.Int64(n)]

        mg = m_of(0)
        for s in range_constexpr(SPLK - 1):
            mg = mg.maximumf(m_of(s + 1))
        sc = []
        den = fx.Float32(0.0)
        for s in range_constexpr(SPLK):
            s_w = _exp2(m_of(s) - mg)
            sc.append(s_w)
            den = den + l_of(s) * s_w
        inv = vd / den

        o_row_base = fx.Int64(token) * (H * D) + (fx.Int64(h) * 16 + fx.Int64(n)) * D
        for c in range_constexpr(8):
            for v in range_constexpr(4):
                d_idx = fx.Int32(c * 16) + lg * 4 + fx.Int32(v)
                acc = fx.Float32(0.0)
                for s in range_constexpr(SPLK):
                    base = fx.Int64(bh + fx.Int32(s)) * fx.Int64(WS_PER)
                    acc = (
                        acc
                        + gws[base + fx.Int64(d_idx * BLOCK_M) + fx.Int64(n)] * sc[s]
                    )
                go[o_row_base + fx.Int64(d_idx)] = (acc * inv).to(OUT)

    @flyc.jit
    def combine_launch(
        o_ptr: fx.Pointer,
        ws_ptr: fx.Pointer,
        vd_ptr: fx.Pointer,
        cuq_ptr: fx.Pointer,
        batch_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = _cache_tag
        combine_kernel(o_ptr, ws_ptr, vd_ptr, cuq_ptr).launch(
            grid=(HKV, batch_size, 1),
            block=(WARP, 1, 1),
            stream=stream,
        )

    _dummy_holder = {}

    def _dummy_f32():
        import torch

        if "d" not in _dummy_holder:
            _dummy_holder["d"] = torch.zeros(1, device="cuda", dtype=torch.float32)
        return _dummy_holder["d"]

    def _dummy_i32():
        import torch

        if "i" not in _dummy_holder:
            _dummy_holder["i"] = torch.zeros(1, device="cuda", dtype=torch.int32)
        return _dummy_holder["i"]

    def workspace_elems(batch):
        """f32 workspace element count for a split-K launch (0 when SPLK==1)."""
        return batch * HKV * SPLK * WS_PER if SPLITK else 0

    def mod(
        q,
        k,
        v,
        o,
        batch,
        *,
        seq_len_kv=0,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        block_table,
        block_table_stride,
        q_descale,
        k_descale,
        v_descale,
        workspace=None,
        stream=None,
    ):
        import torch

        if stream is None:
            stream = torch.cuda.current_stream()
        cuq = cu_seqlens_q if cu_seqlens_q is not None else _dummy_i32()
        cukv = cu_seqlens_kv if cu_seqlens_kv is not None else _dummy_i32()
        if SPLITK:
            need = workspace_elems(int(batch))
            if workspace is None or workspace.numel() < need:
                workspace = torch.empty(need, device="cuda", dtype=torch.float32)
            ws = workspace
        else:
            ws = _dummy_f32()
        # K/V are plain pointers: the kernel rebases a BufferDesc per page from
        # the block table (no whole-pool memref), so any pool size is fine.
        _run_compiled(
            launch,
            ptr_arg(q),
            ptr_arg(k),
            ptr_arg(v),
            ptr_arg(o),
            ptr_arg(block_table),
            ptr_arg(cuq),
            ptr_arg(cukv),
            ptr_arg(q_descale),
            ptr_arg(k_descale),
            ptr_arg(v_descale),
            ptr_arg(ws),
            int(batch),
            int(seq_len_kv),
            int(block_table_stride),
            stream,
        )
        if SPLITK:
            _run_compiled(
                combine_launch,
                ptr_arg(o),
                ptr_arg(ws),
                ptr_arg(v_descale),
                ptr_arg(cuq),
                int(batch),
                stream,
            )

    mod.workspace_elems = workspace_elems
    return mod
