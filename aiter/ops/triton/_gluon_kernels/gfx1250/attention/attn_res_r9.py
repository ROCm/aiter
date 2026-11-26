# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# Kimi-K3 AttnRes kernels for gfx1250 (Gluon), r9. Per token t of one call:
#     p      = bf16(prefix + delta)                  (stored back; optionally into a block slot)
#     l_i    = (v_i . w) * rsqrt(mean(v_i^2) + eps)  for v_i in {blocks[0..n-1], p}, w = norm_w * qk_w
#     mixed  = sum_i softmax(l)_i v_i,  out = RMSNorm(mixed) * out_w  (+ optional fp8 per-token quant)
# Persistent CTAs (CTA pid owns tokens pid, pid + grid, ...); per token: TDM rows -> LDS, pass A (sums for the logits,
# one cross-wave exchange), lane-local softmax, pass B (mix + output-norm sum), norm / quant / store.
#
# _attn_res_kernel          plain call (blocks re-read from LDS in pass B)
# _attn_res_produce_kernel  plain call with the block rows kept in registers; AHEAD also writes the block-only
#                           mixtures M_c / stats_c of the next K calls (logits either computed here, or read from
#                           the partial logits of _attn_res_logits_kernel when PL=1)
# _attn_res_consume_kernel  group member: reads M_c + stats_c instead of the blocks
# _attn_res_logits_kernel   WMMA logits of a group head: blocks [T, n, H] x (W_hi + W_lo)^T per 256-column slice
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _bar(UB: gl.constexpr):
    # UB=0: gl.barrier() (also drains this wave's global stores); UB=1: none here -- the Membar pass then inserts
    # LDS-only barriers (s_wait_dscnt + s_barrier) exactly at the hazards (after the TDM wait, before a slot refill)
    if UB == 0:
        gl.barrier()


@gluon.jit
def _fexp(x, FEXP: gl.constexpr):
    if FEXP:  # bare v_exp_f32: equals gl.exp unless x * log2(e) < -126 (then 0 instead of a denormal)
        return gl.inline_asm_elementwise(
            "v_exp_f32 $0, $1", "=v,v", [x * 1.4426950408889634], dtype=gl.float32, is_pure=True, pack=1
        )
    else:
        return gl.exp(x)


@gluon.jit
def _issue_plain(
    prefix_ptr,
    delta_ptr,
    blocks_ptr,
    T,
    stride_prefix_m,
    stride_delta_m,
    stride_block_m,
    pbuf,
    dbuf,
    bbuf,
    tok,
    slot,
    HIDDEN: gl.constexpr,
    NUM_BLOCKS: gl.constexpr,
    HAS_DELTA: gl.constexpr,
    NBP: gl.constexpr,
    BD: gl.constexpr,
    PS: gl.constexpr,
    PW: gl.constexpr,
    smem_layout: gl.constexpr,
):
    # Columns >= HIDDEN are out of bounds for TDM and arrive as zeros.
    desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        prefix_ptr, [T, HIDDEN], [stride_prefix_m, 1], [1, BD], smem_layout
    )
    gl.amd.gfx1250.tdm.async_load(desc, [tok, 0], pbuf.index(slot))
    if HAS_DELTA:
        desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            delta_ptr, [T, HIDDEN], [stride_delta_m, 1], [1, BD], smem_layout
        )
        gl.amd.gfx1250.tdm.async_load(desc, [tok, 0], dbuf.index(slot))
    # Blocks: the token's own [n, HIDDEN] tensor, so rows n..NBP-1 of the tile are zero fill.
    # TDM tiles are powers of two; with 8 rows the tile is split into the column pieces PW
    # (4096 + 2048 + 1024 for K3.1) so that two tokens fit in LDS.
    for i in gl.static_range(len(PW) * (NUM_BLOCKS > 0)):
        desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            blocks_ptr + tok.to(gl.int64) * stride_block_m, [NUM_BLOCKS, HIDDEN], [HIDDEN, 1], [NBP, PW[i]], smem_layout
        )
        gl.amd.gfx1250.tdm.async_load(desc, [0, PS[i]], bbuf[i].index(slot))


@gluon.jit
def _attn_res_kernel(
    prefix_ptr,
    delta_ptr,
    blocks_ptr,
    norm_weight_ptr,
    qk_weight_ptr,
    output_norm_weight_ptr,
    output_ptr,
    output_scale_ptr,
    T,
    stride_prefix_m,
    stride_delta_m,
    stride_block_m,
    stride_output_m,
    eps,
    output_norm_eps,
    HIDDEN: gl.constexpr,
    NUM_BLOCKS: gl.constexpr,
    BLOCK_WRITE_IDX: gl.constexpr,
    HAS_DELTA: gl.constexpr,
    WRITE_BLOCK: gl.constexpr,
    STORE_PREFIX: gl.constexpr,
    APPLY_OUTPUT_NORM: gl.constexpr,
    QUANT_MAX: gl.constexpr,
    NBP: gl.constexpr,  # rows of the block tile: power of two >= NUM_BLOCKS
    EPL: gl.constexpr,  # contiguous columns of every row held by a lane (4 or 2)
    C: gl.constexpr,  # columns per chunk = 32 lanes * EPL * NUM_WARPS
    NCH: gl.constexpr,  # chunks per row = cdiv(HIDDEN, C)
    BD: gl.constexpr,  # prefix/delta tile width: power of two >= HIDDEN
    PS: gl.constexpr,  # column pieces of the block tile: start columns ...
    PW: gl.constexpr,  # ... and widths (powers of two, multiples of C, covering NCH * C)
    NSLOT: gl.constexpr,  # token slots in LDS: 2 = token k + 1 loads while k computes; 1 = load, then compute
    NUM_WARPS: gl.constexpr,
    EARLY: gl.constexpr,  # NSLOT=1 only: free the slot after pass B and issue token k + 1 there
    UB: gl.constexpr = 0,  # see _bar
):
    pid = gl.program_id(0)
    grid = gl.num_programs(0)
    n_tok = gl.cdiv(T - pid, grid)  # this CTA owns tokens pid, pid + grid, ...

    L: gl.constexpr = 1 + HAS_DELTA + (NUM_BLOCKS > 0) * len(PW)  # TDM ops per token
    # [rows, C] chunk: a lane holds EPL contiguous columns of every row; waves tile the columns.
    layout: gl.constexpr = gl.BlockedLayout([NBP, EPL], [1, 32], [1, NUM_WARPS], [1, 0])
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    dtype: gl.constexpr = prefix_ptr.dtype.element_ty

    pbuf = gl.allocate_shared_memory(dtype, [NSLOT, 1, BD], smem_layout)  # [slot, 1, BD]
    dbuf = gl.allocate_shared_memory(dtype, [NSLOT, 1, BD], smem_layout) if HAS_DELTA else pbuf
    bbuf = ()  # per piece: [slot, NBP, PW]
    for i in gl.static_range(len(PW) * (NUM_BLOCKS > 0)):
        bbuf += (gl.allocate_shared_memory(dtype, [NSLOT, NBP, PW[i]], smem_layout),)

    if NSLOT == 2:
        _issue_plain(
            prefix_ptr,
            delta_ptr,
            blocks_ptr,
            T,
            stride_prefix_m,
            stride_delta_m,
            stride_block_m,
            pbuf,
            dbuf,
            bbuf,
            pid,
            0,
            HIDDEN,
            NUM_BLOCKS,
            HAS_DELTA,
            NBP,
            BD,
            PS,
            PW,
            smem_layout,
        )

    cols = gl.arange(0, C, layout=gl.SliceLayout(0, layout))[None, :]
    rows = gl.arange(0, NBP, layout=gl.SliceLayout(1, layout))[:, None]
    w = ()  # norm_weight * qk_weight, per chunk
    ow = ()  # output norm weight, per chunk
    for c in gl.static_range(NCH):
        cc = c * C + cols
        if NUM_BLOCKS > 0:
            w += (
                gl.load(norm_weight_ptr + cc, mask=cc < HIDDEN, other=0.0).to(gl.float32)
                * gl.load(qk_weight_ptr + cc, mask=cc < HIDDEN, other=0.0).to(gl.float32),
            )
        if APPLY_OUTPUT_NORM:
            ow += (gl.load(output_norm_weight_ptr + cc, mask=cc < HIDDEN, other=0.0).to(gl.float32),)

    if EARLY:
        _issue_plain(
            prefix_ptr,
            delta_ptr,
            blocks_ptr,
            T,
            stride_prefix_m,
            stride_delta_m,
            stride_block_m,
            pbuf,
            dbuf,
            bbuf,
            pid,
            0,
            HIDDEN,
            NUM_BLOCKS,
            HAS_DELTA,
            NBP,
            BD,
            PS,
            PW,
            smem_layout,
        )
    for k in range(n_tok):
        tok64 = (pid + k * grid).to(gl.int64)
        slot = k % NSLOT
        if not EARLY:
            _bar(UB)  # every wave is done with token k - 1 (and with its slot)
        if EARLY:
            gl.amd.gfx1250.tdm.async_wait(0)
        elif NSLOT == 1:
            # One slot: load token k, then compute it; the CU's other CTAs compute meanwhile.
            _issue_plain(
                prefix_ptr,
                delta_ptr,
                blocks_ptr,
                T,
                stride_prefix_m,
                stride_delta_m,
                stride_block_m,
                pbuf,
                dbuf,
                bbuf,
                pid + k * grid,
                0,
                HIDDEN,
                NUM_BLOCKS,
                HAS_DELTA,
                NBP,
                BD,
                PS,
                PW,
                smem_layout,
            )
            gl.amd.gfx1250.tdm.async_wait(0)
        elif k + 1 < n_tok:
            # The freed slot takes token k + 1, which loads while token k computes. Token k
            # has landed once only those L ops may be outstanding.
            _issue_plain(
                prefix_ptr,
                delta_ptr,
                blocks_ptr,
                T,
                stride_prefix_m,
                stride_delta_m,
                stride_block_m,
                pbuf,
                dbuf,
                bbuf,
                pid + (k + 1) * grid,
                (k + 1) % 2,
                HIDDEN,
                NUM_BLOCKS,
                HAS_DELTA,
                NBP,
                BD,
                PS,
                PW,
                smem_layout,
            )
            gl.amd.gfx1250.tdm.async_wait(L)
        else:
            gl.amd.gfx1250.tdm.async_wait(0)
        _bar(UB)  # every wave reads tiles issued by other waves

        # Pass A: p = bf16(prefix + delta), stored back; per-lane partial sums for the logits.
        p = ()
        dot_p = gl.zeros([1, C], gl.float32, layout)
        sq_p = gl.zeros([1, C], gl.float32, layout)
        dot_b = gl.zeros([NBP, C], gl.float32, layout)
        sq_b = gl.zeros([NBP, C], gl.float32, layout)
        for i in gl.static_range(len(PW)):
            for j in gl.static_range(min(PW[i], NCH * C - PS[i]) // C):
                cc = PS[i] + j * C + cols
                pc = pbuf.index(slot).slice(PS[i] + j * C, C, 1).load(layout)
                if HAS_DELTA:
                    # bf16 prefix-add; the rounded value is the residual source.
                    dc = dbuf.index(slot).slice(PS[i] + j * C, C, 1).load(layout)
                    pc = (pc.to(gl.float32) + dc.to(gl.float32)).to(dtype)
                    if STORE_PREFIX:
                        gl.store(prefix_ptr + tok64 * stride_prefix_m + cc, pc, mask=cc < HIDDEN)
                if WRITE_BLOCK:
                    gl.store(blocks_ptr + tok64 * stride_block_m + BLOCK_WRITE_IDX * HIDDEN + cc, pc, mask=cc < HIDDEN)
                pc = pc.to(gl.float32)
                p += (pc,)
                if NUM_BLOCKS > 0:
                    v = bbuf[i].index(slot).slice(j * C, C, 1).load(layout).to(gl.float32)
                    dot_p = gl.fma(pc, w[PS[i] // C + j], dot_p)
                    sq_p = gl.fma(pc, pc, sq_p)
                    dot_b = gl.fma(v, w[PS[i] // C + j], dot_b)
                    sq_b = gl.fma(v, v, sq_b)

        if NUM_BLOCKS > 0:
            # One cross-wave exchange per statistic, then the whole softmax is lane-local.
            logit_p = gl.sum(dot_p, axis=1, keep_dims=True) * gl.rsqrt(
                gl.sum(sq_p, axis=1, keep_dims=True) * (1.0 / HIDDEN) + eps
            )
            logit_b = gl.sum(dot_b, axis=1, keep_dims=True) * gl.rsqrt(
                gl.sum(sq_b, axis=1, keep_dims=True) * (1.0 / HIDDEN) + eps
            )
            if NBP > NUM_BLOCKS:
                logit_b = gl.where(rows < NUM_BLOCKS, logit_b, float("-inf"))
            m = gl.maximum(gl.max(logit_b, axis=0, keep_dims=True), logit_p)
            e_p = gl.exp(logit_p - m)
            e_b = gl.exp(logit_b - m)
            rden = 1.0 / (gl.sum(e_b, axis=0, keep_dims=True) + e_p)
            e_p = e_p * rden
            e_b = e_b * rden

        # Pass B: mix with the final weights (blocks re-read from LDS) and output-norm sum.
        mixed = ()
        sq_o = gl.zeros([1, C], gl.float32, layout)
        for i in gl.static_range(len(PW)):
            for j in gl.static_range(min(PW[i], NCH * C - PS[i]) // C):
                mc = p[PS[i] // C + j]
                if NUM_BLOCKS > 0:
                    v = bbuf[i].index(slot).slice(j * C, C, 1).load(layout).to(gl.float32)
                    mc = mc * e_p
                    for r in gl.static_range(NUM_BLOCKS):  # lane-local FMA chain over the rows
                        mc = gl.fma(gl.amd.slice(v, [1, C], [r, 0]), gl.amd.slice(e_b, [1, 1], [r, 0]), mc)
                if APPLY_OUTPUT_NORM:
                    sq_o = gl.fma(mc, mc, sq_o)
                mixed += (mc,)

        if EARLY:
            _bar(UB)  # every wave is done reading this token's slot
            if k + 1 < n_tok:
                _issue_plain(
                    prefix_ptr,
                    delta_ptr,
                    blocks_ptr,
                    T,
                    stride_prefix_m,
                    stride_delta_m,
                    stride_block_m,
                    pbuf,
                    dbuf,
                    bbuf,
                    pid + (k + 1) * grid,
                    0,
                    HIDDEN,
                    NUM_BLOCKS,
                    HAS_DELTA,
                    NBP,
                    BD,
                    PS,
                    PW,
                    smem_layout,
                )
        if APPLY_OUTPUT_NORM:
            rstd = gl.rsqrt(gl.sum(sq_o, axis=1, keep_dims=True) * (1.0 / HIDDEN) + output_norm_eps)
        out = ()
        amax = gl.zeros([1, C], gl.float32, layout)
        for c in gl.static_range(NCH):
            oc = mixed[c]
            if APPLY_OUTPUT_NORM:
                oc = oc * rstd * ow[c]
            if QUANT_MAX > 0:
                # Round to bf16 first, like the unquantized output would be.
                oc = oc.to(dtype).to(gl.float32)
                amax = gl.maximum(amax, gl.abs(oc))
            out += (oc,)

        if QUANT_MAX > 0:
            scale = gl.maximum(gl.max(amax, axis=1, keep_dims=True), 1e-10) * (1.0 / QUANT_MAX)
            rscale = 1.0 / scale
            gl.store(output_scale_ptr + tok64 + cols, scale + gl.zeros([1, C], gl.float32, layout), mask=cols == 0)
        for c in gl.static_range(NCH):
            cc = c * C + cols
            oc = out[c]
            if QUANT_MAX > 0:
                oc = gl.minimum(gl.maximum(oc * rscale, -QUANT_MAX), QUANT_MAX)
            gl.store(output_ptr + tok64 * stride_output_m + cc, oc.to(output_ptr.dtype.element_ty), mask=cc < HIDDEN)


@gluon.jit
def _issue_produce(
    prefix_ptr,
    delta_ptr,
    blocks_ptr,
    T,
    stride_prefix_m,
    stride_delta_m,
    stride_block_m,
    pbuf,
    dbuf,
    bbuf,
    tok,
    slot,
    HIDDEN: gl.constexpr,
    NUM_BLOCKS: gl.constexpr,
    HAS_DELTA: gl.constexpr,
    NBP: gl.constexpr,
    BD: gl.constexpr,
    PS: gl.constexpr,
    PW: gl.constexpr,
    smem_layout: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    if len(PW) == 3 and NUM_BLOCKS > 0 and NUM_WARPS == 8:
        pd = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            prefix_ptr, [T, HIDDEN], [stride_prefix_m, 1], [1, BD], smem_layout
        )
        pd = gl.amd.gfx1250.tdm.update_tensor_descriptor(pd, add_offsets=[tok, 0])
        bb = blocks_ptr + tok.to(gl.int64) * stride_block_m
        b0 = gl.amd.gfx1250.tdm.make_tensor_descriptor(bb, [NUM_BLOCKS, HIDDEN], [HIDDEN, 1], [NBP, PW[0]], smem_layout)
        b0 = gl.amd.gfx1250.tdm.update_tensor_descriptor(b0, add_offsets=[0, PS[0]])
        b1 = gl.amd.gfx1250.tdm.make_tensor_descriptor(bb, [NUM_BLOCKS, HIDDEN], [HIDDEN, 1], [NBP, PW[1]], smem_layout)
        b1 = gl.amd.gfx1250.tdm.update_tensor_descriptor(b1, add_offsets=[0, PS[1]])
        b2 = gl.amd.gfx1250.tdm.make_tensor_descriptor(bb, [NUM_BLOCKS, HIDDEN], [HIDDEN, 1], [NBP, PW[2]], smem_layout)
        b2 = gl.amd.gfx1250.tdm.update_tensor_descriptor(b2, add_offsets=[0, PS[2]])
        gl.amd.gfx1250.tdm.async_load_fused(
            [
                (b0, bbuf[0].index(slot), 0b00001111),
                (b1, bbuf[1].index(slot), 0b00110000),
                (b2, bbuf[2].index(slot), 0b01000000),
                (pd, pbuf.index(slot), 0b10000000),
            ]
        )
        if HAS_DELTA:
            desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                delta_ptr, [T, HIDDEN], [stride_delta_m, 1], [1, BD], smem_layout
            )
            gl.amd.gfx1250.tdm.async_load(desc, [tok, 0], dbuf.index(slot))
    else:
        # Columns >= HIDDEN are out of bounds for TDM and arrive as zeros.
        desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            prefix_ptr, [T, HIDDEN], [stride_prefix_m, 1], [1, BD], smem_layout
        )
        gl.amd.gfx1250.tdm.async_load(desc, [tok, 0], pbuf.index(slot))
        if HAS_DELTA:
            desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                delta_ptr, [T, HIDDEN], [stride_delta_m, 1], [1, BD], smem_layout
            )
            gl.amd.gfx1250.tdm.async_load(desc, [tok, 0], dbuf.index(slot))
        # Blocks: the token's own [n, HIDDEN] tensor, so rows n..NBP-1 of the tile are zero fill.
        # TDM tiles are powers of two; with 8 rows the tile is split into the column pieces PW
        # (4096 + 2048 + 1024 for K3.1) so that two tokens fit in LDS.
        for i in gl.static_range(len(PW) * (NUM_BLOCKS > 0)):
            desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                blocks_ptr + tok.to(gl.int64) * stride_block_m,
                [NUM_BLOCKS, HIDDEN],
                [HIDDEN, 1],
                [NBP, PW[i]],
                smem_layout,
            )
            gl.amd.gfx1250.tdm.async_load(desc, [0, PS[i]], bbuf[i].index(slot))


@gluon.jit
def _mix_from_logits(
    vs,
    l_raw,
    rstd_b,
    rows,
    row64,
    mix_ptr,
    stats_ptr,
    cols,
    HIDDEN: gl.constexpr,
    NUM_BLOCKS: gl.constexpr,
    NBP: gl.constexpr,
    NCH: gl.constexpr,
    C: gl.constexpr,
    layout: gl.constexpr,
    FEXP: gl.constexpr,
):
    # one block-only mixture from its raw logits [NBP, 1]: softmax over the rows, FMA chain, stores
    a_logit = l_raw * rstd_b
    if NBP > NUM_BLOCKS:
        a_logit = gl.where(rows < NUM_BLOCKS, a_logit, float("-inf"))
    a_m = gl.max(a_logit, axis=0, keep_dims=True)
    a_e = _fexp(a_logit - a_m, FEXP)
    a_den = gl.sum(a_e, axis=0, keep_dims=True)
    a_mix = ()
    for q in gl.static_range(NCH):
        a_mix += (gl.amd.slice(vs[q], [1, C], [0, 0]) * gl.amd.slice(a_e, [1, 1], [0, 0]),)
    for r in gl.static_range(NUM_BLOCKS - 1):
        a_nxt = ()
        for q in gl.static_range(NCH):
            a_nxt += (gl.fma(gl.amd.slice(vs[q], [1, C], [r + 1, 0]), gl.amd.slice(a_e, [1, 1], [r + 1, 0]), a_mix[q]),)
        a_mix = a_nxt
    for q in gl.static_range(NCH):
        cc = q * C + cols
        gl.store(mix_ptr + row64 * HIDDEN + cc, a_mix[q], mask=cc < HIDDEN, cache_modifier=".cs")
    gl.store(stats_ptr + row64 * 2 + cols, gl.where(cols == 0, a_m, a_den), mask=cols < 2)


@gluon.jit
def _attn_res_produce_kernel(
    prefix_ptr,
    delta_ptr,
    blocks_ptr,
    norm_weight_ptr,
    qk_weight_ptr,
    output_norm_weight_ptr,
    output_ptr,
    output_scale_ptr,
    w_ptr,
    mix_ptr,
    stats_ptr,
    plog_ptr,
    T,
    K,
    stride_prefix_m,
    stride_delta_m,
    stride_block_m,
    stride_output_m,
    eps,
    output_norm_eps,
    HIDDEN: gl.constexpr,
    NUM_BLOCKS: gl.constexpr,
    BLOCK_WRITE_IDX: gl.constexpr,
    HAS_DELTA: gl.constexpr,
    WRITE_BLOCK: gl.constexpr,
    STORE_PREFIX: gl.constexpr,
    APPLY_OUTPUT_NORM: gl.constexpr,
    QUANT_MAX: gl.constexpr,
    NBP: gl.constexpr,  # rows of the block tile: power of two >= NUM_BLOCKS
    EPL: gl.constexpr,  # contiguous columns of every row held by a lane (4 or 2)
    C: gl.constexpr,  # columns per chunk = 32 lanes * EPL * NUM_WARPS
    NCH: gl.constexpr,  # chunks per row = cdiv(HIDDEN, C)
    BD: gl.constexpr,  # prefix/delta tile width: power of two >= HIDDEN
    PS: gl.constexpr,  # column pieces of the block tile: start columns ...
    PW: gl.constexpr,  # ... and widths (powers of two, multiples of C, covering NCH * C)
    NSLOT: gl.constexpr,  # token slots in LDS: 2 = token k + 1 loads while k computes; 1 = load, then compute
    AHEAD: gl.constexpr,  # also emit the block-only mixtures of the next K calls
    NUM_WARPS: gl.constexpr,
    EARLY: gl.constexpr,  # NSLOT=1 only: free the slot after pass A (rows are in registers) and issue token k + 1
    UB: gl.constexpr = 0,  # see _bar
    PL: gl.constexpr = 0,  # AHEAD: logits from plog [T, SP, 8, 32] (S slices summed in order) instead of in-kernel dots
    S: gl.constexpr = 28,
    SP: gl.constexpr = 32,
    FEXP: gl.constexpr = 0,  # PL: bare v_exp_f32 in the mixture loop
):
    # AHEAD: w_ptr [K, HIDDEN] fp32 (norm_weight * qk_weight of the next K calls), mix_ptr
    # [T, K, HIDDEN] fp32, stats_ptr [T, K, 2] fp32 (max logit, sum of exp); see attn_res_mix.py.
    pid = gl.program_id(0)
    grid = gl.num_programs(0)
    n_tok = gl.cdiv(T - pid, grid)  # this CTA owns tokens pid, pid + grid, ...

    L: gl.constexpr = (
        (1 + HAS_DELTA)
        if (len(PW) == 3 and NUM_BLOCKS > 0 and NUM_WARPS == 8)
        else 1 + HAS_DELTA + (NUM_BLOCKS > 0) * len(PW)
    )  # TDM ops per token (per wave)
    # [rows, C] chunk: a lane holds EPL contiguous columns of every row; waves tile the columns.
    layout: gl.constexpr = gl.BlockedLayout([NBP, EPL], [1, 32], [1, NUM_WARPS], [1, 0])
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    dtype: gl.constexpr = prefix_ptr.dtype.element_ty

    pbuf = gl.allocate_shared_memory(dtype, [NSLOT, 1, BD], smem_layout)  # [slot, 1, BD]
    dbuf = gl.allocate_shared_memory(dtype, [NSLOT, 1, BD], smem_layout) if HAS_DELTA else pbuf
    bbuf = ()  # per piece: [slot, NBP, PW]
    for i in gl.static_range(len(PW) * (NUM_BLOCKS > 0)):
        bbuf += (gl.allocate_shared_memory(dtype, [NSLOT, NBP, PW[i]], smem_layout),)

    if PL:
        lsc = gl.allocate_shared_memory(gl.float32, [32, 8], smem_layout)  # [mixture, row] logits of one token
        lsc3 = lsc.reinterpret(gl.float32, [32, 8, 1], smem_layout)
    if NSLOT == 2:
        _issue_produce(
            prefix_ptr,
            delta_ptr,
            blocks_ptr,
            T,
            stride_prefix_m,
            stride_delta_m,
            stride_block_m,
            pbuf,
            dbuf,
            bbuf,
            pid,
            0,
            HIDDEN,
            NUM_BLOCKS,
            HAS_DELTA,
            NBP,
            BD,
            PS,
            PW,
            smem_layout,
            NUM_WARPS,
        )

    cols = gl.arange(0, C, layout=gl.SliceLayout(0, layout))[None, :]
    rows = gl.arange(0, NBP, layout=gl.SliceLayout(1, layout))[:, None]
    w = ()  # norm_weight * qk_weight, per chunk
    ow = ()  # output norm weight, per chunk
    for c in gl.static_range(NCH):
        cc = c * C + cols
        if NUM_BLOCKS > 0:
            w += (
                gl.load(norm_weight_ptr + cc, mask=cc < HIDDEN, other=0.0).to(gl.float32)
                * gl.load(qk_weight_ptr + cc, mask=cc < HIDDEN, other=0.0).to(gl.float32),
            )
        if APPLY_OUTPUT_NORM:
            ow += (gl.load(output_norm_weight_ptr + cc, mask=cc < HIDDEN, other=0.0).to(gl.float32),)

    if EARLY:
        _issue_produce(
            prefix_ptr,
            delta_ptr,
            blocks_ptr,
            T,
            stride_prefix_m,
            stride_delta_m,
            stride_block_m,
            pbuf,
            dbuf,
            bbuf,
            pid,
            0,
            HIDDEN,
            NUM_BLOCKS,
            HAS_DELTA,
            NBP,
            BD,
            PS,
            PW,
            smem_layout,
            NUM_WARPS,
        )
    for k in range(n_tok):
        tok64 = (pid + k * grid).to(gl.int64)
        slot = k % NSLOT
        if not EARLY:
            _bar(UB)  # every wave is done with token k - 1 (and with its slot)
        if EARLY:
            gl.amd.gfx1250.tdm.async_wait(0)
        elif NSLOT == 1:
            # One slot: load token k, then compute it; the CU's other CTAs compute meanwhile.
            _issue_produce(
                prefix_ptr,
                delta_ptr,
                blocks_ptr,
                T,
                stride_prefix_m,
                stride_delta_m,
                stride_block_m,
                pbuf,
                dbuf,
                bbuf,
                pid + k * grid,
                0,
                HIDDEN,
                NUM_BLOCKS,
                HAS_DELTA,
                NBP,
                BD,
                PS,
                PW,
                smem_layout,
                NUM_WARPS,
            )
            gl.amd.gfx1250.tdm.async_wait(0)
        elif k + 1 < n_tok:
            # The freed slot takes token k + 1, which loads while token k computes. Token k
            # has landed once only those L ops may be outstanding.
            _issue_produce(
                prefix_ptr,
                delta_ptr,
                blocks_ptr,
                T,
                stride_prefix_m,
                stride_delta_m,
                stride_block_m,
                pbuf,
                dbuf,
                bbuf,
                pid + (k + 1) * grid,
                (k + 1) % 2,
                HIDDEN,
                NUM_BLOCKS,
                HAS_DELTA,
                NBP,
                BD,
                PS,
                PW,
                smem_layout,
                NUM_WARPS,
            )
            gl.amd.gfx1250.tdm.async_wait(L)
        else:
            gl.amd.gfx1250.tdm.async_wait(0)
        _bar(UB)  # every wave reads tiles issued by other waves

        # Pass A: p = bf16(prefix + delta), stored back; per-lane partial sums for the logits.
        p = ()
        vs = ()  # the token's block rows, unpacked once and kept in registers
        dot_p = gl.zeros([1, C], gl.float32, layout)
        sq_p = gl.zeros([1, C], gl.float32, layout)
        dot_b = gl.zeros([NBP, C], gl.float32, layout)
        sq_b = gl.zeros([NBP, C], gl.float32, layout)
        for i in gl.static_range(len(PW)):
            for j in gl.static_range(min(PW[i], NCH * C - PS[i]) // C):
                cc = PS[i] + j * C + cols
                pc = pbuf.index(slot).slice(PS[i] + j * C, C, 1).load(layout)
                if HAS_DELTA:
                    # bf16 prefix-add; the rounded value is the residual source.
                    dc = dbuf.index(slot).slice(PS[i] + j * C, C, 1).load(layout)
                    pc = (pc.to(gl.float32) + dc.to(gl.float32)).to(dtype)
                    if STORE_PREFIX and not EARLY:
                        gl.store(prefix_ptr + tok64 * stride_prefix_m + cc, pc, mask=cc < HIDDEN)
                if WRITE_BLOCK and not EARLY:
                    gl.store(blocks_ptr + tok64 * stride_block_m + BLOCK_WRITE_IDX * HIDDEN + cc, pc, mask=cc < HIDDEN)
                pc = pc.to(gl.float32)
                p += (pc,)
                if NUM_BLOCKS > 0:
                    v = bbuf[i].index(slot).slice(j * C, C, 1).load(layout).to(gl.float32)
                    vs += (v,)
                    dot_p = gl.fma(pc, w[PS[i] // C + j], dot_p)
                    sq_p = gl.fma(pc, pc, sq_p)
                    dot_b = gl.fma(v, w[PS[i] // C + j], dot_b)
                    sq_b = gl.fma(v, v, sq_b)
        if EARLY:
            _bar(UB)  # every wave has its rows in registers: the slot is free
            if k + 1 < n_tok:
                _issue_produce(
                    prefix_ptr,
                    delta_ptr,
                    blocks_ptr,
                    T,
                    stride_prefix_m,
                    stride_delta_m,
                    stride_block_m,
                    pbuf,
                    dbuf,
                    bbuf,
                    pid + (k + 1) * grid,
                    0,
                    HIDDEN,
                    NUM_BLOCKS,
                    HAS_DELTA,
                    NBP,
                    BD,
                    PS,
                    PW,
                    smem_layout,
                    NUM_WARPS,
                )
            for c in gl.static_range(NCH):
                cc = c * C + cols
                if HAS_DELTA and STORE_PREFIX:
                    gl.store(prefix_ptr + tok64 * stride_prefix_m + cc, p[c].to(dtype), mask=cc < HIDDEN)
                if WRITE_BLOCK:
                    gl.store(
                        blocks_ptr + tok64 * stride_block_m + BLOCK_WRITE_IDX * HIDDEN + cc,
                        p[c].to(dtype),
                        mask=cc < HIDDEN,
                    )

        if NUM_BLOCKS > 0:
            # One cross-wave exchange for all four statistics ([NBP, C, 2, 2] summed over the
            # columns; the prefix pair is broadcast over the rows), then all is lane-local.
            s = gl.sum(gl.join(gl.join(dot_b, sq_b), gl.join(dot_p, sq_p)), axis=1)
            s_b, s_p = gl.split(s)
            d_b, q_b = gl.split(s_b)
            d_p, q_p = gl.split(s_p)
            row_layout: gl.constexpr = gl.SliceLayout(1, layout)
            d_b = gl.convert_layout(d_b, row_layout)[:, None]
            q_b = gl.convert_layout(q_b, row_layout)[:, None]
            d_p = gl.amd.slice(gl.convert_layout(d_p, row_layout)[:, None], [1, 1], [0, 0])
            q_p = gl.amd.slice(gl.convert_layout(q_p, row_layout)[:, None], [1, 1], [0, 0])
            rstd_b = gl.rsqrt(q_b * (1.0 / HIDDEN) + eps)
            logit_p = d_p * gl.rsqrt(q_p * (1.0 / HIDDEN) + eps)
            logit_b = d_b * rstd_b
            if NBP > NUM_BLOCKS:
                logit_b = gl.where(rows < NUM_BLOCKS, logit_b, float("-inf"))
            m = gl.maximum(gl.max(logit_b, axis=0, keep_dims=True), logit_p)
            e_p = gl.exp(logit_p - m)
            e_b = gl.exp(logit_b - m)
            rden = 1.0 / (gl.sum(e_b, axis=0, keep_dims=True) + e_p)
            e_p = e_p * rden
            e_b = e_b * rden

        if AHEAD and PL:
            # Precomputed logits: sum the S slice partials in order -> LDS; the mixture loop has no reduction/barrier.
            LP: gl.constexpr = gl.BlockedLayout([SP, 1, 1], [1, 32, 1], [1, 1, NUM_WARPS], [2, 1, 0])
            ps_ = gl.arange(0, SP, layout=gl.SliceLayout(1, gl.SliceLayout(2, LP)))[:, None, None]
            pc_ = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(2, LP)))[None, :, None]
            pr_ = gl.arange(0, 8, layout=gl.SliceLayout(0, gl.SliceLayout(1, LP)))[None, None, :]
            plv = gl.load(
                plog_ptr + tok64 * (SP * 256) + ps_ * 256 + pr_ * 32 + pc_,
                mask=(ps_ < S) & (pc_ >= 0) & (pr_ < NUM_BLOCKS),
                other=0.0,
            )
            lsc.store(gl.sum(plv, axis=0))
            l_n = lsc3.index(0).slice(0, NBP, 0).load(layout)
            for kc in range(K):
                l_ = l_n
                l_n = lsc3.index(gl.minimum(kc + 1, K - 1)).slice(0, NBP, 0).load(layout)
                _mix_from_logits(
                    vs,
                    l_,
                    rstd_b,
                    rows,
                    tok64 * K + kc,
                    mix_ptr,
                    stats_ptr,
                    cols,
                    HIDDEN,
                    NUM_BLOCKS,
                    NBP,
                    NCH,
                    C,
                    layout,
                    FEXP,
                )
        elif AHEAD:
            # Block-only mixtures of the next K calls: the rows are already in registers.
            w_cur = ()
            for q in gl.static_range(NCH):
                cc = q * C + cols
                w_cur += (gl.load(w_ptr + cc, mask=cc < HIDDEN, other=0.0),)
            for kc in range(K):
                row64 = tok64 * K + kc  # token-major: a token's K rows are one write stream
                cn = gl.minimum(kc + 1, K - 1)
                w_nxt = ()
                for q in gl.static_range(NCH):
                    cc = q * C + cols
                    w_nxt += (gl.load(w_ptr + cn * HIDDEN + cc, mask=cc < HIDDEN, other=0.0),)
                a_dot = gl.zeros([NBP, C], gl.float32, layout)
                for q in gl.static_range(NCH):
                    a_dot = gl.fma(vs[q], w_cur[q], a_dot)
                w_cur = w_nxt
                a_logit = gl.sum(a_dot, axis=1, keep_dims=True) * rstd_b
                if NBP > NUM_BLOCKS:
                    a_logit = gl.where(rows < NUM_BLOCKS, a_logit, float("-inf"))
                a_m = gl.max(a_logit, axis=0, keep_dims=True)
                a_e = gl.exp(a_logit - a_m)
                a_den = gl.sum(a_e, axis=0, keep_dims=True)
                a_mix = ()
                for q in gl.static_range(NCH):
                    a_mix += (gl.amd.slice(vs[q], [1, C], [0, 0]) * gl.amd.slice(a_e, [1, 1], [0, 0]),)
                for r in gl.static_range(NUM_BLOCKS - 1):  # rows outermost: no back-to-back dependent FMAs
                    a_nxt = ()
                    for q in gl.static_range(NCH):
                        a_nxt += (
                            gl.fma(
                                gl.amd.slice(vs[q], [1, C], [r + 1, 0]), gl.amd.slice(a_e, [1, 1], [r + 1, 0]), a_mix[q]
                            ),
                        )
                    a_mix = a_nxt
                for q in gl.static_range(NCH):
                    cc = q * C + cols
                    gl.store(mix_ptr + row64 * HIDDEN + cc, a_mix[q], mask=cc < HIDDEN, cache_modifier=".cs")
                gl.store(stats_ptr + row64 * 2 + cols, gl.where(cols == 0, a_m, a_den), mask=cols < 2)

        # Pass B: mix the rows with the final weights; output-norm sum.
        mixed = ()
        sq_o = gl.zeros([1, C], gl.float32, layout)
        for i in gl.static_range(len(PW)):
            for j in gl.static_range(min(PW[i], NCH * C - PS[i]) // C):
                mc = p[PS[i] // C + j]
                if NUM_BLOCKS > 0:
                    v = vs[PS[i] // C + j]
                    mc = mc * e_p
                    for r in gl.static_range(NUM_BLOCKS):  # lane-local FMA chain over the rows
                        mc = gl.fma(gl.amd.slice(v, [1, C], [r, 0]), gl.amd.slice(e_b, [1, 1], [r, 0]), mc)
                if APPLY_OUTPUT_NORM:
                    sq_o = gl.fma(mc, mc, sq_o)
                mixed += (mc,)

        if APPLY_OUTPUT_NORM:
            rstd = gl.rsqrt(gl.sum(sq_o, axis=1, keep_dims=True) * (1.0 / HIDDEN) + output_norm_eps)
        out = ()
        amax = gl.zeros([1, C], gl.float32, layout)
        for cq in gl.static_range(NCH):
            oc = mixed[cq]
            if APPLY_OUTPUT_NORM:
                oc = oc * rstd * ow[cq]
            if QUANT_MAX > 0:
                # Round to bf16 first, like the unquantized output would be.
                oc = oc.to(dtype).to(gl.float32)
                amax = gl.maximum(amax, gl.abs(oc))
            out += (oc,)

        if QUANT_MAX > 0:
            scale = gl.maximum(gl.max(amax, axis=1, keep_dims=True), 1e-10) * (1.0 / QUANT_MAX)
            rscale = 1.0 / scale
            gl.store(output_scale_ptr + tok64 + cols, scale + gl.zeros([1, C], gl.float32, layout), mask=cols == 0)
        for cq in gl.static_range(NCH):
            cc = cq * C + cols
            oc = out[cq]
            if QUANT_MAX > 0:
                oc = gl.minimum(gl.maximum(oc * rscale, -QUANT_MAX), QUANT_MAX)
            gl.store(output_ptr + tok64 * stride_output_m + cc, oc.to(output_ptr.dtype.element_ty), mask=cc < HIDDEN)


@gluon.jit
def _issue_consume(
    prefix_ptr,
    delta_ptr,
    mix_ptr,
    T,
    stride_prefix_m,
    stride_delta_m,
    stride_mix_m,
    pbuf,
    dbuf,
    mbuf,
    tok,
    slot,
    HIDDEN: gl.constexpr,
    HAS_DELTA: gl.constexpr,
    smem_layout: gl.constexpr,
    PBD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    pd = gl.amd.gfx1250.tdm.make_tensor_descriptor(prefix_ptr, [T, HIDDEN], [stride_prefix_m, 1], [1, PBD], smem_layout)
    pd = gl.amd.gfx1250.tdm.update_tensor_descriptor(pd, add_offsets=[tok, 0])
    md = gl.amd.gfx1250.tdm.make_tensor_descriptor(mix_ptr, [T, HIDDEN], [stride_mix_m, 1], [1, PBD], smem_layout)
    md = gl.amd.gfx1250.tdm.update_tensor_descriptor(md, add_offsets=[tok, 0])
    Q: gl.constexpr = NUM_WARPS // 4
    if HAS_DELTA:
        dd = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            delta_ptr, [T, HIDDEN], [stride_delta_m, 1], [1, PBD], smem_layout
        )
        dd = gl.amd.gfx1250.tdm.update_tensor_descriptor(dd, add_offsets=[tok, 0])
        gl.amd.gfx1250.tdm.async_load_fused(
            [
                (pd, pbuf[0].index(slot), (1 << Q) - 1),
                (dd, dbuf[0].index(slot), ((1 << Q) - 1) << Q),
                (md, mbuf[0].index(slot), ((1 << (2 * Q)) - 1) << (2 * Q)),
            ]
        )
    else:
        gl.amd.gfx1250.tdm.async_load_fused(
            [(pd, pbuf[0].index(slot), (1 << (2 * Q)) - 1), (md, mbuf[0].index(slot), ((1 << (2 * Q)) - 1) << (2 * Q))]
        )


@gluon.jit
def _attn_res_consume_kernel(
    prefix_ptr,
    delta_ptr,
    blocks_ptr,
    mix_ptr,
    stats_ptr,
    norm_weight_ptr,
    qk_weight_ptr,
    output_norm_weight_ptr,
    output_ptr,
    output_scale_ptr,
    T,
    stride_prefix_m,
    stride_delta_m,
    stride_block_m,
    stride_mix_m,
    stride_stats_m,
    stride_output_m,
    eps,
    output_norm_eps,
    HIDDEN: gl.constexpr,
    BLOCK_WRITE_IDX: gl.constexpr,
    WRITE_BLOCK: gl.constexpr,
    HAS_DELTA: gl.constexpr,
    STORE_PREFIX: gl.constexpr,
    APPLY_OUTPUT_NORM: gl.constexpr,
    QUANT_MAX: gl.constexpr,
    EPL: gl.constexpr,
    C: gl.constexpr,
    NCH: gl.constexpr,
    PS: gl.constexpr,
    PW: gl.constexpr,
    NSLOT: gl.constexpr,
    PBD: gl.constexpr,  # padded row tile width (power of two >= HIDDEN) for prefix, delta and mixture
    NUM_WARPS: gl.constexpr,
):
    pid = gl.program_id(0)
    grid = gl.num_programs(0)
    n_tok = gl.cdiv(T - pid, grid)

    L: gl.constexpr = 1  # TDM ops per token per wave (fused load)
    layout: gl.constexpr = gl.BlockedLayout([1, EPL], [1, 32], [1, NUM_WARPS], [1, 0])
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    dtype: gl.constexpr = prefix_ptr.dtype.element_ty
    pbuf = ()
    dbuf = ()
    mbuf = ()
    pbuf += (gl.allocate_shared_memory(dtype, [NSLOT, 1, PBD], smem_layout),)
    if HAS_DELTA:
        dbuf += (gl.allocate_shared_memory(dtype, [NSLOT, 1, PBD], smem_layout),)

    mbuf = (gl.allocate_shared_memory(mix_ptr.dtype.element_ty, [NSLOT, 1, PBD], smem_layout),)
    if not HAS_DELTA:
        dbuf = pbuf
    for s in gl.static_range(NSLOT):
        if s < n_tok:
            _issue_consume(
                prefix_ptr,
                delta_ptr,
                mix_ptr,
                T,
                stride_prefix_m,
                stride_delta_m,
                stride_mix_m,
                pbuf,
                dbuf,
                mbuf,
                pid + s * grid,
                s,
                HIDDEN,
                HAS_DELTA,
                smem_layout,
                PBD,
                NUM_WARPS,
            )
    tok0 = gl.minimum(pid, T - 1).to(gl.int64)
    cols = gl.arange(0, C, layout=gl.SliceLayout(0, layout))[None, :]
    zc = cols * 0  # stats as per-lane (VGPR) loads: scalar loads spilled SGPRs
    max_n = gl.load(stats_ptr + tok0 * stride_stats_m + zc)
    den_n = gl.load(stats_ptr + tok0 * stride_stats_m + 1 + zc)
    w = ()
    ow = ()
    for c in gl.static_range(NCH):
        co = c * C + cols
        w += (
            gl.load(norm_weight_ptr + co, mask=co < HIDDEN, other=0.0).to(gl.float32)
            * gl.load(qk_weight_ptr + co, mask=co < HIDDEN, other=0.0).to(gl.float32),
        )
        if APPLY_OUTPUT_NORM:
            ow += (gl.load(output_norm_weight_ptr + co, mask=co < HIDDEN, other=0.0).to(gl.float32),)

    o_prev = ()
    for c in gl.static_range(NCH):
        o_prev += (gl.zeros([1, C], gl.float32, layout).to(output_ptr.dtype.element_ty),)
    tok_prev = pid.to(gl.int64)
    for k in range(n_tok):
        tok64 = (pid + k * grid).to(gl.int64)
        slot = k % NSLOT
        if k + NSLOT - 1 < n_tok:
            gl.amd.gfx1250.tdm.async_wait((NSLOT - 1) * L)
        else:
            gl.amd.gfx1250.tdm.async_wait(0)
        gl.barrier()  # every wave reads tiles issued by other waves
        if k > 0:  # token k - 1's output row, after the barrier so the barrier never waits for it
            for c in gl.static_range(NCH):
                gl.store(output_ptr + tok_prev * stride_output_m + c * C + cols, o_prev[c], cache_modifier=".cs")
        max_b = max_n
        den_b = den_n

        p = ()
        mraw = ()
        dot_p = gl.zeros([1, C], gl.float32, layout)
        sq_p = gl.zeros([1, C], gl.float32, layout)
        for i in gl.static_range(len(PW)):
            for j in gl.static_range(PW[i] // C):
                pc = pbuf[0].index(slot).slice(PS[i] + j * C, C, 1).load(layout)
                if HAS_DELTA:
                    dc = dbuf[0].index(slot).slice(PS[i] + j * C, C, 1).load(layout)
                    pc = (pc.to(gl.float32) + dc.to(gl.float32)).to(dtype)
                pc = pc.to(gl.float32)
                p += (pc,)
                mraw += (mbuf[0].index(slot).slice(PS[i] + j * C, C, 1).load(layout).to(gl.float32),)
                dot_p = gl.fma(pc, w[PS[i] // C + j], dot_p)
                sq_p = gl.fma(pc, pc, sq_p)
        gl.barrier()  # every wave is done with this slot: refill it now
        if k + NSLOT < n_tok:
            _issue_consume(
                prefix_ptr,
                delta_ptr,
                mix_ptr,
                T,
                stride_prefix_m,
                stride_delta_m,
                stride_mix_m,
                pbuf,
                dbuf,
                mbuf,
                pid + (k + NSLOT) * grid,
                slot,
                HIDDEN,
                HAS_DELTA,
                smem_layout,
                PBD,
                NUM_WARPS,
            )
        tokn = gl.minimum(pid + (k + 1) * grid, T - 1).to(gl.int64)
        max_n = gl.load(stats_ptr + tokn * stride_stats_m + zc)
        den_n = gl.load(stats_ptr + tokn * stride_stats_m + 1 + zc)
        if HAS_DELTA:
            for c in gl.static_range(NCH):
                cc = c * C + cols
                if STORE_PREFIX:
                    gl.store(prefix_ptr + tok64 * stride_prefix_m + cc, p[c].to(dtype), cache_modifier=".cs")
                if WRITE_BLOCK:
                    gl.store(blocks_ptr + tok64 * stride_block_m + BLOCK_WRITE_IDX * HIDDEN + cc, p[c].to(dtype))
        elif WRITE_BLOCK:
            for c in gl.static_range(NCH):
                cc = c * C + cols
                gl.store(blocks_ptr + tok64 * stride_block_m + BLOCK_WRITE_IDX * HIDDEN + cc, p[c].to(dtype))

        d_p, q_p = gl.split(gl.sum(gl.join(dot_p, sq_p), axis=1))
        d_p = gl.convert_layout(d_p, gl.SliceLayout(1, layout))[:, None]
        q_p = gl.convert_layout(q_p, gl.SliceLayout(1, layout))[:, None]
        logit_p = d_p * gl.rsqrt(q_p * (1.0 / HIDDEN) + eps)
        m = gl.maximum(logit_p, max_b)
        e_b = gl.exp(max_b - m)
        e_p = gl.exp(logit_p - m)
        rden = 1.0 / (den_b * e_b + e_p)
        e_b = e_b * rden
        e_p = e_p * rden

        mixed = ()
        sq_o = gl.zeros([1, C], gl.float32, layout)
        for i in gl.static_range(len(PW)):
            for j in gl.static_range(PW[i] // C):
                mc = gl.fma(mraw[PS[i] // C + j], e_b, p[PS[i] // C + j] * e_p)
                if APPLY_OUTPUT_NORM:
                    sq_o = gl.fma(mc, mc, sq_o)
                mixed += (mc,)

        if APPLY_OUTPUT_NORM:
            rstd = gl.rsqrt(gl.sum(sq_o, axis=1, keep_dims=True) * (1.0 / HIDDEN) + output_norm_eps)
        out = ()
        amax = gl.zeros([1, C], gl.float32, layout)
        for c in gl.static_range(NCH):
            oc = mixed[c]
            if APPLY_OUTPUT_NORM:
                oc = oc * rstd * ow[c]
            if QUANT_MAX > 0:
                oc = oc.to(dtype).to(gl.float32)
                amax = gl.maximum(amax, gl.abs(oc))
            out += (oc,)

        if QUANT_MAX > 0:
            scale = gl.maximum(gl.max(amax, axis=1, keep_dims=True), 1e-10) * (1.0 / QUANT_MAX)
            rscale = 1.0 / scale
            gl.store(output_scale_ptr + tok64 + cols, scale + gl.zeros([1, C], gl.float32, layout), mask=cols == 0)
        o_nxt = ()
        for c in gl.static_range(NCH):
            oc = out[c]
            if QUANT_MAX > 0:
                oc = gl.minimum(gl.maximum(oc * rscale, -QUANT_MAX), QUANT_MAX)
            o_nxt += (oc.to(output_ptr.dtype.element_ty),)
        o_prev = o_nxt
        tok_prev = tok64
    if n_tok > 0:
        for c in gl.static_range(NCH):
            gl.store(output_ptr + tok_prev * stride_output_m + c * C + cols, o_prev[c], cache_modifier=".cs")


@gluon.jit
def _attn_res_logits_kernel(
    blocks_ptr,
    wcat_ptr,
    plog_ptr,
    T,
    K,
    stride_block_m,
    HIDDEN: gl.constexpr,
    NUM_BLOCKS: gl.constexpr,
    S: gl.constexpr,
    SW: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SP: gl.constexpr,
    NA: gl.constexpr = 3,
):
    # CTA = (column slice sl of width SW, persistent over tiles of TT tokens). W = W_hi + W_lo exactly in bf16 (w has
    # <= 16 significant bits); the slice's B operands live in registers. A = TDM tile [TT tokens, 8 rows (>= n zero
    # fill), SW] in an NA-slot ring (padded LDS rows: no bank conflicts). plog[t, sl, r, c] = partial logit (fp32).
    pid = gl.program_id(0)
    grid = gl.num_programs(0)
    sl = pid % S
    pt = pid // S
    npt = grid // S
    TT: gl.constexpr = 2 * NUM_WARPS  # tokens per tile: 16 A rows per warp
    MR: gl.constexpr = 8 * TT
    n_tiles = gl.cdiv(T, TT)
    n_it = gl.cdiv(n_tiles - pt, npt)
    if NUM_WARPS == 8:
        WM: gl.constexpr = gl.amd.AMDWMMALayout(3, True, [[1, 0], [2, 0], [4, 0]], [], [16, 16, 32])
    else:
        WM: gl.constexpr = gl.amd.AMDWMMALayout(3, True, [[1, 0], [2, 0]], [], [16, 16, 32])
    AL: gl.constexpr = gl.DotOperandLayout(0, WM, 8)
    BL: gl.constexpr = gl.DotOperandLayout(1, WM, 8)
    smem_l: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    a3_l: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[SW, 8]], [TT, 8, SW], [2, 1, 0])
    a2_l: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[SW, 8]], [MR, SW], [1, 0])
    abuf = gl.allocate_shared_memory(gl.bfloat16, [NA, TT, 8, SW], a3_l)
    wsm = gl.allocate_shared_memory(gl.bfloat16, [2, SW, 32], smem_l)
    osm = gl.allocate_shared_memory(gl.float32, [TT, 8, 32], gl.SwizzledSharedLayout(1, 1, 1, order=[2, 1, 0]))
    adesc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        blocks_ptr, [T, NUM_BLOCKS, HIDDEN], [stride_block_m, HIDDEN, 1], [TT, 8, SW], a3_l
    )

    # prologue: NA - 1 tiles in flight, W slice -> LDS -> operand registers
    for s_ in gl.static_range(NA - 1):
        gl.amd.gfx1250.tdm.async_load(adesc, [gl.minimum(pt + s_ * npt, n_tiles - 1) * TT, 0, sl * SW], abuf.index(s_))
    LW: gl.constexpr = gl.BlockedLayout([8, 1], [4, 8], [1, NUM_WARPS], [0, 1])
    wj = gl.arange(0, SW, layout=gl.SliceLayout(1, LW))[:, None]
    wc = gl.arange(0, 32, layout=gl.SliceLayout(0, LW))[None, :]
    for h in gl.static_range(2):
        wsm.index(h).store(
            gl.load(wcat_ptr + (h * K + wc) * HIDDEN + sl * SW + wj, mask=(wc < K) & (wj >= 0), other=0.0)
        )
    bhi = ()
    blo = ()
    for st in gl.static_range(SW // 32):
        bhi += (wsm.index(0).slice(st * 32, 32, 0).load(BL),)
        blo += (wsm.index(1).slice(st * 32, 32, 0).load(BL),)
    LO: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [NUM_WARPS, 1], [1, 0])
    oq = gl.arange(0, TT, layout=gl.SliceLayout(1, LO))[:, None]
    oe = gl.arange(0, 256, layout=gl.SliceLayout(0, LO))[None, :]

    # main loop: refill (clamped, unconditional -> partial tensorcnt wait), 4 independent WMMA chains, plog store
    for it in range(n_it):
        tile = pt + it * npt
        nt = gl.minimum(tile + (NA - 1) * npt, n_tiles - 1)
        gl.amd.gfx1250.tdm.async_load(adesc, [nt * TT, 0, sl * SW], abuf.index((it + NA - 1) % NA))
        gl.amd.gfx1250.tdm.async_wait(NA - 1)
        a2 = abuf.index(it % NA).reinterpret(gl.bfloat16, [MR, SW], a2_l)
        acc0 = gl.zeros([MR, 32], gl.float32, WM)
        acc1 = gl.zeros([MR, 32], gl.float32, WM)
        acc2 = gl.zeros([MR, 32], gl.float32, WM)
        acc3 = gl.zeros([MR, 32], gl.float32, WM)
        for st in gl.static_range(SW // 32):
            a_ = a2.slice(st * 32, 32, 1).load(AL)
            if st % 2 == 0:
                acc0 = gl.amd.gfx1250.wmma(a_, bhi[st], acc0)
                acc1 = gl.amd.gfx1250.wmma(a_, blo[st], acc1)
            else:
                acc2 = gl.amd.gfx1250.wmma(a_, bhi[st], acc2)
                acc3 = gl.amd.gfx1250.wmma(a_, blo[st], acc3)
        # [MR, 32] -> LDS [TT, 8, 32] -> b128 rows of plog (rows >= n not written)
        osm.store(gl.reshape((acc0 + acc2) + (acc1 + acc3), [TT, 8, 32]))
        ov = osm.reinterpret(gl.float32, [TT, 256], smem_l).load(LO)
        otk = tile * TT + oq
        gl.store(plog_ptr + otk.to(gl.int64) * (SP * 256) + sl * 256 + oe, ov, mask=(otk < T) & (oe < NUM_BLOCKS * 32))

    # epilogue
    gl.amd.gfx1250.tdm.async_wait(0)
