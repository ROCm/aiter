# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL prefill causal-conv1d kernel with fused split q/k/v output."""

from __future__ import annotations

import functools

import torch

try:
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import arith, rocdl
    from flydsl.expr.arith import CmpIPredicate
    from flydsl.expr.typing import T, Int32
    from flydsl._mlir import ir
    from flydsl.expr import buffer_ops
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.runtime.device import get_rocm_arch
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

    _FLYDSL_AVAILABLE = True
except Exception:  # pragma: no cover - flydsl optional
    _FLYDSL_AVAILABLE = False


PAD_SLOT_ID = -1
_LOG2E = 1.4426950408889634


def is_flydsl_available() -> bool:
    return _FLYDSL_AVAILABLE


def build_causal_conv1d_flydsl_module(
    width: int,
    has_bias: bool,
    silu: bool,
    tm: int = 64,
    tn: int = 64,
    block_threads: int = 256,
    dtype_str: str = "bf16",
):
    """Build the FlyDSL causal conv1d kernel for the given config."""
    assert _FLYDSL_AVAILABLE, "flydsl is not installed"
    assert width in (2, 3, 4)
    assert (
        tm == 64 and tn == 64 and block_threads == 256
    ), "fixed TM=TN=64, 256-thread tile"

    W = width
    KW = W
    SL = W - 1
    TM, TN, BT = tm, tn, block_threads
    LDS_PAD = TM + KW  # halo(KW-1) + body(TM) + pad(1)
    EPT = TM // 4  # outputs per thread (4 token groups)
    FG = BT // TM  # feat-base groups in cooperative load (=4)
    ELEMS = TN * TM // BT  # body features loaded per thread (=16)
    LOG2_TM = TM.bit_length() - 1  # =6
    NLDS = TN * LDS_PAD
    STORE_PAD = TN + 1
    LDS_BYTES = NLDS * 2  # bf16 staging
    HAS_BIAS = bool(has_bias)
    SILU = bool(silu)

    arch = get_rocm_arch()
    allocator = SmemAllocator(
        None,
        arch=arch,
        global_sym_name=f"causal_conv1d_w{W}_tm{TM}_{dtype_str}",
    )
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + LDS_BYTES

    @flyc.kernel
    def conv1d_kernel(
        x_ptr: fx.Tensor,
        w_ptr: fx.Tensor,
        bias_ptr: fx.Tensor,
        cs_ptr: fx.Tensor,
        cache_idx_ptr: fx.Tensor,
        has_init_ptr: fx.Tensor,
        qsl_ptr: fx.Tensor,
        batch_ptr: fx.Tensor,
        chunk_off_ptr: fx.Tensor,
        q_ptr: fx.Tensor,
        k_ptr: fx.Tensor,
        v_ptr: fx.Tensor,
        dim: Int32,
        kd: Int32,
        vd: Int32,
        sx0: Int32,
        sx1: Int32,
        sw0: Int32,
        sw1: Int32,
        scs0: Int32,
        scs1: Int32,
        scs2: Int32,
        sci: Int32,
        qs0: Int32,
        qs1: Int32,
        ks0: Int32,
        ks1: Int32,
        vs0: Int32,
        vs1: Int32,
    ):
        i32 = T.i32
        elem_dtype = T.bf16 if dtype_str == "bf16" else T.f16

        def _v(x):
            return x.ir_value() if hasattr(x, "ir_value") else x

        dim = _v(dim)
        kd = _v(kd)
        vd = _v(vd)
        sx0 = _v(sx0)
        sx1 = _v(sx1)
        sw0 = _v(sw0)
        sw1 = _v(sw1)
        scs0 = _v(scs0)
        scs1 = _v(scs1)
        scs2 = _v(scs2)
        sci = _v(sci)
        qs0 = _v(qs0)
        qs1 = _v(qs1)
        ks0 = _v(ks0)
        ks1 = _v(ks1)
        vs0 = _v(vs0)
        vs1 = _v(vs1)

        def c32(v):
            return arith.constant(int(v), type=i32)

        def cf(v):
            return arith.constant(float(v), type=T.f32)

        def to_i32(v):
            return arith.index_cast(i32, v)

        def mul(a, b):
            return arith.muli(a, b)

        def add(a, b):
            return arith.addi(a, b)

        def sub(a, b):
            return arith.subi(a, b)

        def f32(bf):
            return arith.extf(T.f32, bf)

        def _rsrc(ptr):
            return buffer_ops.create_buffer_resource(ptr, max_size=True)

        x_r = _rsrc(x_ptr)
        w_r = _rsrc(w_ptr)
        b_r = _rsrc(bias_ptr)
        cs_r = _rsrc(cs_ptr)
        ci_r = _rsrc(cache_idx_ptr)
        hi_r = _rsrc(has_init_ptr)
        qsl_r = _rsrc(qsl_ptr)
        batch_r = _rsrc(batch_ptr)
        choff_r = _rsrc(chunk_off_ptr)
        q_r = _rsrc(q_ptr)
        k_r = _rsrc(k_ptr)
        v_r = _rsrc(v_ptr)

        lds = SmemPtr(allocator.get_base(), lds_off, elem_dtype, shape=(NLDS,))
        lds.get()

        def lds_st(val, idx):
            lds.store(val, [idx])

        def lds_ld(idx):
            return lds.load([idx])

        tid = to_i32(fx.thread_idx.x)
        pid_x = to_i32(fx.block_idx.x)
        pid_y = to_i32(fx.block_idx.y)

        seq_idx = buffer_ops.buffer_load(batch_r, pid_x, vec_width=1, dtype=i32)
        chunk_idx = buffer_ops.buffer_load(choff_r, pid_x, vec_width=1, dtype=i32)
        seq_start = buffer_ops.buffer_load(qsl_r, seq_idx, vec_width=1, dtype=i32)
        seq_end = buffer_ops.buffer_load(
            qsl_r, add(seq_idx, c32(1)), vec_width=1, dtype=i32
        )
        seqlen = sub(seq_end, seq_start)

        feat_start = mul(pid_y, c32(TN))
        tok_start = mul(chunk_idx, c32(TM))
        is_chunk0 = arith.cmpi(CmpIPredicate.eq, chunk_idx, c32(0))

        feat_local = arith.shrui(tid, c32(2))
        tok_group = arith.andi(tid, c32(3))
        tok_base = mul(tok_group, c32(EPT))
        gfeat = add(feat_start, feat_local)
        feat_valid = arith.cmpi(CmpIPredicate.slt, gfeat, dim)

        # weights + bias (fp32)
        w_base = mul(gfeat, sw0)
        w_taps = []
        for j in fx.range_constexpr(W):
            w_taps.append(
                f32(
                    buffer_ops.buffer_load(
                        w_r,
                        add(w_base, mul(c32(j), sw1)),
                        vec_width=1,
                        dtype=elem_dtype,
                    )
                )
            )
        if fx.const_expr(HAS_BIAS):
            bias_f = f32(
                buffer_ops.buffer_load(b_r, gfeat, vec_width=1, dtype=elem_dtype)
            )
        else:
            bias_f = cf(0.0)

        # cooperative load into staging buffer
        t_const = arith.andi(tid, c32(TM - 1))
        f_base = arith.shrui(tid, c32(LOG2_TM))
        hc = arith.shrui(tid, c32(6))
        hf = arith.andi(tid, c32(63))
        tok_gbase = add(sub(add(seq_start, tok_start), c32(KW - 1)), c32(0))
        gt1 = add(tok_gbase, add(t_const, c32(KW - 1)))

        all_feat = arith.cmpi(CmpIPredicate.sle, add(feat_start, c32(TN)), dim)
        all_tok1 = arith.cmpi(CmpIPredicate.slt, add(tok_start, c32(TM - 1)), seqlen)
        all_tok2 = arith.cmpi(CmpIPredicate.sge, tok_start, c32(KW - 1))
        fast = arith.andi(arith.andi(all_feat, all_tok1), all_tok2)

        if fast:
            # fast path: fully interior, coalesced, no bounds/state
            cur = add(mul(add(feat_start, f_base), sx0), gt1)
            fstep = mul(c32(FG), sx0)
            raws = []
            for j in fx.range_constexpr(ELEMS):
                raws.append(
                    buffer_ops.buffer_load(x_r, cur, vec_width=1, dtype=elem_dtype)
                )
                if fx.const_expr(j + 1 < ELEMS):
                    cur = add(cur, fstep)
            do_halo = arith.cmpi(CmpIPredicate.slt, hc, c32(KW - 1))
            prefix_off = arith.select(
                do_halo, add(mul(add(feat_start, hf), sx0), add(tok_gbase, hc)), c32(0)
            )
            prefix_v = buffer_ops.buffer_load(
                x_r, prefix_off, vec_width=1, dtype=elem_dtype
            )
            lds_idx = add(mul(f_base, c32(LDS_PAD)), add(t_const, c32(KW - 1)))
            for j in fx.range_constexpr(ELEMS):
                cur_idx = lds_idx if j == 0 else add(lds_idx, c32(j * FG * LDS_PAD))
                lds_st(raws[j], cur_idx)
            if do_halo:
                lds_st(prefix_v, add(mul(hf, c32(LDS_PAD)), hc))
        else:
            # slow path: sequence-relative bounds (still coalesced)
            zero_e = arith.constant(0.0, type=elem_dtype)
            body_wp = add(tok_start, t_const)
            body_ok = arith.cmpi(CmpIPredicate.slt, body_wp, seqlen)
            sl_m1 = arith.select(
                arith.cmpi(CmpIPredicate.sgt, seqlen, c32(0)),
                sub(seqlen, c32(1)),
                c32(0),
            )
            body_gt = add(seq_start, arith.select(body_ok, body_wp, sl_m1))
            for j in fx.range_constexpr(ELEMS):
                gf = add(add(feat_start, f_base), c32(j * FG))
                gf_ok = arith.cmpi(CmpIPredicate.slt, gf, dim)
                safe_gf = arith.select(gf_ok, gf, c32(0))
                raw = buffer_ops.buffer_load(
                    x_r, add(mul(safe_gf, sx0), body_gt), vec_width=1, dtype=elem_dtype
                )
                val = arith.select(arith.andi(body_ok, gf_ok), raw, zero_e)
                lds_st(
                    val,
                    add(
                        mul(add(f_base, c32(j * FG)), c32(LDS_PAD)),
                        add(t_const, c32(KW - 1)),
                    ),
                )
            # halo column with conv_state blend at chunk0
            do_halo = arith.cmpi(CmpIPredicate.slt, hc, c32(KW - 1))
            if do_halo:
                gf = add(feat_start, hf)
                gf_ok = arith.cmpi(CmpIPredicate.slt, gf, dim)
                wp = sub(add(tok_start, hc), c32(KW - 1))
                wp_in = arith.andi(
                    arith.cmpi(CmpIPredicate.sge, wp, c32(0)),
                    arith.cmpi(CmpIPredicate.slt, wp, seqlen),
                )
                safe_xoff = arith.select(
                    arith.andi(wp_in, gf_ok),
                    add(mul(gf, sx0), add(seq_start, wp)),
                    c32(0),
                )
                xv = arith.select(
                    arith.andi(wp_in, gf_ok),
                    buffer_ops.buffer_load(
                        x_r, safe_xoff, vec_width=1, dtype=elem_dtype
                    ),
                    zero_e,
                )
                # pre-seq source: conv_state at chunk0
                hi8 = buffer_ops.buffer_load(hi_r, seq_idx, vec_width=1, dtype=T.i8)
                hi_nz = arith.cmpi(CmpIPredicate.ne, hi8, arith.constant(0, type=T.i8))
                need_cs = arith.andi(
                    arith.andi(arith.cmpi(CmpIPredicate.slt, wp, c32(0)), is_chunk0),
                    arith.andi(hi_nz, gf_ok),
                )
                in_coord = buffer_ops.buffer_load(
                    ci_r, mul(seq_idx, sci), vec_width=1, dtype=i32
                )
                slot = add(c32(KW - 1), wp)
                cs_off = arith.select(
                    need_cs,
                    add(add(mul(in_coord, scs0), mul(gf, scs1)), mul(slot, scs2)),
                    c32(0),
                )
                csv = buffer_ops.buffer_load(
                    cs_r, cs_off, vec_width=1, dtype=elem_dtype
                )
                hv = arith.select(need_cs, csv, xv)
                lds_st(hv, add(mul(hf, c32(LDS_PAD)), hc))

        fx.gpu.barrier()

        # compute: acc[e] = bias + sum_k w[k] * x[...]; the EPT outputs share a
        # contiguous window, loaded once into registers before the MAC.
        row_base = add(mul(feat_local, c32(LDS_PAD)), tok_base)
        NSPAN = EPT + W - 1
        xw = []
        for i in fx.range_constexpr(NSPAN):
            idx = row_base if i == 0 else add(row_base, c32(i))
            xw.append(f32(lds_ld(idx)))
        acc = []
        for e in fx.range_constexpr(EPT):
            a = bias_f
            for kk in fx.range_constexpr(W):
                a = arith.addf(a, arith.mulf(w_taps[kk], xw[e + kk]))
            if fx.const_expr(SILU):
                ex = rocdl.exp2(T.f32, arith.mulf(a, cf(-_LOG2E)))
                a = arith.mulf(a, rocdl.rcp(T.f32, arith.addf(cf(1.0), ex)))
            acc.append(a)

        # store: transpose through staging (fast) or direct (slow)
        store_fast = arith.andi(
            arith.cmpi(CmpIPredicate.sle, add(feat_start, c32(TN)), dim),
            arith.cmpi(CmpIPredicate.slt, add(tok_start, c32(TM - 1)), seqlen),
        )
        vstart = mul(kd, c32(2))
        blk_q = arith.cmpi(CmpIPredicate.sle, add(feat_start, c32(TN)), kd)
        blk_k = arith.andi(
            arith.cmpi(CmpIPredicate.sge, feat_start, kd),
            arith.cmpi(CmpIPredicate.sle, add(feat_start, c32(TN)), vstart),
        )
        blk_v = arith.cmpi(CmpIPredicate.sge, feat_start, vstart)

        if store_fast:
            fx.gpu.barrier()
            for e in fx.range_constexpr(EPT):
                lds_st(
                    arith.truncf(elem_dtype, acc[e]),
                    add(mul(add(tok_base, c32(e)), c32(STORE_PAD)), feat_local),
                )
            fx.gpu.barrier()
            sf = arith.andi(tid, c32(TN - 1))
            tg = arith.shrui(tid, c32(6))
            tg_ept = mul(tg, c32(EPT))
            tok0 = add(add(seq_start, tok_start), tg_ept)

            def emit_fast(cond, res, ts, ds, fo):
                if cond:
                    of = sub(add(feat_start, sf), fo)
                    base_off = add(mul(tok0, ts), mul(of, ds))
                    cur = base_off
                    for e in fx.range_constexpr(EPT):
                        val = lds_ld(add(mul(add(tg_ept, c32(e)), c32(STORE_PAD)), sf))
                        buffer_ops.buffer_store(val, res, cur)
                        if fx.const_expr(e + 1 < EPT):
                            cur = add(cur, ts)

            emit_fast(blk_q, q_r, qs0, qs1, c32(0))
            emit_fast(blk_k, k_r, ks0, ks1, kd)
            emit_fast(blk_v, v_r, vs0, vs1, vstart)
        else:

            def emit_slow(cond, res, ts, ds, fo):
                if arith.andi(cond, feat_valid):
                    of = sub(gfeat, fo)
                    base_off = add(
                        mul(add(add(seq_start, tok_start), tok_base), ts), mul(of, ds)
                    )
                    cur = base_off
                    for e in fx.range_constexpr(EPT):
                        tok_ok = arith.cmpi(
                            CmpIPredicate.slt,
                            add(add(tok_start, tok_base), c32(e)),
                            seqlen,
                        )
                        if tok_ok:
                            buffer_ops.buffer_store(
                                arith.truncf(elem_dtype, acc[e]), res, cur
                            )
                        if fx.const_expr(e + 1 < EPT):
                            cur = add(cur, ts)

            emit_slow(blk_q, q_r, qs0, qs1, c32(0))
            emit_slow(blk_k, k_r, ks0, ks1, kd)
            emit_slow(blk_v, v_r, vs0, vs1, vstart)

        # conv_state writeback (chunk 0)
        if fx.const_expr(SL > 0):
            if is_chunk0:
                zero_e = arith.constant(0.0, type=elem_dtype)
                slot = tok_group
                should = arith.andi(
                    arith.cmpi(CmpIPredicate.slt, slot, c32(KW - 1)),
                    arith.cmpi(CmpIPredicate.slt, gfeat, dim),
                )
                if should:
                    in_coord = buffer_ops.buffer_load(
                        ci_r, mul(seq_idx, sci), vec_width=1, dtype=i32
                    )
                    pos_x = add(sub(seqlen, c32(KW - 1)), slot)
                    x_in = arith.cmpi(CmpIPredicate.sge, pos_x, c32(0))
                    safe_x = arith.select(
                        x_in, add(mul(gfeat, sx0), add(seq_start, pos_x)), c32(0)
                    )
                    val_x = buffer_ops.buffer_load(
                        x_r, safe_x, vec_width=1, dtype=elem_dtype
                    )
                    hi8 = buffer_ops.buffer_load(hi_r, seq_idx, vec_width=1, dtype=T.i8)
                    hi_nz = arith.cmpi(
                        CmpIPredicate.ne, hi8, arith.constant(0, type=T.i8)
                    )
                    need_pr = arith.andi(
                        arith.cmpi(CmpIPredicate.slt, pos_x, c32(0)), hi_nz
                    )
                    src = add(slot, seqlen)
                    safe_pr = arith.select(
                        need_pr,
                        add(add(mul(in_coord, scs0), mul(gfeat, scs1)), mul(src, scs2)),
                        c32(0),
                    )
                    val_pr = buffer_ops.buffer_load(
                        cs_r, safe_pr, vec_width=1, dtype=elem_dtype
                    )
                    wb_val = arith.select(
                        x_in, val_x, arith.select(need_pr, val_pr, zero_e)
                    )
                    cs_wr = add(
                        add(mul(in_coord, scs0), mul(gfeat, scs1)), mul(slot, scs2)
                    )
                    buffer_ops.buffer_store(wb_val, cs_r, cs_wr)

    @flyc.jit
    def launch(
        x_ptr: fx.Tensor,
        w_ptr: fx.Tensor,
        bias_ptr: fx.Tensor,
        cs_ptr: fx.Tensor,
        cache_idx_ptr: fx.Tensor,
        has_init_ptr: fx.Tensor,
        qsl_ptr: fx.Tensor,
        batch_ptr: fx.Tensor,
        chunk_off_ptr: fx.Tensor,
        q_ptr: fx.Tensor,
        k_ptr: fx.Tensor,
        v_ptr: fx.Tensor,
        dim: Int32,
        kd: Int32,
        vd: Int32,
        sx0: Int32,
        sx1: Int32,
        sw0: Int32,
        sw1: Int32,
        scs0: Int32,
        scs1: Int32,
        scs2: Int32,
        sci: Int32,
        qs0: Int32,
        qs1: Int32,
        ks0: Int32,
        ks1: Int32,
        vs0: Int32,
        vs1: Int32,
        num_programs: Int32,
        grid_y_dim: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        gx = arith.index_cast(T.index, num_programs)
        gy = arith.index_cast(T.index, grid_y_dim)
        conv1d_kernel(
            x_ptr,
            w_ptr,
            bias_ptr,
            cs_ptr,
            cache_idx_ptr,
            has_init_ptr,
            qsl_ptr,
            batch_ptr,
            chunk_off_ptr,
            q_ptr,
            k_ptr,
            v_ptr,
            dim,
            kd,
            vd,
            sx0,
            sx1,
            sw0,
            sw1,
            scs0,
            scs1,
            scs2,
            sci,
            qs0,
            qs1,
            ks0,
            ks1,
            vs0,
            vs1,
        ).launch(grid=(gx, gy, 1), block=(BT, 1, 1), stream=stream)

    launch._tn = TN
    launch._tm = TM
    return launch


@functools.lru_cache(maxsize=None)
def _get_compiled(width, has_bias, silu, tm, tn, block_threads, dtype_str):
    return build_causal_conv1d_flydsl_module(
        width, has_bias, silu, tm, tn, block_threads, dtype_str
    )


def _build_chunk_metadata(query_start_loc: torch.Tensor, block_m: int):
    """Build (num_programs, batch_ptr, token_chunk_offset_ptr) like the Triton wrapper."""
    device = query_start_loc.device
    seqlens = query_start_loc.diff().to("cpu")
    nums = (-(-seqlens // block_m)).to(torch.int64)  # ceil-div per sequence
    n_seqs = nums.numel()
    tot = int(nums.sum().item())
    if tot == 0:
        z = torch.zeros(0, dtype=torch.int32, device=device)
        return 0, z, z
    seq_ids = torch.arange(n_seqs, dtype=torch.int32)
    batch_ptr = torch.repeat_interleave(seq_ids, nums)
    starts = nums.cumsum(0) - nums  # exclusive prefix sum
    base = torch.repeat_interleave(starts, nums)
    tco = (torch.arange(tot, dtype=torch.int64) - base).to(torch.int32)
    return tot, batch_ptr.to(device), tco.to(device)


def causal_conv1d_split_qkv_flydsl_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    k_dim_size: int,
    v_dim_size: int,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    block_m: int = 64,
    **kwargs,
):
    """FlyDSL prefill causal conv1d with fused split q/k/v. Returns (q, k, v)."""
    if x.dtype != conv_states.dtype:  # avoid no-op .to() dispatch on the hot path
        x = x.to(conv_states.dtype)
    dim, cu_seqlen = x.shape
    _, width = weight.shape
    silu = activation in ("silu", "swish")

    if cache_indices is None:
        cache_indices = torch.arange(
            query_start_loc.numel() - 1, dtype=torch.int32, device=x.device
        )
    if has_initial_state is None:
        has_initial_state = torch.zeros(
            query_start_loc.numel() - 1, dtype=torch.bool, device=x.device
        )

    # chunk schedule (reuse harness metadata when provided)
    if (
        metadata is not None
        and hasattr(metadata, "nums_dict")
        and block_m in metadata.nums_dict
    ):
        entry = metadata.nums_dict[block_m]
        tot = int(entry["tot"])
        batch_ptr = entry["batch_ptr"]
        chunk_off_ptr = entry["token_chunk_offset_ptr"]
        if batch_ptr.device != x.device:
            batch_ptr = batch_ptr.to(x.device)
            chunk_off_ptr = chunk_off_ptr.to(x.device)
    else:
        tot, batch_ptr, chunk_off_ptr = _build_chunk_metadata(query_start_loc, block_m)

    query = torch.empty([cu_seqlen, k_dim_size], dtype=x.dtype, device=x.device)
    key = torch.empty([cu_seqlen, k_dim_size], dtype=x.dtype, device=x.device)
    value = torch.empty([cu_seqlen, v_dim_size], dtype=x.dtype, device=x.device)

    if tot == 0:
        return query, key, value

    dtype_str = "bf16" if x.dtype == torch.bfloat16 else "fp16"
    launcher = _get_compiled(
        int(width), bias is not None, bool(silu), int(block_m), 64, 256, dtype_str
    )
    tn = launcher._tn
    grid_y_dim = (dim + tn - 1) // tn

    bias_arg = bias if bias is not None else x  # dummy ptr when HAS_BIAS=False

    launch_args = (
        x,
        weight,
        bias_arg,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        batch_ptr,
        chunk_off_ptr,
        query,
        key,
        value,
        int(dim),
        int(k_dim_size),
        int(v_dim_size),
        int(x.stride(0)),
        int(x.stride(1)),
        int(weight.stride(0)),
        int(weight.stride(1)),
        int(conv_states.stride(0)),
        int(conv_states.stride(1)),
        int(conv_states.stride(2)),
        int(cache_indices.stride(0)),
        int(query.stride(0)),
        int(query.stride(1)),
        int(key.stride(0)),
        int(key.stride(1)),
        int(value.stride(0)),
        int(value.stride(1)),
        int(tot),
        int(grid_y_dim),
        torch.cuda.current_stream(),
    )

    # First call compiles and executes in one step; later calls reuse the
    # cached CompiledFunction.
    compiled = getattr(launcher, "_fast_compiled", None)
    if compiled is None:
        try:
            launcher._fast_compiled = flyc.compile(launcher, *launch_args)
        except Exception:
            launcher._fast_compiled = False  # fall back permanently
            launcher(*launch_args)
    elif compiled is not False:
        compiled(*launch_args)
    else:
        launcher(*launch_args)
    return query, key, value
