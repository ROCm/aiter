# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL prefill causal-conv1d kernel with fused split q/k/v output."""

import functools

import torch

try:
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import arith
    from flydsl.expr.typing import T, Int32
    from flydsl.expr import buffer_ops

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
    HAS_BIAS = bool(has_bias)
    SILU = bool(silu)

    fx_elem_dtype = fx.BFloat16 if dtype_str == "bf16" else fx.Float16

    @fx.struct
    class SharedStorage:
        lds: fx.Array[fx_elem_dtype, NLDS, 16]

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
        # dtype args for buffer_ops loads (MLIR types, not values)
        i32 = T.i32
        elem_dtype = T.bf16 if dtype_str == "bf16" else T.f16

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

        lds_base = fx.SharedAllocator().allocate(SharedStorage).peek().lds.ptr

        def lds_st(val, idx):
            fx.ptr_store(val, lds_base + fx.Int64(idx))

        def lds_ld(idx):
            return fx.ptr_load(lds_base + fx.Int64(idx))

        tid = fx.thread_idx.x
        pid_x = fx.block_idx.x
        pid_y = fx.block_idx.y

        seq_idx = fx.Int32(
            buffer_ops.buffer_load(batch_r, pid_x, vec_width=1, dtype=i32)
        )
        chunk_idx = fx.Int32(
            buffer_ops.buffer_load(choff_r, pid_x, vec_width=1, dtype=i32)
        )
        seq_start = fx.Int32(
            buffer_ops.buffer_load(qsl_r, seq_idx, vec_width=1, dtype=i32)
        )
        seq_end = fx.Int32(
            buffer_ops.buffer_load(qsl_r, seq_idx + 1, vec_width=1, dtype=i32)
        )
        seqlen = seq_end - seq_start

        feat_start = pid_y * TN
        tok_start = chunk_idx * TM
        is_chunk0 = chunk_idx == 0

        feat_local = tid >> 2
        tok_group = tid & 3
        tok_base = tok_group * EPT
        gfeat = feat_start + feat_local
        feat_valid = gfeat < dim

        # weights + bias (fp32)
        w_base = gfeat * sw0
        w_taps = []
        for j in fx.range_constexpr(W):
            w_taps.append(
                fx.Float32(
                    buffer_ops.buffer_load(
                        w_r,
                        w_base + j * sw1,
                        vec_width=1,
                        dtype=elem_dtype,
                    )
                )
            )
        if fx.const_expr(HAS_BIAS):
            bias_f = fx.Float32(
                buffer_ops.buffer_load(b_r, gfeat, vec_width=1, dtype=elem_dtype)
            )
        else:
            bias_f = fx.Float32(0.0)

        # cooperative load into staging buffer
        t_const = tid & (TM - 1)
        f_base = tid >> LOG2_TM
        hc = tid >> 6
        hf = tid & 63
        tok_gbase = (seq_start + tok_start) - (KW - 1)
        gt1 = tok_gbase + (t_const + (KW - 1))

        all_feat = (feat_start + TN) <= dim
        all_tok1 = (tok_start + (TM - 1)) < seqlen
        all_tok2 = tok_start >= (KW - 1)
        fast = all_feat & all_tok1 & all_tok2

        if fast:
            # fast path: fully interior, coalesced, no bounds/state
            cur = (feat_start + f_base) * sx0 + gt1
            fstep = FG * sx0
            raws = []
            for j in fx.range_constexpr(ELEMS):
                raws.append(
                    fx_elem_dtype(
                        buffer_ops.buffer_load(x_r, cur, vec_width=1, dtype=elem_dtype)
                    )
                )
                if fx.const_expr(j + 1 < ELEMS):
                    cur = cur + fstep
            do_halo = hc < (KW - 1)
            prefix_off = do_halo.select((feat_start + hf) * sx0 + (tok_gbase + hc), 0)
            prefix_v = fx_elem_dtype(
                buffer_ops.buffer_load(x_r, prefix_off, vec_width=1, dtype=elem_dtype)
            )
            lds_idx = f_base * LDS_PAD + (t_const + (KW - 1))
            for j in fx.range_constexpr(ELEMS):
                cur_idx = lds_idx if j == 0 else lds_idx + (j * FG * LDS_PAD)
                lds_st(raws[j], cur_idx)
            if do_halo:
                lds_st(prefix_v, hf * LDS_PAD + hc)
        else:
            # slow path: sequence-relative bounds (still coalesced)
            zero_e = fx_elem_dtype(0.0)
            body_wp = tok_start + t_const
            body_ok = body_wp < seqlen
            sl_m1 = (seqlen > 0).select(seqlen - 1, 0)
            body_gt = seq_start + body_ok.select(body_wp, sl_m1)
            for j in fx.range_constexpr(ELEMS):
                gf = (feat_start + f_base) + (j * FG)
                gf_ok = gf < dim
                safe_gf = gf_ok.select(gf, 0)
                raw = fx_elem_dtype(
                    buffer_ops.buffer_load(
                        x_r, safe_gf * sx0 + body_gt, vec_width=1, dtype=elem_dtype
                    )
                )
                val = (body_ok & gf_ok).select(raw, zero_e)
                lds_st(
                    val,
                    (f_base + (j * FG)) * LDS_PAD + (t_const + (KW - 1)),
                )
            # halo column with conv_state blend at chunk0
            do_halo = hc < (KW - 1)
            if do_halo:
                gf = feat_start + hf
                gf_ok = gf < dim
                wp = (tok_start + hc) - (KW - 1)
                wp_in = (wp >= 0) & (wp < seqlen)
                both = wp_in & gf_ok
                safe_xoff = both.select(gf * sx0 + (seq_start + wp), 0)
                xv = both.select(
                    fx_elem_dtype(
                        buffer_ops.buffer_load(
                            x_r, safe_xoff, vec_width=1, dtype=elem_dtype
                        )
                    ),
                    zero_e,
                )
                # pre-seq source: conv_state at chunk0
                hi8 = fx.Int8(
                    buffer_ops.buffer_load(hi_r, seq_idx, vec_width=1, dtype=T.i8)
                )
                hi_nz = hi8 != 0
                need_cs = ((wp < 0) & is_chunk0) & (hi_nz & gf_ok)
                in_coord = fx.Int32(
                    buffer_ops.buffer_load(ci_r, seq_idx * sci, vec_width=1, dtype=i32)
                )
                slot = (KW - 1) + wp
                cs_off = need_cs.select((in_coord * scs0 + gf * scs1) + slot * scs2, 0)
                csv = fx_elem_dtype(
                    buffer_ops.buffer_load(cs_r, cs_off, vec_width=1, dtype=elem_dtype)
                )
                hv = need_cs.select(csv, xv)
                lds_st(hv, hf * LDS_PAD + hc)

        fx.gpu.barrier()

        # compute: acc[e] = bias + sum_k w[k] * x[...]; the EPT outputs share a
        # contiguous window, loaded once into registers before the MAC.
        row_base = feat_local * LDS_PAD + tok_base
        NSPAN = EPT + W - 1
        xw = []
        for i in fx.range_constexpr(NSPAN):
            idx = row_base if i == 0 else row_base + i
            xw.append(fx.Float32(lds_ld(idx)))
        acc = []
        for e in fx.range_constexpr(EPT):
            a = bias_f
            for kk in fx.range_constexpr(W):
                a = a + w_taps[kk] * xw[e + kk]
            if fx.const_expr(SILU):
                ex = fx.math.exp2(a * fx.Float32(-_LOG2E))
                a = a / (fx.Float32(1.0) + ex)
            acc.append(a)

        # store: transpose through staging (fast) or direct (slow)
        store_fast = ((feat_start + TN) <= dim) & ((tok_start + (TM - 1)) < seqlen)
        vstart = kd * 2
        blk_q = (feat_start + TN) <= kd
        blk_k = (feat_start >= kd) & ((feat_start + TN) <= vstart)
        blk_v = feat_start >= vstart

        if store_fast:
            fx.gpu.barrier()
            for e in fx.range_constexpr(EPT):
                lds_st(
                    acc[e].to(fx_elem_dtype),
                    (tok_base + e) * STORE_PAD + feat_local,
                )
            fx.gpu.barrier()
            sf = tid & (TN - 1)
            tg = tid >> 6
            tg_ept = tg * EPT
            tok0 = (seq_start + tok_start) + tg_ept

            def emit_fast(cond, res, ts, ds, fo):
                if cond:
                    of = (feat_start + sf) - fo
                    base_off = tok0 * ts + of * ds
                    cur = base_off
                    for e in fx.range_constexpr(EPT):
                        val = lds_ld((tg_ept + e) * STORE_PAD + sf)
                        buffer_ops.buffer_store(val, res, cur)
                        if fx.const_expr(e + 1 < EPT):
                            cur = cur + ts

            emit_fast(blk_q, q_r, qs0, qs1, 0)
            emit_fast(blk_k, k_r, ks0, ks1, kd)
            emit_fast(blk_v, v_r, vs0, vs1, vstart)
        else:

            def emit_slow(cond, res, ts, ds, fo):
                if cond & feat_valid:
                    of = gfeat - fo
                    base_off = ((seq_start + tok_start) + tok_base) * ts + of * ds
                    cur = base_off
                    for e in fx.range_constexpr(EPT):
                        tok_ok = ((tok_start + tok_base) + e) < seqlen
                        if tok_ok:
                            buffer_ops.buffer_store(acc[e].to(fx_elem_dtype), res, cur)
                        if fx.const_expr(e + 1 < EPT):
                            cur = cur + ts

            emit_slow(blk_q, q_r, qs0, qs1, 0)
            emit_slow(blk_k, k_r, ks0, ks1, kd)
            emit_slow(blk_v, v_r, vs0, vs1, vstart)

        # conv_state writeback (chunk 0)
        if fx.const_expr(SL > 0):
            if is_chunk0:
                zero_e = fx_elem_dtype(0.0)
                slot = tok_group
                should = (slot < (KW - 1)) & (gfeat < dim)
                if should:
                    in_coord = fx.Int32(
                        buffer_ops.buffer_load(
                            ci_r, seq_idx * sci, vec_width=1, dtype=i32
                        )
                    )
                    pos_x = (seqlen - (KW - 1)) + slot
                    x_in = pos_x >= 0
                    safe_x = x_in.select(gfeat * sx0 + (seq_start + pos_x), 0)
                    val_x = fx_elem_dtype(
                        buffer_ops.buffer_load(
                            x_r, safe_x, vec_width=1, dtype=elem_dtype
                        )
                    )
                    hi8 = fx.Int8(
                        buffer_ops.buffer_load(hi_r, seq_idx, vec_width=1, dtype=T.i8)
                    )
                    hi_nz = hi8 != 0
                    need_pr = (pos_x < 0) & hi_nz
                    src = slot + seqlen
                    safe_pr = need_pr.select(
                        (in_coord * scs0 + gfeat * scs1) + src * scs2, 0
                    )
                    val_pr = fx_elem_dtype(
                        buffer_ops.buffer_load(
                            cs_r, safe_pr, vec_width=1, dtype=elem_dtype
                        )
                    )
                    wb_val = x_in.select(val_x, need_pr.select(val_pr, zero_e))
                    cs_wr = (in_coord * scs0 + gfeat * scs1) + slot * scs2
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

    # Reuse precomputed chunk schedule metadata when provided.
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
