# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL prefill causal-conv1d kernel with fused split q/k/v output."""

import functools
from typing import Optional, Sequence

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr.typing import Int32

from aiter.ops.flydsl.kernels.kernels_common import LOG2E as _LOG2E
from aiter.ops.flydsl.kernels.tensor_shim import ptr_buf_tensor

from ..prefill_batch_metadata import CausalConvPrefillMetadata
from .kernels.causal_conv1d_prefill import create_causal_conv1d_prefill_kernel

PAD_SLOT_ID = -1


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
        # Every access here is one element wide, so each buffer gets a single
        # view typed by its own element -- `t[i]` is the whole access.
        x_r = ptr_buf_tensor(x_ptr, fx_elem_dtype)
        w_r = ptr_buf_tensor(w_ptr, fx_elem_dtype)
        b_r = ptr_buf_tensor(bias_ptr, fx_elem_dtype)
        cs_r = ptr_buf_tensor(cs_ptr, fx_elem_dtype)
        ci_r = ptr_buf_tensor(cache_idx_ptr, fx.Int32)
        hi_r = ptr_buf_tensor(has_init_ptr, fx.Int8)
        qsl_r = ptr_buf_tensor(qsl_ptr, fx.Int32)
        batch_r = ptr_buf_tensor(batch_ptr, fx.Int32)
        choff_r = ptr_buf_tensor(chunk_off_ptr, fx.Int32)
        q_r = ptr_buf_tensor(q_ptr, fx_elem_dtype)
        k_r = ptr_buf_tensor(k_ptr, fx_elem_dtype)
        v_r = ptr_buf_tensor(v_ptr, fx_elem_dtype)

        lds_base = fx.SharedAllocator().allocate(SharedStorage).peek().lds.ptr

        def lds_st(val, idx):
            fx.ptr_store(val, lds_base + fx.Int64(idx))

        def lds_ld(idx):
            return fx.ptr_load(lds_base + fx.Int64(idx))

        tid = fx.thread_idx.x
        pid_x = fx.block_idx.x
        pid_y = fx.block_idx.y

        seq_idx = fx.Int32(batch_r[pid_x])
        chunk_idx = fx.Int32(choff_r[pid_x])
        seq_start = fx.Int32(qsl_r[seq_idx])
        seq_end = fx.Int32(qsl_r[seq_idx + 1])
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
            w_taps.append(fx.Float32(w_r[w_base + j * sw1]))
        if fx.const_expr(HAS_BIAS):
            bias_f = fx.Float32(b_r[gfeat])
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
                raws.append(fx_elem_dtype(x_r[cur]))
                if fx.const_expr(j + 1 < ELEMS):
                    cur = cur + fstep
            do_halo = hc < (KW - 1)
            prefix_off = do_halo.select((feat_start + hf) * sx0 + (tok_gbase + hc), 0)
            prefix_v = fx_elem_dtype(x_r[prefix_off])
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
                raw = fx_elem_dtype(x_r[safe_gf * sx0 + body_gt])
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
                    fx_elem_dtype(x_r[safe_xoff]),
                    zero_e,
                )
                # pre-seq source: conv_state at chunk0
                hi8 = fx.Int8(hi_r[seq_idx])
                hi_nz = hi8 != 0
                need_cs = ((wp < 0) & is_chunk0) & (hi_nz & gf_ok)
                in_coord = fx.Int32(ci_r[seq_idx * sci])
                slot = (KW - 1) + wp
                cs_off = need_cs.select((in_coord * scs0 + gf * scs1) + slot * scs2, 0)
                csv = fx_elem_dtype(cs_r[cs_off])
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
                        res[cur] = val
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
                            res[cur] = acc[e].to(fx_elem_dtype)
                        if fx.const_expr(e + 1 < EPT):
                            cur = cur + ts

            emit_slow(blk_q, q_r, qs0, qs1, 0)
            emit_slow(blk_k, k_r, ks0, ks1, kd)
            emit_slow(blk_v, v_r, vs0, vs1, vstart)

        # conv_state writeback (chunk 0)
        if fx.const_expr(SL > 0) and is_chunk0:
            zero_e = fx_elem_dtype(0.0)
            slot = tok_group
            should = (slot < (KW - 1)) & (gfeat < dim)
            if should:
                in_coord = fx.Int32(ci_r[seq_idx * sci])
                pos_x = (seqlen - (KW - 1)) + slot
                x_in = pos_x >= 0
                safe_x = x_in.select(gfeat * sx0 + (seq_start + pos_x), 0)
                val_x = fx_elem_dtype(x_r[safe_x])
                hi8 = fx.Int8(hi_r[seq_idx])
                hi_nz = hi8 != 0
                need_pr = (pos_x < 0) & hi_nz
                src = slot + seqlen
                safe_pr = need_pr.select(
                    (in_coord * scs0 + gfeat * scs1) + src * scs2, 0
                )
                val_pr = fx_elem_dtype(cs_r[safe_pr])
                wb_val = x_in.select(val_x, need_pr.select(val_pr, zero_e))
                cs_wr = (in_coord * scs0 + gfeat * scs1) + slot * scs2
                cs_r[cs_wr] = wb_val

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
        stream: fx.Stream,
    ):
        gx = fx.Int64(num_programs)
        gy = fx.Int64(grid_y_dim)
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


@functools.cache
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
    if isinstance(metadata, CausalConvPrefillMetadata):
        metadata.validate(
            query_start_loc,
            total_tokens=cu_seqlen,
            num_sequences=query_start_loc.numel() - 1,
        )
        grid = metadata.get_chunk_grid(block_m)
        tot = grid.total_chunks
        batch_ptr = grid.sequence_ids
        chunk_off_ptr = grid.chunk_ids
    elif (
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
        except Exception:  # noqa: BLE001
            launcher._fast_compiled = False  # fall back permanently
            launcher(*launch_args)
    elif compiled is not False:
        compiled(*launch_args)
    else:
        launcher(*launch_args)
    return query, key, value


def causal_conv1d_prefill_flydsl_fn(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor], query_start_loc: torch.Tensor,
    seq_lens_cpu: Sequence[int], cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu", pad_slot_id: Optional[int] = -1,
    validate_data: bool = False, *, block: int = 128, tokens: int = 16,
    prefetch: int = 16, channels_per_thread: Optional[int] = None,
) -> torch.Tensor:
    """``causal_conv1d_fn`` contract: returns the output, updates ``conv_states``.

    Active cache indices must be unique and in bounds. CPU lengths must match
    GPU cumulative offsets. Padded-slot outputs are unspecified, like SGLang.
    No host tensor reads occur unless validate_data=True (not graph safe).
    prefetch is the number of token loads grouped before consumption. The default
    channel mapping pairs adjacent BF16/FP16 channels for channel-contiguous inputs
    with at least 8192 packed tokens; other inputs use one channel per thread.
    """
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("expected x [channels,total_tokens], weight [channels,width]")
    dim, total = x.shape
    width = weight.shape[1]
    batch = len(seq_lens_cpu)
    if weight.shape[0] != dim or width not in (2, 3, 4, 5):
        raise ValueError("weight must have matching channels and width 2..5")
    if activation not in (None, False, True, "silu", "swish"):
        raise ValueError("activation must be None, silu, or swish")
    if block not in (64, 128, 256) or not width - 1 <= tokens <= 64:
        raise ValueError("block must be 64/128/256 and width-1 <= tokens <= 64")
    if prefetch not in (1, 2, 4, 8, 16):
        raise ValueError("prefetch must be 1/2/4/8/16")
    if channels_per_thread not in (None, 1, 2):
        raise ValueError("channels_per_thread must be None, 1, or 2")
    if channels_per_thread == 2 and dim % 2:
        raise ValueError("two channels per thread requires an even channel count")
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("expected a GPU BF16/FP16/FP32 tensor")
    if 1 not in x.stride() or 1 not in weight.stride():
        raise ValueError("input and weight need at least one unit-stride axis")
    tensors = [weight, query_start_loc, bias, conv_states, cache_indices, has_initial_state]
    if any(t is not None and t.device != x.device for t in tensors):
        raise ValueError("all tensors must be on the input device")
    if weight.dtype != x.dtype or (conv_states is not None and conv_states.dtype != x.dtype):
        raise ValueError("input, weight, and state must have matching dtype")
    if query_start_loc.shape != (batch + 1,) or not query_start_loc.is_contiguous():
        raise ValueError("query_start_loc must be contiguous [batch+1]")
    if query_start_loc.dtype not in (torch.int32, torch.int64):
        raise ValueError("query_start_loc must contain integers")
    for name, tensor in (("cache_indices", cache_indices), ("has_initial_state", has_initial_state)):
        if tensor is not None and (tensor.shape != (batch,) or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be contiguous [batch]")
    if cache_indices is not None and cache_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("cache_indices must contain integers")
    if bias is not None and (bias.shape != (dim,) or not bias.is_contiguous()):
        raise ValueError("bias must be contiguous [channels]")
    if conv_states is not None and (conv_states.ndim != 3 or conv_states.shape[1] != dim or conv_states.shape[2] < width - 1):
        raise ValueError("state must be [slots,channels,at_least_width-1]")
    if conv_states is not None and 1 not in conv_states.stride():
        raise ValueError("state needs at least one unit-stride axis")
    if conv_states is not None and cache_indices is None and conv_states.shape[0] < batch:
        raise ValueError("state needs at least batch slots without cache_indices")
    if has_initial_state is not None and conv_states is None:
        raise ValueError("initial history requires conv_states")
    if any(n < 0 for n in seq_lens_cpu) or sum(seq_lens_cpu) > total:
        raise ValueError("invalid sequence lengths")
    if validate_data:
        starts = query_start_loc.cpu().tolist()
        expected = [0]
        for n in seq_lens_cpu:
            expected.append(expected[-1] + n)
        if starts != expected:
            raise ValueError("GPU offsets do not match CPU sequence lengths")
        if conv_states is not None:
            slots = list(range(batch)) if cache_indices is None else cache_indices.cpu().tolist()
            active = [s for s, n in zip(slots, seq_lens_cpu) if s != pad_slot_id and n > 0]
            if len(set(active)) != len(active) or any(s < 0 or s >= conv_states.shape[0] for s in active):
                raise ValueError("active cache indices must be unique and in bounds")
    out = torch.empty_like(x)
    max_len = max(seq_lens_cpu, default=0)
    if not max_len or not dim:
        return out
    lanes = channels_per_thread
    if lanes is None:
        lanes = 2 if (dim % 2 == 0 and x.stride(0) == 1
                      and x.dtype in (torch.bfloat16, torch.float16)
                      and total >= 8192) else 1
    ss = conv_states.stride() if conv_states is not None else (0, 0, 0)
    def span(t: Optional[torch.Tensor]) -> int:
        return 1 if t is None else 1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride()))
    launch = create_causal_conv1d_prefill_kernel(dim, width, x.stride(), weight.stride(), ss, out.stride(),
                    bias is not None, conv_states is not None, cache_indices is not None,
                    has_initial_state is not None, activation in (True, "silu", "swish"),
                    pad_slot_id, block, tokens, x.dtype, prefetch, lanes)
    # Absent optional pointers are never dereferenced by the specialized kernel.
    args = (x, weight, bias if bias is not None else x,
            conv_states if conv_states is not None else x, query_start_loc,
            cache_indices if cache_indices is not None else query_start_loc,
            has_initial_state if has_initial_state is not None else query_start_loc, out)
    runtime = (batch, max_len, *(span(t) for t in (x, weight, conv_states, out)))
    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream(x.device)
        compiled = getattr(launch, "_compiled", None)
        # FlyDSL specializes dtype, rank, and the first unit-stride axis.
        # Sizes are dynamic; physical indexing strides remain in _build's key.
        key = (x.device.index, tuple((t.dtype, t.ndim, t.stride().index(1)) for t in args))
        if compiled is None:
            launch._compiled = {}
        if key not in launch._compiled:
            launch._compiled[key] = flyc.compile(launch, *args, *runtime, fx.Stream(stream))
        else:
            launch._compiled[key](*args, *runtime, fx.Stream(stream))
    return out
