# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL decode + speculative-verify causal-conv1d update kernels.

Two builders, one per upstream interface, sharing a module because they are the
same convolution behind two shells:

* ``build_causal_conv1d_update_module`` ports vLLM's Triton
  ``_causal_conv1d_update_kernel``, i.e. the *chain* (linear) speculative path.
* ``build_causal_conv1d_update_sglang_module`` ports SGLang's kernel of the same
  name. The convolution core is bit-identical; only the shell differs -- the
  accepted-token argument is spelled ``num_accept_tokens``, ``SAVE_INTERMEDIATE``
  snapshots each step's window into ``intermediate_conv_window``, and an EAGLE
  **tree** path convolves along each token's parent chain instead of its linear
  predecessor, fusing the ``retrieve_parent_token`` mapping out.

Torch-facing wrappers live in ``aiter.ops.flydsl.causal_conv1d_update_kernels``,
keeping this module torch-free.

Design (shared by both)
-----------------------
* One thread owns ``C`` feature channels (``C`` = ``channels_per_thread``).
  ``grid = (batch, cdiv(dim, BLOCK_N * C))``, ``block = (BLOCK_N, 1, 1)``, and
  channel ``c`` of a thread is ``pid_y * (BLOCK_N * C) + c * BLOCK_N + tid`` so
  neighbouring lanes touch neighbouring channels and the loads stay coalesced.
  This resolves the upstreams' ``idx_feats`` down to single channels, so the
  convolution runs entirely in registers with no shared memory.
* ``seqlen`` and the effective ``state_len`` are compile-time constants, as they
  are ``tl.constexpr`` upstream, so every loop is unrolled.
* Channel-independent work -- the cache-line index, the rollback offset, the
  tree parent chain -- is computed once and shared by the ``C`` channels.
* Addressing splits the way vLLM's does: terms scaling with the batch or the
  cache size (the ones vLLM casts to ``tl.int64``) are folded into the 64-bit
  descriptor base, leaving the per-channel and per-token remainder in the 32-bit
  offset. Without the split a conv_state or packed ``x`` above 4 GiB wraps and
  reads the wrong lines with no fault. A descriptor base must be uniform, so the
  indices feeding one are loaded scalar (see ``_cache_line``).

The upstream STEP 1-5 map across directly: read the ``width-1`` history columns
at the rollback point (1), slide the conv_state window by ``1 if spec else
seqlen`` and blend in the new ``x`` tokens (2), preload the weight taps, bias
and tree chains (3/4), then convolve token by token, apply SiLU and store (5).

Scope
-----
Each builder covers every mode of the kernel it ports: for vLLM that includes
varlen packing and prefix-caching copy-on-write, including both at once, which
is how its speculative sites call it; for SGLang the snapshots and the EAGLE
tree. ``cache_seqlens`` (circular buffer) is unimplemented, matching SGLang.

Upstream's ``NP2_STATELEN`` / ``NP2_SEQLEN`` have no counterpart because they
pad Triton's power-of-two tiles, which unrolled loops do not need, and
``USE_GDC`` / ``launch_pdl`` gate a CUDA-only launch mode.
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl

from aiter.ops.flydsl.kernels import buffer_ops

_LOG2E = 1.4426950408889634


def build_causal_conv1d_update_module(
    width: int,
    seqlen: int,
    has_bias: bool,
    silu: bool,
    is_spec_decoding: bool,
    has_null_block: bool,
    block_n: int = 256,
    dtype_str: str = "bf16",
    weight_contig: bool = True,
    cs_vec: bool = False,
    o_vec: bool = False,
    channels_per_thread: int = 1,
    is_apc_enabled: bool = False,
    is_varlen: bool = False,
):
    """Build a FlyDSL decode/verify conv1d-update kernel for a fixed config.

    ``seqlen`` is the number of tokens processed per sequence (``1`` for plain
    decode, ``1 + K`` for chain verify) and is baked in as a compile-time
    constant, exactly like vLLM's ``seqlen: tl.constexpr``.

    ``is_apc_enabled`` is vLLM's ``IS_APC_ENABLED``: it splits the cache line the
    history is read from off the one the rolled window is written to, so a
    prefix-cached block is copied rather than clobbered.

    ``is_varlen`` is vLLM's ``IS_VARLEN``: ``x`` arrives packed as
    ``(cu_tokens, dim)`` and each sequence's token count comes from
    ``query_start_loc``, so ``seqlen`` becomes the *maximum* over the batch and
    the per-sequence count is a runtime bound. ``state_len`` shrinks with it by
    the same amount upstream does, which leaves ``VAL`` -- the number of window
    slots still fed by ``conv_state`` -- a compile-time constant either way.

    ``channels_per_thread`` (CPT) makes one thread own that many feature
    channels, trading workgroup count for more independent loads in flight per
    thread; see this module's docstring for the channel mapping.
    """
    assert 2 <= width <= 6
    assert seqlen >= 1
    assert channels_per_thread >= 1

    W = width
    S = seqlen
    BN = block_n
    CPT = int(channels_per_thread)
    HAS_BIAS = bool(has_bias)
    SILU = bool(silu)
    IS_SPEC = bool(is_spec_decoding)
    HAS_NULL_BLOCK = bool(has_null_block)
    IS_APC = bool(is_apc_enabled)
    IS_VARLEN = bool(is_varlen)
    ELEM_BYTES = 2  # bf16 / fp16 are the only element types this kernel takes

    # Effective conv_state window length (matches vLLM wrapper):
    #   spec  -> width - 1 + (seqlen - 1)   (history + K candidate slots)
    #   decode-> width - 1
    ST = (W - 1 + (S - 1)) if IS_SPEC else (W - 1)
    VAL = ST - S  # tokens before the first x column enters the rolled window
    SHIFT = 1 if IS_SPEC else S  # conv_state slide, per vLLM's STEP 2

    # Only this combination slides by a runtime amount: upstream rewrites
    # `seqlen` to the per-sequence count first, so without a rollback point the
    # source column stops being a compile-time index into the history taps. With
    # a rollback point the slide is 1, and at VAL <= 0 no slot comes from
    # conv_state at all.
    RUNTIME_SHIFT = IS_VARLEN and not IS_SPEC and VAL > 0

    # One channel's W taps are contiguous, so fetch them in a single vector load.
    # vec_width supports 2/4; other widths keep the scalar path.
    WEIGHT_VEC = bool(weight_contig) and W in (2, 4)

    # With a contiguous token axis one channel's written slots are adjacent and
    # collapse into a few vector stores plus a scalar remainder. The wrapper
    # decides; see it for why an odd channel stride is still worth vectorizing.
    CS_VEC = bool(cs_vec)
    O_VEC = bool(o_vec)

    def _vec_chunks(total):
        """Cover [0, total) with aligned vector widths (4, 2) + scalar."""
        chunks, i = [], 0
        for wd in (4, 2):
            while total - i >= wd:
                chunks.append((i, wd))
                i += wd
        while i < total:
            chunks.append((i, 1))
            i += 1
        return chunks

    @flyc.kernel
    def update_kernel(
        x_ptr: fx.Tensor,
        w_ptr: fx.Tensor,
        bias_ptr: fx.Tensor,
        cs_ptr: fx.Tensor,
        csi_ptr: fx.Tensor,
        nacc_ptr: fx.Tensor,
        qsl_ptr: fx.Tensor,
        blst_ptr: fx.Tensor,
        isi_ptr: fx.Tensor,
        o_ptr: fx.Tensor,
        dim: fx.Int32,
        num_cache_lines: fx.Int32,
        null_block_id: fx.Int32,
        sx_seq: fx.Int32,
        sx_dim: fx.Int32,
        sx_tok: fx.Int32,
        sw_dim: fx.Int32,
        sw_width: fx.Int32,
        scs_seq: fx.Int32,
        scs_dim: fx.Int32,
        scs_tok: fx.Int32,
        scsi: fx.Int32,
        so_seq: fx.Int32,
        so_dim: fx.Int32,
        so_tok: fx.Int32,
    ):
        elem_dtype = fx.BFloat16 if dtype_str == "bf16" else fx.Float16

        def _rsrc(ptr):
            return buffer_ops.create_buffer_resource(ptr, max_size=True)

        def _rsrc_at(ptr, index, stride):
            """Descriptor whose base already includes ``index * stride`` elements.

            The buffer offset operand is 32 bits wide in hardware, so a term that
            scales with the batch or the cache size cannot live there. Folding it
            into the descriptor base instead is the same split vLLM makes when it
            promotes exactly these quantities to ``tl.int64`` and leaves the
            per-channel arithmetic in 32 bits. The two factors are widened
            separately: their product is what overflows.
            """
            byte_offset = (
                index.to(fx.Int64) * stride.to(fx.Int64) * fx.Int64(ELEM_BYTES)
            )
            return buffer_ops.create_buffer_resource(
                ptr, max_size=True, base_byte_offset=byte_offset.ir_value()
            )

        # Only the descriptors actually used: the rest are dummy (x) pointers, and
        # skipping them keeps their kernargs out of the prologue.
        w_r = _rsrc(w_ptr)
        b_r = _rsrc(bias_ptr) if fx.const_expr(HAS_BIAS) else None
        csi_r = _rsrc(csi_ptr)
        nacc_r = _rsrc(nacc_ptr) if fx.const_expr(IS_SPEC) else None
        qsl_r = _rsrc(qsl_ptr) if fx.const_expr(IS_VARLEN) else None
        blst_r = _rsrc(blst_ptr) if fx.const_expr(IS_APC) else None
        isi_r = _rsrc(isi_ptr) if fx.const_expr(IS_APC) else None
        # x / out / conv_state are the ones whose offsets scale with the batch and
        # the cache size, so their descriptors are built further down, once the
        # sequence-level part of their address is known.

        tid = fx.Int32(fx.thread_idx.x)
        idx_seq = fx.Int32(fx.block_idx.x)
        pid_y = fx.Int32(fx.block_idx.y)

        # ============ sequence-level work, shared by the CPT channels ========
        # Everything here depends only on block_idx.x, so with CPT > 1 it is
        # computed once instead of once per channel.

        # Cache line(s) for this sequence. Without APC one index serves both the
        # history load and the write-back; with APC conv_state_indices is 2D and
        # the ends differ, the history coming from the block the state was last
        # computed into and the rolled window going to the block scheduled for
        # this step, which is what makes a prefix-cached block copy-on-write.
        #
        # Loaded scalar because the result reaches a descriptor base, which must
        # be uniform. A per-lane load does not tell the compiler that, and it
        # answers by serializing every access built on the result.
        def _cache_line(block_offset):
            return fx.Int32(
                buffer_ops.buffer_load(
                    csi_r, block_offset, vec_width=1, dtype=fx.Int32, is_scalar=True
                )
            )

        # Spelled as a branch rather than adding an index that is zero when APC
        # is off, so the offset expression stays byte-identical in that case.
        if fx.const_expr(IS_APC):
            init_i = fx.Int32(
                buffer_ops.buffer_load(
                    isi_r, idx_seq, vec_width=1, dtype=fx.Int32, is_scalar=True
                )
            )
            last_i = fx.Int32(
                buffer_ops.buffer_load(
                    blst_r, idx_seq, vec_width=1, dtype=fx.Int32, is_scalar=True
                )
            )
            in_coord = _cache_line(idx_seq * scsi + init_i)
            out_coord = _cache_line(idx_seq * scsi + last_i)
        else:
            in_coord = _cache_line(idx_seq * scsi)
            out_coord = in_coord

        # Packed layout: this sequence owns x[qs:qe], a runtime count bounded by
        # the compile-time S. Upstream shrinks state_len by the same slack, which
        # is why VAL stays constant and only the window tail and output need
        # runtime bounds.
        if fx.const_expr(IS_VARLEN):
            qs = fx.Int32(
                buffer_ops.buffer_load(
                    qsl_r, idx_seq, vec_width=1, dtype=fx.Int32, is_scalar=True
                )
            )
            qe = fx.Int32(
                buffer_ops.buffer_load(
                    qsl_r,
                    idx_seq + fx.Int32(1),
                    vec_width=1,
                    dtype=fx.Int32,
                    is_scalar=True,
                )
            )
            s_len = qe - qs
            # An empty slot stores nothing but still issues its loads, so aim
            # them at the start of the tensor: its own qs can sit one past the
            # last token when it trails the batch.
            nonempty = s_len > fx.Int32(0)
            qs_eff = nonempty.select(qs, fx.Int32(0))
            tok_hi = nonempty.select(s_len - fx.Int32(1), fx.Int32(0))
            x_seq_idx, x_seq_stride = qs_eff, sx_tok
            o_seq_idx, o_seq_stride = qs_eff, so_tok
        else:
            s_len = fx.Int32(S)
            x_seq_idx, x_seq_stride = idx_seq, sx_seq
            o_seq_idx, o_seq_stride = idx_seq, so_seq

        # Sequence-level address terms go into the 64-bit descriptor base; the
        # per-channel and per-token remainder stays in the 32-bit offset. Read and
        # write bases differ only under APC.
        x_r = _rsrc_at(x_ptr, x_seq_idx, x_seq_stride)
        o_r = _rsrc_at(o_ptr, o_seq_idx, o_seq_stride)
        cs_r = _rsrc_at(cs_ptr, in_coord, scs_seq)
        cs_out_r = _rsrc_at(cs_ptr, out_coord, scs_seq) if IS_APC else cs_r

        # rollback point: spec -> num_accepted - 1, decode -> 0. Left per-lane
        # rather than scalar, since it never reaches a descriptor base.
        if fx.const_expr(IS_SPEC):
            nacc = fx.Int32(
                buffer_ops.buffer_load(nacc_r, idx_seq, vec_width=1, dtype=fx.Int32)
            )
            offset_dyn = nacc - fx.Int32(1)
            if fx.const_expr(IS_VARLEN):
                # Upstream returns on an empty slot before it ever reads the accept
                # count, so nothing constrains that entry. This kernel issues its
                # loads unconditionally and only guards the stores, so pin the
                # offset for empty slots rather than address conv_state with
                # whatever the padding happens to hold.
                offset_dyn = (s_len > fx.Int32(0)).select(offset_dyn, fx.Int32(0))
        else:
            offset_dyn = fx.Int32(0)

        def _store_run(vals, rsrc, base, stride, total, vectorize):
            # Callers only set ``vectorize`` when the axis stride is 1, so the
            # slot offset is a constant here; spelling it as one keeps the byte
            # address out of the runtime path.
            if fx.const_expr(vectorize):
                for start, wd in _vec_chunks(total):
                    off = base if fx.const_expr(start == 0) else base + fx.Int32(start)
                    if fx.const_expr(wd == 1):
                        buffer_ops.buffer_store(vals[start], rsrc, off)
                    else:
                        chunk = fx.Vector.from_elements(
                            [vals[start + j] for j in range(wd)], elem_dtype
                        )
                        buffer_ops.buffer_store(chunk, rsrc, off)
            else:
                for t in fx.range_constexpr(total):
                    buffer_ops.buffer_store(vals[t], rsrc, base + fx.Int32(t) * stride)

        # ================= per-channel work ==================================
        def _channel(gfeat):
            # active guard: valid feature, in-range cache line, non-null block.
            # The cache-line half is loop-invariant across the CPT channels but is
            # left here rather than hoisted, which CSE already covers and which
            # measured worse.
            feat_ok = gfeat < dim
            active = feat_ok & (in_coord < num_cache_lines)
            if fx.const_expr(HAS_NULL_BLOCK):
                active = active & (in_coord != null_block_id)
            if fx.const_expr(IS_VARLEN):
                # An empty slot in the packed batch is skipped whole, as upstream
                # returns before it writes anything.
                active = active & (s_len > fx.Int32(0))

            # Both cache coordinates already sit in their descriptor bases, so
            # what is left is the per-channel term, shared by read and write.
            cs_base = gfeat * scs_dim
            x_base = gfeat * sx_dim
            o_base = gfeat * so_dim

            # ============ PHASE 1: issue ALL loads up front ==================
            # Every load is issued before any convert or compute, so they stay in
            # flight together. The conv_state addresses depend on csi and nacc,
            # which are themselves still in flight, so the loads independent of
            # those go first to overlap that round-trip. In decode offset_dyn is
            # constant and the ordering is merely harmless.

            # The S new x tokens, loaded once and reused for both the rolled
            # conv_state slots and the convolution.
            x_raw = []
            for tok in fx.range_constexpr(S):
                if fx.const_expr(IS_VARLEN and tok > 0):
                    # Past this sequence's tokens the value is unused (its
                    # consumers are store-guarded) but the address must stay
                    # inside the packed tensor, so clamp rather than mask. Token 0
                    # needs no clamp, tok_hi being 0 when the slot is empty.
                    tok_idx = (fx.Int32(tok) < s_len).select(fx.Int32(tok), tok_hi)
                else:
                    tok_idx = fx.Int32(tok)
                off = x_base + tok_idx * sx_tok
                x_raw.append(
                    elem_dtype(
                        buffer_ops.buffer_load(x_r, off, vec_width=1, dtype=elem_dtype)
                    )
                )

            # STEP 4 weights (raw) + optional bias (raw)
            w_base = gfeat * sw_dim
            if fx.const_expr(WEIGHT_VEC):
                wv = fx.Vector(
                    buffer_ops.buffer_load(w_r, w_base, vec_width=W, dtype=elem_dtype)
                )
                w_raw = [wv[j] for j in fx.range_constexpr(W)]
            else:
                w_raw = []
                for j in fx.range_constexpr(W):
                    off = w_base + fx.Int32(j) * sw_width
                    w_raw.append(
                        elem_dtype(
                            buffer_ops.buffer_load(
                                w_r, off, vec_width=1, dtype=elem_dtype
                            )
                        )
                    )
            if fx.const_expr(HAS_BIAS):
                bias_raw = elem_dtype(
                    buffer_ops.buffer_load(b_r, gfeat, vec_width=1, dtype=elem_dtype)
                )

            # ---- loads needing cs_base and offset_dyn ------------------------
            # STEP 1 history: W-1 taps starting at conv_state[offset + 0].
            col_raw = []
            for k in fx.range_constexpr(W - 1):
                off = cs_base + (offset_dyn + fx.Int32(k)) * scs_tok
                col_raw.append(
                    elem_dtype(
                        buffer_ops.buffer_load(cs_r, off, vec_width=1, dtype=elem_dtype)
                    )
                )

            if fx.const_expr(RUNTIME_SHIFT):
                roll_raw = []
                for t in fx.range_constexpr(VAL):
                    off = cs_base + (offset_dyn + s_len + fx.Int32(t)) * scs_tok
                    roll_raw.append(
                        elem_dtype(
                            buffer_ops.buffer_load(
                                cs_r, off, vec_width=1, dtype=elem_dtype
                            )
                        )
                    )

            # STEP 2 needs no loads of its own. A rolled slot reads
            # conv_state[offset + SHIFT + t], and the roll condition bounds that
            # to offset + (W-2), which col_raw already holds -- so the store below
            # reuses col_raw[SHIFT + t] instead of re-reading.

            # ============ PHASE 2: convert + convolution =====================
            cols = [c.to(fx.Float32) for c in col_raw]
            w_col = [w.to(fx.Float32) for w in w_raw]
            xt = [v.to(fx.Float32) for v in x_raw]
            if fx.const_expr(HAS_BIAS):
                bias_f = bias_raw.to(fx.Float32)
            else:
                bias_f = fx.Float32(0.0)

            # ---- STEP 5: running causal convolution over the S tokens -------
            window = list(cols)  # W-1 f32 history taps
            o_vals = []
            for idx_token in fx.range_constexpr(S):
                win = window + [xt[idx_token]]  # W taps: [hist..., current]
                acc = bias_f
                for j in fx.range_constexpr(W):
                    acc = acc + w_col[j] * win[j]
                if fx.const_expr(SILU):
                    # The bare intrinsics, which is what the Triton oracles lower
                    # to and what keeps the parity suite bit-exact. Deliberately
                    # not fx.math.exp2, whose denormal/range fixup this kernel
                    # never needs.
                    f32_ty = fx.Float32.ir_type
                    ex = fx.Float32(
                        rocdl.exp2(f32_ty, (acc * fx.Float32(-_LOG2E)).ir_value())
                    )
                    acc = acc * fx.Float32(rocdl.rcp(f32_ty, fx.Float32(1.0) + ex))
                o_vals.append(acc.to(elem_dtype))
                window = win[1:]  # slide the register window left by one

            # ============ PHASE 3: guarded stores ============================
            # Rolled window slot t: history below VAL (col_raw, or roll_raw when
            # the slide is only known at runtime), new x token at or above it.
            cs_vals = []
            for t in fx.range_constexpr(ST):
                if fx.const_expr((t + S) >= ST):
                    cs_vals.append(x_raw[t - VAL])
                elif fx.const_expr(RUNTIME_SHIFT):
                    cs_vals.append(roll_raw[t])
                else:
                    cs_vals.append(col_raw[SHIFT + t])

            if fx.const_expr(IS_VARLEN):
                # Only VAL + s_len slots and s_len outputs belong to this
                # sequence and the rest must keep what was there, so every store
                # is predicated and the vectorized runs do not apply. The first
                # VAL slots come from history and are always in range.
                for t in fx.range_constexpr(ST):
                    keep = (
                        active
                        if fx.const_expr(t < VAL)
                        else active & (fx.Int32(t - VAL) < s_len)
                    )
                    if keep:
                        buffer_ops.buffer_store(
                            cs_vals[t], cs_out_r, cs_base + fx.Int32(t) * scs_tok
                        )
                for t in fx.range_constexpr(S):
                    keep = (
                        active
                        if fx.const_expr(t == 0)
                        else active & (fx.Int32(t) < s_len)
                    )
                    if keep:
                        buffer_ops.buffer_store(
                            o_vals[t], o_r, o_base + fx.Int32(t) * so_tok
                        )
            elif active:
                _store_run(cs_vals, cs_out_r, cs_base, scs_tok, ST, CS_VEC)
                _store_run(o_vals, o_r, o_base, so_tok, S, O_VEC)

        for c in fx.range_constexpr(CPT):
            _channel(pid_y * fx.Int32(BN * CPT) + fx.Int32(c * BN) + tid)

    @flyc.jit
    def launch(
        x_ptr: fx.Tensor,
        w_ptr: fx.Tensor,
        bias_ptr: fx.Tensor,
        cs_ptr: fx.Tensor,
        csi_ptr: fx.Tensor,
        nacc_ptr: fx.Tensor,
        qsl_ptr: fx.Tensor,
        blst_ptr: fx.Tensor,
        isi_ptr: fx.Tensor,
        o_ptr: fx.Tensor,
        dim: fx.Int32,
        num_cache_lines: fx.Int32,
        null_block_id: fx.Int32,
        sx_seq: fx.Int32,
        sx_dim: fx.Int32,
        sx_tok: fx.Int32,
        sw_dim: fx.Int32,
        sw_width: fx.Int32,
        scs_seq: fx.Int32,
        scs_dim: fx.Int32,
        scs_tok: fx.Int32,
        scsi: fx.Int32,
        so_seq: fx.Int32,
        so_dim: fx.Int32,
        so_tok: fx.Int32,
        batch: fx.Int32,
        grid_y_dim: fx.Int32,
        stream: fx.Stream,
    ):
        update_kernel(
            x_ptr,
            w_ptr,
            bias_ptr,
            cs_ptr,
            csi_ptr,
            nacc_ptr,
            qsl_ptr,
            blst_ptr,
            isi_ptr,
            o_ptr,
            dim,
            num_cache_lines,
            null_block_id,
            sx_seq,
            sx_dim,
            sx_tok,
            sw_dim,
            sw_width,
            scs_seq,
            scs_dim,
            scs_tok,
            scsi,
            so_seq,
            so_dim,
            so_tok,
        ).launch(grid=(batch, grid_y_dim, 1), block=(BN, 1, 1), stream=stream)

    launch._bn = BN
    launch._cpt = CPT
    launch._state_len = ST
    launch._seqlen = S
    return launch


@functools.cache
def compile_causal_conv1d_update(
    width: int,
    seqlen: int,
    has_bias: bool,
    silu: bool,
    is_spec_decoding: bool,
    has_null_block: bool,
    block_n: int,
    dtype_str: str,
    weight_contig: bool,
    cs_vec: bool,
    o_vec: bool,
    channels_per_thread: int,
    is_apc_enabled: bool = False,
    is_varlen: bool = False,
):
    """Memoized :func:`build_causal_conv1d_update_module`.

    Every argument is a compile-time specialization, so one launcher is built
    per distinct configuration and reused for the life of the process.
    """
    return build_causal_conv1d_update_module(
        width,
        seqlen,
        has_bias,
        silu,
        is_spec_decoding,
        has_null_block,
        block_n,
        dtype_str,
        weight_contig,
        cs_vec,
        o_vec,
        channels_per_thread,
        is_apc_enabled,
        is_varlen,
    )


def build_causal_conv1d_update_sglang_module(
    width: int,
    seqlen: int,
    has_bias: bool,
    silu: bool,
    is_spec_decoding: bool,
    has_null_block: bool,
    block_n: int = 256,
    dtype_str: str = "bf16",
    weight_contig: bool = True,
    cs_vec: bool = False,
    o_vec: bool = False,
    i_vec: bool = False,
    save_intermediate: bool = False,
    save_stream: bool = False,
    has_tree: bool = False,
    channels_per_thread: int = 1,
):
    """Build a FlyDSL SGLang-flavoured conv1d-update kernel for a fixed config.

    ``seqlen`` is the number of tokens processed per sequence (``1`` for plain
    decode, ``1 + K`` for verify) and is baked in as a compile-time constant,
    exactly like SGLang's ``seqlen: tl.constexpr``.
    """
    assert 2 <= width <= 4, "SGLang's kernel only implements width 2/3/4"
    assert seqlen >= 1

    W = width
    S = seqlen
    BN = block_n
    CPT = int(channels_per_thread)
    HAS_BIAS = bool(has_bias)
    SILU = bool(silu)
    IS_SPEC = bool(is_spec_decoding)
    HAS_NULL_BLOCK = bool(has_null_block)
    SAVE_INTER = bool(save_intermediate)
    # Same snapshot without the aliasing: consecutive windows overlap in W-2 taps,
    # so against SGLang's deduplicated layout the per-step stores would write
    # every address W-1 times. This walks the run once for identical bytes. The
    # wrapper picks the mode from the snapshot's strides.
    SAVE_STREAM = bool(save_stream)
    assert not (SAVE_INTER and SAVE_STREAM), "pick one snapshot representation"
    SAVE_ANY = SAVE_INTER or SAVE_STREAM
    TREE = bool(has_tree)

    # Effective conv_state window length (matches the SGLang wrapper):
    #   spec  -> width - 1 + (seqlen - 1)   (history + K candidate slots)
    #   decode-> width - 1
    ST = (W - 1 + (S - 1)) if IS_SPEC else (W - 1)
    VAL = ST - S  # tokens before the first x column enters the rolled window
    SHIFT = 1 if IS_SPEC else S  # conv_state slide, per SGLang's STEP 2

    # History columns 1..W-2 followed by the S new tokens: the union of every
    # per-step window, laid out so step t's window is positions t .. t+W-2.
    STREAM_LEN = S + W - 2

    # One vector load for a channel's W taps; width 3 keeps the scalar path.
    WEIGHT_VEC = bool(weight_contig) and W in (2, 4)
    CS_VEC = bool(cs_vec)
    O_VEC = bool(o_vec)
    I_VEC = bool(i_vec)

    def _vec_chunks(total):
        """Cover [0, total) with aligned vector widths (4, 2) + scalar."""
        chunks, i = [], 0
        for wd in (4, 2):
            while total - i >= wd:
                chunks.append((i, wd))
                i += wd
        while i < total:
            chunks.append((i, 1))
            i += 1
        return chunks

    @flyc.kernel
    def update_kernel(
        x_ptr: fx.Tensor,
        w_ptr: fx.Tensor,
        bias_ptr: fx.Tensor,
        cs_ptr: fx.Tensor,
        csi_ptr: fx.Tensor,
        nacc_ptr: fx.Tensor,
        o_ptr: fx.Tensor,
        inter_ptr: fx.Tensor,
        isi_ptr: fx.Tensor,
        rnt_ptr: fx.Tensor,
        rns_ptr: fx.Tensor,
        rpt_ptr: fx.Tensor,
        dim: fx.Int32,
        num_cache_lines: fx.Int32,
        null_block_id: fx.Int32,
        sx_seq: fx.Int32,
        sx_dim: fx.Int32,
        sx_tok: fx.Int32,
        sw_dim: fx.Int32,
        sw_width: fx.Int32,
        scs_seq: fx.Int32,
        scs_dim: fx.Int32,
        scs_tok: fx.Int32,
        scsi: fx.Int32,
        so_seq: fx.Int32,
        so_dim: fx.Int32,
        so_tok: fx.Int32,
        si_seq: fx.Int32,
        si_step: fx.Int32,
        si_dim: fx.Int32,
        si_win: fx.Int32,
        sisi: fx.Int32,
        srnt_seq: fx.Int32,
        srnt_tok: fx.Int32,
        srns_seq: fx.Int32,
        srns_tok: fx.Int32,
        srpt_seq: fx.Int32,
        srpt_tok: fx.Int32,
    ):
        elem_dtype = fx.BFloat16 if dtype_str == "bf16" else fx.Float16

        def _rsrc(ptr):
            return buffer_ops.create_buffer_resource(ptr, max_size=True)

        def _load_i32(rsrc, off, is_scalar=False):
            return fx.Int32(
                buffer_ops.buffer_load(
                    rsrc, off, vec_width=1, dtype=fx.Int32, is_scalar=is_scalar
                )
            )

        def _load_elem(rsrc, off):
            return elem_dtype(
                buffer_ops.buffer_load(rsrc, off, vec_width=1, dtype=elem_dtype)
            )

        # Only the descriptors actually used; the rest are dummy pointers.
        x_r = _rsrc(x_ptr)
        w_r = _rsrc(w_ptr)
        b_r = _rsrc(bias_ptr) if fx.const_expr(HAS_BIAS) else None
        cs_r = _rsrc(cs_ptr)
        csi_r = _rsrc(csi_ptr)
        nacc_r = _rsrc(nacc_ptr) if fx.const_expr(IS_SPEC) else None
        o_r = _rsrc(o_ptr)
        inter_r = _rsrc(inter_ptr) if fx.const_expr(SAVE_ANY) else None
        isi_r = _rsrc(isi_ptr) if fx.const_expr(SAVE_ANY) else None
        rnt_r = _rsrc(rnt_ptr) if fx.const_expr(TREE) else None
        rns_r = _rsrc(rns_ptr) if fx.const_expr(TREE) else None
        rpt_r = _rsrc(rpt_ptr) if fx.const_expr(TREE) else None

        tid = fx.Int32(fx.thread_idx.x)
        idx_seq = fx.Int32(fx.block_idx.x)
        pid_y = fx.Int32(fx.block_idx.y)

        # ============ channel-independent prologue (shared by CPT lanes) =====
        # The cache-line address is workgroup-uniform, so decode loads it scalar
        # rather than having every lane repeat the same load. Verify keeps the
        # per-lane load, which measured better inside its longer pipeline.
        in_coord = _load_i32(csi_r, idx_seq * scsi, is_scalar=not IS_SPEC)

        seq_ok = in_coord < num_cache_lines
        if fx.const_expr(HAS_NULL_BLOCK):
            seq_ok = seq_ok & (in_coord != null_block_id)

        # rollback point: spec -> num_accept_tokens - 1, decode -> 0.
        if fx.const_expr(IS_SPEC):
            offset_dyn = _load_i32(nacc_r, idx_seq) - fx.Int32(1)
        else:
            offset_dyn = fx.Int32(0)

        if fx.const_expr(SAVE_ANY):
            inter_coord = _load_i32(isi_r, idx_seq * sisi)

        # ---- EAGLE tree: parent map + per-token tap chain -------------------
        # Channel-independent, so built once here; the per-channel loop below only
        # turns the tap indices into loads.
        if fx.const_expr(TREE):
            rnt_base = idx_seq * srnt_seq
            rns_base = idx_seq * srns_seq
            rnt_v = [
                _load_i32(rnt_r, rnt_base + fx.Int32(t) * srnt_tok)
                for t in fx.range_constexpr(S)
            ]
            rns_v = [
                _load_i32(rns_r, rns_base + fx.Int32(t) * srns_tok)
                for t in fx.range_constexpr(S)
            ]

            par = [fx.Int32(0) for _ in fx.range_constexpr(S)]

            def _par_gather(idx):
                """parent[idx] for a runtime idx, as a select chain."""
                got = par[0]
                for k in fx.range_constexpr(S):
                    if fx.const_expr(k > 0):
                        got = (idx == fx.Int32(k)).select(par[k], got)
                return got

            # chain[t] entry j says how tap j+1 is reached from tap j: the parent
            # x token, or a history column once the walk leaves this chunk.
            chain = []
            for t in fx.range_constexpr(S):
                # A child's parent is the current token, a sibling inherits it.
                # -1 never matches a slot index, so the guard folds into the test.
                for k in fx.range_constexpr(S):
                    par[k] = (rnt_v[t] == fx.Int32(k)).select(fx.Int32(t), par[k])
                p_cur = par[t]
                for k in fx.range_constexpr(S):
                    par[k] = (rns_v[t] == fx.Int32(k)).select(p_cur, par[k])

                steps = []
                cur = fx.Int32(t)
                cur_const = t  # None once the walk depends on runtime parents
                for _j in fx.range_constexpr(W - 1):
                    if fx.const_expr(cur_const is not None):
                        if fx.const_expr(cur_const > 0):
                            pidx = par[cur_const]
                            steps.append((True, pidx, None))
                            cur, cur_const = pidx, None
                        else:
                            # Token 0 and below only ever read history columns.
                            steps.append((False, None, cur_const))
                            cur_const = cur_const - 1
                            cur = fx.Int32(cur_const)
                    else:
                        cond = cur > fx.Int32(0)
                        pidx = _par_gather(cur)
                        steps.append((cond, pidx, cur))
                        cur = cond.select(pidx, cur - fx.Int32(1))
                chain.append(steps)

        # ================= per-channel work ==================================
        def _channel(gfeat):
            active = (gfeat < dim) & seq_ok

            cs_base = in_coord * scs_seq + gfeat * scs_dim
            x_base = idx_seq * sx_seq + gfeat * sx_dim
            o_base = idx_seq * so_seq + gfeat * so_dim

            # ============ PHASE 1: issue ALL loads up front ==================
            # All loads before any compute, so they stay in flight together. The
            # conv_state addresses depend on csi/nacc, still in flight, so the
            # loads independent of those go first to hide that round-trip.
            x_raw = [
                _load_elem(x_r, x_base + fx.Int32(tok) * sx_tok)
                for tok in fx.range_constexpr(S)
            ]

            w_base = gfeat * sw_dim
            if fx.const_expr(WEIGHT_VEC):
                wv = fx.Vector(
                    buffer_ops.buffer_load(w_r, w_base, vec_width=W, dtype=elem_dtype)
                )
                w_raw = [wv[j] for j in fx.range_constexpr(W)]
            else:
                w_raw = [
                    _load_elem(w_r, w_base + fx.Int32(j) * sw_width)
                    for j in fx.range_constexpr(W)
                ]

            if fx.const_expr(HAS_BIAS):
                bias_raw = _load_elem(b_r, gfeat)

            # STEP 1 history columns, starting at conv_state[offset + 0].
            col_raw = [
                _load_elem(cs_r, cs_base + (offset_dyn + fx.Int32(k)) * scs_tok)
                for k in fx.range_constexpr(W - 1)
            ]

            # Tree taps live at runtime indices, so they cannot join the static
            # batch above; issue them right after so they still overlap.
            if fx.const_expr(TREE):
                tree_x = []
                for t in fx.range_constexpr(S):
                    per_tok = []
                    for j in fx.range_constexpr(W - 1):
                        _cond, pidx, _hist = chain[t][j]
                        if fx.const_expr(pidx is None):
                            per_tok.append(None)
                        else:
                            per_tok.append(_load_elem(x_r, x_base + pidx * sx_tok))
                    tree_x.append(per_tok)

            # ================= PHASE 2: convert + convolution ================
            cols = [c.to(fx.Float32) for c in col_raw]
            w_col = [w.to(fx.Float32) for w in w_raw]
            xt = [v.to(fx.Float32) for v in x_raw]
            bias_f = (
                bias_raw.to(fx.Float32) if fx.const_expr(HAS_BIAS) else fx.Float32(0.0)
            )

            def _mac(acc, w, v):
                # SGLang's Triton kernel keeps only the accumulator in fp32, so
                # each product is rounded to the input dtype before it is added.
                # Reproducing that rounding is what makes this bit-exact.
                return acc + (w * v).to(elem_dtype).to(fx.Float32)

            def _silu(acc):
                # The bare intrinsics, which is what the Triton oracle lowers to
                # and what keeps the parity suite bit-exact. Deliberately not
                # fx.math.exp2, whose denormal/range fixup this kernel never
                # needs.
                f32_ty = fx.Float32.ir_type
                ex = fx.Float32(
                    rocdl.exp2(f32_ty, (acc * fx.Float32(-_LOG2E)).ir_value())
                )
                return acc * fx.Float32(rocdl.rcp(f32_ty, fx.Float32(1.0) + ex))

            def _hist(cur, cur_const):
                """History column feeding a tap that left the current chunk.

                ``cur == 0`` picks the newest history column, ``-1`` the one
                before it, and anything older saturates at the oldest column.
                """
                if fx.const_expr(cur_const is not None):
                    return cols[max(W - 2 + cur_const, 0)]
                got = cols[0]
                for cval in fx.range_constexpr(-(W - 3), 1):
                    got = (cur == fx.Int32(cval)).select(cols[W - 2 + cval], got)
                return got

            inter_vals = []  # per token: the W-1 window slots to snapshot
            o_vals = []

            if fx.const_expr(TREE):
                # Convolve along the parent chain: itself * w[W-1],
                # parent * w[W-2], grand-parent * w[W-3], ...
                for t in fx.range_constexpr(S):
                    acc = bias_f
                    tap = xt[t]
                    slots = [None] * (W - 1)
                    for j in fx.range_constexpr(W):
                        acc = _mac(acc, w_col[W - 1 - j], tap)
                        if fx.const_expr(SAVE_INTER and (W - j - 2) >= 0):
                            slots[W - j - 2] = tap
                        if fx.const_expr(j < W - 1):
                            cond, _pidx, hist_cur = chain[t][j]
                            if fx.const_expr(cond is True):
                                tap = tree_x[t][j].to(fx.Float32)
                            elif fx.const_expr(cond is False):
                                tap = _hist(None, hist_cur)
                            else:
                                tap = cond.select(
                                    tree_x[t][j].to(fx.Float32),
                                    _hist(hist_cur, None),
                                )
                    if fx.const_expr(SILU):
                        acc = _silu(acc)
                    o_vals.append(acc.to(elem_dtype))
                    inter_vals.append(slots)
            else:
                # Chain: a register window that slides left by one per token.
                window = list(cols)
                for idx_token in fx.range_constexpr(S):
                    win = window + [xt[idx_token]]  # W taps: [hist..., current]
                    acc = bias_f
                    for j in fx.range_constexpr(W):
                        acc = _mac(acc, w_col[j], win[j])
                    if fx.const_expr(SILU):
                        acc = _silu(acc)
                    o_vals.append(acc.to(elem_dtype))
                    window = win[1:]
                    inter_vals.append(list(window))

            # ================= PHASE 3: guarded stores =======================
            # Rolled window slot t: history while t + S < ST, else the new x
            # token. The roll condition bounds the history index to W-2, which
            # col_raw already holds, so no slot needs a re-read.
            cs_vals = [
                col_raw[SHIFT + t] if fx.const_expr((t + S) < ST) else x_raw[t - VAL]
                for t in fx.range_constexpr(ST)
            ]

            def _store_run(vals, rsrc, base, stride, total, vectorize):
                # Callers only set ``vectorize`` when the axis stride is 1, so
                # the slot offset is a constant here; spelling it as one keeps
                # the byte address out of the runtime path.
                if fx.const_expr(vectorize):
                    for start, wd in _vec_chunks(total):
                        off = (
                            base
                            if fx.const_expr(start == 0)
                            else base + fx.Int32(start)
                        )
                        if fx.const_expr(wd == 1):
                            buffer_ops.buffer_store(vals[start], rsrc, off)
                        else:
                            chunk = fx.Vector.from_elements(
                                [vals[start + j] for j in range(wd)], elem_dtype
                            )
                            buffer_ops.buffer_store(chunk, rsrc, off)
                else:
                    for t in fx.range_constexpr(total):
                        buffer_ops.buffer_store(
                            vals[t], rsrc, base + fx.Int32(t) * stride
                        )

            if active:
                _store_run(cs_vals, cs_r, cs_base, scs_tok, ST, CS_VEC)
                _store_run(o_vals, o_r, o_base, so_tok, S, O_VEC)

                if fx.const_expr(SAVE_STREAM):
                    # Every tap any token's window can reach, in one run. History
                    # slot 0 is the only column no window reaches, so dropping it
                    # lines the stream up with conv_state's rolled layout.
                    _store_run(
                        [col_raw[k] for k in fx.range_constexpr(1, W - 1)] + x_raw,
                        inter_r,
                        inter_coord * si_seq + gfeat * si_dim,
                        si_win,
                        STREAM_LEN,
                        I_VEC,
                    )

                if fx.const_expr(SAVE_INTER):
                    i_base = inter_coord * si_seq + gfeat * si_dim
                    for t in fx.range_constexpr(S):
                        row = i_base + fx.Int32(t) * si_step
                        _store_run(
                            [v.to(elem_dtype) for v in inter_vals[t]],
                            inter_r,
                            row,
                            si_win,
                            W - 1,
                            I_VEC,
                        )

        for c in fx.range_constexpr(CPT):
            _channel(pid_y * fx.Int32(BN * CPT) + fx.Int32(c * BN) + tid)

        # The fused parent map is channel-independent; one lane writes it.
        if fx.const_expr(TREE):
            lane0 = (tid == fx.Int32(0)) & (pid_y == fx.Int32(0))
            if lane0 & seq_ok:
                rpt_base = idx_seq * srpt_seq
                for k in fx.range_constexpr(S):
                    buffer_ops.buffer_store(
                        par[k], rpt_r, rpt_base + fx.Int32(k) * srpt_tok
                    )

    @flyc.jit
    def launch(
        x_ptr: fx.Tensor,
        w_ptr: fx.Tensor,
        bias_ptr: fx.Tensor,
        cs_ptr: fx.Tensor,
        csi_ptr: fx.Tensor,
        nacc_ptr: fx.Tensor,
        o_ptr: fx.Tensor,
        inter_ptr: fx.Tensor,
        isi_ptr: fx.Tensor,
        rnt_ptr: fx.Tensor,
        rns_ptr: fx.Tensor,
        rpt_ptr: fx.Tensor,
        dim: fx.Int32,
        num_cache_lines: fx.Int32,
        null_block_id: fx.Int32,
        sx_seq: fx.Int32,
        sx_dim: fx.Int32,
        sx_tok: fx.Int32,
        sw_dim: fx.Int32,
        sw_width: fx.Int32,
        scs_seq: fx.Int32,
        scs_dim: fx.Int32,
        scs_tok: fx.Int32,
        scsi: fx.Int32,
        so_seq: fx.Int32,
        so_dim: fx.Int32,
        so_tok: fx.Int32,
        si_seq: fx.Int32,
        si_step: fx.Int32,
        si_dim: fx.Int32,
        si_win: fx.Int32,
        sisi: fx.Int32,
        srnt_seq: fx.Int32,
        srnt_tok: fx.Int32,
        srns_seq: fx.Int32,
        srns_tok: fx.Int32,
        srpt_seq: fx.Int32,
        srpt_tok: fx.Int32,
        batch: fx.Int32,
        grid_y_dim: fx.Int32,
        stream: fx.Stream,
    ):
        update_kernel(
            x_ptr,
            w_ptr,
            bias_ptr,
            cs_ptr,
            csi_ptr,
            nacc_ptr,
            o_ptr,
            inter_ptr,
            isi_ptr,
            rnt_ptr,
            rns_ptr,
            rpt_ptr,
            dim,
            num_cache_lines,
            null_block_id,
            sx_seq,
            sx_dim,
            sx_tok,
            sw_dim,
            sw_width,
            scs_seq,
            scs_dim,
            scs_tok,
            scsi,
            so_seq,
            so_dim,
            so_tok,
            si_seq,
            si_step,
            si_dim,
            si_win,
            sisi,
            srnt_seq,
            srnt_tok,
            srns_seq,
            srns_tok,
            srpt_seq,
            srpt_tok,
        ).launch(grid=(batch, grid_y_dim, 1), block=(BN, 1, 1), stream=stream)

    launch._bn = BN
    launch._cpt = CPT
    launch._state_len = ST
    launch._seqlen = S
    return launch


@functools.cache
def compile_causal_conv1d_update_sglang(
    width: int,
    seqlen: int,
    has_bias: bool,
    silu: bool,
    is_spec_decoding: bool,
    has_null_block: bool,
    block_n: int,
    dtype_str: str,
    weight_contig: bool,
    cs_vec: bool,
    o_vec: bool,
    i_vec: bool,
    save_intermediate: bool,
    save_stream: bool,
    has_tree: bool,
    channels_per_thread: int,
):
    """Memoized :func:`build_causal_conv1d_update_sglang_module`.

    Every argument is a compile-time specialization, so one launcher is built
    per distinct configuration and reused for the life of the process.
    """
    return build_causal_conv1d_update_sglang_module(
        width,
        seqlen,
        has_bias,
        silu,
        is_spec_decoding,
        has_null_block,
        block_n,
        dtype_str,
        weight_contig,
        cs_vec,
        o_vec,
        i_vec,
        save_intermediate,
        save_stream,
        has_tree,
        channels_per_thread,
    )
