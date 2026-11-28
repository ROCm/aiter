# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL AttnRes training kernels (forward and backward) for gfx1250.

Each token mixes ``L = K + 1`` candidates, the ``K`` banked block representations
``bank[t, 0..K)`` and the intra-block ``prefix[t]``::

    r_l = rsqrt(mean(v_l ** 2) + eps)          z_l = r_l * <v_l, q * w>
    p   = softmax(scale * z)                   o   = sum_l p_l * v_l
    y   = o * rsqrt(mean(o ** 2) + out_eps) * w_o      (optional output RMSNorm)

The forward keeps only per-token scalars (``z``, ``r``, the log-sum-exp and the
output rstd) for the backward, which rebuilds ``o`` from the candidates.

Both kernels are persistent: a workgroup walks tokens in a grid-stride loop with
one token's L rows staged in LDS. Rows reach LDS through per-lane
GLOBAL_LOAD_ASYNC_TO_LDS_B128, so rows in flight cost no VGPRs, and each slot is
refilled with the next token's row as soon as the last pass over it is done: the
next token streams in while this one computes, and every row is read from memory
once. Each thread reads back only the bytes it loaded and consecutive lanes touch
consecutive 16-byte chunks, so the LDS traffic needs no barrier and hits all 64
banks once per 16 lanes.

Block reductions are batched across candidates: the forward reduces every
candidate's two row sums at once and the backward every candidate's ``dp`` at
once, so a token costs one or two barriers instead of one per candidate.

The backward sums the ``q * w`` and ``w_o`` gradients over its tokens, so each
workgroup writes one fp32 partial row instead of one row per token; a small
kernel then sums the partial rows into the three weight gradients.
"""

# Do not add ``from __future__ import annotations``: FlyDSL inspects annotations
# at trace time.

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from .kernels_common import create_llvm_ptr
from .tensor_shim import buf_base_i64, buf_scalar_load

WARP_SIZE = 32  # gfx1250
_VEC = 8  # bf16 per 128-bit access
_F32_VEC = 4  # fp32 per 128-bit access
_TILE_BYTES = 16
_CACHELINE = 128
_LOG2E = math.log2(math.e)
# gfx1250 cache policy: TH in bits [2:0], SCOPE in bits [4:3]. TH = 1 is
# non-temporal, for rows each kernel reads or writes once.
_TH_NT = 1
# GLOBAL_PREFETCH_B8 into GL2: SE scope, TH = 3 (high-priority temporal, address
# known to be valid).
_PREFETCH_L2 = (1 << 3) | 3

_MAX_TILES = 4  # 128-bit tiles per thread; each row costs 4 * _MAX_TILES fp32 VGPRs
_MAX_THREADS = 1024


def pick_threads(hidden_size: int) -> int:
    """Fewest whole waves that split the row into at most ``_MAX_TILES`` tiles per thread.

    The backward holds about five fp32 rows per thread, and gfx1250 has 1024 VGPRs
    per SIMD lane to share between its waves, so the per-thread row slice sets the
    occupancy. Kimi-K3's 7168 runs 224 threads (7 waves) of 4 tiles.
    """
    if hidden_size <= 0 or hidden_size % _VEC:
        raise ValueError(
            f"hidden size {hidden_size} must be a positive multiple of {_VEC}"
        )
    num_vec = hidden_size // _VEC
    for threads in range(WARP_SIZE, _MAX_THREADS + 1, WARP_SIZE):
        if num_vec % threads == 0 and num_vec // threads <= _MAX_TILES:
            return threads
    raise ValueError(f"hidden size {hidden_size} has no supported thread split")


def _bwd_stages_rows(num_bank: int, output_norm: bool) -> bool:
    # With one candidate and no output norm, dv is dy and the row is never read.
    return num_bank > 0 or output_norm


def bwd_lds_bytes(hidden_size: int, num_bank: int, output_norm: bool) -> int:
    rows = (num_bank + 1) * hidden_size * 2
    if not _bwd_stages_rows(num_bank, output_norm):
        rows = 0
    return rows + (hidden_size * 4 if output_norm else 0)


def _pin(values):
    """Materialize ``values`` in VGPRs here.

    FlyDSL has no scheduling barrier for values. Without one LLVM sinks each
    per-row update to its last use and keeps every candidate row live, which at
    K = 8 takes most of the register file and leaves one wave per SIMD.
    """
    llvm.inline_asm(
        None,
        [value.ir_value() for value in values],
        "",
        ",".join("v" for _ in values),
        has_side_effects=True,
    )


def _fma(a, b, c):
    """``a * b + c`` on 8-lane fp32 vectors, scalars broadcast.

    FlyDSL emits ``*`` and ``+`` without contraction, so ``c + a * b`` costs a
    V_PK_MUL_F32 and a V_PK_ADD_F32 where this is one V_PK_FMA_F32.
    """

    def lanes(x):
        return x if isinstance(x, fx.Vector) else fx.full(_VEC, x, fx.Float32)

    return fmath.fma(lanes(a), lanes(b), lanes(c))


def _lane_sums(s_part, base, n_values, n_waves, lane):
    """The block totals of ``n_values`` wave partials laid out ``[value][wave]``.

    Lane ``i`` adds up value ``i`` of each 32-value chunk and V_READLANE_B32 hands
    the totals to SGPRs, so a wave issues ``n_waves`` LDS reads per chunk instead of
    ``n_waves`` per value.
    """
    sums = []
    for start in range_constexpr(0, n_values, WARP_SIZE):
        count = min(WARP_SIZE, n_values - start)
        mine = start + (lane < count).select(lane, fx.Int32(0))
        total = fx.memref_load(s_part, base + mine * n_waves)
        for w in range_constexpr(1, n_waves):
            total = total + fx.memref_load(s_part, base + mine * n_waves + w)
        bits = total.bitcast(fx.Int32)
        sums += [
            fx.Int32(fx.rocdl.readlane(T.i32, bits, fx.Int32(i))).bitcast(fx.Float32)
            for i in range_constexpr(count)
        ]
    return sums


def _swap16(x, y):
    """V_PERMLANE16_SWAP_B32: lanes 16..31 of ``x`` trade places with lanes 0..15 of ``y``."""
    pair = fx.rocdl.permlane16_swap(
        llvm.StructType.get_literal([T.i32, T.i32]),
        x.bitcast(fx.Int32).ir_value(),
        y.bitcast(fx.Int32).ir_value(),
        False,
        False,
    )
    return [
        fx.Int32(llvm.extractvalue(T.i32, pair, [j])).bitcast(fx.Float32)
        for j in range(2)
    ]


def _wave_scatter_sums(values, lane):
    """Wave totals of ``2^k`` per-lane values, scattered so lane ``l`` holds value ``l >> (5 - k)``.

    Each halving step pairs value ``j`` with ``j + s/2`` between lanes ``d`` apart,
    and each lane keeps one of the two plus its partner's copy: a swap and an add
    per pair at ``d = 16``, two selects, a shuffle and an add below; xor steps then
    finish the lane groups. ``2^k - 1`` pairs, where reducing every value whole
    costs five shuffle-adds each.
    """
    vals = list(values)
    d = WARP_SIZE // 2
    while len(vals) > 1:
        half = len(vals) // 2
        nxt = []
        if d == WARP_SIZE // 2:
            for j in range(half):
                lo, hi = _swap16(vals[j], vals[j + half])
                nxt.append(lo + hi)
        else:
            upper = (lane & d) != 0
            for j in range(half):
                send = upper.select(vals[j], vals[j + half])
                keep = upper.select(vals[j + half], vals[j])
                nxt.append(keep + send.shuffle_xor(fx.Int32(d), fx.Int32(WARP_SIZE)))
        vals = nxt
        d //= 2
    total = vals[0]
    while d >= 1:
        total = total + total.shuffle_xor(fx.Int32(d), fx.Int32(WARP_SIZE))
        d //= 2
    return total


def _uniform(value):
    """A wave-uniform fp32 value moved to an SGPR, so it costs no VGPR."""
    bits = value.bitcast(fx.Int32)
    return fx.Int32(fx.rocdl.readfirstlane(T.i32, bits)).bitcast(fx.Float32)


def _load_uniform(buf, index):
    """fp32 ``buf[index]`` for a wave-uniform ``index``, loaded straight to an SGPR."""
    return fx.Int32(buf_scalar_load(buf, index)).bitcast(fx.Float32)


def _row_helpers(tid, threads, tiles, fm):
    """Per-thread row access: thread ``tid`` owns 128-bit tiles ``tid + i * threads``."""
    bf16_nt_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(_TH_NT), fx.BFloat16)
    bf16_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    f32_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    f32x1_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

    def tile_index(tile):
        return tid + tile * threads

    def load_bf16_nt(row):
        div = fx.logical_divide(row, fx.make_layout(_VEC, 1))
        values = []
        for tile in range_constexpr(tiles):
            reg = fx.make_rmem_tensor(_VEC, fx.BFloat16)
            fx.copy(bf16_nt_atom, fx.slice(div, (None, tile_index(tile))), reg)
            values.append(fx.memref_load_vec(reg).to(fx.Float32))
        return values

    def load_f32(row):
        div = fx.logical_divide(row, fx.make_layout(_F32_VEC, 1))
        values = []
        for tile in range_constexpr(tiles):
            halves = []
            for half in range_constexpr(2):
                reg = fx.make_rmem_tensor(_F32_VEC, fx.Float32)
                fx.copy(
                    f32_atom, fx.slice(div, (None, 2 * tile_index(tile) + half)), reg
                )
                halves.append(fx.memref_load_vec(reg))
            values.append(halves[0].shuffle(halves[1], list(range(_VEC))))
        return values

    def load_weight(row, bf16):
        """A weight row in fp32; weights are shared by every workgroup, so cached."""
        if const_expr(not bf16):
            return load_f32(row)
        div = fx.logical_divide(row, fx.make_layout(_VEC, 1))
        values = []
        for tile in range_constexpr(tiles):
            reg = fx.make_rmem_tensor(_VEC, fx.BFloat16)
            fx.copy(bf16_atom, fx.slice(div, (None, tile_index(tile))), reg)
            values.append(fx.memref_load_vec(reg).to(fx.Float32))
        return values

    def store_bf16_nt(row, values):
        div = fx.logical_divide(row, fx.make_layout(_VEC, 1))
        for tile in range_constexpr(tiles):
            reg = fx.make_rmem_tensor(_VEC, fx.BFloat16)
            fx.memref_store_vec(values[tile].to(fx.BFloat16), reg)
            fx.copy(bf16_nt_atom, reg, fx.slice(div, (None, tile_index(tile))))

    def store_f32(row, values):
        div = fx.logical_divide(row, fx.make_layout(_F32_VEC, 1))
        for tile in range_constexpr(tiles):
            for half in range_constexpr(2):
                reg = fx.make_rmem_tensor(_F32_VEC, fx.Float32)
                lanes = list(range(half * _F32_VEC, (half + 1) * _F32_VEC))
                fx.memref_store_vec(values[tile].shuffle(values[tile], lanes), reg)
                fx.copy(
                    f32_atom, reg, fx.slice(div, (None, 2 * tile_index(tile) + half))
                )

    def store_scalar(row, index, value):
        div = fx.logical_divide(row, fx.make_layout(1, 1))
        reg = fx.make_rmem_tensor(1, fx.Float32)
        fx.memref_store_vec(fx.full(1, value, fx.Float32), reg)
        fx.copy(f32x1_atom, reg, fx.slice(div, (None, index)))

    def dot(a, b):
        # Accumulate lane-wise across tiles, then reduce the 8 lanes once.
        acc = a[0] * b[0]
        for tile in range_constexpr(1, tiles):
            acc = _fma(a[tile], b[tile], acc)
        return acc.reduce(ReductionOp.ADD, fastmath=fm)

    return load_bf16_nt, load_weight, store_bf16_nt, store_f32, store_scalar, dot


def _lds_rows(rows, n_slots, hidden_size, tid, threads, tiles):
    """Rows staged in LDS slots: ``issue`` starts an async copy, ``read`` loads one.

    A thread's chunks sit at the same offsets in LDS as in the row, so each wave
    copies and reads 512 contiguous bytes per tile.
    """
    row_bytes = hidden_size * 2
    base = fx.Int64(fx.ptrtoint(fx.recast_iter(fx.Uint8, rows.ptr)))
    view = fx.logical_divide(
        rows.view(fx.make_layout(n_slots * hidden_size, 1)), fx.make_layout(_VEC, 1)
    )
    lds_atom = fx.make_copy_atom(fx.UniversalCopy(128), fx.BFloat16)
    lane_bytes = fx.Int64(tid * _TILE_BYTES)

    def issue(row_addr, slot):
        # The op takes the raw CPOL bits as an i32 attribute. Its immediate offset
        # applies to both addresses, so the tiles of a row share one pair of
        # pointers: an SGPR row base plus the lane's byte offset.
        cpol = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), _TH_NT)
        global_ptr = create_llvm_ptr(row_addr + lane_bytes, address_space=1)
        lds_ptr = create_llvm_ptr(base + slot * row_bytes + lane_bytes, address_space=3)
        for tile in range_constexpr(tiles):
            fx.rocdl.global_load_async_to_lds_b128(
                global_ptr, lds_ptr, tile * threads * _TILE_BYTES, aux=cpol
            )

    def read(slot):
        values = []
        for tile in range_constexpr(tiles):
            reg = fx.make_rmem_tensor(_VEC, fx.BFloat16)
            index = slot * (hidden_size // _VEC) + tid + tile * threads
            fx.copy(lds_atom, fx.slice(view, (None, index)), reg)
            values.append(fx.memref_load_vec(reg).to(fx.Float32))
        return values

    return issue, read


def _wave_sum(value):
    for shift in range_constexpr(4, -1, -1):
        value = value + value.shuffle_xor(fx.Int32(1 << shift), fx.Int32(WARP_SIZE))
    return value


def _row_addresses(bank, prefix, num_bank, bank_strides, prefix_stride):
    """``source_addr(token, src)``: byte address of candidate ``src`` of ``token``."""
    prefix_base = buf_base_i64(prefix)
    bank_base = buf_base_i64(bank) if num_bank else None

    def source_addr(token, src):
        tok64 = fx.Int64(token)
        if const_expr(src < num_bank):
            return bank_base + tok64 * (bank_strides[0] * 2) + src * bank_strides[1] * 2
        return prefix_base + tok64 * (prefix_stride * 2)

    return source_addr


def build_attn_res_fwd(
    hidden_size: int,
    num_bank: int,
    output_norm: bool,
    eps: float,
    out_eps: float,
    scale: float,
    bank_strides: tuple,
    prefix_stride: int,
    weight_bf16: tuple,
):
    """``bank_strides`` (token, candidate) and ``prefix_stride`` are in elements;
    ``weight_bf16`` says which of query, rms_weight and the output weight are bf16
    rather than fp32.

    Pass 1 takes both row sums of every candidate as its row lands and reduces
    them across the block at once; pass 2 mixes the staged rows.
    """
    threads = pick_threads(hidden_size)
    tiles = hidden_size // _VEC // threads
    n_waves = threads // WARP_SIZE
    n_src = num_bank + 1
    red = 2 * n_src
    inv_h = 1.0 / hidden_size
    q_bf16, w_bf16, wo_bf16 = weight_bf16
    name = f"attn_res_train_fwd_h{hidden_size}_k{num_bank}" + (
        "_onorm" if output_norm else ""
    )

    @fx.struct
    class SharedStorage:
        partials: fx.Array[fx.Float32, 2 * red * n_waves, 16]
        rows: fx.Array[fx.BFloat16, n_src * hidden_size, 128]

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def fwd_kernel(
        bank: fx.Tensor,
        prefix: fx.Tensor,
        query: fx.Tensor,
        rms_weight: fx.Tensor,
        out_weight: fx.Tensor,
        y: fx.Tensor,
        logit: fx.Tensor,
        rstd: fx.Tensor,
        lse: fx.Tensor,
        out_rstd: fx.Tensor,
        num_tokens: fx.Int32,
        num_groups: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        group = fx.Int32(fx.block_idx.x)
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE
        fm = arith.FastMathFlags.fast
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_part = lds.partials.view(fx.make_layout(2 * red * n_waves, 1))
        (
            _load_bf16_nt,
            load_weight,
            store_bf16_nt,
            _store_f32,
            store_scalar,
            dot,
        ) = _row_helpers(tid, threads, tiles, fm)

        # Partials alternate between two LDS buffers, so back-to-back reductions
        # on different buffers need no barrier between them.
        def block_sum(values, buf):
            if const_expr(n_waves == 1):
                return [_uniform(_wave_sum(value)) for value in values]
            base = buf * red * n_waves
            if const_expr(len(values) > 2):
                # The first 2^k values come out scattered over lane groups, one
                # store per group; the rest are reduced whole.
                k = min(len(values), WARP_SIZE).bit_length() - 1
                group = WARP_SIZE >> k
                scattered = _wave_scatter_sums(values[: 1 << k], lane)
                rest = [_wave_sum(value) for value in values[1 << k :]]
                if lane % group == 0:
                    index = base + (lane // group) * n_waves + wave
                    fx.memref_store(scattered, s_part, index)
                if lane == 0:
                    for i in range_constexpr(len(rest)):
                        index = base + ((1 << k) + i) * n_waves + wave
                        fx.memref_store(rest[i], s_part, index)
                fx.gpu.barrier()
                return _lane_sums(s_part, base, len(values), n_waves, lane)
            totals = [_wave_sum(value) for value in values]
            if lane == 0:
                for i in range_constexpr(len(totals)):
                    fx.memref_store(totals[i], s_part, base + i * n_waves + wave)
            fx.gpu.barrier()
            sums = []
            for i in range_constexpr(len(totals)):
                total = fx.memref_load(s_part, base + i * n_waves)
                for w in range_constexpr(1, n_waves):
                    total = total + fx.memref_load(s_part, base + i * n_waves + w)
                sums.append(_uniform(total))
            return sums

        issue_row, read_row = _lds_rows(
            lds.rows, n_src, hidden_size, tid, threads, tiles
        )
        source_addr = _row_addresses(
            bank, prefix, num_bank, bank_strides, prefix_stride
        )

        y_buf = fx.rocdl.make_buffer_tensor(y)
        logit_buf = fx.rocdl.make_buffer_tensor(logit)
        rstd_buf = fx.rocdl.make_buffer_tensor(rstd)
        lse_buf = fx.rocdl.make_buffer_tensor(lse)
        out_rstd_buf = fx.rocdl.make_buffer_tensor(out_rstd)
        out_weight_buf = fx.rocdl.make_buffer_tensor(out_weight)

        # Prologue: the first token's rows. Later tokens' rows are issued from
        # pass 2 of the token before, one slot at a time.
        for src in range_constexpr(n_src):
            issue_row(source_addr(group, src), src)
        q_vals = load_weight(fx.rocdl.make_buffer_tensor(query), q_bf16)
        w_vals = load_weight(fx.rocdl.make_buffer_tensor(rms_weight), w_bf16)
        qw_vals = [q_vals[t] * w_vals[t] for t in range_constexpr(tiles)]

        def token_step(token):
            next_token = token + num_groups
            partial = []
            for src in range_constexpr(n_src):
                # Async loads complete in order: leave the later rows in flight.
                fx.rocdl.s_wait_asynccnt(tiles * (n_src - 1 - src))
                vals = read_row(src)
                partial.append(dot(vals, vals))
                partial.append(dot(vals, qw_vals))
                _pin(partial[-2:])
            sums = block_sum(partial, 0)
            zs = []
            rs = []
            for src in range_constexpr(n_src):
                r = _uniform(fmath.rsqrt(sums[2 * src] * inv_h + eps, fastmath=fm))
                rs.append(r)
                zs.append(sums[2 * src + 1] * r)
            run_max = zs[0] * scale
            for src in range_constexpr(1, n_src):
                run_max = run_max.maximumf(zs[src] * scale)
            weights = [
                fmath.exp2((zs[src] * scale - run_max) * _LOG2E, fastmath=fm)
                for src in range_constexpr(n_src)
            ]
            denom = weights[0]
            for src in range_constexpr(1, n_src):
                denom = denom + weights[src]
            inv_denom = 1.0 / denom
            if tid == 0:
                logit_row = fx.slice(logit_buf, (token, None))
                rstd_row = fx.slice(rstd_buf, (token, None))
                for src in range_constexpr(n_src):
                    store_scalar(logit_row, src, zs[src])
                    store_scalar(rstd_row, src, rs[src])
                store_scalar(lse_buf, token, run_max + fmath.log(denom, fastmath=fm))
            if const_expr(output_norm):
                w_out = load_weight(out_weight_buf, wo_bf16)

            acc = None
            for src in range_constexpr(n_src):
                vals = read_row(src)
                p = _uniform(weights[src] * inv_denom)
                if const_expr(src == 0):
                    acc = [vals[t] * p for t in range_constexpr(tiles)]
                else:
                    acc = [_fma(vals[t], p, acc[t]) for t in range_constexpr(tiles)]
                _pin(acc)
                # This thread is done with its part of the slot: refill it.
                if next_token < num_tokens:
                    issue_row(source_addr(next_token, src), src)

            out = acc
            if const_expr(output_norm):
                (out_sumsq,) = block_sum([dot(out, out)], 1)
                out_r = fmath.rsqrt(out_sumsq * inv_h + out_eps, fastmath=fm)
                out = [out[t] * out_r * w_out[t] for t in range_constexpr(tiles)]
            store_bf16_nt(fx.slice(y_buf, (token, None)), out)
            if const_expr(output_norm):
                if tid == 0:
                    store_scalar(out_rstd_buf, token, out_r)
            else:
                # The next token's reduction reuses partial buffer 0.
                fx.gpu.barrier()

        for it in range(group, num_tokens, num_groups):
            token_step(fx.Int32(it))

    @flyc.jit
    def launch_fwd(
        bank: fx.Tensor,
        prefix: fx.Tensor,
        query: fx.Tensor,
        rms_weight: fx.Tensor,
        out_weight: fx.Tensor,
        y: fx.Tensor,
        logit: fx.Tensor,
        rstd: fx.Tensor,
        lse: fx.Tensor,
        out_rstd: fx.Tensor,
        num_tokens: fx.Int32,
        num_groups: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        fwd_kernel(
            bank,
            prefix,
            query,
            rms_weight,
            out_weight,
            y,
            logit,
            rstd,
            lse,
            out_rstd,
            num_tokens,
            num_groups,
        ).launch(grid=(num_groups, 1, 1), block=(threads, 1, 1), stream=stream)

    # Two 7-wave workgroups put four waves on a SIMD; past 256 VGPRs only one fits.
    launch_fwd.compile_hints = {"waves_per_eu": 4}
    return launch_fwd


def build_attn_res_bwd(
    hidden_size: int,
    num_bank: int,
    output_norm: bool,
    scale: float,
    bank_strides: tuple,
    prefix_stride: int,
    dy_stride: int,
    weight_bf16: tuple,
):
    """``bank_strides`` (token, candidate), ``prefix_stride`` and ``dy_stride`` are
    in elements; ``weight_bf16`` is as for :func:`build_attn_res_fwd`.

    With the output norm, pass 1 rebuilds ``o`` and one reduction gives ``g``,
    pass 2 takes every ``dp_l = <v_l, g>`` in a second reduction, and pass 3
    writes ``dv``. Without it ``g`` is ``dy``, so pass 1 takes every ``dp_l`` as
    its row lands and pass 2 writes ``dv``.
    """
    threads = pick_threads(hidden_size)
    tiles = hidden_size // _VEC // threads
    n_waves = threads // WARP_SIZE
    n_src = num_bank + 1
    red = max(2, n_src)
    inv_h = 1.0 / hidden_size
    q_bf16, w_bf16, wo_bf16 = weight_bf16
    stage_rows = _bwd_stages_rows(num_bank, output_norm)
    dy_lines = (hidden_size * 2 + _CACHELINE - 1) // _CACHELINE
    # A token whose only reduction shares a buffer with the next token's first
    # needs a barrier between the two.
    end_barrier = (n_src == 1) == output_norm
    name = f"attn_res_train_bwd_h{hidden_size}_k{num_bank}" + (
        "_onorm" if output_norm else ""
    )
    row_elems = n_src * hidden_size if stage_rows else _VEC
    # The w_o gradient sums in LDS rather than in a loop-carried register row,
    # which would be live across the dv pass and cost 32 VGPRs at Kimi-K3's width.
    dwo_elems = hidden_size if output_norm else _F32_VEC

    @fx.struct
    class SharedStorage:
        partials: fx.Array[fx.Float32, 2 * red * n_waves, 16]
        rows: fx.Array[fx.BFloat16, row_elems, 128]
        dwo: fx.Array[fx.Float32, dwo_elems, 16]

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def bwd_kernel(
        dy: fx.Tensor,
        bank: fx.Tensor,
        prefix: fx.Tensor,
        query: fx.Tensor,
        rms_weight: fx.Tensor,
        out_weight: fx.Tensor,
        logit: fx.Tensor,
        rstd: fx.Tensor,
        lse: fx.Tensor,
        out_rstd: fx.Tensor,
        dbank: fx.Tensor,
        dprefix: fx.Tensor,
        part_dqw: fx.Tensor,
        part_dwo: fx.Tensor,
        num_tokens: fx.Int32,
        num_groups: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        group = fx.Int32(fx.block_idx.x)
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE
        fm = arith.FastMathFlags.fast
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_part = lds.partials.view(fx.make_layout(2 * red * n_waves, 1))
        s_dwo = fx.logical_divide(
            lds.dwo.view(fx.make_layout(dwo_elems, 1)), fx.make_layout(_F32_VEC, 1)
        )
        lds_f32_atom = fx.make_copy_atom(fx.UniversalCopy(128), fx.Float32)
        (
            load_bf16_nt,
            load_weight,
            store_bf16_nt,
            store_f32,
            _store_scalar,
            dot,
        ) = _row_helpers(tid, threads, tiles, fm)
        issue_row, read_row = _lds_rows(
            lds.rows, n_src, hidden_size, tid, threads, tiles
        )
        source_addr = _row_addresses(
            bank, prefix, num_bank, bank_strides, prefix_stride
        )

        # Each thread owns its own elements of s_dwo, so the accumulation needs no
        # barrier. Chunk (tile, half) of every lane sits at one 16-byte slot, lanes
        # adjacent, so a wave's access covers each bank once per 16 lanes.
        def dwo_index(tile, half):
            return (2 * tile + half) * threads + tid

        def load_dwo(tile):
            halves = []
            for half in range_constexpr(2):
                reg = fx.make_rmem_tensor(_F32_VEC, fx.Float32)
                fx.copy(
                    lds_f32_atom, fx.slice(s_dwo, (None, dwo_index(tile, half))), reg
                )
                halves.append(fx.memref_load_vec(reg))
            return halves[0].shuffle(halves[1], list(range(_VEC)))

        def store_dwo(tile, value):
            for half in range_constexpr(2):
                reg = fx.make_rmem_tensor(_F32_VEC, fx.Float32)
                lanes = list(range(half * _F32_VEC, (half + 1) * _F32_VEC))
                fx.memref_store_vec(value.shuffle(value, lanes), reg)
                fx.copy(
                    lds_f32_atom, reg, fx.slice(s_dwo, (None, dwo_index(tile, half)))
                )

        if const_expr(output_norm):
            for tile in range_constexpr(tiles):
                store_dwo(tile, fx.Vector.filled(_VEC, 0.0, fx.Float32))

        # Partials alternate between two LDS buffers, so back-to-back reductions
        # on different buffers need no barrier between them.
        def block_sum(values, buf):
            if const_expr(n_waves == 1):
                return [_uniform(_wave_sum(value)) for value in values]
            base = buf * red * n_waves
            if const_expr(len(values) > 2):
                # The first 2^k values come out scattered over lane groups, one
                # store per group; the rest are reduced whole.
                k = min(len(values), WARP_SIZE).bit_length() - 1
                group = WARP_SIZE >> k
                scattered = _wave_scatter_sums(values[: 1 << k], lane)
                rest = [_wave_sum(value) for value in values[1 << k :]]
                if lane % group == 0:
                    index = base + (lane // group) * n_waves + wave
                    fx.memref_store(scattered, s_part, index)
                if lane == 0:
                    for i in range_constexpr(len(rest)):
                        index = base + ((1 << k) + i) * n_waves + wave
                        fx.memref_store(rest[i], s_part, index)
                fx.gpu.barrier()
                return _lane_sums(s_part, base, len(values), n_waves, lane)
            totals = [_wave_sum(value) for value in values]
            if lane == 0:
                for i in range_constexpr(len(totals)):
                    fx.memref_store(totals[i], s_part, base + i * n_waves + wave)
            fx.gpu.barrier()
            sums = []
            for i in range_constexpr(len(totals)):
                total = fx.memref_load(s_part, base + i * n_waves)
                for w in range_constexpr(1, n_waves):
                    total = total + fx.memref_load(s_part, base + i * n_waves + w)
                sums.append(_uniform(total))
            return sums

        dy_buf = fx.rocdl.make_buffer_tensor(dy)
        dprefix_buf = fx.rocdl.make_buffer_tensor(dprefix)
        if const_expr(num_bank):
            dbank_buf = fx.rocdl.make_buffer_tensor(dbank)
        logit_buf = fx.rocdl.make_buffer_tensor(logit)
        rstd_buf = fx.rocdl.make_buffer_tensor(rstd)
        lse_buf = fx.rocdl.make_buffer_tensor(lse)
        out_rstd_buf = fx.rocdl.make_buffer_tensor(out_rstd)
        query_buf = fx.rocdl.make_buffer_tensor(query)
        rms_weight_buf = fx.rocdl.make_buffer_tensor(rms_weight)
        out_weight_buf = fx.rocdl.make_buffer_tensor(out_weight)
        dy_base = buf_base_i64(dy)
        prefetch_policy = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(32), _PREFETCH_L2
        )

        def prefetch_dy(token):
            # One cacheline per thread brings the row into GL2 without VGPRs.
            if tid < dy_lines:
                addr = (
                    dy_base
                    + fx.Int64(token) * (dy_stride * 2)
                    + fx.Int64(tid * _CACHELINE)
                )
                fx.rocdl.global_prefetch(
                    create_llvm_ptr(addr, address_space=1), cache_policy=prefetch_policy
                )

        # Prologue: the first token's rows. Later tokens' rows are issued from
        # the last pass of the token before, one slot at a time.
        if const_expr(stage_rows):
            for src in range_constexpr(n_src):
                issue_row(source_addr(group, src), src)
        prefetch_dy(group)

        def token_step(token, dqw_acc):
            next_token = token + num_groups
            dy_vals = load_bf16_nt(fx.slice(dy_buf, (token, None)))
            # The weight row hits in L2; issued here, its latency hides behind pass 1.
            if const_expr(output_norm):
                w_out = load_weight(out_weight_buf, wo_bf16)
            # Per-token scalars are loaded where they are used, straight to SGPRs.
            stat_base = token * n_src
            lse_t = _load_uniform(lse_buf, token)

            def candidate_stats(src):
                z = _load_uniform(logit_buf, stat_base + src)
                p = fmath.exp2((z * scale - lse_t) * _LOG2E, fastmath=fm)
                return z, p

            def wait_row(src):
                # Async loads complete in order: leave the later rows in flight.
                fx.rocdl.s_wait_asynccnt(tiles * (n_src - 1 - src))

            def row_dots(vec, wait):
                partial = []
                for src in range_constexpr(n_src):
                    if const_expr(wait):
                        wait_row(src)
                    partial.append(dot(read_row(src), vec))
                    _pin(partial[-1:])
                    # Without a wait in between, the scheduler would hoist every
                    # row's LDS reads to the top and keep all of them live.
                    fx.rocdl.sched_barrier(0)
                return partial

            if const_expr(output_norm):
                # Pass 1: rebuild o = sum_l p_l v_l as the rows land.
                out = None
                for src in range_constexpr(n_src):
                    wait_row(src)
                    vals = read_row(src)
                    _, p = candidate_stats(src)
                    if const_expr(src == 0):
                        out = [vals[t] * p for t in range_constexpr(tiles)]
                    else:
                        out = [_fma(vals[t], p, out[t]) for t in range_constexpr(tiles)]
                    _pin(out)
                # Output RMSNorm backward and delta = <g, o>, in one reduction:
                # with xhat = o * r_o, <xhat, w_o dy> is r_o * <w_o dy, o>.
                out_r = _load_uniform(out_rstd_buf, token)
                for t in range_constexpr(tiles):
                    store_dwo(t, _fma(out[t], dy_vals[t] * out_r, load_dwo(t)))
                wdy = [w_out[t] * dy_vals[t] for t in range_constexpr(tiles)]
                b, c = block_sum([dot(wdy, out), dot(out, out)], 0)
                c1 = out_r * b * inv_h
                delta = out_r * b - out_r * out_r * c1 * c
                xhat_coef = out_r * c1
                g = [
                    _fma(out[t], -xhat_coef, wdy[t]) * out_r
                    for t in range_constexpr(tiles)
                ]
                _pin(g)
                if const_expr(n_src > 1):
                    # Pass 2: every candidate's dp = <v, g>, reduced at once.
                    dps = block_sum(row_dots(g, False), 1)
            else:
                g = dy_vals
                if const_expr(n_src > 1):
                    # Pass 1: every dp = <v, dy> as the rows land, reduced at once;
                    # delta = <dy, o> is then sum_l p_l dp_l.
                    dps = block_sum(row_dots(g, True), 0)
                    delta = None
                    for src in range_constexpr(n_src):
                        _, p = candidate_stats(src)
                        term = p * dps[src]
                        delta = term if delta is None else delta + term

            if const_expr(n_src > 1):
                q_vals = load_weight(query_buf, q_bf16)
                w_vals = load_weight(rms_weight_buf, w_bf16)
                qw_vals = [q_vals[t] * w_vals[t] for t in range_constexpr(tiles)]
            # The next token's dy lands in L2 while the dv pass runs.
            if next_token < num_tokens:
                prefetch_dy(next_token)

            # Last pass: per-candidate softmax and RMSNorm backward from the staged
            # rows. A single candidate has p = 1 whatever its score, so its score
            # gets no gradient.
            for src in range_constexpr(n_src):
                if const_expr(n_src == 1):
                    dv = g
                else:
                    vals = read_row(src)
                    z, p = candidate_stats(src)
                    r = _load_uniform(rstd_buf, stat_base + src)
                    dz = (dps[src] - delta) * (p * scale)
                    key_coef = dz * r
                    key_shift = r * z * inv_h
                    dv = [
                        _fma(_fma(vals[t], -key_shift, qw_vals[t]), key_coef, g[t] * p)
                        for t in range_constexpr(tiles)
                    ]
                    dqw_acc = [
                        _fma(vals[t], key_coef, dqw_acc[t])
                        for t in range_constexpr(tiles)
                    ]
                    _pin(dqw_acc)
                if const_expr(src < num_bank):
                    store_bf16_nt(fx.slice(dbank_buf, (token, src, None)), dv)
                else:
                    store_bf16_nt(fx.slice(dprefix_buf, (token, None)), dv)
                if const_expr(stage_rows):  # noqa: SIM102
                    # This thread is done with its part of the slot: refill it.
                    if next_token < num_tokens:
                        issue_row(source_addr(next_token, src), src)
            if const_expr(end_barrier):
                fx.gpu.barrier()
            return dqw_acc

        # With one candidate the scores get no gradient and nothing is carried;
        # otherwise the q * w gradient is. A single carried value comes back bare
        # rather than in a list.
        if const_expr(n_src > 1):
            init = [
                fx.Vector.filled(_VEC, 0.0, fx.Float32) for _ in range_constexpr(tiles)
            ]
            for it, state in range(group, num_tokens, num_groups, init=init):
                dqw_acc = token_step(fx.Int32(it), list(state))
                results = yield dqw_acc
            if const_expr(tiles == 1):
                results = [results]
            store_f32(
                fx.slice(fx.rocdl.make_buffer_tensor(part_dqw), (group, None)),
                list(results),
            )
        else:
            for it in range(group, num_tokens, num_groups):
                token_step(fx.Int32(it), [])
        if const_expr(output_norm):
            store_f32(
                fx.slice(fx.rocdl.make_buffer_tensor(part_dwo), (group, None)),
                [load_dwo(t) for t in range_constexpr(tiles)],
            )

    @flyc.jit
    def launch_bwd(
        dy: fx.Tensor,
        bank: fx.Tensor,
        prefix: fx.Tensor,
        query: fx.Tensor,
        rms_weight: fx.Tensor,
        out_weight: fx.Tensor,
        logit: fx.Tensor,
        rstd: fx.Tensor,
        lse: fx.Tensor,
        out_rstd: fx.Tensor,
        dbank: fx.Tensor,
        dprefix: fx.Tensor,
        part_dqw: fx.Tensor,
        part_dwo: fx.Tensor,
        num_tokens: fx.Int32,
        num_groups: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        bwd_kernel(
            dy,
            bank,
            prefix,
            query,
            rms_weight,
            out_weight,
            logit,
            rstd,
            lse,
            out_rstd,
            dbank,
            dprefix,
            part_dqw,
            part_dwo,
            num_tokens,
            num_groups,
        ).launch(grid=(num_groups, 1, 1), block=(threads, 1, 1), stream=stream)

    # Two 7-wave workgroups put four waves on a SIMD; past 256 VGPRs only one fits.
    launch_bwd.compile_hints = {"waves_per_eu": 4}
    return launch_bwd


_WGRAD_VECS = 8  # four-column fp32 vectors per workgroup: 128 bytes of a row
_WGRAD_LANES = 32  # threads that split the partial rows of one vector
_WGRAD_UNROLL = 16  # partial rows each lane has in flight


def build_attn_res_weight_grads(
    hidden_size: int,
    num_bank: int,
    output_norm: bool,
    weight_bf16: tuple,
    warp_size: int = WARP_SIZE,
):
    """``dquery``, ``drms`` and ``dwo`` from the backward's per-workgroup partial rows.

    A workgroup owns 32 columns, a 128-byte line of each partial row: each of 32
    lanes walks every 32nd row with 16-byte loads, sixteen rows in flight. The
    lanes of a wave that share columns (four on wave32, eight on wave64) combine
    with xor shuffles, and the wave partials add up in LDS in a fixed order, so the
    result does not depend on scheduling. ``dquery = dqw * w`` and
    ``drms = dqw * q``; with no bank the scores get no gradient and both are zero.
    """
    threads = _WGRAD_VECS * _WGRAD_LANES
    n_waves = threads // warp_size
    lane_xors = [
        _WGRAD_VECS << s for s in range((warp_size // _WGRAD_VECS).bit_length() - 1)
    ]
    blocks = -(-hidden_size // (_WGRAD_VECS * _F32_VEC))
    q_bf16, w_bf16, wo_bf16 = weight_bf16
    has_dqw = num_bank > 0
    # Always two carried sums: a single loop-carried value comes back bare.
    n_sums = 2
    name = (
        f"attn_res_train_wgrad_h{hidden_size}_k{num_bank}"
        + ("_onorm" if output_norm else "")
        + ("" if warp_size == WARP_SIZE else f"_w{warp_size}")
    )

    @fx.struct
    class SharedStorage:
        waves: fx.Array[fx.Float32, n_sums * n_waves * _WGRAD_VECS * _F32_VEC, 16]

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def wgrad_kernel(
        part_dqw: fx.Tensor,
        part_dwo: fx.Tensor,
        query: fx.Tensor,
        rms_weight: fx.Tensor,
        dquery: fx.Tensor,
        drms: fx.Tensor,
        dwo: fx.Tensor,
        num_groups: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        vec = tid % _WGRAD_VECS
        lane = tid // _WGRAD_VECS
        col_vec = fx.Int32(fx.block_idx.x) * _WGRAD_VECS + vec
        wave = tid // warp_size
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_waves = fx.logical_divide(
            lds.waves.view(
                fx.make_layout(n_sums * n_waves * _WGRAD_VECS * _F32_VEC, 1)
            ),
            fx.make_layout(_F32_VEC, 1),
        )
        f32_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        bf16_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
        lds_atom = fx.make_copy_atom(fx.UniversalCopy(128), fx.Float32)

        def vec_slot(row):
            return fx.slice(
                fx.logical_divide(row, fx.make_layout(_F32_VEC, 1)), (None, col_vec)
            )

        def load_vec(row, bf16):
            if const_expr(bf16):
                reg = fx.make_rmem_tensor(_F32_VEC, fx.BFloat16)
                fx.copy(bf16_atom, vec_slot(row), reg)
                return fx.memref_load_vec(reg).to(fx.Float32)
            reg = fx.make_rmem_tensor(_F32_VEC, fx.Float32)
            fx.copy(f32_atom, vec_slot(row), reg)
            return fx.memref_load_vec(reg)

        def store_vec(row, bf16, value):
            if const_expr(bf16):
                reg = fx.make_rmem_tensor(_F32_VEC, fx.BFloat16)
                fx.memref_store_vec(value.to(fx.BFloat16), reg)
                fx.copy(bf16_atom, reg, vec_slot(row))
            else:
                reg = fx.make_rmem_tensor(_F32_VEC, fx.Float32)
                fx.memref_store_vec(value, reg)
                fx.copy(f32_atom, reg, vec_slot(row))

        def wave_slot(i, w):
            return fx.slice(s_waves, (None, (i * n_waves + w) * _WGRAD_VECS + vec))

        def wave_sum(value):
            # The lanes of a wave that share ``vec`` sit _WGRAD_VECS threads apart.
            lanes = []
            for e in range_constexpr(_F32_VEC):
                x = value[e]
                for d in lane_xors:
                    x = x + x.shuffle_xor(fx.Int32(d), fx.Int32(warp_size))
                lanes.append(x)
            return fx.Vector.from_elements(lanes, fx.Float32)

        parts = [
            fx.rocdl.make_buffer_tensor(part_dqw),
            fx.rocdl.make_buffer_tensor(part_dwo),
        ]
        zero = fx.Vector.filled(_F32_VEC, 0.0, fx.Float32)
        init = [zero for _ in range_constexpr(n_sums)]
        sums_on = (has_dqw, output_norm)
        step = _WGRAD_LANES * _WGRAD_UNROLL
        for g, state in range(lane, num_groups, step, init=init):
            acc = list(state)
            rows = [
                fx.Int32(g) + u * _WGRAD_LANES for u in range_constexpr(_WGRAD_UNROLL)
            ]
            # A row slice moves the buffer base, so rows past the end are not
            # bounds-checked: those lanes read row 0 and drop it.
            valid = [row < num_groups for row in rows]
            safe = [ok.select(row, fx.Int32(0)) for ok, row in zip(valid, rows)]
            for i in range_constexpr(n_sums):
                if const_expr(sums_on[i]):
                    loaded = [
                        load_vec(fx.slice(parts[i], (row, None)), False) for row in safe
                    ]
                    for ok, value in zip(valid, loaded):
                        acc[i] = acc[i] + ok.select(value, zero)
            results = yield acc
        partials = [
            wave_sum(results[i]) if sums_on[i] else zero
            for i in range_constexpr(n_sums)
        ]
        if tid % warp_size < _WGRAD_VECS:
            for i in range_constexpr(n_sums):
                if const_expr(sums_on[i]):
                    reg = fx.make_rmem_tensor(_F32_VEC, fx.Float32)
                    fx.memref_store_vec(partials[i], reg)
                    fx.copy(lds_atom, reg, wave_slot(i, wave))
        fx.gpu.barrier()

        if tid < _WGRAD_VECS:
            totals = []
            for i in range_constexpr(n_sums):
                total = zero
                if const_expr(sums_on[i]):
                    for w in range_constexpr(n_waves):
                        reg = fx.make_rmem_tensor(_F32_VEC, fx.Float32)
                        fx.copy(lds_atom, wave_slot(i, w), reg)
                        value = fx.memref_load_vec(reg)
                        total = value if w == 0 else total + value
                totals.append(total)
            dqw = totals[0] if has_dqw else zero
            q_vals = load_vec(fx.rocdl.make_buffer_tensor(query), q_bf16)
            w_vals = load_vec(fx.rocdl.make_buffer_tensor(rms_weight), w_bf16)
            store_vec(fx.rocdl.make_buffer_tensor(dquery), q_bf16, dqw * w_vals)
            store_vec(fx.rocdl.make_buffer_tensor(drms), w_bf16, dqw * q_vals)
            if const_expr(output_norm):
                store_vec(fx.rocdl.make_buffer_tensor(dwo), wo_bf16, totals[1])

    @flyc.jit
    def launch_wgrad(
        part_dqw: fx.Tensor,
        part_dwo: fx.Tensor,
        query: fx.Tensor,
        rms_weight: fx.Tensor,
        dquery: fx.Tensor,
        drms: fx.Tensor,
        dwo: fx.Tensor,
        num_groups: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        wgrad_kernel(
            part_dqw, part_dwo, query, rms_weight, dquery, drms, dwo, num_groups
        ).launch(grid=(blocks, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch_wgrad
