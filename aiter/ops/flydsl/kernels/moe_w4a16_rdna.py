# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL W4A16 grouped (MoE) GEMM for RDNA3.5 (gfx115x, wave32).

One kernel serves both MoE GEMMs: ``out[r, :] = W[expert(r)] @ x[row(r), :]``
for every routed row ``r``, with rows grouped by expert into 16-row blocks by
``moe_align_block_size``.

A wave computes a 16 x (16 * REG_N) output tile with bf16 WMMA. Weights stay
packed int4 until the WMMA operand: a nibble ``q`` is placed in the mantissa
of the bf16 value ``1 + q / 16`` by shift/mask/or, so no int->float conversion
is needed. Per quantization group the kernel therefore accumulates
``P = sum_k x_k * (1 + q_k / 16)`` and ``S = sum_k x_k`` (WMMA against ones),
and folds scale and zero point in f32:

    sum_k x_k * s * (q_k - zp) = (16 * s) * P - s * (16 + zp) * S

``pack_w4a16_weights`` produces the kernel's weight and coefficient layout.
"""

import functools
from types import SimpleNamespace

import torch


def _vector_store(value, memref, indices):
    from flydsl._mlir.dialects import vector as _vector
    from flydsl.expr import arith

    idx = [arith.unwrap(i, index=True) for i in indices]
    _vector.store(arith.unwrap(value), arith.unwrap(memref), idx)


def _vector_load_op(result_type, memref, indices):
    from flydsl._mlir.dialects import vector as _vector
    from flydsl.expr import arith

    idx = [arith.unwrap(i, index=True) for i in indices]
    return _vector.LoadOp(result_type, arith.unwrap(memref), idx).result


# flydsl.expr.vector.store / load_op of FlyDSL 0.1.8; FlyDSL 0.5 removed them.
vector = SimpleNamespace(store=_vector_store, load_op=_vector_load_op)

BLOCK_M = 16
REG_N = 4
TILE_N = 16 * REG_N
# K-steps whose loads are kept in flight ahead of the one being computed.
_LOOKAHEAD = 6
# Experts per token of the reduce kernel: every second row of a WMMA A operand.
MAX_REDUCE_ROWS = 8
# Up to this many tokens every routed row gets its own wave (the row and
# reduce kernels); more tokens are grouped into BLOCK_M-row blocks per expert.
ROW_PATH_MAX_TOKENS = 5

# A packed dword holds the nibbles of four element pairs, pair p at bits 4p
# and 4p + 16. In an operand pair a nibble sits in the top mantissa bits of
# 1.0: (1.0, lowest mantissa bit used) for bf16 and float16.
_OPERAND_FORMAT = {False: (0x3F80, 3), True: (0x3C00, 6)}

# DRAM channels interleave every 256 B on gfx115x. Waves of a launch read
# their tiles in lockstep, so tile strides that are a multiple of 512 B would
# make them all pull from the same channel at the same time.
_CHANNEL_BYTES = 256
# The same holds across experts: the one-token reduce kernel reads the same
# tile offset of every routed expert at once, so the expert stride gets one
# more granule.
_EXPERT_PAD_DWORDS = _CHANNEL_BYTES // 4
# slc | dlc: weights of one token are read once, so they are loaded without
# being kept in the caches, like a non-temporal load.
_NON_TEMPORAL = 6


def _tile_stride(payload_dwords: int) -> int:
    """Dwords per tile, padded to an odd multiple of the channel granule."""
    granules = -(-payload_dwords * 4 // _CHANNEL_BYTES)
    return (granules | 1) * _CHANNEL_BYTES // 4


def pad_expert_stride(t: torch.Tensor) -> torch.Tensor:
    """Return ``t`` (``[E, tiles, tile stride]``) as a view of a buffer whose
    expert stride is ``_EXPERT_PAD_DWORDS`` longer."""
    E, flat = t.size(0), t[0].numel()
    extra = _EXPERT_PAD_DWORDS * 4 // t.element_size()
    buf = t.new_zeros((E, flat + extra))
    buf[:, :flat] = t.reshape(E, flat)
    return buf[:, :flat].view(t.shape)


def _expert_stride_dwords(t: torch.Tensor) -> int:
    return t.stride(0) * t.element_size() // 4


def _require(cond, msg: str) -> None:
    """Input validation that also holds under ``python -O``: the buffer
    resources are not bounds checked, so a bad input reads or writes out of
    bounds instead of failing."""
    if not cond:
        raise ValueError(msg)


def _check_coef(coef: torch.Tensor, K: int, group: int, pairs: bool) -> None:
    _require(coef.dtype == torch.float16, "coefficients are packed float16")
    _require(
        group > 0 and group % 16 == 0 and K % group == 0,
        "group size a multiple of 16 dividing K",
    )
    stride = _tile_stride(_coef_tile_dwords(K, group, pairs))
    _require(coef.size(-1) * coef.element_size() == stride * 4, "coefficient layout")


def _check_w(w: torch.Tensor, K: int) -> None:
    _require(w.dtype == torch.int32, "weights are packed int32")
    _require(w.size(-1) == _tile_stride(_w_tile_dwords(K)), "weight layout")


def row_lds_bytes(K: int, group: int) -> int:
    """LDS of the row kernel: one activation row and one group of a tile."""
    return (K // 2 + group * 8) * 4


def reduce_lds_bytes(K: int, top_k: int) -> int:
    """LDS of the reduce kernel: the activation rows of one token."""
    return top_k * K * 2


@functools.cache
def lds_capacity_bytes() -> int:
    from flydsl.runtime.device import get_rocm_arch

    from aiter.jit.utils.lds_capacity import get_lds_capacity_bytes

    return get_lds_capacity_bytes(get_rocm_arch())


def _w_tile_dwords(K: int) -> int:
    return (K // 16) * 16 * 2 * REG_N


def _coef_tile_dwords(K: int, group: int, pairs: bool) -> int:
    return (K // group) * 16 * REG_N * (2 if pairs else 1) // 2


def _kernel_name(
    kind: str, K: int, N: int, group: int, pairs: bool, silu: bool, fp16: bool
) -> str:
    """Symbol name of a kernel build; it is what profiles show."""
    name = f"flydsl_rdna_moe_{kind}_k{K}_n{N}_g{group}"
    name += ("_zp" if pairs else "") + ("_silu" if silu else "")
    return name + ("_f16" if fp16 else "")


def _operand_ops(fp16: bool):
    """``(unpack, wmma, ones)`` for bf16 or float16 activations: ``unpack``
    turns two packed dwords into the WMMA operand ``1 + q / 16``."""
    import flydsl.expr as fx
    from flydsl.expr import rocdl
    from flydsl.expr.typing import Vector

    one, lsb = _OPERAND_FORMAT[fp16]
    one_pair = one << 16 | one
    mask = (0xF << lsb) * 0x10001
    op = rocdl.wmma_f32_16x16x16_f16 if fp16 else rocdl.wmma_f32_16x16x16_bf16

    def unpack(wv):
        words = []
        for i in range(2):
            for p in range(4):
                sh = lsb - 4 * p
                w = wv[i] << sh if sh >= 0 else wv[i] >> -sh
                words.append((w & mask) | one_pair)
        return Vector.from_elements(words, fx.Int32).bitcast(fx.Int16)

    def wmma(a, b, acc):
        if fp16:
            a, b = a.bitcast(fx.Float16), b.bitcast(fx.Float16)
        return op(acc.type, a, b, acc).result

    return unpack, wmma, fx.full(16, one, fx.Int16)


def _scale_terms(c, pairs: bool):
    """Float32 ``(16 s, -s (16 + zp))`` from the loaded coefficient(s)."""
    import flydsl.expr as fx

    if pairs:
        cf = c.to(fx.Float32)
        return cf[0], cf[1]
    s16 = fx.Float16(c).to(fx.Float32)
    return s16, s16 * -1.5


def gate_up_tile_order(n_out: int, device: torch.device) -> torch.Tensor:
    """Column order that puts each gate column 16 * REG_N / 2 columns before
    its up column inside one tile, as the fused SiLU-and-multiply needs."""
    half = TILE_N // 2
    cols = torch.arange(n_out, device=device).view(2, -1, half)
    return cols.transpose(0, 1).reshape(-1)


def pack_w4a16_weights(
    q: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack unsigned int4 weights for the kernel.

    Args:
        q: ``[E, N, K]`` unsigned nibbles (0..15).
        scale: ``[E, N, K // group]`` group scales; the group size is a
            multiple of 16.
        zp: ``[E, N, K // group]`` group zero points (0..15), or None for
            symmetric weights (zero point 8).

    Returns:
        Per ``(expert, 64-column tile)`` rows of dwords: ``w`` int32 with the
        nibbles ordered ``[K/16, REG_N, 16, 2]`` and ``coef`` float16 pairs
        ``(16 s, -s (16 + zp))`` ordered ``[K/group, REG_N, 16]``. Rows are padded
        to the tile stride. Without ``zp`` only ``16 s`` is stored: the second
        coefficient is -1.5 times the first.

    """
    E, N, K = q.shape
    # [E, N, K // group]; checkpoints often store scales as [K // group, N].
    _require(scale.dim() == 3 and scale.shape[:2] == q.shape[:2], "scale shape")
    G = scale.shape[2]
    # Values outside a nibble would spill into the neighbouring ones.
    _require(int(q.min()) >= 0 and int(q.max()) <= 15, "q must be in 0..15")
    if zp is not None:
        _require(zp.shape == scale.shape, "zp must have the shape of scale")
        _require(int(zp.min()) >= 0 and int(zp.max()) <= 15, "zp must be in 0..15")
    tiles = N // TILE_N
    _require(
        N % TILE_N == 0 and K % G == 0 and (K // G) % 16 == 0,
        "N a multiple of 64, group size a multiple of 16 dividing K",
    )
    # [E, tile, rn, lane, kstep, dword, half, pair] -> nibble at 4 * pair + 16 * half
    qv = q.to(torch.int64).view(E, tiles, REG_N, 16, K // 16, 2, 4, 2)
    pair = torch.arange(4, device=q.device).view(4, 1)
    half = torch.arange(2, device=q.device).view(1, 2)
    w = (qv << (4 * pair + 16 * half)).sum(dim=(-1, -2))
    w = torch.where(w >= 2**31, w - 2**32, w).to(torch.int32)
    w = w.permute(0, 1, 4, 2, 3, 5).reshape(E, tiles, -1)

    s = scale.to(torch.float32)
    coef = (16.0 * s).unsqueeze(-1)
    if zp is not None:
        coef = torch.cat((coef, -s.unsqueeze(-1) * (16.0 + zp.unsqueeze(-1))), dim=-1)
    coef = coef.view(E, tiles, REG_N, 16, G, -1).permute(0, 1, 4, 2, 3, 5)
    coef = coef.to(torch.float16).reshape(E, tiles, -1)

    def _pad(t: torch.Tensor, dwords: int) -> torch.Tensor:
        per_dword = 4 // t.element_size()
        extra = (_tile_stride(dwords) - dwords) * per_dword
        return torch.nn.functional.pad(t, (0, extra)).contiguous()

    return (
        _pad(w, _w_tile_dwords(K)),
        _pad(coef, _coef_tile_dwords(K, K // G, zp is not None)),
    )


@functools.cache
def _build_gemm(
    K: int,
    N: int,
    x_row_div: int,
    group: int,
    silu_mul: bool,
    w_expert: int,
    c_expert: int,
    coef_pairs: bool,
    fp16: bool,
):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import gpu, math, range_constexpr

    from aiter.ops.flydsl.kernels import buffer_ops

    _require(
        K % group == 0 and group % 16 == 0 and N % TILE_N == 0,
        "group size a multiple of 16 dividing K, N a multiple of 64",
    )
    out_type = fx.Float16 if fp16 else fx.BFloat16
    n_groups = K // group
    steps = group // 16  # WMMA K-steps per group
    k_dwords = K // 2
    w_step = 16 * 2 * REG_N  # dwords per K-step of one output tile
    w_tile = _tile_stride(_w_tile_dwords(K))
    # Coefficient offsets are in float16 elements.
    cw = 2 if coef_pairs else 1
    c_group = 16 * REG_N * cw
    c_tile = 2 * _tile_stride(_coef_tile_dwords(K, group, coef_pairs))
    c_expert *= 2
    # With ``silu_mul`` a tile holds gate columns then the matching up columns.
    out_reg = REG_N // 2 if silu_mul else REG_N
    n_out = N // 2 if silu_mul else N
    name = _kernel_name("gemm", K, N, group, coef_pairs, silu_mul, fp16)

    @flyc.kernel(name=name)
    def gemm_kernel(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_coef: fx.Tensor,
        arg_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_num_valid: fx.Tensor,
        n_rows: fx.Int32,
    ):
        lane = gpu.thread_idx.x
        tile_n = gpu.block_idx.x
        block_m = gpu.block_idx.y
        lane16 = lane % 16
        klane = lane // 16

        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
        x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=True)
        w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(arg_coef, max_size=True)
        ids_rsrc = buffer_ops.create_buffer_resource(arg_ids, max_size=True)
        eid_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids, max_size=True)
        nv_rsrc = buffer_ops.create_buffer_resource(arg_num_valid, max_size=True)

        num_valid = buffer_ops.buffer_load(nv_rsrc, 0, vec_width=1, dtype=fx.Int32)
        row0 = block_m * BLOCK_M
        # Padding blocks and blocks without an expert (id -1) run zero groups
        # and store nothing.
        expert = buffer_ops.buffer_load(eid_rsrc, block_m, vec_width=1, dtype=fx.Int32)
        block_valid = (row0 < num_valid) & (expert >= 0)
        expert = block_valid.select(expert, fx.Int32(0))
        rid = buffer_ops.buffer_load(
            ids_rsrc, row0 + lane16, vec_width=1, dtype=fx.Int32
        )
        x_row = (block_valid & (rid < n_rows)).select(rid // x_row_div, fx.Int32(0))
        x_base = x_row * k_dwords
        w_base = expert * w_expert + tile_n * w_tile + lane16 * 2
        c_base = expert * c_expert + tile_n * c_tile + lane16 * cw

        zero = fx.full(8, 0.0, fx.Float32)

        def load_step(kstep):
            """Issue the loads of one K-step: 2 x and REG_N w vectors."""
            vecs = [
                buffer_ops.buffer_load(
                    x_rsrc, x_base + kstep * 8 + j * 4, vec_width=4, dtype=fx.Int32
                )
                for j in range_constexpr(2)
            ]
            for rn in range_constexpr(REG_N):
                vecs.append(
                    buffer_ops.buffer_load(
                        w_rsrc,
                        w_base + kstep * w_step + rn * 32,
                        vec_width=2,
                        dtype=fx.Int32,
                    )
                )
            return vecs

        def load_coef(g):
            return [
                buffer_ops.buffer_load(
                    c_rsrc,
                    c_base + g * c_group + rn * (16 * cw),
                    vec_width=cw,
                    dtype=fx.Float16,
                )
                for rn in range_constexpr(REG_N)
            ]

        unpack, wmma, ones = _operand_ops(fp16)

        # The loads of the next _LOOKAHEAD K-steps stay in flight while a step
        # is computed; across groups they are carried as loop state.
        n_ksteps = K // 16
        n_step_loads = 2 + REG_N
        n_iter = block_valid.select(fx.Int32(n_groups), fx.Int32(0))
        init = [zero for _ in range_constexpr(REG_N)]
        for d in range_constexpr(_LOOKAHEAD):
            init = init + load_step(fx.Int32(d % n_ksteps))
        init = init + load_coef(fx.Int32(0))
        for i, state in range(0, n_iter, 1, init=init):  # type: ignore[call-overload]
            tot = list(state[:REG_N])
            queue = list(state[REG_N:-REG_N])
            coef = list(state[-REG_N:])
            g = fx.Int32(i)
            # Loads past the end repeat the last ones, which are cached.
            coef_next = load_coef((g + 1 < n_groups).select(g + 1, g))

            acc = [zero for _ in range_constexpr(REG_N)]
            xsum = zero
            for step in range_constexpr(steps):
                ahead = g * steps + (step + _LOOKAHEAD)
                ahead = (ahead < n_ksteps).select(ahead, fx.Int32(n_ksteps - 1))
                queue = queue + load_step(ahead)
                cur = queue[:n_step_loads]
                queue = queue[n_step_loads:]
                a = cur[0].shuffle(cur[1], list(range(8))).bitcast(fx.Int16)
                xsum = wmma(a, ones, xsum)
                for rn in range_constexpr(REG_N):
                    wv = cur[2 + rn]
                    acc[rn] = wmma(a, unpack(wv), acc[rn])
            for rn in range_constexpr(REG_N):
                s16, szp = _scale_terms(coef[rn], coef_pairs)
                s16 = fx.full(8, s16, fx.Float32)
                szp = fx.full(8, szp, fx.Float32)
                tot[rn] = math.fma(acc[rn], s16, math.fma(xsum, szp, tot[rn]))
            results = yield tot + queue + coef_next
        tot = list(results[:REG_N])
        if silu_mul:
            tot = [
                tot[rn] / (1.0 + math.exp(-tot[rn])) * tot[rn + out_reg]
                for rn in range_constexpr(out_reg)
            ]

        for si in range_constexpr(8):
            out_rid = buffer_ops.buffer_load(
                ids_rsrc, row0 + 2 * si + klane, vec_width=1, dtype=fx.Int32
            )
            # RDNA buffer resources are not bounds checked, so padding rows
            # have to be skipped rather than masked.
            if block_valid & (out_rid < n_rows):
                out_base = out_rid * n_out + tile_n * (16 * out_reg) + lane16
                for rn in range_constexpr(out_reg):
                    buffer_ops.buffer_store(
                        tot[rn][si].to(out_type), out_rsrc, out_base + rn * 16
                    )

    @flyc.jit
    def launch(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_coef: fx.Tensor,
        arg_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_num_valid: fx.Tensor,
        n_rows: fx.Int32,
        n_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        gemm_kernel(
            arg_out,
            arg_x,
            arg_w,
            arg_coef,
            arg_ids,
            arg_expert_ids,
            arg_num_valid,
            n_rows,
        ).launch(grid=(N // TILE_N, n_blocks, 1), block=(32, 1, 1), stream=stream)

    return launch


def _copy_to_lds(lds, rsrc, n_dwords: int, lane, base=0) -> None:
    """Copy ``n_dwords`` of a buffer from dword ``base`` to the start of LDS,
    4 dwords per lane and round."""
    import flydsl.expr as fx
    from flydsl.expr import arith, range_constexpr
    from flydsl.expr.typing import T

    from aiter.ops.flydsl.kernels import buffer_ops

    for j in range_constexpr(-(-n_dwords // 128)):
        idx = lane * 4 + j * 128
        idx = (idx < n_dwords).select(idx, fx.Int32(n_dwords - 4))
        v = buffer_ops.buffer_load(rsrc, base + idx, vec_width=4, dtype=fx.Int32)
        vector.store(v, lds, [arith.index_cast(T.index, idx)])


@functools.cache
def _build_gemm_row(
    K: int,
    N: int,
    x_row_div: int,
    group: int,
    silu_mul: bool,
    w_expert: int,
    c_expert: int,
    coef_pairs: bool,
    fp16: bool,
):
    """``out[r] = W[expert_ids[r]] @ x[r // x_row_div]``, one row per block.

    A wave needs each weight in two lanes, as WMMA operands are laid out, but
    it loads a group of K-steps with 32 distinct lanes (512 B per load) into
    LDS and reads the operands from there. The activations are in LDS too.
    """
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import arith, gpu, math, range_constexpr
    from flydsl.expr.typing import T
    from flydsl.runtime.device import get_rocm_arch
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

    from aiter.ops.flydsl.kernels import buffer_ops

    _require(
        K % group == 0 and group % 16 == 0 and N % TILE_N == 0,
        "group size a multiple of 16 dividing K, N a multiple of 64",
    )
    out_type = fx.Float16 if fp16 else fx.BFloat16
    n_groups = K // group
    steps = group // 16
    k_dwords = K // 2
    w_step = 16 * 2 * REG_N
    chunk = steps * w_step  # dwords of one group of one tile
    w_tile = _tile_stride(_w_tile_dwords(K))
    # Coefficient offsets are in float16 elements.
    cw = 2 if coef_pairs else 1
    c_group = 16 * REG_N * cw
    c_tile = 2 * _tile_stride(_coef_tile_dwords(K, group, coef_pairs))
    c_expert *= 2
    out_reg = REG_N // 2 if silu_mul else REG_N
    n_out = N // 2 if silu_mul else N
    _require(row_lds_bytes(K, group) <= lds_capacity_bytes(), "row kernel LDS")
    allocator = SmemAllocator(None, arch=get_rocm_arch(), global_sym_name="moe_row")
    # One wave per workgroup: workgroups of several waves (and their larger
    # LDS allocation) cost up to 5% of a decode step. Declaring the block
    # size matters here (59 vs 64 us per layer at K=2048, N=1024); on the
    # reduce kernel the same declaration is slower.
    lds_dwords = k_dwords + chunk
    allocator.ptr = lds_dwords * 4
    name = _kernel_name("row", K, N, group, coef_pairs, silu_mul, fp16)

    @flyc.kernel(name=name, known_block_size=[32, 1, 1])
    def row_kernel(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_coef: fx.Tensor,
        arg_expert_ids: fx.Tensor,
    ):
        lane = gpu.thread_idx.x
        tile_n = gpu.block_idx.x
        row = gpu.block_idx.y
        lane16 = lane % 16
        klane = lane // 16

        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
        x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=True)
        w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(arg_coef, max_size=True)
        eid_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids, max_size=True)

        # Rows without an expert (id -1) run zero groups and store zeros.
        expert = buffer_ops.buffer_load(eid_rsrc, row, vec_width=1, dtype=fx.Int32)
        valid = expert >= 0
        expert = valid.select(expert, fx.Int32(0))
        w_base = expert * w_expert + tile_n * w_tile
        c_base = expert * c_expert + tile_n * c_tile + lane16 * cw

        lds = SmemPtr(allocator.get_base(), 0, T.i32, shape=(lds_dwords,)).get()
        _copy_to_lds(lds, x_rsrc, k_dwords, lane, (row // x_row_div) * k_dwords)

        def lds_at(dword):
            return [arith.index_cast(T.index, dword)]

        def load_chunk(start, spread):
            """Loads of one group; with ``spread`` 0 all lanes load the same
            16 B, for the group after the last."""
            return [
                buffer_ops.buffer_load(
                    w_rsrc,
                    w_base + start + (lane * 4 + s * w_step) * spread,
                    vec_width=4,
                    dtype=fx.Int32,
                    cache_modifier=_NON_TEMPORAL,
                )
                for s in range_constexpr(steps)
            ]

        def load_coef(g):
            return [
                buffer_ops.buffer_load(
                    c_rsrc,
                    c_base + g * c_group + rn * (16 * cw),
                    vec_width=cw,
                    dtype=fx.Float16,
                )
                for rn in range_constexpr(REG_N)
            ]

        unpack, wmma, ones = _operand_ops(fp16)

        zero = fx.full(8, 0.0, fx.Float32)
        n_iter = valid.select(fx.Int32(n_groups), fx.Int32(0))
        init = [zero for _ in range_constexpr(REG_N)]
        init = init + load_chunk(fx.Int32(0), fx.Int32(1)) + load_coef(fx.Int32(0))
        for i, state in range(0, n_iter, 1, init=init):  # type: ignore[call-overload]
            tot = list(state[:REG_N])
            cur = list(state[REG_N:-REG_N])
            coef = list(state[-REG_N:])
            g = fx.Int32(i)
            for s in range_constexpr(steps):
                vector.store(cur[s], lds, lds_at(lane * 4 + (k_dwords + s * w_step)))
            more = g + 1 < n_groups
            nxt = load_chunk(
                more.select((g + 1) * chunk, fx.Int32(0)),
                more.select(fx.Int32(1), fx.Int32(0)),
            )
            coef_next = load_coef(more.select(g + 1, g))

            acc = [zero for _ in range_constexpr(REG_N)]
            xsum = zero
            for s in range_constexpr(steps):
                a = vector.load_op(
                    T.vec(8, T.i32), lds, lds_at((g * steps + s) * 8)
                ).bitcast(fx.Int16)
                xsum = wmma(a, ones, xsum)
                for rn in range_constexpr(REG_N):
                    wv = vector.load_op(
                        T.vec(2, T.i32),
                        lds,
                        lds_at(lane16 * 2 + (k_dwords + s * w_step + rn * 32)),
                    )
                    acc[rn] = wmma(a, unpack(wv), acc[rn])
            for rn in range_constexpr(REG_N):
                s16, szp = _scale_terms(coef[rn], coef_pairs)
                s16 = fx.full(8, s16, fx.Float32)
                szp = fx.full(8, szp, fx.Float32)
                tot[rn] = math.fma(acc[rn], s16, math.fma(xsum, szp, tot[rn]))
            results = yield tot + nxt + coef_next
        tot = list(results[:REG_N])
        if silu_mul:
            tot = [
                tot[rn] / (1.0 + math.exp(-tot[rn])) * tot[rn + out_reg]
                for rn in range_constexpr(out_reg)
            ]

        # Row 0 of the WMMA result is element 0 of the first 16 lanes.
        if klane == 0:
            out_base = row * n_out + tile_n * (16 * out_reg) + lane16
            for rn in range_constexpr(out_reg):
                buffer_ops.buffer_store(
                    tot[rn][0].to(out_type), out_rsrc, out_base + rn * 16
                )

    @flyc.jit
    def launch(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_coef: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        n_rows: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        row_kernel(arg_out, arg_x, arg_w, arg_coef, arg_expert_ids).launch(
            grid=(N // TILE_N, n_rows, 1), block=(32, 1, 1), stream=stream
        )

    return launch


@functools.cache
def _build_gemm_reduce(
    K: int,
    N: int,
    group: int,
    top_k: int,
    w_expert: int,
    c_expert: int,
    coef_pairs: bool,
    fp16: bool,
):
    """Per token ``t``: ``out[t, n] = sum_e weight[t, e] * (W[expert[t, e]] @
    x[t * top_k + e])[n]``.

    Row ``2 e`` of the WMMA A operand holds ``x[e]``, so one WMMA per expert
    yields its 16 output columns in the lanes of the first half-wave and a
    single WMMA against ones yields all activation sums.

    ``x`` is copied to LDS first. Buffer loads complete in order, so loading
    ``x`` when it is needed would wait for the weight loads issued ahead.
    """
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import arith, gpu, math, range_constexpr
    from flydsl.expr.typing import T
    from flydsl.runtime.device import get_rocm_arch
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

    from aiter.ops.flydsl.kernels import buffer_ops

    _require(
        K % group == 0 and group % 16 == 0 and N % TILE_N == 0,
        "group size a multiple of 16 dividing K, N a multiple of 64",
    )
    out_type = fx.Float16 if fp16 else fx.BFloat16
    _require(top_k <= MAX_REDUCE_ROWS, "top_k above MAX_REDUCE_ROWS")
    n_groups = K // group
    steps = group // 16
    n_ksteps = K // 16
    k_dwords = K // 2
    w_step = 16 * 2 * REG_N
    w_tile = _tile_stride(_w_tile_dwords(K))
    # Coefficient offsets are in float16 elements.
    cw = 2 if coef_pairs else 1
    c_group = 16 * REG_N * cw
    c_tile = 2 * _tile_stride(_coef_tile_dwords(K, group, coef_pairs))
    c_expert *= 2
    x_dwords = top_k * k_dwords
    _require(reduce_lds_bytes(K, top_k) <= lds_capacity_bytes(), "reduce kernel LDS")
    allocator = SmemAllocator(None, arch=get_rocm_arch(), global_sym_name="moe_x")
    allocator.ptr = x_dwords * 4
    name = _kernel_name("reduce", K, N, group, coef_pairs, False, fp16)

    @flyc.kernel(name=name)
    def reduce_kernel(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_coef: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_weights: fx.Tensor,
    ):
        lane = gpu.thread_idx.x
        col16 = gpu.block_idx.x
        token = gpu.block_idx.y
        lane16 = lane % 16
        klane = lane // 16

        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
        x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=True)
        w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(arg_coef, max_size=True)
        eid_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids, max_size=True)
        tw_rsrc = buffer_ops.create_buffer_resource(arg_weights, max_size=True)

        lds = SmemPtr(allocator.get_base(), 0, T.i32, shape=(x_dwords,)).get()
        _copy_to_lds(lds, x_rsrc, x_dwords, lane, token * x_dwords)
        x_row = lane16 // 2
        x_base = (x_row < top_k).select(x_row, fx.Int32(0)) * k_dwords
        tile_n = col16 // REG_N
        rn = col16 % REG_N
        weight = []
        w_base = []
        c_base = []
        for e in range_constexpr(top_k):
            expert = buffer_ops.buffer_load(
                eid_rsrc, token * top_k + e, vec_width=1, dtype=fx.Int32
            )
            valid = expert >= 0
            tw = buffer_ops.buffer_load(
                tw_rsrc, token * top_k + e, vec_width=1, dtype=fx.Float32
            )
            # Experts with id -1 contribute nothing (the engine warm-up).
            weight.append(valid.select(tw, fx.Float32(0.0)))
            expert = valid.select(expert, fx.Int32(0))
            w_base.append(expert * w_expert + tile_n * w_tile + rn * 32 + lane16 * 2)
            c_base.append(expert * c_expert + tile_n * c_tile + (rn * 16 + lane16) * cw)

        zero = fx.full(8, 0.0, fx.Float32)

        def load_w(kstep):
            return [
                buffer_ops.buffer_load(
                    w_rsrc,
                    w_base[e] + kstep * w_step,
                    vec_width=2,
                    dtype=fx.Int32,
                    cache_modifier=_NON_TEMPORAL,
                )
                for e in range_constexpr(top_k)
            ]

        def load_coef(g):
            return [
                buffer_ops.buffer_load(
                    c_rsrc, c_base[e] + g * c_group, vec_width=cw, dtype=fx.Float16
                )
                for e in range_constexpr(top_k)
            ]

        unpack, wmma, ones = _operand_ops(fp16)

        init = [fx.Float32(0.0)]
        for d in range_constexpr(_LOOKAHEAD):
            init = init + load_w(fx.Int32(d % n_ksteps))
        init = init + load_coef(fx.Int32(0))
        for i, state in range(0, n_groups, 1, init=init):  # type: ignore[call-overload]
            tot = fx.Float32(state[0])
            queue = list(state[1:-top_k])
            coef = list(state[-top_k:])
            g = fx.Int32(i)
            coef_next = load_coef((g + 1 < n_groups).select(g + 1, g))

            acc = [zero for _ in range_constexpr(top_k)]
            xsum = zero
            for step in range_constexpr(steps):
                ahead = g * steps + (step + _LOOKAHEAD)
                ahead = (ahead < n_ksteps).select(ahead, fx.Int32(n_ksteps - 1))
                queue = queue + load_w(ahead)
                a = vector.load_op(
                    T.vec(8, T.i32),
                    lds,
                    [arith.index_cast(T.index, x_base + (g * steps + step) * 8)],
                ).bitcast(fx.Int16)
                xsum = wmma(a, ones, xsum)
                for e in range_constexpr(top_k):
                    acc[e] = wmma(a, unpack(queue[e]), acc[e])
                queue = queue[top_k:]
            for e in range_constexpr(top_k):
                s16, szp = _scale_terms(coef[e], coef_pairs)
                val = math.fma(acc[e][e], s16, xsum[e] * szp)
                tot = math.fma(val, weight[e], tot)
            results = yield [tot] + queue + coef_next
        tot = fx.Float32(results[0])

        if klane == 0:
            buffer_ops.buffer_store(
                tot.to(out_type), out_rsrc, token * N + col16 * 16 + lane16
            )

    @flyc.jit
    def launch(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_coef: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_weights: fx.Tensor,
        n_tokens: fx.Int32,
        block_x: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        # block_x is always 32. A runtime value keeps FlyDSL from inferring
        # known_block_size, which makes this kernel spill (284 B of scratch
        # instead of none at K=1024, group 32 with zero points).
        reduce_kernel(
            arg_out, arg_x, arg_w, arg_coef, arg_expert_ids, arg_weights
        ).launch(grid=(N // 16, n_tokens, 1), block=(block_x, 1, 1), stream=stream)

    return launch
