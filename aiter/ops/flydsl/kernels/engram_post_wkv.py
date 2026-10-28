# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Small-M Engram gate exchange and residual for TP2/4/8 on gfx950.

Each CTA owns one local Key branch (half a branch on TP8). Gate messages
contain an epoch and FP32 gate in one atomic i64. Ping-pong slots need no ack:
publishing n+2 requires consuming peers' n+1, hence peers have consumed n.
TP8 first exchanges FP32 reduction statistics within pairs of ranks.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, math, range_constexpr, rocdl
from flydsl.expr.typing import ReductionOp

from .communication_ops_utils import (
    fence_system_acquire,
    load_i64_acquire,
    store_i64_global_system,
)

DIM, HC, GRID_CAP, SLOT_BYTES = 5120, 4, 256, 128


def load_vec(tensor, off, width, dtype):
    sizes = {
        8: rocdl.BufferCopy64b,
        4: rocdl.BufferCopy32b,
        2: rocdl.BufferCopy16b,
        1: rocdl.BufferCopy8b,
    }
    view = fx.make_view(fx.get_iter(tensor) + fx.Int32(off), fx.make_layout(width, 1))
    frag = fx.make_rmem_tensor(width, dtype)
    fx.copy(fx.make_copy_atom(sizes[width * (dtype.width // 8)](), dtype), view, frag)
    return fx.Vector(frag.load())


def store_vec(tensor, off, value):
    view = fx.make_view(fx.get_iter(tensor) + fx.Int32(off), fx.make_layout(2, 1))
    frag = fx.make_rmem_tensor(2, fx.BFloat16)
    frag.store(value)
    fx.copy(fx.make_copy_atom(rocdl.BufferCopy32b(), fx.BFloat16), frag, view)


def message(seq, value):
    return (fx.Int64(seq) << 32) | (fx.Int64(value.bitcast(fx.Int32)) & 0xFFFFFFFF)


@flyc.jit
def receive(addr, seq):
    value = fx.Int64(load_i64_acquire(addr))
    spins = fx.Int32(0)
    while (fx.Int32(value >> 32) != seq) & (spins < fx.Int32(1000000)):
        rocdl.s_sleep(1)
        value = fx.Int64(load_i64_acquire(addr))
        spins = spins + fx.Int32(1)
    return value


@fx.struct
class Shared:
    words: fx.Array[fx.Float32, 32, 16]


@flyc.kernel
def fused_engram_post_wkv(
    Hidden: fx.Tensor,
    KV: fx.Tensor,
    Weight: fx.Tensor,
    Mask: fx.Tensor,
    Output: fx.Tensor,
    Epoch: fx.Tensor,
    Error: fx.Tensor,
    Peers: fx.Tensor,
    M: fx.Constexpr[int],
    TP: fx.Constexpr[int],
    RANK: fx.Constexpr[int],
    GRID: fx.Constexpr[int],
    HAS_MASK: fx.Constexpr[bool],
    VALUE_START: fx.Constexpr[int],
    EPS: fx.Constexpr[float],
):
    GROUPS = min(TP, HC)
    LOCAL_HC = HC // GROUPS
    KEY_DIM = DIM // (TP // GROUPS)
    KEY_ROWS = HC * DIM // TP
    OWNER = RANK if TP <= HC else RANK // 2
    WIDTH = 2 if TP <= HC else 1
    tid = fx.Int32(gpu.thread_id("x"))
    lane, wave = tid % 64, tid // 64
    block = fx.Int32(gpu.block_id("x"))
    h = rocdl.make_buffer_tensor(Hidden, max_size=False)
    kv = rocdl.make_buffer_tensor(KV, max_size=False)
    w = rocdl.make_buffer_tensor(Weight, max_size=False)
    mask = rocdl.make_buffer_tensor(Mask, max_size=False)
    out = rocdl.make_buffer_tensor(Output, max_size=False)
    ep, err, peers = fx.get_iter(Epoch), fx.get_iter(Error), fx.get_iter(Peers)
    sp = fx.SharedAllocator().allocate(Shared).peek().words.ptr
    own = fx.Int64(fx.ptr_load(peers + fx.Int32(RANK))) + fx.Int64(block) * SLOT_BYTES
    seq = fx.Int32(fx.ptr_load(ep + block))
    row = block
    while row < fx.Int32(M * LOCAL_HC):
        token, local_branch = row // LOCAL_HC, row % LOCAL_HC
        branch = fx.Int32(OWNER * LOCAL_HC) + local_branch
        seq = seq + fx.Int32(1)
        sh, sk, dot = fx.Float32(0.0), fx.Float32(0.0), fx.Float32(0.0)
        for j in range_constexpr(5):
            col = tid * WIDTH + fx.Int32(j * 512 * WIDTH)
            hcol = col + fx.Int32((RANK % 2) * KEY_DIM if TP == 8 else 0)
            hv = load_vec(h, (token * HC + branch) * DIM + hcol, WIDTH, fx.BFloat16).to(
                fx.Float32
            )
            key = load_vec(
                kv,
                token * (KEY_ROWS + DIM) + local_branch * KEY_DIM + col,
                WIDTH,
                fx.BFloat16,
            ).to(fx.Float32)
            weight = load_vec(w, branch * DIM + hcol, WIDTH, fx.Float32)
            sh = sh + (hv * hv).reduce(ReductionOp.ADD)
            sk = sk + (key * key).reduce(ReductionOp.ADD)
            dot = dot + ((hv * weight) * key).reduce(ReductionOp.ADD)
        for offset in range_constexpr(6):
            sh = sh + fx.Float32(gpu.shuffle_xor(sh, 1 << (5 - offset), 64))
            sk = sk + fx.Float32(gpu.shuffle_xor(sk, 1 << (5 - offset), 64))
            dot = dot + fx.Float32(gpu.shuffle_xor(dot, 1 << (5 - offset), 64))
        if lane == 0:
            fx.ptr_store(sh, sp + wave)
            fx.ptr_store(sk, sp + wave + 8)
            fx.ptr_store(dot, sp + wave + 16)
        gpu.barrier()
        if tid == 0:
            sh, sk, dot = fx.Float32(0.0), fx.Float32(0.0), fx.Float32(0.0)
            for i in range_constexpr(8):
                sh = sh + fx.Float32(fx.ptr_load(sp + fx.Int32(i)))
                sk = sk + fx.Float32(fx.ptr_load(sp + fx.Int32(i + 8)))
                dot = dot + fx.Float32(fx.ptr_load(sp + fx.Int32(i + 16)))
            if fx.const_expr(TP == 8):
                off = fx.Int64(64) + fx.Int64(seq & 1) * 16
                pair = (
                    fx.Int64(fx.ptr_load(peers + fx.Int32(RANK ^ 1)))
                    + fx.Int64(block) * SLOT_BYTES
                )
                norms = (fx.Int64(sk.bitcast(fx.Int32)) << 32) | (
                    fx.Int64(sh.bitcast(fx.Int32)) & 0xFFFFFFFF
                )
                store_i64_global_system(pair + off, norms)
                store_i64_global_system(pair + off + 8, message(seq, dot))
                remote = receive(own + off + 8, seq)
                if fx.Int32(remote >> 32) != seq:
                    fx.ptr_store(fx.Int32(1), err + block * 4)
                # The ready message releases the separately stored norms.
                fence_system_acquire()
                norms = fx.Int64(load_i64_acquire(own + off))
                sh = sh + fx.Int32(norms).bitcast(fx.Float32)
                sk = sk + fx.Int32(norms >> 32).bitcast(fx.Float32)
                dot = dot + fx.Int32(remote).bitcast(fx.Float32)
            rstd = math.rsqrt(sh / fx.Float32(DIM) + fx.Float32(EPS))
            rstd = rstd * math.rsqrt(sk / fx.Float32(DIM) + fx.Float32(EPS))
            score = dot * rstd * fx.Float32(DIM**-0.5)
            mag = math.sqrt(fx.max(abs(score), fx.Float32(1e-6)))
            signed = (
                (mag.bitcast(fx.Int32) & fx.Int32(0x7FFFFFFF))
                | (score.bitcast(fx.Int32) & fx.Int32(-2147483648))
            ).bitcast(fx.Float32)
            gate = fx.Float32(1.0) / (fx.Float32(1.0) + math.exp(-signed))
            if fx.const_expr(HAS_MASK):
                live = load_vec(mask, token, 1, fx.Int8)[0] != fx.Int8(0)
                gate = live.select(gate, fx.Float32(0.0))
            fx.ptr_store(gate, sp + fx.Int32(24 + OWNER))
            if fx.const_expr(TP == 2):
                peer = (
                    fx.Int64(fx.ptr_load(peers + fx.Int32(1 - RANK)))
                    + fx.Int64(block) * SLOT_BYTES
                )
                slot = fx.Int64(seq & 1) * 8
                store_i64_global_system(
                    peer + fx.Int64(RANK * 16) + slot, message(seq, gate)
                )
                remote = receive(own + fx.Int64((1 - RANK) * 16) + slot, seq)
                if fx.Int32(remote >> 32) != seq:
                    fx.ptr_store(fx.Int32(2), err + block * 4)
                fx.ptr_store(
                    fx.Int32(remote).bitcast(fx.Float32), sp + fx.Int32(24 + 1 - RANK)
                )
        gpu.barrier()
        if fx.const_expr(TP > 2):
            if (tid < GROUPS) & (tid != OWNER):
                target = tid * (TP // GROUPS) + fx.Int32(RANK % (TP // GROUPS))
                peer = (
                    fx.Int64(fx.ptr_load(peers + target)) + fx.Int64(block) * SLOT_BYTES
                )
                slot = fx.Int64(seq & 1) * 8
                gate = fx.Float32(fx.ptr_load(sp + fx.Int32(24 + OWNER)))
                store_i64_global_system(
                    peer + fx.Int64(OWNER * 16) + slot, message(seq, gate)
                )
                remote = receive(own + fx.Int64(tid) * 16 + slot, seq)
                if fx.Int32(remote >> 32) != seq:
                    fx.ptr_store(fx.Int32(2), err + block * 4 + tid)
                fx.ptr_store(
                    fx.Int32(remote).bitcast(fx.Float32), sp + fx.Int32(24) + tid
                )
            gpu.barrier()
        for part in range_constexpr(GROUPS):
            owner = (OWNER + part) % GROUPS
            out_branch = fx.Int32(owner * LOCAL_HC) + local_branch
            gate = fx.Float32(fx.ptr_load(sp + fx.Int32(24 + owner)))
            for j in range_constexpr(5):
                col = tid * 2 + fx.Int32(j * 1024)
                off = (token * HC + out_branch) * DIM + col
                value_col = (col + fx.Int32(DIM - VALUE_START)) % DIM
                hv = load_vec(h, off, 2, fx.BFloat16).to(fx.Float32)
                value = load_vec(
                    kv, token * (KEY_ROWS + DIM) + KEY_ROWS + value_col, 2, fx.BFloat16
                ).to(fx.Float32)
                store_vec(out, off, (hv + value * gate).to(fx.BFloat16))
        gpu.barrier()
        row = row + fx.Int32(GRID)
    if tid == 0:
        fx.ptr_store(seq, ep + block)


@flyc.jit
def launch(
    H: fx.Tensor,
    KV: fx.Tensor,
    W: fx.Tensor,
    Mask: fx.Tensor,
    O: fx.Tensor,
    Epoch: fx.Tensor,
    Error: fx.Tensor,
    Peers: fx.Tensor,
    M: fx.Constexpr[int],
    TP: fx.Constexpr[int],
    Rank: fx.Constexpr[int],
    Grid: fx.Constexpr[int],
    HasMask: fx.Constexpr[bool],
    ValueStart: fx.Constexpr[int],
    Eps: fx.Constexpr[float],
    stream: fx.Stream,
):
    fused_engram_post_wkv(
        H,
        KV,
        W,
        Mask,
        O,
        Epoch,
        Error,
        Peers,
        M,
        TP,
        Rank,
        Grid,
        HasMask,
        ValueStart,
        Eps,
    ).launch(grid=(Grid,), block=(512,), stream=stream)
