# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gfx942 plain-paged Gemma sliding attention, independent of the gfx950 bodies.

M64 packs two query heads per KV head; each wave owns 32 packed rows. K and
transposed V use two N32 LDS slots. Legacy FP8 MFMA operands contain eight
FNUZ bytes per lane. This is a direct-launch prototype, not a dispatch target.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.flash_attn_dualwave_common import _pack_bf16_pair


def _load(ptr, offset, dtype, alignment):
    p = buffer_ops.get_element_ptr(fx.to_llvm_ptr(ptr), byte_offset=offset)
    return llvm.LoadOp(dtype, p, alignment=alignment).result


def _store(ptr, offset, value, alignment):
    p = buffer_ops.get_element_ptr(fx.to_llvm_ptr(ptr), byte_offset=offset)
    llvm.StoreOp(
        value.ir_value() if hasattr(value, "ir_value") else value,
        p,
        alignment=alignment,
    )


def _mfma(a, b, c):
    return fx.Vector(
        rocdl.mfma_f32_32x32x16_fp8_fp8(
            fx.Vector.make_type(16, fx.Float32),
            a,
            b,
            c.ir_value(),
            0,
            0,
            ir.Attribute.parse("#rocdl<mfma_perm_b none>"),
        ).result
    )


def _exp2(x):
    return fx.Float32(rocdl.exp2(T.f32, x.ir_value()))


def _pack4(values):
    lo = rocdl.cvt_pk_fp8_f32(
        T.i32, values[0].ir_value(), values[1].ir_value(), fx.Int32(0).ir_value(), 0
    )
    return fx.Int32(
        rocdl.cvt_pk_fp8_f32(
            T.i32, values[2].ir_value(), values[3].ir_value(), lo, 1
        )
    )


_workspaces = {}


def _decode_splits(batch, max_q, max_k, page_size, cu_count, forced=None):
    if max_q != 1:
        return 1
    if forced is not None:
        if not isinstance(forced, int) or not 1 <= forced <= 16:
            raise ValueError("forced split count must be in [1, 16]")
        return forced
    length = min(max_k, 1024) if max_k is not None else 0
    # The second wave of split workgroups loses to unsplit at B16 on gfx942.
    if length < 512 or batch >= 16 or batch * 16 >= cu_count:
        return 1
    return max(1, min(16, (cu_count + batch * 16 - 1) // (batch * 16),
                      length // 64, (length + page_size - 1) // page_size))


@lru_cache(maxsize=32)
def build_flash_attn_fp8_gfx942(page_size=32, _num_splits=1):
    """Build D256, Hq32/Hkv16, causal window1024 attention for page32/64.

    Each sequence has 1 <= q_len <= kv_len; decode and prefix-cached chunks
    share one launch. Lengths and page IDs are device metadata supplied by the
    caller. max_seqlen_q must bound every q_len, and each block-table row must
    cover its kv_len. Descales are per-tensor FP32, not per-head or per-page.
    """
    if page_size not in (32, 64):
        raise ValueError("gfx942 sliding attention requires page size 32 or 64")

    @fx.struct
    class SharedStorage:
        q: fx.Array[fx.Int8, 64 * 260, 16]
        k: fx.Array[fx.Int8, 2 * 32 * 260, 16]
        v: fx.Array[fx.Int8, 2 * 256 * 32, 16]
        bt: fx.Array[fx.Int32, 1024, 16]

    @flyc.kernel(known_block_size=(128, 1, 1))
    def attention(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        CuQ: fx.Tensor,
        UsedK: fx.Tensor,
        BT: fx.Tensor,
        QD: fx.Tensor,
        KD: fx.Tensor,
        VD: fx.Tensor,
        bt_stride: fx.Int32,
        scale: fx.Float32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid % 64
        wave = tid // 64
        half = lane // 32
        row = wave * 32 + lane % 32
        head = fx.Int32(gpu.block_id("x"))
        seq = fx.Int32(gpu.block_id("y"))
        split = fx.Int32(gpu.block_id("z"))
        tile = fx.Int32(0) if const_expr(_num_splits > 1) else split
        q0 = fx.Int32(fx.memref_load(CuQ, seq))
        qlen = fx.Int32(fx.memref_load(CuQ, seq + 1)) - q0
        klen = fx.Int32(fx.memref_load(UsedK, seq))
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        log_scale = (
            fx.Float32(fx.memref_load(QD, 0))
            * fx.Float32(fx.memref_load(KD, 0))
            * scale
            * 1.4426950408889634
        )
        vscale = fx.Float32(fx.memref_load(VD, 0)) / 240.0
        qp, kp, vp, op = fx.get_iter(Q), fx.get_iter(K), fx.get_iter(V), fx.get_iter(O)
        btp = fx.get_iter(BT)

        if tile * 32 < qlen:
            # A block-uniform guard keeps both waves participating in barriers.
            if const_expr(_num_splits > 1):
                lower = klen - 1024
                start = (lower > 0).select(lower, fx.Int32(0)) // 32
                end = (klen + 31) // 32
                page_lo = start * 32 // page_size
                page_hi = (end * 32 + page_size - 1) // page_size
                pages = page_hi - page_lo
                p0 = page_lo + pages * split // _num_splits
                p1 = page_lo + pages * (split + 1) // _num_splits
            for i in range_constexpr(8):
                j = tid + i * 128
                copy_page = j < (klen + page_size - 1) // page_size
                if const_expr(_num_splits > 1):
                    copy_page = (j >= p0) & (j < p1)
                if copy_page:
                    page = _load(
                        btp,
                        (fx.Int64(seq) * fx.Int64(bt_stride) + fx.Int64(j)) * 4,
                        T.i32,
                        4,
                    )
                    _store(lds.bt.ptr, j * 4, page, 4)
            for i in range_constexpr(8):
                off = tid * 16 + i * 2048
                r = off // 256
                d = off % 256
                qr = tile * 32 + r // 2
                safe_qr = (qr < qlen).select(qr, fx.Int32(0))
                src = (
                    (
                        (fx.Int64(q0) + fx.Int64(safe_qr)) * 32
                        + fx.Int64(head * 2 + r % 2)
                    )
                    * 256
                    + fx.Int64(d)
                )
                data = _load(qp, src, fx.Vector.make_type(4, fx.Int32), 16)
                # Like K, an odd dword row stride spreads MFMA reads over all banks.
                _store(lds.q.ptr, r * 260 + d, data, 4)
            gpu.barrier()

            lower = tile * 32 + klen - qlen - 1023
            start = (lower > 0).select(lower, fx.Int32(0)) // 32
            upper = (tile * 32 + 32 < qlen).select(tile * 32 + 32, qlen)
            end = (upper + klen - qlen + 31) // 32
            if const_expr(_num_splits > 1):
                start = (start > p0 * (page_size // 32)).select(
                    start, p0 * (page_size // 32)
                )
                end = (end < p1 * (page_size // 32)).select(
                    end, p1 * (page_size // 32)
                )

            def stage(block, slot):
                for i in range_constexpr(4):
                    off = tid * 16 + i * 2048
                    token = block * 32 + off // 256
                    d = off % 256
                    safe_token = (token < klen).select(token, fx.Int32(0))
                    if const_expr(_num_splits > 1):
                        # Tail lanes must read an initialized split-local page ID.
                        safe_token = (token < klen).select(token, start * 32)
                    page = fx.Int32(
                        _load(lds.bt.ptr, (safe_token // page_size) * 4, T.i32, 4)
                    )
                    src = (
                        (fx.Int64(page) * page_size + fx.Int64(safe_token % page_size))
                        * 16
                        + fx.Int64(head)
                    ) * 256 + fx.Int64(d)
                    kval = _load(kp, src, fx.Vector.make_type(4, fx.Int32), 16)
                    vval = fx.Vector(
                        _load(vp, src, fx.Vector.make_type(4, fx.Int32), 16)
                    )
                    _store(
                        lds.k.ptr, slot * (32 * 260) + (off // 256) * 260 + d, kval, 4
                    )
                    # Transpose four tokens in registers before writing a dword.
                    # XOR depth bits into the bank index for both access orders.
                    for j in range_constexpr(4):
                        word = vval[j]
                        peer = word.shuffle_xor(fx.Int32(16), fx.Int32(64))
                        pair = fx.Int32(rocdl.perm_b32(
                            peer, word,
                            ((lane // 16) % 2 == 0).select(
                                fx.Int32(0x06020400), fx.Int32(0x03070105)
                            ),
                        ))
                        peer = pair.shuffle_xor(fx.Int32(32), fx.Int32(64))
                        packed = rocdl.perm_b32(
                            peer, pair,
                            (half == 0).select(
                                fx.Int32(0x05040100), fx.Int32(0x03020706)
                            ),
                        )
                        depth = d + j * 4 + lane // 16
                        token4 = (off // 256) // 4
                        bank = (depth % 32) ^ (depth // 8)
                        _store(
                            lds.v.ptr,
                            slot * 8192 + (depth // 32) * 1024
                            + token4 * 128 + bank * 4,
                            packed,
                            4,
                        )

            q_frag = [
                _load(lds.q.ptr, row * 260 + depth * 16 + half * 8, T.i64, 4)
                for depth in range(16)
            ]
            if const_expr(_num_splits > 1):
                if start < end:
                    stage(start, fx.Int32(0))
            else:
                stage(start, fx.Int32(0))
            gpu.barrier()
            init = [fx.Float32(-1.0e30), fx.Float32(0.0)] + [
                fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(8)
            ]
            for block, state in range(start, end, fx.Int32(1), init=init):
                block = fx.Int32(block)
                slot = (block - start) % 2
                if block + 1 < end:
                    stage(block + 1, 1 - slot)
                score = fx.Vector.filled(16, 0.0, fx.Float32)
                for depth in range_constexpr(16):
                    a = _load(
                        lds.k.ptr,
                        slot * 8320 + (lane % 32) * 260 + depth * 16 + half * 8,
                        T.i64,
                        4,
                    )
                    b = q_frag[depth]
                    score = _mfma(a, b, score)
                qpos = tile * 32 + row // 2 + klen - qlen
                scores = []
                m = fx.Float32(state[0])
                for r in range_constexpr(16):
                    col = block * 32 + half * 4 + (r // 4) * 8 + r % 4
                    valid = (
                        (col <= qpos)
                        & (col > qpos - 1024)
                        & (col < klen)
                        & (tile * 32 + row // 2 < qlen)
                    )
                    s = valid.select(score[r] * log_scale, fx.Float32(-1.0e30))
                    scores.append(s)
                    m = m.maximumf(s)
                m = m.maximumf(m.shuffle_xor(fx.Int32(32), fx.Int32(64)))
                correction = _exp2(fx.Float32(state[0]) - m)
                probs = []
                psum = fx.Float32(0.0)
                for r in range_constexpr(16):
                    col = block * 32 + half * 4 + (r // 4) * 8 + r % 4
                    valid = (
                        (col <= qpos)
                        & (col > qpos - 1024)
                        & (col < klen)
                        & (tile * 32 + row // 2 < qlen)
                    )
                    p = valid.select(_exp2(scores[r] - m), fx.Float32(0.0))
                    psum = psum + p
                    probs.append(p * 240.0)
                denom = (
                    fx.Float32(state[1]) * correction
                    + psum
                    + psum.shuffle_xor(fx.Int32(32), fx.Int32(64))
                )
                words = [_pack4(probs[g * 4 : g * 4 + 4]) for g in range(4)]
                peers = [w.shuffle_xor(fx.Int32(32), fx.Int32(64)) for w in words]
                p0 = fx.Vector.from_elements(
                    [(half == 0).select(words[0], peers[1]),
                     (half == 0).select(peers[0], words[1])], fx.Int32
                ).bitcast(fx.Int64)[0]
                p1 = fx.Vector.from_elements(
                    [(half == 0).select(words[2], peers[3]),
                     (half == 0).select(peers[2], words[3])], fx.Int32
                ).bitcast(fx.Int64)[0]
                def load_v(depth, token):
                    bank = (depth % 32) ^ (depth // 8)
                    offset = slot * 8192 + (depth // 32) * 1024 + bank * 4
                    words = [
                        fx.Int32(_load(lds.v.ptr, offset + (token // 4 + i) * 128, T.i32, 4))
                        for i in range(2)
                    ]
                    return fx.Vector.from_elements(words, fx.Int32).bitcast(fx.Int64)[0]

                accum = []
                for dc in range_constexpr(8):
                    o = fx.Vector(state[dc + 2]) * fx.Vector.filled(
                        16, correction, fx.Float32
                    )
                    v0 = load_v(dc * 32 + lane % 32, half * 8)
                    o = _mfma(v0.ir_value(), p0.ir_value(), o)
                    v1 = load_v(dc * 32 + lane % 32, 16 + half * 8)
                    o = _mfma(v1.ir_value(), p1.ir_value(), o)
                    accum.append(o)
                gpu.barrier()
                result = yield [m, denom] + accum

            if const_expr(_num_splits > 1):
                if row < 2:
                    base = (((fx.Int64(seq) * 16 + fx.Int64(head)) * _num_splits
                             + fx.Int64(split)) * 2 + fx.Int64(row)) * 258
                    for dc in range_constexpr(8):
                        vals = fx.Vector(result[dc + 2])
                        for r in range_constexpr(16):
                            d = dc * 32 + half * 4 + (r // 4) * 8 + r % 4
                            _store(op, (base + fx.Int64(d)) * 4, vals[r], 4)
                    if half == 0:
                        denom = fx.Float32(result[1])
                        maximum = (denom > 0.0).select(
                            fx.Float32(result[0]), fx.Float32(float("-inf"))
                        )
                        _store(op, (base + 256) * 4, maximum, 4)
                        _store(op, (base + 257) * 4, denom, 4)
            else:
                qr = tile * 32 + row // 2
                if qr < qlen:
                    norm = vscale / fx.Float32(result[1])
                    for dc in range_constexpr(8):
                        vals = fx.Vector(result[dc + 2])
                        for r in range_constexpr(8):
                            d = dc * 32 + half * 4 + ((r * 2) // 4) * 8 + (r * 2) % 4
                            dest = (
                                (
                                    (fx.Int64(q0) + fx.Int64(qr)) * 32
                                    + fx.Int64(head * 2 + row % 2)
                                )
                                * 256
                                + fx.Int64(d)
                            ) * 2
                            packed = _pack_bf16_pair(
                                vals[r * 2] * norm, vals[r * 2 + 1] * norm
                            )
                            _store(op, dest, packed, 4)

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        CuQ: fx.Tensor,
        UsedK: fx.Tensor,
        BT: fx.Tensor,
        QD: fx.Tensor,
        KD: fx.Tensor,
        VD: fx.Tensor,
        batch: fx.Int32,
        tiles: fx.Int32,
        bt_stride: fx.Int32,
        scale: fx.Float32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        attention(Q, K, V, O, CuQ, UsedK, BT, QD, KD, VD, bt_stride, scale).launch(
            grid=(16, batch, tiles), block=(128, 1, 1), stream=stream
        )

    @flyc.kernel(known_block_size=(128, 1, 1))
    def combine(Part: fx.Tensor, O: fx.Tensor, CuQ: fx.Tensor, VD: fx.Tensor):
        tid = fx.Int32(gpu.thread_id("x"))
        lane, row = tid % 64, tid // 64
        head = fx.Int32(gpu.block_id("x"))
        seq = fx.Int32(gpu.block_id("y"))
        pp, op = fx.get_iter(Part), fx.get_iter(O)
        base = ((fx.Int64(seq) * 16 + fx.Int64(head)) * _num_splits * 2
                + fx.Int64(row)) * 258
        maximum = fx.Float32(float("-inf"))
        maxima, denoms = [], []
        for s in range_constexpr(_num_splits):
            offset = base + s * 2 * 258
            m = fx.Float32(_load(pp, (offset + 256) * 4, T.f32, 4))
            l = fx.Float32(_load(pp, (offset + 257) * 4, T.f32, 4))
            maxima.append(m)
            denoms.append(l)
            maximum = maximum.maximumf((l > 0.0).select(m, fx.Float32(float("-inf"))))
        total = fx.Float32(0.0)
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for s in range_constexpr(_num_splits):
            valid = denoms[s] > 0.0
            delta = valid.select(maxima[s], fx.Float32(0.0)) - valid.select(
                maximum, fx.Float32(0.0)
            )
            weight = valid.select(_exp2(delta), fx.Float32(0.0))
            total = total + weight * denoms[s]
            vals = fx.Vector(_load(pp, (base + s * 2 * 258 + fx.Int64(lane * 4)) * 4,
                                   fx.Vector.make_type(4, fx.Float32), 4))
            acc = acc + vals * fx.Vector.filled(4, weight, fx.Float32)
        norm = fx.Float32(fx.memref_load(VD, 0)) / (240.0 * (total > 0.0).select(
            total, fx.Float32(1.0)
        ))
        norm = (total > 0.0).select(norm, fx.Float32(0.0))
        q0 = fx.Int64(fx.memref_load(CuQ, seq))
        dest = ((q0 * 32 + fx.Int64(head * 2 + row)) * 256 + fx.Int64(lane * 4)) * 2
        for i in range_constexpr(2):
            _store(op, dest + i * 4, _pack_bf16_pair(acc[i * 2] * norm,
                                                   acc[i * 2 + 1] * norm), 4)

    @flyc.jit
    def launch_combine(Part: fx.Tensor, O: fx.Tensor, CuQ: fx.Tensor,
                       VD: fx.Tensor, batch: fx.Int32,
                       stream: fx.Stream = fx.Stream(None)):  # noqa: B008
        combine(Part, O, CuQ, VD).launch(grid=(16, batch), block=(128, 1, 1), stream=stream)

    def mod(
        q, k, v, out, *, cu_seqlens_q, seqused_k, max_seqlen_q,
        block_table, softmax_scale, q_descale, k_descale, v_descale,
        window_size=(1023, 0), causal=True, max_seqlen_k=None, softcap=0,
        workspace=None, _force_splits=None,
    ):
        import torch

        if torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0] != "gfx942":
            raise ValueError("this kernel requires gfx942")
        if (
            q.shape[1:] != (32, 256)
            or k.shape[1:] != (page_size, 16, 256)
            or v.shape != k.shape
        ):
            raise ValueError("expected Q[T,32,256], K/V[pages,page_size,16,256]")
        if not causal or window_size not in ((1023, 0), (1023, -1)) or softcap != 0:
            raise ValueError("only causal window1024 without softcap is supported")
        if (
            any(t.dtype != torch.float8_e4m3fnuz for t in (q, k, v))
            or out.dtype != torch.bfloat16
        ):
            raise ValueError("expected FNUZ Q/K/V and BF16 output")
        if any(
            not t.is_contiguous()
            for t in (q, k, v, out, cu_seqlens_q, seqused_k, block_table)
        ):
            raise ValueError("direct launcher requires contiguous tensors")
        batch = seqused_k.numel()
        if (
            out.shape != q.shape
            or cu_seqlens_q.shape != (batch + 1,)
            or seqused_k.ndim != 1
        ):
            raise ValueError("output and sequence metadata shapes do not match Q")
        if block_table.ndim != 2 or block_table.shape[0] != batch:
            raise ValueError("expected one block-table row per sequence")
        if any(t.dtype != torch.int32 for t in (cu_seqlens_q, seqused_k, block_table)):
            raise ValueError("sequence metadata must be int32")
        descales = (q_descale, k_descale, v_descale)
        if any(t.dtype != torch.float32 or t.numel() != 1 for t in descales):
            raise ValueError("descales must be per-tensor FP32 scalars")
        if any(
            t.device != q.device
            for t in (k, v, out, cu_seqlens_q, seqused_k, block_table) + descales
        ):
            raise ValueError("all tensors must be on the same device")
        if block_table.shape[1] > 1024:
            raise ValueError("block table exceeds the 1024-entry LDS capacity")
        splits = _decode_splits(
            batch, max_seqlen_q, max_seqlen_k, page_size,
            torch.cuda.get_device_properties(q.device).multi_processor_count, _force_splits,
        )
        if splits != _num_splits:
            return build_flash_attn_fp8_gfx942(page_size, splits)(
                q, k, v, out, cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k,
                max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                block_table=block_table, softmax_scale=softmax_scale,
                q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
                window_size=window_size, causal=causal, softcap=softcap,
                workspace=workspace, _force_splits=_force_splits,
            )
        stream = torch.cuda.current_stream(q.device)
        target = out
        if splits > 1:
            needed = batch * 16 * splits * 2 * 258
            if workspace is None:
                key = (q.device, stream.cuda_stream)
                workspace = _workspaces.get(key)
                if workspace is None or workspace.numel() < needed:
                    workspace = torch.empty(needed, device=q.device, dtype=torch.float32)
                    _workspaces[key] = workspace
            if (workspace.device != q.device or workspace.dtype != torch.float32
                    or not workspace.is_contiguous() or workspace.numel() < needed):
                raise ValueError("workspace must be contiguous FP32 on Q's device with sufficient capacity")
            target = workspace.view(-1)[:needed].view(batch, 16, splits, 2, 258)
        # FlyDSL packs each dynamic extent as i32. Preserve axes instead of
        # flattening multi-GiB tensors; global byte offsets remain 64-bit.
        launch(
            q.view(torch.int8),
            k.view(torch.int8),
            v.view(torch.int8),
            target,
            cu_seqlens_q,
            seqused_k,
            block_table,
            q_descale.reshape(1),
            k_descale.reshape(1),
            v_descale.reshape(1),
            seqused_k.numel(),
            splits if splits > 1 else (max_seqlen_q + 31) // 32,
            block_table.stride(0),
            softmax_scale,
            stream=stream,
        )
        if splits > 1:
            launch_combine(target, out, cu_seqlens_q, v_descale.reshape(1), batch, stream=stream)
        return out

    return mod
