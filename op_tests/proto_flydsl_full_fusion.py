#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Prototype C: the *full* fused kernel — Prototype B's compute (RMSNorm +
interleaved 3D-mrope RoPE + per-tensor FP8 quant) fused with Prototype A's
coalesced KV shuffle-layout cache write, in one kernel launch for K/V.

Why K/V get one fused kernel and Q doesn't (see ``flydsl-best-practices.md``
S6): Q has no cache write-amplification problem (plain contiguous
``[T, H_q, D]`` bf16 output), so it stays Prototype B's simple
one-wave-per-(token,head) kernel unchanged. K/V *do* have the problem (fp8
shuffle-layout KV cache), so that's the one that gets re-gridded and fused:

  grid  = (num_kv_heads, num_page_blocks)         # was (kv_head, token)
  block = 512 threads = 8 waves x 64 lanes

Phase 1 (compute, into LDS): each of the 8 waves owns pair-index (lane) and
loops over 8 of the page-block's 64 tokens (``token_local = wave_id +
it*8``), running *exactly* Prototype B's RMSNorm+RoPE+quant (K) / raw+quant
(V) math, but storing the resulting fp8 bytes into an LDS staging tile
``[BLOCK_SIZE, HEAD_SIZE]`` (i8) per K/V instead of a strided global store.

Phase 2 (coalesced write, from LDS): after one ``gpu.barrier()``, the same
512 threads are re-purposed exactly as Prototype A's ``*_coalesced`` kernels
-- K: thread t owns a 16 B run -> one dwordx4 LDS load + dwordx4 global
store (512 threads == the whole per-block byte count / 16, no loop). V: 512
threads = 4 tiles-of-16-tokens x 128 head positions -> gather 16 LDS bytes
(stride HEAD_SIZE), pack, one dwordx4 global store per thread.

Net effect vs the naive per-(token,head) compute+strided-write kernel: same
FLOPs, same *logical* bytes written, but the KV cache HBM store traffic
collapses from ~32x amplified (V) / ~2x amplified (K) scalar transactions to
fully coalesced dwordx4 ones (see Prototype A's measured ratios).

    python op_tests/proto_flydsl_full_fusion.py --tokens 4096
    python op_tests/proto_flydsl_full_fusion.py --tokens 32768 --bench
"""

import argparse
import sys
from pathlib import Path

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, range_constexpr, vector
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.tensor_shim import GTensor, _run_compiled
from aiter.test_common import checkAllclose
from aiter.utility import dtypes
from aiter import per_tensor_quant

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fused_qk_norm_mrope_cache_quant import (  # noqa: E402
    rms_norm_forward,
    apply_interleaved_rope,
    apply_rotary_emb_torch,
)

# --- worst-case workload constants (Qwen3-VL MLPerf) -------------------------
HEAD_SIZE = 128
HALF = HEAD_SIZE // 2  # 64 == WAVE
NUM_Q_HEADS = 64
NUM_KV_HEADS = 4
MROPE_SECTION = [24, 20, 20]
EPS = 1e-6
WAVE = 64
BLOCK_SIZE = 64  # KV cache page size (tokens/page)
X = 16  # fp8 shuffle innermost dim
NUM_BLOCKS = 22988
WAVES_PER_BLOCK = BLOCK_SIZE // 8  # 8 waves; each loops 8x to cover 64 tokens
KV_THREADS = WAVE * WAVES_PER_BLOCK  # 512 -- also == HEAD_SIZE*BLOCK_SIZE/16
assert HALF == WAVE
assert sum(MROPE_SECTION) == HALF
assert KV_THREADS == (HEAD_SIZE * BLOCK_SIZE) // 16  # K-coalesced needs this
assert KV_THREADS == (BLOCK_SIZE // X) * HEAD_SIZE  # V-coalesced needs this


def _v_head_stride():
    return (BLOCK_SIZE // X) * HEAD_SIZE * X


def _v_per_block():
    return NUM_KV_HEADS * _v_head_stride()


def _k_head_stride():
    return HEAD_SIZE * BLOCK_SIZE


def _k_per_block():
    return NUM_KV_HEADS * _k_head_stride()


# ============================================================================
# Q kernel: unchanged from Prototype B (no cache write, no fusion needed).
# ============================================================================
def build_q_kernel():
    H_Q, H_K, H_V, D = NUM_Q_HEADS, NUM_KV_HEADS, NUM_KV_HEADS, HEAD_SIZE

    @flyc.kernel(name="q_norm_mrope_proto")
    def kernel(
        qkv: fx.Pointer,
        positions: fx.Pointer,
        cos_sin: fx.Pointer,
        q_norm_w: fx.Pointer,
        q_out: fx.Pointer,
        num_tokens: fx.Int32,
    ):
        fm_fast = arith.FastMathFlags.fast

        def wave_reduce_add(x):
            v = x
            for sh_exp in range_constexpr(6):
                off = WAVE // (2 << sh_exp)
                peer = v.shuffle_xor(off, WAVE)
                v = v.addf(peer, fastmath=fm_fast)
            return v

        def mrope_cos_sin(tid, bid_t, num_tokens, positions_t, cos_sin_t):
            mid = tid % fx.Int32(3)
            is_mid1 = mid == fx.Int32(1)
            boundary = is_mid1.select(
                fx.Int32(MROPE_SECTION[1] * 3), fx.Int32(MROPE_SECTION[2] * 3)
            )
            in_range = tid < boundary
            use_mid = (mid != fx.Int32(0)) and in_range
            sect_idx = use_mid.select(mid, fx.Int32(0))

            pos_i64 = fx.Int64(positions_t[sect_idx * num_tokens + bid_t])
            pos = pos_i64.to(fx.Int32)
            cos_v = fx.BFloat16(cos_sin_t[pos, tid]).to(fx.Float32)
            sin_v = fx.BFloat16(cos_sin_t[pos, tid + fx.Int32(HALF)]).to(fx.Float32)
            return cos_v, sin_v

        def norm_rope(x0, x1, w0, w1, tid, bid_t, num_tokens, positions_t, cos_sin_t):
            x0f = fx.BFloat16(x0).to(fx.Float32)
            x1f = fx.BFloat16(x1).to(fx.Float32)
            sumsq_local = x0f * x0f + x1f * x1f
            sumsq = wave_reduce_add(sumsq_local)
            rstd = fmath.rsqrt(sumsq * (1.0 / D) + EPS, fastmath=fm_fast)

            w0f = fx.BFloat16(w0).to(fx.Float32)
            w1f = fx.BFloat16(w1).to(fx.Float32)
            xn0 = x0f * rstd * w0f
            xn1 = x1f * rstd * w1f

            cos_v, sin_v = mrope_cos_sin(tid, bid_t, num_tokens, positions_t, cos_sin_t)
            o0 = xn0 * cos_v - xn1 * sin_v
            o1 = xn1 * cos_v + xn0 * sin_v
            return o0, o1

        bid_x = fx.block_idx.x  # 0..H_Q-1
        bid_t = fx.block_idx.y
        tid = fx.thread_idx.x

        qkv_t = GTensor(qkv, dtype=T.bf16, shape=(1, H_Q + H_K + H_V, D))
        positions_t = GTensor(positions, dtype=T.i64, shape=(1,))
        cos_sin_t = GTensor(cos_sin, dtype=T.bf16, shape=(1, D))
        qw_t = GTensor(q_norm_w, dtype=T.bf16, shape=(D,))
        q_out_t = GTensor(q_out, dtype=T.bf16, shape=(1, H_Q, D))

        o0, o1 = norm_rope(
            qkv_t[bid_t, bid_x, tid],
            qkv_t[bid_t, bid_x, tid + fx.Int32(HALF)],
            qw_t[tid],
            qw_t[tid + fx.Int32(HALF)],
            tid,
            bid_t,
            num_tokens,
            positions_t,
            cos_sin_t,
        )
        q_out_t[bid_t, bid_x, tid] = o0.to(fx.BFloat16)
        q_out_t[bid_t, bid_x, tid + fx.Int32(HALF)] = o1.to(fx.BFloat16)

    @flyc.jit
    def launch(
        qkv: fx.Pointer,
        positions: fx.Pointer,
        cos_sin: fx.Pointer,
        q_norm_w: fx.Pointer,
        q_out: fx.Pointer,
        num_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        k = kernel(qkv, positions, cos_sin, q_norm_w, q_out, num_tokens)
        k.launch(
            grid=(H_Q, fx.Index(num_tokens), 1),
            block=(WAVE, 1, 1),
            stream=stream,
        )

    return launch


# ============================================================================
# Fused KV kernel: compute (RMSNorm+RoPE+quant for K, raw+quant for V) into
# LDS, barrier, then Prototype-A-style coalesced write from LDS to the
# shuffle-layout cache. Grid = (kv_head, page_block); block = 512 threads.
# ============================================================================
def build_kv_kernel():
    H_Q, H_K, H_V, D = NUM_Q_HEADS, NUM_KV_HEADS, NUM_KV_HEADS, HEAD_SIZE
    STAGE_ELEMS = BLOCK_SIZE * HEAD_SIZE  # 8192 bytes per K or V stage tile

    @fx.struct
    class SharedStorage:
        k_lds: fx.Array[fx.Int8, STAGE_ELEMS, 16]
        v_lds: fx.Array[fx.Int8, STAGE_ELEMS, 16]

    @flyc.kernel(
        name="kv_norm_mrope_quant_shuffle_write_fused_proto",
        known_block_size=[KV_THREADS, 1, 1],
    )
    def kernel(
        qkv: fx.Pointer,  # [T, H_Q+H_K+H_V, D] bf16, contig
        positions: fx.Pointer,  # [3, T] i64, contig (flat mid*T + tok)
        cos_sin: fx.Pointer,  # [max_pos, D] bf16
        k_norm_w: fx.Pointer,  # [D] bf16
        k_cache: fx.Pointer,  # shuffle-layout fp8 K cache (flat u8)
        v_cache: fx.Pointer,  # shuffle-layout fp8 V cache (flat u8)
        slot_mapping: fx.Pointer,  # [T] i64
        num_tokens: fx.Int32,
        k_scale: fx.Float32,
        v_scale: fx.Float32,
    ):
        # See flydsl-best-practices.md S1/S3: helpers nested here (not as
        # siblings in build_kv_kernel) so the AST rewriter processes their
        # `and`/`select` control flow; GTensor views are cheap Python
        # wrappers, fine to build up front here since there's no dynamic
        # if/elif/else branching in this kernel (unlike Prototype B's
        # Q/K/V dispatch) -- every thread in this grid does the same K-then-V
        # work.
        fm_fast = arith.FastMathFlags.fast

        def wave_reduce_add(x):
            v = x
            for sh_exp in range_constexpr(6):
                off = WAVE // (2 << sh_exp)
                peer = v.shuffle_xor(off, WAVE)
                v = v.addf(peer, fastmath=fm_fast)
            return v

        def mrope_cos_sin(tid, tok, num_tokens, positions_t, cos_sin_t):
            mid = tid % fx.Int32(3)
            is_mid1 = mid == fx.Int32(1)
            boundary = is_mid1.select(
                fx.Int32(MROPE_SECTION[1] * 3), fx.Int32(MROPE_SECTION[2] * 3)
            )
            in_range = tid < boundary
            use_mid = (mid != fx.Int32(0)) and in_range
            sect_idx = use_mid.select(mid, fx.Int32(0))

            pos_i64 = fx.Int64(positions_t[sect_idx * num_tokens + tok])
            pos = pos_i64.to(fx.Int32)
            cos_v = fx.BFloat16(cos_sin_t[pos, tid]).to(fx.Float32)
            sin_v = fx.BFloat16(cos_sin_t[pos, tid + fx.Int32(HALF)]).to(fx.Float32)
            return cos_v, sin_v

        def norm_rope(x0, x1, w0, w1, tid, tok, num_tokens, positions_t, cos_sin_t):
            x0f = fx.BFloat16(x0).to(fx.Float32)
            x1f = fx.BFloat16(x1).to(fx.Float32)
            sumsq_local = x0f * x0f + x1f * x1f
            sumsq = wave_reduce_add(sumsq_local)
            rstd = fmath.rsqrt(sumsq * (1.0 / D) + EPS, fastmath=fm_fast)

            w0f = fx.BFloat16(w0).to(fx.Float32)
            w1f = fx.BFloat16(w1).to(fx.Float32)
            xn0 = x0f * rstd * w0f
            xn1 = x1f * rstd * w1f

            cos_v, sin_v = mrope_cos_sin(tid, tok, num_tokens, positions_t, cos_sin_t)
            o0 = xn0 * cos_v - xn1 * sin_v
            o1 = xn1 * cos_v + xn0 * sin_v
            return o0, o1

        def quant_pair_fp8(v0, v1, scale):
            s0 = v0 / scale
            s1 = v1 / scale
            packed = fx.Int32(fx.rocdl.cvt_pk_fp8_f32(T.i32, s0, s1, fx.Int32(0), 0))
            byte0 = packed.to(fx.Int8)
            byte1 = (packed >> fx.Int32(8)).to(fx.Int8)
            return byte0, byte1

        def slot_i32(slot_tensor, tok):
            return fx.Int32(arith.trunci(T.i32, slot_tensor[tok]))

        head = fx.block_idx.x  # kv head 0..NUM_KV_HEADS-1
        blk = fx.block_idx.y  # page-block index (contiguous-slot assumption)
        t = fx.thread_idx.x  # 0..KV_THREADS-1

        qkv_t = GTensor(qkv, dtype=T.bf16, shape=(-1, H_Q + H_K + H_V, D))
        positions_t = GTensor(positions, dtype=T.i64, shape=(1,))
        cos_sin_t = GTensor(cos_sin, dtype=T.bf16, shape=(1, D))
        kw_t = GTensor(k_norm_w, dtype=T.bf16, shape=(D,))
        slot_t = GTensor(slot_mapping, dtype=T.i64, shape=(-1,))

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        k_lds = lds.k_lds
        v_lds = lds.v_lds

        tok0 = blk * BLOCK_SIZE

        # ---------------- Phase 1: compute -> LDS stage ----------------
        wid = t // WAVE
        lane = t % WAVE
        for it in range_constexpr(WAVES_PER_BLOCK):
            token_local = wid + WAVES_PER_BLOCK * it
            tok = tok0 + token_local

            # ---- K: RMSNorm + RoPE + per-tensor fp8 quant ----
            o0, o1 = norm_rope(
                qkv_t[tok, H_Q + head, lane],
                qkv_t[tok, H_Q + head, lane + HALF],
                kw_t[lane],
                kw_t[lane + HALF],
                lane,
                tok,
                num_tokens,
                positions_t,
                cos_sin_t,
            )
            kb0, kb1 = quant_pair_fp8(o0, o1, k_scale)
            k_row = token_local * HEAD_SIZE
            k_lds[k_row + lane] = kb0
            k_lds[k_row + lane + HALF] = kb1

            # ---- V: raw + per-tensor fp8 quant (no norm/rope) ----
            v0 = fx.BFloat16(qkv_t[tok, H_Q + H_K + head, lane]).to(fx.Float32)
            v1 = fx.BFloat16(qkv_t[tok, H_Q + H_K + head, lane + HALF]).to(fx.Float32)
            vb0, vb1 = quant_pair_fp8(v0, v1, v_scale)
            v_row = token_local * HEAD_SIZE
            v_lds[v_row + lane] = vb0
            v_lds[v_row + lane + HALF] = vb1

        gpu.barrier()

        # ---------------- Phase 2a: K coalesced write, LDS -> global ----
        # Identical addressing to Prototype A's k_coalesced; src is now the
        # LDS stage (read as i32x4) instead of a second global buffer.
        PER = 16
        block_id = slot_i32(slot_t, tok0) // BLOCK_SIZE
        head_base = block_id * _k_per_block() + head * _k_head_stride()

        f0 = t * PER
        chunk_k = f0 // (BLOCK_SIZE * X)
        block_off = (f0 % (BLOCK_SIZE * X)) // X
        d0 = chunk_k * X

        k_lds_i32 = fx.recast_iter(fx.Int32, k_lds.ptr)
        src_elem_i32 = (block_off * HEAD_SIZE + d0) // 4
        vec4_k = fx.ptr_load(
            k_lds_i32 + src_elem_i32, result_type=fx.Vector.make_type(4, fx.Int32)
        )
        cache_k_i32 = GTensor(k_cache, dtype=T.i32, shape=(-1,))
        cache_k_i32.vec_store(((head_base + f0) // 4,), vec4_k, PER // 4)

        # ---------------- Phase 2b: V coalesced write, LDS -> global -----
        # 512 threads = 4 tiles-of-X-tokens x HEAD_SIZE positions (matches
        # Prototype A's v_coalesced grain exactly, just 4 tiles/block here).
        tile = t // HEAD_SIZE  # 0..(BLOCK_SIZE/X - 1)
        d = t % HEAD_SIZE  # 0..HEAD_SIZE-1
        chunk_v = tile  # token-tile == shuffle chunk index (contiguous slots)
        v_head_base = block_id * _v_per_block() + head * _v_head_stride()
        v_dst_base = v_head_base + chunk_v * (HEAD_SIZE * X) + d * X

        vals = [v_lds[(tile * X + j) * HEAD_SIZE + d] for j in range_constexpr(X)]
        vec16_v = vector.from_elements(T.vec(X, T.i8), vals)
        vec4_v = vector.bitcast(T.vec(X // 4, T.i32), vec16_v)
        cache_v_i32 = GTensor(v_cache, dtype=T.i32, shape=(-1,))
        cache_v_i32.vec_store((v_dst_base // 4,), vec4_v, X // 4)

    @flyc.jit
    def launch(
        qkv: fx.Pointer,
        positions: fx.Pointer,
        cos_sin: fx.Pointer,
        k_norm_w: fx.Pointer,
        k_cache: fx.Pointer,
        v_cache: fx.Pointer,
        slot_mapping: fx.Pointer,
        num_tokens: fx.Int32,
        num_page_blocks: fx.Int32,
        k_scale: fx.Float32,
        v_scale: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ):
        k = kernel(
            qkv,
            positions,
            cos_sin,
            k_norm_w,
            k_cache,
            v_cache,
            slot_mapping,
            num_tokens,
            k_scale,
            v_scale,
        )
        k.launch(
            grid=(NUM_KV_HEADS, fx.Index(num_page_blocks), 1),
            block=(KV_THREADS, 1, 1),
            stream=stream,
        )

    return launch


# ============================================================================
# Torch reference: Prototype B's exact compute math + Prototype A's exact
# shuffle-layout scatter, composed end to end.
# ============================================================================
def torch_ref_full(qkv, qw, kw, cos_sin, positions, slot_mapping, num_tokens, k_scale, v_scale, kv_dtype):
    H_Q, H_K, H_V, D = NUM_Q_HEADS, NUM_KV_HEADS, NUM_KV_HEADS, HEAD_SIZE
    q_size, k_size, v_size = H_Q * D, H_K * D, H_V * D

    qkv2 = qkv.view(num_tokens, q_size + k_size + v_size)
    q, k, v = qkv2.split([q_size, k_size, v_size], dim=-1)

    q_by_head = rms_norm_forward(q.view(num_tokens, H_Q, D), qw, EPS)
    k_by_head = rms_norm_forward(k.view(num_tokens, H_K, D), kw, EPS)
    v_by_head = v.view(num_tokens, H_V, D)

    cos_sin_v = cos_sin.view(-1, D)
    pos3 = positions.view(3, num_tokens)
    cs = cos_sin_v[pos3]
    cos, sin = cs.chunk(2, dim=-1)
    cos = apply_interleaved_rope(cos, MROPE_SECTION)
    sin = apply_interleaved_rope(sin, MROPE_SECTION)

    q_r = apply_rotary_emb_torch(q_by_head, cos, sin, is_neox_style=True)
    k_r = apply_rotary_emb_torch(k_by_head, cos, sin, is_neox_style=True)

    k_q, _ = per_tensor_quant(
        k_r, scale=torch.tensor(k_scale, device=qkv.device), quant_dtype=kv_dtype
    )
    v_q, _ = per_tensor_quant(
        v_by_head, scale=torch.tensor(v_scale, device=qkv.device), quant_dtype=kv_dtype
    )

    # Prototype A's shuffle-layout scatter (flat-index, uint8 space).
    k_i = k_q.view(torch.uint8)
    v_i = v_q.view(torch.uint8)
    slots = slot_mapping.to(torch.int64)
    block_id = slots // BLOCK_SIZE
    block_off = slots % BLOCK_SIZE

    k_cache = torch.zeros(NUM_BLOCKS * _k_per_block(), dtype=torch.uint8, device=qkv.device)
    d = torch.arange(D, device=qkv.device)
    chunk = d // X
    in_x = d % X
    for h in range(NUM_KV_HEADS):
        base = block_id * _k_per_block() + h * _k_head_stride() + block_off * X
        dst = base[:, None] + chunk[None, :] * (BLOCK_SIZE * X) + in_x[None, :]
        k_cache[dst.reshape(-1)] = k_i[:, h, :].reshape(-1)

    v_cache = torch.zeros(NUM_BLOCKS * _v_per_block(), dtype=torch.uint8, device=qkv.device)
    chunk_v = block_off // X
    in_x_v = block_off % X
    for h in range(NUM_KV_HEADS):
        base = block_id * _v_per_block() + h * _v_head_stride() + chunk_v * (HEAD_SIZE * X) + in_x_v
        dst = base[:, None] + d[None, :] * X
        v_cache[dst.reshape(-1)] = v_i[:, h, :].reshape(-1)

    return q_r, k_cache, v_cache


def _ptr(t):
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def _time(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--max_positions", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", action="store_true", help="also time vs the production CUDA/HIP op")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    assert args.tokens % BLOCK_SIZE == 0, f"tokens must be a multiple of BLOCK_SIZE={BLOCK_SIZE}"
    torch.manual_seed(args.seed)
    dev = "cuda"
    kv_dtype = dtypes.fp8
    T_tok = args.tokens
    H_Q, H_K, H_V, D = NUM_Q_HEADS, NUM_KV_HEADS, NUM_KV_HEADS, HEAD_SIZE
    k_scale, v_scale = 1.5, 2.0

    qkv = torch.randn(T_tok, H_Q + H_K + H_V, D, dtype=torch.bfloat16, device=dev)
    qw = torch.randn(D, dtype=torch.bfloat16, device=dev).abs() + 0.5
    kw = torch.randn(D, dtype=torch.bfloat16, device=dev).abs() + 0.5
    cos_sin = torch.randn(args.max_positions, D, dtype=torch.bfloat16, device=dev) * 0.5
    positions = torch.randint(
        0, args.max_positions, (3, T_tok), dtype=torch.int64, device=dev
    ).contiguous()
    # Contiguous prefill slot_mapping (worst-case M-large prefill launch),
    # matching Prototype A's assumption: token t..t+BLOCK_SIZE-1 -> one page.
    slot_mapping = torch.arange(0, T_tok, device=dev, dtype=torch.int64)
    assert T_tok <= NUM_BLOCKS * BLOCK_SIZE

    q_out = torch.empty(T_tok, H_Q, D, dtype=torch.bfloat16, device=dev)
    k_cache = torch.zeros(NUM_BLOCKS * _k_per_block(), dtype=kv_dtype, device=dev)
    v_cache = torch.zeros(NUM_BLOCKS * _v_per_block(), dtype=kv_dtype, device=dev)

    q_launch = build_q_kernel()
    kv_launch = build_kv_kernel()
    stream = fx.Stream(torch.cuda.current_stream())
    num_page_blocks = T_tok // BLOCK_SIZE

    def run_full():
        _run_compiled(
            q_launch, _ptr(qkv), _ptr(positions), _ptr(cos_sin), _ptr(qw), _ptr(q_out),
            T_tok, stream,
        )
        _run_compiled(
            kv_launch, _ptr(qkv), _ptr(positions), _ptr(cos_sin), _ptr(kw),
            _ptr(k_cache), _ptr(v_cache), _ptr(slot_mapping),
            T_tok, num_page_blocks, float(k_scale), float(v_scale), stream,
        )

    run_full()
    torch.cuda.synchronize()

    q_ref, k_ref, v_ref = torch_ref_full(
        qkv, qw, kw, cos_sin, positions, slot_mapping, T_tok, k_scale, v_scale, kv_dtype
    )

    print(f"[correctness] T={T_tok} H_q={H_Q} H_k=H_v={H_K} D={D} page_blocks={num_page_blocks}")
    q_bad = checkAllclose(q_out.float(), q_ref.float(), rtol=1e-2, atol=0.05, msg="q_out ")
    k_bad = checkAllclose(
        k_cache.view(kv_dtype).float(), k_ref.view(kv_dtype).float(), rtol=1e-2, atol=0.05, msg="k_cache "
    )
    v_bad = checkAllclose(
        v_cache.view(kv_dtype).float(), v_ref.view(kv_dtype).float(), rtol=1e-2, atol=0.05, msg="v_cache "
    )
    print(
        f"  q_out mismatch ratio={q_bad:.3%}  k_cache={k_bad:.3%}  v_cache={v_bad:.3%}"
        "  (catastrophic would have raised above)"
    )
    print("[PASS] full fused kernel (compute + coalesced shuffle-layout KV write) matches torch reference.")

    if args.bench:
        fused_us = _time(run_full, args.iters, args.warmup)
        logical_mb = T_tok * (H_Q + H_K + H_V) * D / 1e6
        print(f"[perf] fused prototype (Q kernel + fused KV kernel): {fused_us:8.2f} us  "
              f"({logical_mb / fused_us * 1000:7.0f} GB/s eff, logical i/o only)")

        try:
            import aiter as aiter_mod

            rope_emb = cos_sin
            mrope_pos = positions
            q_out_prod = torch.empty(T_tok, H_Q * D, dtype=torch.bfloat16, device=dev)
            kv_k = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, D, dtype=kv_dtype, device=dev)
            kv_v = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, D, dtype=kv_dtype, device=dev)
            per_tensor_k_scale = torch.tensor(float(k_scale), dtype=torch.float32, device=dev)
            per_tensor_v_scale = torch.tensor(float(v_scale), dtype=torch.float32, device=dev)
            qkv_flat = qkv.view(T_tok, (H_Q + H_K + H_V) * D)

            def run_prod():
                aiter_mod.fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(
                    qkv_flat, qw, kw, rope_emb, mrope_pos, T_tok,
                    H_Q, H_K, H_V, D, True, MROPE_SECTION, True, EPS,
                    q_out_prod, kv_k, kv_v, slot_mapping,
                    per_tensor_k_scale, per_tensor_v_scale,
                    None, None, False, True, BLOCK_SIZE, X, D, False,
                )

            run_prod()
            torch.cuda.synchronize()
            prod_us = _time(run_prod, args.iters, args.warmup)
            print(f"[perf] production CUDA/HIP op                         : {prod_us:8.2f} us  "
                  f"({logical_mb / prod_us * 1000:7.0f} GB/s eff, logical i/o only)")
            print(f"[perf] fused prototype vs production                  : {prod_us / fused_us:5.2f}x")
        except Exception as e:  # pragma: no cover - benchmark is best-effort
            print(f"[perf] skipping production-op comparison ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
