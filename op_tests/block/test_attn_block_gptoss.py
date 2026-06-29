# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Functional + perf test for the whole GPT-OSS attention block.

This runs the complete attention block of OpenAI's GPT-OSS (mirroring ATOM's
``GptOssForCausalLM`` / ``OAIAttention``), end to end:

    hidden_states
        -> qkv_proj (Linear, bias)        # generic bf16 GEMM (torch F.linear)
        -> split q, k, v
        -> RoPE (YaRN) + write paged KV    # aiter fused_qk_rope_reshape_and_cache
        -> attention                       # prefill: flash_attn_varlen_func
                                           # decode : pa_decode_gluon
           with per-head attention sinks + alternating sliding window + GQA
        -> o_proj (Linear, bias)           # generic bf16 GEMM (torch F.linear)

GPT-OSS attention specifics that drive the kernel selection:
  * head_dim = 64  -> ATOM forces the triton/gluon path (head_dim != 128).
  * GQA          : num_attention_heads / num_key_value_heads (e.g. 64 / 8).
  * attention sinks : one learnable logit per query head, acting as a zero-value
    virtual KV column (``denom += exp(sink_h - max)``).
  * sliding window : applied on even-indexed layers (window=128), odd layers are
    full causal. Selected here via --layer-parity / --sliding-window.
  * RoPE         : YaRN scaled, neox style, theta=150000.

aiter kernels exercised (same as ATOM's PagedAttentionImpl for GPT-OSS):
  * aiter.rotary_embedding.get_rope               (YaRN cos/sin cache)
  * aiter.ops.triton.fusions.fused_kv_cache
        .fused_qk_rope_reshape_and_cache          (RoPE + paged cache write, NHD)
  * aiter.flash_attn_varlen_func (sink_ptr=...)   (prefill)
  * aiter.ops.triton.gluon.pa_decode_gluon        (decode)

Style mirrors op_tests/test_pa_decode_bf16_asm.py / test_pa_ps.py: a torch host
reference compared against the kernel block via aiter.test_common.checkAllclose
(no pytest), driven by argparse over a config grid, with a perf summary.

Validated environment: MI355X / gfx950, torch 2.10, aiter main.
"""

import argparse
import itertools
import sys
from dataclasses import dataclass, field

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.rotary_embedding import get_rope
from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_reshape_and_cache
from aiter.ops.triton.gluon.pa_decode_gluon import (
    get_recommended_splits,
    pa_decode_gluon,
)
from aiter.test_common import checkAllclose, run_perftest

torch.set_default_device("cuda")
torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# Model configuration                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class GptOssCfg:
    name: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int = 64
    rope_theta: float = 150000.0
    max_position_embeddings: int = 131072
    sliding_window: int = 128
    # YaRN rope scaling (gpt-oss defaults).
    yarn_factor: float = 32.0
    yarn_orig_max_pos: int = 4096
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0

    def rope_scaling(self) -> dict:
        return {
            "rope_type": "yarn",
            "factor": self.yarn_factor,
            "original_max_position_embeddings": self.yarn_orig_max_pos,
            "beta_fast": self.yarn_beta_fast,
            "beta_slow": self.yarn_beta_slow,
        }


# gpt-oss-20b and gpt-oss-120b share the same attention shape (only the number of
# layers / experts differ, which the attention block does not see). "small" keeps
# the real attention dims but is cheap to run.
CONFIGS = {
    "small": GptOssCfg(
        "small", hidden_size=2880, num_attention_heads=64, num_key_value_heads=8
    ),
    "gpt-oss-20b": GptOssCfg(
        "gpt-oss-20b", hidden_size=2880, num_attention_heads=64, num_key_value_heads=8
    ),
    "gpt-oss-120b": GptOssCfg(
        "gpt-oss-120b", hidden_size=2880, num_attention_heads=64, num_key_value_heads=8
    ),
}


# --------------------------------------------------------------------------- #
# The attention block under test                                              #
# --------------------------------------------------------------------------- #
@dataclass
class BlockWeights:
    qkv_w: torch.Tensor  # [(q+2kv)*hd, hidden]
    qkv_b: torch.Tensor  # [(q+2kv)*hd]
    o_w: torch.Tensor  # [hidden, q*hd]
    o_b: torch.Tensor  # [hidden]
    sinks: torch.Tensor  # [num_q_heads] fp32
    rotary_emb: torch.nn.Module = field(default=None)


def make_weights(cfg: GptOssCfg, dtype: torch.dtype) -> BlockWeights:
    hd = cfg.head_dim
    nq = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    qkv_out = (nq + 2 * nkv) * hd

    def randn(*shape):
        return (torch.randn(*shape, dtype=torch.float32) * 0.05).to(dtype)

    rotary_emb = get_rope(
        head_size=hd,
        rotary_dim=hd,
        max_position=cfg.max_position_embeddings,
        base=cfg.rope_theta,
        is_neox_style=True,
        rope_scaling=cfg.rope_scaling(),
        dtype=dtype,
    )
    return BlockWeights(
        qkv_w=randn(qkv_out, cfg.hidden_size),
        qkv_b=randn(qkv_out),
        o_w=randn(cfg.hidden_size, nq * hd),
        o_b=randn(cfg.hidden_size),
        # Per-head sink logits in the scaled (q·k·scale) domain.
        sinks=(torch.randn(nq, dtype=torch.float32)),
        rotary_emb=rotary_emb,
    )


def _qkv_split(qkv, cfg):
    hd, nq, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
    q, k, v = torch.split(qkv, [nq * hd, nkv * hd, nkv * hd], dim=-1)
    return (
        q.view(-1, nq, hd).contiguous(),
        k.view(-1, nkv, hd).contiguous(),
        v.view(-1, nkv, hd).contiguous(),
    )


def kernel_block_prefill(
    cfg: GptOssCfg,
    w: BlockWeights,
    hidden: torch.Tensor,  # [B*S, hidden]
    cu_seqlens: torch.Tensor,  # [B+1] int32
    positions: torch.Tensor,  # [B*S] int32
    max_seqlen: int,
    sliding_window: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Full GPT-OSS attention block, prefill path (varlen)."""
    hd, nq, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
    scale = hd**-0.5

    qkv = F.linear(hidden, w.qkv_w, w.qkv_b)
    q, k, v = _qkv_split(qkv, cfg)

    # RoPE + (throwaway) cache write. Prefill attention reads k/v directly, but
    # GPT-OSS still applies RoPE via the fused kernel and writes the cache.
    total = hidden.shape[0]
    block_size = 16
    x = 16 // torch.empty(0, dtype=dtype).element_size()
    num_blocks = (total + block_size - 1) // block_size + 1
    k_cache = torch.zeros(num_blocks, nkv, hd // x, block_size, x, dtype=dtype)
    v_cache = torch.zeros(num_blocks, nkv, hd, block_size, dtype=dtype)
    slot_mapping = torch.arange(total, dtype=torch.int64)
    q, k, _, _ = fused_qk_rope_reshape_and_cache(
        q,
        k,
        v,
        k_cache,
        v_cache,
        slot_mapping,
        positions,
        w.rotary_emb.cos_cache,
        w.rotary_emb.sin_cache,
        None,
        None,
        is_neox=w.rotary_emb.is_neox_style,
        flash_layout=False,
        apply_scale=False,
        output_zeros=False,
    )

    window = (sliding_window, 0, 0) if sliding_window > 0 else (-1, -1, 0)
    o = aiter.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=scale,
        causal=True,
        window_size=window,
        sink_ptr=w.sinks,
    )
    o = o.view(total, nq * hd)
    return F.linear(o, w.o_w, w.o_b)


def kernel_block_decode(
    cfg: GptOssCfg,
    w: BlockWeights,
    hidden_ctx: torch.Tensor,  # [B*L, hidden]  (full context per seq)
    ctx_len: int,
    batch: int,
    sliding_window: int,
    kv_cache_dtype: torch.dtype,  # bf16 or fp8 (paged cache precision)
) -> torch.Tensor:
    """Full GPT-OSS attention block, decode path (paged, query_len=1).

    Activations stay bf16; only the paged KV cache may be fp8. With fp8 a single
    per-tensor scale is used for K and V (ATOM's convention): the fused kernel
    stores ``kv / scale`` as e4m3 and pa_decode_gluon recovers ``stored * scale``.
    """
    hd, nq, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
    scale = hd**-0.5
    act_dtype = dtypes.bf16
    is_fp8 = kv_cache_dtype == dtypes.fp8

    qkv = F.linear(hidden_ctx, w.qkv_w, w.qkv_b)
    q, k, v = _qkv_split(qkv, cfg)

    block_size = 16
    x = 16 // torch.empty(0, dtype=kv_cache_dtype).element_size()
    blocks_per_seq = (ctx_len + block_size - 1) // block_size
    num_blocks = batch * blocks_per_seq
    k_cache = torch.zeros(num_blocks, nkv, hd // x, block_size, x, dtype=kv_cache_dtype)
    v_cache = torch.zeros(num_blocks, nkv, hd, block_size, dtype=kv_cache_dtype)

    block_tables = torch.arange(num_blocks, dtype=torch.int32).view(
        batch, blocks_per_seq
    )
    # positions and slot_mapping over the whole context of every sequence.
    positions = torch.arange(ctx_len, dtype=torch.int32).repeat(batch)
    seq_ids = torch.arange(batch, dtype=torch.int64).repeat_interleave(ctx_len)
    in_seq = torch.arange(ctx_len, dtype=torch.int64).repeat(batch)
    slot_mapping = block_tables[seq_ids, in_seq // block_size].to(
        torch.int64
    ) * block_size + (in_seq % block_size)

    if is_fp8:
        kv_scale = (
            max(k.abs().max(), v.abs().max()).float() / torch.finfo(dtypes.fp8).max
        )
        kv_scale = torch.tensor([kv_scale.item()], dtype=torch.float32)
    else:
        kv_scale = None

    q_roped, _, _, _ = fused_qk_rope_reshape_and_cache(
        q,
        k,
        v,
        k_cache,
        v_cache,
        slot_mapping,
        positions,
        w.rotary_emb.cos_cache,
        w.rotary_emb.sin_cache,
        kv_scale,
        kv_scale,
        is_neox=w.rotary_emb.is_neox_style,
        flash_layout=False,
        apply_scale=is_fp8,
        output_zeros=False,
    )

    # decode query = last token of each sequence.
    q_dec = q_roped.view(batch, ctx_len, nq, hd)[torch.arange(batch), -1].contiguous()
    q_dec = q_dec.view(batch, nq, hd)

    context_lens = torch.full((batch,), ctx_len, dtype=torch.int32)
    num_seqs = batch
    if sliding_window > 0:
        max_part = 1
        part_size = 128
    else:
        max_part = get_recommended_splits(num_seqs, nkv)
        part_size = 256
    group = nq // nkv
    inter = (num_seqs, nkv, max_part, group)
    exp_sums = torch.empty(inter, dtype=torch.float32)
    max_logits = torch.empty(inter, dtype=torch.float32)
    tmp_out = torch.empty(*inter, hd, dtype=act_dtype)

    o = torch.empty(batch, nq, hd, dtype=act_dtype)
    pa_decode_gluon(
        o,
        q_dec,
        k_cache,
        v_cache,
        context_lens,
        block_tables,
        scale,
        1,  # query_length
        max_part,
        part_size,
        compute_type=kv_cache_dtype if is_fp8 else act_dtype,
        query_scale=None,
        key_scale=kv_scale,
        value_scale=kv_scale,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=tmp_out,
        alibi_slopes=None,
        sinks=w.sinks,
        sliding_window=sliding_window,
        ps=True,
    )
    o = o.view(batch, nq * hd)
    return F.linear(o, w.o_w, w.o_b)


# --------------------------------------------------------------------------- #
# Torch reference                                                             #
# --------------------------------------------------------------------------- #
def _ref_one_seq(q, k, v, sinks, scale, sliding_window, q_offset):
    """Reference attention for one sequence. q:[sq,nq,hd], k/v:[sk,nkv,hd] (RoPE'd).
    Bottom-right causal: query row i is at abs pos q_offset+i, attends key j iff
    j <= q_offset+i (and j > q_offset+i-window when a window is set)."""
    sq, nq, hd = q.shape
    sk, nkv, _ = k.shape
    group = nq // nkv
    qf, kf, vf = q.float(), k.float(), v.float()
    kf = kf.repeat_interleave(group, dim=1)  # GQA expand -> [sk, nq, hd]
    vf = vf.repeat_interleave(group, dim=1)

    scores = torch.einsum("qhd,khd->hqk", qf, kf) * scale  # [nq, sq, sk]
    qi = torch.arange(sq, device=q.device).view(sq, 1) + q_offset
    kj = torch.arange(sk, device=q.device).view(1, sk)
    mask = kj <= qi
    if sliding_window > 0:
        mask = mask & (kj > qi - sliding_window)
    scores = scores.masked_fill(~mask.view(1, sq, sk), float("-inf"))

    m = scores.max(dim=-1, keepdim=True).values  # [nq, sq, 1]
    sink = sinks.view(nq, 1, 1).float()
    m = torch.maximum(m, sink)
    p = torch.exp(scores - m)
    denom = p.sum(dim=-1, keepdim=True) + torch.exp(sink - m)
    p = p / denom
    out = torch.einsum("hqk,khd->qhd", p, vf)  # [sq, nq, hd]
    return out


def _fp8_round_trip(t, s):
    """e4m3 round-trip: stored = clamp(t/s) cast to fp8, recovered = *s. Clamp
    matches the kernel's saturating store (torch .to(e4m3fn) NaNs on overflow)."""
    fp8_max = torch.finfo(dtypes.fp8).max
    return ((t.float() / s).clamp(-fp8_max, fp8_max).to(dtypes.fp8).float() * s).to(
        t.dtype
    )


def ref_block(
    cfg,
    w,
    hidden,
    cu_seqlens,
    positions,
    sliding_window,
    kv_cache_dtype=None,
    decode=False,
):
    """Torch reference for the whole GPT-OSS attention block (prefill + decode).

    Per sequence (delimited by cu_seqlens): bottom-right causal + optional
    sliding window + sinks over GQA-expanded K/V. decode=True keeps only the last
    query per sequence (paged decode); kv_cache_dtype=fp8 round-trips K/V to
    mirror the fp8 cache (shared scale from PRE-RoPE max of K and V).
    """
    hd, nq = cfg.head_dim, cfg.num_attention_heads
    scale = hd**-0.5
    q, k, v = _qkv_split(F.linear(hidden, w.qkv_w, w.qkv_b), cfg)
    kv_scale = None
    if kv_cache_dtype == dtypes.fp8:
        kv_scale = (
            max(k.abs().max(), v.abs().max()).float() / torch.finfo(dtypes.fp8).max
        )
    q, k = w.rotary_emb.forward_native(positions.long(), q, k)
    if kv_cache_dtype == dtypes.fp8:
        # Stored as e4m3: post-RoPE K and raw V, both with the pre-RoPE scale.
        k, v = _fp8_round_trip(k, kv_scale), _fp8_round_trip(v, kv_scale)
    outs = []
    cu = cu_seqlens.tolist()
    for b in range(len(cu) - 1):
        s, e = cu[b], cu[b + 1]
        qs = q[e - 1 : e] if decode else q[s:e]
        q_off = (e - s - 1) if decode else 0
        outs.append(
            _ref_one_seq(qs, k[s:e], v[s:e], w.sinks, scale, sliding_window, q_off)
        )
    o = torch.cat(outs, dim=0).to(hidden.dtype).view(-1, nq * hd)
    return F.linear(o, w.o_w, w.o_b)


# --------------------------------------------------------------------------- #
# Drivers                                                                     #
# --------------------------------------------------------------------------- #
def run_prefill(cfg, batch, seqlen, sliding_window, args):
    # Prefill reads K/V directly in bf16; the KV-cache precision does not affect
    # this path, so prefill always runs bf16.
    dtype = dtypes.bf16
    w = make_weights(cfg, dtype)
    total = batch * seqlen
    hidden = (torch.randn(total, cfg.hidden_size, dtype=torch.float32) * 0.1).to(dtype)
    cu = torch.arange(0, (batch + 1) * seqlen, seqlen, dtype=torch.int32)
    positions = torch.arange(seqlen, dtype=torch.int32).repeat(batch)

    out, us = run_perftest(
        kernel_block_prefill,
        cfg,
        w,
        hidden,
        cu,
        positions,
        seqlen,
        sliding_window,
        dtype,
        num_iters=args.iters,
        num_warmup=args.warmup,
    )
    ref = ref_block(cfg, w, hidden, cu, positions, sliding_window)
    err = checkAllclose(
        ref,
        out,
        rtol=args.rtol,
        atol=args.atol,
        msg=f"prefill b{batch} s{seqlen} win{sliding_window} {dtype}",
    )
    return us, err


def run_decode(cfg, kv_cache_dtype, batch, ctx_len, sliding_window, args):
    w = make_weights(cfg, dtypes.bf16)
    total = batch * ctx_len
    hidden = (torch.randn(total, cfg.hidden_size, dtype=torch.float32) * 0.1).to(
        dtypes.bf16
    )

    out, us = run_perftest(
        kernel_block_decode,
        cfg,
        w,
        hidden,
        ctx_len,
        batch,
        sliding_window,
        kv_cache_dtype,
        num_iters=args.iters,
        num_warmup=args.warmup,
    )
    cu = torch.arange(0, (batch + 1) * ctx_len, ctx_len, dtype=torch.int32)
    positions = torch.arange(ctx_len, dtype=torch.int32).repeat(batch)
    ref = ref_block(
        cfg, w, hidden, cu, positions, sliding_window, kv_cache_dtype, decode=True
    )
    # fp8 e4m3 KV adds quantization noise; relax tolerance for that path.
    rtol = args.rtol if kv_cache_dtype == dtypes.bf16 else max(args.rtol, 6e-2)
    atol = args.atol if kv_cache_dtype == dtypes.bf16 else max(args.atol, 6e-2)
    err = checkAllclose(
        ref,
        out,
        rtol=rtol,
        atol=atol,
        msg=f"decode  b{batch} ctx{ctx_len} win{sliding_window} {kv_cache_dtype}",
    )
    return us, err


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="small", choices=list(CONFIGS.keys()))
    p.add_argument("--phase", default="both", choices=["prefill", "decode", "both"])
    p.add_argument(
        "--layer-parity",
        default="both",
        choices=["even", "odd", "both"],
        help="even=sliding window, odd=full causal",
    )
    p.add_argument(
        "--sliding-window",
        type=int,
        default=None,
        help="override the even-layer window size",
    )
    p.add_argument("--kv-cache-dtype", default="bf16", choices=["bf16", "fp8"])
    p.add_argument("--batch", type=int, nargs="+", default=[1, 2])
    p.add_argument(
        "--seqlen",
        type=int,
        nargs="+",
        default=[64, 256],
        help="prefill sequence length(s)",
    )
    p.add_argument(
        "--ctx-len",
        type=int,
        nargs="+",
        default=[64, 300],
        help="decode context length(s)",
    )
    p.add_argument("--rtol", type=float, default=2e-2)
    p.add_argument("--atol", type=float, default=2e-2)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args()

    cfg = CONFIGS[args.model]
    kv_cache_dtype = dtypes.bf16 if args.kv_cache_dtype == "bf16" else dtypes.fp8
    if kv_cache_dtype == dtypes.fp8:
        print(
            "NOTE: fp8 KV-cache affects the decode path only "
            "(activations stay bf16; prefill stays bf16)."
        )

    win = cfg.sliding_window if args.sliding_window is None else args.sliding_window
    parities = {"even": [win], "odd": [-1], "both": [win, -1]}[args.layer_parity]
    phases = ["prefill", "decode"] if args.phase == "both" else [args.phase]

    rows = []
    fail = 0
    for phase in phases:
        for sw in parities:
            sizes = args.seqlen if phase == "prefill" else args.ctx_len
            for batch, sz in itertools.product(args.batch, sizes):
                if phase == "prefill":
                    us, err = run_prefill(cfg, batch, sz, sw, args)
                else:
                    us, err = run_decode(cfg, kv_cache_dtype, batch, sz, sw, args)
                ok = err == 0 or (isinstance(err, float) and err < 0.02)
                fail += 0 if ok else 1
                rows.append(
                    {
                        "phase": phase,
                        "window": sw,
                        "batch": batch,
                        "size": sz,
                        "us": round(us, 2),
                        "err_ratio": err,
                        "pass": ok,
                    }
                )

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    if fail:
        print(f"\n{fail} configuration(s) FAILED")
        sys.exit(1)
    print("\nAll configurations passed.")


if __name__ == "__main__":
    main()
