# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Functional + perf test for the whole GPT-OSS attention block.

Runs OpenAI's GPT-OSS attention block end to end (mirrors ATOM's OAIAttention):
qkv_proj -> RoPE(YaRN) + paged KV write -> attention (sinks + alternating sliding
window + GQA) -> o_proj, checked against a torch reference via checkAllclose.

aiter kernels: get_rope (YaRN) + fused_qk_rope_reshape_and_cache, then attention via
--backend: asm = flash_attn_varlen_func(sink_ptr) + pa_decode_gluon (gfx942/gfx950);
triton = unified_attention for both phases (portable, incl. gfx1250). bf16 + fp8 KV.

Sinks = one learnable per-head logit acting as a zero-value KV column
(denom += exp(sink_h - max)). Validated on gfx950 / gfx942 / gfx1250.
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
from aiter.ops.triton.attention.unified_attention import unified_attention
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


# 20b/120b share the attention shape (only layers/experts differ); "small" is an alias.
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


def _attn_asm(
    cfg,
    w,
    q,
    k,
    v,
    positions,
    cu_seqlens,
    seq_len,
    batch,
    sliding_window,
    kv_cache_dtype,
    phase,
):
    """CDNA3/CDNA4 path (ATOM AiterBackend): SHUFFLE/NHD paged cache, prefill via
    flash_attn_varlen_func, decode via pa_decode_gluon. gfx942/gfx950 only."""
    hd, nq, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
    scale = hd**-0.5
    act_dtype = dtypes.bf16
    is_fp8 = kv_cache_dtype == dtypes.fp8
    total = q.shape[0]

    block_size = 16
    x = 16 // torch.empty(0, dtype=kv_cache_dtype).element_size()
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = batch * blocks_per_seq
    k_cache = torch.zeros(num_blocks, nkv, hd // x, block_size, x, dtype=kv_cache_dtype)
    v_cache = torch.zeros(num_blocks, nkv, hd, block_size, dtype=kv_cache_dtype)
    block_tables = torch.arange(num_blocks, dtype=torch.int32).view(
        batch, blocks_per_seq
    )
    seq_ids = torch.arange(batch, dtype=torch.int64).repeat_interleave(seq_len)
    in_seq = torch.arange(seq_len, dtype=torch.int64).repeat(batch)
    slot_mapping = block_tables[seq_ids, in_seq // block_size].to(
        torch.int64
    ) * block_size + (in_seq % block_size)

    kv_scale = None
    if is_fp8:
        kv_scale = (
            (torch.maximum(k.abs().max(), v.abs().max()) / torch.finfo(dtypes.fp8).max)
            .reshape(1)
            .float()
        )

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
        kv_scale,
        kv_scale,
        is_neox=w.rotary_emb.is_neox_style,
        flash_layout=False,
        apply_scale=is_fp8,
        output_zeros=False,
    )

    if phase == "prefill":
        window = (sliding_window, 0, 0) if sliding_window > 0 else (-1, -1, 0)
        o = aiter.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            softmax_scale=scale,
            causal=True,
            window_size=window,
            sink_ptr=w.sinks,
        )
        return o.view(total, nq * hd)

    q_dec = q.view(batch, seq_len, nq, hd)[:, -1].contiguous().view(batch, nq, hd)
    context_lens = torch.full((batch,), seq_len, dtype=torch.int32)
    if sliding_window > 0:
        max_part, part_size = 1, 128
    else:
        max_part, part_size = get_recommended_splits(batch, nkv), 256
    group = nq // nkv
    inter = (batch, nkv, max_part, group)
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
        1,
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
    return o.view(batch, nq * hd)


def _attn_triton(
    cfg,
    w,
    q,
    k,
    v,
    positions,
    cu_seqlens,
    seq_len,
    batch,
    sliding_window,
    phase,
    kv_cache_dtype,
):
    """Portable path (ATOM TritonMHABackend): flash-layout KV cache + aiter
    unified_attention for prefill AND decode; runs on gfx1250 (where
    flash_attn_varlen / pa_decode_gluon don't). fp8 KV (both phases read it): fused
    stores kv/scale as e4m3 (shared PRE-RoPE-max scale), unified_attention dequants
    via k/v_descale. Decode = single last-token query (max_seqlen_q=1)."""
    hd, nq, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
    scale = hd**-0.5
    total = q.shape[0]
    is_fp8 = kv_cache_dtype == dtypes.fp8
    # unified_attention window is (left, right) with left = sliding_window - 1.
    window = (sliding_window - 1, 0) if sliding_window > 0 else (-1, -1)

    # Flash-layout cache [num_blocks, block_size, num_kv_heads, head_dim].
    block_size = 16
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = batch * blocks_per_seq
    k_cache = torch.zeros(num_blocks, block_size, nkv, hd, dtype=kv_cache_dtype)
    v_cache = torch.zeros(num_blocks, block_size, nkv, hd, dtype=kv_cache_dtype)
    block_tables = torch.arange(num_blocks, dtype=torch.int32).view(
        batch, blocks_per_seq
    )
    seq_ids = torch.arange(batch, dtype=torch.int64).repeat_interleave(seq_len)
    in_seq = torch.arange(seq_len, dtype=torch.int64).repeat(batch)
    slot_mapping = block_tables[seq_ids, in_seq // block_size].to(
        torch.int64
    ) * block_size + (in_seq % block_size)

    # Graph-safe shared K/V scale from PRE-RoPE max (no .item()/Python max).
    kv_scale = None
    if is_fp8:
        kv_scale = (
            (torch.maximum(k.abs().max(), v.abs().max()) / torch.finfo(dtypes.fp8).max)
            .reshape(1)
            .float()
        )

    q, _, _, _ = fused_qk_rope_reshape_and_cache(
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
        flash_layout=True,
        apply_scale=is_fp8,
        output_zeros=False,
    )

    context_lens = torch.full((batch,), seq_len, dtype=torch.int32)
    # descale for unified_attention: bf16 -> ones (no-op); fp8 -> shared scale.
    if is_fp8:
        descale = kv_scale.view(1, 1).expand(batch, nkv).contiguous()
    else:
        descale = torch.ones((batch, nkv), dtype=torch.float32)
    ones = descale
    if phase == "prefill":
        o = torch.empty(total, nq, hd, dtype=dtypes.bf16)
        unified_attention(
            q,
            k_cache,
            v_cache,
            o,
            cu_seqlens_q=cu_seqlens,
            max_seqlen_q=seq_len,
            seqused_k=context_lens,
            max_seqlen_k=seq_len,
            softmax_scale=scale,
            causal=True,
            window_size=window,
            block_table=block_tables,
            softcap=0,
            q_descale=None,
            k_descale=ones,
            v_descale=ones,
            sinks=w.sinks,
        )
        return o.view(total, nq * hd)

    # decode: single last-token query per seq over the full-context cache.
    q_dec = q.view(batch, seq_len, nq, hd)[:, -1].contiguous()
    cu_q = torch.arange(batch + 1, dtype=torch.int32)
    o = torch.empty(batch, nq, hd, dtype=dtypes.bf16)
    unified_attention(
        q_dec,
        k_cache,
        v_cache,
        o,
        cu_seqlens_q=cu_q,
        max_seqlen_q=1,
        seqused_k=context_lens,
        max_seqlen_k=seq_len,
        softmax_scale=scale,
        causal=True,
        window_size=window,
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=ones,
        v_descale=ones,
        sinks=w.sinks,
    )
    return o.view(batch, nq * hd)


def attention_block(
    cfg: GptOssCfg,
    w: BlockWeights,
    hidden: torch.Tensor,  # [B*T, hidden]  (T = seq_len for prefill / ctx_len for decode)
    positions: torch.Tensor,  # [B*T] int32
    cu_seqlens: torch.Tensor,  # [B+1] int32
    seq_len: int,
    batch: int,
    sliding_window: int,
    kv_cache_dtype: torch.dtype,  # paged cache precision (bf16 or fp8)
    phase: str,  # "prefill" or "decode"
    backend: str,  # "asm" (gfx942/950) or "triton" (portable, incl. gfx1250)
) -> torch.Tensor:
    """Whole GPT-OSS attention block: qkv_proj -> RoPE + paged KV write (fused) ->
    attention -> o_proj. Two arch backends mirror ATOM: "asm" (AiterBackend,
    flash_attn_varlen + pa_decode_gluon, gfx942/950) and "triton" (TritonMHA /
    use_flash_layout, unified_attention, portable incl. gfx1250). Activations bf16;
    the asm decode cache may be fp8 (shared K/V scale from PRE-RoPE max)."""
    qkv = F.linear(hidden, w.qkv_w, w.qkv_b)
    q, k, v = _qkv_split(qkv, cfg)
    if backend == "triton":
        o = _attn_triton(
            cfg,
            w,
            q,
            k,
            v,
            positions,
            cu_seqlens,
            seq_len,
            batch,
            sliding_window,
            phase,
            kv_cache_dtype,
        )
    else:
        o = _attn_asm(
            cfg,
            w,
            q,
            k,
            v,
            positions,
            cu_seqlens,
            seq_len,
            batch,
            sliding_window,
            kv_cache_dtype,
            phase,
        )
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
def run_phase(
    cfg, phase, kv_cache_dtype, batch, seq_len, sliding_window, hipgraph, backend, args
):
    decode = phase == "decode"
    # KV-cache precision: triton reads the cache in BOTH phases (fp8 applies to
    # both); asm prefill reads K/V directly (bf16), so fp8 affects asm decode only.
    if backend == "triton":
        cache_dtype = kv_cache_dtype
    else:
        cache_dtype = kv_cache_dtype if decode else dtypes.bf16
    w = make_weights(cfg, dtypes.bf16)
    total = batch * seq_len
    hidden = (torch.randn(total, cfg.hidden_size, dtype=torch.float32) * 0.1).to(
        dtypes.bf16
    )
    cu = torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int32)
    positions = torch.arange(seq_len, dtype=torch.int32).repeat(batch)

    out, us = run_perftest(
        attention_block,
        cfg,
        w,
        hidden,
        positions,
        cu,
        seq_len,
        batch,
        sliding_window,
        cache_dtype,
        phase,
        backend,
        testGraph=hipgraph,
        num_iters=args.iters,
        num_warmup=args.warmup,
    )
    ref = ref_block(cfg, w, hidden, cu, positions, sliding_window, cache_dtype, decode)
    # fp8 e4m3 KV adds quantization noise; relax tolerance for that path.
    relax = cache_dtype == dtypes.fp8
    rtol = max(args.rtol, 6e-2) if relax else args.rtol
    atol = max(args.atol, 6e-2) if relax else args.atol
    err = checkAllclose(
        ref,
        out,
        rtol=rtol,
        atol=atol,
        msg=f"{phase[:7]:7} {backend:6} b{batch} t{seq_len} win{sliding_window} "
        f"{cache_dtype} graph={hipgraph}",
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
    p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "asm", "triton"],
        help="asm = flash_attn_varlen + pa_decode_gluon (gfx942/950); "
        "triton = unified_attention flash-layout (portable, incl. gfx1250); "
        "auto = triton on gfx1250, else asm",
    )
    p.add_argument(
        "--hipgraph",
        default="auto",
        choices=["auto", "on", "off"],
        help="HIP-graph capture/replay; auto = off for prefill, on for decode",
    )
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
    if args.backend == "auto":
        backend = "triton" if aiter.get_gfx() == "gfx1250" else "asm"
    else:
        backend = args.backend
    print(f"backend={backend} (gfx={aiter.get_gfx()})")
    kv_cache_dtype = dtypes.bf16 if args.kv_cache_dtype == "bf16" else dtypes.fp8
    if kv_cache_dtype == dtypes.fp8:
        scope = "both phases" if backend == "triton" else "decode only (prefill bf16)"
        print(f"NOTE: fp8 KV-cache, {backend} backend -> applies to {scope}.")

    win = cfg.sliding_window if args.sliding_window is None else args.sliding_window
    parities = {"even": [win], "odd": [-1], "both": [win, -1]}[args.layer_parity]
    phases = ["prefill", "decode"] if args.phase == "both" else [args.phase]

    rows = []
    fail = 0
    for phase in phases:
        # auto: prefill off, decode on; otherwise honor the explicit flag.
        hipgraph = {"on": True, "off": False}.get(args.hipgraph, phase == "decode")
        for sw in parities:
            sizes = args.seqlen if phase == "prefill" else args.ctx_len
            for batch, sz in itertools.product(args.batch, sizes):
                us, err = run_phase(
                    cfg, phase, kv_cache_dtype, batch, sz, sw, hipgraph, backend, args
                )
                ok = err == 0 or (isinstance(err, float) and err < 0.02)
                fail += 0 if ok else 1
                rows.append(
                    {
                        "phase": phase,
                        "backend": backend,
                        "window": sw,
                        "batch": batch,
                        "size": sz,
                        "graph": hipgraph,
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
