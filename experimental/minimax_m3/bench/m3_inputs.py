"""Synthetic MiniMax-M3 TP4 inputs for the mono microbenchmark (one rank): sparse layers
in the tensors ``vllm_runner.SparseLayer`` binds (FP8 ptpc qkv / o, BF16 gate, MXFP4
experts as raw bytes), vLLM's cache contract (fp8 [nb, 2, 128, 128] with page-16 K / V
views, bf16 index cache), and one decode step: ``rows`` requests of context ``ctx``,
one row each (the serving decode at concurrency ``rows``)."""
from __future__ import annotations

import functools
from types import SimpleNamespace

import torch

from vllm.models.minimax_m3.amd.mono.config import (
    HEAD_DIM, HIDDEN, INTER, LOCAL_Q_HEADS, N_ROUTED, ONE_INDEX_HEAD, ROTARY_DIM, TP, qkv_rows,
)
from vllm.models.minimax_m3.amd.mono.vllm_runner import M3MonoRunner, SparseLayer

BLOCK = 128
BT_WIDTH = 256  # vLLM's persistent block table row (max_model_len 32768 / 128)
MAX_POS = 32768


def _gen(seed, dev):
    return torch.Generator(device=dev).manual_seed(seed)


def _fp8(shape, g, dev, sd=1.0):
    return (torch.randn(shape, generator=g, device=dev) * sd).clamp(-448, 448).to(torch.float8_e4m3fn)


def _bytes(shape, g, dev, lo=0, hi=256):
    return torch.randint(lo, hi, shape, generator=g, device=dev, dtype=torch.uint8)


def _gain(n, g, dev):
    return (0.1 * torch.randn(n, generator=g, device=dev)).to(torch.bfloat16)  # Gemma: (1 + g)


class SynthAttn:
    """What the runner reads off vLLM's MiniMaxM3SparseAttention: the page-16 K / V views."""

    def __init__(self, nb, g, dev):
        self.kv_cache = _fp8((nb, 2, BLOCK, HEAD_DIM), g, dev)
        x = 16
        self.k16 = self.kv_cache.view(nb * 16, 1, HEAD_DIM // x, 16, x)
        self.v16 = self.kv_cache.view(nb * 16, 1, 16 // x, HEAD_DIM, x)[8:]
        self._aiter_sparse_pa_block_page_stride = 16

    def get_aiter_sparse_pa_kv_cache(self):
        return self.k16, self.v16


def make_layers(n_layers, nb, rank, dev, seed=0, max_pos=MAX_POS):
    """``n_layers`` sparse layers (ordinal % 4 == 0 selects, as layers 3, 7, .. 59)."""
    rows = qkv_rows(ONE_INDEX_HEAD.count)
    pos = torch.arange(max_pos, device=dev, dtype=torch.float32)[:, None]
    inv = 1.0 / (5e6 ** (torch.arange(0, ROTARY_DIM // 2, device=dev, dtype=torch.float32) / (ROTARY_DIM // 2)))
    ang = pos * inv[None, :]
    cos_sin = torch.cat([ang.cos(), ang.sin()], dim=1).to(torch.bfloat16).contiguous()
    out = []
    for i in range(n_layers):
        g = _gen(1000 * seed + 97 * i + rank, dev)
        gs = _gen(1000 * seed + 97 * i, dev)  # rank-shared tensors (norm gains, router)
        out.append(SparseLayer(
            layer_id=3 + i, attn=SynthAttn(nb, g, dev),
            index_cache=SimpleNamespace(kv_cache=(torch.randn(nb, BLOCK, HEAD_DIM, generator=g, device=dev)
                                                  ).to(torch.bfloat16)),
            index_topk=(i % 4 == 0),
            g_in=_gain(HIDDEN, gs, dev),
            w_qkv=_fp8((rows, HIDDEN), g, dev),
            s_qkv=torch.full((rows,), HIDDEN ** -0.5, device=dev),
            g_q=_gain(HEAD_DIM, gs, dev), g_k=_gain(HEAD_DIM, gs, dev),
            g_iq=_gain(HEAD_DIM, gs, dev), g_ik=_gain(HEAD_DIM, gs, dev),
            cos_sin=cos_sin,
            w_o=_fp8((HIDDEN, LOCAL_Q_HEADS * HEAD_DIM), g, dev),
            s_o=torch.full((HIDDEN,), 0.5 * (TP * LOCAL_Q_HEADS * HEAD_DIM) ** -0.5, device=dev),
            g_post=_gain(HIDDEN, gs, dev),
            gate=(torch.randn(N_ROUTED, HIDDEN, generator=gs, device=dev) * 0.02).to(torch.bfloat16),
            bias=(torch.randn(N_ROUTED, generator=gs, device=dev) * 0.05).float(),
            # MXFP4: e2m1 nibbles, e8m0 scales ~2^-7 (gate | up rows, aiter-shuffled order)
            w13=_bytes((N_ROUTED + 1, 2 * INTER, HIDDEN // 2), g, dev),
            s13=_bytes((N_ROUTED + 1, 2 * INTER, HIDDEN // 32), g, dev, 119, 122),
            w2=_bytes((N_ROUTED + 1, HIDDEN, INTER // 2), g, dev),
            s2=_bytes((N_ROUTED + 1, HIDDEN, INTER // 32), g, dev, 119, 122),
        ))
    return out


def make_step(rows, ctx, nb, dev, seed=0):
    """One decode step: row r is request r, context ``ctx`` (its token at position ctx-1)."""
    g = _gen(4242 + seed, dev)
    need = (ctx + BLOCK - 1) // BLOCK
    assert rows * need <= nb, "cache too small"
    perm = torch.randperm(nb, generator=g, device=dev).to(torch.int32)
    bt_buf = torch.zeros(64, BT_WIDTH, dtype=torch.int32, device=dev)  # persistent, as vLLM's
    import os
    for r in range(rows):
        q = r % 16 if os.environ.get("M3_DUP_ROWS") else r
        bt_buf[r, :need] = perm[q * need:(q + 1) * need]
    block_table = bt_buf[:rows]
    seq_lens = torch.full((rows,), ctx, dtype=torch.int32, device=dev)
    positions = torch.full((rows,), ctx - 1, dtype=torch.int64, device=dev)
    p = ctx - 1
    slot = (block_table[:, p // BLOCK].long() * BLOCK + p % BLOCK).contiguous()
    md = SimpleNamespace(
        decode=SimpleNamespace(block_table=block_table, seq_lens=seq_lens, decode_query_len=1),
        slot_mapping=slot, num_prefills=0,
    )
    return md, positions


def make_spec_step(reqs, ql, ctx, nb, dev, seed=0):
    """One speculative-verify decode step after a ``ctx``-token prefill: ``reqs`` requests of
    ``ql`` rows each (1 + draft tokens), request r's rows at positions ctx .. ctx + ql - 1 (their
    K / V / index keys are inserted by the step itself); vLLM's metadata: seq_lens = ctx + ql per
    request, decode_query_len = ql, one slot per row. The block table is as wide as the context
    needs (vLLM's persistent table is max_model_len / 128 wide)."""
    g = _gen(4242 + seed, dev)
    total = ctx + ql
    need = (total + BLOCK - 1) // BLOCK
    assert reqs * need <= nb, "cache too small"
    width = max(BT_WIDTH, need)
    perm = torch.randperm(nb, generator=g, device=dev).to(torch.int32)
    bt_buf = torch.zeros(64, width, dtype=torch.int32, device=dev)
    for r in range(reqs):
        bt_buf[r, :need] = perm[r * need:(r + 1) * need]
    block_table = bt_buf[:reqs]
    seq_lens = torch.full((reqs,), total, dtype=torch.int32, device=dev)
    pos = torch.arange(ctx, total, device=dev, dtype=torch.int64)
    positions = pos.repeat(reqs).contiguous()
    rows_bt = block_table.repeat_interleave(ql, dim=0)
    slot = (rows_bt.gather(1, (positions // BLOCK)[:, None]).flatten().long() * BLOCK + positions % BLOCK).contiguous()
    md = SimpleNamespace(
        decode=SimpleNamespace(block_table=block_table, seq_lens=seq_lens, decode_query_len=ql),
        slot_mapping=slot, num_prefills=0,
    )
    return md, positions


class StandaloneRunner(M3MonoRunner):
    """``M3MonoRunner`` on synthetic layers and a fixed step's metadata (no vLLM model)."""

    def __init__(self, layers, rank, cpu_group, dev, debug=False, eps=1e-6, route_scale=2.0, shared_weight=1.0,
                 swiglu_limit=7.0, timeline=False):
        from vllm.models.minimax_m3.amd.mono.fw.runtime.widths import WidthBuilds
        from vllm.models.minimax_m3.amd.mono.kernels.post_attn import build_post_attn_kernel

        self.npes, self.rank, self.dev, self.sparse, self.debug = TP, rank, dev, layers, debug
        self.md = None
        self._build_k4 = functools.partial(
            build_post_attn_kernel, TP, HEAD_DIM ** -0.5, eps, route_scale, shared_weight, swiglu_limit, 0, 1,
            fuse_k1=True, heads=ONE_INDEX_HEAD, debug=debug, timeline=timeline,
        )
        if timeline:
            from vllm.models.minimax_m3.amd.mono.kernels.stages.shared import TL_POINTS

            self.tl_buf = torch.zeros(len(layers), 256, TL_POINTS, dtype=torch.int64, device=dev)
        self._allocate(cpu_group)
        self.k4 = WidthBuilds(self._build_width, self._width_kernels, cpu_group, "M3 mono ubench (FlyDSL)")
        if timeline:  # a timeline build spills a little: diagnostics only, run under a timeout
            self.k4._scratch_free = lambda *_: True

    def sparse_metadata(self):
        return self.md
