# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4.1-Flash mono decode layer: host side.

One ``DSV41MonoLayer`` a TP rank runs every mono layer of a decode step (M <= 48
rows): two persistent launches a layer (``kernels/dsv41_mono``) from the layer's
inputs at the attention seam to its outputs at the next one -- the MoE's
reduced output, the residual after the FFN seam and that seam's mixes -- with
both TP all-reduces inside the kernels (symmetric peer memory). Every argument
is a device pointer: the launches can be captured in a HIP graph. The kernels
move their mailbox epoch on themselves; every rank runs the same launches, so
the ranks' epochs agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .dsv41_mega_attn import MegaLayerWeights, _ensure_writable_flydsl_cache
from .kernels.dsv41_mega_attn.plan import HEAD_DIM, HIDDEN, KEYS, Dims
from .kernels.dsv41_mono.layer import (
    MAX_TOKENS,
    MonoBuild,
    build_mono_k1,
    build_mono_k2,
    peer_half_bytes,
    scratch_bytes,
)

__all__ = ["DSV41MonoLayer", "MonoLayerWeights", "MAX_TOKENS"]

_LOG2E = 1.4426950408889634
HC = 4


@dataclass
class MonoLayerWeights:
    """One layer's tensors on this rank, in vLLM's loaded layout."""

    attn: MegaLayerWeights
    hc_attn_fn: torch.Tensor  # [24, 4 * 5120] f32
    hc_attn_scale: torch.Tensor  # [3] f32
    hc_attn_base: torch.Tensor  # [24] f32
    attn_norm: torch.Tensor  # [5120] bf16
    hc_ffn_fn: torch.Tensor
    hc_ffn_scale: torch.Tensor
    hc_ffn_base: torch.Tensor
    ffn_norm: torch.Tensor
    gate_w: torch.Tensor  # [384, 5120] bf16
    bias: torch.Tensor  # [384] f32 (e_score_correction_bias)
    w13: torch.Tensor  # [384, 2 inter, 2560] fp4x2, aiter (16, 16) shuffle
    w13_s: torch.Tensor  # its scales, aiter shuffle_scale order
    w2: torch.Tensor  # [384, 5120, inter / 2] fp4x2
    w2_s: torch.Tensor
    sgu: torch.Tensor  # shared gate_up [2 inter, 5120] e4m3, row-major
    sgu_s: torch.Tensor  # [2 inter / 32, 160] E8M0
    sw2: torch.Tensor  # shared down [5120, inter] e4m3
    sw2_s: torch.Tensor


class DSV41MonoLayer:
    """The mono decode runner of one TP rank (``group``: its TP group, for the
    peer-memory handle exchange; a gloo / CPU group)."""

    def __init__(self, tp: int, rank: int, group, device: torch.device | str = "cuda"):
        _ensure_writable_flydsl_cache()
        from .kernels.dsv41_mono.atomfw.runtime.peer_memory import PeerBuffer

        self.tp, self.rank = tp, rank
        self.d = Dims(tp)
        dev = self.device = torch.device(device)
        self.scratch = torch.zeros(scratch_bytes(MAX_TOKENS, tp), dtype=torch.uint8, device=dev)
        # [epoch, -, -, -, a mark per CTA]
        self.epoch = torch.zeros(4 + 256, dtype=torch.int32, device=dev)
        self.q = torch.zeros(MAX_TOKENS, self.d.heads, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        self.kt = torch.zeros(MAX_TOKENS * KEYS, dtype=torch.int32, device=dev)
        self.klen = torch.zeros(2 * MAX_TOKENS, dtype=torch.int32, device=dev)
        self._dummy = torch.zeros(4, dtype=torch.int32, device=dev)
        self._zero_rec = torch.zeros(1024, dtype=torch.uint8, device=dev)
        # the attention seam's outputs, K1 -> K2
        self.res_mid = torch.zeros(MAX_TOKENS, HC, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.post_a = torch.zeros(MAX_TOKENS, HC, dtype=torch.float32, device=dev)
        self.comb_a = torch.zeros(MAX_TOKENS, HC, HC, dtype=torch.float32, device=dev)
        self.pre_a = torch.zeros(MAX_TOKENS, HC, dtype=torch.float32, device=dev)
        self.peer = PeerBuffer(2 * peer_half_bytes(tp), group, rank, tp, dev)
        self.peer.bytes.zero_()
        self._qk_scale = (
            torch.tensor([HEAD_DIM**-0.5 * _LOG2E], dtype=torch.float32).view(torch.int32).item()
        )
        self._kernels: dict = {}

    @staticmethod
    def supports(tokens: int) -> bool:
        return 1 <= tokens <= MAX_TOKENS

    def kernels(self, tokens: int, ratio: int):
        key = (tokens, ratio)
        if key not in self._kernels:
            b = MonoBuild(tokens, self.tp, ratio)
            self._kernels[key] = (build_mono_k1(b), build_mono_k2(b))
        return self._kernels[key]

    def forward(
        self,
        w: MonoLayerWeights,
        x: torch.Tensor,  # [M, 5120] bf16: the previous FFN output (reduced)
        residual: torch.Tensor,  # [M, 4, 5120] bf16
        post_mix: torch.Tensor,  # [M, 4, 1] f32
        res_mix: torch.Tensor,  # [M, 4, 4] f32
        pre_mix: torch.Tensor,  # [M, 4] f32
        positions: torch.Tensor,  # [M] int64
        slot_mapping: torch.Tensor,  # [M] int64
        swa_cache: torch.Tensor,  # [blocks, block, 584] uint8
        swa_indices: torch.Tensor,  # [>= M, 1, 128] int32
        swa_lens: torch.Tensor,  # [>= M] int32
        token_to_req: torch.Tensor,  # [>= M] int32
        topk_indices: torch.Tensor | None = None,
        comp_cache: torch.Tensor | None = None,
        comp_block_table: torch.Tensor | None = None,
        outs: tuple | None = None,
    ):
        """-> (out, residual, post_mix, res_mix, pre_mix): the layer's outputs
        (``outs``: buffers to write them into)."""
        M = x.shape[0]
        assert self.supports(M), M
        a = w.attn
        ratio = a.ratio
        k1, k2 = self.kernels(M, ratio)
        st = torch.cuda.current_stream()
        if outs is None:
            outs = (
                torch.empty(M, HIDDEN, dtype=torch.bfloat16, device=x.device),
                torch.empty_like(residual),
                torch.empty(M, HC, 1, dtype=torch.float32, device=x.device),
                torch.empty(M, HC, HC, dtype=torch.float32, device=x.device),
                torch.empty(M, HC, dtype=torch.float32, device=x.device),
            )
        out, res_out, post_out, comb_out, pre_out = outs
        for t_ in (x, residual, post_mix, res_mix, pre_mix, positions, slot_mapping):
            assert t_.is_contiguous()
        swa_block = swa_cache.shape[1]
        if ratio:
            comp, comp_stride, comp_block = comp_cache, comp_cache.stride(0), comp_cache.shape[1]
            bt, bt_stride, topk = comp_block_table, comp_block_table.stride(0), topk_indices
        else:
            comp, comp_stride, comp_block = self._dummy, 0, 1
            bt, bt_stride, topk = self._dummy, 0, self._dummy
        k1(
            residual.data_ptr(), x.data_ptr(), post_mix.data_ptr(), res_mix.data_ptr(),
            pre_mix.data_ptr(), w.hc_attn_fn.data_ptr(), w.hc_attn_scale.data_ptr(),
            w.hc_attn_base.data_ptr(), w.attn_norm.data_ptr(),
            self.res_mid.data_ptr(), self.post_a.data_ptr(), self.comb_a.data_ptr(),
            self.pre_a.data_ptr(),
            a.wqkv.data_ptr(), a.wqkv_scale.data_ptr(), a.q_norm.data_ptr(), a.kv_norm.data_ptr(),
            a.wq_b.data_ptr(), a.wq_b_scale.data_ptr(), a.cos_sin.data_ptr(),
            positions.data_ptr(), slot_mapping.data_ptr(), swa_cache.data_ptr(),
            swa_cache.stride(0), swa_block, self.q.data_ptr(), swa_indices.data_ptr(),
            swa_lens.data_ptr(), comp_block, bt.data_ptr(), bt_stride, topk.data_ptr(),
            token_to_req.data_ptr(), self.kt.data_ptr(), self.klen.data_ptr(),
            self.scratch.data_ptr(), self.epoch.data_ptr(), 0, stream=st,
        )  # fmt: skip
        k2(
            self.q.data_ptr(), swa_cache.data_ptr(), swa_cache.stride(0), swa_block,
            comp.data_ptr(), comp_stride, comp_block, self.kt.data_ptr(), self.klen.data_ptr(),
            positions.data_ptr(), a.attn_sink.data_ptr(), self._qk_scale, a.cos_sin.data_ptr(),
            a.wo_a.data_ptr(), a.wo_a_scale.data_ptr(), a.wo_b.data_ptr(), a.wo_b_scale.data_ptr(),
            self._zero_rec.data_ptr(),
            self.res_mid.data_ptr(), self.post_a.data_ptr(), self.comb_a.data_ptr(),
            self.pre_a.data_ptr(), w.hc_ffn_fn.data_ptr(), w.hc_ffn_scale.data_ptr(),
            w.hc_ffn_base.data_ptr(), w.ffn_norm.data_ptr(),
            res_out.data_ptr(), post_out.data_ptr(), comb_out.data_ptr(), pre_out.data_ptr(),
            w.gate_w.data_ptr(), w.bias.data_ptr(), w.w13.data_ptr(), w.w13_s.data_ptr(),
            w.w2.data_ptr(), w.w2_s.data_ptr(), w.sgu.data_ptr(), w.sgu_s.data_ptr(),
            w.sw2.data_ptr(), w.sw2_s.data_ptr(), out.data_ptr(),
            self.scratch.data_ptr(), self.peer.local, self.peer.addresses.data_ptr(),
            self.rank, self.epoch.data_ptr(), 0, stream=st,
        )  # fmt: skip
        return outs
