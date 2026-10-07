# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4.1-Flash attention mega kernel: host side.

One ``DSV41MegaAttention`` a TP rank serves every eligible layer (no own
compressor / indexer) of a decode step with M <= 48 rows: two resident launches
a layer (``kernels/dsv41_mega_attn``) from the attention seam's normed rows to
the rank's wo_b partial, the step's KV rows written into the layer's
sliding-window cache. Every argument is a device pointer, so the launches can
be captured in a HIP graph. The kernels move their own mailbox epoch on (the
last CTA of a layer's second launch), so a launch pair's hand-offs never meet
an earlier pair's flags, however the launches are captured or replayed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from .kernels.dsv41_mega_attn.back import BackBuild, build_back
from .kernels.dsv41_mega_attn.front import FrontBuild, build_front
from .kernels.dsv41_mega_attn.plan import (
    HEAD_DIM,
    HIDDEN,
    KEYS,
    MAX_TOKENS,
    Dims,
    scratch_bytes,
)

__all__ = ["DSV41MegaAttention", "MegaLayerWeights", "MAX_TOKENS"]

_LOG2E = 1.4426950408889634


def _ensure_writable_flydsl_cache() -> None:
    """aiter points FlyDSL at its bundled, read-only cache; new kernels need a
    writable one."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


@dataclass
class MegaLayerWeights:
    """One layer's tensors on this rank, in vLLM's loaded layout."""

    layer_id: int
    wqkv: torch.Tensor  # [1792, 5120] e4m3: [wq_a; wkv]
    wqkv_scale: torch.Tensor  # [56, 160] uint8 E8M0 (32 x 32 blocks)
    q_norm: torch.Tensor  # [1280] bf16
    kv_norm: torch.Tensor  # [512] bf16
    wq_b: torch.Tensor  # [H * 512, 1280] e4m3
    wq_b_scale: torch.Tensor  # [H * 16, 40] uint8
    wo_a: torch.Tensor  # [G * 1024, 4096] e4m3
    wo_a_scale: torch.Tensor  # [G * 32, 128] uint8
    wo_b: torch.Tensor  # [5120, G * 1024] e4m3
    wo_b_scale: torch.Tensor  # [160, G * 32] uint8
    attn_sink: torch.Tensor  # [>= H] f32
    cos_sin: torch.Tensor  # [positions, 64] f32 (the layer's rope table)
    ratio: int  # 0 / 1 / 2: the layer's compress ratio

    def check(self, d: Dims) -> None:
        h, g = d.heads, d.groups
        want = {
            "wqkv": ((1792, HIDDEN), torch.float8_e4m3fn),
            "wqkv_scale": ((56, HIDDEN // 32), torch.uint8),
            "wq_b": ((h * HEAD_DIM, 1280), torch.float8_e4m3fn),
            "wq_b_scale": ((h * HEAD_DIM // 32, 40), torch.uint8),
            "wo_a": ((g * 1024, d.group_k), torch.float8_e4m3fn),
            "wo_a_scale": ((g * 32, d.group_k // 32), torch.uint8),
            "wo_b": ((HIDDEN, g * 1024), torch.float8_e4m3fn),
            "wo_b_scale": ((HIDDEN // 32, g * 32), torch.uint8),
        }
        for name, (shape, dtype) in want.items():
            t = getattr(self, name)
            assert tuple(t.shape) == shape and t.dtype == dtype and t.is_contiguous(), (
                f"layer {self.layer_id} {name}: {tuple(t.shape)} {t.dtype}, want {shape} {dtype}"
            )
        assert self.cos_sin.dtype == torch.float32 and self.cos_sin.shape[-1] == 64


class DSV41MegaAttention:
    """The mega-kernel runner of one TP rank."""

    def __init__(self, tp: int, device: torch.device | str = "cuda", timeline: bool = False):
        _ensure_writable_flydsl_cache()
        self.d = Dims(tp)
        self.tp = tp
        self.device = torch.device(device)
        self.timeline = timeline
        dev = self.device
        self.scratch = torch.zeros(scratch_bytes(MAX_TOKENS, self.d), dtype=torch.uint8, device=dev)
        # [epoch, -, -, -, a mark per CTA (back.EPOCH_MARKS)]
        self.epoch = torch.zeros(4 + 256, dtype=torch.int32, device=dev)
        self.q = torch.zeros(MAX_TOKENS, self.d.heads, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        self.kt = torch.zeros(MAX_TOKENS * KEYS, dtype=torch.int32, device=dev)
        self.klen = torch.zeros(2 * MAX_TOKENS, dtype=torch.int32, device=dev)
        self._dummy = torch.zeros(4, dtype=torch.int32, device=dev)
        # what K2 reads for an absent key: a zero fp8_ds_mla record (+ scales)
        self._zero_rec = torch.zeros(1024, dtype=torch.uint8, device=dev)
        self._qk_scale = (
            torch.tensor([HEAD_DIM**-0.5 * _LOG2E], dtype=torch.float32).view(torch.int32).item()
        )
        self._kernels: dict = {}
        self._warm: set = set()
        self._warm_bufs: dict | None = None
        self.tl_front = self.tl_back = None
        if timeline:
            self.tl_front = torch.zeros(256 * 6, dtype=torch.int64, device=dev)
            self.tl_back = torch.zeros(256 * 16, dtype=torch.int64, device=dev)

    def kernels(self, tokens: int, ratio: int):
        key = (tokens, ratio)
        if key not in self._kernels:
            self._kernels[key] = (
                build_front(FrontBuild(tokens, self.tp, ratio, self.timeline)),
                build_back(BackBuild(tokens, self.tp, ratio, self.timeline)),
            )
        return self._kernels[key]

    def begin_step(self) -> None:
        """Optional: skip an epoch value (the kernels move it on themselves)."""
        self.epoch[:1].add_(1)

    @staticmethod
    def supports(tokens: int) -> bool:
        return 1 <= tokens <= MAX_TOKENS

    def built(self, tokens: int, ratio: int) -> bool:
        return (tokens, ratio) in self._warm

    def warmup(self, w: MegaLayerWeights, tokens: int) -> None:
        """Compile and load the (tokens, w.ratio) kernels with one real launch
        that reads no key and writes no cache row (every row a pad: slot -1),
        outside any graph capture: a capture must not build a kernel."""
        if (tokens, w.ratio) in self._warm:
            return
        dev = self.device
        z = self._warm_bufs
        if z is None or z["rows"] < tokens:
            n = MAX_TOKENS
            z = self._warm_bufs = {
                "rows": n,
                "x": torch.zeros(n, HIDDEN, dtype=torch.bfloat16, device=dev),
                "out": torch.empty(n, HIDDEN, dtype=torch.bfloat16, device=dev),
                "pos": torch.zeros(n, dtype=torch.int64, device=dev),
                "slot": torch.full((n,), -1, dtype=torch.int64, device=dev),
                "cache": torch.zeros(1, 64, 584, dtype=torch.uint8, device=dev),
                "swa_idx": torch.zeros(n, 1, 128, dtype=torch.int32, device=dev),
                "swa_len": torch.zeros(n, dtype=torch.int32, device=dev),
                "t2r": torch.zeros(n, dtype=torch.int32, device=dev),
                "topk": torch.full((n, 512), -1, dtype=torch.int32, device=dev),
                "bt": torch.zeros(1, 1, dtype=torch.int32, device=dev),
            }
        self.begin_step()
        self.forward(
            w, z["x"][:tokens], z["out"][:tokens], z["pos"][:tokens], z["slot"][:tokens],
            z["cache"][:, :32], z["swa_idx"], z["swa_len"], z["t2r"],
            topk_indices=z["topk"] if w.ratio else None,
            comp_cache=z["cache"] if w.ratio else None,
            comp_block_table=z["bt"] if w.ratio else None,
        )  # fmt: skip
        torch.cuda.current_stream().synchronize()
        self._warm.add((tokens, w.ratio))

    def forward(
        self,
        w: MegaLayerWeights,
        hidden: torch.Tensor,  # [M, 5120] bf16 (rows padded to M)
        out: torch.Tensor,  # [M, 5120] bf16: the rank's wo_b partial
        positions: torch.Tensor,  # [M] int64
        slot_mapping: torch.Tensor,  # [M] int64 (-1: pad row)
        swa_cache: torch.Tensor,  # [blocks, block, 584] uint8 view (the layer's)
        swa_indices: torch.Tensor,  # [>= M, ..., 128] int32
        swa_lens: torch.Tensor,  # [>= M] int32
        token_to_req: torch.Tensor,  # [>= M] int32
        topk_indices: torch.Tensor | None = None,  # [>= M, 512] int32 (local)
        comp_cache: torch.Tensor | None = None,  # [blocks, entries, 584] uint8 view
        comp_block_table: torch.Tensor | None = None,  # [reqs, max_blocks] int32
    ) -> torch.Tensor:
        M = hidden.shape[0]
        assert self.supports(M), M
        assert hidden.dtype == torch.bfloat16 and hidden.stride(-1) == 1
        assert out.shape == (M, HIDDEN) and out.dtype == torch.bfloat16 and out.stride(-1) == 1
        # the kernels read the int64 rows' low words and fixed row strides
        assert positions.dtype == torch.int64 and positions.is_contiguous()
        assert slot_mapping.dtype == torch.int64 and slot_mapping.is_contiguous()
        assert swa_indices.dtype == torch.int32 and swa_indices.stride(0) == 128
        assert swa_lens.dtype == torch.int32 and token_to_req.dtype == torch.int32
        assert swa_cache.dtype == torch.uint8 and swa_cache.shape[-1] == 584
        ratio = w.ratio
        front, back = self.kernels(M, ratio)
        st = torch.cuda.current_stream()
        swa_block = swa_cache.shape[1]
        if ratio:
            assert comp_cache is not None and comp_block_table is not None
            assert topk_indices is not None and topk_indices.shape[-1] == 512
            assert topk_indices.dtype == torch.int32 and topk_indices.stride(0) == 512
            assert comp_cache.dtype == torch.uint8 and comp_cache.shape[-1] == 584
            assert comp_block_table.dtype == torch.int32
            comp, comp_stride, comp_block = comp_cache, comp_cache.stride(0), comp_cache.shape[1]
            bt, bt_stride = comp_block_table, comp_block_table.stride(0)
            topk = topk_indices
        else:
            comp, comp_stride, comp_block = self._dummy, 0, 1
            bt, bt_stride, topk = self._dummy, 0, self._dummy
        tlf = 0 if self.tl_front is None else self.tl_front.data_ptr()
        tlb = 0 if self.tl_back is None else self.tl_back.data_ptr()
        front(
            hidden.data_ptr(), hidden.stride(0), w.wqkv.data_ptr(), w.wqkv_scale.data_ptr(),
            w.q_norm.data_ptr(), w.kv_norm.data_ptr(), w.wq_b.data_ptr(), w.wq_b_scale.data_ptr(),
            w.cos_sin.data_ptr(), positions.data_ptr(), slot_mapping.data_ptr(),
            swa_cache.data_ptr(), swa_cache.stride(0), swa_block, self.q.data_ptr(),
            swa_indices.data_ptr(), swa_lens.data_ptr(), comp_block, bt.data_ptr(), bt_stride,
            topk.data_ptr(), token_to_req.data_ptr(), self.kt.data_ptr(), self.klen.data_ptr(),
            self.scratch.data_ptr(), self.epoch.data_ptr(), w.layer_id, tlf, stream=st,
        )  # fmt: skip
        back(
            self.q.data_ptr(), swa_cache.data_ptr(), swa_cache.stride(0), swa_block,
            comp.data_ptr(), comp_stride, comp_block, self.kt.data_ptr(), self.klen.data_ptr(),
            positions.data_ptr(), w.attn_sink.data_ptr(), self._qk_scale, w.cos_sin.data_ptr(),
            w.wo_a.data_ptr(), w.wo_a_scale.data_ptr(), w.wo_b.data_ptr(), w.wo_b_scale.data_ptr(),
            out.data_ptr(), out.stride(0), self.scratch.data_ptr(), self.epoch.data_ptr(),
            w.layer_id, tlb, self._zero_rec.data_ptr(), stream=st,
        )  # fmt: skip
        return out
