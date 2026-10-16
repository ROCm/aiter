# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused MiniMax-M3 sparse-layer decode: one launch per layer and rank.

Each launch runs the layer's input norm, QKV GEMV, head norm / RoPE / KV and
index-cache insert, indexer scoring and top-k, sparse attention, o_proj and its
all-reduce, router, MoE and its all-reduce. The all-reduces go through a
peer-mapped buffer inside the kernel, so every rank of the TP group must launch
the same layers with the same step shape, or the group deadlocks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import NamedTuple

import torch
import torch.distributed as dist

from aiter.ops.flydsl.kernels.minimax_m3_mono.config import (
    BLOCKS,
    HEAD_DIM,
    HIDDEN,
    INTER,
    MAX_TOKENS,
    N_ROUTED,
    O_K,
    ONE_INDEX_HEAD,
    ROTARY_DIM,
    TP,
    CacheLayout,
)
from aiter.ops.flydsl.kernels.minimax_m3_mono.layout import SCRATCH_BYTES as K4_SCRATCH
from aiter.ops.flydsl.kernels.minimax_m3_mono.layout import sym_layout
from aiter.ops.flydsl.kernels.minimax_m3_mono.post_attn import build_post_attn_kernel
from aiter.ops.flydsl.kernels.minimax_m3_mono.pre_attn import K1_ARGS
from aiter.ops.flydsl.kernels.minimax_m3_mono.pre_attn import (
    SCRATCH_BYTES as K1_SCRATCH,
)
from aiter.ops.flydsl.quick_allreduce_int4_ipc import UncachedIpcHeap

__all__ = [
    "MAX_TOKENS",
    "CacheLayout",
    "MiniMaxM3MonoDecode",
    "MonoLayerCaches",
    "MonoLayerWeights",
]

_FP8 = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
_EXPERTS = N_ROUTED + 1


class MonoLayerCaches(NamedTuple):
    """The engine-owned tensors of one layer that can be reallocated: the op does
    not keep them alive, and ``run`` checks them against the registered ones."""

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    index_cache: torch.Tensor


@dataclass(frozen=True)
class MonoLayerWeights:
    """The tensors one sparse MoE layer reads, as the unfused path holds them.

    ``w_qkv`` / ``w_o``: per-channel FP8 weights preshuffled to aiter's (16, 16)
    layout, rows ``q | k | v | index_q | index_k`` for ``w_qkv``. ``w13`` /
    ``w2``: MXFP4 experts (routed then the fused shared one), gate rows before up
    rows, shuffled by ``shuffle_weight``; their E8M0 scales by ``shuffle_scale``.
    ``k_cache`` / ``v_cache``: page-16 SHUFFLE views addressed by K-side page id
    (``CacheLayout.block_pages`` pages a block). ``k_scale`` / ``v_scale``: fp32,
    one value each when the layout's scale is fixed.
    """

    input_norm: torch.Tensor
    w_qkv: torch.Tensor
    s_qkv: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    index_q_norm: torch.Tensor
    index_k_norm: torch.Tensor
    cos_sin: torch.Tensor
    w_o: torch.Tensor
    s_o: torch.Tensor
    post_norm: torch.Tensor
    gate: torch.Tensor
    gate_bias: torch.Tensor
    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    index_cache: torch.Tensor

    @property
    def caches(self) -> MonoLayerCaches:
        return MonoLayerCaches(*(getattr(self, f) for f in MonoLayerCaches._fields))


def _check(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(f"MiniMax-M3 mono decode: {what}")


def _validate(w: MonoLayerWeights, cache: CacheLayout, gate_fp32: bool) -> None:
    for f in fields(w):
        t = getattr(w, f.name)
        _check(t.is_cuda, f"{f.name} is not on the GPU")
    for name in ("input_norm", "post_norm"):
        t = getattr(w, name)
        _check(t.dtype == torch.bfloat16 and t.numel() == HIDDEN, name)
    for name in ("q_norm", "k_norm", "index_q_norm", "index_k_norm"):
        t = getattr(w, name)
        _check(t.dtype == torch.bfloat16 and t.numel() == HEAD_DIM, name)
    qkv_rows = ONE_INDEX_HEAD.rows
    _check(
        w.w_qkv.dtype in _FP8 and tuple(w.w_qkv.shape) == (qkv_rows, HIDDEN),
        f"w_qkv must be fp8 [{qkv_rows}, {HIDDEN}], got "
        f"{w.w_qkv.dtype} {tuple(w.w_qkv.shape)}",
    )
    _check(
        w.w_o.dtype in _FP8 and tuple(w.w_o.shape) == (HIDDEN, O_K),
        f"w_o must be fp8 [{HIDDEN}, {O_K}], got {w.w_o.dtype} {tuple(w.w_o.shape)}",
    )
    _check(
        w.s_qkv.dtype == torch.float32 and w.s_qkv.numel() == qkv_rows,
        "s_qkv must hold one fp32 scale per output row",
    )
    _check(
        w.s_o.dtype == torch.float32 and w.s_o.numel() == HIDDEN,
        "s_o must hold one fp32 scale per output row",
    )
    _check(
        w.cos_sin.dtype == torch.bfloat16
        and w.cos_sin.dim() == 2
        and w.cos_sin.shape[1] == ROTARY_DIM,
        "cos_sin must be bf16 [max_position, rotary_dim], cos then sin halves",
    )
    gate_dtype = torch.float32 if gate_fp32 else torch.bfloat16
    _check(
        w.gate.dtype == gate_dtype and tuple(w.gate.shape) == (N_ROUTED, HIDDEN),
        f"gate must be {gate_dtype} [{N_ROUTED}, {HIDDEN}]",
    )
    _check(
        w.gate_bias.dtype == torch.float32 and w.gate_bias.numel() == N_ROUTED,
        "gate_bias must be fp32 [n_routed]",
    )
    _check(
        w.w13.numel() * w.w13.element_size() == _EXPERTS * 2 * INTER * HIDDEN // 2
        and w.w2.numel() * w.w2.element_size() == _EXPERTS * HIDDEN * INTER // 2,
        "expert weights must be MXFP4 with the shared expert fused last",
    )
    _check(
        w.s13.numel() == _EXPERTS * 2 * INTER * HIDDEN // 32
        and w.s2.numel() == _EXPERTS * HIDDEN * INTER // 32,
        "expert scales must be E8M0, one per 32 weights",
    )
    _check(
        w.k_cache.dtype in _FP8 and w.v_cache.dtype in _FP8,
        "KV cache must be fp8",
    )
    _check(w.index_cache.dtype in _FP8, "index cache must be fp8")
    if cache.scalar_kv_scale:
        _check(
            w.k_scale.numel() == 1 and w.v_scale.numel() == 1,
            "fixed KV scales must be single values",
        )
    for name in ("w_qkv", "w_o", "w13", "w2", "s13", "s2", "gate", "cos_sin"):
        _check(getattr(w, name).is_contiguous(), f"{name} must be contiguous")


class _Layer(NamedTuple):
    layer_id: int
    weights: tuple[torch.Tensor, ...]  # held: the kernel reads them every step
    ptrs: dict[str, int]
    k1_args: torch.Tensor


class _PeerBuffer:
    """One uncached buffer per rank, every rank holding every peer's address."""

    def __init__(self, nbytes: int, group, rank: int, device: torch.device):
        self.local = UncachedIpcHeap.alloc_uncached(nbytes)
        self._opened: list[int] = []
        handles = UncachedIpcHeap.gather_object_list_via_broadcast(
            group, UncachedIpcHeap.get_mem_handle_bytes(self.local)
        )
        addresses = []
        for peer, handle in enumerate(handles):
            if peer == rank:
                addresses.append(self.local)
            else:
                base = UncachedIpcHeap.open_mem_handle(handle)
                self._opened.append(base)
                addresses.append(base)
        dist.barrier(group=group)
        self.addresses = torch.tensor(addresses, dtype=torch.int64, device=device)

    def close(self) -> None:
        opened, self._opened = self._opened, []
        for base in opened:
            UncachedIpcHeap.close_mem_handle(base)
        if self.local:
            UncachedIpcHeap.free_device_mem(self.local)
            self.local = 0


class MiniMaxM3MonoDecode:
    """Owns the fused sparse-layer kernels, their scratch and the peer buffer.

    ``group``: the TP group's CPU (gloo) process group, used once to exchange the
    peer buffer handles. Build it outside CUDA graph capture.

    The first ``run`` of each token count compiles that count's kernel, so run
    every count a graph will capture (1..MAX_TOKENS) eagerly first; capturing an
    uncompiled count raises.
    """

    def __init__(
        self,
        group,
        rank: int,
        world_size: int,
        device: torch.device,
        *,
        sm_scale: float,
        eps: float,
        route_scale: float,
        shared_weight: float,
        swiglu_limit: float,
        init_blocks: int,
        local_blocks: int,
        scalar_kv_scale: bool,
        gate_fp32: bool,
    ):
        _check(world_size == TP, f"needs tensor parallel size {TP}, got {world_size}")
        cus = torch.cuda.get_device_properties(device).multi_processor_count
        _check(cus == BLOCKS, f"needs {BLOCKS} compute units, got {cus}")
        self.rank = rank
        self.cache = CacheLayout(scalar_kv_scale=scalar_kv_scale)
        self.gate_fp32 = gate_fp32
        self._build_args = (
            world_size, sm_scale, eps, route_scale, shared_weight, swiglu_limit,
            init_blocks, local_blocks,
        )  # fmt: skip
        self._kernels: dict[int, object] = {}
        self._layers: list[_Layer] = []

        u8, bf16 = torch.uint8, torch.bfloat16
        self._scratch1 = torch.zeros(K1_SCRATCH, dtype=u8, device=device)
        self._scratch4 = torch.zeros(K4_SCRATCH, dtype=u8, device=device)
        self._step = torch.zeros(1, dtype=torch.int32, device=device)
        # layer i reads ars[i % 2] / writes ars[(i + 1) % 2], likewise h_mids
        self._ars = [torch.empty(MAX_TOKENS, HIDDEN, dtype=bf16, device=device)
                     for _ in range(2)]  # fmt: skip
        self._h_mids = [torch.empty(MAX_TOKENS, HIDDEN, dtype=bf16, device=device)
                        for _ in range(2)]  # fmt: skip
        self._h = torch.empty(MAX_TOKENS, HIDDEN, dtype=bf16, device=device)
        self._q = torch.empty(MAX_TOKENS, O_K, dtype=bf16, device=device)
        self._iq = torch.empty(MAX_TOKENS, HEAD_DIM, dtype=bf16, device=device)
        self._peers = _PeerBuffer(sym_layout(world_size)["_bytes"], group, rank, device)

    def register_layers(
        self, layers: list[tuple[int, MonoLayerWeights]], *, block_pages: int
    ) -> None:
        """Set the sparse layers, in model order, replacing any registered before.

        ``block_pages``: page-16 ids one KV cache block spans in the K-side
        numbering (``CacheLayout.block_pages``). The kernels hold raw pointers:
        call this again whenever a tensor it was given is reallocated (an
        engine's KV cache after memory profiling); ``run`` raises until then. The
        weights are kept alive, the ``MonoLayerCaches`` tensors are not.
        """
        cache = CacheLayout(block_pages, self.cache.scalar_kv_scale)
        if cache != self.cache:
            self.cache = cache
            self._kernels.clear()
        self._layers = []
        for layer_id, weights in layers:
            self._register(layer_id, weights)

    def _register(self, layer_id: int, weights: MonoLayerWeights) -> None:
        _validate(weights, self.cache, self.gate_fp32)
        i = len(self._layers)
        ptrs = {
            "ar": self._ars[i % 2].data_ptr(),
            "g_in": weights.input_norm.data_ptr(),
            "w_qkv": weights.w_qkv.data_ptr(),
            "s_qkv": weights.s_qkv.data_ptr(),
            "g_q": weights.q_norm.data_ptr(),
            "g_k": weights.k_norm.data_ptr(),
            "g_iq": weights.index_q_norm.data_ptr(),
            "g_ik": weights.index_k_norm.data_ptr(),
            "cos_sin": weights.cos_sin.data_ptr(),
            "index_cache": weights.index_cache.data_ptr(),
            "iq_out": self._iq.data_ptr(),
            "scratch": self._scratch1.data_ptr(),
        }
        k1_args = torch.tensor(
            [ptrs[a] for a in K1_ARGS], dtype=torch.int64, device=self._step.device
        )
        tensors = {f.name: getattr(weights, f.name) for f in fields(weights)}
        self._layers.append(
            _Layer(
                layer_id,
                tuple(
                    t for name, t in tensors.items()
                    if name not in MonoLayerCaches._fields
                ),
                {name: t.data_ptr() for name, t in tensors.items()},
                k1_args,
            )
        )  # fmt: skip

    def _check_caches(self, caches: Sequence[MonoLayerCaches]) -> None:
        if len(caches) != len(self._layers):
            raise RuntimeError(
                f"MiniMax-M3 mono decode: {len(caches)} layers' caches given, "
                f"{len(self._layers)} registered"
            )
        for layer, current in zip(self._layers, caches):
            for name, t in zip(MonoLayerCaches._fields, current):
                if t.data_ptr() != layer.ptrs[name]:
                    raise RuntimeError(
                        f"MiniMax-M3 mono decode: layer {layer.layer_id}'s {name} "
                        "is not the registered tensor; call register_layers() "
                        "after reallocating a cache"
                    )

    def _kernel(self, tokens: int):
        k = self._kernels.get(tokens)
        if k is None:
            k = build_post_attn_kernel(
                *self._build_args,
                tokens=tokens,
                fuse_k1=True,
                cache=self.cache,
                gate_fp32=self.gate_fp32,
            )
            self._kernels[tokens] = k
        return k

    def run(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        caches: Sequence[MonoLayerCaches],
        query_len: int = 1,
        aux_layers: tuple[int, ...] = (),
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Run every registered layer on one decode step of ``n`` tokens.

        ``hidden``: the TP-reduced output of the layer before the first sparse
        one and ``residual`` its residual stream, both bf16 [n, hidden].
        ``positions`` / ``slot_mapping``: int64 [n], a slot being
        ``block * SPARSE_BLOCK + offset`` in cache blocks (negative to skip the
        write). ``block_table`` / ``seq_lens``: int32, a row per token (a request
        of ``query_len`` tokens repeats its row, token j seeing
        ``seq_len - (query_len - 1 - j)`` keys). ``caches``: every registered
        layer's caches as the engine holds them now, in registration order; any
        that is not the registered tensor raises before a launch. ``aux_layers``:
        registered layer ids whose output stream (FFN output plus residual) to
        return, in order. Returns the last layer's TP-reduced FFN output and its
        residual, the final norm's inputs, and the requested layer outputs.
        """
        n = hidden.shape[0]
        _check(1 <= n <= MAX_TOKENS, f"step of {n} tokens")
        _check(n % query_len == 0, "tokens must be whole requests")
        _check(
            positions.dtype == torch.int64 and slot_mapping.dtype == torch.int64,
            "positions and slot_mapping must be int64",
        )
        _check(
            block_table.dtype == torch.int32
            and seq_lens.dtype == torch.int32
            and block_table.shape[0] >= n
            and seq_lens.shape[0] >= n
            and block_table.stride(0) == block_table.shape[1]
            and block_table.stride(1) == 1,
            "block_table and seq_lens must be int32 with a contiguous row per token",
        )
        self._check_caches(caches)
        if n not in self._kernels and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"MiniMax-M3 mono decode: the {n}-token kernel is not compiled; "
                "run each step size eagerly before CUDA graph capture"
            )
        kernel = self._kernel(n)
        self._ars[0][:n].copy_(hidden)
        res = residual
        aux = []
        stream = torch.cuda.current_stream()
        for i, layer in enumerate(self._layers):
            p = layer.ptrs
            out = self._h_mids[(i + 1) % 2]
            kernel(
                self._h.data_ptr(), self._q.data_ptr(), block_table.data_ptr(),
                seq_lens.data_ptr(), p["k_cache"], p["v_cache"], p["k_scale"],
                p["v_scale"], p["w_o"], p["s_o"], p["post_norm"], p["gate"],
                p["gate_bias"], p["w13"], p["s13"], p["w2"], p["s2"], out.data_ptr(),
                self._ars[(i + 1) % 2].data_ptr(), self._scratch4.data_ptr(),
                self._peers.local, self._peers.addresses.data_ptr(),
                self._step.data_ptr(), self.rank, layer.layer_id,
                block_table.shape[1], query_len, layer.k1_args.data_ptr(),
                positions.data_ptr(), slot_mapping.data_ptr(), res.data_ptr(),
                stream=stream,
            )  # fmt: skip
            res = out[:n]
            if layer.layer_id in aux_layers:
                aux.append(self._ars[(i + 1) % 2][:n] + res)
        self._step.add_(1)
        return self._ars[len(self._layers) % 2][:n], res, aux

    def close(self) -> None:
        self._peers.close()
