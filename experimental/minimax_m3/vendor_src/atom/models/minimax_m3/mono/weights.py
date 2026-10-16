# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Zero-copy views of one sparse MoE layer (``SparseMoeLayer``) or dense layer
(``DenseLayer``), checked against what the kernels read.

The mono kernels read the tensors the original path already loaded, quantized
and shuffled. Every layout assumption is asserted here once, so a model loaded
differently is refused (``MonoUnsupported``) instead of being misread.
"""

from dataclasses import dataclass

import torch
from aiter import QuantType, dtypes

from atom.model_ops.linear import weight_is_stored_preshuffled
from atom.models.minimax_m3.mono.config import (
    DENSE_INTER,
    HEAD_DIM,
    HIDDEN,
    INTER,
    LOCAL_Q_HEADS,
    N_ROUTED,
    O_K,
    ROTARY_DIM,
    TOP_K,
    TOPK_BLOCKS,
    IndexHeads,
)
from atom.mono.runtime.consensus import MonoUnsupported


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


def _ptpc_fp8(
    linear, rows: int, cols: int, name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weight + per-output-channel scale of a ptpc-FP8 linear (per-token activation
    quant, per-channel weight scale), the weight preshuffled (16, 16)."""
    w, s = linear.weight, getattr(linear, "weight_scale", None)
    _need(
        linear.quant_type.value == QuantType.per_Token.value,
        f"{name}: not per-token FP8",
    )
    _need(
        w.dtype == dtypes.fp8 and tuple(w.shape) == (rows, cols),
        f"{name}: weight {w.dtype} {tuple(w.shape)}",
    )
    _need(
        weight_is_stored_preshuffled(linear.quant_type, linear.params_dtype),
        f"{name}: not preshuffled",
    )
    _need(not getattr(linear, "is_output_padded", False), f"{name}: output padded")
    _need(
        s is not None and s.dtype == torch.float32 and s.numel() == rows,
        f"{name}: scale",
    )
    return w.data, s.data.view(rows)


def _bf16_vec(t: torch.Tensor, n: int, name: str) -> torch.Tensor:
    _need(
        t.dtype == torch.bfloat16 and t.numel() == n and t.is_contiguous(),
        f"{name}: {t.dtype} {tuple(t.shape)}",
    )
    return t.data


def _cos_sin(attn, layer_id: int) -> torch.Tensor:
    """The [max_pos, rotary_dim] bf16 cos|sin table the attention module registered
    on its rotary embedding (``_minimax_m3_cos_sin_cache``)."""
    t = getattr(attn.rotary_emb, "_minimax_m3_cos_sin_cache", None)
    _need(
        t is not None
        and t.dtype == torch.bfloat16
        and t.dim() == 2
        and t.shape[1] == ROTARY_DIM,
        f"layer {layer_id}: cos/sin cache",
    )
    return t


@dataclass(frozen=True)
class SparseMoeLayer:
    """Everything K1 / K4 read for one sparse-attention MoE layer."""

    layer_id: int
    attn_impl: object  # SparseMHAPagedAttentionImpl: index cache + page-16 KV views
    # the layer selects its blocks; else it reuses the step's last selecting
    # layer's (the original path's ``skip_index_topk``)
    index_topk: bool
    g_in: torch.Tensor
    w_qkv: torch.Tensor
    s_qkv: torch.Tensor
    g_q: torch.Tensor
    g_k: torch.Tensor
    g_iq: torch.Tensor
    g_ik: torch.Tensor
    cos_sin: torch.Tensor
    w_o: torch.Tensor
    s_o: torch.Tensor
    g_post: torch.Tensor
    gate: torch.Tensor
    bias: torch.Tensor
    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor

    @staticmethod
    def from_layer(layer, layer_id: int, heads: IndexHeads) -> "SparseMoeLayer":
        """``heads``: the index q heads the deployment's fused projection holds."""
        attn = layer.self_attn
        _need(
            getattr(attn, "is_indexed_sparse_attention", False) and layer.is_moe_layer,
            f"layer {layer_id}: not sparse MoE",
        )
        _need(
            (attn.num_heads, attn.num_kv_heads, attn.num_idx_heads, attn.head_dim)
            == (LOCAL_Q_HEADS, 1, heads.count, HEAD_DIM),
            f"layer {layer_id}: heads {attn.num_heads}/{attn.num_kv_heads}/{attn.num_idx_heads}",
        )
        _need(attn.rotary_emb.rotary_dim == ROTARY_DIM, f"layer {layer_id}: rotary_dim")
        _need(
            (attn.topk_blocks, attn.init_blocks, attn.local_blocks)
            == (TOPK_BLOCKS, 0, 1),
            f"layer {layer_id}: sparse selection config",
        )
        impl = attn.attn.impl
        _need(
            impl.kv_cache_dtype == "fp8" and impl.index_cache is not None,
            f"layer {layer_id}: caches",
        )
        _need(
            impl.index_cache.dtype == dtypes.fp8,
            f"layer {layer_id}: index cache {impl.index_cache.dtype}",
        )
        w_qkv, s_qkv = _ptpc_fp8(
            attn.qkv_proj, heads.rows, HIDDEN, f"layer {layer_id} qkv_proj"
        )
        w_o, s_o = _ptpc_fp8(attn.o_proj, HIDDEN, O_K, f"layer {layer_id} o_proj")

        moe = layer.block_sparse_moe
        experts = moe.experts
        qm = experts.quant_method
        E = N_ROUTED + 1
        _need(
            moe.fuse_shared_experts and experts.num_fused_shared_experts == 1,
            f"layer {layer_id}: shared expert",
        )
        _need(
            experts.top_k == TOP_K and not getattr(qm, "is_guinterleave", True),
            f"layer {layer_id}: moe layout",
        )
        _need(
            getattr(qm, "intermediate_pad", 1) == 0
            and getattr(qm, "hidden_pad", 1) == 0,
            f"layer {layer_id}: pad",
        )
        w13, w2 = experts.w13_weight, experts.w2_weight
        _need(
            getattr(w13, "is_shuffled", False) and getattr(w2, "is_shuffled", False),
            f"layer {layer_id}: shuffle",
        )
        _need(
            w13.numel() * w13.element_size() == E * 2 * INTER * HIDDEN // 2
            and w2.numel() * w2.element_size() == E * HIDDEN * INTER // 2,
            f"layer {layer_id}: expert weight bytes",
        )
        s13, s2 = experts.w13_weight_scale, experts.w2_weight_scale
        _need(
            s13.numel() == E * 2 * INTER * HIDDEN // 32
            and s2.numel() == E * HIDDEN * INTER // 32,
            f"layer {layer_id}: expert scale bytes",
        )
        gate = moe.gate.weight
        _need(
            gate.dtype == torch.bfloat16 and tuple(gate.shape) == (N_ROUTED, HIDDEN),
            f"layer {layer_id}: gate",
        )
        bias = moe.e_score_correction_bias
        _need(
            bias is not None
            and bias.dtype == torch.float32
            and bias.numel() == N_ROUTED,
            f"layer {layer_id}: bias",
        )
        return SparseMoeLayer(
            layer_id=layer_id,
            attn_impl=impl,
            index_topk=not attn.skip_index_topk,
            g_in=_bf16_vec(layer.input_layernorm.weight, HIDDEN, "input_layernorm"),
            w_qkv=w_qkv,
            s_qkv=s_qkv,
            g_q=_bf16_vec(attn.q_norm.weight, HEAD_DIM, "q_norm"),
            g_k=_bf16_vec(attn.k_norm.weight, HEAD_DIM, "k_norm"),
            g_iq=_bf16_vec(attn.index_q_norm.weight, HEAD_DIM, "index_q_norm"),
            g_ik=_bf16_vec(attn.index_k_norm.weight, HEAD_DIM, "index_k_norm"),
            cos_sin=_cos_sin(attn, layer_id),
            w_o=w_o,
            s_o=s_o,
            g_post=_bf16_vec(
                layer.post_attention_layernorm.weight,
                HIDDEN,
                "post_attention_layernorm",
            ),
            gate=gate.data,
            bias=bias.data,
            w13=w13.data,
            s13=s13.data,
            w2=w2.data,
            s2=s2.data,
        )


@dataclass(frozen=True)
class DenseLayer:
    """Everything dense_pre / dense_post read for one dense layer, and the original
    attention its decode step calls between them."""

    layer_id: int
    attn_impl: object  # the MHA impl: its decode kernel and the page-128 caches
    g_in: torch.Tensor
    w_qkv: torch.Tensor
    s_qkv: torch.Tensor
    g_q: torch.Tensor
    g_k: torch.Tensor
    cos_sin: torch.Tensor
    w_o: torch.Tensor
    s_o: torch.Tensor
    g_post: torch.Tensor
    w_gu: torch.Tensor
    s_gu: torch.Tensor
    w_dn: torch.Tensor
    s_dn: torch.Tensor
    swiglu_alpha: float
    swiglu_beta: float
    swiglu_limit: float

    @staticmethod
    def from_layer(layer, layer_id: int) -> "DenseLayer":
        attn = layer.self_attn
        _need(
            not getattr(attn, "is_indexed_sparse_attention", False)
            and not layer.is_moe_layer,
            f"layer {layer_id}: not dense",
        )
        _need(
            (attn.num_heads, attn.num_kv_heads, attn.head_dim)
            == (LOCAL_Q_HEADS, 1, HEAD_DIM),
            f"layer {layer_id}: heads {attn.num_heads}/{attn.num_kv_heads}",
        )
        _need(attn.rotary_emb.rotary_dim == ROTARY_DIM, f"layer {layer_id}: rotary_dim")
        impl = attn.attn.impl
        _need(
            impl.kv_cache_dtype == "fp8" and getattr(impl, "sliding_window", -1) == -1,
            f"layer {layer_id}: cache dtype / window",
        )
        rows = (LOCAL_Q_HEADS + 2) * HEAD_DIM
        w_qkv, s_qkv = _ptpc_fp8(
            attn.qkv_proj, rows, HIDDEN, f"layer {layer_id} qkv_proj"
        )
        w_o, s_o = _ptpc_fp8(attn.o_proj, HIDDEN, O_K, f"layer {layer_id} o_proj")
        mlp = layer.mlp
        w_gu, s_gu = _ptpc_fp8(
            mlp.gate_up_proj, 2 * DENSE_INTER, HIDDEN, f"layer {layer_id} gate_up_proj"
        )
        w_dn, s_dn = _ptpc_fp8(
            mlp.down_proj, HIDDEN, DENSE_INTER, f"layer {layer_id} down_proj"
        )
        _need(mlp.swiglu_limit is not None, f"layer {layer_id}: swiglu limit")
        return DenseLayer(
            layer_id=layer_id,
            attn_impl=impl,
            g_in=_bf16_vec(layer.input_layernorm.weight, HIDDEN, "input_layernorm"),
            w_qkv=w_qkv,
            s_qkv=s_qkv,
            g_q=_bf16_vec(attn.q_norm.weight, HEAD_DIM, "q_norm"),
            g_k=_bf16_vec(attn.k_norm.weight, HEAD_DIM, "k_norm"),
            cos_sin=_cos_sin(attn, layer_id),
            w_o=w_o,
            s_o=s_o,
            g_post=_bf16_vec(
                layer.post_attention_layernorm.weight,
                HIDDEN,
                "post_attention_layernorm",
            ),
            w_gu=w_gu,
            s_gu=s_gu,
            w_dn=w_dn,
            s_dn=s_dn,
            swiglu_alpha=float(mlp.swiglu_alpha),
            swiglu_beta=float(mlp.swiglu_beta),
            swiglu_limit=float(mlp.swiglu_limit),
        )
