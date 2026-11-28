# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Inputs for opt-in MiniMax-M3 cross-repository regression tests."""

import importlib
import os
from types import SimpleNamespace

import torch
import triton

from aiter.ops.triton.attention.minimax_m3 import minimax_m3_prefill_score


def router_inputs(tokens):
    return {
        "x": torch.randn(tokens, 6144, dtype=torch.bfloat16, device="cuda"),
        "w": torch.randn(6144, 128, dtype=torch.bfloat16, device="cuda").T,
    }


def atom_module(module):
    """Load a public ATOM wrapper only when the caller opts in."""
    if os.environ.get("AITER_TEST_MINIMAX_M3_ATOM") != "1":
        raise RuntimeError("Set AITER_TEST_MINIMAX_M3_ATOM=1 and install ATOM")
    return importlib.import_module("atom.model_ops." + module)


def atom_op(module, name):
    return getattr(atom_module(module), name)


def swiglu_inputs(tokens):
    return {"gate_up": torch.randn(tokens, 6144, dtype=torch.bfloat16, device="cuda")}


def swiglu(gate_up):
    return atom_op("swiglu_oai", "swiglu_oai_split")(gate_up, 1.702, 1.0, 7.0)


def index_inputs(batch=2, query=4, context=4096, heads=4):
    blocks = (context + 127) // 128
    return {
        "idx_q": torch.randn(
            batch * query, heads, 128, dtype=torch.bfloat16, device="cuda"
        )
        * 0.1,
        "index_cache": torch.randn(
            batch * blocks, 128, 128, dtype=torch.bfloat16, device="cuda"
        ).to(torch.float8_e4m3fn),
        "block_table": torch.randperm(batch * blocks, device="cuda")
        .to(torch.int32)
        .reshape(batch, blocks),
        "seq_lens": torch.full((batch,), context, dtype=torch.int32, device="cuda"),
        "context": context,
        "query": query,
    }


def context_score(idx_q, index_cache, block_table, seq_lens, context, query):
    return atom_op("minimax_m3.indexer_context_parallel", "indexer_context_scores")(
        idx_q, index_cache, block_table, seq_lens, context, 0, 4, query, 128**-0.5
    )


def prefill_inputs(query=31, context=4096):
    inputs = index_inputs(1, query, context, 1)
    # Exact FP8 values give a distinct score to each block of this oracle case.
    if context == 4096:
        inputs["idx_q"].fill_(1)
        block = torch.arange(32, 64, dtype=torch.uint8, device="cuda").view(
            torch.float8_e4m3fn
        )
        inputs["index_cache"] = (
            block[:, None, None]
            .expand(32, 128, 128)
            .contiguous()
            .to(torch.float8_e4m3fn)
        )
        inputs["block_table"] = torch.arange(32, dtype=torch.int32, device="cuda")[None]
    inputs["cu_seqlens_q"] = torch.tensor([0, query], dtype=torch.int32, device="cuda")
    inputs["prefix_lens"] = torch.tensor(
        [context - query], dtype=torch.int32, device="cuda"
    )
    return inputs


def index_prefill(
    idx_q, index_cache, block_table, seq_lens, context, query, cu_seqlens_q, prefix_lens
):
    return atom_op("minimax_m3.index_topk", "minimax_m3_index_topk")(
        idx_q,
        index_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        query,
        context,
        16,
        0,
        0,
        1,
        128**-0.5,
        emit_sparse_block_table=True,
    )


def prefill_score_inputs(query=31, context=4096):
    """Preallocate the output of the pinned ATOM prefill score launch."""
    data = prefill_inputs(query, context)
    data["score"] = torch.empty(
        (1, query, triton.cdiv(context, 128)), dtype=torch.float32, device="cuda"
    )
    return data


def packed_topk_inputs(query=31, context=4096):
    """Produce real score values and allocate selector outputs before timing."""
    data = prefill_score_inputs(query, context)
    score = minimax_m3_prefill_score(**data)
    op = atom_module("minimax_m3.index_topk")
    _, emit_args = op._alloc_emit(
        query,
        1,
        16,
        data["block_table"],
        True,
        score.device,
        op.PAGES_PER_SPARSE_BLOCK,
    )
    return {
        "score": score,
        "topk_idx": torch.empty((1, query, 16), dtype=torch.int32, device="cuda"),
        "cu_seqlens_q": data["cu_seqlens_q"],
        "prefix_lens": data["prefix_lens"],
        "block_table": data["block_table"],
        "emit_args": emit_args,
        "query": query,
    }


def candidate_inputs(tokens=8, context=4096):
    blocks = (context + 127) // 128
    local = (blocks + 3) // 4
    scores = torch.arange(local, dtype=torch.float32, device="cuda") * 4
    return {
        "scores": scores[None, None].expand(4, tokens, local).contiguous(),
        "seq_lens": torch.full(
            (tokens // 4,), context, dtype=torch.int32, device="cuda"
        ),
        "context": context,
    }


def local_topk(scores, seq_lens, context):
    return atom_op("minimax_m3.indexer_candidate_exchange", "local_candidate_keys")(
        scores, seq_lens, 16, 0, 4, 4, (context + 127) // 128
    )


def merge_inputs(tokens=8, context=4096):
    data = candidate_inputs(tokens, context)
    local = atom_op("minimax_m3.indexer_candidate_exchange", "local_candidate_keys")
    keys = []
    for rank in range(4):
        keys.append(
            local(
                data["scores"] + rank,
                data["seq_lens"],
                16,
                rank,
                4,
                4,
                (context + 127) // 128,
            )[0]
        )
    blocks = (context + 127) // 128
    return {
        "keys": torch.stack(keys),
        "seq_lens": data["seq_lens"],
        "block_table": torch.arange(
            (tokens // 4) * blocks, dtype=torch.int32, device="cuda"
        ).reshape(tokens // 4, blocks),
    }


def merge_topk(keys, seq_lens, block_table):
    return atom_op("minimax_m3.indexer_candidate_exchange", "merge_candidate_keys")(
        keys, block_table, seq_lens, 16, 0, 0, 4
    )


def norm_rope_inputs(tokens):
    head_dim, heads, page, x = 128, 16, 128, 16
    pages = (tokens + page - 1) // page
    # A fused QKV projection returns strided bands, not three dense allocations.
    fused = torch.randn(
        tokens, (heads + 2) * head_dim, dtype=torch.bfloat16, device="cuda"
    )
    q, k, v = fused.split([heads * head_dim, head_dim, head_dim], dim=-1)
    angles = torch.randn(tokens + 1, head_dim // 2, dtype=torch.float32, device="cuda")

    def norm():
        return SimpleNamespace(
            weight=torch.randn(head_dim, dtype=torch.bfloat16, device="cuda") * 0.1,
            variance_epsilon=1e-6,
        )

    return {
        "q": q,
        "k": k,
        "v": v,
        "positions": torch.arange(tokens, dtype=torch.int64, device="cuda"),
        "q_norm": norm(),
        "k_norm": norm(),
        "rotary_emb": SimpleNamespace(
            rotary_dim=head_dim, cos_cache=angles.cos(), sin_cache=angles.sin()
        ),
        "num_heads": heads,
        "num_kv_heads": 1,
        "head_dim": head_dim,
        "k_cache": torch.zeros(
            pages, 1, head_dim // x, page, x, dtype=torch.float8_e4m3fn, device="cuda"
        ),
        "v_cache": torch.zeros(
            pages, 1, page // x, head_dim, x, dtype=torch.float8_e4m3fn, device="cuda"
        ),
        "k_scale": torch.zeros(pages, 1, page, dtype=torch.float32, device="cuda"),
        "v_scale": torch.zeros(pages, 1, page, dtype=torch.float32, device="cuda"),
        "slot_mapping": torch.arange(tokens, dtype=torch.int64, device="cuda"),
        "kv_cache_dtype": "fp8",
    }


def norm_rope(**kwargs):
    return atom_op("triton_fused_qkv_norm_rope_cache", "triton_fused_norm_rope_cache")(
        **kwargs
    )


def gather_inputs(tokens=1024):
    page, dim, x = 128, 128, 16
    pages = (tokens + page - 1) // page
    return {
        "key_cache": torch.randn(
            pages, 1, dim // x, page, x, dtype=torch.bfloat16, device="cuda"
        ).to(torch.float8_e4m3fn),
        "value_cache": torch.randn(
            pages, 1, page // x, dim, x, dtype=torch.bfloat16, device="cuda"
        ).to(torch.float8_e4m3fn),
        "key": torch.empty(tokens, 1, dim, dtype=torch.bfloat16, device="cuda"),
        "value": torch.empty(tokens, 1, dim, dtype=torch.bfloat16, device="cuda"),
        "block_tables": torch.arange(pages, dtype=torch.int32, device="cuda")[None],
        "k_scales": torch.rand(pages, 1, page, dtype=torch.float32, device="cuda")
        + 0.1,
        "v_scales": torch.rand(pages, 1, page, dtype=torch.float32, device="cuda")
        + 0.2,
        "cu_seqlens_kv": torch.tensor([0, tokens], dtype=torch.int32, device="cuda"),
        "batch_id_per_k_token": torch.zeros(tokens, dtype=torch.int32, device="cuda"),
        "seq_starts": torch.zeros(1, dtype=torch.int32, device="cuda"),
        "dequant": True,
        "kv_cache_layout": "SHUFFLE",
        "total_tokens": tokens,
        "per_token_quant": True,
    }


def cache_gather(**kwargs):
    atom_op("base_attention", "cp_mha_gather_cache")(**kwargs)
    return kwargs["key"], kwargs["value"]
