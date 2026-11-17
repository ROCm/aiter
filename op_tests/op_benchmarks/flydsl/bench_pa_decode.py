# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL pa_decode over the attention layers of aiter's models.

Each MODELS row is one attention-layer type of a model, from its Hugging Face
``config.json`` (``num_attention_heads``, ``num_key_value_heads``, ``head_dim``,
``sliding_window``, sinks), plus the tensor-parallel sizes it is usually served
with. A rank holds ``q_heads // tp`` query heads and ``max(1, kv_heads // tp)``
KV heads (KV heads replicate past ``kv_heads`` ranks). MLA models (DeepSeek
V3/R1/V4, Kimi K2/K3, GLM-5, Mistral Large 4, GLM-4.7-Flash) and MiMo-V2.5-Pro
(192-dim QK, 128-dim V) do not run through pa_decode and are left out.

The call is the serving one: a preallocated output, a plan from
``plan_pa_decode`` and its workspace. The K/V caches are call arguments, so
``run_perftest`` rotates copies of them and every timed call reads cold K/V.
``--table 0`` bounds ``max_context_length`` by the longest sequence; a positive
``--table`` pads the block table to that many tokens and passes no bound, as
ATOM does (the wrapper then uses the table's capacity).

Run on two trees (e.g. main and a PR) and compare the tables to A/B a change:
    python op_tests/op_benchmarks/flydsl/bench_pa_decode.py
    python op_tests/op_benchmarks/flydsl/bench_pa_decode.py -m qwen3-235b-a22b -d fp8 -q 1 4
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import per_tensor_quant, pertoken_quant
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]

# q_heads / kv_heads / head_dim / sliding_window / sinks from each config.json;
# tp = the tensor-parallel sizes the model is commonly served with.
MODELS = {
    # meta-llama/Llama-3.1-{8B,70B,405B} (model_shapes.json)
    "llama3-8b": {"q_heads": 32, "kv_heads": 8, "head_dim": 128, "tp": (1,)},
    "llama3-70b": {"q_heads": 64, "kv_heads": 8, "head_dim": 128, "tp": (4, 8)},
    "llama3-405b": {"q_heads": 128, "kv_heads": 8, "head_dim": 128, "tp": (8,)},
    # meta-llama/Llama-4-Maverick-17B-128E (model_shapes.json)
    "llama4-maverick": {"q_heads": 40, "kv_heads": 8, "head_dim": 128, "tp": (8,)},
    # openai/gpt-oss-120b: sinks on every layer, half the layers windowed
    "gpt-oss-120b": {
        "q_heads": 64,
        "kv_heads": 8,
        "head_dim": 64,
        "sinks": True,
        "tp": (1, 8),
    },
    "gpt-oss-120b-swa": {
        "q_heads": 64,
        "kv_heads": 8,
        "head_dim": 64,
        "window": 128,
        "sinks": True,
        "tp": (1, 8),
    },
    # Qwen/Qwen3-{8B,14B,32B}; Qwen3-VL-32B has the 32B attention
    "qwen3-8b": {"q_heads": 32, "kv_heads": 8, "head_dim": 128, "tp": (1,)},
    "qwen3-14b": {"q_heads": 40, "kv_heads": 8, "head_dim": 128, "tp": (1,)},
    "qwen3-32b": {"q_heads": 64, "kv_heads": 8, "head_dim": 128, "tp": (1, 2)},
    # Qwen/Qwen3-235B-A22B, Qwen3-VL-235B-A22B
    "qwen3-235b-a22b": {"q_heads": 64, "kv_heads": 4, "head_dim": 128, "tp": (4, 8)},
    # Full-attention layers of Qwen/Qwen3-Next-80B-A3B, Qwen3.5-35B-A3B,
    # Qwen3.6-35B-A3B
    "qwen3-next-80b-a3b": {"q_heads": 16, "kv_heads": 2, "head_dim": 256, "tp": (1, 4)},
    # Full-attention layers of Qwen/Qwen3.5-397B-A17B, Qwen3.5-122B-A10B
    "qwen3.5-397b-a17b": {"q_heads": 32, "kv_heads": 2, "head_dim": 256, "tp": (4, 8)},
    # Full-attention layers of Qwen/Qwen3.6-27B, Qwen3.8-27B
    "qwen3.6-27b": {"q_heads": 24, "kv_heads": 4, "head_dim": 256, "tp": (1, 2)},
    # Full-attention layers of Qwen/Qwen3.8-Flash-Next
    "qwen3.8-flash-next": {"q_heads": 24, "kv_heads": 2, "head_dim": 256, "tp": (1, 2)},
    # MiniMaxAI/MiniMax-M2.5, MiniMax-M3 (and M3.1-Flash)
    "minimax-m2.5": {"q_heads": 48, "kv_heads": 8, "head_dim": 128, "tp": (4, 8)},
    "minimax-m3": {"q_heads": 64, "kv_heads": 4, "head_dim": 128, "tp": (4, 8)},
    # zai-org/GLM-4.7
    "glm-4.7": {"q_heads": 96, "kv_heads": 8, "head_dim": 128, "tp": (4, 8)},
    # tencent/Hy3
    "hy3": {"q_heads": 64, "kv_heads": 8, "head_dim": 128, "tp": (4, 8)},
    # google/gemma-4-31B and gemma-4-26B-A4B: 1024-token sliding layers, and
    # global layers with their own KV heads and a 512 head_dim
    "gemma4-31b-swa": {
        "q_heads": 32,
        "kv_heads": 16,
        "head_dim": 256,
        "window": 1024,
        "tp": (1, 2),
    },
    "gemma4-31b-global": {"q_heads": 32, "kv_heads": 4, "head_dim": 512, "tp": (1, 2)},
    "gemma4-26b-a4b-swa": {
        "q_heads": 16,
        "kv_heads": 8,
        "head_dim": 256,
        "window": 1024,
        "tp": (1,),
    },
    "gemma4-26b-a4b-global": {
        "q_heads": 16,
        "kv_heads": 2,
        "head_dim": 512,
        "tp": (1,),
    },
}

KV_DTYPES = ["bf16", "fp8", "fp8_token"]  # fp8 = per-tensor scales


def fp8_dtype():
    return torch.float8_e4m3fnuz if get_gfx() == "gfx942" else torch.float8_e4m3fn


def unpack_pages(key_cache, value_cache, pages):
    """[n, H, page, D] K/V of ``pages`` from the vectorized 5D caches."""
    key = key_cache[pages].permute(0, 1, 3, 2, 4)
    value = value_cache[pages].permute(0, 1, 2, 4, 3)
    n, heads, page = key.shape[:3]
    return (
        key.reshape(n, heads, page, -1),
        value.reshape(n, heads, page, -1),
    )


def run_torch(
    query,
    key_cache,
    value_cache,
    key_scale,
    value_scale,
    table,
    lengths,
    query_length,
    window,
    sinks,
):
    """fp32 paged attention with MTP causality, window and sinks (reference only)."""
    batch = lengths.numel()
    heads, dim = query.shape[1:]
    kv_heads = key_cache.shape[1]
    group = heads // kv_heads
    queries = query.float().reshape(batch, query_length, kv_heads, group, dim)
    output = torch.zeros_like(queries)
    positions = torch.arange(query_length)
    for seq, length in enumerate(lengths.tolist()):
        tokens = torch.arange(length)
        pages = table[seq, : (length + key_cache.shape[3] - 1) // key_cache.shape[3]]
        key, value = unpack_pages(key_cache, value_cache, pages.long())
        key, value = key.float(), value.float()
        for cache, scale in ((key, key_scale), (value, value_scale)):
            if scale is not None:
                cache.mul_(scale[pages.long()] if scale.dim() > 1 else scale)
        # [n, H, page, D] -> [tokens, H, D]
        key = key.transpose(1, 2).reshape(-1, kv_heads, dim)[:length]
        value = value.transpose(1, 2).reshape(-1, kv_heads, dim)[:length]
        scores = torch.einsum("qhgd,khd->qhgk", queries[seq], key) * dim**-0.5
        visible = length - query_length + 1 + positions
        masked = tokens[None, :] >= visible[:, None]
        if window > 0:
            masked |= tokens[None, :] < (visible - window)[:, None]
        scores.masked_fill_(masked[:, None, None, :], float("-inf"))
        denominator = torch.logsumexp(scores, dim=-1, keepdim=True)
        if sinks is not None:
            denominator = torch.logaddexp(
                denominator, sinks.float().reshape(1, kv_heads, group, 1)
            )
        output[seq] = torch.einsum(
            "qhgk,khd->qhgd", torch.exp(scores - denominator), value
        )
    return output.reshape_as(query)


def make_caches(batch, ctx, kv_heads, dim, page, kv, table_tokens):
    """Random pages in pa_decode's vectorized 5D layouts, quantized for FP8."""
    pages_per_seq = (ctx + page - 1) // page
    table = torch.randperm(batch * pages_per_seq, dtype=torch.int32).reshape(
        batch, pages_per_seq
    )
    if table_tokens:
        # Block-table padding beyond the live pages, as a serving table has.
        width = max((table_tokens + page - 1) // page, pages_per_seq)
        table = torch.nn.functional.pad(table, (0, width - pages_per_seq))
    shape = (batch * pages_per_seq, kv_heads, page, dim)
    key = torch.empty(shape, dtype=torch.bfloat16).uniform_(-0.5, 0.5)
    value = torch.empty(shape, dtype=torch.bfloat16).uniform_(-0.5, 0.5)
    key_scale = value_scale = None
    vector = 8
    if kv != "bf16":
        quantize = pertoken_quant if kv == "fp8_token" else per_tensor_quant
        key, key_scale = quantize(key, quant_dtype=fp8_dtype())
        value, value_scale = quantize(value, quant_dtype=fp8_dtype())
        vector = 16
    pages = shape[0]
    key_cache = (
        key.reshape(pages, kv_heads, page, dim // vector, vector)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    value_cache = (
        value.reshape(pages, kv_heads, page // vector, vector, dim)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
    )
    return key_cache, value_cache, key_scale, value_scale, table


def run_pa(
    output,
    query,
    key_cache,
    value_cache,
    key_scale,
    value_scale,
    context,
    table,
    *,
    query_length,
    plan,
    exp_sums,
    max_logits,
    temporary_output,
    **options,
):
    pa_decode(
        output,
        query,
        key_cache,
        value_cache,
        context,
        table,
        query.shape[-1] ** -0.5,
        query_length,
        key_scale=key_scale,
        value_scale=value_scale,
        work_plan=plan,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        **options,
    )
    return output


@benchmark()
def test_pa_decode(model, tp, kv, page, mtp, batch, ctx, table):
    cfg = MODELS[model]
    q_heads = cfg["q_heads"] // tp
    kv_heads = max(1, cfg["kv_heads"] // tp)
    dim, window = cfg["head_dim"], cfg.get("window", 0)
    group = q_heads // kv_heads
    torch.manual_seed(0)
    query = torch.empty((batch * mtp, q_heads, dim), dtype=torch.bfloat16).uniform_(
        -0.5, 0.5
    )
    key_cache, value_cache, key_scale, value_scale, block_table = make_caches(
        batch, ctx, kv_heads, dim, page, kv, table
    )
    context = torch.full((batch,), ctx, dtype=torch.int32)
    sinks = torch.randn(q_heads, dtype=torch.float32) if cfg.get("sinks") else None
    plan = plan_pa_decode(context, kv_heads, query_length=mtp, sliding_window=window)
    scalars = (kv_heads, plan.capacity, mtp * group)
    workspace = {
        "exp_sums": torch.empty(scalars, dtype=torch.float32),
        "max_logits": torch.empty(scalars, dtype=torch.float32),
        "temporary_output": torch.empty((*scalars, dim), dtype=torch.bfloat16),
    }
    options = {
        "compute_type": torch.bfloat16 if kv == "bf16" else fp8_dtype(),
        "sinks": sinks,
        "sliding_window": window,
        "max_context_length": None if table else ctx,
    }
    output = torch.empty_like(query)
    args = (output, query, key_cache, value_cache, key_scale, value_scale)
    args += (context, block_table)
    ret = {"gfx": get_gfx(), "q_heads": q_heads, "kv_heads": kv_heads, "head_dim": dim}
    ret.update(window=window, sinks=sinks is not None)
    try:
        run_pa(*args, query_length=mtp, plan=plan, **workspace, **options)
    except (NotImplementedError, ValueError) as error:
        # Configurations pa_decode rejects on this arch (e.g. LDS limits) stay NaN.
        aiter.logger.warning("pa_decode skips %s tp%d %s: %s", model, tp, kv, error)
        return ret

    out, us = run_perftest(
        run_pa, *args, query_length=mtp, plan=plan, **workspace, **options
    )
    ref = run_torch(
        query,
        key_cache,
        value_cache,
        key_scale,
        value_scale,
        block_table,
        context,
        mtp,
        window,
        sinks,
    )
    err = checkAllclose(
        ref, out.float(), rtol=5e-3, atol=5e-3, msg=f"pa_decode {model} tp{tp} {kv}: "
    )
    visible = min(ctx, window + mtp - 1) if window else ctx
    flops = 4 * batch * mtp * q_heads * visible * dim  # QK and PV
    scale_bytes = 0 if kv != "fp8_token" else 2 * batch * visible * kv_heads * 4
    nbytes = (
        2 * batch * visible * kv_heads * dim * key_cache.element_size()
        + scale_bytes
        + 2 * query.numel() * query.element_size()
    )
    ret["flydsl us"] = us
    ret["flydsl TFLOPS"] = flops / us / 1e6
    ret["flydsl TB/s"] = nbytes / us / 1e6
    ret["flydsl err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "pa_decode benchmark unsupported on %s; skipping", get_gfx()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL pa_decode on aiter's model attention configs",
    )
    parser.add_argument(
        "-m",
        "--model",
        nargs="*",
        choices=list(MODELS),
        default=list(MODELS),
        help="attention layers to run (default: all)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        nargs="*",
        default=None,
        help="tensor-parallel sizes (default: each model's own)",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        nargs="*",
        choices=KV_DTYPES,
        default=["bf16", "fp8"],
        help="KV cache dtype; fp8 = per-tensor scales, fp8_token = per-token",
    )
    parser.add_argument(
        "-p", "--page", type=int, nargs="*", default=[16], help="KV page size"
    )
    parser.add_argument(
        "-q",
        "--mtp",
        type=int,
        nargs="*",
        default=[1, 4],
        help="query positions per sequence (1 = decode, >1 = MTP verify)",
    )
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[1, 32])
    parser.add_argument(
        "-c", "--ctx", type=int, nargs="*", default=[4096, 32768], help="context length"
    )
    parser.add_argument(
        "-t",
        "--table",
        type=int,
        nargs="*",
        default=[0],
        help="block-table capacity in tokens; 0 = live pages only, bounded call",
    )
    args = parser.parse_args()

    for kv in args.dtype:
        df = []
        for model in args.model:
            tps = [
                tp
                for tp in (args.tp or MODELS[model]["tp"])
                if MODELS[model]["q_heads"] % tp == 0
            ]
            for tp, page, mtp, batch, ctx, table in itertools.product(
                tps, args.page, args.mtp, args.batch, args.ctx, args.table
            ):
                df.append(test_pa_decode(model, tp, kv, page, mtp, batch, ctx, table))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "pa_decode %s summary (markdown):\n%s", kv, df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
