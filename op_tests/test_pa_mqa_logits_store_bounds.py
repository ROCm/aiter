# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression coverage for OutLogits_buffer upper-bound store masks."""

import ast
from pathlib import Path

import pytest

KERNEL_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "aiter"
    / "ops"
    / "triton"
    / "gluon"
    / "pa_mqa_logits.py"
)
KERNEL_NAMES = (
    "_gluon_deepgemm_fp8_paged_mqa_logits",
    "_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle",
    "_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle_varctx",
)


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next(
        (keyword.value for keyword in call.keywords if keyword.arg == name), None
    )


def _references_name(node: ast.AST | None, name: str) -> bool:
    return node is not None and any(
        isinstance(child, ast.Name) and child.id == name for child in ast.walk(node)
    )


def _is_out_logits_store(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "buffer_store"
        and _references_name(_keyword(node, "ptr"), "OutLogits_buffer")
    )


def _has_max_model_len_upper_bound(
    mask: ast.AST | None, offsets: ast.AST | None
) -> bool:
    if mask is None or offsets is None:
        return False
    offsets_ast = ast.dump(offsets, include_attributes=False)
    return any(
        isinstance(node, ast.Compare)
        and ast.dump(node.left, include_attributes=False) == offsets_ast
        and any(
            isinstance(operator, ast.Lt)
            and isinstance(comparator, ast.Name)
            and comparator.id == "max_model_len"
            for operator, comparator in zip(node.ops, node.comparators)
        )
        for node in ast.walk(mask)
    )


@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
def test_out_logits_stores_guard_max_model_len_boundary(kernel_name: str) -> None:
    tree = ast.parse(KERNEL_SOURCE.read_text(encoding="utf-8"))
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == kernel_name
    )
    stores = [node for node in ast.walk(kernel) if _is_out_logits_store(node)]

    assert stores, f"{kernel_name} has no OutLogits_buffer stores"
    unbounded_lines = [
        store.lineno
        for store in stores
        if not _has_max_model_len_upper_bound(
            _keyword(store, "mask"), _keyword(store, "offsets")
        )
    ]
    assert not unbounded_lines, (
        f"{kernel_name} has OutLogits_buffer stores without "
        f"'offset < max_model_len' masks at lines {unbounded_lines}"
    )


@pytest.mark.parametrize("use_varctx", [False, True], ids=["fixed-context", "varctx"])
def test_preshuffle_does_not_write_past_max_model_len(use_varctx: bool) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")

    from aiter.ops.triton.pa_mqa_logits import (
        deepgemm_fp8_paged_mqa_logits,
    )

    from aiter import dtypes

    context_len = 2300
    block_size = 64
    chunk_k = 256
    guard_size = chunk_k
    hidden_dim = 128
    heads = 128
    sentinel = 12345.0
    num_blocks = (context_len + block_size - 1) // block_size

    q_bits = torch.randint(
        1, 64, (1, 1, heads, hidden_dim), dtype=torch.uint8, device="cuda"
    )
    kv_bits = torch.randint(
        1,
        64,
        (num_blocks, block_size, 1, hidden_dim + 4),
        dtype=torch.uint8,
        device="cuda",
    )
    q_fp8 = q_bits.view(dtypes.fp8)
    kv_cache = kv_bits.view(dtypes.fp8)
    weights = torch.ones((1, heads), dtype=torch.float32, device="cuda")
    context_lens = torch.tensor([context_len], dtype=torch.int32, device="cuda")
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device="cuda")[None, :]
    out = torch.full(
        (1, context_len + guard_size),
        sentinel,
        dtype=torch.float32,
        device="cuda",
    )
    varctx_schedule = (
        torch.ones((1,), dtype=torch.int32, device="cuda") if use_varctx else None
    )

    deepgemm_fp8_paged_mqa_logits(
        q_fp8,
        kv_cache,
        weights,
        out,
        context_lens,
        block_tables,
        context_len,
        Preshuffle=True,
        KVBlockSize=block_size,
        ChunkK=chunk_k,
        TotalCuCount=8,
        WavePerEU=2,
        VarCtxSchedule=varctx_schedule,
    )
    torch.cuda.synchronize()

    assert torch.all(out[:, context_len:] == sentinel), (
        f"{'varctx' if use_varctx else 'fixed-context'} kernel wrote into "
        "the guard region past max_model_len"
    )
