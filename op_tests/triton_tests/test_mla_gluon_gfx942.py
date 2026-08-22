# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the gfx942 Gluon MLA decode path.

The QLEN inputs are random tensors. They validate kernel math and graph replay,
not speculative-decoding model quality.
"""

import math
from dataclasses import dataclass

import pytest
import torch

try:
    from triton.experimental import gluon as _gluon  # noqa: F401
except ImportError:
    pytest.skip("triton.experimental.gluon is unavailable", allow_module_level=True)

if not torch.cuda.is_available():
    pytest.skip("a ROCm GPU is required", allow_module_level=True)

from aiter.ops.triton.utils._triton import arch_info

if arch_info.get_arch() != "gfx942":
    pytest.skip("the gfx942 MLA kernel requires gfx942", allow_module_level=True)

from aiter.ops.triton._gluon_kernels.gfx942.attention.mla import (
    _mla_gluon_gfx942,
)
from aiter.ops.triton.attention.mla import mla_gluon_gfx942

BLOCK_SIZE = 768
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
NHEAD = 12
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)


@dataclass
class MlaCase:
    q_nope: torch.Tensor
    q_pe: torch.Tensor
    kv_buffer: torch.Tensor
    output: torch.Tensor
    kv_indices: torch.Tensor
    kv_indptr: torch.Tensor
    seq_lens: list[int]


def _physical_indices(seq_lens: list[int], num_physical_blocks: int) -> list[int]:
    logical_blocks = sum(math.ceil(seq_len / BLOCK_SIZE) for seq_len in seq_lens)
    assert logical_blocks <= num_physical_blocks

    # Consecutive logical blocks deliberately map to non-contiguous physical
    # blocks while retaining a one-to-one allocation.
    physical_blocks = list(range(1, num_physical_blocks, 2))
    physical_blocks += list(range(0, num_physical_blocks, 2))
    physical_blocks = physical_blocks[:logical_blocks]
    indices = []
    logical_block = 0
    for seq_len in seq_lens:
        remaining = seq_len
        while remaining:
            physical_block = physical_blocks[logical_block]
            block_tokens = min(remaining, BLOCK_SIZE)
            block_start = physical_block * BLOCK_SIZE
            indices.extend(range(block_start, block_start + block_tokens))
            remaining -= block_tokens
            logical_block += 1
    return indices


def _make_case(
    batch_size: int,
    qlen: int,
    seq_lens: list[int],
    seed: int,
) -> MlaCase:
    assert len(seq_lens) == batch_size
    logical_blocks = sum(math.ceil(seq_len / BLOCK_SIZE) for seq_len in seq_lens)
    num_physical_blocks = 2 * logical_blocks + 3

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    kv_buffer = (
        torch.randn(
            (num_physical_blocks * BLOCK_SIZE, HEAD_DIM_CKV + HEAD_DIM_KPE),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.2
    )
    q_shape = (
        (batch_size, NHEAD, HEAD_DIM_CKV)
        if qlen == 1
        else (batch_size, qlen, NHEAD, HEAD_DIM_CKV)
    )
    q_pe_shape = (
        (batch_size, NHEAD, HEAD_DIM_KPE)
        if qlen == 1
        else (batch_size, qlen, NHEAD, HEAD_DIM_KPE)
    )
    q_nope = (
        torch.randn(q_shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        * 0.2
    )
    q_pe = (
        torch.randn(
            q_pe_shape,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.2
    )
    output = torch.empty_like(q_nope)

    indices = _physical_indices(seq_lens, num_physical_blocks)
    indptr = [0]
    for seq_len in seq_lens:
        indptr.append(indptr[-1] + seq_len)

    return MlaCase(
        q_nope=q_nope,
        q_pe=q_pe,
        kv_buffer=kv_buffer,
        output=output,
        kv_indices=torch.tensor(indices, device="cuda", dtype=torch.int32),
        kv_indptr=torch.tensor(indptr, device="cuda", dtype=torch.int32),
        seq_lens=seq_lens,
    )


def _reference(case: MlaCase) -> torch.Tensor:
    q_nope = case.q_nope.unsqueeze(1) if case.q_nope.dim() == 3 else case.q_nope
    q_pe = case.q_pe.unsqueeze(1) if case.q_pe.dim() == 3 else case.q_pe
    batch_size, qlen, _, _ = q_nope.shape
    output = torch.empty_like(q_nope)

    for batch_idx in range(batch_size):
        start = int(case.kv_indptr[batch_idx].item())
        end = int(case.kv_indptr[batch_idx + 1].item())
        token_indices = case.kv_indices[start:end].long()
        kv = case.kv_buffer.index_select(0, token_indices)
        k_nope = kv[:, :HEAD_DIM_CKV].float()
        k_pe = kv[:, HEAD_DIM_CKV:].float()
        value = k_nope

        for q_pos in range(qlen):
            causal_end = case.seq_lens[batch_idx] - qlen + q_pos + 1
            scores = torch.einsum(
                "hd,td->ht", q_nope[batch_idx, q_pos].float(), k_nope[:causal_end]
            )
            scores += torch.einsum(
                "hd,td->ht", q_pe[batch_idx, q_pos].float(), k_pe[:causal_end]
            )
            probs = torch.softmax(scores * SM_SCALE, dim=-1)
            output[batch_idx, q_pos] = (probs @ value[:causal_end]).to(output.dtype)

    return output.squeeze(1) if case.q_nope.dim() == 3 else output


def _assert_matches_reference(case: MlaCase) -> None:
    assert torch.isfinite(case.output).all().item()
    torch.testing.assert_close(
        case.output,
        _reference(case),
        atol=1.5e-2,
        rtol=1.0e-2,
    )


@pytest.mark.parametrize(
    ("batch_size", "qlen", "seq_lens"),
    [
        pytest.param(1, 1, [25], id="qlen1-short-sequence"),
        pytest.param(4, 1, [145, 163, 181, 199], id="qlen1"),
        # The graph launch uses 24 non-empty splits. For the earliest query,
        # the final two splits are entirely beyond its causal boundary.
        pytest.param(
            1,
            4,
            [25],
            id="qlen4-fully-masked-future-splits",
        ),
        pytest.param(1, 4, [801], id="qlen4-b1"),
        pytest.param(4, 4, [129, 145, 161, 177], id="qlen4-b4"),
        pytest.param(
            8,
            4,
            [129, 137, 145, 153, 161, 169, 177, 185],
            id="qlen4-b8-fused-qh",
        ),
        pytest.param(
            16,
            4,
            [
                129,
                133,
                137,
                141,
                145,
                149,
                153,
                157,
                161,
                165,
                169,
                173,
                177,
                181,
                185,
                189,
            ],
            id="qlen4-b16-fused-qh",
        ),
    ],
)
def test_combined_physical_cache_ragged(
    batch_size: int,
    qlen: int,
    seq_lens: list[int],
) -> None:
    case = _make_case(batch_size, qlen, seq_lens, seed=17 + batch_size + qlen)

    mla_gluon_gfx942(
        case.q_nope,
        case.q_pe,
        case.kv_buffer,
        case.output,
        case.kv_indices,
        case.kv_indptr,
        SM_SCALE,
    )

    _assert_matches_reference(case)


def test_low_level_split_reduction_forced_64bit_load() -> None:
    case = _make_case(2, 1, [97, 121], seed=41)

    # Force the pointer-based variant on a small allocation so the test does
    # not need to reserve more than 2 GiB of device memory.
    _mla_gluon_gfx942(
        case.q_nope,
        case.q_pe,
        case.kv_buffer[:, :HEAD_DIM_CKV],
        case.kv_buffer[:, HEAD_DIM_CKV:],
        case.kv_indices,
        case.kv_indptr,
        case.output,
        SM_SCALE,
        num_kv_splits=5,
        block_n=32,
        use_2d_view=False,
        within_2gb_override=False,
    )

    _assert_matches_reference(case)


@pytest.mark.parametrize("invalid_q_pe", ["none", "rank", "leading_shape"])
def test_q_pe_validation(invalid_q_pe: str) -> None:
    case = _make_case(1, 4, [25], seed=47)
    if invalid_q_pe == "none":
        q_pe = None
        match = "must not be None"
    elif invalid_q_pe == "rank":
        q_pe = case.q_pe[0]
        match = "same rank"
    else:
        q_pe = case.q_pe[:, :, :-1, :]
        match = "leading dimensions"

    with pytest.raises(AssertionError, match=match):
        _mla_gluon_gfx942(
            case.q_nope,
            q_pe,
            case.kv_buffer[:, :HEAD_DIM_CKV],
            case.kv_buffer[:, HEAD_DIM_CKV:],
            case.kv_indices,
            case.kv_indptr,
            case.output,
            SM_SCALE,
            use_2d_view=False,
        )


def test_cuda_graph_replay_with_metadata_mutation() -> None:
    """Random QLEN data checks graph replay mechanics, not model quality."""
    case = _make_case(4, 4, [101, 113, 127, 139], seed=53)
    alternate = _make_case(4, 4, [139, 127, 113, 101], seed=59)

    # Compile both stages before capture. Capture then owns the split
    # workspaces allocated by the wrapper.
    mla_gluon_gfx942(
        case.q_nope,
        case.q_pe,
        case.kv_buffer,
        case.output,
        case.kv_indices,
        case.kv_indptr,
        SM_SCALE,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mla_gluon_gfx942(
            case.q_nope,
            case.q_pe,
            case.kv_buffer,
            case.output,
            case.kv_indices,
            case.kv_indptr,
            SM_SCALE,
        )

    case.q_nope.copy_(alternate.q_nope)
    case.q_pe.copy_(alternate.q_pe)
    case.kv_indices.copy_(alternate.kv_indices)
    case.kv_indptr.copy_(alternate.kv_indptr)
    case.seq_lens = alternate.seq_lens

    graph.replay()
    torch.cuda.synchronize()

    _assert_matches_reference(case)
