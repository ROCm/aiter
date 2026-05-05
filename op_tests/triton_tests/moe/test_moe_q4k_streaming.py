# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sergey Subbotin <ssubbotin@gmail.com>
#
# Correctness tests for the scattered-pointer Q4_K_M MoE matvec kernel.
# Reference: pure-numpy q4k_pack_reference, byte-exact to llama.cpp's
# dequantize_row_q4_K.

import numpy as np
import pytest
import torch

from aiter.ops.triton.moe.moe_op_q4k_streaming import (
    fused_moe_q4k_streaming,
    QK_K,
)
from op_tests.triton_tests.moe.q4k_pack_reference import (
    build_random_expert,
    moe_matvec_scattered_ref,
)


def _build_setup(n_tokens, n_used, n_unique, n_dim_in, n_dim_out, seed=0):
    rng = np.random.default_rng(seed)
    expert_bufs = [
        build_random_expert(rng, n_dim_in, n_dim_out) for _ in range(n_unique)
    ]
    expert_tensors = [
        torch.frombuffer(bytearray(b), dtype=torch.uint8).cuda()
        for b in expert_bufs
    ]
    expert_ptrs = torch.tensor(
        [t.data_ptr() for t in expert_tensors],
        dtype=torch.uint64,
        device="cuda",
    )
    remap_np = rng.integers(0, n_unique, size=(n_tokens, n_used), dtype=np.int32)
    remap = torch.from_numpy(remap_np).cuda()
    a_np = rng.standard_normal((n_tokens, n_dim_in)).astype(np.float32)
    a = torch.from_numpy(a_np).cuda()
    c = torch.zeros(
        (n_tokens, n_used, n_dim_out), dtype=torch.float32, device="cuda"
    )
    return expert_bufs, expert_tensors, expert_ptrs, remap, a, a_np, remap_np, c


@pytest.mark.parametrize(
    "n_tokens,n_used,n_unique,n_dim_in,n_dim_out",
    [
        # Smallest case: 1 K-block, 4 output rows, single token/slot/expert
        (1, 1, 1, 256, 4),
        # Multiple K blocks
        (1, 1, 1, 512, 8),
        # Multiple slots, multiple unique experts
        (1, 2, 2, 512, 16),
        # Multiple tokens, deduplicated unique-expert set
        (4, 2, 3, 512, 32),
        # Larger row count
        (2, 2, 2, 1024, 64),
        # Odd row counts
        (1, 1, 1, 512, 17),
        (3, 2, 2, 768, 19),
        # Production-shaped Qwen3 MoE expert (gate/up: 1408 rows, K=2048, K=8)
        (1, 8, 8, 2048, 1408),
    ],
)
def test_moe_q4k_streaming_matches_reference(
    n_tokens, n_used, n_unique, n_dim_in, n_dim_out
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    (
        expert_bufs,
        _expert_tensors,
        expert_ptrs,
        remap,
        a,
        a_np,
        remap_np,
        c,
    ) = _build_setup(n_tokens, n_used, n_unique, n_dim_in, n_dim_out, seed=42)

    fused_moe_q4k_streaming(a, expert_ptrs, remap, c, n_dim_out=n_dim_out)

    expected = moe_matvec_scattered_ref(expert_bufs, remap_np, a_np, n_dim_out)
    # fp32 reduction noise grows with K. The reference accumulates in
    # double-precision-equivalent numpy float32 with row-major order;
    # the kernel's order differs slightly, so tolerance scales with K.
    rtol = max(1e-3, 5e-6 * n_dim_in)
    atol = max(1e-3, 5e-6 * n_dim_in)
    np.testing.assert_allclose(c.cpu().numpy(), expected, rtol=rtol, atol=atol)
