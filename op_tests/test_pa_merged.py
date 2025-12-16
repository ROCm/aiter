# SPDX-License-Identifier: MIT
# Consolidated PA tests (repro + stress) added during this session.

import os
import random
from typing import Tuple

import pytest
import torch

from aiter import dtypes
from aiter.ops.attention import pa_fwd_asm, paged_attention_v1_core
from aiter.test_common import checkAllclose, perftest


def _make_alias_view_5d(base_5d: torch.Tensor, num_blocks: int) -> torch.Tensor:
    """
    Create a view with first dimension = num_blocks without allocating num_blocks storage.
    We do this by setting stride(0)=0 so every "block" aliases block 0.
    """
    assert base_5d.dim() == 5 and base_5d.size(0) == 1
    _, s1, s2, s3, s4 = base_5d.stride()
    return base_5d.as_strided(
        size=(num_blocks, base_5d.size(1), base_5d.size(2), base_5d.size(3), base_5d.size(4)),
        stride=(0, s1, s2, s3, s4),
    )


def _asm_v_shuffle(value_cache_4d: torch.Tensor) -> torch.Tensor:
    """Convert V from [B, H, D, BS] -> ASM 5D [B, H, BS/x, D, x]."""
    x = 16 // value_cache_4d.element_size()
    num_blocks, num_kv_heads, head_size, block_size = value_cache_4d.shape
    assert block_size % x == 0
    v5 = value_cache_4d.view(num_blocks, num_kv_heads, head_size, block_size // x, x)
    return v5.permute(0, 1, 3, 2, 4).contiguous()


def _kv_cache_factory_asm_kv(
    *,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create KV in ASM-compatible layouts:
    - K: 5D ASM layout [B, H, D/x, BS, x]
    - V: 5D ASM layout [B, H, BS/x, D, x] (via shuffle from 4D)
    """
    torch.manual_seed(seed)
    random.seed(seed)

    x = 16 // torch.tensor([], dtype=dtype).element_size()
    assert head_size % x == 0

    k = torch.empty(
        (num_blocks, num_kv_heads, head_size // x, block_size, x),
        device=device,
        dtype=dtype,
    )
    k.uniform_(-1, 1)

    v4 = torch.empty(
        (num_blocks, num_kv_heads, head_size, block_size),
        device=device,
        dtype=dtype,
    )
    v4.uniform_(-1, 1)
    v5 = _asm_v_shuffle(v4)
    return k, v5


def _make_workspace_buffer(
    *,
    num_seqs: int,
    num_heads: int,
    head_size: int,
    max_seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    partition_size: int = 256,
) -> torch.Tensor:
    # Mirrors workspace sizing used by `paged_attention_v1` wrapper.
    max_num_partitions = (max_seq_len + partition_size - 1) // partition_size
    nbytes_per_elem = torch.finfo(dtype).bits // 8
    nbytes = (
        (num_seqs * num_heads * max_num_partitions * head_size) * nbytes_per_elem
        + 2 * (num_seqs * num_heads * max_num_partitions) * 4
    )
    return torch.empty((nbytes,), dtype=torch.uint8, device=device)


def _random_block_tables(
    *,
    num_seqs: int,
    max_seq_len: int,
    block_size: int,
    num_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    bt = torch.empty((num_seqs, max_num_blocks_per_seq), dtype=torch.int32, device="cpu")
    for i in range(num_seqs):
        bt[i].random_(0, num_blocks)
    return bt.to(device=device)


@perftest(num_iters=200, num_warmup=2, num_rotate_args=1)
def _perf_v1(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache_asm5d: torch.Tensor,
    workspace: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    out = torch.empty_like(query)
    paged_attention_v1_core(
        out,
        workspace,
        query,
        key_cache,
        value_cache_asm5d,
        float(1.0 / (query.size(2) ** 0.5)),
        block_tables,
        None,  # cu_query_lens
        context_lens,
        int(max_seq_len),
        None,  # alibi_slopes
        "auto",
        "HND",
        0.0,
        None,
        None,
    )
    return out


@perftest(num_iters=200, num_warmup=2, num_rotate_args=1)
def _perf_asm(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache_asm5d: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
) -> torch.Tensor:
    return pa_fwd_asm(
        query.contiguous(),
        key_cache,
        value_cache_asm5d,
        block_tables,
        context_lens,
        int(block_tables.stride(0)),
        max_qlen=1,
        K_QScale=None,
        V_QScale=None,
        out_=None,
        qo_indptr=None,
        high_precision=1,
        kernelName=None,
    )


@pytest.mark.repro
def test_pa_repro_enginecore_shapes_compare_v1_vs_asm() -> None:
    """
    Repro using your logged shapes, then compare HIP v1 vs ASM outputs.
    Opt-in:
      AITER_RUN_REPRO_SHAPES=1 pytest -q op_tests/test_pa_merged.py -k repro -s
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device not available")
    if os.getenv("AITER_RUN_REPRO_SHAPES", "0") != "1":
        pytest.skip("Set AITER_RUN_REPRO_SHAPES=1 to enable this repro test")

    device = torch.device("cuda:0")

    # Exact shapes from your log
    Q = torch.empty((1, 32, 128), device=device, dtype=torch.bfloat16).uniform_(-1, 1)
    K_base = torch.empty((1, 4, 16, 16, 8), device=device, dtype=torch.bfloat16).uniform_(-1, 1)
    V_base = torch.empty((1, 4, 2, 128, 8), device=device, dtype=torch.bfloat16).uniform_(-1, 1)
    K = _make_alias_view_5d(K_base, 134921)
    V = _make_alias_view_5d(V_base, 134921)

    max_seq_len = 17
    block_tables = torch.zeros((1, 16384), device=device, dtype=torch.int32)
    assert block_tables.stride(0) == 16384
    context_lens = torch.tensor([17], device=device, dtype=torch.int32)

    workspace = torch.empty((8448,), device=device, dtype=torch.uint8)

    # 1) correctness compare
    out_v1 = torch.empty_like(Q)
    paged_attention_v1_core(
        out_v1,
        workspace,
        Q,
        K,
        V,
        float(1.0 / (128**0.5)),
        block_tables,
        None,
        context_lens,
        int(max_seq_len),
        None,
        "auto",
        "HND",
        0.0,
        None,
        None,
    )
    out_asm = pa_fwd_asm(
        Q.contiguous(),
        K,
        V,
        block_tables,
        context_lens,
        16384,
        max_qlen=1,
        K_QScale=None,
        V_QScale=None,
        out_=None,
        qo_indptr=None,
        high_precision=1,
        kernelName=None,
    )
    checkAllclose(out_v1, out_asm, msg="repro shapes: v1_core vs pa_fwd_asm")

    # 2) perf (200 iters each), env-gated so repro correctness can run fast if desired
    if os.getenv("AITER_REPRO_PERF", "1") == "1":
        _, v1_us = _perf_v1(Q, K, V, workspace, block_tables, context_lens, max_seq_len)
        _, asm_us = _perf_asm(Q, K, V, block_tables, context_lens)
        print(
            f"[perf repro] HIP(paged_attention_v1_core) avg_us/iter={v1_us:.2f} | "
            f"ASM(pa_fwd_asm) avg_us/iter={asm_us:.2f} | "
            f"winner={'ASM' if asm_us < v1_us else 'HIP' if v1_us < asm_us else 'tie'}"
        )
        assert v1_us > 0 and asm_us > 0


@pytest.mark.stress
def test_pa_stress_asm_layout_compare_v1_vs_asm() -> None:
    """
    Randomized stress: compare HIP v1 core vs ASM kernel on ASM-shaped KV-cache layouts.
    Opt-in:
      AITER_RUN_STRESS=1 pytest -q op_tests/test_pa_merged.py -k stress
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device not available")
    if os.getenv("AITER_RUN_STRESS", "0") != "1":
        pytest.skip("Set AITER_RUN_STRESS=1 to enable stress tests")

    device = torch.device("cuda:0")
    head_size = 128
    block_size = 16
    dtype = dtypes.bf16 if torch.cuda.is_bf16_supported() else torch.float16

    num_heads = 32
    num_kv_heads = 4
    assert num_heads % num_kv_heads == 0

    trials = int(os.getenv("AITER_STRESS_TRIALS", "25"))
    max_seq_len_choices = [64, 256, 1024, 2048, 4096]
    num_seqs_choices = [1, 2, 4, 8]

    for t in range(trials):
        seed = 1337 + t
        max_seq_len = random.choice(max_seq_len_choices)
        num_seqs = random.choice(num_seqs_choices)

        max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        num_blocks = num_seqs * max_num_blocks_per_seq
        if num_blocks > 4096:
            num_blocks = 4096

        query = torch.empty((num_seqs, num_heads, head_size), device=device, dtype=dtype).uniform_(
            -1, 1
        )
        context_lens = torch.randint(
            low=1,
            high=max_seq_len + 1,
            size=(num_seqs,),
            device=device,
            dtype=torch.int32,
        )
        block_tables = _random_block_tables(
            num_seqs=num_seqs,
            max_seq_len=max_seq_len,
            block_size=block_size,
            num_blocks=num_blocks,
            device=device,
        )
        key_cache, value_cache_asm5d = _kv_cache_factory_asm_kv(
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            device=device,
            seed=seed,
        )
        workspace = _make_workspace_buffer(
            num_seqs=num_seqs,
            num_heads=num_heads,
            head_size=head_size,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=device,
        )

        out_v1 = torch.empty_like(query)
        paged_attention_v1_core(
            out_v1,
            workspace,
            query,
            key_cache,
            value_cache_asm5d,
            float(1.0 / (head_size**0.5)),
            block_tables,
            None,
            context_lens,
            int(max_seq_len),
            None,
            "auto",
            "HND",
            0.0,
            None,
            None,
        )
        out_asm = pa_fwd_asm(
            query.contiguous(),
            key_cache,
            value_cache_asm5d,
            block_tables,
            context_lens,
            int(block_tables.stride(0)),
            max_qlen=1,
            K_QScale=None,
            V_QScale=None,
            out_=None,
            qo_indptr=None,
            high_precision=1,
            kernelName=None,
        )
        checkAllclose(out_v1, out_asm, msg=f"trial {t}: v1_core vs pa_fwd_asm")

