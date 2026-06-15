# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reproducers for FP8 rope_norm_store_kv pos/slot handoff bugs.

These tests are intentionally opt-in because one path is expected to trigger a
GPU memory fault.  Run the non-crashing semantic check with:

    AITER_RUN_ROPE_FP8_REPRO=1 pytest -q \
      op_tests/triton_tests/fusions/test_rope_norm_store_kv_fp8_pos_slot_repro.py

Run the crash reproducer directly:

    python op_tests/triton_tests/fusions/test_rope_norm_store_kv_fp8_pos_slot_repro.py \
      --main-slot-oob
"""

import argparse
import os

import pytest
import torch

from aiter.ops.triton._triton_kernels.fusions.rope_norm_store_kv_fp8 import (
    _rope_norm_store_kv_fp8_compute_pos_slot_kernel,
    _rope_norm_store_kv_fp8_kernel,
)
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device is required")


def _launch_compute_pos_slot(
    *,
    q_index: torch.Tensor,
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    slot_indices: torch.Tensor,
    req_ids: torch.Tensor,
    local_idx: torch.Tensor,
    kvcache_indices: torch.Tensor,
    block_size: int,
) -> None:
    _rope_norm_store_kv_fp8_compute_pos_slot_kernel[(seq_lens.numel(),)](
        q_index_ptr=q_index,
        num_seqlen_per_req_ptr=seq_lens,
        kvcache_indices_ptr=kvcache_indices,
        positions_ptr=positions,
        slot_indices_ptr=slot_indices,
        req_ids_ptr=req_ids,
        local_idx_ptr=local_idx,
        stride_kvi_r=kvcache_indices.stride(0),
        stride_kvi_b=kvcache_indices.stride(1),
        BLOCK_R=32,
        BLOCK_SIZE=block_size,
    )


@pytest.mark.skipif(
    os.environ.get("AITER_RUN_ROPE_FP8_REPRO") != "1",
    reason="opt-in repro: set AITER_RUN_ROPE_FP8_REPRO=1",
)
def test_zero_seq_request_must_not_overwrite_canary_rows():
    """A zero-length request used to be skipped by the helper kernel.

    The PR changed the helper to enter the loop even when seq_len == 0 and to
    write sentinel values with mask=row_local<num_rows_req.  If q_index carries
    a stale non-empty range for such a request, the helper now overwrites rows
    that used to remain untouched.
    """

    _require_cuda()
    device = "cuda"
    block_size = 16

    # req0 is a normal decode-like row, req1 is inactive (seq_len == 0) but
    # still has a stale q_index range.  This is the metadata shape that exposes
    # the changed helper semantics without intentionally crashing the process.
    q_index = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([16, 0], dtype=torch.int32, device=device)
    kvcache_indices = torch.zeros((2, 1), dtype=torch.int32, device=device)

    positions = torch.full((2,), 777, dtype=torch.int32, device=device)
    slot_indices = torch.full((2,), -123, dtype=torch.int64, device=device)
    req_ids = torch.full((2,), 555, dtype=torch.int32, device=device)
    local_idx = torch.full((2,), 444, dtype=torch.int32, device=device)

    _launch_compute_pos_slot(
        q_index=q_index,
        seq_lens=seq_lens,
        positions=positions,
        slot_indices=slot_indices,
        req_ids=req_ids,
        local_idx=local_idx,
        kvcache_indices=kvcache_indices,
        block_size=block_size,
    )
    torch.cuda.synchronize()

    got = {
        "position": int(positions[1].item()),
        "slot": int(slot_indices[1].item()),
        "req_id": int(req_ids[1].item()),
        "local_idx": int(local_idx[1].item()),
    }
    assert got == {
        "position": 777,
        "slot": -123,
        "req_id": 555,
        "local_idx": 444,
    }, (
        "zero-seq request overwrote helper outputs; these tensors are consumed "
        f"directly by _rope_norm_store_kv_fp8_kernel. got={got}"
    )


def _run_main_slot_load_oob() -> None:
    """Trigger the same main-kernel site shown in service_vllm.log.

    This directly feeds an undersized slot_indices buffer into
    _rope_norm_store_kv_fp8_kernel so the fault is expected at/near:

        slots = tl.load(slot_indices_ptr + t_offs, mask=t_mask, other=-1)

    It is intentionally destructive to the current process/GPU context.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is required")

    device = "cuda"
    fp8_dtype = get_fp8_e4m3_dtype()

    num_rows = 16
    num_q_heads = 8
    num_kv_heads = 1
    qk_head_dim = 128
    v_head_dim = 128
    block_size = 16
    cache_x = 16
    num_blocks = 8
    hidden = (
        num_q_heads * qk_head_dim
        + num_kv_heads * qk_head_dim
        + num_kv_heads * v_head_dim
    )

    qkv = torch.randn(num_rows, hidden, dtype=torch.bfloat16, device=device)
    cos_sin = torch.randn(64, qk_head_dim, dtype=torch.float32, device=device)
    positions = torch.zeros(num_rows, dtype=torch.int32, device=device)

    # Deliberately too small.  t_mask is true for t_offs=[0..15], so line 141
    # in the main kernel reads past this allocation.
    slot_indices = torch.zeros(1, dtype=torch.int64, device=device)
    req_ids = torch.zeros(num_rows, dtype=torch.int32, device=device)
    local_idx = torch.zeros(num_rows, dtype=torch.int32, device=device)

    q_norm_weight = torch.ones(qk_head_dim, dtype=torch.float32, device=device)
    k_norm_weight = torch.ones(qk_head_dim, dtype=torch.float32, device=device)
    hadamard = torch.eye(qk_head_dim, dtype=torch.bfloat16, device=device)
    q_scale_inv = torch.ones(1, dtype=torch.float32, device=device)
    k_scale = torch.zeros(
        (num_blocks, 1, num_kv_heads, block_size),
        dtype=torch.float32,
        device=device,
    )
    v_scale = torch.ones(num_kv_heads, dtype=torch.float32, device=device)
    q_scale_out = torch.empty(
        (num_rows, num_q_heads), dtype=torch.float32, device=device
    )
    out_q = torch.empty(
        (num_rows, num_q_heads, qk_head_dim), dtype=fp8_dtype, device=device
    )
    key_cache = torch.empty(
        (num_blocks, num_kv_heads, qk_head_dim // cache_x, block_size, cache_x),
        dtype=fp8_dtype,
        device=device,
    )
    value_cache = torch.empty(
        (num_blocks, num_kv_heads, v_head_dim, block_size),
        dtype=fp8_dtype,
        device=device,
    )

    print("Launching _rope_norm_store_kv_fp8_kernel with undersized slot_indices")
    print("Expected fault site: rope_norm_store_kv_fp8.py line 141 slots tl.load")

    grid = (1, num_q_heads)
    _rope_norm_store_kv_fp8_kernel[grid](
        qkv_ptr=qkv,
        cos_sin_ptr=cos_sin,
        positions_ptr=positions,
        slot_indices_ptr=slot_indices,
        req_ids_ptr=req_ids,
        local_idx_ptr=local_idx,
        q_norm_weight_ptr=q_norm_weight,
        k_norm_weight_ptr=k_norm_weight,
        hadamard_ptr=hadamard,
        q_scale_inv_ptr=q_scale_inv,
        k_scale_ptr=k_scale,
        v_scale_ptr=v_scale,
        q_scale_out_ptr=q_scale_out,
        out_q_ptr=out_q,
        out_k_ptr=key_cache,
        out_v_ptr=value_cache,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        eps=1e-5,
        num_rows=num_rows,
        total_num_kv_cache_tokens=num_blocks * block_size,
        fp8_max=torch.finfo(fp8_dtype).max,
        stride_qkv_t=qkv.stride(0),
        stride_qkv_d=qkv.stride(1),
        stride_cos_t=cos_sin.stride(0),
        stride_cos_d=cos_sin.stride(1),
        stride_out_q_t=out_q.stride(0),
        stride_out_q_h=out_q.stride(1),
        stride_out_q_d=out_q.stride(2),
        stride_out_k_t=0,
        stride_out_k_h=0,
        stride_out_k_d=0,
        stride_out_v_t=0,
        stride_out_v_h=0,
        stride_out_v_d=0,
        stride_kc_b=key_cache.stride(0),
        stride_kc_h=key_cache.stride(1),
        stride_kc_g=key_cache.stride(2),
        stride_kc_t=key_cache.stride(3),
        stride_kc_x=key_cache.stride(4),
        stride_vc_b=value_cache.stride(0),
        stride_vc_h=value_cache.stride(1),
        stride_vc_d=value_cache.stride(2),
        stride_vc_t=value_cache.stride(3),
        stride_ks_b=k_scale.stride(0),
        stride_ks_r=k_scale.stride(1),
        stride_ks_h=k_scale.stride(2),
        stride_ks_l=k_scale.stride(3),
        stride_qs_0=q_scale_out.stride(0),
        stride_qs_1=q_scale_out.stride(1),
        stride_qs_2=0,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        QK_HEAD_DIM=qk_head_dim,
        QK_HEAD_DIM_HALF=qk_head_dim // 2,
        V_HEAD_DIM=v_head_dim,
        V_HEAD_DIM_PAD=v_head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_T=num_rows,
        QK_NORM_POLICY=2,
        APPLY_Q_NORM=True,
        APPLY_K_NORM=True,
        Q_QUANT_DYNAMIC=True,
        K_QUANT_DYNAMIC=True,
        V_QUANT_PERHEAD=True,
        APPLY_HADAMARD=True,
        IS_PREFILL=False,
        WRITE_K_TO_CACHE=True,
        WRITE_V_TO_CACHE=True,
        K_SCALE_L=block_size,
        K_CACHE_X=cache_x,
    )
    torch.cuda.synchronize()


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-slot-oob",
        action="store_true",
        help="run destructive main-kernel slot_indices OOB reproducer",
    )
    args = parser.parse_args()
    if args.main_slot_oob:
        _run_main_slot_load_oob()
    else:
        parser.error("pass --main-slot-oob to run the destructive reproducer")


if __name__ == "__main__":
    _main()
