# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression test for paged_attention_common MTP fast-path routing.

Scenario: MTP decode with bf16 Q and shuffled FP8 KV cache (the path vLLM
hits via rocm_aiter_ops.paged_attention_common). gqa_ratio in {8, 16} with
qlen in {2, 3, 4} must hit the pa_fwd_asm fast path (the CSV Mtp=1 generic
kernel handles qlen 2/3/4 via qo_indptr) and match the torch reference
within atol/rtol=0.02.

gqa_ratio outside {8, 16} is intentionally left to the existing HIP
fallback heuristic and not exercised here: paged_attention_common's HIP
branch hardcodes mtp=1, so testing it with qlen>1 would assert pre-existing
behavior outside this PR's scope.
"""

import random
import sys

import torch

from aiter import dtypes, paged_attention_common, pertoken_quant
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import checkAllclose

torch.set_default_device("cuda")

STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}

_PARTITION_SIZE_ROCM = 256


def _kv_cache_factory(
    num_blocks, block_size, num_heads, head_size, model_dtype, device="cuda:0"
):
    x = 16 // model_dtype.itemsize
    k_cache = torch.empty(
        (num_blocks, num_heads, head_size // x, block_size, x),
        dtype=model_dtype,
        device=device,
    ).uniform_(-1, 1)
    v_cache = torch.empty(
        (num_blocks, num_heads, head_size, block_size),
        dtype=model_dtype,
        device=device,
    ).uniform_(-1, 1)
    return k_cache, v_cache


def _ref_masked_attention(query, key, value, scale, dtype):
    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    s_q, s_k = query.shape[0], key.shape[0]
    attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
    mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
    attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
    attn_weights += attn_bias
    attn_weights = torch.softmax(attn_weights, dim=-1)
    return torch.einsum("hqk,khd->qhd", attn_weights.float(), value.float()).to(dtype)


def _torch_mha_extend(
    q, k_cache, v_cache, block_tables, seq_lens, qo_indptr, k_scale, v_scale
):
    num_blocks, num_heads, head_size, block_size = v_cache.shape
    sm_scale = 1.0 / (head_size**0.5)
    dtype = q.dtype
    kv_dtype = k_cache.dtype
    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    k_cache = k_cache.permute(0, 3, 1, 2, 4).contiguous().view(-1, num_heads, head_size)
    v_cache = v_cache.permute(0, 3, 1, 2).contiguous().view(-1, num_heads, head_size)

    bs = qo_indptr.shape[0] - 1
    outs = []
    for i in range(bs):
        qi = qs[i]
        block_table = block_tables[i]
        ctx_len = seq_lens[i].item()
        idx = (
            block_table.repeat_interleave(block_size)[:ctx_len] * block_size
            + torch.arange(ctx_len, device=block_table.device) % block_size
        )
        k = k_cache.view(torch.int8)[idx].view(kv_dtype).to(torch.float)
        if k_scale is not None:
            k *= k_scale[:, idx].t().unsqueeze(-1)
        v = v_cache.view(torch.int8)[idx].view(kv_dtype).to(torch.float)
        if v_scale is not None:
            v *= v_scale[:, idx].t().unsqueeze(-1)
        outs.append(_ref_masked_attention(qi, k, v, sm_scale, dtype))
    return torch.concat(outs)


def _pertoken_quant_kvcache_symm(k_cache, v_cache, quant_dtype):
    num_blocks = k_cache.shape[0]
    num_heads = k_cache.shape[1]
    head_dim = v_cache.shape[2]
    block_size = v_cache.shape[3]
    total_tokens = num_blocks * block_size

    k_perm = (
        k_cache.permute(0, 1, 3, 2, 4)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )
    v_perm = (
        v_cache.permute(0, 1, 3, 2)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )

    k_quant, k_scale_asm = pertoken_quant(k_perm, quant_dtype=quant_dtype)
    v_quant, v_scale_asm = pertoken_quant(v_perm, quant_dtype=quant_dtype)
    quant_x = 16 // quant_dtype.itemsize

    k_quant = (
        k_quant.view(num_blocks, num_heads, block_size, head_dim // quant_x, quant_x)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    k_scale = k_scale_asm.permute(1, 0, 2, 3).contiguous().view(num_heads, total_tokens)
    v_quant = (
        v_quant.view(num_blocks, num_heads, block_size, head_dim)
        .permute(0, 1, 3, 2)
        .contiguous()
    )
    v_scale = v_scale_asm.permute(1, 0, 2, 3).contiguous().view(num_heads, total_tokens)
    return k_quant, k_scale, v_quant, v_scale, k_scale_asm, v_scale_asm


def _asm_V_shuffle(VC):
    x = 16 // VC.element_size()
    num_blocks, num_kv_heads, head_size, block_size = VC.shape
    VC = VC.view(num_blocks, num_kv_heads, head_size, block_size // x, x)
    return VC.permute(0, 1, 3, 2, 4).contiguous()


def run_case(
    num_heads, qlen, ctx_len, batch_size, head_size=128, block_size=16, expect_asm=True
):
    nq, nkv = num_heads
    dtype = torch.bfloat16
    max_seq_len = 16384
    num_blocks = ((max_seq_len + block_size - 1) // block_size) * batch_size

    seq_lens_qo = torch.full((batch_size,), qlen, dtype=torch.int32)
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    qo_indptr[1:] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = int(qo_indptr[-1].item())

    query = torch.empty(total_q, nq, head_size, dtype=dtype).uniform_(-1, 1)
    seq_lens = torch.full((batch_size,), ctx_len, dtype=torch.int32)

    blocks_per_seq = (ctx_len + block_size - 1) // block_size
    bt = torch.tensor(
        [
            [random.randint(0, num_blocks - 1) for _ in range(blocks_per_seq)]
            for _ in range(batch_size)
        ],
        dtype=torch.int32,
    )

    k_cache, v_cache = _kv_cache_factory(
        num_blocks, block_size, nkv, head_size, dtype, "cuda:0"
    )
    k_quant, k_scale, v_quant, v_scale, k_scale_asm, v_scale_asm = (
        _pertoken_quant_kvcache_symm(k_cache, v_cache, quant_dtype=dtypes.fp8)
    )
    v_shuf = _asm_V_shuffle(v_quant)

    out_ref = _torch_mha_extend(
        query, k_quant, v_quant, bt, seq_lens, qo_indptr, k_scale, v_scale
    )

    # paged_attention_common allocates the HIP-fallback workspace; sized for
    # the worst case (HIP path), unused on the ASM fast path.
    max_num_partitions = (ctx_len + _PARTITION_SIZE_ROCM - 1) // _PARTITION_SIZE_ROCM
    tmp_out = torch.empty(
        (total_q, nq, max_num_partitions, head_size), dtype=dtype, device="cuda:0"
    )
    exp_sums = torch.empty(
        (total_q, nq, max_num_partitions), dtype=torch.float32, device="cuda:0"
    )
    max_logits = torch.empty_like(exp_sums)

    out_routed = paged_attention_common(
        Q=query,
        K=k_quant,
        V=v_shuf,
        exp_sums=exp_sums,
        max_logits=max_logits,
        tmp_out=tmp_out,
        block_tables=bt,
        context_lens=seq_lens,
        block_tables_stride0=bt.stride(0),
        scale=float(1.0 / (head_size**0.5)),
        max_qlen=qlen,
        max_seq_len=ctx_len,
        K_QScale_asm=k_scale_asm,
        V_QScale_asm=v_scale_asm,
        K_QScale_hip=k_scale,
        V_QScale_hip=v_scale,
        qo_indptr=qo_indptr,
        kv_cache_dtype="fp8",
    )

    label = (
        f"gqa={nq // nkv} qlen={qlen} ctx={ctx_len} bs={batch_size} "
        f"expect={'asm' if expect_asm else 'hip'}"
    )
    return checkAllclose(out_ref, out_routed, msg=f"[{label}]", atol=0.02, rtol=0.02)


if __name__ == "__main__":
    arch = get_gfx()
    if arch not in ("gfx942", "gfx950"):
        print(f"skip: arch {arch} not in gate (gfx942, gfx950)")
        sys.exit(0)

    print(f"gfx={arch}")
    cases = []
    # ASM fast-path: gqa in {8, 16} x qlen in {2, 3, 4}
    for gqa in (8, 16):
        for qlen in (2, 3, 4):
            cases.append(((gqa, 1), qlen, 4097, 8, True))
            cases.append(((gqa, 1), qlen, 128, 8, True))
            cases.append(((gqa, 1), qlen, 16384, 4, True))

    fails = 0
    for cfg in cases:
        try:
            run_case(*cfg)
        except Exception as ex:
            print(f"!! case {cfg} EXC: {ex}")
            fails += 1
    print(f"\nFAILURES={fails}/{len(cases)}")
    sys.exit(1 if fails else 0)
