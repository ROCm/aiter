import torch
import aiter
from aiter.test_common import checkAllclose, perftest, benchmark, run_perftest
from aiter import dtypes
import argparse
import pandas as pd
import random

# torch.set_printoptions(threshold=torch.inf)


@perftest()
def run_aiter(
    kv_c,
    k_pe,
    kv_cache,
    slot_mapping,
    kv_cache_dtype: str,
    scale,
):
    aiter.concat_and_cache_mla(
        kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
    )
    return kv_cache


# @perftest()
def aiter_fused_rope_concat_and_cache_mla(
    q_nope,
    q_pe,
    kv_c,
    k_pe,  # key tensor
    kv_cache,
    q_out,
    slot_mapping,
    kv_cache_dtype,
    k_scale,
    q_scale,
    positions,
    cos_cache,
    sin_cache,
    is_neox,
    is_nope_first,
    q_out_dtype=None,
):
    aiter.fused_qk_rope_concat_and_cache_mla(
        q_nope,
        q_pe,
        kv_c,
        k_pe,
        kv_cache,
        q_out,
        slot_mapping,
        # kv_cache_dtype,
        k_scale,
        q_scale,
        positions,
        cos_cache,
        sin_cache,
        is_neox,
        is_nope_first,
        # q_out_dtype,
    )
    return kv_cache, q_out


@perftest(3)
def run_torch_fused(
    q_pe,
    k_pe,
    q_nope,
    k_nope,
    kv_cache,
    q_out,
    slot_mapping,
    kv_cache_dtype,
    k_scale,
    q_scale,
    positions,
    cos_cache,
    sin_cache,
    is_neox,
    is_nope_first,
    out_dtype,
):
    #
    q_pe_reshaped = q_pe.unsqueeze(0)
    num_tokens = k_pe.shape[0]
    qk_rope_head_dim = k_pe.shape[-1]
    num_kv_heads = k_pe.shape[1]
    k_pe_reshaped = k_pe.reshape(1, num_tokens, num_kv_heads, qk_rope_head_dim)

    cos_cache_reshaped = cos_cache.reshape(cos_cache.shape[0], 1, 1, cos_cache.shape[1])
    sin_cache_reshaped = sin_cache.reshape(sin_cache.shape[0], 1, 1, sin_cache.shape[1])
    positions = positions.unsqueeze(0)
    ## [s,b,h,d]
    q_pe_out = aiter.rope_cached_positions_fwd(
        q_pe_reshaped,  # [s,b,h,d]
        cos_cache_reshaped,  # [s,1,1,d]
        sin_cache_reshaped,  # [s,1,1,d]
        positions,  # [s,b]
        0 if is_neox else 1,
        True,
        is_nope_first,
    )
    k_pe_out = aiter.rope_cached_positions_fwd(
        k_pe_reshaped,
        cos_cache_reshaped,
        sin_cache_reshaped,
        positions,
        0 if is_neox else 1,
        True,
        is_nope_first,
    )
    q_pe = q_pe_out.squeeze(0)
    k_pe = k_pe_out.reshape(num_tokens, num_kv_heads, qk_rope_head_dim)

    num_kv_heads = kv_cache.shape[2]
    if num_kv_heads == 1:
        k_nope = k_nope.reshape(num_tokens, k_nope.shape[-1])
        k_pe = k_pe.reshape(num_tokens, k_pe.shape[-1])
        kv_cache = kv_cache.reshape(
            kv_cache.shape[0], kv_cache.shape[1], kv_cache.shape[-1]
        )
        aiter.concat_and_cache_mla(
            k_nope, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale
        )
        kv_cache = kv_cache.reshape(
            kv_cache.shape[0], kv_cache.shape[1], 1, kv_cache.shape[-1]
        )
    else:
        block_size = kv_cache.shape[1]
        num_tokens = k_nope.shape[0]
        # Vectorized version - much faster than nested for loops
        # Concatenate k_nope and k_pe along the last dimension: [num_tokens, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
        k_concat = torch.cat([k_nope, k_pe], dim=-1)

        # Compute block indices and offsets for all tokens at once
        block_indices = slot_mapping // block_size
        block_offsets = slot_mapping % block_size

        # Use advanced indexing to write all data at once
        # kv_cache[block_indices, block_offsets, :, :] = k_concat
        # Note: We need to handle each token separately due to potentially different block_idx/offset combinations
        # But we can still avoid the inner loop over heads
        for i in range(num_tokens):
            kv_cache[block_indices[i], block_offsets[i], :, :] = k_concat[i]
        ##
        if kv_cache_dtype == "fp8":
            kv_cache = (kv_cache.to(torch.float32) / k_scale.item()).to(out_dtype)
        else:
            pass
    if is_nope_first:
        kv_cache_swapped = kv_cache
    else:
        kv_cache_swapped = torch.cat(
            [kv_cache[..., k_nope.shape[-1] :], kv_cache[..., : k_nope.shape[-1]]],
            dim=-1,
        )
    if out_dtype == dtypes.fp8:
        q_nope_scale = (q_nope.to(torch.float32) / q_scale.item()).to(out_dtype)
        q_pe_scale = (q_pe.to(torch.float32) / q_scale.item()).to(out_dtype)
        if is_nope_first:
            q_out = torch.cat((q_nope_scale, q_pe_scale), dim=-1)
        else:
            q_out = torch.cat((q_pe_scale, q_nope_scale), dim=-1)
    else:
        if is_nope_first:
            q_out = torch.cat((q_nope, q_pe), dim=-1)
        else:
            q_out = torch.cat((q_pe, q_nope), dim=-1)
    return kv_cache_swapped, q_out


@perftest(3)
def run_torch_concat(
    kv_c,
    k_pe,
    kv_cache,
    slot_mapping,
    kv_cache_dtype: str,
    scale,
    dtype,
):

    block_size = kv_cache.shape[1]
    num_tokens = kv_c.shape[0]
    kv_lora_rank = kv_c.shape[-1]

    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        kv_cache[block_idx, block_offset, :kv_lora_rank] = kv_c[i]
        kv_cache[block_idx, block_offset, kv_lora_rank:] = k_pe[i]

    if kv_cache_dtype == "fp8":
        ref_kv_cache = (kv_cache.to(torch.float32) / scale.item()).to(dtype)
    else:
        ref_kv_cache = kv_cache
    return ref_kv_cache


## compare with vllm impl
# from vllm import _custom_ops as ops
# @perftest()
# def run_vllm(
#    kv_c,
#    k_pe,
#    kv_cache,
#    slot_mapping,
#    kv_cache_dtype: str,
#    scale,
# ):
#    ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale)
#    return kv_cache


@benchmark()
def test_concat_and_cache_mla(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_tokens: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    device: str,
    kv_cache_dtype: str,
) -> None:
    ret = {}
    torch.set_default_device(device)
    total_slots = num_blocks * block_size
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)
    kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(num_tokens, qk_rope_head_dim, dtype=dtype, device=device)
    entry_size = kv_lora_rank + qk_rope_head_dim
    scale = torch.tensor(0.1, dtype=torch.float32, device=device)
    cache_dtype = dtypes.fp8 if kv_cache_dtype == "fp8" else dtype
    kv_cache = torch.zeros(
        num_blocks, block_size, entry_size, dtype=cache_dtype, device=device
    )
    kv_cache, avg_us = run_aiter(
        kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
    )
    ref_temp = torch.zeros(*kv_cache.shape, dtype=dtype, device=device)
    ref_kv_cache, ref_us = run_torch_concat(
        kv_c, k_pe, ref_temp, slot_mapping, kv_cache_dtype, scale, kv_cache.dtype
    )
    # vllm_temp = torch.zeros(*kv_cache.shape, dtype=cache_dtype, device=device)
    # vllm_kv_cache, vllm_us = run_vllm(
    #    kv_c, k_pe, vllm_temp, slot_mapping, kv_cache_dtype, scale
    # )
    if kv_cache_dtype == "fp8":
        result_temp = kv_cache.to(torch.float32) * scale
        expected_temp = ref_kv_cache.to(torch.float32) * scale
        # result_temp = torch.empty_like(kv_cache, dtype=torch.float32)
        # ops.convert_fp8(result_temp, kv_cache, scale.item(), kv_dtype=kv_cache_dtype)
        # expected_vllm = torch.empty_like(vllm_kv_cache, dtype=torch.float32)
        # ops.convert_fp8(
        #    expected_vllm, vllm_kv_cache, scale.item(), kv_dtype=kv_cache_dtype
        # )
        checkAllclose(result_temp, expected_temp, atol=0.01, rtol=0.01)
    else:
        checkAllclose(kv_cache, ref_kv_cache)
    ret["aiter_us"] = avg_us
    ret["torch_us"] = ref_us
    # ret["vllm_us"] = vllm_us
    ret["aiter_bw(TB/s)"] = (
        num_tokens
        * (kv_lora_rank + qk_rope_head_dim)
        * 2
        * (torch.finfo(dtype).bits // 8)
        / (avg_us * 1e6)
    )
    return ret


def compute_cache(
    seq_len: int, freqs_dim: int, dtype: torch.dtype, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:

    cos_cache = torch.zeros(seq_len, freqs_dim)
    sin_cache = torch.zeros(seq_len, freqs_dim)

    # freq for every position
    # theta_i = 1 / (base^(2*(i//2) / dim))
    div_term = 1.0 / (base ** (torch.arange(0, freqs_dim, 1).float() / (freqs_dim)))
    positions = torch.arange(seq_len).float().unsqueeze(1)  # [seq_len, 1]

    freqs = positions * div_term.unsqueeze(0)  # [seq_len, dim//2]
    cos_cache = torch.cos(freqs).to(dtype)
    sin_cache = torch.sin(freqs).to(dtype)
    return cos_cache, sin_cache


@benchmark()
def test_fused_rope_concat_and_cache_mla(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_tokens: int,
    block_size: int,
    num_blocks: int,
    num_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    device: str,
    kv_cache_dtype: str,
    q_dtype: str,
    is_neox: bool,
):
    ret = {}
    torch.set_default_device(device)

    total_slots = num_blocks * block_size
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

    kv_c = torch.randn(
        num_tokens, num_kv_heads, kv_lora_rank, dtype=dtype, device=device
    )
    k_pe = torch.randn(
        num_tokens, num_kv_heads, qk_rope_head_dim, dtype=dtype, device=device
    )
    q_nope = torch.randn(
        num_tokens, num_heads, kv_lora_rank, dtype=dtype, device=device
    )
    q_pe = torch.randn(
        num_tokens, num_heads, qk_rope_head_dim, dtype=dtype, device=device
    )
    entry_size = kv_lora_rank + qk_rope_head_dim
    cos_cache, sin_cache = compute_cache(num_tokens, qk_rope_head_dim // 2, dtype)
    cos_cache = cos_cache.to(device)
    sin_cache = sin_cache.to(device)

    pos = torch.randint(0, num_tokens, (num_tokens,), device=device)
    scale = torch.tensor(0.5, dtype=torch.float32, device=device)
    q_scale = torch.tensor(1, dtype=torch.float32, device=device)
    cache_dtype = dtypes.fp8 if kv_cache_dtype == "fp8" else dtype
    q_out_dtype = dtypes.fp8 if q_dtype == "fp8" else dtype
    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        entry_size,
        dtype=cache_dtype,
        device=device,
    )
    q_out = torch.empty(
        (num_tokens, num_heads, qk_rope_head_dim + kv_lora_rank),
        dtype=q_out_dtype,  # cache_dtype,
        device=q_nope.device,
    )
    is_nope_first = True

    ref_q_out = torch.empty(
        (num_tokens, num_heads, qk_rope_head_dim + kv_lora_rank),
        dtype=q_out_dtype,
        device=q_nope.device,
    )
    ref_temp = torch.zeros(*kv_cache.shape, dtype=cache_dtype, device=device)
    (ref_kv_cache, ref_q_out), ref_us = run_torch_fused(
        q_pe,
        k_pe,
        q_nope,
        kv_c,
        ref_temp,
        ref_q_out,
        slot_mapping,
        kv_cache_dtype,
        scale,
        q_scale,
        pos,
        cos_cache,
        sin_cache,
        is_neox,
        is_nope_first,
        q_out_dtype,
    )
    ############################################################
    # triton test
    ############################################################
    # triton_q_out = torch.empty(
    #  (num_tokens, num_heads, qk_rope_head_dim + kv_lora_rank),
    #  dtype=q_out_dtype,
    #  device=q_nope.device,
    # )
    # from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_cat_and_cache_mla
    #
    # triton_temp = torch.zeros(
    #  (num_tokens, num_kv_heads, entry_size), dtype=cache_dtype, device=device
    # )
    # if block_size == 1 and is_nope_first and (num_heads % num_kv_heads == 0):
    #  (triton_q_out, _, _, _), triton_us = (
    #      run_perftest(
    #          fused_qk_rope_cat_and_cache_mla,
    #          q_nope,
    #          q_pe,
    #          kv_c,
    #          k_pe,
    #          triton_temp,
    #          slot_mapping,
    #          pos,
    #          cos_cache,
    #          sin_cache,
    #          scale,
    #          is_neox,
    #          0,
    #          True if kv_cache_dtype == "fp8" else False,
    #          triton_q_out,
    #      )
    #  )
    # else:
    #  (triton_q_out, decode_q_pe_out, k_pe_out, triton_temp), triton_us = (
    #      triton_q_out,
    #      None,
    #      None,
    #      triton_temp,
    #  ), None
    # triton_temp = triton_temp.reshape(
    #  num_tokens // block_size, block_size, num_kv_heads, entry_size
    # )
    #############################################################
    if num_kv_heads == 1:
        kv_c = kv_c.squeeze(1)
        k_pe = k_pe.squeeze(1)
        kv_cache = kv_cache.squeeze(1)
    (kv_cache, q_out), avg_us = run_perftest(
        aiter_fused_rope_concat_and_cache_mla,
        q_nope,
        q_pe,
        kv_c,
        k_pe,
        kv_cache,
        q_out,
        slot_mapping,
        kv_cache_dtype,
        scale,
        q_scale,
        pos,
        cos_cache,
        sin_cache,
        is_neox,
        is_nope_first,
        q_out_dtype,
    )
    # err_triton_kv = 0
    # err_triton_q_out = 0
    kv_cache = kv_cache.reshape(
        num_tokens // block_size, block_size, num_kv_heads, entry_size
    )
    if kv_cache_dtype == "fp8" and q_dtype == "fp8":
        kv_result_temp = kv_cache.to(torch.float32)
        kv_expected_temp = ref_kv_cache.to(torch.float32)
        q_result_tmp = q_out.to(torch.float32) * q_scale
        q_expected_tmp = ref_q_out.to(torch.float32) * q_scale
        err_kv = checkAllclose(kv_result_temp, kv_expected_temp, atol=0.01, rtol=0.01)
        err_q_out = checkAllclose(q_result_tmp, q_expected_tmp, atol=0.01, rtol=0.01)
        ## compare with qscale=1.0
        # if block_size == 1 and is_nope_first and (num_heads % num_kv_heads == 0):
        #  err_triton_kv = checkAllclose(
        #      triton_temp.to(torch.float32),
        #      kv_expected_temp,
        #      atol=0.01,
        #      rtol=0.01,
        #      msg="fp8 kv result compared with triton",
        #  )
        #  err_triton_q_out = checkAllclose(
        #      triton_q_out.to(torch.float32) * q_scale,
        #      q_expected_tmp,
        #      msg="fp8 qout result compared with triton",
        #  )
    elif kv_cache_dtype == "fp8" and q_dtype == "auto":
        kv_result_temp = kv_cache.to(torch.float32)
        kv_expected_temp = ref_kv_cache.to(torch.float32)
        err_kv = checkAllclose(
            kv_result_temp,
            kv_expected_temp,
            atol=0.01,
            rtol=0.01,
            msg="fp8 kv result compared with ref",
        )
        err_q_out = checkAllclose(
            q_out, ref_q_out, msg="bf16 qout result compared with ref"
        )
        # if block_size == 1 and is_nope_first and (num_heads % num_kv_heads == 0):
        #  err_triton_q_out = checkAllclose(
        #      triton_q_out, ref_q_out, msg="bf16 triton qout result compared with ref"
        #  )
        #  err_triton_kv = checkAllclose(
        #      triton_temp.to(torch.float32),
        #      kv_expected_temp,
        #      msg="fp8 triton kv result compared with ref",
        #  )
    else:
        err_kv = checkAllclose(
            kv_cache, ref_kv_cache, msg="bf16 kv result compared with ref"
        )
        err_q_out = checkAllclose(
            q_out, ref_q_out, msg="bf16 qout result compared with ref"
        )

        # if block_size == 1 and is_nope_first and (num_heads % num_kv_heads == 0):
        #  err_triton_q_out = checkAllclose(
        #      triton_q_out, ref_q_out, msg="bf16 triton qout result compared with ref"
        #  )
        #  err_triton_kv = checkAllclose(
        #      triton_temp, ref_kv_cache, msg="bf16 triton kv result compared with ref"
        #  )
    # ret["triton_us"] = triton_us
    # ret['triton_kv_err'] = err_triton_kv
    # ret['triton_q_err'] = err_triton_q_out
    ret["fused_qk_us"] = avg_us
    # ret["unfused_us"] = ref_us
    ret["hip_kv_err"] = err_kv
    ret["hip_q_err"] = err_q_out
    ####
    ret["aiter_bw(TB/s)"] = (
        num_tokens
        * (
            kv_lora_rank * num_kv_heads
            + qk_rope_head_dim * num_kv_heads
            + num_heads * kv_lora_rank
            + num_heads * qk_rope_head_dim
        )
        * (torch.finfo(dtype).bits // 8)
        + num_tokens
        * (kv_lora_rank + qk_rope_head_dim)
        * num_kv_heads
        * (torch.finfo(cache_dtype).bits // 8)
        + num_tokens
        * num_heads
        * (kv_lora_rank + qk_rope_head_dim)
        * (torch.finfo(q_out_dtype).bits // 8)
    ) / (avg_us * 1e6)
    return ret


def rms_norm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float):
    input_dtype = x.dtype
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(input_dtype)
    return weight * x


def run_torch_fused_norm_rope_group_quant(
    q_nope,
    q_pe,
    kv_c,
    k_pe,
    k_weight,
    kv_cache,
    q_out,
    slot_mapping,
    k_scale,
    q_scale,
    positions,
    cos_cache,
    sin_cache,
    eps,
    group_size,
    is_neox,
    is_nope_first,
    out_dtype,
    kv_cache_dtype,
):
    q_pe_reshaped = q_pe.unsqueeze(0)
    num_tokens = k_pe.shape[0]
    qk_rope_head_dim = k_pe.shape[-1]
    num_kv_heads = k_pe.shape[1]
    k_pe_reshaped = k_pe.reshape(1, num_tokens, num_kv_heads, qk_rope_head_dim)

    cos_cache_reshaped = cos_cache.reshape(cos_cache.shape[0], 1, 1, cos_cache.shape[1])
    sin_cache_reshaped = sin_cache.reshape(sin_cache.shape[0], 1, 1, sin_cache.shape[1])
    positions = positions.unsqueeze(0)
    ## [s,b,h,d]
    q_pe_out = aiter.rope_cached_positions_fwd(
        q_pe_reshaped,  # [s,b,h,d]
        cos_cache_reshaped,  # [s,1,1,d]
        sin_cache_reshaped,  # [s,1,1,d]
        positions,  # [s,b]
        0 if is_neox else 1,
        True,
        is_nope_first,
    )
    k_pe_out = aiter.rope_cached_positions_fwd(
        k_pe_reshaped,
        cos_cache_reshaped,
        sin_cache_reshaped,
        positions,
        0 if is_neox else 1,
        True,
        is_nope_first,
    )
    q_pe = q_pe_out.squeeze(0)
    k_pe = k_pe_out.reshape(num_tokens, num_kv_heads, qk_rope_head_dim)
    ## RMs
    k_nope = rms_norm_forward(kv_c, k_weight, eps)

    # Store original k_nope for group quantization (before writing to fp8 cache)
    k_nope_original = k_nope.clone() if kv_cache_dtype == "fp8" else None
    k_nope_original_shape = k_nope.shape

    num_kv_heads = kv_cache.shape[2]

    block_size = kv_cache.shape[1]
    num_tokens = k_nope.shape[0]
    # Vectorized version - much faster than nested for loops
    # Concatenate k_nope and k_pe along the last dimension: [num_tokens, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
    k_concat = torch.cat([k_nope, k_pe], dim=-1)

    # Compute block indices and offsets for all tokens at once
    block_indices = slot_mapping // block_size
    block_offsets = slot_mapping % block_size

    # Use advanced indexing to write all data at once
    # kv_cache[block_indices, block_offsets, :, :] = k_concat
    # Note: We need to handle each token separately due to potentially different block_idx/offset combinations
    # But we can still avoid the inner loop over heads
    for i in range(num_tokens):
        kv_cache[block_indices[i], block_offsets[i], :, :] = k_concat[i]
    ##
    if kv_cache_dtype == "fp8":
        # kv_cache = (kv_cache.to(torch.float32) / k_scale.item()).to(out_dtype)
        ## k_nope group quant
        # Use original k_nope (before fp8 conversion) for quantization
        k_nope_for_quant = k_nope_original.reshape(-1, group_size).contiguous()
        print(f"k_nope shape: {k_nope_for_quant.shape}")
        k_nope_quant, k_scale = aiter.ops.quant.per_token_quant_hip(
            k_nope_for_quant,
            None,
            dtypes.fp8,
            None,
            1,
        )
        # k_nope_quant = k_nope_for_quant
        # k_scale = None
        print(f"k_nope_quant shape: {k_nope_quant.shape}", flush=True)
        k_nope_quant = k_nope_quant.reshape(k_nope_original_shape)
        # Write quantized k_nope back to kv_cache
        original_kv_cache_shape = kv_cache.shape
        kv_cache_flat = kv_cache.reshape(-1, kv_cache.shape[-1])
        k_nope_quant_flat = k_nope_quant.reshape(-1, k_nope_quant.shape[-1])
        kv_cache_flat[: k_nope_quant_flat.shape[0], : k_nope_quant_flat.shape[-1]] = (
            k_nope_quant_flat
        )
        kv_cache = kv_cache_flat.reshape(original_kv_cache_shape)
    else:
        pass
    if is_nope_first:
        kv_cache_swapped = kv_cache
    else:
        kv_cache_swapped = torch.cat(
            [
                kv_cache[..., k_nope_quant.shape[-1] :],
                kv_cache[..., : k_nope_quant.shape[-1]],
            ],
            dim=-1,
        )
    if out_dtype == dtypes.fp8:
        q_nope_scale = (q_nope.to(torch.float32) / q_scale.item()).to(out_dtype)
        q_pe_scale = (q_pe.to(torch.float32) / q_scale.item()).to(out_dtype)
        if is_nope_first:
            q_out = torch.cat((q_nope_scale, q_pe_scale), dim=-1)
        else:
            q_out = torch.cat((q_pe_scale, q_nope_scale), dim=-1)
    else:
        if is_nope_first:
            q_out = torch.cat((q_nope, q_pe), dim=-1)
        else:
            q_out = torch.cat((q_pe, q_nope), dim=-1)
    return kv_cache_swapped, q_out


def run_aiter_fused_norm_rope_group_quant(
    q_nope,
    q_pe,
    kv_c,
    k_pe,
    k_weight,
    kv_cache,
    q_out,
    slot_mapping,
    k_scale,
    q_scale,
    positions,
    cos_cache,
    sin_cache,
    eps,
    group_size,
    is_neox,
    is_nope_first,
):

    aiter.fused_qk_norm_rope_group_quantconcat_and_cache_mla(
        q_nope,
        q_pe,
        kv_c,
        k_pe,
        k_weight,
        kv_cache,
        q_out,
        slot_mapping,
        k_scale,
        q_scale,
        positions,
        cos_cache,
        sin_cache,
        eps,
        group_size,
        is_neox,
        is_nope_first,
    )
    return kv_cache, q_out


@benchmark()
def test_fused_qk_norm_rope_group_quant_concat_and_cache_mla(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_tokens: int,
    block_size: int,
    num_blocks: int,
    num_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    device: str,
    kv_cache_dtype: str,
    q_dtype: str,
    is_neox: bool,
    group_size: int,
):
    ret = {}
    group_size = 64
    torch.set_default_device(device)

    total_slots = num_blocks * block_size
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

    kv_c = torch.randn(
        num_tokens, num_kv_heads, kv_lora_rank, dtype=dtype, device=device
    )
    k_pe = torch.randn(
        num_tokens, num_kv_heads, qk_rope_head_dim, dtype=dtype, device=device
    )
    q_nope = torch.randn(
        num_tokens, num_heads, kv_lora_rank, dtype=dtype, device=device
    )
    q_pe = torch.randn(
        num_tokens, num_heads, qk_rope_head_dim, dtype=dtype, device=device
    )
    k_weight = torch.randn(kv_lora_rank, dtype=dtype, device=device)
    entry_size = kv_lora_rank + qk_rope_head_dim
    cos_cache, sin_cache = compute_cache(num_tokens, qk_rope_head_dim // 2, dtype)
    cos_cache = cos_cache.to(device)
    sin_cache = sin_cache.to(device)

    pos = torch.randint(0, num_tokens, (num_tokens,), device=device)
    k_scale = torch.empty(
        num_blocks,
        block_size,
        (kv_lora_rank + qk_rope_head_dim) // group_size,
        dtype=dtypes.fp8,
        device=device,
    )
    q_scale = torch.tensor(1, dtype=torch.float32, device=device)
    cache_dtype = dtypes.fp8 if kv_cache_dtype == "fp8" else dtype
    q_out_dtype = dtypes.fp8 if q_dtype == "fp8" else dtype
    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        entry_size,
        dtype=cache_dtype,
        device=device,
    )
    q_out = torch.empty(
        (num_tokens, num_heads, qk_rope_head_dim + kv_lora_rank),
        dtype=q_out_dtype,  # cache_dtype,
        device=q_nope.device,
    )
    is_nope_first = True

    ref_q_out = torch.empty(
        (num_tokens, num_heads, qk_rope_head_dim + kv_lora_rank),
        dtype=q_out_dtype,
        device=q_nope.device,
    )
    ref_temp = torch.zeros(*kv_cache.shape, dtype=cache_dtype, device=device)
    ref_kv_cache, ref_q_out = run_torch_fused_norm_rope_group_quant(
        q_nope,
        q_pe,
        kv_c,
        k_pe,
        k_weight,
        kv_cache,
        q_out,
        slot_mapping,
        k_scale,
        q_scale,
        pos,
        cos_cache,
        sin_cache,
        1e-6,
        group_size,
        is_neox,
        is_nope_first,
        q_out_dtype,
        kv_cache_dtype,
    )

    (kv_cache, q_out), avg_us = run_perftest(
        run_aiter_fused_norm_rope_group_quant,
        q_nope,
        q_pe,
        kv_c,
        k_pe,
        k_weight,
        kv_cache,
        q_out,
        slot_mapping,
        k_scale,
        q_scale,
        pos,
        cos_cache,
        sin_cache,
        1e-6,
        group_size,
        is_neox,
        is_nope_first,
    )
    # err_triton_kv = 0
    # err_triton_q_out = 0
    kv_cache = kv_cache.reshape(
        num_tokens // block_size, block_size, num_kv_heads, entry_size
    )
    if kv_cache_dtype == "fp8" and q_dtype == "fp8":
        kv_result_temp = kv_cache.to(torch.float32)
        kv_expected_temp = ref_kv_cache.to(torch.float32)
        q_result_tmp = q_out.to(torch.float32) * q_scale
        q_expected_tmp = ref_q_out.to(torch.float32) * q_scale
        err_kv = checkAllclose(kv_result_temp, kv_expected_temp, atol=0.01, rtol=0.01)
        err_q_out = checkAllclose(q_result_tmp, q_expected_tmp, atol=0.01, rtol=0.01)

    elif kv_cache_dtype == "fp8" and q_dtype == "auto":
        kv_result_temp = kv_cache.to(torch.float32)
        kv_expected_temp = ref_kv_cache.to(torch.float32)
        print(
            f"kv_result_temp: {kv_result_temp}, kv_expected_temp shape: {kv_expected_temp}"
        )
        err_kv = checkAllclose(
            kv_result_temp,
            kv_expected_temp,
            atol=0.01,
            rtol=0.01,
            msg="fp8 kv result compared with ref",
        )
        err_q_out = checkAllclose(
            q_out, ref_q_out, msg="bf16 qout result compared with ref"
        )

    else:
        err_kv = checkAllclose(
            kv_cache, ref_kv_cache, msg="bf16 kv result compared with ref"
        )

        err_q_out = checkAllclose(
            q_out, ref_q_out, msg="bf16 qout result compared with ref"
        )

    ret["fused_qk_us"] = avg_us
    # ret["unfused_us"] = ref_us
    ret["hip_kv_err"] = err_kv
    ret["hip_q_err"] = err_q_out
    ####
    ret["aiter_bw(TB/s)"] = (
        num_tokens
        * (
            kv_lora_rank * num_kv_heads
            + qk_rope_head_dim * num_kv_heads
            + num_heads * kv_lora_rank
            + num_heads * qk_rope_head_dim
        )
        * (torch.finfo(dtype).bits // 8)
        + num_tokens
        * (kv_lora_rank + qk_rope_head_dim)
        * num_kv_heads
        * (torch.finfo(cache_dtype).bits // 8)
        + num_tokens
        * num_heads
        * (kv_lora_rank + qk_rope_head_dim)
        * (torch.finfo(q_out_dtype).bits // 8)
    ) / (avg_us * 1e6)
    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-k",
    "--kv_lora_rank",
    type=int,
    default=512,
    help="""kv lora rank.
    e.g.: -k 512""",
)
parser.add_argument(
    "-qr",
    "--qk_rope_head_dim",
    type=int,
    default=64,
    help="""qk rope head dim.
    e.g.: -qr 64""",
)

parser.add_argument(
    "-blk",
    "--block_size",
    type=int,
    default=1,
    help="""Block size.
    e.g.: -blk 1""",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["bf16"]],
    default="bf16",
    metavar="{bf16}",
    help="""Data type of input.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-kvd",
    "--kv_dtype",
    type=str,
    choices=["auto", "fp8"],
    nargs="*",
    default=["auto", "fp8"],
    help="""Data type of KV cache.
    e.g.: -kvd auto""",
)
parser.add_argument(
    "-dev",
    "--device",
    type=str,
    default="cuda",
    help="""Device.
    e.g.: -dev cuda""",
)
parser.add_argument(
    "-t",
    "--token",
    type=int,
    nargs="*",
    default=[4, 128, 256, 512, 1024, 2048],  # , 4096 , 8192, 16384,
    help="""token nums.
    e.g.: -t 128""",
)
parser.add_argument(
    "-hd",
    "--head",
    type=int,
    nargs="*",
    default=[2, 8],
    help="""num heads.
    e.g.: -hd 1""",
)
parser.add_argument(
    "-nkh",
    "--num_kv_heads",
    type=int,
    nargs="*",
    default=[1, 2],
    help="""num kv heads.
    e.g.: -nkh 1""",
)
parser.add_argument(
    "-qd",
    "--q_dtype",
    type=str,
    choices=["auto", "fp8"],
    nargs="*",
    default=["auto", "fp8"],
    help="""Data type of Q out.
    e.g.: -qd auto""",
)
parser.add_argument(
    "-n",
    "--is_neox",
    type=dtypes.str2bool,
    nargs="*",
    default=[True, False],
    help="""true: GPT-NeoX style rotary embedding or false: GPT-J style rotary embedding.
    e.g.: --is_neox false
          or --is_neox true""",
)

parser.add_argument(
    "-c",
    "--case",
    type=str,
    choices=["normal", "fused_qk", "fused_norm_group_quant"],
    nargs="*",
    default=["normal", "fused_qk", "fused_norm_group_quant"],
    help="""tests concat and cache or fused_qk.
    e.g.: -c normal""",
)

args = parser.parse_args()

if "normal" in args.case:
    df = []
    for num_token in args.token:
        num_blocks = num_token // args.block_size
        for kv_cache_dtype in args.kv_dtype:
            ret = test_concat_and_cache_mla(
                args.kv_lora_rank,
                args.qk_rope_head_dim,
                num_token,
                args.block_size,
                num_blocks,
                args.dtype,
                args.device,
                kv_cache_dtype,
            )
            df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("concat_and_cache_mla summary (markdown):\n%s", df_md)


if "fused_qk" in args.case:
    df = []
    for num_token in args.token:
        num_blocks = num_token // args.block_size
        for num_heads in args.head:
            for num_kv_heads in args.num_kv_heads:
                for kv_cache_dtype in args.kv_dtype:
                    for is_neox in args.is_neox:
                        for q_dtype in args.q_dtype:
                            if q_dtype == "fp8" and kv_cache_dtype != "fp8":
                                continue
                            if num_kv_heads > num_heads:
                                continue
                            ret = test_fused_rope_concat_and_cache_mla(
                                args.kv_lora_rank,
                                args.qk_rope_head_dim,
                                num_token,
                                args.block_size,
                                num_blocks,
                                num_heads,
                                num_kv_heads,
                                args.dtype,
                                args.device,
                                kv_cache_dtype,
                                q_dtype,
                                is_neox,
                            )
                            df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("fused_rope_concat_and_cache_mla summary (markdown):\n%s", df_md)

if "fused_norm_group_quant" in args.case:
    df = []
    for num_token in args.token:
        num_blocks = num_token // args.block_size
        for num_heads in args.head:
            for num_kv_heads in args.num_kv_heads:
                for kv_cache_dtype in args.kv_dtype:
                    for is_neox in args.is_neox:
                        for q_dtype in args.q_dtype:
                            if q_dtype == "fp8" and kv_cache_dtype != "fp8":
                                continue
                            ret = test_fused_qk_norm_rope_group_quant_concat_and_cache_mla(
                                kv_lora_rank=args.kv_lora_rank,
                                qk_rope_head_dim=args.qk_rope_head_dim,
                                num_tokens=num_token,
                                block_size=args.block_size,
                                num_blocks=num_blocks,
                                num_heads=num_heads,
                                num_kv_heads=num_kv_heads,
                                dtype=args.dtype,
                                device=args.device,
                                kv_cache_dtype=kv_cache_dtype,
                                q_dtype=q_dtype,
                                is_neox=is_neox,
                                group_size=64,
                            )
                            df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info(
        "fused_qk_norm_rope_group_quant_concat_and_cache_mla summary (markdown):\n%s",
        df_md,
    )
