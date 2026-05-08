import torch
import aiter
from aiter.test_common import checkAllclose, perftest, benchmark
from aiter import dtypes
import argparse
import pandas as pd
import random


def compute_cache(
    seq_len: int, freqs_dim: int, dtype: torch.dtype, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:

    cos_cache = torch.zeros(seq_len, freqs_dim)
    sin_cache = torch.zeros(seq_len, freqs_dim)

    div_term = 1.0 / (base ** (torch.arange(0, freqs_dim, 1).float() / (freqs_dim)))
    positions = torch.arange(seq_len).float().unsqueeze(1)

    freqs = positions * div_term.unsqueeze(0)
    cos_cache = torch.cos(freqs).to(dtype)
    sin_cache = torch.sin(freqs).to(dtype)
    return cos_cache, sin_cache


def run_torch_fused_norm_rope_group_quant(
    q,
    kv,
    k_weight,
    kv_cache,
    q_out,
    slot_mapping,
    positions,
    cos_cache,
    sin_cache,
    eps,
    is_neox,
    is_nope_first,
    kv_cache_dtype,
):
    num_tokens = kv.shape[0]
    head_dim = kv.shape[-1]
    num_kv_heads = kv.shape[1] if kv.dim() == 3 else 1
    qk_rope_head_dim = 64
    nope_dim = head_dim - qk_rope_head_dim

    if is_nope_first:
        q_nope = q[..., :nope_dim]
        q_pe_input = q[..., nope_dim:]
    else:
        q_pe_input = q[..., :qk_rope_head_dim]
        q_nope = q[..., qk_rope_head_dim:]

    q_rms_scale = torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + eps)
    q_nope_normed = (q_nope.float() * q_rms_scale).to(q.dtype)
    q_pe_normed = (q_pe_input.float() * q_rms_scale).to(q.dtype)

    if is_nope_first:
        kv_nope = kv[..., :nope_dim]
        kv_pe = kv[..., nope_dim:]
    else:
        kv_pe = kv[..., :qk_rope_head_dim]
        kv_nope = kv[..., qk_rope_head_dim:]

    kv_f32 = kv.float()
    kv_rms_scale = torch.rsqrt(kv_f32.pow(2).mean(-1, keepdim=True) + eps)

    if is_nope_first:
        k_weight_nope = k_weight[:nope_dim]
        k_weight_pe = k_weight[nope_dim:]
    else:
        k_weight_pe = k_weight[:qk_rope_head_dim]
        k_weight_nope = k_weight[qk_rope_head_dim:]

    k_nope_normed = (kv_nope.float() * kv_rms_scale * k_weight_nope.float()).to(
        kv.dtype
    )
    k_pe_normed = (kv_pe.float() * kv_rms_scale * k_weight_pe.float()).to(kv.dtype)

    q_pe_reshaped = q_pe_normed.unsqueeze(0)
    k_pe_reshaped = k_pe_normed.reshape(1, num_tokens, num_kv_heads, qk_rope_head_dim)
    cos_cache_reshaped = cos_cache.reshape(cos_cache.shape[0], 1, 1, cos_cache.shape[1])
    sin_cache_reshaped = sin_cache.reshape(sin_cache.shape[0], 1, 1, sin_cache.shape[1])
    positions = positions.unsqueeze(0)

    q_pe_out = aiter.rope_cached_positions_fwd(
        q_pe_reshaped,
        cos_cache_reshaped,
        sin_cache_reshaped,
        positions,
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
    q_pe_roped = q_pe_out.squeeze(0)
    k_pe_roped = k_pe_out.reshape(num_tokens, num_kv_heads, qk_rope_head_dim)

    if kv_cache_dtype == "fp8":
        k_nope_f32 = kv_nope.float() * kv_rms_scale * k_weight_nope.float()
    else:
        k_nope_f32 = None
    num_kv_heads = kv_cache.shape[2] if kv_cache.dim() == 4 else 1

    block_size = kv_cache.shape[1]

    block_indices = slot_mapping // block_size
    block_offsets = slot_mapping % block_size

    for i in range(num_tokens):
        kv_cache[block_indices[i], block_offsets[i], :, :nope_dim] = k_nope_normed[i]

    if kv_cache_dtype == "fp8":
        group_size = 64
        num_tiles = nope_dim // group_size
        fp8_max = torch.finfo(dtypes.fp8).max
        inv_fp8_max = 1.0 / fp8_max

        k_tiled = k_nope_f32.reshape(num_tokens, num_kv_heads, num_tiles, group_size)
        k_group_max = k_tiled.abs().amax(dim=-1)
        k_scales_f32 = k_group_max * inv_fp8_max
        k_u32 = k_scales_f32.view(torch.int32)
        k_exponents = ((k_u32 >> 23) & 0xFF).to(torch.int32)
        k_has_mantissa = (k_u32 & 0x7FFFFF) != 0
        k_exponents = k_exponents + k_has_mantissa.to(torch.int32)
        k_e8m0_u32 = (k_exponents << 23).view(torch.float32)
        k_inv_scale = k_e8m0_u32.unsqueeze(-1).expand_as(k_tiled)
        k_quantized = (
            (k_tiled / k_inv_scale)
            .to(dtypes.fp8)
            .reshape(num_tokens, num_kv_heads, nope_dim)
        )

        for i in range(num_tokens):
            bi, bo = block_indices[i], block_offsets[i]
            kv_cache[bi, bo, :, :nope_dim] = k_quantized[i]
            kv_cache.view(torch.uint8)[bi, bo, :, nope_dim : nope_dim + num_tiles] = (
                k_exponents[i].to(torch.uint8)
            )

    if is_nope_first:
        q_out = torch.cat((q_nope_normed, q_pe_roped), dim=-1)
    else:
        q_out = torch.cat((q_pe_roped, q_nope_normed), dim=-1)
    return kv_cache, k_pe_roped, q_out


@perftest()
def run_aiter_fused_norm_rope_group_quant(
    q,
    kv,
    k_pe_out,
    k_weight,
    kv_cache,
    q_out,
    slot_mapping,
    positions,
    cos_cache,
    sin_cache,
    eps,
    is_neox,
    is_nope_first,
):
    aiter.fused_qk_norm_rope_group_quant_cache(
        q,
        kv,
        k_pe_out,
        k_weight,
        kv_cache,
        q_out,
        slot_mapping,
        positions,
        cos_cache,
        sin_cache,
        eps,
        is_neox,
        is_nope_first,
    )
    return kv_cache


@benchmark()
def test_fused_qk_norm_rope_group_quant_cache(
    head_dim: int,
    qk_rope_head_dim: int,
    num_tokens: int,
    block_size: int,
    num_blocks: int,
    num_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    device: str,
    kv_cache_dtype: str,
    is_neox: bool,
):
    ret = {}
    torch.set_default_device(device)
    nope_dim = head_dim - qk_rope_head_dim

    total_slots = num_blocks * block_size
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

    kv = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    q = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=device)
    k_weight = torch.ones(head_dim, dtype=dtype, device=device)
    cos_cache, sin_cache = compute_cache(num_tokens, qk_rope_head_dim // 2, dtype)
    cos_cache = cos_cache.to(device)
    sin_cache = sin_cache.to(device)

    pos = torch.randint(0, num_tokens, (num_tokens,), device=device)
    cache_dtype = dtypes.fp8 if kv_cache_dtype == "fp8" else dtype
    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=cache_dtype,
        device=device,
    )
    q_out = torch.empty(
        (num_tokens, num_heads, head_dim),
        dtype=dtype,
        device=device,
    )
    is_nope_first = True

    ref_temp = torch.zeros(*kv_cache.shape, dtype=cache_dtype, device=device)
    ref_kv_cache, ref_k_pe, ref_q_out = run_torch_fused_norm_rope_group_quant(
        q,
        kv,
        k_weight,
        ref_temp,
        q_out,
        slot_mapping,
        pos,
        cos_cache,
        sin_cache,
        1e-6,
        is_neox,
        is_nope_first,
        kv_cache_dtype,
    )
    k_pe_out = torch.empty(
        num_tokens, num_kv_heads, qk_rope_head_dim, dtype=dtype, device=device
    )
    # Correctness run
    aiter.fused_qk_norm_rope_group_quant_cache(
        q,
        kv,
        k_pe_out,
        k_weight,
        kv_cache,
        q_out,
        slot_mapping,
        pos,
        cos_cache,
        sin_cache,
        1e-6,
        is_neox,
        is_nope_first,
    )
    # Perf run
    kv_cache, avg_us = run_aiter_fused_norm_rope_group_quant(
        q,
        kv,
        k_pe_out,
        k_weight,
        kv_cache,
        q_out,
        slot_mapping,
        pos,
        cos_cache,
        sin_cache,
        1e-6,
        is_neox,
        is_nope_first,
    )
    kv_cache = kv_cache.reshape(
        num_tokens // block_size, block_size, num_kv_heads, head_dim
    )
    num_tiles = nope_dim // 64
    # K pe check (always)
    err_k_pe = checkAllclose(
        k_pe_out.to(torch.float32),
        ref_k_pe.to(torch.float32),
        atol=0.01,
        rtol=0.01,
        msg="k_pe rope compared with ref",
    )
    if kv_cache_dtype == "fp8":
        err_kv = checkAllclose(
            kv_cache[..., :nope_dim].to(torch.float32),
            ref_kv_cache[..., :nope_dim].to(torch.float32),
            atol=0.01,
            rtol=0.01,
            msg="fp8 kv nope data compared with ref",
        )
        err_k_scale = checkAllclose(
            kv_cache.view(torch.uint8)[..., nope_dim : nope_dim + num_tiles],
            ref_kv_cache.view(torch.uint8)[..., nope_dim : nope_dim + num_tiles],
            msg="fp8 kscale (in kv_cache) compared with ref",
        )
    else:
        err_kv = checkAllclose(
            kv_cache,
            ref_kv_cache,
            msg="bf16 kv result compared with ref",
        )
        err_k_scale = 0.0
    err_q_out = checkAllclose(
        q_out,
        ref_q_out,
        atol=0.01,
        rtol=0.01,
        msg="bf16 qout (norm+rope) result compared with ref",
    )

    ret["fused_qk_us"] = avg_us
    ret["hip_kv_err"] = err_kv
    ret["hip_q_err"] = err_q_out
    ret["hip_k_scale_err"] = err_k_scale
    ret["hip_k_pe_err"] = err_k_pe
    ret["aiter_bw(TB/s)"] = (
        num_tokens
        * (head_dim * num_kv_heads + num_heads * head_dim)
        * (torch.finfo(dtype).bits // 8)
        + num_tokens * head_dim * num_kv_heads * (torch.finfo(cache_dtype).bits // 8)
        + num_tokens * num_heads * head_dim * (torch.finfo(dtype).bits // 8)
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
    default=[4, 137, 512],
    help="""token nums.
    e.g.: -t 128""",
)
parser.add_argument(
    "-hd",
    "--head",
    type=int,
    nargs="*",
    default=[8],
    help="""num heads.
    e.g.: -hd 1""",
)
parser.add_argument(
    "-nkh",
    "--num_kv_heads",
    type=int,
    nargs="*",
    default=[2],
    help="""num kv heads.
    e.g.: -nkh 1""",
)
parser.add_argument(
    "-n",
    "--is_neox",
    type=dtypes.str2bool,
    nargs="*",
    default=[True],
    help="""true: GPT-NeoX style rotary embedding or false: GPT-J style rotary embedding.
    e.g.: --is_neox false
          or --is_neox true""",
)

args = parser.parse_args()

df = []
for num_token in args.token:
    num_blocks = num_token // args.block_size
    for num_heads in args.head:
        for num_kv_heads in args.num_kv_heads:
            for kv_cache_dtype in args.kv_dtype:
                for is_neox in args.is_neox:
                    ret = test_fused_qk_norm_rope_group_quant_cache(
                        head_dim=args.kv_lora_rank + args.qk_rope_head_dim,
                        qk_rope_head_dim=args.qk_rope_head_dim,
                        num_tokens=num_token,
                        block_size=args.block_size,
                        num_blocks=num_blocks,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        dtype=args.dtype,
                        device=args.device,
                        kv_cache_dtype=kv_cache_dtype,
                        is_neox=is_neox,
                    )
                    df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info(
    "fused_qk_norm_rope_group_quant_cache summary (markdown):\n%s",
    df_md,
)
