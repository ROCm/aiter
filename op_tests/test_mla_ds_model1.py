# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import random
import aiter
from aiter import dtypes
from aiter.ops import quant
from aiter.test_common import benchmark, checkAllclose, run_perftest
import argparse


def to_2buff_for_asm(quant_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    split quant_input to nope_scale_buff and rope_buff for ASM kernel.

    original layotu (per token): [nope(448) | rope(128) | scale(8)] = 584 bytes
    new layout:
        - nope_scale_buff: [nope(448) | scale(8) | padding(56)] = 512 bytes (aligned)
        - rope_buff: [rope(64 bf16)] = 128 bytes
    """
    d, d_nope, d_rope, tile_size, num_tiles = quant.MODEL1_FP8Sparse
    num_blocks, block_size, h_k, _ = quant_input.shape

    quant_kv_flat = quant_input.view(num_blocks, block_size, -1)

    # extract each part
    nope_part = quant_kv_flat[:, :, :d_nope]  # (num_blocks, block_size, 448) fp8
    rope_part = quant_kv_flat[
        :, :, d_nope : d_nope + 2 * d_rope
    ]  # (num_blocks, block_size, 128) bytes
    scale_part = quant_kv_flat[
        :, :, d_nope + 2 * d_rope :
    ]  # (num_blocks, block_size, 8) fp8

    nope_scale_size = d_nope + num_tiles + 1 + 56  # 448 + 8 + 56 = 512
    nope_scale_buff = torch.zeros(
        (num_blocks, block_size, nope_scale_size),
        dtype=dtypes.fp8,
        device=quant_input.device,
    )
    nope_scale_buff[:, :, :d_nope] = nope_part  # fill nope
    nope_scale_buff[:, :, d_nope : d_nope + num_tiles + 1] = (
        scale_part  # fill scale (include padding)
    )
    # the remaining 56 bytes are filled with 0

    rope_buff = rope_part.view(torch.bfloat16)  # (num_blocks, block_size, 64)

    return nope_scale_buff, rope_buff


def dequant_2buff_to_kv_cache(
    nope_scale_buff: torch.Tensor,  # (num_blocks, block_size, 512) fp8 container
    rope_buff: torch.Tensor,  # (num_blocks, block_size, 64) bf16
) -> torch.Tensor:
    """
    Dequantize nope_scale_buff and rope_buff back to BF16 KV cache.

    nope_scale_buff layout: [nope(448 fp8) | scale(7 e8m0) + 1 e8m0(padding) | padding(56)]
    rope_buff layout: [rope(64 bf16)]

    Returns:
        kv_cache: (num_blocks, block_size, 1, d) bf16, where d = d_nope + d_rope = 512
    """
    d, d_nope, d_rope, tile_size, num_tiles = quant.MODEL1_FP8Sparse
    num_blocks, block_size, _ = nope_scale_buff.shape

    # Extract nope and scale from nope_scale_buff
    input_nope = nope_scale_buff[:, :, :d_nope]  # (num_blocks, block_size, 448) fp8
    input_scale = (
        nope_scale_buff[:, :, d_nope : d_nope + num_tiles + 1]
        .view(num_blocks, block_size, num_tiles + 1)[:, :, :num_tiles]
        .view(dtypes.fp8_e8m0)
    )  # (num_blocks, block_size, 7) e8m0

    # rope_buff is already bf16
    input_rope = rope_buff  # (num_blocks, block_size, 64) bf16

    # Allocate result
    result = torch.empty(
        (num_blocks, block_size, d), dtype=dtypes.bf16, device=nope_scale_buff.device
    )

    # Dequantize nope: fp8 * scale -> bf16
    for tile_idx in range(num_tiles):
        cur_nope = input_nope[
            :, :, tile_idx * tile_size : (tile_idx + 1) * tile_size
        ].to(dtypes.bf16)
        cur_scales = input_scale[:, :, tile_idx].to(dtypes.bf16).unsqueeze(-1)
        result[:, :, tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
            cur_nope * cur_scales
        )

    # Copy rope directly (already bf16)
    result[:, :, d_nope:] = input_rope

    # Reshape to (num_blocks, block_size, 1, d)
    result = result.view(num_blocks, block_size, 1, d)
    return result


def native_to_2buff_for_asm(
    input: torch.Tensor,  # (num_blocks, block_size, head_num, dim) bf16, dim=512
) -> tuple[torch.Tensor, torch.Tensor]:
    d, d_nope, d_rope, tile_size, num_tiles = quant.MODEL1_FP8Sparse
    # d=512, d_nope=448, d_rope=64, tile_size=64, num_tiles=7

    assert input.shape[-1] == d, f"Expected dim={d}, got {input.shape[-1]}"
    num_blocks, block_size, num_heads, _ = input.shape

    # split nope and rope
    input_nope = input[..., :d_nope]  # (num_blocks, block_size, 448)
    input_rope = input[..., d_nope:]  # (num_blocks, block_size, 64)

    # nope_scale_buff: 448 (nope fp8) + 8 (scale e8m0) + 56 padding = 512 bytes
    nope_scale_size = 512
    nope_scale_buff = torch.zeros(
        (num_blocks, block_size, num_heads, nope_scale_size),
        dtype=dtypes.fp8,
        device=input.device,
    )
    nope_part = nope_scale_buff[..., :d_nope]
    scale_part = nope_scale_buff[..., d_nope : d_nope + num_tiles].view(
        dtypes.fp8_e8m0
    )  # (num_blocks, block_size, num_heads, 7)

    def _cast_scale_inv_to_ue8m0(
        t_input: torch.Tensor, out_dtype=torch.float32
    ) -> torch.Tensor:
        """make scale to 2^log2(scale)"""
        return torch.pow(2, torch.clamp_min(t_input, 1e-4).log2().ceil()).to(out_dtype)

    # quant nope: bf16 -> fp8
    for tile_idx in range(num_tiles):
        tile_start = tile_idx * tile_size
        tile_end = (tile_idx + 1) * tile_size
        cur_tile = input_nope[
            ..., tile_start:tile_end
        ]  # (num_blocks, block_size, num_heads, 64)

        # scale: max(abs(tile)) / max of dtype
        cur_scale_inverse = (
            torch.abs(cur_tile).max(dim=-1).values.float() / torch.finfo(dtypes.fp8).max
        )  # (num_blocks, block_size, num_heads)

        cur_scale_inverse = _cast_scale_inv_to_ue8m0(cur_scale_inverse)
        scale_part[..., tile_idx] = cur_scale_inverse.to(dtypes.fp8_e8m0)

        # quant nope tile
        cur_quantized = (cur_tile.float() / cur_scale_inverse.unsqueeze(-1)).to(
            dtypes.fp8
        )
        nope_part[..., tile_start:tile_end] = cur_quantized

    # rope no quant, direct output
    rope_buff = input_rope.contiguous()  # (num_blocks, block_size, num_heads, 64) bf16

    return nope_scale_buff, rope_buff


def quant_2buff_to_native(
    nope_scale_buff: torch.Tensor,  # (num_blocks, block_size, num_heads, 512) fp8
    rope_buff: torch.Tensor,  # (num_blocks, block_size, num_heads, 64) bf16
) -> torch.Tensor:
    """
    Reverse of native_to_2buff_for_asm: dequantize nope_scale_buff and concat with rope_buff.

    nope_scale_buff layout: [nope(448 fp8) | scale(7 e8m0) | padding(57)]

    Returns:
        torch.Tensor: (num_blocks, block_size, num_heads, 512) bf16
    """
    d, d_nope, d_rope, tile_size, num_tiles = quant.MODEL1_FP8Sparse
    # d=512, d_nope=448, d_rope=64, tile_size=64, num_tiles=7

    num_blocks, block_size, num_heads, _ = nope_scale_buff.shape

    # extract nope (fp8) and scale (e8m0)
    nope_part = nope_scale_buff[
        ..., :d_nope
    ]  # (num_blocks, block_size, num_heads, 448) fp8
    scale_part = nope_scale_buff[..., d_nope : d_nope + num_tiles].view(
        dtypes.fp8_e8m0
    )  # (num_blocks, block_size, num_heads, 7) e8m0

    # allocate output
    result = torch.empty(
        (num_blocks, block_size, num_heads, d),
        dtype=dtypes.bf16,
        device=nope_scale_buff.device,
    )

    # dequant nope: fp8 * scale -> bf16
    for tile_idx in range(num_tiles):
        tile_start = tile_idx * tile_size
        tile_end = (tile_idx + 1) * tile_size

        cur_nope = nope_part[..., tile_start:tile_end].to(dtypes.bf16)
        cur_scale = scale_part[..., tile_idx].to(dtypes.bf16).unsqueeze(-1)

        result[..., tile_start:tile_end] = cur_nope * cur_scale

    # copy rope
    result[..., d_nope:] = rope_buff

    return result


def ref_sparse_attn_decode(
    q_nope_scale_buff: torch.Tensor,  # (batch, s_q, h_q, 512) fp8
    q_rope_buff: torch.Tensor,  # (batch, s_q, h_q, 64) bf16
    kv_nope_scale_buff: torch.Tensor,  # (num_page, block_size, h_kv, 512) fp8
    kv_rope_buff: torch.Tensor,  # (num_page, block_size, h_kv, 64) bf16
    kv_indptr: torch.Tensor,  # (batch + 1,) cumulative kv lengths
    kv_indices: torch.Tensor,  # (total_kv_len,) page indices
    kv_last_page_lens: torch.Tensor,  # (batch,) valid tokens in last page
    sm_scale: float = 1.0,
    attn_sink: torch.Tensor = None,  # (h_q,) optional attention sink
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation of sparse attention decode.
    Based on flash-mla/tests/ref.py ref_sparse_attn_decode.

    Returns:
        output: (batch, s_q, h_q, d_v) bf16
        lse: (batch, h_q, s_q) logsumexp
    """
    d, d_nope, d_rope, tile_size, num_tiles = quant.MODEL1_FP8Sparse
    d_v = d_nope  # value dim = 448

    # 1. Dequantize q and kv to bf16
    q = quant_2buff_to_native(q_nope_scale_buff, q_rope_buff)  # (batch, s_q, h_q, d)
    kv_cache = quant_2buff_to_native(
        kv_nope_scale_buff, kv_rope_buff
    )  # (num_page, block_size, h_kv, d)

    batch = q.shape[0]
    s_q = q.shape[1]
    h_q = q.shape[2]
    h_kv = kv_cache.shape[2]
    num_page, block_size = kv_cache.shape[:2]

    assert h_kv == 1, "Only single KV head supported"

    # 2. Gather KV by page indices: select pages then flatten
    #    kv_cache: (num_page, block_size, h_kv, d) -> select by kv_indices
    kvc = torch.index_select(
        kv_cache, 0, kv_indices
    )  # (total_kv_len, block_size, h_kv, d)
    kvc = kvc.view(-1, h_kv, d)  # (total_kv_len * block_size, h_kv, d)
    # print(f"{kvc=}")
    # print(f"{kvc.shape=}")

    # 3. Split by batch using kv_indptr
    kv_indptr_list = (kv_indptr * block_size).tolist()
    # print(f"{kv_indptr_list=}")
    # print(len(kv_indptr_list))
    kvs = torch.tensor_split(kvc, kv_indptr_list[1:-1])  # list of (seq_len_k, h_kv, d)
    # print(len(kvs))

    # 4. Compute attention for each batch (using fp32, output bf16)
    outputs = []
    lses = []

    for b_idx in range(batch):
        q_b = q[b_idx].float()  # (s_q, h_q, d) fp32
        kv_b = kvs[b_idx].float()  # (seq_len_k, h_kv, d) fp32

        if kv_b.shape[0] == 0:
            # No KV tokens for this batch
            outputs.append(
                torch.zeros(s_q, h_q, d_v, dtype=torch.float32, device=q.device)
            )
            lses.append(
                torch.full(
                    (s_q, h_q), float("+inf"), dtype=torch.float32, device=q.device
                )
            )
            continue

        # Broadcast KV head to Q heads: (seq_len_k, 1, d) -> (seq_len_k, h_q, d)
        kv_b = kv_b.expand(-1, h_q, -1)  # (seq_len_k, h_q, d)

        # q_b: (s_q, h_q, d), kv_b: (seq_len_k, h_q, d)
        # Attention: Q @ K^T -> (s_q, h_q, seq_len_k)
        attn_weight = torch.einsum("shd,khd->shk", q_b, kv_b)  # (s_q, h_q, seq_len_k)
        attn_weight *= sm_scale

        # Softmax
        lse = attn_weight.logsumexp(dim=-1)  # (s_q, h_q)
        attn_weight = torch.exp(
            attn_weight - lse.unsqueeze(-1)
        )  # (s_q, h_q, seq_len_k)

        # Output: attn @ V (V is first d_v dims of kv)
        v_b = kv_b[..., :d_v]  # (seq_len_k, h_q, d_v)
        output = torch.einsum("shk,khv->shv", attn_weight, v_b)  # (s_q, h_q, d_v)

        outputs.append(output)
        lses.append(lse)

    output = torch.stack(outputs, dim=0)  # (batch, s_q, h_q, d_v) fp32
    lse = torch.stack(lses, dim=0)  # (batch, s_q, h_q) fp32

    # 5. Attention sink adjustment
    if attn_sink is not None:
        sink_factor = 1.0 / (1.0 + torch.exp(attn_sink.view(1, 1, h_q) - lse))
        output *= sink_factor.unsqueeze(-1)

    # 6. Handle lonely Q (no valid K)
    lonely_q_mask = lse == float("-inf")
    output[lonely_q_mask.unsqueeze(-1).expand_as(output)] = 0.0
    lse[lonely_q_mask] = float("+inf")

    # Convert to bf16 for output
    return output.to(dtypes.bf16), lse.transpose(1, 2).to(
        dtypes.bf16
    )  # output: (b,s_q,h_q,d_v), lse: (b,h_q,s_q)


def run_asm_sparse_attn_decode():
    pass


def test_sparse_attn_decode(
    batch_size,
    head_num_of_q,
    head_num_of_kv,
    seq_len_of_q,
    seq_len_of_k,
    have_zero_seqlen_k,
    varlen,
    topk,
    var_topk,
    enable_attn_sink,
    extra_s_k,
    extra_topk,
    block_size,
    extra_block_size,
    dim_qk,
):
    #     print(f"""test_sparse_attn_decode(
    #     batch_size={batch_size},
    #     head_num_of_q={head_num_of_q},
    #     head_num_of_kv={head_num_of_kv},
    #     seq_len_of_q={seq_len_of_q},
    #     seq_len_of_k={seq_len_of_k},
    #     varlen={varlen},
    #     topk={topk},
    #     var_topk={var_topk},
    #     enable_attn_sink={enable_attn_sink},
    #     extra_s_k={extra_s_k},
    #     extra_topk={extra_topk},
    #     block_size={block_size},
    #     extra_block_size={extra_block_size},
    #     dim_qk={dim_qk}
    # )""")
    # 1. prepare input
    assert head_num_of_q % head_num_of_kv == 0

    q = torch.randn((batch_size, seq_len_of_q, head_num_of_q, dim_qk))
    q.clamp_(min=-1.0, max=1.0)
    q_nope_scale_buff, q_rope_buff = native_to_2buff_for_asm(q)

    # print(f"{q=}")
    # print(f"{q.shape=}")
    # print(f"{q_nope_scale_buff=}")
    # print(f"{q_nope_scale_buff.shape=}")
    # print(f"{q_rope_buff=}")
    # print(f"{q_rope_buff.shape=}")

    attn_sink = None
    if enable_attn_sink:
        attn_sink = torch.randn((head_num_of_q,), dtype=torch.float32)
        inf_mask = torch.randn((head_num_of_q,), dtype=torch.float32)
        attn_sink[inf_mask > 0.5] = float("inf")
        attn_sink[inf_mask < -0.5] = float("-inf")
    # print(f"{attn_sink=}")

    def gen_one_k_scope(
        s_k: int,
        block_size: int,
        topk: int,
        varlen: bool,
        have_zero_seqlen_k: bool,
        is_all_indices_invalid: bool = False,
        have_topk_length: bool = False,
    ) -> torch.Tensor:
        random.seed(0)

        kv_max_sz = 65536 * 32  # max num of tokens in kv cache
        num_page = (kv_max_sz + block_size - 1) // block_size

        # qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
        kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
        cache_seqlens = torch.full((batch_size,), s_k, dtype=torch.int32)
        kv_last_page_lens = torch.ones(batch_size, dtype=torch.int32)

        if varlen:
            for i in range(batch_size):
                cache_seqlens[i] = max(random.normalvariate(s_k, s_k / 2), seq_len_of_q)
        if have_zero_seqlen_k:
            zeros_mask = torch.randn(batch_size, dtype=torch.float32) > 0
            cache_seqlens[zeros_mask] = 0

        kv_indptr[1 : batch_size + 1] = torch.cumsum(cache_seqlens, dim=0)
        kv_indices = torch.randint(
            0, num_page, (kv_indptr[-1].item(),), dtype=torch.int32
        )

        kv_cache = (
            torch.randn(
                (num_page, block_size, head_num_of_kv, dim_qk), dtype=torch.bfloat16
            )
            / 10
        )
        kv_cache.clamp_(min=-1.0, max=1.0)

        return kv_cache, kv_indptr, kv_indices, kv_last_page_lens

    native_kv_cache, kv_indptr, kv_indices, kv_last_page_lens = gen_one_k_scope(
        seq_len_of_k,
        block_size,
        topk,
        varlen,
        have_zero_seqlen_k,
    )
    # print(f"{native_kv_cache.shape=}")
    # print(f"{kv_indptr=}")
    # print(f"{kv_indices=}")
    # print(f"{kv_last_page_lens=}")

    ### quant for fp8

    # quant_kv_cache = quant.quantize_k_cache_ds_model1(native_kv_cache)
    # kv_nope_scale_buff, kv_rope_buff = to_2buff_for_asm(quant_kv_cache)
    # dequant_kv_cache = dequant_2buff_to_kv_cache(kv_nope_scale_buff, kv_rope_buff)
    q_nope_scale_buff, q_rope_buff = native_to_2buff_for_asm(q)
    kv_nope_scale_buff, kv_rope_buff = native_to_2buff_for_asm(native_kv_cache)

    # 2. call reference implementation
    """
    refer: need use quantized kv cache.

    """
    ref_out, ref_lse = ref_sparse_attn_decode(
        q_nope_scale_buff,
        q_rope_buff,
        kv_nope_scale_buff,
        kv_rope_buff,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
    )

    # 3. call asm implementation
    """
    ASM kernel memory layout:
        buffer1: (d_nope + scale + padding_to_512 ) = 512 bytes per token
                - d_nope: quantized nope part = 448
                - scale = num_tiles + 1 = 8
                - padding_to_512 = 512 - 448 - 8 = 56
        
        buffer2: (2*d_rope) = 128 bytes per token
                - d_rope: 64 bf16 dtype

        out_asm,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,

    Constraint: block_size must be 1
    """

    # 4. compare results

    # 5. print results
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d_qk",
        "--dim_qk",
        type=int,
        default=512,
        help="""dim of qk for attention.
        e.g.: -d_qk 512""",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=[74],
        help="""batch size.
        e.g.: -b 74""",
    )
    parser.add_argument(
        "-s",
        "--block_size",
        type=int,
        default=[1],
        help="""block size.
        e.g.: -s 1""",
    )
    parser.add_argument(
        "-hq",
        "--head_num_of_q",
        type=int,
        default=128,
        help="""head num of q.
        e.g.: -hq 128""",
    )
    parser.add_argument(
        "-hk",
        "--head_num_of_kv",
        type=int,
        default=1,
        help="""head num of kv.
        e.g.: -hk 1""",
    )
    parser.add_argument(
        "-s_q",
        "--seq_len_of_q",
        type=int,
        default=[1],
        help="""sequence length of q.
        e.g.: -s_q 1""",
    )
    parser.add_argument(
        "-s_k",
        "--seq_len_of_k",
        type=int,
        default=[512],
        help="""sequence length of k.
        e.g.: -s_k 512""",
    )
    parser.add_argument(
        "--have_zero_seqlen_k",
        type=dtypes.str2bool,
        default=False,
        help="""Have zero sequence length of K. Default is disabled.
        e.g.: --have_zero_seqlen_k true    # have zero sequence length of K.
        """,
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=[64],
        help="""topk of sparse attn. topk=64 means the top 64 tokens are attended to.
        e.g.: -t 64""",
    )
    parser.add_argument(
        "-e",
        "--extra_topk",
        type=dtypes.str2tuple,
        default=[(512, 64, 2)],
        help="""have extra topk. Format: (extra_s_k, extra_topk, extra_block_size) 
        e.g.: -e 512,64,2""",
    )
    parser.add_argument(
        "--varlen",
        type=dtypes.str2bool,
        default=[True],
        help="""variable sequence length per batch.
        e.g.: -varlen false
        or -varlen true""",
    )
    parser.add_argument(
        "--var_topk",
        type=dtypes.str2bool,
        default=[True],
        help="""variable topk per sequence. Default is enabled.
        e.g.: -var_topk false    # disable variable topk""",
    )
    parser.add_argument(
        "-a_s",
        "--attn_sink",
        type=dtypes.str2bool,
        default=[True],
        help="""enable attention sink. Default is enabled.
        e.g.: -a_s false    # disable attention sink""",
    )

    args = parser.parse_args()
    # d, d_nope, d_rope, tile_size, num_tiles = quant.MODEL1_FP8Sparse

    for batch_size in args.batch_size:
        for seq_len_of_q in args.seq_len_of_q:
            for seq_len_of_k in args.seq_len_of_k:
                for varlen in args.varlen:
                    for topk in args.topk:
                        for var_topk in args.var_topk:
                            for enable_attn_sink in args.attn_sink:
                                for (
                                    extra_s_k,
                                    extra_topk,
                                    extra_block_size,
                                ) in args.extra_topk:
                                    for block_size in args.block_size:
                                        test_sparse_attn_decode(
                                            batch_size=batch_size,
                                            head_num_of_q=args.head_num_of_q,
                                            head_num_of_kv=args.head_num_of_kv,
                                            seq_len_of_q=seq_len_of_q,
                                            seq_len_of_k=seq_len_of_k,
                                            have_zero_seqlen_k=args.have_zero_seqlen_k,
                                            varlen=varlen,
                                            topk=topk,
                                            var_topk=var_topk,
                                            enable_attn_sink=enable_attn_sink,
                                            extra_s_k=extra_s_k,
                                            extra_topk=extra_topk,
                                            block_size=block_size,
                                            extra_block_size=extra_block_size,
                                            dim_qk=args.dim_qk,
                                        )
