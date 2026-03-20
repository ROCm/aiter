# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
import random
import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

# current supported case in decode MLA: mtp == 0, 1, 2, 3 (decode_qlen = 1, 2, 3, 4)
# qdtype bf16, kdtype bf16: nhead16, nhead128
# qdtype fp8, kdtype fp8: nhead16, nhead128


def check_support(dtype, kv_dtype, nhead):
    if dtype == dtypes.fp8 and kv_dtype == dtypes.bf16:
        return False
    return True


def cal_diff(
    x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool = False
) -> None:
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    # print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    if use_fp8:
        assert cos_diff < 3e-2
    else:
        assert cos_diff < 1e-5


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    dtype,
    is_causal=True,
) -> torch.Tensor:
    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    if is_causal:
        s_q = query.shape[0]
        s_k = key.shape[0]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weights += attn_bias
    attn_weights = torch.softmax(attn_weights, dim=-1)

    out = torch.einsum("hqk,khd->qhd", attn_weights.float(), value.float())
    return out.to(dtype)


def torch_mha_extend(
    q,  # [total_q, nheads, headdim_q]
    k,  # [num_page * page_size, nhead_kv, qk_head_dim]
    v,  # [num_page * page_size, nhead_kv, qk_head_dim]
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
    dtype,
):
    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    ks = torch.tensor_split(k, kv_indptr.tolist()[1:])
    vs = torch.tensor_split(v, kv_indptr.tolist()[1:])
    bs = qo_indptr.shape[0] - 1

    os = []
    for i in range(bs):
        q = qs[i]
        k = ks[i]
        v = vs[i]
        o = ref_masked_attention(q, k, v, sm_scale, dtype)
        os.append(o)
    o = torch.concat(os)
    return o


def torch_mla_extend(
    q,  # [total_q, nheads, headdim_q]
    kvc_cache,  # [num_page * page_size, nhead_kv, qk_head_dim]
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype,
    is_causal=True,
):
    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kvc_cache, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    bs = qo_indptr.shape[0] - 1

    os = []
    for i in range(bs):
        kvc = kvs[i]
        q = qs[i]
        k = kvc
        v, _ = torch.split(kvc, [kv_lora_rank, qk_rope_head_dim], dim=-1)
        o = ref_masked_attention(q, k, v, sm_scale, dtype, is_causal=is_causal)
        os.append(o)
    o = torch.concat(os)
    return o


@benchmark()
def test_mla(
    ctx_lens,
    batch_size,
    nhead,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    dtype,
    kvtype,
    page_size,
    varlen,
    decode_qlen,
    split_per_batch=None,
):
    ret = {}

    kv_max_sz = (
        65536 * 32
    )  # calculated by rest of mem after weight loaded in frameworks
    num_page = (kv_max_sz + page_size - 1) // page_size

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    seq_lens_qo = torch.empty(batch_size, dtype=torch.int)
    seq_lens_kv = torch.empty(batch_size, dtype=torch.int)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
    if varlen:
        for i in range(batch_size):
            seq_lens_kv[i] = max(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens)
            seq_lens_qo[i] = max(
                min(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens), 1
            )
    else:
        seq_lens_kv.fill_(ctx_lens)
        seq_lens_qo.fill_(ctx_lens)
    kv_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_kv, dim=0)
    kv_indices = torch.randint(0, num_page, (kv_indptr[-1].item(),), dtype=torch.int)
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    max_seqlen_qo = seq_lens_qo.max().item()
    max_seqlen_kv = seq_lens_kv.max().item()
    total_qo = qo_indptr[-1].item()
    total_kv = kv_indptr[-1].item()
    kv_buffer = torch.randn(
        (num_page * page_size, 1, kv_lora_rank + qk_rope_head_dim),
        dtype=torch.bfloat16,
    )
    ### kv_buffer quant & dequant
    ### Without quant and dequant, it's difficult to distinguish errors caused by an incorrect fp8 kernel.
    def quant_and_dequant_kv_buffer(kv_buffer:torch.Tensor, d_qk:int)->torch.Tensor:
        fp8_kvcache_layout = None   # (d, d_nope, d_rope, tile_size, num_tiles)
        if d_qk == 576:
            fp8_kvcache_layout = (576, 512, 64, 128, 4)
        elif d_qk == 512:
            fp8_kvcache_layout = (512, 448, 64, 64, 7)
        else:
            assert False
        def quantize_k_cache(
            input_k_cache: torch.Tensor,    # (num_blocks, block_size, h_k, d)
            kvcache_layout,
        ) -> torch.Tensor:
            """
            Quantize the k-cache
            For more detail about the layout of K/V, please refer to comments in flash_mla_interface.py
            """
            d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
            assert input_k_cache.shape[-1] == d
            num_blocks, block_size, h_k, _ = input_k_cache.shape
            assert h_k == 1
            input_k_cache = input_k_cache.squeeze(2)    # [num_blocks, block_size, d]
            input_elem_size = input_k_cache.element_size()

            if kvcache_layout == FP8KVCacheLayout.V32_FP8Sparse:
                bytes_per_token = d_nope + num_tiles*4 + input_elem_size*d_rope
                result = torch.empty((num_blocks, block_size+1, bytes_per_token), dtype=torch.float8_e4m3fn, device=input_k_cache.device)[:, :block_size, :]
                result_k_nope_part = result[..., :d_nope]
                result_k_scale_factor = result[..., d_nope: d_nope + num_tiles*4].view(torch.float32)
                result_k_rope_part = result[..., d_nope + num_tiles*4:].view(input_k_cache.dtype)
                result_k_rope_part[:] = input_k_cache[..., d_nope:]

                for tile_idx in range(0, num_tiles):
                    cur_scale_factors_inv = torch.abs(input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size]).max(dim=-1).values.float() / 448.0 # [num_blocks, block_size]
                    cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
                    result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv

                    cur_scale_factors_inv.unsqueeze_(-1)    # [num_blocks, block_size, 1]
                    cur_quantized_nope = (input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size].float() / cur_scale_factors_inv.float()).to(torch.float8_e4m3fn)
                    result_k_nope_part[..., tile_idx*tile_size:(tile_idx+1)*tile_size] = cur_quantized_nope
                
                result = result.view(num_blocks, block_size, 1, -1)
                return result
            
            elif kvcache_layout == FP8KVCacheLayout.MODEL1_FP8Sparse:
                bytes_per_token = d_nope + 2*d_rope + num_tiles + 1
                size_per_block_padded = (block_size*bytes_per_token + 576-1) // 576 * 576
                result = torch.empty((num_blocks, size_per_block_padded), dtype=torch.float8_e4m3fn, device=input_k_cache.device)[:, :block_size*bytes_per_token]
                result_k_nope_rope_part = result[:, :block_size*(d_nope+2*d_rope)].view(num_blocks, block_size, d_nope + 2*d_rope)
                result_k_nope = result_k_nope_rope_part[:, :, :d_nope]  # [num_blocks, block_size, d_nope]
                result_k_rope = result_k_nope_rope_part[:, :, d_nope:].view(input_k_cache.dtype)  # [num_blocks, block_size, d_rope]
                result_k_scale_factor = result[:, block_size*(d_nope+2*d_rope):].view(num_blocks, block_size, 8)[:, :, :7].view(torch.float8_e8m0fnu)   # [num_blocks, block_size, num_tiles]

                result_k_rope[:] = input_k_cache[..., d_nope:]
                for tile_idx in range(0, num_tiles):
                    cur_scale_factors_inv = torch.abs(input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size]).max(dim=-1).values.float() / 448.0 # [num_blocks, block_size]
                    cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
                    result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv.to(torch.float8_e8m0fnu)

                    cur_scale_factors_inv = cur_scale_factors_inv.view(num_blocks, block_size, 1)
                    cur_quantized_nope = (input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size].float() / cur_scale_factors_inv.float()).to(torch.float8_e4m3fn)
                    result_k_nope[:, :, tile_idx*tile_size:(tile_idx+1)*tile_size] = cur_quantized_nope
                
                result = result.view(num_blocks, block_size, 1, -1)
                return result

            else:
                raise NotImplementedError(f"Unsupported kvcache_layout: {kvcache_layout}")
            
        blocked_k_quantized =  

    # for none absorb (mha)
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    sm_scale = 1.0 / (qk_head_dim**0.5)

    # ############################## normal: prefill
    def test_normal_prefill():   ### 这个是纯 MHA，不是MLA。里面的torch_mha_extend 是MHA的reference实现
        q = torch.randn((total_qo, nhead, qk_head_dim), dtype=torch.bfloat16)
        k = torch.randn((total_kv, nhead, qk_head_dim), dtype=torch.bfloat16)
        v = torch.randn((total_kv, nhead, v_head_dim), dtype=torch.bfloat16)

        out_ref = torch_mha_extend(
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            dtype=dtype,
        )

        out_aiter, us_aiter = run_perftest(
            aiter.flash_attn_varlen_func,
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            max_seqlen_qo,
            max_seqlen_kv,
            softmax_scale=sm_scale,
            causal=True,
        )

        flop = (
            batch_size
            * nhead
            * 2
            * (ctx_lens * qk_head_dim * ctx_lens + ctx_lens * ctx_lens * v_head_dim)
        )
        checkAllclose(
            out_ref.to(torch.float),
            out_aiter.to(torch.float),
            msg=f"mla_prefill-normal    [torch vs  aiter_ck]: {us_aiter:>8.2f} us...... {flop/us_aiter/1000/1000:>8.2f} TFlops",
        )
        return us_aiter

    out_dtype = torch.bfloat16

    us_aiter = None
    if (
        dtype == torch.bfloat16 and kvtype == torch.bfloat16
    ) and batch_size * ctx_lens * nhead < 256 * 8192 * 16:
        us_aiter = test_normal_prefill()
        ret["prefill:ck_192"] = us_aiter

    torch.cuda.empty_cache()
    # absorb init
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    nhead_kv = 1
    v_head_dim = kv_lora_rank
    sm_scale = 1.0 / (qk_head_dim**0.5)

    # test prefill
    # ############################## absorb: prefill
    def test_absorb_prefill():     ### 这个带absorb的，是真 MLA ，里面的torch_mla_extend是MLA的reference实现
        q = torch.randn((total_qo, nhead, qk_head_dim), dtype=torch.bfloat16)

        out_ref = torch_mla_extend(
            q,
            kv_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype=out_dtype,
        )

        # #triton version
        # prefix_indptr = kv_indptr - qo_indptr
        # tmp = kv_indptr[1:] - seq_lens_qo
        # tmp_inpptr, _ = torch.concat([kv_indptr[1:], tmp]).sort()
        # prefix_kv_indices = kv_indices.tensor_split(tmp_inpptr.tolist())
        # extend_kv_indices = torch.concat(
        #     [el for i, el in enumerate(prefix_kv_indices) if i % 2 == 1]
        # )
        # prefix_kv_indices = torch.concat(
        #     [el for i, el in enumerate(prefix_kv_indices) if i % 2 == 0]
        # )
        # extend_kvc = torch.index_select(kv_buffer, 0, extend_kv_indices)
        # out_triton = torch.empty((total_qo, nhead, v_head_dim), dtype=dtype).fill_(-1)
        # _, us_triton = run_perftest(
        #     mla_extend_ref.extend_attention_fwd,
        #     q,
        #     extend_kvc,
        #     extend_kvc[..., :kv_lora_rank],
        #     out_triton,
        #     kv_buffer,
        #     kv_buffer[..., :kv_lora_rank],
        #     qo_indptr,
        #     prefix_indptr,
        #     prefix_kv_indices,
        #     None,
        #     None,
        #     max_seqlen_qo,
        #     sm_scale,
        #     num_iters=5,
        # )
        # checkAllclose(
        #     out_ref,
        #     out_triton,
        #     msg=f"mla_prefill-absorb    [torch vs    triton]:{us_torch:>8.2f} us vs {us_triton:>8.2f} us......",
        # )

        out_asm = torch.empty((total_qo, nhead, v_head_dim), dtype=out_dtype).fill_(-1)
        (attn_logits, attn_lse), us_asm = run_perftest(
            aiter.mla.mla_prefill_fwd,
            q,
            kv_buffer.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            sm_scale,
        )

        checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_prefill-absorb    [torch vs aiter_asm]: {us_asm:>8.2f} us......",
        )
        return us_asm

    us_asm = None
    if (
        dtype == torch.bfloat16 and kvtype == torch.bfloat16 and nhead in [16, 128]
    ) and batch_size * ctx_lens * nhead < 32 * 8192 * 16:
        us_asm = test_absorb_prefill()
        ret["prefill:asm_576"] = us_asm

    torch.cuda.empty_cache()

    # ############################## absorb: decode
    # seq_lens_qo = torch.randint(1, 5, (batch_size,), dtype=torch.int)
    # if nhead == 16 and decode_qlen != 1:
    #     return
    seq_lens_qo.fill_(decode_qlen)

    max_seqlen_qo = seq_lens_qo.max().item()
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = qo_indptr[-1].item()
    q = torch.randn((total_q, nhead, qk_head_dim), dtype=torch.bfloat16)

    # troch implementation
    out_ref = torch_mla_extend(      #### 这个看样子既可以prefill，也可以decode
        q,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        kv_lora_rank,
        qk_rope_head_dim,
        is_causal=True,
        dtype=out_dtype,
    )

    # Triton implementation
    # if decode_qlen == 1:
    #     if qk_head_dim != v_head_dim:
    #         out_triton = q.new_empty((total_q, nhead, v_head_dim)).fill_(-1)
    #     else:
    #         out_triton = torch.empty_like(q)

    #     num_kv_splits = 16
    #     attn_logits = torch.empty(
    #         (total_q, nhead, num_kv_splits, v_head_dim + 1),
    #         dtype=dtypes.fp32,
    #     )
    #     _, us_ref = run_perftest(
    #         mla_decode_ref.decode_attention_fwd,
    #         q,
    #         kv_buffer,
    #         kv_buffer[..., :kv_lora_rank],
    #         out_triton,
    #         kv_indptr,
    #         kv_indices,
    #         attn_logits,
    #         num_kv_splits,
    #         sm_scale,
    #         num_iters=5,
    #     )
    #     # logits_ref, lse_ref = attn_logits.split([v_head_dim, 1], dim=-1)
    #     # logits_ref = rearrange(logits_ref, "bs h sp d -> bs sp h d")
    #     # lse_ref = rearrange(lse_ref, "bs h sp d -> bs sp h d")
    #     checkAllclose(
    #         out_ref,
    #         out_triton,
    #         msg=f"mla_decode-absorb    [golden vs    triton]:{us_torch_decode:>8.2f} us vs {us_ref:>8.2f} us......",
    #     )

    def test_absorb_decode_bf16():
        kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)
        (attn_logits, attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q,
            kv_buffer.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            sm_scale,
            num_kv_splits=split_per_batch,
        )

        # print(f"{out_ref.view(total_q, -1)=}")
        # print(f"{out_asm.view(total_q, -1)=}")
        # checkAllclose(logits_ref, attn_logits,
        #               msg=f'attn_logits [golden vs aiter_asm]')
        # checkAllclose(lse_ref, attn_lse,
        #               msg=f'attn_lse    [golden vs aiter_asm]')
        err = checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_decode-absorb    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )
        return err, us_asm_decode

    def test_absorb_decode_fp8():
        if dtype != dtypes.fp8 and nhead == 128:
            aiter.logger.info("don't support this case:\n")
            return None, 1e12
        kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)

        q_fp8 = q.to(dtype)
        q_scale = None
        if dtype == dtypes.fp8:
            q_scale = torch.ones([1], dtype=torch.float, device="cuda")
        else:
            aiter.logger.info("don't support this case.")
            return None, 1e12

        kv_buffer_fp8 = kv_buffer.to(kvtype)
        kv_scale = torch.ones([1], dtype=torch.float, device="cuda")

        (attn_logits, attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q_fp8 if dtype == dtypes.fp8 else q,
            kv_buffer_fp8.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            sm_scale,
            q_scale=q_scale,
            kv_scale=kv_scale,
            num_kv_splits=split_per_batch,
        )

        # print(f"{out_ref.view(total_q, -1)=}")
        # print(f"{out_asm.view(total_q, -1)=}")
        # checkAllclose(logits_ref, attn_logits,
        #               msg=f'attn_logits [golden vs aiter_asm]')
        # checkAllclose(lse_ref, attn_lse, msg="attn_lse    [golden vs aiter_asm]")
        err = checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_decode-absorb_fp8    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )

        cal_diff(out_ref, out_asm, "out", True)
        return err, us_asm_decode

    err = None
    us_asm_decode = 1e12
    if (dtype == torch.bfloat16 and kvtype == torch.bfloat16) and nhead in [
        16,
        32,
        64,
        128,
    ]:
        err, us_asm_decode = test_absorb_decode_bf16()
    elif kvtype == dtypes.fp8 and nhead in [16, 128]:
        err, us_asm_decode = test_absorb_decode_fp8()

    ret["decode:err"] = err
    ret["decode:asm_576"] = us_asm_decode

    flops = decode_qlen * total_kv * nhead * (qk_head_dim + v_head_dim) * 2
    bytes = (
        total_kv * nhead_kv * qk_head_dim * (torch.finfo(kvtype).bits // 8)
        + total_q * nhead * qk_head_dim * (torch.finfo(dtype).bits // 8)
        + total_q * nhead * v_head_dim * (torch.finfo(out_dtype).bits // 8)
    )

    ret["decode:flops"] = flops
    ret["decode:bytes"] = bytes
    ret["decode:TFLOPS"] = flops / us_asm_decode / 1e6
    ret["decode:TB/s"] = bytes / us_asm_decode / 1e6

    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-k",
    "--kv_lora_rank",
    type=int,
    default=[512, 448],
    help="""kv lora rank.
    e.g.: -k 512""",
)
parser.add_argument(
    "-qn",
    "--qk_nope_head_dim",
    type=int,
    default=128,
    help="""qk nope head dim.
    e.g.: -qn 128""",
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
    "-vh",
    "--v_head_dim",
    type=int,
    default=128,
    help="""v head dim.
    e.g.: -vh 128""",
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
    nargs="*",
    default="bf16,",
    choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]],
    metavar="{bf16, fp8}",
    help="""Data type of Q.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-kvd",
    "--kv_dtype",
    nargs="*",
    type=dtypes.str2Dtype,
    default="bf16,",
    choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]],
    metavar="{bf16, fp8}",
    help="""Data type of KV.
    e.g.: -kvd bf16""",
)
parser.add_argument(
    "-c",
    "--ctxLen",
    type=int,
    nargs="*",
    default=[21, 64, 256, 512, 1200, 3200, 5200, 8192],
    help="""Context length.
    e.g.: -c 21""",
)
parser.add_argument(
    "-b",
    "--batchSize",
    type=int,
    nargs="*",
    default=[1, 3, 5, 16, 32, 64, 128, 256],
    help="""Batch size.
    e.g.: -b 16""",
)
parser.add_argument(
    "-n",
    "--nhead",
    type=dtypes.str2tuple,
    choices=[(16, 1), (16, 2), (16, 4), (128, 1), (128, 2)],
    nargs="*",
    const=None,
    default=[(16, 1), (16, 2), (16, 4), (128, 1), (128, 2)],
    help="""Number of nhead and decode_qlen.
    e.g.: -n 16,1""",
)
parser.add_argument(
    "-splits",
    "--split_per_batch",
    type=int,
    nargs="*",
    default=[None],
    help="""kv seqlens split num for per batch.
    e.g.: -splits 32""",
)
parser.add_argument(
    "--varlen",
    action="store_true",
    help="""variable kv seqlens per batch. Default: False.
    --varlen # True""",
)
parser.add_argument(
    "-det",
    "--decode_extra_topk",
    action="store_true",
    help="""decode extra topk. Default: False.
    e.g.: -det # True""",
)

args = parser.parse_args()

for nhead, decode_qlen in args.nhead:
    df = []
    for dtype, kvtype, ctx_len, batch_size, split_per_batch in itertools.product(
        args.dtype, args.kv_dtype, args.ctxLen, args.batchSize, args.split_per_batch
    ):
        if check_support(dtype, kvtype, nhead):
            ret = test_mla(
                ctx_len,
                batch_size,
                nhead,
                args.kv_lora_rank,
                args.qk_nope_head_dim,
                args.qk_rope_head_dim,
                args.v_head_dim,
                dtype,
                kvtype,
                args.block_size,
                varlen=args.varlen,
                decode_qlen=decode_qlen,
                split_per_batch=split_per_batch,
            )
            df.append(ret)
    df = pd.DataFrame(df)
    # df.to_csv(f"mla_nhead{nhead}decode_qlen{decode_qlen}.csv")
    df_md = df.to_markdown(index=False)
    aiter.logger.info("mla summary (markdown):\n%s", df_md)
