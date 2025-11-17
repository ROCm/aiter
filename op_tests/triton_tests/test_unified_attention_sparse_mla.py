# test code is adapted from flashMLA:
# https://github.com/deepseek-ai/FlashMLA/blob/main/tests/test_flash_mla_decoding.py
import random
import dataclasses
from typing import Optional, Tuple

import torch
import pytest
from math import ceil
from aiter.ops.triton.unified_attention_sparse_mla import unified_attention_sparse_mla


def cdiv(a, b):
    return ceil(a / b)


@dataclasses.dataclass
class Param:
    b: int  # Batch size
    s_q: int  # Number of queries for one request
    s_k: int  # Seq len, or mean seq len if varlen == True
    is_varlen: bool
    is_causal: bool
    is_fp8: bool
    topk: Optional[int] = None
    test_performance: bool = True
    is_all_indices_invalid: bool = False
    have_zero_seqlen_k: bool = False
    block_size: int = 64
    h_q: int = 128  # Number of q heads
    h_kv: int = 1  # Number of kv heads
    d: int = 576  # Q/K head dim (= dv + RoPE dim)
    dv: int = 512  # V head dim
    seed: int = 0


def generate_test_data(
    t: Param,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Generate test data from a given configuration
    Return: [cache_seqlens, q, block_table, blocked_k]
    Pay attention: This function changes the random seed
    """
    random.seed(t.seed)
    torch.manual_seed(t.seed)
    torch.cuda.manual_seed(t.seed)
    torch.backends.cudnn.deterministic = True

    assert t.h_q % t.h_kv == 0

    cache_seqlens_cpu = torch.full((t.b,), t.s_k, dtype=torch.int32, device="cpu")
    if t.is_varlen:
        for i in range(t.b):
            cache_seqlens_cpu[i] = max(random.normalvariate(t.s_k, t.s_k / 2), t.s_q)

    if t.have_zero_seqlen_k:
        zeros_mask = torch.randn(t.b, dtype=torch.float32, device="cpu") > 0
        cache_seqlens_cpu[zeros_mask] = 0

    max_seqlen = cache_seqlens_cpu.max().item()
    max_seqlen_pad = cdiv(max_seqlen, 256) * 256
    cache_seqlens = cache_seqlens_cpu.cuda()

    q = torch.randn(t.b, t.s_q, t.h_q, t.d)
    q.clamp_(min=-1.0, max=1.0)

    block_table = torch.arange(
        t.b * max_seqlen_pad // t.block_size, dtype=torch.int32
    ).view(t.b, max_seqlen_pad // t.block_size)
    block_table = block_table.view(-1)[torch.randperm(block_table.numel())].view(
        t.b, -1
    )
    blocked_k = torch.randn(block_table.numel(), t.block_size, t.h_kv, t.d) / 10
    blocked_k.clamp_(min=-1.0, max=1.0)

    if t.topk is None:
        for i in range(t.b):
            cur_len = cache_seqlens_cpu[i].item()
            cur_num_blocks = cdiv(cur_len, t.block_size)
            blocked_k[block_table[i][cur_num_blocks:]] = float("nan")
            if cur_len % t.block_size != 0:
                blocked_k[block_table[i][cur_num_blocks - 1]][
                    cur_len % t.block_size :
                ] = float("nan")
            block_table[i][cur_num_blocks:] = 2147480000
        return cache_seqlens, q, block_table, blocked_k, None, None
    else:
        block_table_cpu = block_table.cpu()
        abs_indices = torch.empty(t.b, t.s_q, t.topk, dtype=torch.int32, device="cpu")
        indices_in_kvcache = torch.empty(
            t.b, t.s_q, t.topk, dtype=torch.int32, device="cpu"
        )
        for i in range(t.b):
            # Generate indices
            for j in range(t.s_q):
                cur_abs_indices = torch.randperm(
                    int(cache_seqlens_cpu[i].item()), device="cpu"
                )[: t.topk]
                cur_blocked_indices = block_table_cpu[
                    i, cur_abs_indices // t.block_size
                ] * t.block_size + (cur_abs_indices % t.block_size)
                if len(cur_abs_indices) < t.topk:
                    pad_len = t.topk - len(cur_abs_indices)
                    cur_abs_indices = torch.cat(
                        [cur_abs_indices, torch.full((pad_len,), -1, device="cpu")]
                    )
                    cur_blocked_indices = torch.cat(
                        [cur_blocked_indices, torch.full((pad_len,), -1, device="cpu")]
                    )

                # Mask KV
                perm = torch.randperm(t.topk, device="cpu")
                cur_abs_indices = cur_abs_indices[perm]
                cur_blocked_indices = cur_blocked_indices[perm]

                # Fill it with invalid indices if needed
                if t.is_all_indices_invalid:
                    cur_abs_indices.fill_(-1)
                    cur_blocked_indices.fill_(-1)

                abs_indices[i, j, :] = cur_abs_indices
                indices_in_kvcache[i, j, :] = cur_blocked_indices

        # Mask nonused KV as NaN
        all_indices = indices_in_kvcache.flatten().tolist()
        all_indices = list(set(all_indices))
        if -1 in all_indices:
            all_indices.remove(-1)
        all_indices = torch.tensor(all_indices, dtype=torch.int32, device="cpu")

        blocked_k = blocked_k.view(-1, t.h_kv, t.d)
        nonused_indices_mask = torch.ones(
            blocked_k.size(0) * blocked_k.size(1), dtype=torch.bool, device="cpu"
        )
        nonused_indices_mask[all_indices] = False
        blocked_k[nonused_indices_mask, :, :] = float("nan")
        blocked_k = blocked_k.view(-1, t.block_size, t.h_kv, t.d)

        abs_indices = abs_indices.to(q.device)
        indices_in_kvcache = indices_in_kvcache.to(q.device)

        return cache_seqlens, q, block_table, blocked_k, abs_indices, indices_in_kvcache


def reference_torch(
    cache_seqlens: torch.Tensor,  # [batch_size]
    block_table: torch.Tensor,  # [batch_size, ?]
    q: torch.Tensor,  # [batch_size, s_q, h_q, d]
    blocked_k: torch.Tensor,  # [?, block_size, h_kv, d]
    dv: int,
    scale: float,
    is_causal: bool,
    indices: Optional[torch.Tensor] = None,  # [batch_size, s_q, topk]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A reference implementation in PyTorch
    """

    def get_topk_attn_mask(s_q: int, s_k: int, indices: torch.Tensor):
        mask = torch.zeros(s_q, s_k, dtype=torch.bool)
        for i in range(s_q):
            cur_indices = indices[i]
            valid_indices = cur_indices[cur_indices != -1]
            mask[i, valid_indices] = True
        return mask

    def scaled_dot_product_attention(
        batch_idx: int,
        query: torch.Tensor,  # [h_q, s_q, d]
        kv: torch.Tensor,  # [h_kv, s_k, d]
        dv: int,
        scale: float,
        is_causal,
        indices: Optional[torch.Tensor],  # [s_q, topk]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_q = query.size(0)
        h_kv = kv.size(0)
        s_q = query.shape[-2]
        s_k = kv.shape[-2]
        query = query.float() * scale
        kv = kv.float()
        if h_kv != 1:
            kv = kv.repeat_interleave(h_q // h_kv, dim=0)
        kv[kv != kv] = 0.0
        attn_weight = query @ kv.transpose(-2, -1)  # [h_q, s_q, s_k]
        if (is_causal and query.size(1) > 1) or indices is not None:
            mask = torch.ones(s_q, s_k, dtype=torch.bool)
            if is_causal:
                assert indices is None
                mask = mask.tril(diagonal=s_k - s_q)
            if indices is not None:
                mask &= get_topk_attn_mask(s_q, s_k, indices)
            attn_bias = torch.zeros(s_q, s_k, dtype=torch.float)
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
            attn_weight += attn_bias.to(q.dtype)
        # attn_weight /= math.sqrt(query.size(-1))
        lse = attn_weight.logsumexp(dim=-1)  # [h_q, s_q]
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
        output = attn_weight @ kv[..., :dv]  # [h_q, s_q, dv]
        # Correct for q tokens which has no attendable k
        lonely_q_mask = lse == float("-inf")
        output[lonely_q_mask.unsqueeze(-1).broadcast_to(h_q, s_q, dv)] = 0.0
        lse[lonely_q_mask] = float("+inf")

        return output

    b, s_q, h_q, d = q.size()
    block_size = blocked_k.size(1)
    h_kv = blocked_k.size(2)
    cache_seqlens_cpu = cache_seqlens.cpu()
    out_ref = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
    for i in range(b):
        cur_len = cache_seqlens_cpu[i].item()
        cur_num_blocks = cdiv(cur_len, block_size)
        cur_block_indices = block_table[i][0:cur_num_blocks]
        cur_kv = blocked_k[cur_block_indices].view(-1, h_kv, d)[:cur_len, ...]
        cur_out = scaled_dot_product_attention(
            i,
            q[i].transpose(0, 1),
            cur_kv.transpose(0, 1),
            dv,
            scale,
            is_causal,
            indices[i] if indices is not None else None,
        )
        out_ref[i] = cur_out.transpose(0, 1)
    out_ref = out_ref.to(torch.bfloat16)
    return out_ref


def chunk_input(
    cache_seqlens,
    q,
    block_table,
    blocked_k,
    abs_indices,
    indices_in_kvcache,
    dtype=torch.bfloat16,
):
    q_new = q.reshape(-1, q.shape[2], q.shape[3])
    abs_indices = abs_indices.reshape(-1, abs_indices.shape[2])
    indices_in_kvcache = indices_in_kvcache.reshape(-1, indices_in_kvcache.shape[2])
    max_q_len = q.shape[1]
    max_kv_len = max(cache_seqlens)
    query_lens = [q.shape[1]] * q.shape[0]  # B * [q_len,]
    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device="cuda"
    ).cumsum(dim=0, dtype=torch.int32)
    cache_seqlens = cache_seqlens.to("cuda")
    q_new = q_new.to("cuda")
    block_table = block_table.to("cuda")
    blocked_k = blocked_k.to("cuda")
    abs_indices = abs_indices.to("cuda")
    indices_in_kvcache = indices_in_kvcache.to("cuda")
    return (
        cu_query_lens,
        max_q_len,
        cache_seqlens,
        max_kv_len,
        q_new.to(dtype),
        block_table,
        blocked_k.to(dtype),
        abs_indices,
        indices_in_kvcache,
    )


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("s_q", [1, 64, 177])
@pytest.mark.parametrize("s_k", [1, 64, 177])
@pytest.mark.parametrize("top_k", [64, 78])
@pytest.mark.parametrize("num_q_heads", [16, 32])
@pytest.mark.parametrize("lora_dim", [256, 512])
@pytest.mark.parametrize(
    "rope_dim",
    [
        64,
    ],
)
@pytest.mark.parametrize("block_size", [16, 64])
@torch.inference_mode()
def test_triton_unified_attn(
    batch: int,
    s_q: int,
    s_k: int,
    top_k: int,
    num_q_heads: int,
    lora_dim: int,
    rope_dim: int,
    block_size: int,
) -> None:
    total_dim = lora_dim + rope_dim
    softmax_scale = lora_dim**-0.5

    test_p = Param(
        batch,
        s_q,
        s_k,
        d=total_dim,
        dv=lora_dim,
        h_q=num_q_heads,
        block_size=block_size,
        is_varlen=True,
        is_causal=False,
        is_fp8=False,
        topk=top_k,
        test_performance=False,
    )
    (cache_seqlens, q, block_table, blocked_k, abs_indices, indices_in_kvcache) = (
        generate_test_data(test_p)
    )
    ref_output = reference_torch(
        cache_seqlens,
        block_table,
        q,
        blocked_k,
        lora_dim,
        softmax_scale,
        False,
        abs_indices,
    )

    (
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        q,
        block_table,
        blocked_k,
        abs_indices,
        indices_in_kvcache,
    ) = chunk_input(
        cache_seqlens, q, block_table, blocked_k, abs_indices, indices_in_kvcache
    )

    output = torch.empty((*q.shape[:-1], lora_dim), device=q.device, dtype=q.dtype)

    unified_attention_sparse_mla(
        q,
        blocked_k,
        output,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        softmax_scale,
        indices_in_kvcache,
        block_table,
        lora_dim,
    )

    ref_output = ref_output.to(output.device).to(q.dtype)
    output = output.reshape(ref_output.shape)

    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(
        output, ref_output, atol=atol, rtol=rtol
    ), f"{torch.max(torch.abs(output - ref_output))}"
