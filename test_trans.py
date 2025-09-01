import torch
import triton
import triton.language as tl
import pytest
from typing import Optional
# @triton.jit
# def _vllm_layout_trans_kernel(
#     k_buffer_ptr,
#     v_buffer_ptr,
#     k_values_ptr,
#     v_values_ptr,
#     b_query_lens_loc,
#     b_seq_lens_loc,
#     block_table,
#     block_table_stride_0,
#     k_scale,
#     v_scale,
#     output_dtype: tl.constexpr,
#     E_DIM: tl.constexpr,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     batch_idx = tl.program_id(0)
#     block_idx = tl.program_id(1)

#     batch_query_indexes = tl.load(b_query_lens_loc + batch_idx +
#                                     tl.arange(0, 2))
#     batch_query_start, batch_query_end = tl.split(batch_query_indexes)
#     query_len = batch_query_end - batch_query_start

#     if query_len <= 1:
#         return

#     batch_token_indexes = tl.load(b_seq_lens_loc + batch_idx +
#                                     tl.arange(0, 2))
#     batch_token_start, batch_token_end = tl.split(batch_token_indexes)
#     seq_len = batch_token_end - batch_token_start

#     if block_idx * BLOCK_SIZE < seq_len:
#         block_mask = (block_idx * BLOCK_SIZE +
#                         tl.arange(0, BLOCK_SIZE)[:, None]) < seq_len

#         kv_idx = tl.load(block_table + batch_idx * block_table_stride_0 +
#                             block_idx).to(tl.int64)

#         kv_buffer_off = kv_idx * BLOCK_SIZE * E_DIM + tl.arange(
#             0, BLOCK_SIZE)[:, None] * E_DIM + tl.arange(0, E_DIM)[None, :]
#         k_vals = tl.load(k_buffer_ptr + kv_buffer_off,
#                             mask=block_mask,
#                             other=0.0)
#         k_vals = k_vals.to(output_dtype)

#         v_vals = tl.load(v_buffer_ptr + kv_buffer_off,
#                             mask=block_mask,
#                             other=0.0)
#         v_vals = v_vals.to(output_dtype)
#         kv_values_off = batch_token_start * E_DIM + \
#             block_idx * BLOCK_SIZE * E_DIM + \
#             tl.arange(0, BLOCK_SIZE)[:, None] * E_DIM + \
#             tl.arange(0, E_DIM)[None, :]
#         tl.store(k_values_ptr + kv_values_off, k_vals, mask=block_mask)
#         tl.store(v_values_ptr + kv_values_off, v_vals, mask=block_mask)

# def vllm_layout_trans(b_query_lens_loc, b_seq_lens_loc, block_table,
#                         k_cache, v_cache, max_seq_len, k_scale, v_scale,
#                         output_dtype, total_tokens):
#     H_KV = v_cache.shape[2]
#     D = v_cache.shape[3]
#     BLOCK_SIZE = v_cache.shape[1]

#     k_values = torch.empty(
#         (total_tokens, H_KV, D),
#         dtype=output_dtype,
#         device=k_cache.device,
#     )
#     v_values = torch.empty(
#         (total_tokens, H_KV, D),
#         dtype=output_dtype,
#         device=v_cache.device,
#     )

#     grid = (block_table.shape[0],
#             (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

#     if output_dtype == torch.float16:
#         output_dtype = tl.float16
#     elif output_dtype == torch.bfloat16:
#         output_dtype = tl.bfloat16
#     else:
#         raise ValueError(f"Unsupported output dtype: {output_dtype}")

#     for i in range(100):
#         _vllm_layout_trans_kernel[grid](k_cache,
#                                         v_cache,
#                                         k_values,
#                                         v_values,
#                                         b_query_lens_loc,
#                                         b_seq_lens_loc,
#                                         block_table,
#                                         block_table.stride(0),
#                                         k_scale,
#                                         v_scale,
#                                         output_dtype=output_dtype,
#                                         E_DIM=H_KV * D,
#                                         BLOCK_SIZE=BLOCK_SIZE)

#     return k_values, v_values  

@triton.jit
def _vllm_layout_trans_kernel2(
    k_buffer_ptr,
    v_buffer_ptr,
    k_values_ptr,
    v_values_ptr,
    b_seq_lens_loc,
    block_table,
    block_table_stride_0,
    X: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    batch_token_indexes = tl.load(b_seq_lens_loc + batch_idx +
                                    tl.arange(0, 2))
    batch_token_start, batch_token_end = tl.split(batch_token_indexes)
    seq_len = batch_token_end - batch_token_start

    DIM0: tl.constexpr = H_KV * D // X
    DIM1: tl.constexpr = X * BLOCK_SIZE
    E_DIM: tl.constexpr = H_KV * D
    if block_idx * BLOCK_SIZE < seq_len:
        # print("block_idx", block_idx)
        k_block_mask = (block_idx * BLOCK_SIZE +
                        tl.arange(0, BLOCK_SIZE)[None, :, None]) < seq_len
        v_block_mask = (block_idx * BLOCK_SIZE +
                        tl.arange(0, BLOCK_SIZE)[None, :]) < seq_len

        kv_idx = tl.load(block_table + batch_idx * block_table_stride_0 +
                            block_idx)

        k_buffer_off = kv_idx * BLOCK_SIZE * E_DIM + tl.arange(
            0, DIM0)[:, None, None] * DIM1 + tl.arange(
                0, BLOCK_SIZE)[None, :, None] * X + tl.arange(
                    0, X)[None, None, :]
        v_buffer_off = kv_idx * BLOCK_SIZE * E_DIM + tl.arange(
            0, E_DIM)[:, None] * BLOCK_SIZE + tl.arange(
                0, BLOCK_SIZE)[None, :]
        k_vals = tl.load(k_buffer_ptr + k_buffer_off,
                            mask=k_block_mask,
                            other=0.0)
        v_vals = tl.load(v_buffer_ptr + v_buffer_off,
                            mask=v_block_mask,
                            other=0.0)
        k_vals = k_vals.trans(0, 2, 1).view(E_DIM, BLOCK_SIZE)
        block_mask = (block_idx * BLOCK_SIZE +
                        tl.arange(0, BLOCK_SIZE)[:, None]) < seq_len

        kv_values_off = batch_token_start * E_DIM + \
            block_idx * BLOCK_SIZE * E_DIM + tl.arange(
            0, BLOCK_SIZE)[:, None] * E_DIM + tl.arange(0, E_DIM)[None, :]
        tl.store(k_values_ptr + kv_values_off, k_vals.T, mask=block_mask)
        tl.store(v_values_ptr + kv_values_off, v_vals.T, mask=block_mask)

def vllm_layout_trans2(b_seq_lens_loc, block_table, k_cache, v_cache,
                        max_seqlen, total_tokens):
    H_KV = v_cache.shape[1]
    D = v_cache.shape[2]
    BLOCK_SIZE = v_cache.shape[3]
    X = k_cache.shape[-1]
    dtype = v_cache.dtype

    k_values = torch.empty((total_tokens, H_KV, D),
                            dtype=dtype,
                            device="cuda")
    v_values = torch.empty((total_tokens, H_KV, D),
                            dtype=dtype,
                            device="cuda")

    grid = (block_table.shape[0],
            (max_seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE)

    _vllm_layout_trans_kernel2[grid](
        k_cache,
        v_cache,
        k_values,
        v_values,
        b_seq_lens_loc,
        block_table,
        block_table.stride(0),
        X=X,
        H_KV=H_KV,
        D=D,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=1,
        num_warps=4,
    )
    for i in range(100):
        _vllm_layout_trans_kernel2[grid](
            k_cache,
            v_cache,
            k_values,
            v_values,
            b_seq_lens_loc,
            block_table,
            block_table.stride(0),
            X=X,
            H_KV=H_KV,
            D=D,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=1,
            num_warps=4,
        )
    return k_values, v_values

def ref_trans(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_lens: list[int],
    block_tables: torch.Tensor,
    k_scale: float,
    v_scale: float,
) -> torch.Tensor:
    num_seqs = len(kv_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    # import pdb; pdb.set_trace()
    keys = []
    values = []
    for i in range(num_seqs):
        kv_len = kv_lens[i]

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len] * k_scale
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len] * v_scale
        keys.append(k)
        values.append(v)

    return torch.cat(keys, dim=0), torch.cat(values, dim=0)

NUM_HEADS = [(4, 4), (8, 2)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16]
DTYPES = [torch.bfloat16]
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768, 2048]
# Test cases covering different configurations
@pytest.mark.parametrize("seq_lens",
                         [[(10, 1328), (5, 18),
                           (129, 463)], [(8, 523), (24, 37), (3, 2011)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@torch.inference_mode()
def test_varlen_with_paged_kv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    num_blocks: int,
) -> None:
    
    torch.set_default_device("cuda")
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    total_tokens =  sum(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)

    print(f"total_tokens: {total_tokens * head_size * num_kv_heads * 2 / 1024 / 1024} MB")
    key_cache = torch.randn(num_blocks,
                            block_size,
                            num_kv_heads,
                            head_size,
                            dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    key_cache_vllm = key_cache.view(num_blocks, block_size, num_kv_heads, head_size // 8, 8).permute(0, 2, 3, 1, 4).contiguous()
    value_cache_vllm = value_cache.permute(0, 2, 3, 1).contiguous()
    cu_query_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32).cumsum(dim=0,
                                                           dtype=torch.int32)

    cu_seq_lens = torch.tensor([0] + kv_lens,
                               dtype=torch.int32).cumsum(dim=0,
                                                         dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 num_blocks,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)


    k_scale = (1)
    v_scale = (1)
    k_scalev = torch.tensor(k_scale, dtype=torch.float32)
    v_scalev = torch.tensor(v_scale, dtype=torch.float32)
    
    k_values, v_values = vllm_layout_trans2(
        # b_query_lens_loc=cu_query_lens,
        b_seq_lens_loc=cu_seq_lens,
        block_table=block_tables,
        k_cache=key_cache_vllm,
        v_cache=value_cache_vllm,
        max_seqlen=max_kv_len,  # Maximum sequence length in batch
        # k_scale=k_scalev,
        # v_scale=v_scalev,
        # output_dtype=dtype,
        total_tokens=total_tokens
    )
    
    # k_values1, v_values1 = vllm_layout_trans(
    #     b_query_lens_loc=cu_query_lens,
    #     b_seq_lens_loc=cu_seq_lens,
    #     block_table=block_tables,
    #     k_cache=key_cache,
    #     v_cache=value_cache,
    #     max_seq_len=max_kv_len,  # Maximum sequence length in batch
    #     k_scale=k_scalev,
    #     v_scale=v_scalev,
    #     output_dtype=dtype,
    #     total_tokens=total_tokens
    # )
    k_ref, v_ref = ref_trans(
        key_cache=key_cache,
        value_cache=value_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.testing.assert_close(k_values, k_ref, rtol=1e-3, atol=1e-2), f"{torch.max(torch.abs(k_ref - k_values))}"
    torch.testing.assert_close(v_values, v_ref, rtol=1e-3, atol=1e-2), f"{torch.max(torch.abs(v_ref - v_values))}"
    # torch.testing.assert_close(k_values1, k_ref, rtol=1e-3, atol=1e-2), f"{torch.max(torch.abs(k_ref - k_values))}"
    # torch.testing.assert_close(v_values1, v_ref, rtol=1e-3, atol=1e-2), f"{torch.max(torch.abs(v_ref - v_values))}"
    # torch.allclose(k_values, k_ref, rtol=1e-3, atol=1e-2)
    # torch.allclose(v_values, v_ref, rtol=1e-3, atol=1e-2)


test_varlen_with_paged_kv(seq_lens=[(10, 1328), (5, 1800), (129, 4630), (129, 1300), (129, 4096)],
                          num_heads=(4, 4),
                          head_size=128,
                          dtype=torch.bfloat16,
                          block_size=16,
                          num_blocks=32768)