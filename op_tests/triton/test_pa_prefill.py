import math
import random
import time
import pytest
import torch
from aiter.ops.triton.pa_prefill import context_attention_fwd

STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)


NUM_HEADS = [64]
NUM_QUERIES_PER_KV = [1, 8, 64]
HEAD_SIZES = [128, 96, 24]
DTYPES = [torch.float16]
CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]
SLIDING_WINDOW = [0, 16, 64, 128, 256, 512, 2048]
KV_CACHE_DTYPES = ["auto", "fp8", "fp8_e5m2"]


def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != total_num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        num_remaining_heads = min(
            closest_power_of_2, total_num_heads - closest_power_of_2
        )
        extra_powers = torch.arange(
            start=1, end=1 + 2 * num_remaining_heads, step=2, dtype=torch.int32
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes

def input_helper(
    BS,
    MAX_SEQ_LEN,
    MAX_CTX_LEN,
    cache_size,
    block_size,
    max_block_per_request,
    num_heads: int,
    head_size: int,
    num_queries_per_kv: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    use_alibi_slope: bool,
):
    seed_everything(0)
    torch.set_default_device(device)

    # Need this, otherwise when we capture the graph the process
    # for GPU 1 would run on both GPU0 and GPU1 and things would hang
    #
    # see also similar issue: https://github.com/Dao-AILab/flash-attention/issues/523
    torch.cuda.set_device(device)

    if use_alibi_slope:
        alibi_slopes = _get_alibi_slopes(num_heads).to(device)

    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    num_kv_heads = num_heads // num_queries_per_kv

    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)

    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    else:
        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    k_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    v_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.long)
    values = values[torch.randperm(cache_size)]
    block_table = values[: BS * max_block_per_request].view(BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.long)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens, dtype=torch.long), dim=0)
    max_input_len = MAX_SEQ_LEN
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_lens[:-1], dtype=torch.long), dim=0
    )
    for i in range(BS):
        for j in range(query_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] + j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] + b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                key[start_loc:end_loc]
            )
            v_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                value[start_loc:end_loc]
            )
            cur_ctx += block_size
            block_id += 1
    # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    k_cache = (
        k_cache.view(-1, block_size, num_kv_heads, head_size // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to V_cache[num_blocks, num_kv_heads, head_size, block_size]
    v_cache = (
        v_cache.view(-1, block_size, num_kv_heads, head_size)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    if use_alibi_slope:
        return query, k, v, output, k_cache, v_cache, block_table, b_start_loc, b_seq_len, max_input_len, k_scale, v_scale, alibi_slopes
    else:
        return query, k, v, output, k_cache, v_cache, block_table, b_start_loc, b_seq_len, max_input_len, k_scale, v_scale, None

# TODO: Implement a test reference and compare it to the Triton output
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOW)
@torch.inference_mode()
def test_contexted_kv_attention(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    sliding_window: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
) -> None:
    query, k, v, output, k_cache, v_cache, block_table, b_start_loc, b_seq_len, max_input_len, k_scale, v_scale, _ = input_helper(
        BS=10,
        MAX_SEQ_LEN=1024,
        MAX_CTX_LEN=1024,
        cache_size=640,
        block_size=32,
        max_block_per_request=64,
        num_heads=num_heads,
        head_size=head_size,
        num_queries_per_kv=num_queries_per_kv,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        device=device,
        use_alibi_slope=False,
    )

    # Warm up the Triton kernel by calling it once before actually measuring
    # generation time
    context_attention_fwd(
        query,
        k,
        v,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        max_input_len,
        k_scale,
        v_scale,
        sliding_window=sliding_window,
    )
    torch.cuda.synchronize()
    start_time = time.time()
    context_attention_fwd(
        query,
        k,
        v,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        max_input_len,
        k_scale,
        v_scale,
        sliding_window=sliding_window,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"triton Time: {(end_time - start_time)*1000:.2f} ms")


# TODO: Implement a test reference and compare it to the Triton output
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_contexted_kv_attention_alibi(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
) -> None:
    query, k, v, output, k_cache, v_cache, block_table, b_start_loc, b_seq_len, max_input_len, k_scale, v_scale, alibi_slopes = input_helper(
        BS=10,
        MAX_SEQ_LEN=1024,
        MAX_CTX_LEN=1024,
        cache_size=640,
        block_size=32,
        max_block_per_request=64,
        num_heads=num_heads,
        head_size=head_size,
        num_queries_per_kv=num_queries_per_kv,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        device=device,
        use_alibi_slope=True,
    )

    # Warm up the Triton kernel by calling it once before actually measuring
    # generation time
    context_attention_fwd(
        query,
        k,
        v,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        max_input_len,
        k_scale,
        v_scale,
        alibi_slopes=alibi_slopes,
    )
    torch.cuda.synchronize()
    start_time = time.time()
    context_attention_fwd(
        query,
        k,
        v,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        max_input_len,
        k_scale,
        v_scale,
        alibi_slopes=alibi_slopes,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"triton Time: {(end_time - start_time)*1000:.2f} ms")
