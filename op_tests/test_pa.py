import random
from typing import List, Optional, Tuple, Union

import torch
import ater
from ater import paged_attn as ops
from test_common import checkAllclose, perftest, tensor_dump, tensor_load

uniform_range = (0.5, 1)
STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}


def get_kv_cache_torch_dtype(
        cache_dtype: Optional[Union[str, torch.dtype]],
        model_dtype: Optional[Union[str, torch.dtype]] = None) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str):
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in ["half", "bfloat16", "float"]:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        elif cache_dtype == "fp8":
            torch_dtype = torch.uint8
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    return torch_dtype


def kv_cache_factory(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=torch_dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape,
                                dtype=torch_dtype,
                                device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(*uniform_range)
        else:
            raise ValueError(
                f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape,
                                  dtype=torch_dtype,
                                  device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(*uniform_range)
        else:
            raise ValueError(
                f"Does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)
    return key_caches, value_caches


FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
MAX_SEQ_LEN = 65536
# There may not be enough gpu memory due to large NUM_BLOCKS.
# Reduce NUM_BLOCKS when it happens.
NUM_BLOCKS = 32768  # Arbitrary values for testing
PARTITION_SIZE = 512
# flshattF and tritonflashattF supported: {torch.float16, torch.bfloat16}
DTYPES = [torch.half, torch.bfloat16]
NUM_GEN_SEQS = [7]  # Arbitrary values for testing
NUM_PREFILL_SEQS = [3]  # Arbitrary values for testing
NUM_HEADS = [(40, 40), (64, 8)]  # Arbitrary values for testing

# FlashAttention forward only supports head dimension at most 128
# https://github.com/ROCmSoftwarePlatform/flash-attention/blob/3d2b6f5d037782cc2c906909a46fb7e2e1b48b25/csrc/flash_attn_rocm/flash_api.cpp#L62
HEAD_SIZES = [64, 80, 96, 112, 120, 128, 192, 256]

BLOCK_SIZES = [16, 32]
USE_ALIBI = [False, True]
KV_CACHE_DTYPE = ["auto", "fp8"]
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


# @perftest()
def run_native(query,
               key_cache,
               value_cache,
               block_tables,
               seq_lens,
               max_seq_len,
               kv_cache_dtype,
               num_kv_heads,
               scale,
               alibi_slopes,
               k_scale,
               v_scale,
               num_queries_per_kv):
    output = torch.zeros_like(query)
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables_lst = block_tables.cpu().tolist()
    seq_lens_lst = seq_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables_lst[i]
        seq_len = int(seq_lens_lst[i])

        keys_lst: List[torch.Tensor] = []
        values_lst: List[torch.Tensor] = []
        for j in range(seq_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys_lst.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values_lst.append(v)
        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(seq_len).int()
            alibi_bias = (position_ids - seq_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(
                1, 1, -1)

        out = ref_masked_attention(q, keys, values, scale, alibi_bias)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)
    return output, 1


@perftest()
def run_ater(query,
             key_cache,
             value_cache,
             block_tables,
             seq_lens,
             max_seq_len,
             kv_cache_dtype,
             num_kv_heads,
             scale,
             alibi_slopes,
             k_scale,
             v_scale,):
    return ops.PagedAttention.forward_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        max_seq_len,
        kv_cache_dtype,
        num_kv_heads,
        scale,
        alibi_slopes,
        k_scale,
        v_scale,
    )


@perftest()
def run_ater_naive(query,
                   key_cache,
                   value_cache,
                   block_tables,
                   seq_lens,
                   max_seq_len,
                   kv_cache_dtype,
                   num_kv_heads,
                   scale,
                   alibi_slopes,
                   k_scale,
                   v_scale,
                   block_size):
    return ater.pa_fwd_naive(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        max_seq_len,
        num_kv_heads,
        scale,
        k_scale,
        v_scale,
        block_size
    )


def dump_input(query: torch.Tensor,
               key_cache: torch.Tensor,
               value_cache: torch.Tensor,
               block_tables: torch.Tensor,
               seq_lens: torch.Tensor,
               max_seq_len: int,
               kv_cache_dtype: str,
               num_kv_heads: int,
               scale: float,
               alibi_slopes: Optional[torch.Tensor],
               k_scale: float,
               v_scale: float,):
    tensor_dump(query, 'Q')
    # qbk = tensor_load('Q.bin')
    # checkAllclose(query, qbk)
    tensor_dump(key_cache, 'K_cache')
    tensor_dump(value_cache, 'V_cache')
    tensor_dump(block_tables, 'block_tables')
    tensor_dump(seq_lens, 'seq_lens')


def load_input():
    return tensor_load('Q.bin'), tensor_load('K_cache.bin'), tensor_load('V_cache.bin'), tensor_load('block_tables.bin'), tensor_load('seq_lens.bin'),


debug_mode = True
# debug_mode = False


def test_paged_attention(
    ctx_lens: int,
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
    device: str,
    w8a16=False,
) -> None:
    torch.set_default_device(device)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
    query.uniform_(*uniform_range)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=torch.float)

    # seq_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
    seq_lens = [ctx_lens for _ in range(num_seqs)]
    max_seq_len = max(seq_lens)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int)

    # Create the block tables.
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = max_num_blocks_per_seq*num_seqs
    block_tables_lst: List[List[int]] = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, num_blocks - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables_lst.append(block_table)

    block_tables = torch.tensor(block_tables_lst, dtype=torch.int)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(num_blocks, block_size, 1,
                                                num_kv_heads, head_size,
                                                kv_cache_dtype, dtype, seed,
                                                device)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Using default kv_scale
    k_scale = v_scale = 1.0

    out_ater, time_ater = run_ater(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        max_seq_len,
        kv_cache_dtype,
        num_kv_heads,
        scale,
        alibi_slopes,
        k_scale,
        v_scale,
    )
    out_ater_naive, time_ater_naive = run_ater_naive(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        max_seq_len,
        kv_cache_dtype,
        num_kv_heads,
        scale,
        alibi_slopes,
        k_scale,
        v_scale,
        block_size
    )
    checkAllclose(out_ater_naive, out_ater)

    if w8a16:
        # [num_blocks, num_kv_heads, head_size/x, block_size, x]
        #   0,         1                 2              3     4
        k16 = key_cache.permute(1, 2, 4, 0, 3).contiguous()
        k16 = k16.view(num_kv_heads, head_size, num_blocks*block_size)
        k8, K_qscale = ater.smoothquant_fwd_native(k16, torch.float)
        k8 = k8.view(num_kv_heads,   # 0
                     head_size//16,  # 1
                     16,             # 2
                     num_blocks,     # 3
                     block_size)     # 4
        k8 = k8.permute(3, 0, 1, 4, 2).contiguous()  # k8 for w8 pa

        # [num_blocks, num_kv_heads, head_size, block_size]
        #        0          1              2           3
        v16 = value_cache.permute(1, 2, 0, 3).contiguous()
        v16 = v16.view(num_kv_heads, head_size, num_blocks*block_size)
        v8, V_qscale = ater.smoothquant_fwd_native(v16, torch.float)
        v8 = v8.view(num_kv_heads,  # 0
                     head_size,     # 1
                     num_blocks,    # 2
                     block_size)    # 3
        v8 = v8.permute(2, 0, 1, 3).contiguous()  # v8 for w8 pa
    if debug_mode:
        if w8a16:
            dump_input(query,
                       key_cache,
                       value_cache,
                       block_tables,
                       seq_lens,
                       max_seq_len,
                       kv_cache_dtype,
                       num_kv_heads,
                       scale,
                       alibi_slopes,
                       k_scale,
                       v_scale,)
            tensor_dump(k8, 'K_8')
            tensor_dump(v8, 'V_8')
            tensor_dump(K_qscale, 'K_qscale')
            tensor_dump(V_qscale, 'V_qscale')
        else:
            dump_input(query,
                       key_cache,
                       value_cache,
                       block_tables,
                       seq_lens,
                       max_seq_len,
                       kv_cache_dtype,
                       num_kv_heads,
                       scale,
                       alibi_slopes,
                       k_scale,
                       v_scale,)

    # if w8a16:
    #     # todo remove when w8a16 is ready
    #     out_ater, time_ater = run_ater(
    #         query,
    #         k8.to(dtype),
    #         v8.to(dtype),
    #         block_tables,
    #         seq_lens,
    #         max_seq_len,
    #         kv_cache_dtype,
    #         num_kv_heads,
    #         scale,
    #         alibi_slopes,
    #         k_scale,
    #         v_scale,
    #     )
    if debug_mode:
        tensor_dump(out_ater, 'out')
        (query,
         key_cache,
         value_cache,
         block_tables,
         seq_lens) = load_input()
    out_native, time_native = run_native(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        max_seq_len,
        kv_cache_dtype,
        num_kv_heads,
        scale,
        alibi_slopes,
        k_scale,
        v_scale,
        num_queries_per_kv,
    )

    atol, rtol = 1e-2, 1e-5
    msg = f"[perf] dim: {str((num_seqs, num_heads, head_size)):<20}, dtype: {dtype}, {time_native=:<8.2f} us, {time_ater=:<8.2f} us, uplift: {time_native/time_ater-1:<5.1%}"
    checkAllclose(out_native, out_ater, atol=atol, rtol=rtol, msg=msg)


# test_paged_attention( 128, (8,1), 128, False, 16, torch.half, "auto", 0, "cuda:0")
# test_paged_attention( 128, (8,1), 128, False, 32, torch.bfloat16, "auto", 0, "cuda:0")
test_paged_attention(4096, 128, (8, 1), 128, False, 16,
                     torch.bfloat16, "auto", 0, "cuda:0")
# # simple version
# test_paged_attention(4096, 2, (8, 1), 128, False, 16,
#                      torch.bfloat16, "auto", 0, "cuda:0", w8a16=True)
