import triton
import triton.language as tl
import torch
import pytest
import random
from aiter.ops.triton.pa_decode import paged_attention_decode
from aiter import pertoken_quant

DEBUG_MODE = False


def paged_attention_decode_ref(
    output,  # [num_seqs, num_q_heads, head_sz]
    query,  # [num_seqs, num_kv_heads, head_sz],
    k_cache,  # [num_seqs, num_kv_heads, head_sz/x, blk_sz, x]
    v_cache,  # [num_seqs, num_kv_heads, head_sz, blk_sz]
    blk_tables,  # [num_seq, max_num_blks_per_seq]
    ctx_lens,  # [num_seqs]
) -> None:
    num_q_heads = query.shape[1]
    num_kv_heads = v_cache.shape[1]
    q_grp_sz = num_q_heads // num_kv_heads
    head_sz = v_cache.shape[2]
    kv_blk_sz = v_cache.shape[3]

    num_seqs = query.shape[0]
    for s in range(num_seqs):
        q = query[s].unsqueeze(0)
        blk_tbl = blk_tables[s]
        ctx_len = ctx_lens[s]

        keys = []
        values = []
        for j in range(ctx_len):
            blk_number = int(blk_tbl[j // kv_blk_sz])
            blk_offset = j % kv_blk_sz

            k = k_cache[blk_number, :, :, blk_offset, :]
            k = k.reshape(num_kv_heads, head_sz)
            if q_grp_sz != 1:
                k = k.repeat_interleave(q_grp_sz, 0)
            keys.append(k)

            v = v_cache[blk_number, :, :, blk_offset]
            if q_grp_sz != 1:
                v = v.repeat_interleave(q_grp_sz, 0)
            values.append(v)

        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)

        scale = 1.0 / (head_sz**0.5)
        q = q * scale
        attn = torch.einsum("qhd,khd->hqk", q, keys)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("hqk, khd->qhd", attn, values)
        out = out.view(num_q_heads, head_sz)
        output[s].copy_(out, non_blocking=True)

tl_to_torch_dtype = {tl.bfloat16: torch.bfloat16, tl.float16: torch.float16}

def input_helper(B, H_Q, H_KV, D, KV_BLK_SZ, SEQ_LEN, dtype, kv_cache_dtype, output_type, num_blocks=4):
    """Helper function to generate input tensors for paged attention testing."""
    # Query tensor generation
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        query = torch.randn(
            B, H_Q, D, dtype=torch.float16, device="cuda"
        )  # assumption dtype is 8bits or lower
        query = query.to(dtype=dtype, device="cuda")
    else:
        query = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

    if kv_cache_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        x = min(D, 16 // torch.tensor([], dtype=torch.float16).element_size())
        key_cache = torch.randn(
            num_blocks, H_KV, D // x, KV_BLK_SZ, x, dtype=torch.float16, device="cuda"
        )
        value_cache = torch.randn(
            num_blocks, H_KV, D, KV_BLK_SZ, dtype=torch.float16, device="cuda"
        )
        key_cache = torch.clamp(key_cache, min=1e-3) #For FP8 case, this is needed to prevent NANs
        value_cache = torch.clamp(value_cache, min=1e-3) #For FP8 case, this is needed to prevent NANs

        # torch doesn't have randn for fp8 data type, so we convert here
        key_cache = key_cache.to(dtype=kv_cache_dtype)
        value_cache = value_cache.to(dtype=kv_cache_dtype)
    else:
        x = min(D, 16 // torch.tensor([], dtype=kv_cache_dtype).element_size())
        key_cache = torch.randn(
            num_blocks, H_KV, D // x, KV_BLK_SZ, x, dtype=kv_cache_dtype, device="cuda"
        )
        value_cache = torch.randn(
            num_blocks, H_KV, D, KV_BLK_SZ, dtype=kv_cache_dtype, device="cuda"
        )
        key_cache = torch.clamp(key_cache, min=1e-3) #For FP8 case, this is needed to prevent NANs
        value_cache = torch.clamp(value_cache, min=1e-3) #For FP8 case, this is needed to prevent NANs

    key_cache_tri = key_cache.permute(0, 1, 3, 2, 4).flatten(3, 4).contiguous().cuda()
    value_cache_tri = value_cache.permute(0, 1, 3, 2).contiguous().cuda()

    context_lens = torch.full((B,), SEQ_LEN, device="cuda")
    max_context_len = max(context_lens)
    max_num_blks_per_seq = (max_context_len + KV_BLK_SZ - 1) // KV_BLK_SZ

    block_tables = []
    for i in range(B):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int32, device="cuda")

    output = torch.zeros(B, H_Q, D, dtype=output_type, device="cuda")

    return query, output, key_cache, value_cache, key_cache_tri, value_cache_tri, context_lens, block_tables, max_context_len

@pytest.mark.parametrize("B", [1, 4, 27])
@pytest.mark.parametrize("H_Q, H_KV", [(1,1), (16, 16), (24,4)])
@pytest.mark.parametrize("D", [1, 64, 128])
@pytest.mark.parametrize("KV_BLK_SZ", [1, 4, 16])
@pytest.mark.parametrize("SEQ_LEN", [1, 57, 10000])
@pytest.mark.parametrize("NUM_BLK", [4, 16])
# q_dtype, kv_dtype, compute_type, output_type
# INT8xINT8 -> BF16-> BF16
# FP8xFP8 -> BF16-> FP8
# BF16xINT8 -> BF16-> BF16
# BF16xFP8 -> BF16-> BF16
# BF16xBF16->BF16->BF16
# FP16xFP16->FP16->FP16
@pytest.mark.parametrize(
    "dtype, kv_cache_dtype, compute_type, output_type",
    [
        (torch.float16, torch.float16, tl.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16, tl.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float8_e4m3fnuz, tl.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.int8, tl.bfloat16, torch.bfloat16),
        (torch.float8_e4m3fnuz, torch.float8_e4m3fnuz, tl.bfloat16, torch.bfloat16),
        (torch.int8, torch.int8, tl.bfloat16, torch.bfloat16),
    ],
)

def test_paged_attn(
    B,
    H_Q,
    H_KV,
    D,
    KV_BLK_SZ,
    SEQ_LEN,
    NUM_BLK,
    dtype,
    kv_cache_dtype,
    compute_type,
    output_type,
):

    if SEQ_LEN >= 8192 and B >=16:
        pytest.skip("B>={4} and SEQ_LEN>={8192} tests are too slow")
    torch.set_printoptions(threshold=100000)
    num_blocks = NUM_BLK

    query, triton_output, key_cache, value_cache, key_cache_tri, value_cache_tri, context_lens, block_tables, max_context_len \
        = input_helper(B, H_Q, H_KV, D, KV_BLK_SZ, SEQ_LEN, dtype, kv_cache_dtype, output_type, num_blocks)

    attn_scale = 1.0 / (D**0.5)

    paged_attention_decode(
        triton_output,
        query,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        attn_scale,
        max_context_len,
        compute_type,
        k_scale=torch.tensor([1.0]),
        v_scale=torch.tensor([1.0]),
        num_seq_partitions=0,
        alibi_slopes=None
    )

    # torch doesn't have support for fp8 data type, so we convert here
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        query = query.to(tl_to_torch_dtype[compute_type])

    if kv_cache_dtype not in (torch.bfloat16, torch.float16):
        key_cache = key_cache.to(dtype=tl_to_torch_dtype[compute_type])
        value_cache = value_cache.to(dtype=tl_to_torch_dtype[compute_type])
    torch_output = torch.zeros(B, H_Q, D, dtype=output_type, device="cuda")
    paged_attention_decode_ref(
        torch_output, query, key_cache, value_cache, block_tables, context_lens
    )

    triton.testing.assert_close(triton_output, torch_output, rtol=1e-02, atol=1e-02)


@pytest.mark.parametrize("B", [1, 4, 57, 64])
#@pytest.mark.parametrize("H_Q, H_KV", [(1,1), (16, 16), (2,1), (24,4)]) #TODO: GQA failing
@pytest.mark.parametrize("H_Q, H_KV", [(1,1), (16, 16)])
@pytest.mark.parametrize("D", [1, 64, 128])
@pytest.mark.parametrize("KV_BLK_SZ", [1, 4, 512])
@pytest.mark.parametrize("SEQ_LEN", [1, 32, 57, 512, 10000])
@pytest.mark.parametrize("NUM_BLK", [4, 32])
@pytest.mark.parametrize(
    "dtype, kv_cache_dtype, compute_type, output_type",
    [
        (torch.float16, torch.float16, tl.float16, torch.float16),
        #(torch.bfloat16, torch.bfloat16, tl.bfloat16, torch.bfloat16),
    ],
)
def test_paged_attn_per_token_quant(
    B,
    H_Q,
    H_KV,
    D,
    KV_BLK_SZ,
    SEQ_LEN,
    NUM_BLK,
    dtype,
    kv_cache_dtype,
    compute_type,
    output_type,
):
    torch.set_printoptions(precision=5, threshold=10000)
    if D == 128 and KV_BLK_SZ == 512: #Causes Shared Memory out of resources on Mi300
        pytest.skip("D={128} and KV_BLK_SZ={512} causes shared memory out of resources")

    if SEQ_LEN >= 8192 and B >=16:
        pytest.skip("B>={4} and SEQ_LEN>={8192} tests are too slow")

    num_blocks =  NUM_BLK

    query, triton_output, key_cache, value_cache, key_cache_tri, value_cache_tri, context_lens, block_tables, max_context_len \
        = input_helper(B, H_Q, H_KV, D, KV_BLK_SZ, SEQ_LEN, dtype, kv_cache_dtype, output_type, num_blocks)

    attn_scale = 1.0 / (D**0.5)

    key_cache_tri_quant, k_scale, = pertoken_quant(key_cache_tri, scale_dtype=torch.float32, quant_dtype=torch.float8_e4m3fnuz)
    value_cache_tri_quant, v_scale, = pertoken_quant(value_cache_tri, scale_dtype=torch.float32, quant_dtype=torch.float8_e4m3fnuz)

    paged_attention_decode(
        triton_output,
        query,
        key_cache_tri_quant,
        value_cache_tri_quant,
        context_lens,
        block_tables,
        attn_scale,
        max_context_len,
        compute_type,
        k_scale=k_scale,
        v_scale=v_scale,
        num_seq_partitions=0,
        alibi_slopes=None
    )

    if DEBUG_MODE:
        print(f"B={B} H_Q={H_Q}, H_KV={H_KV} D={D}, KV_BLK_SZ={KV_BLK_SZ}, SEQ_LEN={SEQ_LEN}, NUM_BLK={NUM_BLK}")
        print(f"query={query}")
        print(f"key_cache_tri.shape={key_cache_tri.shape} key_cache_tri={key_cache_tri}")
        print(f"k_scale.shape={k_scale.shape} k_scale={k_scale}")
        print(f"key_cache_tri_quant.shape={key_cache_tri_quant.shape} key_cache_tri_quant={key_cache_tri_quant}")
        print(f"v_scale.shape={v_scale.shape} v_scale={v_scale}")
        print(f"value_cache_tri.shape={value_cache_tri.shape} value_cache_tri={value_cache_tri}")
        print(f"value_cache_tri_quant.shape={value_cache_tri_quant.shape} value_cache_tri_quant={value_cache_tri_quant}")
        print(f"triton_output={triton_output}")
    # torch doesn't have support for fp8 data type, so we convert here
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        query = query.to(tl_to_torch_dtype[compute_type])

    if kv_cache_dtype not in (torch.bfloat16, torch.float16):
        key_cache = key_cache.to(dtype=tl_to_torch_dtype[compute_type])
        value_cache = value_cache.to(dtype=tl_to_torch_dtype[compute_type])
    torch_output = torch.zeros(B, H_Q, D, dtype=output_type, device="cuda")
    paged_attention_decode_ref(
        torch_output, query, key_cache, value_cache, block_tables, context_lens
    )
    if DEBUG_MODE:
        print(f"torch_output={torch_output}")

    triton.testing.assert_close(triton_output, torch_output, rtol=2.5e-1, atol=2.5e-1)
