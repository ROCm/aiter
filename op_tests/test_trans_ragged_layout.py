import torch
import random
import triton
import triton.language as tl
from aiter import ragged_layout_trans

GLOBAL_BLOCK_SIZE = 512


# copy from SGLang
@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = GLOBAL_BLOCK_SIZE,
):
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        # index into req_to_token_ptr needs to be int64
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)


@triton.jit
def read_sequential_cache(
    k_cache,
    v_cache,
    addrs,
    seqlens,
    n_elements,
    k_buf,
    v_buf,
    BLOCK_SIZE: tl.constexpr = GLOBAL_BLOCK_SIZE,
):
    pid = tl.program_id(axis=0)

    addr = tl.load(addrs + pid)
    seqlen = tl.load(seqlens + pid)
    repeats = tl.cdiv(seqlen, BLOCK_SIZE)
    for i in range(repeats):
        idx = tl.arange(0, BLOCK_SIZE).to(tl.int32) + i * BLOCK_SIZE
        mask = idx < seqlen
        offset = addr + idx * n_elements
        for j in range(n_elements):
            data = tl.load(k_cache + offset + j, mask=mask)
            tl.store(k_buf + offset + j, data, mask=mask)
            data = tl.load(v_cache + offset + j, mask=mask)
            tl.store(v_buf + offset + j, data, mask=mask)

# sequential cache functions
def create_sequential_cache(seqlen, nheads, hdim, dtype):
    return torch.rand((seqlen, nheads, hdim), dtype=dtype, device="cuda")


def get_sequential_addrs(cu_lens, idx):
    return torch.tensor(
        [x for x in range(cu_lens[idx], cu_lens[idx + 1])],
        dtype=torch.int32,
        device="cuda",
    )

def test_ragged_layout_trans(bs, max_seq_len, nheads, hdim, dtype):
    seq_lens_ptr = torch.tensor(
        [random.randint(1, max_seq_len) for i in range(bs)],
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens_sum = seq_lens_ptr.sum()

    stride = triton.next_power_of_2(max_seq_len)

    print(
        f"bs = {bs}, max seqlen = {max_seq_len}, nheads = {nheads}, hdim = {hdim}, stride = {stride}"
    )
    # print(f'seq_lens_ptr = {seq_lens_ptr}')
    # print(f'seq_lens_sum = {seq_lens_sum}')

    zero_start = torch.tensor([0], dtype=torch.int32, device="cuda")
    cumsums = torch.cumsum(seq_lens_ptr, dim=0)
    kv_indptr = torch.cat((zero_start, cumsums), dim=0)

    # print(f'kv_indptr = {kv_indptr}')

    tokens_ptr = torch.empty(stride * bs, dtype=torch.int64, device="cuda")

    k_cache = create_sequential_cache(seq_lens_sum, nheads, hdim, dtype)
    v_cache = create_sequential_cache(seq_lens_sum, nheads, hdim, dtype)

    for i in range(bs):
        seq = get_sequential_addrs(kv_indptr, i)
        offset = stride * i
        seqlen = seq_lens_ptr[i]
        tokens_ptr[offset : offset + seqlen] = seq

    req_pool_indices_ptr = torch.tensor(
        [i for i in range(bs)], dtype=torch.int32, device="cuda"
    )

    # print(f'tokens_ptr = {tokens_ptr}')
    # print(f'req_pool_indices_ptr = {req_pool_indices_ptr}')

    kv_indices = torch.empty(seq_lens_sum + 256, dtype=torch.int32, device="cuda")

    create_flashinfer_kv_indices_triton[(bs,)](
        tokens_ptr,
        req_pool_indices_ptr,
        seq_lens_ptr,
        kv_indptr,
        None,
        kv_indices,
        stride,
    )

    # read cache directly
    k_ref = torch.empty(seq_lens_sum * nheads * hdim, dtype=dtype, device="cuda")

    v_ref = torch.empty(seq_lens_sum * nheads * hdim, dtype=dtype, device="cuda")

    n_elements = nheads * hdim
    read_sequential_cache[(bs,)](
        k_cache,
        v_cache,
        kv_indptr[:-1] * n_elements,
        seq_lens_ptr,
        n_elements,
        k_ref,
        v_ref,
    )

    k_ref = k_ref.view(seq_lens_sum, nheads, hdim)
    v_ref = v_ref.view(seq_lens_sum, nheads, hdim)

    k, v = ragged_layout_trans(
        kv_indptr,
        kv_indices,
        k_cache.view(-1, nheads, hdim),
        v_cache.view(-1, nheads, hdim),
    )

    k = k.view(seq_lens_sum, nheads, hdim)
    v = v.view(seq_lens_sum, nheads, hdim)

    if torch.equal(k, k_ref):
        print("K values verified.")
    else:
        print(f"valid k ? {torch.equal(k, k_cache)}")
        print(f"valid k_ref ? {torch.equal(k_ref, k_cache)}")
        if not torch.equal(k, k_cache):
            torch.testing.assert_close(k, k_cache, rtol=1e-2, atol=1e-2)
        if not torch.equal(k_ref, k_cache):
            torch.testing.assert_close(k_ref, k_cache, rtol=1e-2, atol=1e-2)
        assert False

    if torch.equal(v, v_ref):
        print("V values verified.")
    else:
        print(f"valid v ? {torch.equal(v, v_cache)}")
        print(f"valid v_ref ? {torch.equal(v_ref, v_cache)}")
        if not torch.equal(v, v_cache):
            torch.testing.assert_close(v, v_cache, rtol=1e-2, atol=1e-2)
        if not torch.equal(v_ref, v_cache):
            torch.testing.assert_close(v_ref, v_cache, rtol=1e-2, atol=1e-2)
        assert False


if __name__ == "__main__":
    # parameters for problem
    min_batch_size = 1
    max_batch_size = 8
    min_head_dim = 0
    max_head_dim = 8
    min_seq_len = 6
    max_seq_len = 18
    nheads = 1  # support only nheads = 1, for now

    # 1) Random cases
    num_cases = 100

    for _ in range(num_cases):
        bs = random.randint(min_batch_size, max_batch_size)
        seq_len = 2 ** random.randint(min_seq_len, max_seq_len)
        hdim = 2 ** random.randint(min_head_dim, max_head_dim)
        test_ragged_layout_trans(bs, seq_len, nheads, hdim, torch.bfloat16)

    print(f"{num_cases} random tests done!\n")

    # 2) Fixed tests
    test_ragged_layout_trans(1, 1024, 1, 128, torch.bfloat16)
    test_ragged_layout_trans(1, 2048, 1, 128, torch.bfloat16)
    test_ragged_layout_trans(1, 4096, 1, 128, torch.bfloat16)
    test_ragged_layout_trans(1, 8192, 1, 128, torch.bfloat16)
    test_ragged_layout_trans(1, 16384, 1, 128, torch.bfloat16)
    test_ragged_layout_trans(1, 32768, 1, 128, torch.bfloat16)
    test_ragged_layout_trans(1, 65536, 1, 128, torch.bfloat16)
    test_ragged_layout_trans(1, 131072, 1, 128, torch.bfloat16)
    # test_ragged_layout_trans(1, 16, 1, 128, torch.bfloat16)

    print(f"Fixed tests done!\n")
