import torch
import pytest
import triton
from aiter.ops.triton.gather_and_maybe_dequant_cache import (
    gather_and_maybe_dequant_cache
)


@pytest.mark.parametrize("NUM_BLOCKS", [131072, 262144, 524288])
@pytest.mark.parametrize("BLOCK_SIZE", [1, 4, 8, 16])
@pytest.mark.parametrize("BATCH_SIZE", [1, 2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("MAX_SEQ_LEN", [57000])
@pytest.mark.parametrize("HEAD_DIM", [64, 128, 192, 512, 576])
@pytest.mark.parametrize("NUM_HEADS", [1, 16, 64])
@pytest.mark.parametrize("NUM_SPLITS", [1, 2, 4, 16, 128])
def test_gather_and_maybe_dequant_cache(NUM_BLOCKS: int, BLOCK_SIZE: int, BATCH_SIZE: int, MAX_SEQ_LEN: int, HEAD_DIM: int, NUM_HEADS: int, NUM_SPLITS: int):
    device = torch.device("cuda")

    if (NUM_BLOCKS*BLOCK_SIZE*NUM_HEADS*HEAD_DIM) >= (2*1024*1024*1024):
        pytest.skip("Skipping caches greater than 2 GB")

    # Create test data
    src_cache = torch.randn(
        NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_DIM,
        dtype=torch.float16, device=device
    )

    # Create sequences
    seq_lens = torch.randint(64, MAX_SEQ_LEN, (BATCH_SIZE,), device=device)
    cu_seq_lens = torch.cat([
        torch.tensor([0], device=device),
        seq_lens.cumsum(0)
    ])
    tot_tokens = cu_seq_lens[-1].item()

    if tot_tokens > (NUM_BLOCKS*BLOCK_SIZE):
        pytest.skip("Skipping since cache size might not be large enough")

    dst = torch.zeros(
        tot_tokens, NUM_HEADS, HEAD_DIM,
        dtype=torch.float16, device=device
    )

    # Create block table
    max_blocks = triton.cdiv(MAX_SEQ_LEN, BLOCK_SIZE)
    block_table = torch.randint(
        0, NUM_BLOCKS, (BATCH_SIZE, max_blocks),
        dtype=torch.int32, device=device
    )

    ref = torch.Tensor([]).to(device)
    for batch in range(BATCH_SIZE):
        temp = block_table[batch, :triton.cdiv(seq_lens[batch], BLOCK_SIZE)]
        temp = src_cache[temp, ...]
        temp = temp.view(-1, NUM_HEADS, HEAD_DIM)[:seq_lens[batch], ...]
        ref = torch.cat((ref, temp))

    ref = ref.to(torch.float16)

    gather_and_maybe_dequant_cache(
        src_cache=src_cache,
        dst=dst,
        block_table=block_table,
        cu_seq_lens=cu_seq_lens,
        batch_size=BATCH_SIZE,
        num_splits=NUM_SPLITS,
    )

    # Check for NaNs or Infs
    assert not torch.isnan(dst).any(), f"NaNs found with num_splits={NUM_SPLITS}"
    assert not torch.isinf(dst).any(), f"Infs found with num_splits={NUM_SPLITS}"

    torch.testing.assert_close(ref, dst, atol=1e-2, rtol=1e-2)
