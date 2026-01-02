import argparse

import numpy as np
import pandas as pd
import torch

import aiter
from aiter.test_common import benchmark, perftest


def create_random_logits(
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    dtype: torch.dtype,
    seed: int,
    data_generation: str = "random",
) -> torch.Tensor:
    """Create random logits tensor for testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Generate logits with some structure to make testing more meaningful
    if data_generation == "random":
        logits = torch.randn(
            row_starts.shape[0], max(row_ends), dtype=dtype, device="cuda"
        )
    elif data_generation == "10LSBits" or data_generation == "mixed":
        top_22_bits_mask = 0xFFFFFC00
        last_10_bits_mask = 0x000003FF
        fixed_top_22_bits = 0x3F900000
        # Generate random bits for the last 10 bits
        random_bottom_bits = torch.randint(
            0,
            2**10,
            (row_starts.shape[0], max(row_ends)),
            dtype=torch.int32,
            device="cuda",
        )
        # Combine: fixed top 22 bits with random last 10 bits
        logits_bits = (fixed_top_22_bits & top_22_bits_mask) | (
            random_bottom_bits & last_10_bits_mask
        )
        logits = logits_bits.view(dtype)

    if data_generation == "mixed":
        logits_random = torch.randn(
            row_starts.shape[0], max(row_ends), dtype=dtype, device="cuda"
        )
        # Mix the two logits tensors randomly
        mask = torch.randint(0, 2, (row_starts.shape[0], 1), device="cuda").bool()
        logits = torch.where(mask, logits, logits_random)

    for i, end in enumerate(row_ends):
        logits[i, end:] = float("-inf")
    return logits


def create_row_boundaries(
    num_rows: int, num_prefix: int = 0, top_k: int = 2048
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create row start and end indices for testing."""
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    row_ends = torch.arange(
        num_prefix + 1, num_prefix + num_rows + 1, device="cuda", dtype=torch.int32
    )
    return row_starts, row_ends


def compare_topk_results(
    logits: torch.Tensor,
    cuda_indices: torch.Tensor,
    torch_indices: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    top_k: int,
    tolerance: float = 1e-5,
) -> bool:
    """
    Compare results from CUDA top_k_per_row with torch.topk.
    Both results should be sorted and contain the same top-k elements.
    """
    num_rows = cuda_indices.shape[0]

    for row_idx in range(num_rows):
        # Get valid elements using row boundaries
        row_start = row_starts[row_idx].item()
        row_end = row_ends[row_idx].item()
        row_length = row_end - row_start
        num_valid = min(top_k, row_length)
        cuda_row_indices = cuda_indices[row_idx][:num_valid].cpu()
        torch_row_indices = torch_indices[row_idx][:num_valid].cpu()

        # Compare the sets of indices first
        cuda_set = set(cuda_row_indices.tolist())
        torch_set = set(torch_row_indices.tolist())
        if cuda_set == torch_set:
            continue

        # Any difference in elements, compare the values
        logits_row = logits[row_idx]
        cuda_row_values = [logits_row[i] for i in cuda_row_indices]
        torch_row_values = [logits_row[i] for i in torch_row_indices]

        cuda_only_values, torch_only_values = [], []
        for idx in cuda_set - torch_set:
            cuda_pos = (cuda_row_indices == idx).nonzero(as_tuple=True)[0]
            cuda_only_values.append(cuda_row_values[cuda_pos[0]])

        for idx in torch_set - cuda_set:
            torch_pos = (torch_row_indices == idx).nonzero(as_tuple=True)[0]
            torch_only_values.append(torch_row_values[torch_pos[0]])

        if len(cuda_only_values) != len(torch_only_values):
            return False
        if not torch.allclose(
            torch.tensor(cuda_only_values),
            torch.tensor(torch_only_values),
            rtol=tolerance,
            atol=tolerance,
        ):
            return False

    return True


@perftest()
def run_top_k_per_row_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    num_rows: int,
    stride_row: int,
    stride_col: int,
) -> None:
    """
    Run the top_k_per_row kernel.
    """
    return aiter.top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        indices,
        values,
        num_rows,
        stride_row,
        stride_col,
    )


@perftest()
def run_top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
    fast: bool,
) -> None:
    """
    Run the top_k_per_row kernel.
    """
    if fast:
        return aiter.top_k_per_row_decode_fast(
            logits,
            next_n,
            seqLens,
            indices,
            numRows,
            stride0,
            stride1,
        )
    else:
        return aiter.top_k_per_row_decode(
            logits,
            next_n,
            seqLens,
            indices,
            numRows,
            stride0,
            stride1,
        )


@benchmark()
def test_top_k_per_row_prefill(
    num_rows: int, num_prefix: int, top_k: int, data_generation: str = "random"
) -> dict:
    """
    Test topk_per_row_prefill.
    """
    ret = {}
    torch.set_default_device("cuda:0")

    # Create test data
    row_starts, row_ends = create_row_boundaries(num_rows, num_prefix)
    logits = create_random_logits(
        row_starts, row_ends, torch.float32, 42, data_generation
    )

    # Create output tensors
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")

    values = torch.empty((num_rows, top_k), dtype=torch.float32, device="cuda").fill_(0)

    # Run the kernel
    _, us = run_top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        indices,
        None,  # values
        num_rows,
        logits.stride(0),
        logits.stride(1),
    )

    # Run reference implementation
    torch_indices = logits.topk(min(top_k, max(row_ends)), dim=-1)[1]
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    mask = mask_lo & mask_hi
    torch_indices = torch_indices.masked_fill(~mask, -1)

    # Compare results
    all_close = compare_topk_results(
        logits, indices, torch_indices, row_starts, row_ends, top_k
    )

    # measure performance
    ret["context_len"] = logits.shape[1]
    ret["all_close"] = all_close
    ret["us"] = us
    return ret


@benchmark()
def test_top_k_per_row_decode(
    batch_size: int,
    context_len: int,
    top_k: int,
    next_n: int,
    data_generation: str = "random",
    fast: bool = False,
) -> dict:
    """
    Test top_k_per_row_decode with seq_lens tensor.
    """
    torch.set_default_device("cuda:0")
    ret = {}
    # Create test data
    num_rows = batch_size * next_n
    seq_lens = torch.empty(batch_size, dtype=torch.int32, device="cuda").fill_(
        context_len
    )
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    row_indices = torch.arange(num_rows, device="cuda") // next_n
    next_n_offset = torch.arange(num_rows, device="cuda") % next_n
    row_ends = seq_lens[row_indices] - next_n + next_n_offset + 1
    logits = create_random_logits(
        row_starts, row_ends, torch.float32, 42, data_generation
    )

    # Create output tensors
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")

    # Run the kernel
    _, us = run_top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        indices,
        num_rows,
        logits.stride(0),
        logits.stride(1),
        fast,
    )

    torch.cuda.synchronize()

    # Run reference implementation
    torch_indices = logits.topk(min(top_k, max(row_ends)), dim=-1)[1]
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    mask = mask_lo & mask_hi
    torch_indices = torch_indices.masked_fill(~mask, -1)

    # Compare results
    all_close = compare_topk_results(
        logits, indices, torch_indices, row_starts, row_ends, top_k
    )

    # measure performance
    ret["context_len"] = logits.shape[1]
    ret["all_close"] = all_close
    ret["us"] = us
    ret["fast"] = fast
    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-c",
    "--context_len",
    type=int,
    default=[8, 128, 1024, 3072, 4096, 8192, 16384, 32768, 65536, 90000, 128000],
    nargs="+",
    help="""number of kv.
    e.g.: -c 64""",
)

parser.add_argument(
    "-k",
    "--top_k",
    type=int,
    default=[2048],
    nargs="+",
    help="""top-k elements per row.
    e.g.: -k 2048""",
)

parser.add_argument(
    "--num_prefix",
    type=int,
    default=[0],
    nargs="+",
    help="""top-k elements per row.
    e.g.: --num_prefix 8000 16000 24000 32000 40000 48000 56000""",
)

parser.add_argument(
    "-b",
    "--decode_batch_size",
    type=int,
    default=[4, 8, 16, 24],
    nargs="+",
    help="""decode_batch_size batch size.
    e.g.: -b 4""",
)

parser.add_argument(
    "-n",
    "--next_n",
    type=int,
    default=[1, 2, 3, 4],
    nargs="+",
    help="""next_n elements per sequence in a row.
    e.g.: -n 4""",
)

parser.add_argument(
    "-d",
    "--data_generation",
    type=str,
    default=["random"],
    choices=["random", "10LSBits", "mixed"],
    nargs="+",
    help="""Specify method for generating logits.
    e.g.: -d random""",
)

args = parser.parse_args()


df = []
for data_generation in args.data_generation:
    for m in args.context_len:
        for k in args.top_k:
            for num_prefix in args.num_prefix:
                ret = test_top_k_per_row_prefill(m, num_prefix, k, data_generation)
                df.append(ret)

df = pd.DataFrame(df)
aiter.logger.info(f"summary for top_k_per_row_prefill kernel:\n{df}")


df = []
for data_generation in args.data_generation:
    for m in args.decode_batch_size:
        for ctx in args.context_len:
            for k in args.top_k:
                for n in args.next_n:
                    ret = test_top_k_per_row_decode(
                        m, ctx, k, n, data_generation, False
                    )
                    df.append(ret)
                    ret = test_top_k_per_row_decode(m, ctx, k, n, data_generation, True)
                    df.append(ret)

df = pd.DataFrame(df)
aiter.logger.info(f"summary for top_k_per_row_decode kernel:\n{df}")
