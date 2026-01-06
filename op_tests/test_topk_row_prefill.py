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
    pt_file_path: str = None,
    replicate_first_row: bool = False,
) -> torch.Tensor:
    """Create random logits tensor for testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Using random seed: {seed}")
    # Generate logits with some structure to make testing more meaningful
    if data_generation == "random":
        # Generate random data (each element unique)
        logits = torch.randn(
            (row_starts.shape[0], max(row_ends)), dtype=dtype, device="cuda"
        )
        # Generate a random value and fill the entire logits tensor
        # random_value = torch.randn(1, dtype=dtype, device="cuda").item()
        # logits = torch.full(
        #     (row_starts.shape[0], max(row_ends)),
        #     random_value,
        #     dtype=dtype,
        #     device="cuda"
        # )
        # print(f"Generated logits: all elements are {random_value:.7f}")
        print(
            f"    shape={logits.shape}, min={logits.min().item():.7f}, max={logits.max().item():.7f}, mean={logits.mean().item():.7f}"
        )
    elif data_generation == "10LSBits":
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
    for i, end in enumerate(row_ends):
        logits[i, end:] = float("-inf")
    return logits


def create_row_boundaries(
    split_rows: int,
    context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create row start and end indices for testing."""
    row_starts = torch.zeros(split_rows, dtype=torch.int32, device="cuda")
    row_ends = torch.arange(
        context_len - split_rows + 1, context_len + 1, device="cuda", dtype=torch.int32
    )
    return row_starts, row_ends


def load_prefill_configs():
    """
    Load prefill test configurations.
    Returns list of (num_rows, context_len, top_k) tuples.
    """
    configs = [
        # Group 1: num_rows = 4k
        (1, 65536, 2048),
        # (4096, 4096, 2048),    # num_rows=4k, context_len=4k
        # (4096, 8192, 2048),    # num_rows=4k, context_len=8k
        # (4096, 12288, 2048),   # num_rows=4k, context_len=12k
        # (4096, 16384, 2048),   # num_rows=4k, context_len=16k
        # (4096, 20480, 2048),   # num_rows=4k, context_len=20k
        # (4096, 24576, 2048),   # num_rows=4k, context_len=24k
        # (4096, 28672, 2048),   # num_rows=4k, context_len=28k
        # (4096, 32768, 2048),   # num_rows=4k, context_len=32k
        # # # Group 2: num_rows = 8k
        # (1, 8192, 2048),       # num_rows=4k, context_len=20k
        # (8192, 8192, 2048),    # num_rows=8k, context_len=8k
        # (8192, 16384, 2048),   # num_rows=8k, context_len=16k
        # (8192, 24576, 2048),   # num_rows=8k, context_len=24k
        # (8192, 32768, 2048),   # num_rows=8k, context_len=32k
        # (8192, 40960, 2048),   # num_rows=8k, context_len=40k
        # (8192, 49152, 2048),   # num_rows=8k, context_len=48k
        # (8192, 57344, 2048),   # num_rows=8k, context_len=56k
        # (8192, 65536, 2048),   # num_rows=8k, context_len=64k
        # (8931, 71469, 2048),     # num_rows=8k, context_len=70k
    ]
    return configs


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
    print(f"now compare topk results with num_rows: {num_rows}")
    for row_idx in range(num_rows):
        # Get valid elements using row boundaries
        row_start = row_starts[row_idx].item()
        row_end = row_ends[row_idx].item()
        row_length = row_end - row_start
        num_valid = min(top_k, row_length)
        cuda_row_indices = cuda_indices[row_idx][:num_valid].cpu()
        torch_row_indices = torch_indices[row_idx][:num_valid].cpu()

        logits_row = logits[row_idx]
        max_valid_idx = row_length - 1

        # Check for invalid indices
        cuda_invalid_mask = (cuda_row_indices < 0) | (cuda_row_indices > max_valid_idx)
        torch_invalid_mask = (torch_row_indices < 0) | (
            torch_row_indices > max_valid_idx
        )

        if cuda_invalid_mask.any():
            invalid_indices = cuda_row_indices[cuda_invalid_mask].tolist()
            print(
                f"Row {row_idx}: CUDA has {len(invalid_indices)} invalid indices: {invalid_indices}"
            )
            print(f"    Valid range: [0, {max_valid_idx}]")

        if torch_invalid_mask.any():
            invalid_indices = torch_row_indices[torch_invalid_mask].tolist()
            print(
                f"Row {row_idx}: PyTorch has {len(invalid_indices)} invalid indices: {invalid_indices}"
            )
            print(f"    Valid range: [0, {max_valid_idx}]")

        cuda_row_indices = cuda_row_indices[
            (cuda_row_indices >= 0) & (cuda_row_indices <= max_valid_idx)
        ]
        torch_row_indices = torch_row_indices[
            (torch_row_indices >= 0) & (torch_row_indices <= max_valid_idx)
        ]

        # Compare the sets of indices first
        cuda_set = set(cuda_row_indices.tolist())
        torch_set = set(torch_row_indices.tolist())

        # Calculate differences
        cuda_only_indices = cuda_set - torch_set
        torch_only_indices = torch_set - cuda_set
        common_indices = cuda_set & torch_set

        if cuda_set == torch_set:
            continue

        # Any difference in elements, compare the values
        cuda_row_values = [logits_row[i] for i in cuda_row_indices]
        torch_row_values = [logits_row[i] for i in torch_row_indices]

        cuda_only_values, torch_only_values = [], []

        # Print difference elements
        cuda_only_indices = cuda_set - torch_set
        torch_only_indices = torch_set - cuda_set

        for idx in cuda_set - torch_set:
            cuda_pos = (cuda_row_indices == idx).nonzero(as_tuple=True)[0]
            cuda_only_values.append(cuda_row_values[cuda_pos[0]])

        for idx in torch_set - cuda_set:
            torch_pos = (torch_row_indices == idx).nonzero(as_tuple=True)[0]
            torch_only_values.append(torch_row_values[torch_pos[0]])

        if len(cuda_only_values) != len(torch_only_values):
            # Calculate common elements
            common_indices = cuda_set & torch_set

            # Check for duplicate indices
            cuda_original_count = len(cuda_row_indices)
            torch_original_count = len(torch_row_indices)
            cuda_has_duplicates = cuda_original_count != len(cuda_set)
            torch_has_duplicates = torch_original_count != len(torch_set)

            # Print detailed difference information
            print(f"\n{'='*80}")
            print(f"Row {row_idx} Failed: different number of differences")
            print(f"{'='*80}")

            print("\nElement Statistics:")
            print(f"  o CUDA Top-K original count:    {cuda_original_count}")
            print(f"  o CUDA Top-K unique count:      {len(cuda_set)}")
            print(
                f"  o CUDA has duplicate indices:   {'Yes' if cuda_has_duplicates else 'No'}"
            )
            if cuda_has_duplicates:
                print(f"    Duplicated {cuda_original_count - len(cuda_set)} indices")

            print(f"  o PyTorch Top-K original count: {torch_original_count}")
            print(f"  o PyTorch Top-K unique count:   {len(torch_set)}")
            print(
                f"  o PyTorch has duplicates:       {'Yes' if torch_has_duplicates else 'No'}"
            )
            if torch_has_duplicates:
                print(f"    Duplicated {torch_original_count - len(torch_set)} indices")

            print(f"\n  o Common element count:         {len(common_indices)}")
            print(f"  o CUDA unique element count:    {len(cuda_only_indices)}")
            print(f"  o PyTorch unique element count: {len(torch_only_indices)}")

            # If there are duplicates, find the duplicate indices
            if cuda_has_duplicates:
                from collections import Counter

                cuda_counter = Counter(cuda_row_indices.tolist())
                duplicates = {
                    idx: count for idx, count in cuda_counter.items() if count > 1
                }
                print(f"\n  CUDA duplicate indices (total {len(duplicates)}):")
                for idx, count in sorted(duplicates.items()):
                    print(f"      Index {idx}: appears {count} times")

            print(f"\nCUDA unique {len(cuda_only_values)} elements:")
            for i, (idx, val) in enumerate(
                zip(sorted(cuda_only_indices), cuda_only_values)
            ):
                print(f"  [{i}] index={idx:<6} value={val:.10f}")

            print(f"\nPyTorch unique {len(torch_only_values)} elements:")
            for i, (idx, val) in enumerate(
                zip(sorted(torch_only_indices), torch_only_values)
            ):
                print(f"  [{i}] index={idx:<6} value={val:.10f}")

            if len(cuda_only_values) != len(torch_only_values):
                return False
            # Check if value differences are within tolerance
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
def run_top_k_per_row_prefill_fast(
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
    Run the fast version of top_k_per_row kernel.
    """
    return aiter.top_k_per_row_prefill_fast(
        logits,
        row_starts,
        row_ends,
        indices,
        values,
        num_rows,
        stride_row,
        stride_col,
    )


@benchmark()
def test_top_k_per_row_prefill(
    num_rows: int, context_len: int, top_k: int, data_generation: str = "random"
) -> dict:
    """
    Test topk_per_row_prefill and compare standard vs fast version.
    """
    ret = {}
    torch.set_default_device("cuda:0")

    # Create test data
    if args.load_prefill_data:
        # Load prefill data from .pt files
        print(f"Loading prefill data from: {args.load_prefill_data}")

        import os

        logits_file = os.path.join(args.load_prefill_data, "logits.pt")
        ks_file = os.path.join(args.load_prefill_data, "cu_seqlen_ks.pt")
        ke_file = os.path.join(args.load_prefill_data, "cu_seqlen_ke.pt")

        # Check if files exist
        for file_path, file_name in [
            (logits_file, "logits.pt"),
            (ks_file, "cu_seqlen_ks.pt"),
            (ke_file, "cu_seqlen_ke.pt"),
        ]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

        # Load data
        print(f"  o Loading {os.path.basename(logits_file)}...")
        logits = torch.load(logits_file, weights_only=False)
        print(f"  o Loading {os.path.basename(ks_file)}...")
        cu_seqlen_ks = torch.load(ks_file, weights_only=False)
        print(f"  o Loading {os.path.basename(ke_file)}...")
        cu_seqlen_ke = torch.load(ke_file, weights_only=False)

        # Convert to row_starts and row_ends
        row_starts = cu_seqlen_ks.to(torch.int32).cuda()
        row_ends = cu_seqlen_ke.to(torch.int32).cuda()

        # Ensure logits is on CUDA
        if logits.device.type != "cuda":
            logits = logits.cuda()

        # Ensure correct data type
        if logits.dtype != torch.float32:
            logits = logits.to(torch.float32)

        # Update num_rows to actual loaded row count
        actual_num_rows = len(row_starts)
        if actual_num_rows != num_rows:
            print(
                f"Note: Loaded row count ({actual_num_rows}) differs from specified num_rows ({num_rows})"
            )
            print(f"         Using loaded row count: {actual_num_rows}")
            num_rows = actual_num_rows

        print("Data loading completed:")
        print(
            f"  o logits shape: {logits.shape}, dtype: {logits.dtype}, device: {logits.device}"
        )
        print(f"  o row_starts shape: {row_starts.shape}, dtype: {row_starts.dtype}")
        print(f"  o row_ends shape: {row_ends.shape}, dtype: {row_ends.dtype}")
        print(f"  o num_rows: {num_rows}")
        print(
            f"  o Sequence length range: [{row_ends.min().item()}, {row_ends.max().item()}]"
        )
        print()
    else:
        # Original random generation logic
        row_starts, row_ends = create_row_boundaries(
            split_rows=num_rows, context_len=context_len
        )
        logits = create_random_logits(
            row_starts,
            row_ends,
            torch.float32,
            42,
            data_generation,
            args.pt_file,
            args.replicate_first_row,
        )

    # Create output tensors (initialize with -1 to detect uninitialized values)
    indices_standard = torch.full(
        (num_rows, top_k), -1, dtype=torch.int32, device="cuda"
    )
    indices_fast = torch.full((num_rows, top_k), -1, dtype=torch.int32, device="cuda")

    # Run the standard kernel
    print(f"\n{'='*80}")
    print("Running Standard Kernel performance test (101 iterations)...")
    print(f"{'='*80}")
    _, us_standard = run_top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        indices_standard,
        None,  # values
        num_rows,
        logits.stride(0),
        logits.stride(1),
    )
    print(f"Standard Kernel performance test completed: {us_standard:.2f} us")

    # Run the fast kernel
    print(f"\n{'='*80}")
    print("Running Fast Kernel performance test (101 iterations)...")
    print(f"{'='*80}")
    _, us_fast = run_top_k_per_row_prefill_fast(
        logits,
        row_starts,
        row_ends,
        indices_fast,
        None,  # values
        num_rows,
        logits.stride(0),
        logits.stride(1),
    )
    print(f"Fast Kernel performance test completed: {us_fast:.2f} us")

    # Print indices_fast results
    print(f"\n{'='*80}")
    print("indices_fast all results")
    print(f"{'='*80}")
    print(f"indices_fast first 10 numbers: {indices_fast[0, :10]}")

    # Run reference implementation
    torch_indices = logits.topk(min(top_k, max(row_ends)), dim=-1)[1]
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    mask = mask_lo & mask_hi
    torch_indices = torch_indices.masked_fill(~mask, -1)

    # Compare results for standard version
    print(f"\n{'='*80}")
    print("Comparing Standard Kernel results...")
    print(f"{'='*80}")
    all_close_standard = compare_topk_results(
        logits, indices_standard, torch_indices, row_starts, row_ends, top_k
    )
    print(
        f"Standard Kernel comparison completed: {'Passed' if all_close_standard else 'Failed'}"
    )

    # Compare results for fast version
    print(f"\n{'='*80}")
    print("Comparing Fast Kernel results...")
    print(f"{'='*80}")
    all_close_fast = False
    all_close_fast = compare_topk_results(
        logits, indices_fast, torch_indices, row_starts, row_ends, top_k
    )
    print(
        f"Fast Kernel comparison completed: {'Passed' if all_close_fast else 'Failed'}"
    )

    # Calculate speedup
    speedup = us_standard / us_fast if us_fast > 0 else 0.0
    # Calculate improvement percentage
    improvement_pct = (
        (us_standard - us_fast) / us_standard * 100 if us_standard > 0 else 0.0
    )

    # ANSI color codes
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    # Determine which version is faster
    faster_is_standard = us_standard < us_fast
    standard_color = GREEN if faster_is_standard else RED
    fast_color = GREEN if not faster_is_standard else RED
    speedup_color = GREEN if speedup > 1.0 else RED

    # Print comparison
    print(f"\n{'='*80}")
    print(
        f"Performance Comparison (num_rows={num_rows}, context_len={context_len}, top_k={top_k}):"
    )
    print(
        f"  Standard version: {standard_color}{us_standard:.2f} us{RESET}, correct={all_close_standard}"
    )
    print(
        f"  Fast version:     {fast_color}{us_fast:.2f} us{RESET}, correct={all_close_fast}"
    )
    print(f"  Speedup:          {speedup_color}{speedup:.3f}x{RESET}")
    print(f"  Improvement:      {speedup_color}{improvement_pct:.2f}%{RESET}")
    print(f"{'='*80}\n")

    # measure performance
    ret["num_rows"] = num_rows
    ret["context_len"] = context_len
    ret["top_k"] = top_k
    ret["data_generation"] = data_generation
    ret["us_standard"] = us_standard
    ret["us_fast"] = us_fast
    ret["speedup"] = f"{speedup:.2f}x"
    ret["improvement_pct"] = f"{improvement_pct:.2f}%"
    ret["all_close_standard"] = all_close_standard
    ret["all_close_fast"] = all_close_fast
    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-c",
    "--context_len",
    type=int,
    default=[6912, 13824, 20736, 27648, 35560, 41472, 48384, 55296],
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
    "-b",
    "--decode_batch_size",
    type=int,
    default=[1, 2, 4],
    nargs="+",
    help="""decode_batch_size batch size.
    e.g.: -b 4""",
)

parser.add_argument(
    "-n",
    "--next_n",
    type=int,
    default=[2],
    nargs="+",
    help="""next_n elements per sequence in a row.
    e.g.: -n 4""",
)

parser.add_argument(
    "-d",
    "--data_generation",
    type=str,
    default=["10LSBits"],
    choices=["random", "10LSBits", "load_pt", "threshold_bin_data"],
    nargs="+",
    help="""Specify method for generating logits.
    e.g.: -d random
          -d threshold_bin_data
          -d load_pt (requires --pt-file)""",
)

parser.add_argument(
    "--pt-file",
    type=str,
    default=None,
    help="""Path to .pt file for loading logits data (used with -d load_pt).
    e.g.: --pt-file /path/to/debug_data.pt""",
)

parser.add_argument(
    "--dump-file",
    type=str,
    default="/home/topk_per_row_decode_debug/topk_per_row_decode_dump_1766463708.4711573.pt",
    help="""Path to dump format .pt file for decode_from_dump mode.
    e.g.: --dump-file /path/to/dump.pt""",
)

parser.add_argument(
    "--replicate-first-row",
    action="store_true",
    default=False,
    help="""When loading from PT file, replicate the first row data to all rows.
    Useful for testing with consistent data across all rows.
    e.g.: --replicate-first-row""",
)

parser.add_argument(
    "--load-prefill-data",
    type=str,
    default=None,
    help="""Path to directory containing prefill data files (logits.pt, cu_seqlen_ks.pt, cu_seqlen_ke.pt).
    When specified, row_starts, row_ends, and logits will be loaded from these files.
    e.g.: --load-prefill-data /home/topkPrefill""",
)

parser.add_argument(
    "-m",
    "--mode",
    type=str,
    default="all",
    choices=["prefill", "decode", "decode_from_dump", "all", "prefill_profile"],
    help="""Test mode: prefill, decode, decode_from_dump, prefill_profile, or all (default: all).
    e.g.: -m prefill           (only run prefill tests)
          -m decode            (only run decode tests)
          -m decode_from_dump  (run decode test from dump file)
          -m prefill_profile   (run prefill with fewer iterations for profiling)
          -m all               (run both prefill and decode tests)""",
)

parser.add_argument(
    "--profiling-iterations",
    type=int,
    default=1,
    help="""Number of iterations to run each kernel (for profiling with ATT trace).
    Use higher values (e.g. 100-1000) for capturing fast kernels with ATT.
    e.g.: --profiling-iterations 100""",
)

args = parser.parse_args()


# ========== Run Tests Based on Mode ==========
if args.mode in ["prefill", "all"]:
    # ========== Prefill Tests ==========
    df_prefill = []
    param_info_list = []
    # num_row = 6912
    configs_prefill = load_prefill_configs()
    print("=" * 80)
    print("Starting Prefill Tests...")
    print("=" * 80)
    for data_generation in args.data_generation:
        for num_rows, context_len, top_k in configs_prefill:
            ret = test_top_k_per_row_prefill(
                num_rows, context_len, top_k, data_generation
            )
            df_prefill.append(ret)

    df_prefill = pd.DataFrame(df_prefill)
    df_prefill.to_csv("prefill_topk.csv", index=False)

    # Set pandas display options to show all columns
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    print(f"\n{'=' * 80}")
    print(f"Summary for top_k_per_row_prefill kernel:\n{df_prefill}")
    print(f"{'=' * 80}\n")

print("\n" + "=" * 80)
print(f"Test completed! Mode: {args.mode}")
if args.mode in ["prefill", "all"]:
    print("   - Prefill results saved to: prefill_topk.csv")
print("=" * 80)
