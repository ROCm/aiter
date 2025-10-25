import torch
import aiter
import pytest

import numpy as np


def create_random_logits(
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """Create random logits tensor for testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Generate logits with some structure to make testing more meaningful
    logits = torch.randn(row_starts.shape[0], max(row_ends), dtype=dtype, device="cuda")
    for i, end in enumerate(row_ends):
        logits[i, end:] = float("-inf")
    return logits


def create_row_boundaries(
    num_rows: int, top_k: int = 2048
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create row start and end indices for testing."""
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    row_ends = torch.arange(1, num_rows + 1, device="cuda", dtype=torch.int32) * 128
    return row_starts, row_ends


def compare_top_k_results(
    cuda_indices: torch.Tensor,
    cuda_values: torch.Tensor,
    torch_indices: torch.Tensor,
    torch_values: torch.Tensor,
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
        cuda_row_values = cuda_values[row_idx][:num_valid].cpu()
        torch_row_values = torch_values[row_idx][:num_valid].cpu()

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


@pytest.mark.parametrize("num_rows", [8, 16, 32, 64, 128, 256, 512, 768, 1024])
def test_top_k_per_row(num_rows: int) -> None:
    """
    Test top_k_per_row.
    """
    torch.set_default_device("cuda:0")
    top_k = 2048

    # Create test data
    row_starts, row_ends = create_row_boundaries(num_rows)
    logits = create_random_logits(row_starts, row_ends, torch.float32, 42)

    # Create output tensors
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
    values = torch.empty((num_rows, top_k), dtype=torch.float32, device="cuda")

    # Run the kernel
    aiter.topk_per_row(
            logits,
            row_starts,
            row_ends,
            indices,
            values,
            num_rows,
            logits.stride(0),
            logits.stride(1)
        )

    # Run reference implementation
    torch_values, torch_indices = logits.topk(min(top_k, max(row_ends)), dim=-1)
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    mask = mask_lo & mask_hi
    torch_indices = torch_indices.masked_fill(~mask, -1)

    # Compare results
    assert compare_top_k_results(
        indices, values, torch_indices, torch_values, row_starts, row_ends, top_k
    ), "aiter topk_per_row results don't match with reference torch topk implementation"