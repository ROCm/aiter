"""e8m0 scale tensors followed in memory by known bytes, for ragged-K tests.

On a K block that runs past K, a kernel that does not mask its scale loads reads
whatever memory follows the scale tensor. These tests put a known byte there, so
such a read changes the output on every run: 0xFF is the e8m0 NaN.
"""

import torch


def scales_with_tail(
    scales: torch.Tensor, tail: int, tail_groups: int = 16
) -> torch.Tensor:
    """Return the same [..., rows, groups] scales, followed in memory by
    `tail_groups` groups of `tail` bytes for every row.

    The result is a transposed view of a [..., groups + tail_groups, rows]
    buffer, so a read of group `groups` or later lands on a tail byte. The
    default of 16 groups covers a K block of 512.
    """
    *lead, rows, groups = scales.shape
    buf = torch.full(
        (*lead, groups + tail_groups, rows),
        tail,
        dtype=torch.uint8,
        device=scales.device,
    )
    buf[..., :groups, :] = scales.transpose(-1, -2)
    return buf[..., :groups, :].transpose(-1, -2)
