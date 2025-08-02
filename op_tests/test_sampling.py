# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops import sampling  # noqa: F401

torch.set_default_device("cuda")


def _to_tensor_scalar_tuple(x):
    if isinstance(x, torch.Tensor):
        return (x, 0)
    else:
        return (None, x)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 500, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_top_p_sampling(batch_size, vocab_size, p):
    torch.manual_seed(42)
    eps = 1e-4
    pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32).to(0)
    mask.scatter_add_(1, indices, (cdf > (1 - p) - eps).int())

    num_trials = 1000
    for _ in range(num_trials):
        samples = torch.ops.aiter.top_p_sampling_from_probs(
            normalized_prob, None, *_to_tensor_scalar_tuple(p), deterministic=True
        )
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 500, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_renorm_probs(batch_size, vocab_size, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask = (normalized_prob >= pivot.unsqueeze(-1)).int()
    renorm_prob_ground_truth = normalized_prob.clone()
    renorm_prob_ground_truth[mask == 0] = 0
    renorm_prob_ground_truth = renorm_prob_ground_truth / renorm_prob_ground_truth.sum(
        dim=-1, keepdim=True
    )

    renorm_prob = torch.ops.aiter.top_k_renorm_probs(
        normalized_prob, *_to_tensor_scalar_tuple(k)
    )
    for i in range(batch_size):
        torch.testing.assert_close(
            renorm_prob_ground_truth[i],
            renorm_prob[i],
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 500, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5])
@pytest.mark.parametrize("k", [10, 50])
def test_top_k_top_p_joint_sampling_from_probs(batch_size, vocab_size, p, k):
    torch.manual_seed(42)
    # if p == 0.1:
    #     k = int(vocab_size * 0.5)
    # elif p == 0.5:
    #     k = int(vocab_size * 0.1)
    # else:
    #     raise ValueError("p not recognized")
    eps = 1e-4
    pre_norm_prob = torch.rand(batch_size, vocab_size)
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    # top-p mask
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask_top_p = torch.zeros(batch_size, vocab_size, dtype=torch.int32)
    mask_top_p.scatter_add_(1, indices, (cdf > (1 - p) - eps).int())
    # top-k mask
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask_top_k = (normalized_prob >= pivot.unsqueeze(-1)).int()
    # overall mask
    mask = torch.minimum(mask_top_p, mask_top_k)
    top_p_tensor = torch.full((batch_size,), p)
    top_k_tensor = torch.full((batch_size,), k)

    num_trials = 1000
    for _ in range(num_trials):
        samples = torch.ops.aiter.top_k_top_p_sampling_from_probs(
            normalized_prob,
            None,
            *_to_tensor_scalar_tuple(top_k_tensor),
            *_to_tensor_scalar_tuple(top_p_tensor),
            deterministic=True,
        )
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1), normalized_prob[
            torch.arange(batch_size), samples
        ]


if __name__ == "__main__":
    test_top_k_top_p_joint_sampling_from_probs(40, 129280, 0.6, 20)
    # test_top_k_renorm_probs(1, 129280, 10)
    # test_top_p_sampling(1, 129280, 0.1)
