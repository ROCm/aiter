# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
from scipy import stats

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
@pytest.mark.parametrize("k", [1, 10, 50])
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


def _create_controlled_probs(scenario: str, vocab_size: int = 1000):
    """
    Create probability distributions with well-separated values where
    floating-point calculation errors cannot affect the top-k/top-p boundary.

    Returns: (probs, k, p, expected_valid_tokens)
    """
    probs = torch.zeros(vocab_size)

    if scenario == "dominant":
        # Single dominant token - should always be selected
        probs[0] = 0.92
        probs[1:10] = 0.008  # 9 * 0.008 = 0.072
        probs[10:] = 0.008 / (vocab_size - 10)  # Remaining ~0.008
        probs = probs / probs.sum()  # Normalize
        k, p = 10, 0.9
        # Token 0 alone has prob > 0.9, so it's the only valid token under top-p
        expected_valid = {0}

    elif scenario == "topk_sep":
        # Clear top-k separation: top-10 have 0.09 each, rest have tiny probs
        probs[:10] = 0.09  # Sum = 0.9
        probs[10:] = 0.1 / (vocab_size - 10)  # Sum = 0.1
        probs = probs / probs.sum()
        k, p = 10, 1.0
        # Only top-k matters (p=1.0), gap is 0.09 vs ~0.0001
        expected_valid = set(range(10))

    elif scenario == "topp_boundary":
        # Clear top-p boundary: cumsum reaches p at a well-defined point
        probs[0] = 0.50
        probs[1] = 0.31
        probs[2] = 0.10
        probs[3] = 0.05
        probs[4:] = 0.04 / (vocab_size - 4)
        probs = probs / probs.sum()
        k, p = 100, 0.8
        # Cumsum: 0.50, 0.81, 0.91, 0.96, ...
        # Top-p=0.8 includes tokens 0,1 (cumsum exceeds 0.8 at token 1)
        expected_valid = {0, 1}

    elif scenario == "both_active":
        # Both constraints active: top-k limits more than top-p
        probs[:5] = 0.15  # Sum = 0.75
        probs[5:10] = 0.04  # Sum = 0.20
        probs[10:] = 0.05 / (vocab_size - 10)  # Sum = 0.05
        probs = probs / probs.sum()
        k, p = 5, 0.95
        # Top-k=5 limits to tokens 0-4
        # Top-p=0.95 would allow tokens 0-9, but k=5 is stricter
        # Gap between token 4 (0.15) and token 5 (0.04) is clear
        expected_valid = set(range(5))

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return probs, k, p, expected_valid


@pytest.mark.parametrize(
    "scenario", ["dominant", "topk_sep", "topp_boundary", "both_active"]
)
@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_top_k_top_p_deterministic_controlled(scenario, batch_size):
    """
    Test with controlled probability distributions where
    floating-point calculation errors cannot affect the outcome.
    """
    torch.manual_seed(42)

    probs_single, k, p, expected_valid = _create_controlled_probs(scenario)
    vocab_size = probs_single.shape[0]

    # Expand to batch
    probs = probs_single.unsqueeze(0).expand(batch_size, -1).contiguous()

    num_trials = 100
    for trial in range(num_trials):
        samples = torch.ops.aiter.top_k_top_p_sampling_from_probs(
            probs,
            None,
            *_to_tensor_scalar_tuple(k),
            *_to_tensor_scalar_tuple(p),
            deterministic=True,
        )

        # Verify all samples are within expected valid set
        for b in range(batch_size):
            sample_val = samples[b].item()
            assert sample_val in expected_valid, (
                f"Scenario '{scenario}', trial {trial}, batch {b}: "
                f"sampled token {sample_val} not in expected valid set {expected_valid}"
            )


# Statistical Equivalence Test - Verify the sampling distribution matches the expected theoretical
# distribution using chi-squared goodness-of-fit test.

def _compute_expected_distribution(probs, k, p, eps=1e-4):
    """
    Compute the theoretical probability distribution that the kernel should
    sample from after applying top-k and top-p filtering.

    Args:
        probs: [batch_size, vocab_size] - normalized input probabilities
        k: top-k parameter
        p: top-p parameter
        eps: tolerance for boundary comparison (matches kernel behavior)

    Returns:
        [batch_size, vocab_size] - expected sampling probabilities (normalized)
    """
    batch_size, vocab_size = probs.shape
    expected = torch.zeros_like(probs)

    for b in range(batch_size):
        # Step 1: Find top-k mask (tokens with prob >= k-th highest)
        sorted_probs_desc, _ = torch.sort(probs[b], descending=True)
        pivot_k = sorted_probs_desc[k - 1]
        mask_topk = probs[b] >= pivot_k

        # Step 2: Find top-p mask (tokens in cumulative sum up to p)
        sorted_probs_asc, indices_asc = torch.sort(probs[b], descending=False)
        cdf = torch.cumsum(sorted_probs_asc, dim=0)
        # Tokens where CDF > (1-p) - eps are in the top-p set
        in_topp_sorted = cdf > (1 - p) - eps
        mask_topp = torch.zeros(vocab_size, dtype=torch.bool, device=probs.device)
        mask_topp[indices_asc] = in_topp_sorted

        # Step 3: Valid set = intersection of top-k AND top-p
        valid_mask = mask_topk & mask_topp

        # Step 4: Normalize probabilities among valid tokens
        valid_probs = probs[b] * valid_mask.float()
        prob_sum = valid_probs.sum()
        if prob_sum > 0:
            expected[b] = valid_probs / prob_sum

    return expected


def _compute_frequencies(samples, vocab_size, batch_size):
    """
    Count how often each token was sampled.

    Args:
        samples: list of [batch_size] tensors (num_samples total)
        vocab_size: size of vocabulary
        batch_size: batch dimension

    Returns:
        [batch_size, vocab_size] - observed frequencies (counts)
    """
    # Stack all samples: [num_samples, batch_size]
    stacked = torch.stack(samples, dim=0).cpu()

    freq = torch.zeros(batch_size, vocab_size)
    for b in range(batch_size):
        freq[b] = torch.bincount(stacked[:, b], minlength=vocab_size).float()

    return freq


@pytest.mark.parametrize("batch_size", [1, 10, 50])
@pytest.mark.parametrize("vocab_size", [100, 1000, 10000])
@pytest.mark.parametrize("k,p", [(10, 0.9), (50, 0.5), (100, 0.95), (5, 0.3)])
def test_top_k_top_p_statistical_distribution(batch_size, vocab_size, k, p):
    """
    Verify the sampling distribution matches the expected theoretical
    distribution using chi-squared goodness-of-fit test.
    """
    if k > vocab_size:
        pytest.skip("k > vocab_size")

    torch.manual_seed(42)
    num_samples = 50000

    # 1. Generate random normalized probabilities
    pre_norm_prob = torch.rand(batch_size, vocab_size)
    probs = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    # 2. Compute expected distribution on CPU
    expected_probs = _compute_expected_distribution(probs.cpu(), k, p)

    # 3. Run kernel many times, collect samples
    samples = []
    for _ in range(num_samples):
        sample = torch.ops.aiter.top_k_top_p_sampling_from_probs(
            probs,
            None,
            *_to_tensor_scalar_tuple(k),
            *_to_tensor_scalar_tuple(p),
            deterministic=True,
        )
        samples.append(sample)

    # 4. Compute observed frequencies
    observed_freq = _compute_frequencies(samples, vocab_size, batch_size)

    # 5. Chi-squared goodness-of-fit test for each batch element
    min_expected_count = 5  # Chi-squared requires expected count >= 5 per bin
    for b in range(batch_size):
        # Compute expected counts for all tokens
        expected_counts_all = expected_probs[b] * num_samples

        # Filter to bins with sufficient expected counts (chi-squared requirement)
        sufficient_mask = expected_counts_all >= min_expected_count
        num_sufficient = sufficient_mask.sum().item()

        if num_sufficient <= 1:
            # Not enough bins with sufficient counts to test
            continue

        observed = observed_freq[b][sufficient_mask].cpu().numpy()
        expected_counts = expected_counts_all[sufficient_mask].cpu().numpy()

        # Chi-squared test: H0 = observed matches expected distribution
        chi2, p_value = stats.chisquare(observed, f_exp=expected_counts)

        assert p_value > 0.001, (
            f"Distribution mismatch for batch {b}: chi2={chi2:.2f}, "
            f"p_value={p_value:.6f} (expected > 0.001), "
            f"num_bins={num_sufficient}"
        )


if __name__ == "__main__":
    test_top_k_top_p_joint_sampling_from_probs(40, 129280, 0.6, 20)
    # test_top_k_top_p_statistical_distribution(10, 10000, 5, 0.3)
    # test_top_k_renorm_probs(1, 129280, 10)
    # test_top_p_sampling(1, 129280, 0.1)
