# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise the HIP sampling cores directly, including their CDF boundaries."""

import pytest
import torch

import aiter  # noqa: F401 -- expose the packaged csrc tree for standalone runs
from csrc.cpp_itfs import utils as cpp_utils
from csrc.cpp_itfs.sampling.top_k_renorm_probs import top_k_renorm_probs
from csrc.cpp_itfs.sampling.top_k_top_p_sampling_from_probs import (
    top_k_top_p_sampling_from_probs,
)
from csrc.cpp_itfs.sampling.top_p_sampling_from_probs import top_p_sampling_from_probs

requires_rocm = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="the cpp_itfs sampling cores require a ROCm GPU",
)


@pytest.mark.parametrize(
    "archs, expected",
    [
        ("gfx1250", ["gfx1250"]),
        (" gfx1250:xnack- ; gfx942:sramecc+ ", ["gfx1250", "gfx942"]),
        ("native", ["gfx1250"]),
    ],
)
def test_sampling_arch_validation_gfx1250(monkeypatch, archs, expected):
    monkeypatch.setattr(cpp_utils, "GPU_ARCH", archs)
    monkeypatch.setattr(cpp_utils, "get_gfx_runtime", lambda: "gfx1250")
    assert cpp_utils.validate_and_update_archs() == expected


def test_sampling_arch_validation_rejects_unknown(monkeypatch):
    monkeypatch.setattr(cpp_utils, "GPU_ARCH", "gfx9999")
    with pytest.raises(AssertionError, match="invalid or not supported"):
        cpp_utils.validate_and_update_archs()


def _draw_samples(
    sampler,
    probs,
    deterministic,
    *,
    indices=None,
    top_k_arr=None,
    top_k=0,
    top_p_arr=None,
    top_p=1.0,
    trials=32,
):
    generator = torch.Generator(device=probs.device).manual_seed(0x1250)
    samples = []
    for _ in range(trials):
        if sampler == "top_p":
            sample = top_p_sampling_from_probs(
                probs,
                indices,
                top_p_arr,
                top_p,
                deterministic=deterministic,
                generator=generator,
            )
        else:
            sample = top_k_top_p_sampling_from_probs(
                probs,
                indices,
                top_k_arr,
                top_k,
                top_p_arr,
                top_p,
                deterministic=deterministic,
                generator=generator,
            )
        samples.append(sample)
    return torch.stack(samples).cpu()


def _assert_sample_distribution(samples, expected):
    samples = samples.reshape(-1)
    assert ((samples >= 0) & (samples < expected.numel())).all()
    counts = torch.bincount(samples.long(), minlength=expected.numel())
    assert counts[expected == 0].sum() == 0, "sampled a filtered or zero-mass token"
    observed = counts.double() / samples.numel()
    # Six standard deviations plus a small rounding allowance keep this
    # seeded statistical check stable while detecting missing/biased CDF spans.
    tolerance = 6 * torch.sqrt(expected * (1 - expected) / samples.numel()) + 0.005
    error = (observed - expected).abs()
    assert (error <= tolerance).all(), (
        f"CDF distribution mismatch: max error={error.max().item():.4f}, "
        f"draws={samples.numel()}"
    )


@requires_rocm
@pytest.mark.parametrize("sampler", ["top_p", "joint"])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize(
    "vocab_size, positions",
    [
        (111, [0, 31, 32, 63, 64, 95, 110]),
        (4098, [0, 62, 64, 126, 128, 2048, 4097]),
        (8192, [0, 124, 128, 252, 256, 4096, 8191]),
    ],
)
def test_cpp_itfs_sampling_cdf_boundaries(
    sampler, deterministic, vocab_size, positions
):
    # gcd(vocab, 4) selects vec_size 1/2/4. The sparse mass straddles logical
    # 32-lane scans, physical wave boundaries, later chunks, and the final item.
    expected = torch.zeros(vocab_size, dtype=torch.float64, device="cpu")
    expected[positions] = (
        torch.tensor([3, 5, 7, 9, 11, 13, 16], dtype=torch.float64, device="cpu") / 64
    )
    # Independent Philox subsequences across 64 rows give 2048 draws with only
    # 32 launches; the largest probability tensor occupies 2 MiB.
    probs = expected.float().unsqueeze(0).repeat(64, 1).to("cuda")
    samples = _draw_samples(sampler, probs, deterministic, top_k=vocab_size)
    _assert_sample_distribution(samples, expected)


@requires_rocm
@pytest.mark.parametrize("sampler", ["top_p", "joint"])
@pytest.mark.parametrize("per_row", [False, True])
def test_cpp_itfs_sampling_filters_and_indices(sampler, per_row):
    vocab_size = 111
    base_probs = torch.zeros(3, vocab_size, dtype=torch.float32, device="cpu")
    for row, positions in enumerate(([7, 32, 96], [13, 63, 95], [2, 31, 64])):
        base_probs[row, positions] = torch.tensor(
            [0.5, 0.3125, 0.1875], dtype=torch.float32, device="cpu"
        )
    # Keep the index mapping the same length as probs: the top-p core currently
    # allocates its output using probs.size(0). Distinct row supports and
    # parameters prove that both probabilities and filters follow the mapping.
    probs = base_probs.repeat(16, 1).to("cuda")
    index_cpu = torch.arange(48, device="cpu").reshape(-1, 3)[:, [2, 0, 1]].reshape(-1)
    indices = index_cpu.to(device="cuda", dtype=torch.int32)
    p_cpu = (
        torch.tensor([1.0, 0.7, 0.4], device="cpu")
        if per_row
        else torch.full((3,), 0.7, device="cpu")
    )
    k_cpu = (
        torch.tensor([1, 3, 2], device="cpu")
        if per_row
        else torch.full((3,), 2, device="cpu")
    )
    top_p_arr = p_cpu.repeat(16).to("cuda") if per_row else None
    top_k_arr = (
        k_cpu.repeat(16).to(device="cuda", dtype=torch.int32) if per_row else None
    )
    samples = _draw_samples(
        sampler,
        probs,
        True,
        indices=indices,
        top_k_arr=top_k_arr,
        top_k=2,
        top_p_arr=top_p_arr,
        top_p=0.7,
    )

    sorted_probs, order = base_probs.sort(dim=-1, descending=True)
    mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
    keep = (mass_before < p_cpu[:, None]) & (sorted_probs > 0)
    if sampler == "joint":
        keep &= torch.arange(vocab_size, device="cpu")[None, :] < k_cpu[:, None]
    expected = torch.zeros_like(base_probs)
    expected.scatter_(1, order, sorted_probs * keep)
    expected /= expected.sum(dim=-1, keepdim=True)
    for source_row in range(3):
        columns = index_cpu % 3 == source_row
        _assert_sample_distribution(samples[:, columns], expected[source_row].double())


@requires_rocm
@pytest.mark.parametrize("vocab_size", [111, 4098, 8192])
@pytest.mark.parametrize("per_row", [False, True])
def test_cpp_itfs_top_k_renorm_matches_reference(vocab_size, per_row):
    generator = torch.Generator(device="cpu").manual_seed(0x1250)
    probs_cpu = torch.rand(4, vocab_size, generator=generator, device="cpu")
    probs_cpu /= probs_cpu.sum(dim=-1, keepdim=True)
    ks = [1, 7, vocab_size // 2, vocab_size] if per_row else [7] * 4
    expected = probs_cpu.clone()
    for row, k in enumerate(ks):
        pivot = probs_cpu[row].topk(k).values[-1]
        expected[row, probs_cpu[row] < pivot] = 0
    expected /= expected.sum(dim=-1, keepdim=True)

    probs = probs_cpu.to("cuda")
    top_k_arr = torch.tensor(ks, device="cuda", dtype=torch.int32) if per_row else None
    actual = top_k_renorm_probs(probs, top_k_arr, 0 if per_row else 7).cpu()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
    assert torch.equal(actual == 0, expected == 0)
    torch.testing.assert_close(
        actual.sum(dim=-1), torch.ones(4, device="cpu"), rtol=2e-5, atol=1e-6
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
