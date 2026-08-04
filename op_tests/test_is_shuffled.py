"""Regression tests for the explicit `is_shuffled` layout argument.

shuffle_weight() records the preshuffled MFMA layout in a plain `is_shuffled`
attribute on the tensor it returns. That attribute is not tensor metadata, so
clone(), to(), detach(), slicing, view() and nn.Parameter() all drop it. The
kernels then read the weights back with the wrong layout and silently produce
garbage. `is_shuffled` lets a caller state the layout when the tensor can no
longer carry it.
"""

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.ops.shuffle import shuffle_weight
from aiter.tuned_gemm import gemm_a16w16
from aiter.utility import fp4_utils

ATOL = 0.05


def run_moe(
    hidden, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, is_shuffled=None
):
    output = fused_moe(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=None,
        a2_scale=None,
        hidden_pad=0,
        intermediate_pad=0,
        is_shuffled=is_shuffled,
    )
    torch.cuda.synchronize()
    return output


def test_fused_moe_cloned_fp4_weights():
    num_experts, model_dim, inter_dim, top_k, num_tokens = 257, 4096, 1024, 9, 8
    generator = torch.Generator(device="cuda").manual_seed(42)

    hidden = torch.randn(
        (num_tokens, model_dim),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        device="cuda",
        dtype=torch.int32,
        generator=generator,
    )
    topk_weights = torch.randn(
        (num_tokens, top_k),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).softmax(dim=-1)

    # The unquantized dimensions must match hidden.shape[-1]. The old reproducer
    # used 1024 here, causing the kernel to read past w1 for a 4096-wide input.
    w1_bf16 = torch.randn(
        (num_experts, 2 * inter_dim, model_dim),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    w2_bf16 = torch.randn(
        (num_experts, model_dim, inter_dim),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )

    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1, w1_scale = torch_quant(w1_bf16, quant_dtype=dtypes.fp4x2)
    w2, w2_scale = torch_quant(w2_bf16, quant_dtype=dtypes.fp4x2)
    w1 = w1.view(num_experts, 2 * inter_dim, model_dim // 2)
    w2 = w2.view(num_experts, model_dim, inter_dim // 2)

    # Silu + per_1x32 uses the A4W4 path, which expects the generic FP4 layout.
    w1 = shuffle_weight(w1, layout=(16, 16))
    w2 = shuffle_weight(w2, layout=(16, 16))
    w1_scale = fp4_utils.e8m0_shuffle(w1_scale)
    w2_scale = fp4_utils.e8m0_shuffle(w2_scale)

    cloned = tuple(tensor.clone() for tensor in (w1, w2, w1_scale, w2_scale))
    for original, clone in zip((w1, w2, w1_scale, w2_scale), cloned):
        assert original.data_ptr() != clone.data_ptr()
        assert original.shape == clone.shape
        assert original.stride() == clone.stride()
        assert torch.equal(original, clone)

    for name, derived in (
        ("clone()", w1.clone()),
        ("contiguous()", w1.contiguous()),
        ("to(device)", w1.to(w1.device, copy=True)),
        ("detach()", w1.detach()),
        ("nn.Parameter()", torch.nn.Parameter(w1, requires_grad=False)),
    ):
        print(f"is_shuffled after {name}: {getattr(derived, 'is_shuffled', False)}")

    # Two calls on the very same tensors bound the kernel's own nondeterminism:
    # stage2 reduces with atomics, and the output magnitude is ~1e5, so a fixed
    # atol of 0.05 is unreachable even when the result is correct.
    out_original = run_moe(hidden, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids)
    out_repeated = run_moe(hidden, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids)
    noise_floor = max((out_original - out_repeated).abs().max().item(), ATOL)
    tolerance = 2 * noise_floor
    print(f"Noise floor from identical inputs: {noise_floor}")

    # Declaring the layout makes the clone equivalent again.
    out_declared = run_moe(hidden, *cloned, topk_weights, topk_ids, is_shuffled=True)
    declared_diff = (out_original - out_declared).abs()
    print(f"Clone with is_shuffled=True max diff: {declared_diff.max().item()}")

    # Positive control: the argument is honored in both directions, so claiming
    # the preshuffled weights are unshuffled must break the result.
    out_mislabelled = run_moe(
        hidden, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, is_shuffled=False
    )
    mislabelled_diff = (out_original - out_mislabelled).abs()
    print(
        f"Same weights with is_shuffled=False max diff: {mislabelled_diff.max().item()}"
    )

    # Without the argument the clone is still read as unshuffled -- inference can
    # only look at the tag, which clone() dropped. This is the trap the explicit
    # argument exists to escape, so report it rather than pin it down.
    out_inferred = run_moe(hidden, *cloned, topk_weights, topk_ids)
    inferred_diff = (out_original - out_inferred).abs()
    print(f"Clone without is_shuffled max diff: {inferred_diff.max().item()}")

    assert mislabelled_diff.max().item() > tolerance
    torch.testing.assert_close(out_declared, out_original, rtol=0, atol=tolerance)


def test_gemm_a16w16_cloned_weight():
    m, n, k = 128, 1024, 512
    generator = torch.Generator(device="cuda").manual_seed(42)

    x = torch.randn((m, k), device="cuda", dtype=dtypes.bf16, generator=generator)
    weight = torch.randn((n, k), device="cuda", dtype=dtypes.bf16, generator=generator)
    shuffled = shuffle_weight(weight, layout=(16, 16))
    assert shuffled.is_shuffled is True

    clone = shuffled.clone()
    assert torch.equal(shuffled, clone)
    assert not getattr(clone, "is_shuffled", False)

    # gemm_a16w16 reads the layout off B, so the reference is the call that still
    # carries the tag. Repeat it first: a split-k config reduces with atomics, so
    # bound that here rather than assuming exact equality. The bound collapses to
    # exact equality when the selected kernel is deterministic.
    reference = gemm_a16w16(x, shuffled)
    repeated = gemm_a16w16(x, shuffled)
    torch.cuda.synchronize()
    tolerance = 2 * (reference - repeated).abs().max().item()

    # Declaring the layout has to reproduce it: same kernel over identical data.
    declared = gemm_a16w16(x, clone, is_shuffled=True)
    torch.cuda.synchronize()
    torch.testing.assert_close(declared, reference, rtol=0, atol=tolerance)

    # The untagged clone is what the caller gets today without the argument. It
    # is only reported: whether it diverges depends on the kernel picked for this
    # shape, and a shape whose kernel ignores the layout would not be a failure.
    inferred = gemm_a16w16(x, clone)
    torch.cuda.synchronize()
    print(
        "gemm_a16w16 clone without is_shuffled max diff: "
        f"{(reference - inferred).abs().max().item()}"
    )


if __name__ == "__main__":
    test_fused_moe_cloned_fp4_weights()
    test_gemm_a16w16_cloned_weight()
