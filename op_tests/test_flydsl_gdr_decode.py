# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CI-visible tests for the FlyDSL GDR decode op: its export, and its numerics.

CI shards ``op_tests/test_*.py`` at depth 1 (``.github/scripts/split_tests.sh``)
and runs each as ``python3 <file>`` (``.github/scripts/aiter_test.sh``), so the
in-package suite at ``aiter/ops/flydsl/test_flydsl_linear_attention.py`` never
runs automatically, and a file without the ``__main__`` entry below collects
nothing and still exits 0. Being listed is not the same as being run.

The export checks matter on their own: importing from
``linear_attention_kernels`` keeps working if the export drops back out of
``__init__.py``, so nothing else would catch that.

These cases trust the torch reference and put the *kernel* on trial;
``op_tests/test_kda_torch_ref.py`` runs the arrow backwards. Which one fails
localises the fault: here alone, the per-channel kernel; there alone, the
reference; both, the gate formula they share.

Keep it thin -- shapes belong here, the reference kernel belongs in-package.
"""

import pytest
import torch

import aiter.ops.flydsl as flydsl_ops
from aiter.ops.flydsl import is_flydsl_available

pytestmark = pytest.mark.skipif(
    not is_flydsl_available(), reason="flydsl is not installed"
)

# Skips at import when flydsl or a GPU is missing, taking this module with it.
from aiter.ops.flydsl.test_flydsl_linear_attention import (
    Args,
    check_gdr_decode,
)


def test_flydsl_gdr_decode_is_importable_from_package_namespace():
    assert hasattr(flydsl_ops, "flydsl_gdr_decode"), (
        "flydsl_gdr_decode is missing from aiter.ops.flydsl -- check that its "
        "import is uncommented in aiter/ops/flydsl/__init__.py"
    )
    assert callable(flydsl_ops.flydsl_gdr_decode)


def test_flydsl_gdr_decode_is_advertised_in_all():
    assert "flydsl_gdr_decode" in flydsl_ops.__all__


def test_package_export_is_the_kernel_wrapper_itself():
    from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_decode

    assert flydsl_ops.flydsl_gdr_decode is flydsl_gdr_decode


@pytest.mark.parametrize(
    "args",
    [
        # Smallest GQA shape, the pre-847 scalar path.
        Args(
            dtype=torch.bfloat16,
            b=1,
            sq=1,
            num_k_heads=2,
            num_v_heads=8,
            head_k_dim=128,
            head_v_dim=128,
        ),
        # Batch large enough to exercise the state-shuffle indices.
        Args(
            dtype=torch.bfloat16,
            b=128,
            sq=1,
            num_k_heads=2,
            num_v_heads=8,
            head_k_dim=128,
            head_v_dim=128,
        ),
        # f32 bias against a bf16 query, decoupled from the query dtype for KDA.
        Args(
            dtype=torch.bfloat16,
            b=1,
            sq=1,
            num_k_heads=2,
            num_v_heads=8,
            head_k_dim=128,
            head_v_dim=128,
            dt_bias_dtype=torch.float32,
        ),
    ],
    ids=["gqa_b1", "gqa_b128", "f32_dt_bias"],
)
def test_flydsl_gdr_decode_matches_reference(args):
    check_gdr_decode(args)


G_MIN = -5.0
K = V = 128


def assert_close_rmse(name, ref, tri, ratio=1e-3, err_atol=1e-3):
    """vLLM's own bar (``test_kda.py:92``): RMSE-relative with an absolute
    escape hatch, at the packed-vs-dense decode pair's (1e-3, 1e-3). The ratio
    alone is not the bar.
    """
    ref, tri = ref.detach().float(), tri.detach().float()
    abs_err = (ref - tri).abs().max().item()
    if abs_err <= err_atol:
        return abs_err
    assert not torch.isnan(ref).any(), f"{name}: NaN in ref"
    assert not torch.isnan(tri).any(), f"{name}: NaN in tri"
    rmse_diff = (ref - tri).square().mean().sqrt().item()
    rmse_base = ref.square().mean().sqrt().item()
    rel_err = rmse_diff / (rmse_base + 1e-8)
    assert (
        rel_err < ratio
    ), f"{name}: max abs err {abs_err:.6f}, rmse ratio {rel_err:.6f} >= {ratio}"
    return abs_err


def _kda_inputs(B, H, dt, first_index, padded, shuffle, seed=0, indices_stride=1):
    """Build one KDA decode case.

    ``shuffle`` picks the state layout the caller holds: False is already
    (D_v, D_k), True is (D_k, D_v) and the wrapper transposes. With K == V the
    two are shape-identical, so feeding the wrong one tests the harness.
    """
    torch.manual_seed(seed)
    dev = "cuda"
    T = 1
    args = {
        "q": torch.randn(B, T, H, K, dtype=dt, device=dev),
        "k": torch.randn(B, T, H, K, dtype=dt, device=dev),
        "v": torch.randn(B, T, H, V, dtype=dt, device=dev),
        "a": torch.randn(B, T, H, K, dtype=dt, device=dev),
        "b": torch.randn(B, T, H, dtype=dt, device=dev),
        "dt_bias": (
            torch.randn(H, K, dtype=torch.float32, device=dev) * 0.1
        ).contiguous(),
        "A_log": (torch.randn(H, dtype=torch.float32, device=dev) * 0.5).contiguous(),
        "out": torch.zeros(B, T, H, V, dtype=dt, device=dev),
    }

    d0, d1 = (K, V) if shuffle else (V, K)
    n_slots = B + first_index
    if padded:
        # vLLM's setup: padded rows, so the kernel must honour the strides.
        storage = torch.randn(n_slots, H * K * V + 17, dtype=torch.float32, device=dev)
        pool = storage[:, : H * K * V].view(n_slots, H, d0, d1)
        # vLLM's own assertions (test_kda.py:321-322). The second bites: only
        # the outer stride is padded, each slot still internally contiguous.
        # Asserted, so a storage-shape change cannot silently test another layout.
        assert not pool.is_contiguous()
        assert pool.stride()[1:] == (d0 * d1, d1, 1)
    else:
        pool = torch.randn(n_slots, H, d0, d1, dtype=torch.float32, device=dev)

    if indices_stride > 1:
        # vLLM parametrizes this (state_indices_stride): the serving stack hands
        # over a strided column of a wider table, not a fresh contiguous array.
        storage = torch.zeros(B, indices_stride, dtype=torch.int32, device=dev)
        indices = storage[:, 0]
        assert indices.stride(0) == indices_stride
    else:
        indices = torch.empty(B, dtype=torch.int32, device=dev)
    indices.copy_(
        torch.arange(first_index, first_index + B, dtype=torch.int32, device=dev)
    )
    return args, pool, indices


def _kda_reference(args, initial_state):
    from aiter.ops.torch_ref.kda import kda_gate, l2norm, naive_recurrent_kda

    return naive_recurrent_kda(
        l2norm(args["q"]),
        l2norm(args["k"]),
        args["v"],
        kda_gate(args["a"], args["A_log"], args["dt_bias"], g_min=G_MIN),
        args["b"].float().sigmoid(),
        scale=K**-0.5,
        initial_state=initial_state,
        output_final_state=True,
    )


@pytest.mark.parametrize(
    ("B", "H", "dt", "shuffle", "padded", "first_index", "indices_stride"),
    [
        (1, 8, torch.bfloat16, True, False, 0, 1),
        (4, 12, torch.bfloat16, True, False, 0, 1),
        (2, 12, torch.float16, True, False, 0, 1),
        # K3 as deployed, every landmine at once: (D_v, D_k) state, padded
        # storage, slots numbered from 1, strided index column.
        (4, 12, torch.bfloat16, False, True, 1, 8),
        (4, 12, torch.bfloat16, False, False, 1, 1),
        (4, 12, torch.bfloat16, True, True, 1, 1),
        (4, 12, torch.bfloat16, True, False, 1, 8),
        # The batch sizes with 12x12 rows in gdr_decode_tuned.csv; before
        # these, only B=4's geometry was reachable from a test.
        (1, 12, torch.bfloat16, True, False, 1, 1),
        (64, 12, torch.bfloat16, True, False, 1, 1),
        (256, 12, torch.bfloat16, True, False, 1, 1),
        # Past K3's head count: no tuned row, so the fallback geometry.
        (2, 64, torch.bfloat16, True, False, 1, 1),
    ],
    ids=[
        "b1_h8_bf16",
        "b4_h12_bf16",
        "b2_h12_fp16",
        "k3_deployed",
        "no_shuffle",
        "padded_state",
        "strided_indices",
        "b1_h12_tuned",
        "b64_h12_tuned",
        "b256_h12_tuned",
        "b2_h64",
    ],
)
def test_kda_per_channel_gate_matches_torch_reference(
    B, H, dt, shuffle, padded, first_index, indices_stride
):
    """The KDA gate: 4D `a`, 2D f32 dt_bias, g_min * sigmoid(exp(A_log) * x).

    H is 1:1 with the k heads -- K3's ratio, which the scalar path never saw.
    """
    args, pool, indices = _kda_inputs(
        B, H, dt, first_index, padded, shuffle, indices_stride=indices_stride
    )

    initial_state = pool[indices.long()].clone()
    if not shuffle:
        initial_state = initial_state.transpose(-1, -2)

    kernel_pool = pool.clone()
    flydsl_ops.flydsl_gdr_decode(
        args["q"],
        args["k"],
        args["v"],
        args["a"],
        args["b"],
        args["dt_bias"],
        args["A_log"],
        indices,
        kernel_pool,
        args["out"],
        use_qk_l2norm=True,
        need_shuffle_state=shuffle,
    )

    ref_out, ref_state = _kda_reference(args, initial_state)
    got_state = kernel_pool[indices.long()]
    if not shuffle:
        got_state = got_state.transpose(-1, -2)

    out_err = assert_close_rmse("o", ref_out, args["out"])
    state_err = assert_close_rmse("ht", ref_state, got_state)

    # Tripwire under the bar above: agreement is ~1e-4, so drift to just inside
    # 1e-3 would pass while signalling that something changed.
    assert out_err < 5e-4 and state_err < 5e-4, (out_err, state_err)


def test_channel_strided_a_is_rejected():
    """`a` is vector-loaded along D_k, so a strided channel axis reads garbage.

    Rejected rather than silently wrong: unlike q/k/v the wrapper does not copy
    `a`, it forwards the strides. The other axes stay free -- K3 passes a view of
    a (1, B, H, K) buffer, which is padded in the batch stride.
    """
    B, H, dt = 2, 12, torch.bfloat16
    args, pool, indices = _kda_inputs(
        B, H, dt, first_index=0, padded=False, shuffle=True
    )

    # Every other channel of a 2K-wide buffer: stride(-1) == 2.
    wide = torch.randn(B, 1, H, 2 * K, dtype=dt, device="cuda")
    strided_a = wide[..., ::2]
    assert strided_a.shape == args["a"].shape and strided_a.stride(-1) == 2

    with pytest.raises(AssertionError, match="dense along D_k"):
        flydsl_ops.flydsl_gdr_decode(
            args["q"],
            args["k"],
            args["v"],
            strided_a,
            args["b"],
            args["dt_bias"],
            args["A_log"],
            indices,
            pool,
            args["out"],
            use_qk_l2norm=True,
            need_shuffle_state=True,
        )


def test_consumer_native_a_layout_is_rejected():
    """`a` held as (1, B, H_v, D_k) must be rejected, not read row-shifted.

    Consumers hold the gate that way and are expected to pass a (B, Sq, H_v, D_k)
    view. The un-transposed buffer clears every dtype and stride check, so
    without a full shape check it reaches the kernel and reads the Sq axis as
    batch: wrong for B > 1, out of bounds once B exceeds H_v.
    """
    B, H, dt = 2, 12, torch.bfloat16
    args, pool, indices = _kda_inputs(
        B, H, dt, first_index=0, padded=False, shuffle=True
    )

    consumer_native = args["a"].transpose(0, 1)
    assert consumer_native.shape == (1, B, H, K)
    assert consumer_native.stride(-1) == 1  # passes the D_k density check

    with pytest.raises(ValueError, match=r"`a` must have shape"):
        flydsl_ops.flydsl_gdr_decode(
            args["q"],
            args["k"],
            args["v"],
            consumer_native,
            args["b"],
            args["dt_bias"],
            args["A_log"],
            indices,
            pool,
            args["out"],
            use_qk_l2norm=True,
            need_shuffle_state=True,
        )


def test_staging_copies_are_ordered_against_a_caller_supplied_stream():
    """A staged `.contiguous()` copy must be issued on the launch stream.

    Ordering the inputs against ``side`` is the caller's job, so the sync does it
    and leaves only the wrapper's own copies under test. The sleep then makes the
    failure deterministic: it holds the current stream, so a copy left there
    cannot finish before a launch on ``side`` would start.
    """
    B, H, dt = 2, 12, torch.bfloat16
    args, pool, indices = _kda_inputs(
        B, H, dt, first_index=0, padded=False, shuffle=True
    )

    wide_bias = torch.randn(H, 2 * K, dtype=torch.float32, device="cuda") * 0.1
    args["dt_bias"] = wide_bias[:, ::2]
    assert not args["dt_bias"].is_contiguous()

    initial_state = pool[indices.long()].clone()
    kernel_pool = pool.clone()

    # A compile inside the timed section runs for seconds and outlasts the
    # sleep, closing the window; warm this config on throwaway buffers first.
    flydsl_ops.flydsl_gdr_decode(
        args["q"],
        args["k"],
        args["v"],
        args["a"],
        args["b"],
        args["dt_bias"],
        args["A_log"],
        indices,
        pool.clone(),
        torch.empty_like(args["out"]),
        use_qk_l2norm=True,
        need_shuffle_state=True,
    )

    side = torch.cuda.Stream()
    torch.cuda.synchronize()
    torch.cuda._sleep(100_000_000)
    flydsl_ops.flydsl_gdr_decode(
        args["q"],
        args["k"],
        args["v"],
        args["a"],
        args["b"],
        args["dt_bias"],
        args["A_log"],
        indices,
        kernel_pool,
        args["out"],
        use_qk_l2norm=True,
        need_shuffle_state=True,
        stream=side,
    )
    torch.cuda.synchronize()

    ref_out, ref_state = _kda_reference(args, initial_state)
    assert_close_rmse("o", ref_out, args["out"])
    assert_close_rmse("ht", ref_state, kernel_pool[indices.long()])


def test_non_contiguous_dt_bias_is_copied_not_rejected():
    """The counterpart: dt_bias is copied contiguous, so its strides are free.

    Pinned because the obvious "be strict everywhere" edit is to assert
    contiguity here too, which would reject an input that computes correctly.
    """
    B, H, dt = 2, 12, torch.bfloat16
    args, pool, indices = _kda_inputs(
        B, H, dt, first_index=0, padded=False, shuffle=True
    )

    wide_bias = torch.randn(H, 2 * K, dtype=torch.float32, device="cuda") * 0.1
    args["dt_bias"] = wide_bias[:, ::2]
    assert not args["dt_bias"].is_contiguous()

    initial_state = pool[indices.long()].clone()
    kernel_pool = pool.clone()
    flydsl_ops.flydsl_gdr_decode(
        args["q"],
        args["k"],
        args["v"],
        args["a"],
        args["b"],
        args["dt_bias"],
        args["A_log"],
        indices,
        kernel_pool,
        args["out"],
        use_qk_l2norm=True,
        need_shuffle_state=True,
    )

    ref_out, ref_state = _kda_reference(args, initial_state)
    assert_close_rmse("o", ref_out, args["out"])
    assert_close_rmse("ht", ref_state, kernel_pool[indices.long()])


def test_negative_slot_is_skipped_and_zero_is_not():
    """Pins this kernel's invalid-slot contract, which is not the reference's.

    aiter skips ``pool_idx < 0`` and writes nothing, leaving the caller's buffer
    untouched; the reference treats ``state_idx <= 0`` as invalid and zero-fills.
    They differ on both boundary and behaviour, so slot 0 being *processed* is
    the point.
    """
    B, H, dt = 4, 12, torch.bfloat16
    args, pool, _ = _kda_inputs(B, H, dt, first_index=0, padded=False, shuffle=True)

    indices = torch.tensor([-1, 0, 1, 2], dtype=torch.int32, device="cuda")
    args["out"].fill_(7.0)
    kernel_pool = pool.clone()

    flydsl_ops.flydsl_gdr_decode(
        args["q"],
        args["k"],
        args["v"],
        args["a"],
        args["b"],
        args["dt_bias"],
        args["A_log"],
        indices,
        kernel_pool,
        args["out"],
        use_qk_l2norm=True,
        need_shuffle_state=True,
    )

    # Row 0 asked for slot -1: untouched, not zeroed.
    assert (args["out"][0] == 7.0).all()
    # Slot 0 is valid here even though the reference would call it invalid.
    assert not (args["out"][1] == 7.0).all()
    assert not torch.equal(kernel_pool[0], pool[0])


def test_kda_tuned_rows_are_not_reachable_from_the_scalar_gate():
    """The 12:12 rows time a per-channel kernel; scalar GDR keeps its fallback.

    Both gates share (arch, dtypes, B, Sq, heads, dims), so without gate_mode in
    the key a scalar call at K3's geometry silently inherits a config tuned for
    a different binary.
    """
    from aiter.ops.flydsl.linear_attention_kernels import (
        GDR_GPU_ARCH,
        get_default_kwargs,
    )

    fallback = {"NUM_BLOCKS_PER_V_DIM": 1, "NUM_WARPS": 4, "WARP_THREADS_K": 8}
    tuned_batches = (1, 4, 64, 256)

    def lookup(B, gate_mode):
        return get_default_kwargs(
            "torch.bfloat16", "torch.float32", B, 1, 12, 12, 128, 128, gate_mode
        )

    if all(lookup(B, "kda") == fallback for B in tuned_batches):
        pytest.skip(f"no 12x12 kda row for {GDR_GPU_ARCH}")

    for B in tuned_batches:
        assert lookup(B, "gdr") == fallback, f"scalar GDR at B={B} took a kda row"


@pytest.mark.parametrize("act_dtype", [torch.bfloat16, torch.float16])
def test_state_store_follows_the_pool_dtype_not_the_activations(act_dtype):
    """A bf16 pool must be written as bf16 whatever the activations are.

    The store used to take its vector type from the activations, so fp16 ones
    truncated to fp16 and landed in a bf16 pool: same width, other exponent.
    bf16 activations hid it, both types agreeing by luck.
    """
    B, Sq, H, D = 2, 1, 4, 128

    def run(state_dtype):
        torch.manual_seed(0)
        kw = dict(dtype=act_dtype, device="cuda")
        q, k, v = (torch.randn(B, Sq, H, D, **kw) for _ in range(3))
        a, b = (torch.randn(B, Sq, H, **kw) for _ in range(2))
        dt_bias = torch.rand(H, dtype=torch.float32, device="cuda") + 1.0
        A_log = torch.rand(H, dtype=torch.float32, device="cuda") * 4.0
        indices = torch.arange(B, dtype=torch.int32, device="cuda")
        torch.manual_seed(99)
        state = torch.randn(B, H, D, D, dtype=torch.float32, device="cuda").to(
            state_dtype
        )
        out = torch.zeros(B, Sq, H, D, **kw)
        flydsl_ops.flydsl_gdr_decode(
            q,
            k,
            v,
            a,
            b,
            dt_bias,
            A_log,
            indices,
            state,
            out,
            use_qk_l2norm=True,
            need_shuffle_state=True,
        )
        return state.float()

    ref = run(torch.float32)
    got = run(torch.bfloat16)
    # bf16 keeps 8 mantissa bits, so rounding alone stays well inside 5%.
    assert (got - ref).abs().max() < 0.05 * ref.abs().max()


def test_fp32_activations_are_rejected():
    """The out store converts to fp16 for anything but bf16, so f32 must not
    reach the kernel and be silently narrowed."""
    B, Sq, H, D = 1, 1, 4, 128
    kw = dict(dtype=torch.float32, device="cuda")
    q, k, v = (torch.randn(B, Sq, H, D, **kw) for _ in range(3))
    a, b = (torch.randn(B, Sq, H, **kw) for _ in range(2))

    with pytest.raises(ValueError, match=r"`query` must be fp16 or bf16"):
        flydsl_ops.flydsl_gdr_decode(
            q,
            k,
            v,
            a,
            b,
            torch.rand(H, dtype=torch.float32, device="cuda"),
            torch.rand(H, dtype=torch.float32, device="cuda"),
            torch.zeros(B, dtype=torch.int32, device="cuda"),
            torch.randn(B, H, D, D, dtype=torch.float32, device="cuda"),
            torch.zeros(B, Sq, H, D, **kw),
            use_qk_l2norm=True,
            need_shuffle_state=True,
        )


# CI runs each file as `python3 <file>`, which collects nothing on its own.
if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
