# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CI-visible numerics tests for the FlyDSL GDR decode kernel.

CI shards ``op_tests/test_*.py`` at depth 1 (``.github/scripts/split_tests.sh``)
and runs each as ``python3 <file>`` (``.github/scripts/aiter_test.sh``), so the
in-package suite at ``aiter/ops/flydsl/test_flydsl_linear_attention.py`` never
runs automatically, and a file without the ``__main__`` entry below collects
nothing and still exits 0. Being listed is not the same as being run.
"""

import pytest
import torch

import aiter.ops.flydsl as flydsl_ops
from aiter.ops.flydsl import is_flydsl_available
from aiter.test_common import checkAllclose

# CI runs this file as a script, so ``op_tests`` is not a package on sys.path;
# under pytest from the repo root it is.
try:
    from kda_ref import kda_gate, l2norm, naive_recurrent_kda
except ModuleNotFoundError as e:
    if e.name != "kda_ref":
        raise
    from op_tests.kda_ref import kda_gate, l2norm, naive_recurrent_kda

pytestmark = pytest.mark.skipif(
    not is_flydsl_available(), reason="flydsl is not installed"
)

# This import skips the whole module when flydsl or a GPU is missing.
from aiter.ops.flydsl.test_flydsl_linear_attention import (
    Args,
    check_gdr_decode,
)


@pytest.mark.parametrize(
    "args",
    [
        # Smallest GQA shape on the scalar-gate path.
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
RTOL = ATOL = 1e-3


def assert_close(name, ref, out):
    """Every element must sit within rtol/atol of the torch reference."""
    ref, out = ref.detach().float(), out.detach().float()
    mismatched = checkAllclose(out, ref, rtol=RTOL, atol=ATOL, msg=f"{name} ")
    assert mismatched == 0, (
        f"{name}: {mismatched:.3%} of elements outside rtol={RTOL} atol={ATOL}, "
        f"max abs delta {(out - ref).abs().max().item():.3e}"
    )


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
        "dt_bias": torch.randn(H, K, dtype=torch.float32, device=dev) * 0.1,
        "A_log": torch.randn(H, dtype=torch.float32, device=dev) * 0.5,
        "out": torch.zeros(B, T, H, V, dtype=dt, device=dev),
    }

    d0, d1 = (K, V) if shuffle else (V, K)
    n_slots = B + first_index
    if padded:
        # A serving stack hands the state pool over as one field of a wider
        # per-slot allocation, so the slot stride exceeds the state size. The
        # kernel must index slots by that stride, not assume they are packed.
        storage = torch.randn(n_slots, H * K * V + 17, dtype=torch.float32, device=dev)
        pool = storage[:, : H * K * V].view(n_slots, H, d0, d1)
        # Only the outer stride is padded; each slot stays internally contiguous.
        assert not pool.is_contiguous()
        assert pool.stride()[1:] == (d0 * d1, d1, 1)
    else:
        pool = torch.randn(n_slots, H, d0, d1, dtype=torch.float32, device=dev)

    if indices_stride > 1:
        # The serving stack passes a strided column of a wider index table, not
        # a fresh contiguous array, so the kernel must honour the index stride.
        storage = torch.zeros(B, indices_stride, dtype=torch.int32, device=dev)
        indices = storage[:, 0]
    else:
        indices = torch.empty(B, dtype=torch.int32, device=dev)
    indices.copy_(
        torch.arange(first_index, first_index + B, dtype=torch.int32, device=dev)
    )
    return args, pool, indices


def _clone_pool(pool):
    """Copy the pool keeping its layout; ``clone()`` would densify a padded view
    and hand the kernel a contiguous pool instead of the strided one."""
    out = torch.empty_strided(
        pool.shape, pool.stride(), dtype=pool.dtype, device=pool.device
    )
    out.copy_(pool)
    assert out.stride() == pool.stride(), "kernel pool lost the padded stride"
    return out


def _kda_reference(args, initial_state):
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
        # Every awkward input at once: (D_v, D_k) state layout, padded storage,
        # slots numbered from 1, and a strided index column.
        (4, 12, torch.bfloat16, False, True, 1, 8),
        (4, 12, torch.bfloat16, False, False, 1, 1),
        (4, 12, torch.bfloat16, True, True, 1, 1),
        (4, 12, torch.bfloat16, True, False, 1, 8),
        # The remaining batch sizes that select a tuned launch config, so every
        # tuned geometry runs rather than only the one B=4 picks.
        (1, 12, torch.bfloat16, True, False, 1, 1),
        (64, 12, torch.bfloat16, True, False, 1, 1),
        (256, 12, torch.bfloat16, True, False, 1, 1),
        # A large head count with no tuned config, on the fallback geometry.
        (2, 64, torch.bfloat16, True, False, 1, 1),
    ],
    ids=[
        "b1_h8_bf16",
        "b4_h12_bf16",
        "b2_h12_fp16",
        "all_at_once",
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
    """The KDA gate: 4D `a`, 2D f32 dt_bias, g_min * sigmoid(exp(A_log) * x)."""
    args, pool, indices = _kda_inputs(
        B, H, dt, first_index, padded, shuffle, indices_stride=indices_stride
    )

    initial_state = pool[indices.long()].clone()
    if not shuffle:
        initial_state = initial_state.transpose(-1, -2)

    kernel_pool = _clone_pool(pool)
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

    assert_close("o", ref_out, args["out"])
    assert_close("ht", ref_state, got_state)


def test_channel_strided_a_is_rejected():
    """A gap between `a`'s channels is rejected rather than read as garbage.

    The kernel vector-loads `a` along D_k, and the wrapper forwards `a`'s strides
    instead of copying it the way it copies q/k/v, so a non-dense channel axis
    would read the wrong elements. Only D_k must be dense; other axes stay free.
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
    assert_close("o", ref_out, args["out"])
    assert_close("ht", ref_state, kernel_pool[indices.long()])


def test_negative_slot_is_skipped_and_zero_is_not():
    """A negative slot is skipped; slot 0 is valid and is decoded.

    The kernel guards each row on ``read_pool_idx >= 0 & write_pool_idx >= 0``,
    so a negative index leaves both the output row and the pool untouched, while
    slot 0 goes through like any other slot.
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


def test_tuned_config_lookup_is_keyed_by_gate_mode(monkeypatch):
    """Each gate picks its own tuned config, or the fallback when it has none.

    Both gates share (arch, dtypes, B, Sq, heads, dims), so without gate_mode in
    the key a scalar call at a per-channel-tuned shape would inherit a config
    tuned for a different binary.
    """
    from aiter.ops.flydsl import linear_attention_kernels as lak

    fallback = {"NUM_BLOCKS_PER_V_DIM": 1, "NUM_WARPS": 4, "WARP_THREADS_K": 8}
    kda_config = {"NUM_BLOCKS_PER_V_DIM": 4, "NUM_WARPS": 2, "WARP_THREADS_K": 32}
    gdr_config = {"NUM_BLOCKS_PER_V_DIM": 8, "NUM_WARPS": 4, "WARP_THREADS_K": 16}
    dtypes = ("torch.bfloat16", "torch.float32")
    geometry = (4, 1, 12, 12, 128, 128)
    key = (*dtypes, lak.GDR_GPU_ARCH, *geometry)

    monkeypatch.setattr(
        lak,
        "GDR_GLOBAL_CONFIG_MAP",
        {(*key, "kda"): kda_config, (*key, "gdr"): gdr_config},
    )
    assert lak.get_default_kwargs(*dtypes, *geometry, "kda") == kda_config
    assert lak.get_default_kwargs(*dtypes, *geometry, "gdr") == gdr_config

    # With only a per-channel row, the scalar gate must not inherit it.
    monkeypatch.setattr(lak, "GDR_GLOBAL_CONFIG_MAP", {(*key, "kda"): kda_config})
    assert lak.get_default_kwargs(*dtypes, *geometry, "kda") == kda_config
    assert lak.get_default_kwargs(*dtypes, *geometry, "gdr") == fallback


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
        kw = {"dtype": act_dtype, "device": "cuda"}
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
    kw = {"dtype": torch.float32, "device": "cuda"}
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
