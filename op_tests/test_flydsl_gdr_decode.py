# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CI-visible tests for the FlyDSL GDR decode op: its export, and its numerics.

CI collects only ``op_tests/test_*.py`` at depth 1 (see
``.github/scripts/split_tests.sh``), so the kernel's in-package suite at
``aiter/ops/flydsl/test_flydsl_linear_attention.py`` never runs automatically.
This file puts both properties under CI.

The export checks matter on their own: reaching into ``linear_attention_kernels``
keeps working if the export drops back out of ``__init__.py``, so no other test
would notice that regression. The scalar numeric checks reuse the in-package
harness rather than carrying a second copy of its Triton reference.

The KDA cases trust the torch reference and put the *kernel* on trial.
``op_tests/test_kda_torch_ref.py`` runs that arrow backwards -- it trusts the
shipping scalar kernel and puts the *reference* on trial. The two therefore look
alike without either subsuming the other, and which one fails says where the
fault is: here alone means the per-channel kernel, there alone means the
reference, both at once means the gate formula they share.

Keep it thin -- shapes belong here, the reference kernel belongs in-package.
"""

import pytest
import torch

import aiter.ops.flydsl as flydsl_ops
from aiter.ops.flydsl import is_flydsl_available

pytestmark = pytest.mark.skipif(
    not is_flydsl_available(), reason="flydsl is not installed"
)

# Skips itself at import when flydsl or a GPU is missing, which skips this module
# with it.
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
    """The reference implementation's own bar, from vLLM ``test_kda.py:92``.

    RMSE-relative with an absolute escape hatch, used at the packed-vs-dense
    decode pair (1e-3, 1e-3). The ratio alone is not the bar.
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

    ``need_shuffle_state`` decides which state layout the caller holds: False
    means it is already (D_v, D_k), True means (D_k, D_v) and the wrapper
    transposes. With K == V the two are shape-identical and differ only in
    meaning, so feeding the wrong one measures the harness, not the kernel.
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
        # vLLM's setup: rows are padded, so the state is deliberately
        # non-contiguous and the kernel has to honour the strides it is given.
        storage = torch.randn(n_slots, H * K * V + 17, dtype=torch.float32, device=dev)
        pool = storage[:, : H * K * V].view(n_slots, H, d0, d1)
        # Both assertions are vLLM's (test_kda.py:321-322). The second is the
        # one that bites: a slot must still be internally contiguous, so only
        # the outer stride is padded. Assert it rather than assume .view() gave
        # us that, or a future change to the storage shape could silently start
        # testing a layout the kernel never sees.
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
        # K3 as deployed: state already (D_v, D_k), padded storage, slots
        # numbered from 1 because the serving stack treats 0 as invalid, and a
        # strided index column -- every landmine in vLLM's setup at once.
        (4, 12, torch.bfloat16, False, True, 1, 8),
        (4, 12, torch.bfloat16, False, False, 1, 1),
        (4, 12, torch.bfloat16, True, True, 1, 1),
        (4, 12, torch.bfloat16, True, False, 1, 8),
    ],
    ids=[
        "b1_h8_bf16",
        "b4_h12_bf16",
        "b2_h12_fp16",
        "k3_deployed",
        "no_shuffle",
        "padded_state",
        "strided_indices",
    ],
)
def test_kda_per_channel_gate_matches_torch_reference(
    B, H, dt, shuffle, padded, first_index, indices_stride
):
    """The KDA gate: 4D `a`, 2D f32 dt_bias, g_min * sigmoid(exp(A_log) * x).

    H is 1:1 with the k heads, which is K3's ratio and a case the scalar path
    never exercised.
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

    # Tripwire under the acceptance bar above: agreement is ~1e-4 in practice,
    # so a drift to just inside 1e-3 would pass the bar while signalling that
    # something in the gate or the recurrence changed.
    assert out_err < 5e-4 and state_err < 5e-4, (out_err, state_err)


def test_negative_slot_is_skipped_and_zero_is_not():
    """Pins this kernel's invalid-slot contract, which is not the reference's.

    aiter skips ``pool_idx < 0`` and writes nothing -- leaving the caller's
    output buffer untouched. The KDA reference instead treats ``state_idx <= 0``
    as invalid and zero-fills. They differ on both the boundary and the
    behaviour, so slot 0 being *processed* here is the point of the test.
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
