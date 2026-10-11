# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the fused FlyDSL KDA decode specialization."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

pytest.importorskip("flydsl")
from aiter.ops.flydsl.kda_decode import (
    flydsl_kda_decode,
    flydsl_kda_decode_with_f_b,
    is_flydsl_kda_decode_supported,
)


def _gfx950_flydsl_available() -> bool:
    return is_flydsl_kda_decode_supported()


pytestmark = pytest.mark.skipif(
    not _gfx950_flydsl_available(),
    reason="gfx950 FlyDSL required",
)

_DEVICE = torch.device("cuda")
_HEADS = 12
_DIM = 128
_CHANNELS = 3 * _HEADS * _DIM
_CONV_WIDTH = 4
_LOWER_BOUND = -5.0
_NORM_EPS = 1e-5


@dataclass
class Inputs:
    x: torch.Tensor
    conv_weight: torch.Tensor
    conv_state: torch.Tensor
    raw_g: torch.Tensor
    raw_beta: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    state: torch.Tensor
    state_indices: torch.Tensor
    output_gate: torch.Tensor
    norm_weight: torch.Tensor


def _make_inputs(batch: int, seed: int = 20260728) -> Inputs:
    generator = torch.Generator(device=_DEVICE).manual_seed(seed + batch)
    slots = batch + 2

    x_storage = torch.randn(
        (batch, _CHANNELS + 17),
        dtype=torch.bfloat16,
        device=_DEVICE,
        generator=generator,
    )
    x = x_storage[:, :_CHANNELS]
    conv_weight = 0.1 * torch.randn(
        (_CHANNELS, _CONV_WIDTH),
        dtype=torch.float32,
        device=_DEVICE,
        generator=generator,
    )

    # Kimi's hybrid cache can pad the slot stride. Exercise that layout
    # explicitly while keeping each slot's inner dimensions contiguous.
    conv_storage = torch.randn(
        (slots, _CHANNELS * (_CONV_WIDTH - 1) + 19),
        dtype=torch.bfloat16,
        device=_DEVICE,
        generator=generator,
    )
    conv_state = conv_storage[:, : _CHANNELS * (_CONV_WIDTH - 1)].view(
        slots, _CHANNELS, _CONV_WIDTH - 1
    )
    state_storage = 0.01 * torch.randn(
        (slots, _HEADS * _DIM * _DIM + 23),
        dtype=torch.float32,
        device=_DEVICE,
        generator=generator,
    )
    state = state_storage[:, : _HEADS * _DIM * _DIM].view(slots, _HEADS, _DIM, _DIM)

    raw_beta_storage = torch.randn(
        (1, batch, _HEADS + 1),
        dtype=torch.bfloat16,
        device=_DEVICE,
        generator=generator,
    )
    output_gate_storage = torch.randn(
        (batch, _HEADS * _DIM + 7),
        dtype=torch.bfloat16,
        device=_DEVICE,
        generator=generator,
    )
    return Inputs(
        x=x,
        conv_weight=conv_weight,
        conv_state=conv_state,
        raw_g=torch.randn(
            (1, batch, _HEADS, _DIM),
            dtype=torch.bfloat16,
            device=_DEVICE,
            generator=generator,
        ),
        raw_beta=raw_beta_storage[:, :, :_HEADS],
        A_log=0.5
        * torch.randn(
            (_HEADS,),
            dtype=torch.float32,
            device=_DEVICE,
            generator=generator,
        ),
        dt_bias=0.1
        * torch.randn(
            (_HEADS * _DIM,),
            dtype=torch.float32,
            device=_DEVICE,
            generator=generator,
        ),
        state=state,
        state_indices=torch.arange(
            1,
            batch + 1,
            dtype=torch.int32,
            device=_DEVICE,
        ),
        output_gate=output_gate_storage[:, : _HEADS * _DIM].view(batch, _HEADS, _DIM),
        norm_weight=torch.randn(
            (_DIM,),
            dtype=torch.bfloat16,
            device=_DEVICE,
            generator=generator,
        ),
    )


def _copy_inputs(inputs: Inputs) -> Inputs:
    def clone_preserving_strides(tensor: torch.Tensor) -> torch.Tensor:
        clone = torch.empty_strided(
            tensor.shape,
            tensor.stride(),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        clone.copy_(tensor)
        return clone

    return Inputs(
        **{
            name: clone_preserving_strides(getattr(inputs, name))
            for name in Inputs.__dataclass_fields__
        }
    )


def _reference(inputs: Inputs) -> torch.Tensor:
    batch = inputs.x.shape[0]
    output = torch.zeros(
        (1, batch, _HEADS, _DIM),
        dtype=torch.bfloat16,
        device=_DEVICE,
    )
    dt_bias = inputs.dt_bias.view(_HEADS, _DIM)

    for batch_idx in range(batch):
        state_idx = int(inputs.state_indices[batch_idx])
        if state_idx <= 0:
            continue

        history = inputs.conv_state[state_idx]
        conv_values = torch.cat(
            (
                history.float(),
                inputs.x[batch_idx, :, None].float(),
            ),
            dim=-1,
        )
        packed_qkv = F.silu((conv_values * inputs.conv_weight).sum(dim=-1)).to(
            torch.bfloat16
        )
        history[:, 0].copy_(history[:, 1])
        history[:, 1].copy_(history[:, 2])
        history[:, 2].copy_(inputs.x[batch_idx])

        q, k, v = packed_qkv.view(
            3,
            _HEADS,
            _DIM,
        ).unbind(0)
        for head_idx in range(_HEADS):
            q_head = q[head_idx].float()
            k_head = k[head_idx].float()
            q_head = q_head * torch.rsqrt(q_head.square().sum() + 1e-6)
            q_head = q_head * (_DIM**-0.5)
            k_head = k_head * torch.rsqrt(k_head.square().sum() + 1e-6)

            a = inputs.A_log[head_idx].exp()
            decay = (
                _LOWER_BOUND
                * torch.sigmoid(
                    (
                        inputs.raw_g[
                            0,
                            batch_idx,
                            head_idx,
                        ].float()
                        + dt_bias[head_idx]
                    )
                    * a
                )
            ).exp()
            decayed_state = inputs.state[state_idx, head_idx] * decay[None, :]
            state_dot_k = decayed_state @ k_head
            state_dot_q = decayed_state @ q_head
            k_dot_q = torch.dot(k_head, q_head)
            beta = torch.sigmoid(
                inputs.raw_beta[
                    0,
                    batch_idx,
                    head_idx,
                ].float()
            )
            v_new = (v[head_idx].float() - state_dot_k) * beta
            inputs.state[state_idx, head_idx].copy_(
                decayed_state + v_new[:, None] * k_head[None, :]
            )

            # The model materializes recurrent output in BF16 before the
            # normalization/gating operation.
            recurrent = (state_dot_q + v_new * k_dot_q).to(torch.bfloat16)
            recurrent_f32 = recurrent.float()
            inv_rms = torch.rsqrt(recurrent_f32.square().mean() + _NORM_EPS)
            output[0, batch_idx, head_idx] = (
                recurrent_f32
                * inv_rms
                * inputs.norm_weight.float()
                * torch.sigmoid(
                    inputs.output_gate[
                        batch_idx,
                        head_idx,
                    ].float()
                )
            ).to(torch.bfloat16)
    return output


def _relative_rmse(
    reference: torch.Tensor,
    actual: torch.Tensor,
) -> float:
    delta = actual.float() - reference.float()
    return float(
        delta.square().mean().sqrt() / (reference.float().square().mean().sqrt() + 1e-8)
    )


def _run(inputs: Inputs, out: torch.Tensor | None = None) -> torch.Tensor:
    return flydsl_kda_decode(
        x=inputs.x,
        conv_weight=inputs.conv_weight,
        conv_bias=None,
        conv_state=inputs.conv_state,
        raw_g=inputs.raw_g,
        raw_beta=inputs.raw_beta,
        A_log=inputs.A_log,
        dt_bias=inputs.dt_bias,
        lower_bound=_LOWER_BOUND,
        state=inputs.state,
        state_indices=inputs.state_indices,
        output_gate=inputs.output_gate,
        norm_weight=inputs.norm_weight,
        norm_eps=_NORM_EPS,
        out=out,
    )


def _make_fb_inputs(
    batch: int,
    seed: int = 20260728,
) -> tuple[torch.Tensor, torch.Tensor, Inputs]:
    generator = torch.Generator(device=_DEVICE).manual_seed(seed + 10_000 + batch)
    f_a_storage = torch.randn(
        (batch, _DIM + 5),
        dtype=torch.bfloat16,
        device=_DEVICE,
        generator=generator,
    )
    f_a = f_a_storage[:, :_DIM]
    f_b_weight = (
        0.05
        * torch.randn(
            (_HEADS, _DIM, _DIM),
            dtype=torch.bfloat16,
            device=_DEVICE,
            generator=generator,
        )
    ).to(torch.bfloat16)
    inputs = _make_inputs(batch, seed)
    projected = F.linear(
        f_a.float(),
        f_b_weight.view(_HEADS * _DIM, _DIM).float(),
    ).to(torch.bfloat16)
    inputs.raw_g.copy_(projected.view(1, batch, _HEADS, _DIM))
    return f_a, f_b_weight, inputs


def _run_with_f_b(
    f_a: torch.Tensor,
    f_b_weight: torch.Tensor,
    inputs: Inputs,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return flydsl_kda_decode_with_f_b(
        f_a=f_a,
        f_b_weight=f_b_weight,
        x=inputs.x,
        conv_weight=inputs.conv_weight,
        conv_bias=None,
        conv_state=inputs.conv_state,
        raw_beta=inputs.raw_beta,
        A_log=inputs.A_log,
        dt_bias=inputs.dt_bias,
        lower_bound=_LOWER_BOUND,
        state=inputs.state,
        state_indices=inputs.state_indices,
        output_gate=inputs.output_gate,
        norm_weight=inputs.norm_weight,
        norm_eps=_NORM_EPS,
        out=out,
    )


def test_public_api_and_support_predicate() -> None:
    import aiter.ops.flydsl as flydsl_ops

    assert flydsl_ops.flydsl_kda_decode is flydsl_kda_decode
    assert flydsl_ops.is_flydsl_kda_decode_supported is is_flydsl_kda_decode_supported
    assert is_flydsl_kda_decode_supported(0)
    assert not is_flydsl_kda_decode_supported("cpu")


def test_f_b_public_api() -> None:
    import aiter.ops.flydsl as flydsl_ops

    assert flydsl_ops.flydsl_kda_decode_with_f_b is flydsl_kda_decode_with_f_b


def test_decode_module_exports() -> None:
    import aiter.ops.flydsl as flydsl_ops
    from aiter.ops.flydsl import kda_decode

    expected = {
        "flydsl_kda_decode",
        "flydsl_kda_decode_with_f_b",
        "is_flydsl_kda_decode_supported",
    }
    assert set(kda_decode.__all__) == expected
    assert expected <= set(flydsl_ops.__all__)


@pytest.mark.parametrize("batch", [1, 8, 16])
def test_kda_decode_matches_reference(batch: int) -> None:
    seed = _make_inputs(batch)
    reference_inputs = _copy_inputs(seed)
    actual_inputs = _copy_inputs(seed)

    reference = _reference(reference_inputs)
    actual = _run(actual_inputs)
    torch.cuda.synchronize()

    assert is_flydsl_kda_decode_supported(_DEVICE)
    assert not torch.isnan(actual).any()
    assert _relative_rmse(reference, actual) < 1e-3
    assert (
        _relative_rmse(
            reference_inputs.state,
            actual_inputs.state,
        )
        < 1e-3
    )
    assert torch.equal(
        reference_inputs.conv_state,
        actual_inputs.conv_state,
    )


def test_non_positive_slots_do_not_modify_caches() -> None:
    inputs = _make_inputs(batch=2)
    inputs.state_indices.copy_(torch.tensor([0, -1], dtype=torch.int32, device=_DEVICE))
    conv_before = inputs.conv_state.clone()
    state_before = inputs.state.clone()

    actual = _run(inputs)
    torch.cuda.synchronize()

    assert torch.count_nonzero(actual) == 0
    assert torch.equal(inputs.conv_state, conv_before)
    assert torch.equal(inputs.state, state_before)


@pytest.mark.parametrize("batch", [1, 8, 16])
def test_kda_decode_with_f_b_matches_reference(batch: int) -> None:
    f_a, f_b_weight, seed = _make_fb_inputs(batch)
    reference_inputs = _copy_inputs(seed)
    actual_inputs = _copy_inputs(seed)

    reference = _reference(reference_inputs)
    actual = _run_with_f_b(f_a, f_b_weight, actual_inputs)
    torch.cuda.synchronize()

    assert not torch.isnan(actual).any()
    assert _relative_rmse(reference, actual) < 1e-3
    assert _relative_rmse(reference_inputs.state, actual_inputs.state) < 1e-3
    assert torch.equal(reference_inputs.conv_state, actual_inputs.conv_state)


def test_f_b_non_positive_slots_do_not_modify_caches() -> None:
    f_a, f_b_weight, inputs = _make_fb_inputs(batch=2)
    inputs.state_indices.copy_(torch.tensor([0, -1], dtype=torch.int32, device=_DEVICE))
    conv_before = inputs.conv_state.clone()
    state_before = inputs.state.clone()

    actual = _run_with_f_b(f_a, f_b_weight, inputs)
    torch.cuda.synchronize()

    assert torch.count_nonzero(actual) == 0
    assert torch.equal(inputs.conv_state, conv_before)
    assert torch.equal(inputs.state, state_before)


@pytest.mark.parametrize("fused_f_b", [False, True])
def test_out_of_range_slots_do_not_modify_caches(fused_f_b: bool) -> None:
    batch = 4
    if fused_f_b:
        f_a, f_b_weight, inputs = _make_fb_inputs(batch)
    else:
        inputs = _make_inputs(batch)
    # Give the recurrent pool fewer slots than the convolution pool so the
    # bound must come from the smaller of the two caches.
    state_slots = inputs.state.shape[0] - 2
    assert state_slots < inputs.conv_state.shape[0]
    inputs.state = inputs.state[:state_slots]
    inputs.state_indices.copy_(
        torch.tensor(
            [state_slots, state_slots + 1, 2**31 - 1, 0],
            dtype=torch.int32,
            device=_DEVICE,
        )
    )
    conv_before = inputs.conv_state.clone()
    state_before = inputs.state.clone()

    if fused_f_b:
        actual = _run_with_f_b(f_a, f_b_weight, inputs)
    else:
        actual = _run(inputs)
    torch.cuda.synchronize()

    assert torch.count_nonzero(actual) == 0
    assert torch.equal(inputs.conv_state, conv_before)
    assert torch.equal(inputs.state, state_before)


def _move_to_large_caches(
    inputs: Inputs,
    slot_indices: tuple[int, ...],
    slots: int,
    conv_slot_stride: int,
) -> Inputs:
    """Copy each batch slot into zeroed caches large enough to overflow 32-bit
    byte offsets, at the high ``slot_indices``."""
    conv_storage = torch.zeros(
        (slots, conv_slot_stride),
        dtype=torch.bfloat16,
        device=_DEVICE,
    )
    conv_state = conv_storage[:, : _CHANNELS * (_CONV_WIDTH - 1)].view(
        slots, _CHANNELS, _CONV_WIDTH - 1
    )
    state = torch.zeros(
        (slots, _HEADS, _DIM, _DIM),
        dtype=torch.float32,
        device=_DEVICE,
    )
    for batch_idx, slot in enumerate(slot_indices):
        small_slot = int(inputs.state_indices[batch_idx])
        conv_state[slot].copy_(inputs.conv_state[small_slot])
        state[slot].copy_(inputs.state[small_slot])
    large = _copy_inputs(inputs)
    large.conv_state = conv_state
    large.state = state
    large.state_indices = torch.tensor(
        slot_indices,
        dtype=torch.int32,
        device=_DEVICE,
    )
    return large


@pytest.mark.parametrize("fused_f_b", [False, True])
def test_large_cache_slot_offsets(fused_f_b: bool) -> None:
    # 11,000 [12, 128, 128] FP32 slots cross 2**31 elements of state, and a
    # 2**18-element BF16 conv slot stride crosses 2**32 bytes past slot 8,192.
    slots = 11_000
    slot_indices = (5_500, 10_999)
    batch = len(slot_indices)
    if fused_f_b:
        f_a, f_b_weight, seed = _make_fb_inputs(batch)
    else:
        seed = _make_inputs(batch)
    reference_inputs = _copy_inputs(seed)
    actual_inputs = _move_to_large_caches(
        seed,
        slot_indices,
        slots,
        conv_slot_stride=1 << 18,
    )

    reference = _reference(reference_inputs)
    if fused_f_b:
        actual = _run_with_f_b(f_a, f_b_weight, actual_inputs)
    else:
        actual = _run(actual_inputs)
    torch.cuda.synchronize()

    assert _relative_rmse(reference, actual) < 1e-3
    for batch_idx, slot in enumerate(slot_indices):
        small_slot = int(reference_inputs.state_indices[batch_idx])
        assert (
            _relative_rmse(
                reference_inputs.state[small_slot],
                actual_inputs.state[slot],
            )
            < 1e-3
        )
        assert torch.equal(
            reference_inputs.conv_state[small_slot],
            actual_inputs.conv_state[slot],
        )
        actual_inputs.state[slot].zero_()
        actual_inputs.conv_state[slot].zero_()
    # Nothing outside the addressed slots may be touched by wrapped offsets.
    assert torch.count_nonzero(actual_inputs.state) == 0
    assert torch.count_nonzero(actual_inputs.conv_state) == 0


def test_f_b_api_rejects_invalid_projection_inputs() -> None:
    f_a, f_b_weight, inputs = _make_fb_inputs(batch=1)

    with pytest.raises(ValueError, match="`f_a` must have rank 2"):
        _run_with_f_b(f_a.unsqueeze(0), f_b_weight, inputs)
    with pytest.raises(ValueError, match="`f_a` must have dtype"):
        _run_with_f_b(f_a.float(), f_b_weight, inputs)
    with pytest.raises(ValueError, match="`f_b_weight` must have shape"):
        _run_with_f_b(f_a, f_b_weight[:, :, :-1], inputs)
    with pytest.raises(ValueError, match="`f_b_weight` must have inner strides"):
        _run_with_f_b(f_a, f_b_weight.transpose(1, 2), inputs)


def test_decode_api_rejects_invalid_input_rank() -> None:
    inputs = _make_inputs(batch=1)
    inputs.x = inputs.x.unsqueeze(0)

    with pytest.raises(ValueError, match="`x` must have rank 2"):
        _run(inputs)


@pytest.mark.parametrize("fused_f_b", [False, True])
@pytest.mark.parametrize(
    "name, strides",
    [
        ("conv_state", (0, 3, 1)),
        ("conv_state", (_CHANNELS * 3 - 1, 3, 1)),
        ("conv_state", (_CHANNELS * 3, 0, 1)),
        ("conv_state", (_CHANNELS * 3, 3, 0)),
        ("conv_state", (_CHANNELS * 6, 6, 2)),
        ("state", (0, _DIM * _DIM, _DIM, 1)),
        ("state", (_HEADS * _DIM * _DIM - 1, _DIM * _DIM, _DIM, 1)),
        ("out", (0, _HEADS * _DIM, 0, 1)),
        ("out", (0, _HEADS * _DIM, _DIM - 1, 1)),
        ("out", (0, 0, _DIM, 1)),
        ("out", (0, (_HEADS - 1) * (_DIM + 7) + _DIM - 1, _DIM + 7, 1)),
    ],
)
def test_api_rejects_unsafe_write_layouts(
    monkeypatch: pytest.MonkeyPatch,
    fused_f_b: bool,
    name: str,
    strides: tuple[int, ...],
) -> None:
    from aiter.ops.flydsl import kda_decode

    def unexpected_compile(*args):
        pytest.fail("Unsafe write layouts must be rejected before kernel compilation.")

    monkeypatch.setattr(kda_decode, "create_kda_decode_kernel", unexpected_compile)
    monkeypatch.setattr(
        kda_decode, "create_kda_decode_fused_projection_kernel", unexpected_compile
    )
    with FakeTensorMode():
        f_a, f_b_weight, inputs = _make_fb_inputs(batch=2)
        out = torch.empty((1, 2, _HEADS, _DIM), dtype=torch.bfloat16, device=_DEVICE)
        tensor = out if name == "out" else getattr(inputs, name)
        view = torch.empty_strided(
            tensor.shape, strides, dtype=tensor.dtype, device=_DEVICE
        )
        if name == "out":
            out = view
        else:
            setattr(inputs, name, view)
        with pytest.raises(ValueError, match=f"`{name}`.*non-overlapping"):
            if fused_f_b:
                _run_with_f_b(f_a, f_b_weight, inputs, out=out)
            else:
                _run(inputs, out=out)


@pytest.mark.parametrize("fused_f_b", [False, True])
@pytest.mark.parametrize("padding", [0, 7])
@pytest.mark.parametrize("trailing_padding", [0, 17])
def test_api_accepts_non_overlapping_write_layouts(
    fused_f_b: bool,
    padding: int,
    trailing_padding: int,
) -> None:
    f_a, f_b_weight, inputs = _make_fb_inputs(batch=2)
    if padding == 0:
        inputs.conv_state = inputs.conv_state.contiguous()
        inputs.state = inputs.state.contiguous()
    reference_inputs = _copy_inputs(inputs)
    reference = _reference(reference_inputs)
    head_stride = _DIM + padding
    batch_stride = (_HEADS - 1) * head_stride + _DIM + trailing_padding
    out = torch.empty_strided(
        (1, 2, _HEADS, _DIM),
        (0, batch_stride, head_stride, 1),
        dtype=torch.bfloat16,
        device=_DEVICE,
    )
    if fused_f_b:
        actual = _run_with_f_b(f_a, f_b_weight, inputs, out=out)
    else:
        actual = _run(inputs, out=out)
    torch.cuda.synchronize()

    assert actual is out
    assert _relative_rmse(reference, actual) < 1e-3
    assert _relative_rmse(reference_inputs.state, inputs.state) < 1e-3
    assert torch.equal(reference_inputs.conv_state, inputs.conv_state)


@pytest.mark.parametrize(
    "fused_f_b, name, dim",
    # Convolution inner strides are fixed by dense-cache validation, so their
    # byte span cannot overflow. Invalid layouts are covered above instead.
    [
        (fused_f_b, name, dim)
        for fused_f_b in (False, True)
        for name, dim in (
            ("x", 0),
            ("conv_weight", 0),
            ("conv_weight", 1),
            ("raw_beta", 1),
            ("output_gate", 0),
            ("output_gate", 1),
            ("out", 1),
            ("out", 2),
        )
    ]
    + [(False, "raw_g", 1), (True, "f_a", 0), (True, "f_b_weight", 0)],
)
@pytest.mark.parametrize("stride", [2**30, 2**31])
def test_api_rejects_i32_address_overflow(
    monkeypatch: pytest.MonkeyPatch,
    fused_f_b: bool,
    name: str,
    dim: int,
    stride: int,
) -> None:
    from aiter.ops.flydsl import kda_decode

    def unexpected_compile(*args):
        pytest.fail("Unsafe addressing must be rejected before kernel compilation.")

    monkeypatch.setattr(kda_decode, "create_kda_decode_kernel", unexpected_compile)
    monkeypatch.setattr(
        kda_decode, "create_kda_decode_fused_projection_kernel", unexpected_compile
    )
    # Fake views exercise multi-GiB spans without allocating or accessing them.
    # 2**30 fits the launcher parameter but overflows the resulting byte span;
    # 2**31 also overflows the stride parameter itself.
    with FakeTensorMode():
        inputs = _make_inputs(batch=2)
        f_a = torch.empty((2, _DIM), dtype=torch.bfloat16, device=_DEVICE)
        f_b_weight = torch.empty(
            (_HEADS, _DIM, _DIM), dtype=torch.bfloat16, device=_DEVICE
        )
        out = torch.empty((1, 2, _HEADS, _DIM), dtype=torch.bfloat16, device=_DEVICE)
        tensor = {"f_a": f_a, "f_b_weight": f_b_weight, "out": out}.get(name)
        if tensor is None:
            tensor = getattr(inputs, name)
        strides = list(tensor.stride())
        strides[dim] = stride
        view = torch.empty_strided(
            tensor.shape, strides, dtype=tensor.dtype, device=_DEVICE
        )
        if name == "f_a":
            f_a = view
        elif name == "f_b_weight":
            f_b_weight = view
        elif name == "out":
            out = view
        else:
            setattr(inputs, name, view)
        with pytest.raises(ValueError, match=f"`{name}`.*int32"):
            if fused_f_b:
                _run_with_f_b(f_a, f_b_weight, inputs, out=out)
            else:
                _run(inputs, out=out)


def test_unused_singleton_stride_must_still_fit_launcher_int32() -> None:
    from aiter.ops.flydsl.kda_decode import _check_i32_addressing

    tensor = torch.empty_strided(
        (1, _CHANNELS), (2**31, 1), dtype=torch.bfloat16, device="meta"
    )
    with pytest.raises(ValueError, match="`x` strides.*int32"):
        _check_i32_addressing("x", tensor)


def test_i32_byte_span_boundary() -> None:
    from aiter.ops.flydsl.kda_decode import _check_i32_addressing

    # The starting offset fits for both tensors; the entire final BF16 element
    # must also fit. Use meta tensors to avoid allocating nearly two GiB.
    safe = torch.empty_strided((2,), (2**30 - 2,), dtype=torch.bfloat16, device="meta")
    unsafe = torch.empty_strided(
        (2,), (2**30 - 1,), dtype=torch.bfloat16, device="meta"
    )
    _check_i32_addressing("x", safe)
    with pytest.raises(ValueError, match="`x` byte span.*int32"):
        _check_i32_addressing("x", unsafe)
