# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch.nn import functional as F

from aiter import gdr_decode_packed_bf16

NUM_QK_HEADS = 8
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
QKV_DIM = 6144
SCALE = HEAD_K_DIM**-0.5
RTOL = 1.0e-2
ATOL = 1.0e-3


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return getattr(props, "gcnArchName", "").split(":")[0] == "gfx950"
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _is_gfx950(), reason="packed BF16 GDR decode requires gfx950"
)


def _normal(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    std: float,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return (
        torch.empty(shape, device="cuda", dtype=torch.float32)
        .normal_(0.0, std, generator=generator)
        .to(dtype)
    )


def _make_inputs(
    batch: int,
    pool: int,
    *,
    seed: int,
    strided: bool = False,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    if strided:
        mixed_storage = _normal((batch, 2, QKV_DIM), generator=generator, std=0.1)
        a_storage = _normal((batch, 2, NUM_V_HEADS), generator=generator, std=0.2)
        b_storage = _normal((batch, 2, NUM_V_HEADS), generator=generator, std=0.2)
        state_storage = _normal(
            (pool, 2, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM),
            generator=generator,
            std=0.02,
        )
        mixed_qkv = mixed_storage[:, 0, :]
        a = a_storage[:, 0, :]
        b = b_storage[:, 0, :]
        state = state_storage[:, 0, :, :, :]
        assert mixed_qkv.stride(0) > QKV_DIM
        assert a.stride(0) > NUM_V_HEADS
        assert state.stride(0) > NUM_V_HEADS * HEAD_V_DIM * HEAD_K_DIM
    else:
        mixed_qkv = _normal((batch, QKV_DIM), generator=generator, std=0.1)
        a = _normal((batch, NUM_V_HEADS), generator=generator, std=0.2)
        b = _normal((batch, NUM_V_HEADS), generator=generator, std=0.2)
        state = _normal(
            (pool, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM),
            generator=generator,
            std=0.02,
        )

    dt_bias = _normal((NUM_V_HEADS,), generator=generator, std=0.1)
    A_log = torch.empty((NUM_V_HEADS,), device="cuda", dtype=torch.float32).uniform_(
        -3.0, -1.0, generator=generator
    )
    out = torch.empty(
        (batch, 1, NUM_V_HEADS, HEAD_V_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )
    return {
        "mixed_qkv": mixed_qkv,
        "a": a,
        "b": b,
        "dt_bias": dt_bias,
        "A_log": A_log,
        "state": state,
        "out": out,
    }


def _reference(
    inputs: dict[str, torch.Tensor], indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mixed_qkv = inputs["mixed_qkv"]
    batch = mixed_qkv.shape[0]
    state = inputs["state"].clone()

    q = mixed_qkv[:, : NUM_QK_HEADS * HEAD_K_DIM].reshape(
        batch, NUM_QK_HEADS, HEAD_K_DIM
    )
    k = mixed_qkv[:, NUM_QK_HEADS * HEAD_K_DIM : 2 * NUM_QK_HEADS * HEAD_K_DIM].reshape(
        batch, NUM_QK_HEADS, HEAD_K_DIM
    )
    v = mixed_qkv[:, 2 * NUM_QK_HEADS * HEAD_K_DIM :].reshape(
        batch, NUM_V_HEADS, HEAD_V_DIM
    )

    q = q.float()
    k = k.float()
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1.0e-6) * SCALE
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1.0e-6)
    q = q.repeat_interleave(NUM_V_HEADS // NUM_QK_HEADS, dim=1)
    k = k.repeat_interleave(NUM_V_HEADS // NUM_QK_HEADS, dim=1)

    gate_x = inputs["a"].float() + inputs["dt_bias"].float().unsqueeze(0)
    softplus = torch.where(gate_x <= 20.0, F.softplus(gate_x), gate_x)
    decay = torch.exp(-torch.exp(inputs["A_log"].float()).unsqueeze(0) * softplus)
    beta = torch.sigmoid(inputs["b"].float()).to(torch.bfloat16).float()

    out = torch.zeros(
        (batch, 1, NUM_V_HEADS, HEAD_V_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )
    pool = state.shape[0]
    for row, state_idx in enumerate(indices.cpu().tolist()):
        if not 0 <= state_idx < pool:
            continue
        recurrent = state[state_idx].float() * decay[row, :, None, None]
        sum_hk = (recurrent * k[row, :, None, :]).sum(-1)
        sum_hq = (recurrent * q[row, :, None, :]).sum(-1)
        dot_kq = (k[row] * q[row]).sum(-1)
        residual = (v[row].float() - sum_hk) * beta[row, :, None]
        out[row, 0] = (sum_hq + residual * dot_kq[:, None]).to(torch.bfloat16)
        state[state_idx] = (recurrent + residual[:, :, None] * k[row, :, None, :]).to(
            torch.bfloat16
        )
    return out, state


def _run(
    inputs: dict[str, torch.Tensor], indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return gdr_decode_packed_bf16(
        mixed_qkv=inputs["mixed_qkv"],
        a=inputs["a"],
        b=inputs["b"],
        dt_bias=inputs["dt_bias"],
        A_log=inputs["A_log"],
        indices=indices,
        state=inputs["state"],
        out=inputs["out"],
        scale=SCALE,
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.float(), expected.float(), rtol=RTOL, atol=ATOL)


def test_valid_accuracy_alias_and_untouched_slots() -> None:
    inputs = _make_inputs(batch=4, pool=7, seed=2026080301)
    indices = torch.tensor([5, 0, 6, 2], device="cuda", dtype=torch.int32)
    state_before = inputs["state"].clone()
    expected_out, expected_state = _reference(inputs, indices)

    actual_out, actual_state = _run(inputs, indices)
    torch.cuda.synchronize()

    assert actual_out.data_ptr() == inputs["out"].data_ptr()
    assert actual_state.data_ptr() == inputs["state"].data_ptr()
    _assert_close(actual_out, expected_out)
    _assert_close(actual_state, expected_state)
    untouched = torch.tensor([1, 3, 4], device="cuda")
    assert torch.equal(
        actual_state.index_select(0, untouched).view(torch.int16),
        state_before.index_select(0, untouched).view(torch.int16),
    )


def test_invalid_indices_write_positive_zero_without_state_update() -> None:
    inputs = _make_inputs(batch=4, pool=3, seed=2026080302)
    state_before = inputs["state"].clone()
    indices = torch.tensor([-1, 3, 17, 2**31 - 1], device="cuda", dtype=torch.int32)

    actual_out, actual_state = _run(inputs, indices)
    torch.cuda.synchronize()

    assert torch.count_nonzero(actual_out.view(torch.int16)).item() == 0
    assert torch.equal(actual_state.view(torch.int16), state_before.view(torch.int16))


def test_mixed_indices_and_strided_rows() -> None:
    batch, pool = 6, 7
    inputs = _make_inputs(batch=batch, pool=pool, seed=2026080303, strided=True)
    index_storage = torch.full((batch * 2,), -99, device="cuda", dtype=torch.int32)
    index_storage[::2] = torch.tensor(
        [0, -1, pool, 4, 2**31 - 1, 6], device="cuda", dtype=torch.int32
    )
    indices = index_storage[::2]
    assert indices.stride(0) == 2
    state_before = inputs["state"].clone()
    expected_out, expected_state = _reference(inputs, indices)

    actual_out, actual_state = _run(inputs, indices)
    torch.cuda.synchronize()

    _assert_close(actual_out, expected_out)
    _assert_close(actual_state, expected_state)
    invalid_rows = torch.tensor([1, 2, 4], device="cuda")
    assert (
        torch.count_nonzero(
            actual_out.index_select(0, invalid_rows).view(torch.int16)
        ).item()
        == 0
    )
    untouched = torch.tensor([1, 2, 3, 5], device="cuda")
    assert torch.equal(
        actual_state.index_select(0, untouched).view(torch.int16),
        state_before.index_select(0, untouched).view(torch.int16),
    )


def test_multi_step_bf16_state_drift() -> None:
    batch, pool, steps = 4, 6, 16
    indices = torch.tensor([5, 0, 3, 1], device="cuda", dtype=torch.int32)
    initial = _make_inputs(batch=batch, pool=pool, seed=2026080600)
    actual_state = initial["state"].clone()
    expected_state = initial["state"].clone()
    untouched_before = actual_state[2].clone()

    for step in range(steps):
        step_inputs = _make_inputs(
            batch=batch,
            pool=pool,
            seed=2026080601 + step,
        )

        reference_inputs = {**step_inputs, "state": expected_state}
        expected_out, expected_state = _reference(reference_inputs, indices)

        actual_inputs = {**step_inputs, "state": actual_state}
        actual_out, returned_state = _run(actual_inputs, indices)
        assert returned_state.data_ptr() == actual_state.data_ptr()
        _assert_close(actual_out, expected_out)
        _assert_close(returned_state, expected_state)
        actual_state = returned_state

    torch.cuda.synchronize()
    assert torch.equal(
        actual_state[2].view(torch.int16), untouched_before.view(torch.int16)
    )


def _bad_scale(inputs: dict[str, torch.Tensor]) -> dict[str, object]:
    return {"scale": 1.0}


def _bad_indices_dtype(inputs: dict[str, torch.Tensor]) -> dict[str, object]:
    return {"indices": torch.arange(2, device="cuda", dtype=torch.int64)}


def _bad_state_layout(inputs: dict[str, torch.Tensor]) -> dict[str, object]:
    return {"state": inputs["state"].transpose(-1, -2)}


def _bad_out_layout(inputs: dict[str, torch.Tensor]) -> dict[str, object]:
    out = torch.empty(
        (2, 1, NUM_V_HEADS, HEAD_V_DIM * 2),
        device="cuda",
        dtype=torch.bfloat16,
    )[..., ::2]
    return {"out": out}


@pytest.mark.parametrize(
    "mutation",
    [_bad_scale, _bad_indices_dtype, _bad_state_layout, _bad_out_layout],
)
def test_validation_errors(
    mutation: Callable[[dict[str, torch.Tensor]], dict[str, object]],
) -> None:
    inputs = _make_inputs(batch=2, pool=2, seed=2026080304)
    kwargs: dict[str, object] = {
        **inputs,
        "indices": torch.arange(2, device="cuda", dtype=torch.int32),
        "scale": SCALE,
    }
    kwargs.update(mutation(inputs))
    with pytest.raises(ValueError):
        gdr_decode_packed_bf16(**kwargs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
