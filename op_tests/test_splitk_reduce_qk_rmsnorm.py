# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import pytest
import torch

from aiter import splitk_reduce_qk_rmsnorm
from aiter.ops.enum import QuantType
from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm

PLANES = 6
Q_DIM = 2048
KV_DIM = 512
OUT_DIM = 2624
EPS = 1e-5
# The HIP reducer supports arbitrary row counts; producer dispatch is separate.
TEST_M = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return getattr(props, "gcnArchName", "").split(":")[0] == "gfx950"
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _is_gfx950(), reason="splitk_reduce_qk_rmsnorm is built for gfx950"
)


def _inputs(m: int, seed: int = 2624):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    partial = torch.randn(
        (PLANES, m, OUT_DIM), device="cuda", dtype=torch.float32, generator=generator
    )
    q_weight = (torch.rand(Q_DIM, device="cuda", generator=generator) + 0.5).to(
        torch.bfloat16
    )
    k_weight = (torch.rand(KV_DIM, device="cuda", generator=generator) + 0.5).to(
        torch.bfloat16
    )
    return partial, q_weight, k_weight


def _fold(partial: torch.Tensor) -> torch.Tensor:
    total = partial[0] + 0
    for plane in range(1, PLANES):
        total = total + partial[plane]
    return total.to(torch.bfloat16)


def _rmsnorm_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    scale = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + EPS)
    return (xf * scale * weight.float()).to(torch.bfloat16)


@pytest.mark.parametrize("m", TEST_M)
def test_matches_fixed_order_fold_and_fused_qk_rmsnorm(m: int) -> None:
    partial, q_weight, k_weight = _inputs(m)

    out, q_out, k_out = splitk_reduce_qk_rmsnorm(partial, q_weight, EPS, k_weight, EPS)

    expected = _fold(partial)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))

    q_ref = torch.empty_like(q_out)
    k_ref = torch.empty_like(k_out)
    fused_qk_rmsnorm(
        q_out_quantized=q_ref,
        k_out=k_ref,
        q=expected[:, :Q_DIM],
        q_weight=q_weight,
        q_epsilon=EPS,
        k=expected[:, Q_DIM : Q_DIM + KV_DIM],
        k_weight=k_weight,
        k_epsilon=EPS,
        quant_type=QuantType.No,
    )
    assert torch.equal(q_out.view(torch.int16), q_ref.view(torch.int16))
    assert torch.equal(k_out.view(torch.int16), k_ref.view(torch.int16))

    torch.testing.assert_close(
        q_out, _rmsnorm_fp32(expected[:, :Q_DIM], q_weight), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        k_out,
        _rmsnorm_fp32(expected[:, Q_DIM : Q_DIM + KV_DIM], k_weight),
        atol=2e-2,
        rtol=2e-2,
    )


def test_writes_into_given_outputs() -> None:
    partial, q_weight, k_weight = _inputs(128)
    out = torch.empty((128, OUT_DIM), device="cuda", dtype=torch.bfloat16)
    q_out = torch.empty((128, Q_DIM), device="cuda", dtype=torch.bfloat16)
    k_out = torch.empty((128, KV_DIM), device="cuda", dtype=torch.bfloat16)

    returned = splitk_reduce_qk_rmsnorm(
        partial, q_weight, EPS, k_weight, EPS, out=out, q_out=q_out, k_out=k_out
    )

    assert all(a is b for a, b in zip(returned, (out, q_out, k_out)))
    assert torch.equal(out.view(torch.int16), _fold(partial).view(torch.int16))


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((5, 128, OUT_DIM), torch.float32),
        ((PLANES, 128, OUT_DIM - 64), torch.float32),
        ((PLANES, 128, OUT_DIM), torch.bfloat16),
    ],
)
def test_rejects_other_partial_layouts(shape, dtype) -> None:
    _, q_weight, k_weight = _inputs(128)
    partial = torch.zeros(shape, device="cuda", dtype=dtype)
    with pytest.raises(ValueError):
        splitk_reduce_qk_rmsnorm(partial, q_weight, EPS, k_weight, EPS)


def test_rejects_non_contiguous_partial() -> None:
    _, q_weight, k_weight = _inputs(128)
    partial = torch.zeros(
        (PLANES, 128, 2 * OUT_DIM), device="cuda", dtype=torch.float32
    )[..., ::2]
    with pytest.raises(ValueError):
        splitk_reduce_qk_rmsnorm(partial, q_weight, EPS, k_weight, EPS)


@pytest.mark.parametrize("m", TEST_M)
def test_graph_replay_overwrites_outputs_and_preserves_inputs(m):
    partial, qw, kw = _inputs(m)
    outputs = (
        torch.empty((m, OUT_DIM), device="cuda", dtype=torch.bfloat16),
        torch.empty((m, Q_DIM), device="cuda", dtype=torch.bfloat16),
        torch.empty((m, KV_DIM), device="cuda", dtype=torch.bfloat16),
    )
    q_eps, k_eps = 1e-5, 1e-3

    def call():
        return splitk_reduce_qk_rmsnorm(
            partial,
            qw,
            q_eps,
            kw,
            k_eps,
            out=outputs[0],
            q_out=outputs[1],
            k_out=outputs[2],
        )

    call()  # Compile before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for pattern in ("random", "zero", "cancellation"):
        fresh, new_qw, new_kw = _inputs(m, seed=9000)
        if pattern == "zero":
            fresh.zero_()
        elif pattern == "cancellation":
            # Cancel large adjacent planes, retain small later contributions.
            fresh[0].mul_(1024)
            fresh[1].copy_(-fresh[0])
        partial.copy_(fresh)
        qw.copy_(new_qw)
        kw.copy_(new_kw)
        saved = tuple(t.clone() for t in (partial, qw, kw))
        call()
        eager = tuple(t.clone() for t in outputs)
        expected = _fold(partial)
        assert torch.equal(eager[0].view(torch.int16), expected.view(torch.int16))
        for got, x, weight, epsilon in (
            (eager[1], expected[:, :Q_DIM], qw, q_eps),
            (eager[2], expected[:, Q_DIM : Q_DIM + KV_DIM], kw, k_eps),
        ):
            xf = x.float()
            ref = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + epsilon)
            ref *= weight.float()
            torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)
            if ref.square().mean() > 0:
                nrmse = (
                    got.float() - ref
                ).square().mean().sqrt() / ref.square().mean().sqrt()
                assert nrmse < 0.01
        for _ in range(3):
            for output in outputs:
                output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            for actual, reference in zip(outputs, eager):
                assert torch.equal(
                    actual.view(torch.int16), reference.view(torch.int16)
                )
            for actual, reference in zip((partial, qw, kw), saved):
                assert torch.equal(actual, reference)


def _arguments(device="cuda"):
    return dict(
        partial=torch.empty((PLANES, 1, OUT_DIM), dtype=torch.float32, device=device),
        q_weight=torch.ones(Q_DIM, dtype=torch.bfloat16, device=device),
        k_weight=torch.ones(KV_DIM, dtype=torch.bfloat16, device=device),
        out=torch.empty((1, OUT_DIM), dtype=torch.bfloat16, device=device),
        q_out=torch.empty((1, Q_DIM), dtype=torch.bfloat16, device=device),
        k_out=torch.empty((1, KV_DIM), dtype=torch.bfloat16, device=device),
        q_eps=EPS,
        k_eps=EPS,
    )


@pytest.mark.parametrize(
    "name", ["partial", "q_weight", "k_weight", "out", "q_out", "k_out"]
)
def test_rejects_contiguous_misaligned_storage(name):
    args = _arguments()
    tensor = args[name]
    # Slicing by one element preserves shape/contiguity but breaks vector alignment.
    args[name] = torch.empty(
        tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device
    )[1:].view(tensor.shape)
    assert args[name].is_contiguous()
    with pytest.raises(ValueError, match="aligned"):
        splitk_reduce_qk_rmsnorm(**args)


def test_rejects_cpu_tensors_before_launch():
    with pytest.raises(ValueError, match="GPU tensor"):
        splitk_reduce_qk_rmsnorm(**_arguments("cpu"))


@pytest.mark.parametrize("name", ["q_weight", "k_weight", "out", "q_out", "k_out"])
@pytest.mark.parametrize("native", [False, True])
def test_rejects_mixed_gpu_devices(name, native):
    from aiter.ops.splitk_reduce_qk_rmsnorm import _splitk_reduce_qk_rmsnorm

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    args = _arguments("cuda:0")
    args[name] = args[name].to("cuda:1")
    fn = _splitk_reduce_qk_rmsnorm if native else splitk_reduce_qk_rmsnorm
    with pytest.raises(
        RuntimeError if native else ValueError, match="same (device|GPU)"
    ):
        fn(**args)


@pytest.mark.parametrize("fault", ["misaligned partial", "cpu output"])
def test_native_binding_rejects_unsafe_buffers(fault):
    from aiter.ops.splitk_reduce_qk_rmsnorm import _splitk_reduce_qk_rmsnorm

    args = _arguments()
    if fault == "misaligned partial":
        p = args["partial"]
        args["partial"] = torch.empty(p.numel() + 1, dtype=p.dtype, device=p.device)[
            1:
        ].view(p.shape)
        message = "aligned"
    else:
        args["out"] = args["out"].cpu()
        message = "GPU tensor"
    # Bypass the Python public wrapper: the C++ boundary must reject these too.
    with pytest.raises(RuntimeError, match=message):
        _splitk_reduce_qk_rmsnorm(**args)


def test_empty_rows_are_a_noop():
    partial, qw, kw = _inputs(0)
    out, q, k = splitk_reduce_qk_rmsnorm(partial, qw, EPS, kw, EPS)
    assert tuple(out.shape) == (0, OUT_DIM)
    assert tuple(q.shape) == (0, Q_DIM)
    assert tuple(k.shape) == (0, KV_DIM)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("fault", ["output/output", "output/input"])
def test_rejects_overlapping_buffers(native, fault):
    from aiter.ops.splitk_reduce_qk_rmsnorm import _splitk_reduce_qk_rmsnorm

    args = _arguments()
    if fault == "output/output":
        args["q_out"] = args["out"][:, :Q_DIM]
    else:
        args["q_weight"] = args["q_out"].view(Q_DIM)
    fn = _splitk_reduce_qk_rmsnorm if native else splitk_reduce_qk_rmsnorm
    with pytest.raises(RuntimeError if native else ValueError, match="overlap"):
        fn(**args)


def test_restores_current_device_after_other_gpu_call():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    stream0 = torch.cuda.Stream(device=0)
    stream1 = torch.cuda.Stream(device=1)
    with torch.cuda.stream(stream1):
        args = _arguments("cuda:1")
        args["partial"].zero_()
        with torch.cuda.stream(stream0):
            result = splitk_reduce_qk_rmsnorm(**args)
            assert torch.cuda.current_device() == 0
            assert torch.cuda.current_stream() == stream0
        stream1.synchronize()
        assert all(t.device.index == 1 for t in result)
        assert all(torch.count_nonzero(t).item() == 0 for t in result)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
