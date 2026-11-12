# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""BF16 latent-projection GEMMs: caller buffers and changed graph inputs.

The defaults exercise both projection groups at their optimized M cases:
M128/M16384 for N7168/K3584 and M16384 for N3584/K7168, plus padded-M boundaries.
The M128/N3584/K7168 case exercises deterministic FP32 partial reduction. Bias, scaling and preshuffled
weights are excluded. Torch FP32 linear is an untimed reference.
"""

import argparse
import itertools
import statistics

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.tuned_gemm import gemm_a16w16, gemm_a16w16_out

# The changed M128 key also serves M65..128 through padded-M lookup.
# Both large projections have exact-shape implementation specializations.
# Their ragged-M boundaries keep the corresponding original device code.
SHAPES = (
    (128, 3584, 7168),
    (128, 7168, 3584),
    (16384, 3584, 7168),
    (16384, 7168, 3584),
    (65, 7168, 3584),
    (127, 7168, 3584),
    (8193, 3584, 7168),
    (16383, 3584, 7168),
    (16257, 7168, 3584),
    (16383, 7168, 3584),
)


def run_torch(x, weight):
    return F.linear(x.float(), weight.float()).to(x.dtype)


def assert_result(expected, actual, label):
    assert expected.shape == actual.shape and expected.dtype == actual.dtype
    error = checkAllclose(
        expected.float(),
        actual.float(),
        rtol=1e-2,
        atol=1e-2,
        tol_err_ratio=0,
        msg=label,
    )
    assert error == 0, (label, error)
    return error


@benchmark()
def test_latent_gemm(m, n, k, seed):
    torch.manual_seed(seed)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    # Exact trailing capacities for inputs; nonzero output halos expose overwrites.
    storage = torch.full((m * n + 32,), 19.0, device=x.device, dtype=x.dtype)
    out = storage[16:-16].view(m, n)
    expected = run_torch(x, weight)

    def into():
        assert gemm_a16w16_out(x, weight, out) is None
        return out

    candidates = {"functional": lambda: gemm_a16w16(x, weight), "out": into}
    for name, fn in candidates.items():
        result = fn()
        assert_result(expected, result, name)

    graphs = []
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    others = [torch.empty_like(out) for _ in streams]
    for stream, other in zip(streams, others):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            gemm_a16w16_out(x, weight, other)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                gemm_a16w16_out(x, weight, other)
        stream.synchronize()
        graphs.append(graph)

    for pattern in ("small", "dyadic"):
        if pattern == "small":
            x.mul_(0.125)
            weight.mul_(0.125)
        else:
            x.copy_(torch.randint(-4, 5, x.shape, device=x.device).float() / 16)
            weight.copy_(
                torch.randint(-4, 5, weight.shape, device=x.device).float() / 16
            )
        before = (x.clone(), weight.clone())
        expected = run_torch(x, weight)
        for stream, graph in zip(streams, graphs):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
                graph.replay()
        for stream, other in zip(streams, others):
            stream.synchronize()
            assert_result(expected, other, pattern)
        assert torch.equal(x, before[0]) and torch.equal(weight, before[1])
        into()
        assert_result(expected, out, "out " + pattern)
        assert (storage[:16] == 19).all().item()
        assert (storage[-16:] == 19).all().item()

    # Restore a random distribution before timing. Include allocation for the
    # functional call; the out= variant reuses the supplied output.
    x.normal_()
    weight.normal_()
    expected = run_torch(x, weight)
    # Amortize Python launch gaps when comparing GPU kernel configurations.
    graph = torch.cuda.CUDAGraph()
    into()
    with torch.cuda.graph(graph):
        for _ in range(4):
            into()

    def replay():
        graph.replay()
        return out

    candidates["out_graph"] = replay
    ret = {"gfx": get_gfx_runtime()}
    flops = 2 * m * n * k
    nbytes = 2 * (m * k + n * k + m * n)
    for name, fn in candidates.items():
        samples = []
        for _ in range(3):
            result, us = run_perftest(
                fn,
                num_iters=30,
                num_warmup=5,
                num_rotate_args=1,
                use_cuda_event=True,
            )
            error = assert_result(expected, result, name + " measured")
            samples.append(us)
        us = statistics.median(samples) / (4 if name == "out_graph" else 1)
        ret.update(
            {
                name + " us": us,
                name + " TFLOPS": flops / us / 1e6,
                name + " TB/s": nbytes / us / 1e6,
                name + " err": error,
            }
        )
    return ret


def check_ring_specialization():
    from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import (
        _use_unrolled_lds_ring,
        make_gemm_a16w16_gfx950_kernel_name,
        make_gemm_a16w16_param_and_validate,
    )

    cfg = {
        "in_dtype_id": 2,
        "out_dtype_id": 2,
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "stages": 5,
        "split_k": 1,
        "m_waves": 2,
        "n_waves": 2,
        "k_waves": 1,
        "group_m": 0,
        "use_half_tile_interleaved": False,
        "a_is_transposed": False,
        "b_is_transposed": True,
        "has_bias": False,
    }

    original = cfg.copy()
    selected = []
    for m in range(1, 257):
        want = 65 <= m <= 128
        assert _use_unrolled_lds_ring(m, 7168, 3584, cfg) == want
        param = make_gemm_a16w16_param_and_validate(m, 7168, 3584, cfg)
        assert param is not None
        name = make_gemm_a16w16_gfx950_kernel_name(param)
        assert name.endswith("_ring5") == want
        selected.append(param)
    # One specialization per policy, not one binary per exact padded M.
    assert selected[64] is selected[127]
    assert selected[63] is selected[128]
    assert selected[63] is not selected[64]
    for n, k in ((7168, 512), (7168, 3520), (7160, 3584)):
        assert not _use_unrolled_lds_ring(128, n, k, cfg)
    for field, value in (
        ("in_dtype_id", 3),
        ("out_dtype_id", 1),
        ("has_bias", True),
        ("a_is_transposed", True),
        ("b_is_transposed", False),
        ("use_half_tile_interleaved", True),
        ("b_non_temporal", True),
        ("k_stagger", 1),
        ("group_m", 4),
        ("split_k", 2),
        ("block_m", 128),
        ("block_n", 128),
        ("block_k", 128),
        ("stages", 7),
        ("m_waves", 4),
        ("n_waves", 4),
        ("k_waves", 2),
    ):
        assert not _use_unrolled_lds_ring(128, 7168, 3584, {**cfg, field: value})
    # A caller-provided override must not broaden the internal specialization.
    param = make_gemm_a16w16_param_and_validate(
        128, 7168, 512, {**cfg, "unroll_lds_ring": True}
    )
    assert not make_gemm_a16w16_gfx950_kernel_name(param).endswith("_ring5")
    assert cfg == original
    aiter.logger.info("LDS-ring shape/config/cache specialization checks passed")


def check_n_grouped_specialization():
    from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import (
        _use_n_grouped_loads,
        make_gemm_a16w16_gfx950_kernel_name,
        make_gemm_a16w16_param_and_validate,
    )

    cfg = {
        "in_dtype_id": 2,
        "out_dtype_id": 2,
        "block_m": 256,
        "block_n": 256,
        "block_k": 64,
        "stages": 2,
        "split_k": 1,
        "m_waves": 2,
        "n_waves": 4,
        "k_waves": 1,
        "group_m": 4,
        "use_half_tile_interleaved": True,
        "a_is_transposed": False,
        "b_is_transposed": True,
        "has_bias": False,
    }
    original = cfg.copy()
    for m, n, k in (
        (16384, 7168, 3584),
        (16383, 7168, 3584),
        (16257, 7168, 3584),
        (16385, 7168, 3584),
        (16384, 3584, 7168),
        (16384, 7168, 3456),
        (16384, 7152, 3584),
        (128, 7168, 3584),
    ):
        want = (m, n, k) == (16384, 7168, 3584)
        assert _use_n_grouped_loads(m, n, k, cfg) == want
        param = make_gemm_a16w16_param_and_validate(m, n, k, cfg)
        assert param is not None
        assert make_gemm_a16w16_gfx950_kernel_name(param).endswith("_ng4sk") == want
    for field, value in (
        ("in_dtype_id", 3),
        ("out_dtype_id", 1),
        ("has_bias", True),
        ("a_is_transposed", True),
        ("b_is_transposed", False),
        ("use_half_tile_interleaved", False),
        ("b_non_temporal", True),
        ("k_stagger", 1),
        ("group_m", 0),
        ("split_k", 2),
        ("block_m", 128),
        ("block_n", 128),
        ("block_k", 128),
        ("stages", 3),
        ("m_waves", 4),
        ("n_waves", 2),
        ("k_waves", 2),
    ):
        assert not _use_n_grouped_loads(16384, 7168, 3584, {**cfg, field: value})
    selected = make_gemm_a16w16_param_and_validate(16384, 7168, 3584, cfg)
    assert selected is make_gemm_a16w16_param_and_validate(16384, 7168, 3584, cfg)
    declined = make_gemm_a16w16_param_and_validate(
        16383, 7168, 3584, {**cfg, "n_grouped_loads": True}
    )
    assert selected is not declined
    assert not make_gemm_a16w16_gfx950_kernel_name(declined).endswith("_ng4sk")
    assert cfg == original
    aiter.logger.info("N-grouped shape/config/cache specialization checks passed")


def check_hti_unroll_specialization():
    from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import (
        _use_unrolled_hti,
        make_gemm_a16w16_gfx950_kernel_name,
        make_gemm_a16w16_param_and_validate,
    )

    cfg = {
        "in_dtype_id": 2,
        "out_dtype_id": 2,
        "block_m": 256,
        "block_n": 256,
        "block_k": 64,
        "stages": 2,
        "split_k": 1,
        "m_waves": 2,
        "n_waves": 4,
        "k_waves": 1,
        "group_m": 4,
        "use_half_tile_interleaved": True,
        "a_is_transposed": False,
        "b_is_transposed": True,
        "has_bias": False,
    }
    original = cfg.copy()
    for m, n, k in (
        (16384, 3584, 7168),
        (16383, 3584, 7168),
        (16257, 3584, 7168),
        (16385, 3584, 7168),
        (16384, 7168, 3584),
        (16384, 7168, 3456),
        (16384, 7152, 3584),
        (128, 3584, 7168),
    ):
        want = (m, n, k) == (16384, 3584, 7168)
        assert _use_unrolled_hti(m, n, k, cfg) == want
        param = make_gemm_a16w16_param_and_validate(m, n, k, cfg)
        assert param is not None
        assert make_gemm_a16w16_gfx950_kernel_name(param).endswith("_hti5sk") == want
    for field, value in (
        ("in_dtype_id", 3),
        ("out_dtype_id", 1),
        ("has_bias", True),
        ("a_is_transposed", True),
        ("b_is_transposed", False),
        ("use_half_tile_interleaved", False),
        ("b_non_temporal", True),
        ("k_stagger", 1),
        ("group_m", 0),
        ("split_k", 2),
        ("block_m", 128),
        ("block_n", 128),
        ("block_k", 128),
        ("stages", 3),
        ("m_waves", 4),
        ("n_waves", 2),
        ("k_waves", 2),
    ):
        assert not _use_unrolled_hti(16384, 3584, 7168, {**cfg, field: value})
    selected = make_gemm_a16w16_param_and_validate(16384, 3584, 7168, cfg)
    assert selected is make_gemm_a16w16_param_and_validate(16384, 3584, 7168, cfg)
    declined = make_gemm_a16w16_param_and_validate(
        16383, 3584, 7168, {**cfg, "unroll_hti": True}
    )
    assert selected is not declined
    assert not make_gemm_a16w16_gfx950_kernel_name(declined).endswith("_hti5sk")
    assert cfg == original
    aiter.logger.info("HTI period shape/config/cache specialization checks passed")


def check_packed_split_specialization():
    from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import (
        _use_packed_split,
        make_gemm_a16w16_gfx950_kernel_name,
        make_gemm_a16w16_param_and_validate,
    )

    cfg = {
        "in_dtype_id": 2,
        "out_dtype_id": 2,
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "stages": 4,
        "split_k": 4,
        "m_waves": 2,
        "n_waves": 2,
        "k_waves": 1,
        "group_m": 0,
        "use_half_tile_interleaved": False,
        "a_is_transposed": False,
        "b_is_transposed": True,
        "has_bias": False,
    }
    original = cfg.copy()
    for m in (64, 65, 127, 128, 129, 256):
        want = m == 128
        assert _use_packed_split(m, 3584, 7168, cfg) == want
        param = make_gemm_a16w16_param_and_validate(m, 3584, 7168, cfg)
        assert param is not None
        assert make_gemm_a16w16_gfx950_kernel_name(param).endswith("_ps2r4_ptr") == want
    old = make_gemm_a16w16_param_and_validate(128, 3584, 7168, cfg)
    new = make_gemm_a16w16_param_and_validate(
        128, 3584, 7168, {**cfg, "stages": 5, "split_k": 2}
    )
    assert old is new
    assert int(old.stages) == 5 and not bool(old.is_split_k)
    for field, value in (
        ("in_dtype_id", 3),
        ("out_dtype_id", 1),
        ("has_bias", True),
        ("a_is_transposed", True),
        ("b_is_transposed", False),
        ("use_half_tile_interleaved", True),
        ("b_non_temporal", True),
        ("k_stagger", 1),
        ("group_m", 4),
        ("split_k", 1),
        ("block_m", 128),
        ("block_n", 128),
        ("block_k", 128),
        ("stages", 7),
        ("m_waves", 4),
        ("n_waves", 4),
        ("k_waves", 2),
    ):
        assert not _use_packed_split(128, 3584, 7168, {**cfg, field: value})
    for n, k in ((3584, 3584), (7168, 7168), (3576, 7168)):
        assert not _use_packed_split(128, n, k, cfg)
    declined = make_gemm_a16w16_param_and_validate(
        127, 3584, 7168, {**cfg, "packed_split": True}
    )
    assert not make_gemm_a16w16_gfx950_kernel_name(declined).endswith("_ps2r4_ptr")
    assert cfg == original
    aiter.logger.info("Packed split shape/config/cache specialization checks passed")


@benchmark()
def test_packed_split_lifetime(padding, seed):
    """Independent scratch under graph capture, explicit streams and legal strides."""
    from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import gemm_a16w16 as fly_gemm

    m, n, k = 128, 3584, 7168
    torch.manual_seed(seed)
    cfg = {
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "stages": 4,
        "split_k": 4,
        "m_waves": 2,
        "n_waves": 2,
        "k_waves": 1,
        "group_m": 0,
        "use_half_tile_interleaved": False,
    }
    inputs, halos, outputs = [], [], []
    for _ in range(2):
        x = torch.randn((m, k + padding), device="cuda", dtype=torch.bfloat16)
        w = torch.randn((n, k + 2 * padding), device="cuda", dtype=torch.bfloat16)
        # Nonzero, 16-byte-aligned offsets, exact logical K and padded strides.
        x = x[:, padding // 2 : padding // 2 + k]
        w = w[:, padding : padding + k]
        inputs.append((x, w))
        halo = torch.full((m * n + 32,), 37.0, device=x.device, dtype=x.dtype)
        halos.append(halo)
        outputs.append(halo[16:-16].view(m, n))

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs = []
    # Compile on the ambient stream, then capture on two new streams without
    # preallocating scratch there. Each captured invocation owns its workspace.
    for (x, w), out in zip(inputs, outputs):
        fly_gemm(x, w.T, out=out, user_kwargs=cfg)
    for stream, (x, w), out in zip(streams, inputs, outputs):
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(4):
                fly_gemm(x, w.T, out=out, user_kwargs=cfg)
        stream.synchronize()
        graphs.append(graph)

    for step in range(6):
        for x, w in inputs:
            if step % 3 == 0:
                x.normal_()
                w.normal_()
            elif step % 3 == 1:
                x.mul_(0.125)
                w.mul_(0.125)
            else:
                x.copy_(torch.randint(-4, 5, x.shape, device=x.device).float() / 16)
                w.copy_(torch.randint(-4, 5, w.shape, device=w.device).float() / 16)
        saved = [(x.clone(), w.clone()) for x, w in inputs]
        expected = [run_torch(x, w) for x, w in inputs]
        for stream, graph, out in zip(streams, graphs, outputs):
            out.fill_(float("nan"))
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
                graph.replay()
        # Pass an explicit launch stream while the ambient stream is different.
        for stream, (x, w), out in zip(streams, inputs, outputs):
            fly_gemm(x, w.T, out=out, user_kwargs=cfg, stream=stream)
        pressure = [
            torch.full((256, 3584), 91.0, device=outputs[0].device) for _ in range(3)
        ]
        for stream in streams:
            stream.synchronize()
        for (x, w), (sx, sw), ref, out, halo in zip(
            inputs, saved, expected, outputs, halos
        ):
            assert_result(ref, out, "packed scratch lifetime")
            assert torch.equal(x, sx) and torch.equal(w, sw)
            assert (halo[:16] == 37).all().item()
            assert (halo[-16:] == 37).all().item()
        del pressure

    ret = {"gfx": get_gfx_runtime()}
    candidates = {f"graph{i}": graph.replay for i, graph in enumerate(graphs)}
    for i, (name, fn) in enumerate(candidates.items()):
        _, us = run_perftest(
            fn, num_iters=30, num_warmup=5, num_rotate_args=1, use_cuda_event=True
        )
        err = assert_result(expected[i], outputs[i], name)
        us /= 4
        ret.update(
            {
                name + " us": us,
                name + " TFLOPS": 2 * m * n * k / us / 1e6,
                name + " TB/s": 2 * (m * k + n * k + m * n) / us / 1e6,
                name + " err": err,
            }
        )
    return ret


def check_packed_pointer_inputs():
    """Reject invalid pointer launches before invoking any compiled handle."""
    from unittest.mock import Mock, patch

    import aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 as impl

    cfg = {
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "stages": 4,
        "split_k": 4,
        "m_waves": 2,
        "n_waves": 2,
        "k_waves": 1,
        "group_m": 0,
        "use_half_tile_interleaved": False,
    }
    stream = torch.cuda.current_stream()
    x = torch.empty((128, 7168), device="cuda", dtype=torch.bfloat16)
    w = torch.empty((3584, 7168), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((128, 3584), device="cuda", dtype=torch.bfloat16)
    cpu_x = torch.empty_like(x, device="cpu")
    cpu_w = torch.empty_like(w, device="cpu")
    cpu_out = torch.empty_like(out, device="cpu")
    unaligned = torch.empty((128 * 7168 + 1,), device="cuda", dtype=x.dtype)[
        1:
    ].view_as(x)
    invalid_unit_stride = torch.empty((128, 2 * 7168), device="cuda", dtype=x.dtype)[
        :, ::2
    ]

    # These metadata-only mocks have real device/dtype/address metadata but
    # illegal strides; no giant GPU allocation or out-of-bounds launch is made.
    huge_x = Mock(spec=torch.Tensor)
    huge_x.device, huge_x.dtype, huge_x.ndim = x.device, x.dtype, 2
    huge_x.shape = x.shape
    huge_x.data_ptr.return_value = x.data_ptr()
    huge_x.element_size.return_value = x.element_size()
    huge_x.stride.side_effect = lambda axis: (1 << 24) if axis == 0 else 1

    wrong_stream = Mock(spec=torch.cuda.Stream)
    wrong_stream.device = torch.device("cuda", x.device.index + 1)
    bad_calls = (
        lambda: impl.gemm_a16w16(
            cpu_x, cpu_w.T, out=cpu_out, user_kwargs=cfg, stream=stream
        ),
        lambda: impl.gemm_a16w16(x, cpu_w.T, out=out, user_kwargs=cfg, stream=stream),
        lambda: impl.gemm_a16w16(x, w.T, out=cpu_out, user_kwargs=cfg, stream=stream),
        lambda: impl.gemm_a16w16(x, w.T, out=out, user_kwargs=cfg, stream=wrong_stream),
        lambda: impl.gemm_a16w16(
            unaligned, w.T, out=out, user_kwargs=cfg, stream=stream
        ),
        lambda: impl.gemm_a16w16(
            invalid_unit_stride, w.T, out=out, user_kwargs=cfg, stream=stream
        ),
        lambda: impl.gemm_a16w16(huge_x, w.T, out=out, user_kwargs=cfg, stream=stream),
    )
    with patch.object(impl, "_run_compiled") as launch:
        for fn in bad_calls:
            try:
                fn()
            except (ValueError, AssertionError):
                pass
            else:
                raise AssertionError("invalid packed pointer input was accepted")
        launch.assert_not_called()
    aiter.logger.info("Packed pointer device/layout/span rejection checks passed")


def main():
    if get_gfx_runtime() != "gfx950":
        aiter.logger.warning("This tuning qualification targets gfx950; skipping")
        return
    check_ring_specialization()
    check_n_grouped_specialization()
    check_hti_unroll_specialization()
    check_packed_split_specialization()
    check_packed_pointer_inputs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-s", "--mnk", nargs="+", type=aiter.dtypes.str2tuple, default=SHAPES
    )
    parser.add_argument("--seed", nargs="+", type=int, default=[912, 913])
    parser.add_argument("--leading-padding", type=int, nargs="+", default=[0, 16])
    args = parser.parse_args()
    if any(p < 0 or p % 16 for p in args.leading_padding):
        parser.error("--leading-padding must contain nonnegative multiples of 16")
    rows = [
        test_latent_gemm(*shape, seed)
        for shape, seed in itertools.product(args.mnk, args.seed)
    ]
    aiter.logger.info(
        "BF16 latent projection GEMMs:\n%s", pd.DataFrame(rows).to_markdown(index=False)
    )
    lifetime_rows = [
        test_packed_split_lifetime(padding, seed)
        for padding, seed in itertools.product(args.leading_padding, args.seed)
    ]
    aiter.logger.info(
        "Packed split scratch lifetime:\n%s",
        pd.DataFrame(lifetime_rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
