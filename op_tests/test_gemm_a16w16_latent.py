# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""BF16 latent-projection GEMMs: caller buffers and changed graph inputs.

The two exact M16384 projections and down-M128 have specialized schedules.
Up-M128 selects the counter WS kernel through its model CSV for contiguous
aligned inputs; non-target padded-M consumers retain the original kernel.
Down-M128 retains the original four-way BF16 atomic arithmetic, but publishes
output initialization before input prefetch and unrolls the LDS ring. Its
exact-shape pointer launcher reuses the existing split-K synchronization.
This does not repair the original strict random-reference precision limitation;
the separate exact-input check below verifies dispatch/ownership without
relaxing the strict tests for the other paths.
Bias, scaling and preshuffled weights are excluded. Torch FP32 linear is an
untimed reference.
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

# Up-M128 WS and the new large-M row are exact-only CSV selections.
# Both large projections have exact-shape implementation specializations.
# Their ragged-M boundaries keep the corresponding original device code.
SHAPES = (
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


def check_up_m128_csv_dispatch():
    """Prove the real CSV selects WS, with legacy padded/layout fallbacks."""
    from unittest.mock import patch

    import aiter.ops.flydsl.gemm_kernels as kernels
    import aiter.tuned_gemm as tuned

    torch.manual_seed(2911)
    dtype = torch.bfloat16
    cfg = tuned.get_GEMM_A16W16_config(
        128,
        7168,
        3584,
        False,
        str(dtype),
        str(dtype),
        gfx="gfx950",
        cu_num=256,
    )
    assert cfg["kernelName"] == kernels.WS_UP_M128_KERNEL_NAME
    real_ws = tuned._flydsl_ws_launcher(cfg["kernelName"])
    a = torch.randn((128, 3584), device="cuda", dtype=dtype)
    b = torch.randn((7168, 3584), device=a.device, dtype=dtype)
    out = torch.empty((128, 7168), device=a.device, dtype=dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # Cold capture exercises CSV selection, JIT and current-stream launch.
    graph = torch.cuda.CUDAGraph()
    with (
        patch.object(tuned, "_flydsl_ws_launcher", return_value=real_ws) as ws,
        patch.object(kernels, "flydsl_hgemm", wraps=kernels.flydsl_hgemm) as old,
    ):
        with torch.cuda.graph(graph, stream=stream):
            gemm_a16w16_out(a, b, out)
        stream.synchronize()
        assert ws.call_count >= 1
        old.assert_not_called()
    for _ in range(3):
        a.normal_()
        b.normal_()
        expected = run_torch(a, b)
        out.fill_(float("nan"))
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        assert_result(expected, out, "CSV WS modified-input graph")

    for case in ("padded_m", "row_padding", "k_stride"):
        if case == "padded_m":
            x, w = a[:127], b
        elif case == "row_padding":
            x = torch.randn((128, 3600), device=a.device, dtype=dtype)[:, :3584]
            w = b
        else:
            x = torch.randn((128, 7168), device=a.device, dtype=dtype)[:, ::2]
            w = b
        expected = run_torch(x, w)
        target = torch.empty_like(expected)
        with (
            patch.object(tuned, "_flydsl_ws_launcher") as ws,
            patch.object(kernels, "flydsl_hgemm", wraps=kernels.flydsl_hgemm) as old,
        ):
            gemm_a16w16_out(x, w, target)
            torch.cuda.synchronize()
            ws.assert_not_called()
            assert old.call_count == 1
            assert old.call_args.kwargs["block_k"] == 64
            assert old.call_args.kwargs["stages"] == 7
        assert_result(expected, target, "CSV legacy " + case)
    aiter.logger.info("CSV WS exact dispatch and legacy fallbacks: PASS")


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


@benchmark()
def test_down_projection_split(seed, padding):
    """Same atomic arithmetic: exact-input, ownership and pointer-dispatch guard.

    This does NOT certify strict random-input accuracy. The old four-way BF16
    atomic reduction has a documented random-reference error; the slower FP32
    repair was withdrawn, rather than widening test_latent_gemm's tolerance.
    """
    from unittest.mock import patch

    import aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 as impl

    m, n, k = 128, 3584, 7168
    torch.manual_seed(seed)
    x_mem = torch.zeros((m, k + 2 * padding), device="cuda", dtype=torch.bfloat16)
    w_mem = torch.zeros((n, k + 2 * padding), device=x_mem.device, dtype=x_mem.dtype)
    x = x_mem[:, padding : padding + k]
    weight = w_mem[:, padding : padding + k]
    weight.copy_(
        (torch.randint(0, 2, (n, k), device=x.device) * 2 - 1).to(x.dtype) / 16
    )
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

    def direct(out, stream):
        return impl.gemm_a16w16(x, weight.T, out=out, user_kwargs=cfg, stream=stream)

    # Eight nonzero terms per partition. Every product is +/-1/256; all
    # partial/intermediate sums are exact BF16 for any atomic arrival order.
    active = torch.tensor(
        [part * 1792 + offset for part in range(4) for offset in range(8)],
        device=x.device,
    )
    x[:, active] = (torch.randint(0, 2, (m, 32), device=x.device) * 2 - 1).to(
        x.dtype
    ) / 16
    halos = [
        torch.full((m * n + 64,), 23.0, device=x.device, dtype=x.dtype)
        for _ in range(2)
    ]
    outs = [h[32:-32].view(m, n) for h in halos]
    streams = [torch.cuda.Stream() for _ in outs]
    graphs = []
    get_buffers = impl.get_split_k_buffers

    def checked_buffers(stream, device, split_k):
        # Cold capture followed by first eager use can allocate new counters.
        # Their initialization must be queued on the actual launch stream.
        assert torch.cuda.current_stream(device) == stream
        return get_buffers(stream, device, split_k)

    with (
        patch.object(
            impl, "get_split_k_buffers", side_effect=checked_buffers
        ) as scratch,
        patch.object(impl, "_run_compiled", wraps=impl._run_compiled) as pointer,
    ):
        for stream, out in zip(streams, outs):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    direct(out, stream)
                    direct(out, stream)
            stream.synchronize()
            assert scratch.call_args.args[2] == 4
            assert pointer.call_args.args[0] is impl._split_init_ring_ptr
            graphs.append(graph)
        for _ in range(8):
            x.neg_()
            saved = (x.clone(), weight.clone())
            expected = run_torch(x, weight)
            for stream, graph, out in zip(streams, graphs, outs):
                out.fill_(float("nan"))
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    graph.replay()
                    graph.replay()
            # Eager use of the same explicit stream outside its context.
            for stream, out in zip(streams, outs):
                direct(out, stream)
            pressure = [
                torch.empty((128, 3584), device=x.device, dtype=x.dtype)
                for _ in range(4)
            ]
            for stream, out, halo in zip(streams, outs, halos):
                stream.synchronize()
                assert_result(expected, out, "split down exact input")
                assert torch.equal(expected, out)
                assert (halo[:32] == 23).all() and (halo[-32:] == 23).all()
            assert torch.equal(x, saved[0]) and torch.equal(weight, saved[1])
            del pressure

    result = gemm_a16w16(x, weight)
    assert_result(expected, result, "functional down exact")
    ret = {"m": m, "n": n, "k": k, "gfx": get_gfx_runtime()}
    candidates = {f"graph{i}": graph.replay for i, graph in enumerate(graphs)}
    for i, (name, fn) in enumerate(candidates.items()):
        _, us = run_perftest(
            fn, num_iters=30, num_warmup=5, num_rotate_args=1, use_cuda_event=True
        )
        us /= 2
        error = assert_result(expected, outs[i], "split " + name)
        ret.update(
            {
                name + " us": us,
                name + " TFLOPS": 2 * m * n * k / us / 1e6,
                name + " TB/s": 2 * (m * k + n * k + m * n) / us / 1e6,
                name + " err": error,
            }
        )
    return ret


def check_split_pointer_routing():
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
    x = torch.empty((129, 7168), device="cuda", dtype=torch.bfloat16)
    w = torch.empty((3584, 7168), device=x.device, dtype=x.dtype)
    o = torch.empty((129, 3584), device=x.device, dtype=x.dtype)
    stream = torch.cuda.current_stream()
    sem = torch.empty((2048,), device=x.device, dtype=torch.int64)
    sig = torch.empty((2048,), device=x.device, dtype=torch.int32)
    with (
        patch.object(impl, "_run_compiled") as pointer,
        patch.object(impl, "run_cached") as tensor,
        patch.object(impl, "get_split_k_buffers", return_value=(sem, sig)),
        patch.object(impl, "_dynamic_tensor_arg", return_value=None),
    ):
        for m in (65, 127, 128, 129):
            pointer.reset_mock()
            tensor.reset_mock()
            impl.gemm_a16w16(x[:m], w.T, out=o[:m], user_kwargs=cfg, stream=stream)
            if m == 128:
                pointer.assert_called_once()
                tensor.assert_not_called()
                assert pointer.call_args.args[0] is impl._split_init_ring_ptr
                key = pointer.call_args.kwargs["specialization_key"]
                assert key[:3] == (x.device.index, x.dtype, o.dtype)
            else:
                pointer.assert_not_called()
                tensor.assert_called_once()
        huge = Mock(spec=torch.Tensor)
        huge.device, huge.dtype, huge.ndim, huge.shape = (
            x.device,
            x.dtype,
            2,
            (128, 7168),
        )
        huge.data_ptr.return_value = x.data_ptr()
        huge.element_size.return_value = 2
        huge.stride.side_effect = lambda axis: 1 << 24 if axis == 0 else 1
        pointer.reset_mock()
        tensor.reset_mock()
        impl.gemm_a16w16(huge, w.T, out=o[:128], user_kwargs=cfg, stream=stream)
        pointer.assert_not_called()
        tensor.assert_called_once()
        wrong = Mock(spec=torch.cuda.Stream)
        wrong.device = torch.device("cuda", x.device.index + 1)
        cpu_x = x[:128].cpu()
        cpu_w = w.cpu()
        cpu_o = o[:128].cpu()
        misaligned = torch.empty((128 * 7168 + 1,), device=x.device, dtype=x.dtype)[
            1:
        ].view(128, 7168)
        for fn in (
            lambda: impl.gemm_a16w16(
                cpu_x, cpu_w.T, out=cpu_o, user_kwargs=cfg, stream=stream
            ),
            lambda: impl.gemm_a16w16(
                x[:128], w.T, out=o[:128], user_kwargs=cfg, stream=wrong
            ),
            lambda: impl.gemm_a16w16(
                misaligned, w.T, out=o[:128], user_kwargs=cfg, stream=stream
            ),
            lambda: impl.gemm_a16w16(
                x[:128], cpu_w.T, out=o[:128], user_kwargs=cfg, stream=stream
            ),
        ):
            pointer.reset_mock()
            tensor.reset_mock()
            try:
                fn()
            except (AssertionError, ValueError):
                pass
            else:
                raise AssertionError("invalid split pointer input accepted")
            pointer.assert_not_called()
            tensor.assert_not_called()


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


def main():
    if get_gfx_runtime() != "gfx950":
        aiter.logger.warning("This tuning qualification targets gfx950; skipping")
        return
    check_up_m128_csv_dispatch()
    check_n_grouped_specialization()
    check_hti_unroll_specialization()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-s", "--mnk", nargs="+", type=aiter.dtypes.str2tuple, default=SHAPES
    )
    parser.add_argument("--seed", nargs="+", type=int, default=[912, 913])
    args = parser.parse_args()
    rows = [
        test_latent_gemm(*shape, seed)
        for shape, seed in itertools.product(args.mnk, args.seed)
    ]
    aiter.logger.info(
        "BF16 latent projection GEMMs:\n%s", pd.DataFrame(rows).to_markdown(index=False)
    )
    check_split_pointer_routing()
    restored_rows = [
        test_down_projection_split(seed, padding)
        for seed, padding in itertools.product(args.seed, (0, 16))
    ]
    aiter.logger.info(
        "Down-M128 split schedule (exact-input test, not random-accuracy repair):"
        "\n%s",
        pd.DataFrame(restored_rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
