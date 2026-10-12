# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""GPU launch semantics for AITER's cached FlyDSL dispatch helper."""

import sys

import pytest
import torch


@pytest.mark.parametrize("null_optional", [False, True])
@pytest.mark.parametrize("specialization_key", [None, "increment"])
def test_run_compiled_launches_once(null_optional, specialization_key, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    flyc = pytest.importorskip("flydsl.compiler")
    fx = pytest.importorskip("flydsl.expr")
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg

    @flyc.kernel(known_block_size=(64, 1, 1))
    def increment(output: fx.Pointer, optional: fx.Pointer):
        out = fx.recast_iter(fx.Int32, output)
        if fx.thread_idx.x == 0:
            out[0] = out[0] + 1

    @flyc.jit
    def launch(output: fx.Pointer, optional: fx.Pointer, stream: fx.Stream):
        increment(output, optional).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        counters = [torch.zeros(1, dtype=torch.int32, device="cuda") for _ in range(2)]
        optional = (
            flyc.from_c_void_p(fx.Int32, 0)
            if null_optional
            else ptr_arg(counters[0], fx.Int32)
        )

        def run(output):
            _run_compiled(
                launch,
                ptr_arg(output, fx.Int32),
                optional,
                stream,
                specialization_key=specialization_key,
            )

        with monkeypatch.context() as compile_only:
            compile_only.setenv("COMPILE_ONLY", "1")
            run(counters[0])
        stream.synchronize()
        assert counters[0].item() == 0

        # Check the initial dispatch and cached calls, including a new pointer.
        # Incrementing makes both a missing launch and a double launch observable.
        for index, expected in ((0, 1), (1, 1), (0, 2)):
            run(counters[index])
            stream.synchronize()
            assert counters[index].item() == expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", *sys.argv[1:]]))
