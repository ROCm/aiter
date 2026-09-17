# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

import aiter.test_common
from aiter.utility.mp_tuner import worker


def test_worker_forwards_requested_benchmark_iterations(monkeypatch):
    captured = {}

    def fake_run_perftest(func, *args, **kwargs):
        captured.update(kwargs)
        return object(), 12.5

    monkeypatch.setattr(aiter.test_common, "run_perftest", fake_run_perftest)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    info, latency, error = worker(
        0,
        "shape",
        lambda: None,
        [],
        {},
        num_iters=17,
        num_warmup=9,
    )

    assert (info, latency, error) == ("shape", 12.5, 0.0)
    assert captured["num_iters"] == 17
    assert captured["num_warmup"] == 9
