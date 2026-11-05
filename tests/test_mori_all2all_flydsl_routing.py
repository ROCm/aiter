# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which implementation MoriAll2AllManager.get_handle hands out (CPU only).

Supported intra-node high-throughput handles go to the FlyDSL kernels without
touching mori; every other case (inter-node, low latency, unsupported shape /
codec / arch, ``backend="mori"``) builds a mori handle after initialising
mori's shmem heap.
"""

import pytest
import torch

from aiter.dist.device_communicators import all2all
from aiter.dist.device_communicators.base_device_communicator import Cache


class _Handle:
    """Stand-in handle (the handle cache keeps weak references)."""

    def __init__(self, impl):
        self.impl = impl


BASE = {
    "rank": 0,
    "num_ep_ranks": 8,
    "input_dtype": torch.bfloat16,
    "quant_dtype": torch.float8_e4m3fn,
    "token_hidden_size": 5120,
    "scale_dim": 160,
    "scale_type_size": 1,
    "max_num_tokens_per_dp_rank": 1024,
    "num_local_experts": 48,
    "num_experts_per_token": 6,
    "gpu_per_node": 8,
    "low_latency": False,
}


@pytest.fixture
def manager(monkeypatch):
    calls = {"shmem": 0}
    monkeypatch.setattr(all2all, "_has_module", lambda name: True)
    monkeypatch.setattr("aiter.jit.utils.chip_info.get_gfx_runtime", lambda: "gfx950")

    def fake_shmem():
        calls["shmem"] += 1

    monkeypatch.setattr(all2all, "ensure_mori_shmem", fake_shmem)
    m = all2all.MoriAll2AllManager.__new__(all2all.MoriAll2AllManager)
    m.internode = False
    m.cpu_group = None
    m.handle_cache = Cache()
    m._flydsl_handles = []
    monkeypatch.setattr(
        m,
        "_make_flydsl_handle",
        lambda impl, **kw: _Handle("flydsl"),
        raising=False,
    )
    monkeypatch.setattr(m, "_make_all2all_kwargs", lambda **kw: kw)
    monkeypatch.setattr(m, "_make_handle", lambda **kw: _Handle("mori"))
    return m, calls


def _impl(m, **override):
    kwargs = {**BASE, **override}
    backend = kwargs.pop("backend", "auto")
    handle = m.get_handle(kwargs, backend=backend)
    return handle.impl


def test_supported_intranode_uses_flydsl(manager):
    m, calls = manager
    assert _impl(m) == "flydsl"
    assert _impl(m, quant_type="fp8_direct_cast") == "flydsl"
    assert _impl(m, quant_dtype=torch.bfloat16, scale_dim=0, scale_type_size=0) == (
        "flydsl"
    )
    assert calls["shmem"] == 0


@pytest.mark.parametrize(
    "override",
    [
        {"backend": "mori"},
        {"low_latency": True},
        {"quant_type": "fp8_blockwise"},
        {"num_ep_ranks": 16},
        {"token_hidden_size": 5124},
        {"input_dtype": torch.float16},
        {"scale_dim": 512, "scale_type_size": 4},
    ],
)
def test_other_handles_use_mori(manager, override):
    m, calls = manager
    assert _impl(m, **override) == "mori"
    assert calls["shmem"] == 1


def test_internode_uses_mori(manager):
    m, calls = manager
    m.internode = True
    assert _impl(m) == "mori"
    assert calls["shmem"] == 1


def test_unsupported_arch_uses_mori(manager, monkeypatch):
    m, _ = manager
    monkeypatch.setattr("aiter.jit.utils.chip_info.get_gfx_runtime", lambda: "gfx1250")
    assert _impl(m) == "mori"


def test_bad_backend_rejected(manager):
    m, _ = manager
    with pytest.raises(ValueError, match="backend"):
        _impl(m, backend="flydsl")
