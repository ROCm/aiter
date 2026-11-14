# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which implementation MoriAll2AllManager.get_handle hands out (CPU only).

Supported intra-node high-throughput handles go to the FlyDSL kernels without
touching mori; every other case (inter-node, low latency, a config the FlyDSL
op rejects, an unsupported arch, ``backend="mori"``) builds a mori handle after
initialising mori's shmem heap. The FlyDSL decision is the op's own static
config check, the op accepts mori's ``dispatch(x, w, None, ids)`` on a handle
configured with scales, and the manager keeps its FlyDSL ops until
``destroy()``.
"""

import gc
import itertools

import pytest
import torch

pytest.importorskip("flydsl")

from aiter.dist.device_communicators import all2all
from aiter.dist.device_communicators.base_device_communicator import (
    Cache,
)
from aiter.ops.flydsl.kernels import (
    flydsl_dispatch_combine_intranode_op as op_mod,
)


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

# ATOM's handle for a model whose routed rows travel in bf16 while its experts
# take FP8 block-scale activations (e.g. DeepSeek-R1 FP8 on EP8): scales are
# configured (hidden // 128 fp32 values) but dispatch() is called with None.
BF16_WIRE = {
    **BASE,
    "quant_dtype": torch.bfloat16,
    "token_hidden_size": 7168,
    "scale_dim": 7168 // 128,
    "scale_type_size": 4,
    "max_num_tokens_per_dp_rank": 256,
    "num_local_experts": 32,
    "num_experts_per_token": 8,
}


def _bare_manager(monkeypatch, internode=False):
    monkeypatch.setattr(all2all, "_has_module", lambda name: True)
    monkeypatch.setattr("aiter.jit.utils.chip_info.get_gfx_runtime", lambda: "gfx950")
    m = all2all.MoriAll2AllManager.__new__(all2all.MoriAll2AllManager)
    m.internode = internode
    m.cpu_group = None
    m.handle_cache = Cache()
    m._flydsl_handles = []
    return m


@pytest.fixture
def manager(monkeypatch):
    calls = {"shmem": 0}

    def fake_shmem():
        calls["shmem"] += 1

    monkeypatch.setattr(all2all, "ensure_mori_shmem", fake_shmem)
    m = _bare_manager(monkeypatch)
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
        # the op's check rejects an fp8 combine cast of fp32 experts' output
        {"input_dtype": torch.float32, "quant_type": "fp8_direct_cast"},
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


def test_gate_matches_op_check(manager):
    """get_handle picks FlyDSL exactly when the op's static check accepts the
    config, and never raises for a config the op rejects."""
    m, _ = manager
    dtypes = (
        torch.bfloat16,
        torch.float32,
        torch.float16,
        torch.float8_e4m3fn,
        torch.float4_e2m1fn_x2,
    )
    grid = itertools.product(
        (2, 8, 9),  # ranks
        dtypes[:3],  # experts' output (combine) dtype
        dtypes,  # wire (dispatch) dtype
        (7168, 5124),  # hidden
        ((0, 0), (56, 4), (512, 4)),  # scale_dim, scale_type_size
        ("none", "fp8_direct_cast", "fp8_blockwise"),
        (8, 65),  # top-k
    )
    n_fly = n_mori = 0
    for world, in_dt, wire, hidden, (sd, sts), qt, k in grid:
        kwargs = {
            **BF16_WIRE,
            "num_ep_ranks": world,
            "gpu_per_node": world,
            "input_dtype": in_dt,
            "quant_dtype": wire,
            "token_hidden_size": hidden,
            "scale_dim": sd,
            "scale_type_size": sts,
            "num_experts_per_token": k,
            "quant_type": qt,
        }
        got = m.get_handle(kwargs).impl
        cfg_kwargs = all2all._flydsl_op_kwargs(
            **{
                key: v
                for key, v in kwargs.items()
                if key not in ("gpu_per_node", "low_latency")
            }
        )
        try:
            op_mod.FlyDSLDispatchCombineIntraNodeOp.check_config(
                op_mod.FlyDSLDispatchCombineConfig(**cfg_kwargs), arch="gfx950"
            )
            want = "flydsl"
        except (ValueError, NotImplementedError, RuntimeError):
            want = "mori"
        assert got == want, kwargs
        n_fly += got == "flydsl"
        n_mori += got == "mori"
    assert n_fly and n_mori


def test_bf16_wire_with_configured_scales(manager):
    """ATOM's bf16-wire handle goes to FlyDSL, and the op accepts its
    dispatch(a1, w, None, ids) call (mori does)."""
    m, _ = manager
    assert m.get_handle(dict(BF16_WIRE)).impl == "flydsl"

    cfg = op_mod.FlyDSLDispatchCombineConfig(
        **all2all._flydsl_op_kwargs(
            **{
                k: v
                for k, v in BF16_WIRE.items()
                if k not in ("gpu_per_node", "low_latency")
            }
        )
    )
    op = op_mod.FlyDSLDispatchCombineIntraNodeOp.__new__(
        op_mod.FlyDSLDispatchCombineIntraNodeOp
    )
    op.cfg = cfg
    op._check_tensor_device = lambda name, t: None  # CPU tensors stand in
    T, K = 5, cfg.num_experts_per_token
    a1 = torch.zeros(T, cfg.hidden_dim, dtype=torch.bfloat16)
    w = torch.zeros(T, K, dtype=torch.float32)
    ids = torch.zeros(T, K, dtype=torch.int32)
    op._check_dispatch_inputs(a1, w, None, ids, None)  # must not raise
    scales = torch.zeros(T, cfg.scale_dim, dtype=torch.float32)
    op._check_dispatch_inputs(a1, w, scales, ids, None)

    # a scales tensor on a handle without scales is still a caller error
    op.cfg = op_mod.FlyDSLDispatchCombineConfig(
        **{**cfg.__dict__, "scale_dim": 0, "scale_type_size": 0}
    )
    with pytest.raises(ValueError, match="scales"):
        op._check_dispatch_inputs(a1, w, scales, ids, None)


class _FakeHeap:
    closed = 0

    def close(self):
        _FakeHeap.closed += 1


class _FakeOp:
    built = 0
    check_config = staticmethod(op_mod.FlyDSLDispatchCombineIntraNodeOp.check_config)

    def __init__(self, cfg, group=None):
        _FakeOp.built += 1
        self.cfg = cfg
        self._heap = _FakeHeap()


def test_flydsl_ops_live_until_destroy(monkeypatch):
    """A handle the caller dropped is returned again (no second collective
    allocation); destroy() closes every op's IPC heap once."""
    m = _bare_manager(monkeypatch)
    monkeypatch.setattr(all2all, "ensure_mori_shmem", lambda: None)
    monkeypatch.setattr(op_mod, "FlyDSLDispatchCombineIntraNodeOp", _FakeOp)
    _FakeOp.built = _FakeHeap.closed = 0

    h = m.get_handle(dict(BASE))
    op = h._op
    del h
    gc.collect()
    assert m.get_handle(dict(BASE))._op is op
    assert _FakeOp.built == 1
    assert m.get_handle(dict(BASE), index=1)._op is not op
    assert _FakeOp.built == 2

    m.destroy()
    assert _FakeHeap.closed == 2
    m.destroy()  # idempotent
    assert _FakeHeap.closed == 2
    assert m.get_handle(dict(BASE))._op is not op
    assert _FakeOp.built == 3
