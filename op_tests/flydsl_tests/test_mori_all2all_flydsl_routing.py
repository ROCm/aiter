# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which implementation MoriAll2AllManager.get_handle hands out (CPU only).

Supported intra-node high-throughput handles go to the FlyDSL kernels without
touching mori; every other case (inter-node, low latency, a config the FlyDSL
op rejects, an unsupported arch, an EP rank that is not the current device,
``backend="mori"``) builds a mori handle after initialising mori's shmem heap.
The FlyDSL decision is the op's own static config check, the op accepts
mori's ``dispatch(x, w, None, ids)`` on a handle configured with scales, and
the manager keeps its FlyDSL ops until ``destroy()``. Also the launch
geometry the op picks on a device with fewer CUs, and the one combine key the
fused-combine preload and launch share.

    python -m pytest -q op_tests/flydsl_tests/test_mori_all2all_flydsl_routing.py
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
    # EP rank 0 of BASE / BF16_WIRE on its own device
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
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
        # gpt-oss MXFP4 rows: 90 B of e8m0 scales, not whole 4-byte words
        {
            "quant_dtype": torch.float4_e2m1fn_x2,
            "token_hidden_size": 2880,
            "scale_dim": 90,
            "scale_type_size": 1,
            "num_local_experts": 16,
            "num_experts_per_token": 4,
        },
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


def test_rank_not_current_device_uses_mori(manager, monkeypatch):
    """The op keeps its buffers on cuda:{rank}; another current device (e.g.
    ranks placed with an offset) keeps mori."""
    m, _ = manager
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert _impl(m) == "mori"
    assert _impl(m, rank=1) == "flydsl"


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
        ((0, 0), (56, 4), (90, 1), (512, 4)),  # scale_dim, scale_type_size
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


class _FakeOp:
    built = closed = 0
    check_config = staticmethod(op_mod.FlyDSLDispatchCombineIntraNodeOp.check_config)

    def __init__(self, cfg, group=None):
        _FakeOp.built += 1
        self.cfg = cfg

    def close(self):
        _FakeOp.closed += 1


def test_flydsl_ops_live_until_destroy(monkeypatch):
    """A handle the caller dropped is returned again (no second collective
    allocation); destroy() closes every op once."""
    m = _bare_manager(monkeypatch)
    monkeypatch.setattr(all2all, "ensure_mori_shmem", lambda: None)
    monkeypatch.setattr(op_mod, "FlyDSLDispatchCombineIntraNodeOp", _FakeOp)
    _FakeOp.built = _FakeOp.closed = 0

    h = m.get_handle(dict(BASE))
    op = h._op
    del h
    gc.collect()
    assert m.get_handle(dict(BASE))._op is op
    assert _FakeOp.built == 1
    assert m.get_handle(dict(BASE), index=1)._op is not op
    assert _FakeOp.built == 2

    m.destroy()
    assert _FakeOp.closed == 2
    m.destroy()  # idempotent
    assert _FakeOp.closed == 2
    assert m.get_handle(dict(BASE))._op is not op
    assert _FakeOp.built == 3


def test_mori_symmetric_heap_initialises_mori_first(monkeypatch):
    """The attention all-to-all's MoRI heap initialises mori's shmem (over the
    registered EP group) before it reads mori's PE and npes."""
    import sys
    import types

    from aiter.ops.flydsl import communication_ops_host_utils as heaps

    calls = []
    shmem = types.ModuleType("mori.shmem")
    shmem.shmem_mype = lambda: calls.append("mype") or 1
    shmem.shmem_npes = lambda: calls.append("npes") or 2
    mori = types.ModuleType("mori")
    mori.shmem = shmem
    monkeypatch.setitem(sys.modules, "mori", mori)
    monkeypatch.setitem(sys.modules, "mori.shmem", shmem)
    monkeypatch.setattr(all2all, "ensure_mori_shmem", lambda: calls.append("init"))

    heaps.MoriSymmetricHeap(rank=1, world_size=2, device="cpu", group=None)
    assert calls[0] == "init" and calls.count("init") == 1
    with pytest.raises(ValueError, match="MoRI"):
        heaps.MoriSymmetricHeap(rank=0, world_size=2, device="cpu", group=None)


def _table(dispatch, combine):
    from aiter.ops.flydsl.kernels.communication_ops_utils import GeometryTuningTable

    return GeometryTuningTable(dispatch=dispatch, combine=combine)


def test_geometry_capped_at_cu_count(monkeypatch):
    """Tuned rows and the default target a full device: on a device with
    fewer CUs (MI308X: 80) the block count is capped, so the grid-wide barrier
    still has every block resident. A pinned geometry is not capped (the
    residency check rejects it)."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(op_mod, "_device_cu_count", lambda _i: 80)
    table = _table({8: (256, 4), 4096: (64, 8)}, {})
    resolve = op_mod._resolve_launch_geometry
    assert resolve("dispatch", None, None, table, 8, 128, 4) == (80, 4)
    assert resolve("dispatch", None, None, table, 100, 128, 4) == (64, 8)
    assert resolve("combine", None, None, table, 8, 128, 8) == (80, 8)
    assert resolve("combine", 256, 8, table, 8, 128, 8) == (256, 8)
    with pytest.raises(ValueError, match="CU count"):
        op_mod._check_block_num_resident("combine", 256)


def _bare_op(monkeypatch, **cfg_kwargs):
    """An op without buffers: what the CPU tests below need of one."""
    cfg = op_mod.FlyDSLDispatchCombineConfig(
        **{
            "rank": 0,
            "world_size": 8,
            "hidden_dim": 2048,
            "max_num_inp_token_per_rank": 64,
            "num_experts_per_rank": 4,
            "num_experts_per_token": 4,
            **cfg_kwargs,
        }
    )
    op = op_mod.FlyDSLDispatchCombineIntraNodeOp.__new__(
        op_mod.FlyDSLDispatchCombineIntraNodeOp
    )
    op.cfg = cfg
    op._disp_compiled_cache, op._comb_compiled_cache = {}, {}
    op._disp_jit_cache, op._comb_jit_cache, op._loaded = {}, {}, set()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(op_mod, "_device_cu_count", lambda _i: 80)
    return op


def test_every_tuned_geometry_is_preloaded(monkeypatch):
    """Building the op loads the dispatch / combine kernel of every geometry a
    token count can pick (capped), not just one; a new variant loads all of
    them on its first call."""
    op = _bare_op(
        monkeypatch,
        tuning_table=_table(
            {4: (64, 4), 1024: (128, 4), 2048: (128, 8), 4096: (256, 4)},
            {8: (64, 4), 1024: (128, 16)},
        ),
    )
    monkeypatch.setattr(op_mod, "_device_cu_count", lambda _i: 160)
    loaded = []
    monkeypatch.setattr(op, "_preload_dispatch", lambda key: loaded.append(key))
    monkeypatch.setattr(op, "_preload_combine", lambda key: loaded.append(key))
    variant = op._default_variant()
    op._preload_geometries([(torch.bfloat16, False)], [variant])
    assert {tuple(k[2:]) for k in loaded if isinstance(k, op_mod._DispatchKey)} == {
        (64, 4),
        (128, 4),
        (128, 8),
        (160, 4),
    }
    assert {
        (k.block_num, k.warp_num_per_block)
        for k in loaded
        if isinstance(k, op_mod._CombineKey)
    } == {(64, 4), (128, 16)}
    assert all(
        k.variant == variant for k in loaded if isinstance(k, op_mod._CombineKey)
    )

    # no rows: the default geometry, capped
    op = _bare_op(monkeypatch)
    op.cfg.tuning_table = None
    assert op._launch_geometries("dispatch") == [(80, 4)]
    assert op._launch_geometries("combine") == [(80, 8)]


def test_fused_combine_preload_key_is_the_launch_key(monkeypatch):
    """preload_combine_no_stage1() compiles the kernel under the key that
    combine_no_stage1() launches for the same call, so MegaMoE's preload is
    what the first live call runs (nothing is compiled there)."""
    monkeypatch.setattr(
        op_mod.FlyDSLDispatchCombineIntraNodeOp, "_ENABLE_COMBINE_NO_STAGE1", True
    )
    op = _bare_op(monkeypatch, combine_dtype=torch.bfloat16)
    op.cfg.tuning_table = None
    op._dev = torch.device("cpu")
    op._check_tensor_device = lambda name, t: None  # CPU tensors stand in
    mt, h, k = 64, 2048, 4
    op.shmem_disp_out_wts = torch.zeros(8 * mt * k)
    op.shmem_comb_out_tok = torch.zeros(mt * h, dtype=torch.int16)
    op.shmem_comb_out_wts = torch.zeros(mt * k)
    op._combine_args = lambda *a, **kw: ()
    op._get_combine_jit = lambda key: key
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: None)
    preloads, launches = [], []
    monkeypatch.setattr(op_mod, "_preload_compiled", lambda fn, *a: preloads.append(fn))
    monkeypatch.setattr(
        op_mod.flyc,
        "compile",
        lambda fn, *a: launches.append(fn) or (lambda *b: launches.append(fn)),
    )
    x = torch.zeros(16, h, dtype=torch.bfloat16)
    keys = []
    for quant in ("none", "fp8_blockwise_1x32"):
        keys.append(
            op.preload_combine_no_stage1(
                x, cur_tok=16, enable_weights=False, stage2_p2p_quant=quant
            )
        )
        assert preloads == keys
        op.combine_no_stage1(
            x, None, None, cur_tok=16, enable_weights=False, stage2_p2p_quant=quant
        )
        op.combine_no_stage1(
            x, None, None, cur_tok=16, enable_weights=False, stage2_p2p_quant=quant
        )
    # each call launched its preloaded key and compiled nothing more
    assert preloads == keys and len(set(keys)) == 2
    assert launches == [keys[0], keys[0], keys[1], keys[1]]


@pytest.mark.parametrize("shared", ["gm_payload_arena", "combine_input"])
def test_shared_mori_buffers_need_no_group(shared):
    """The buffers MegaMoE v2 shares between ops (a group-major payload arena,
    a combine input) live on mori's heap: an op with a group (HIP IPC buffers)
    rejects them before it allocates anything."""
    cfg = op_mod.FlyDSLDispatchCombineConfig(
        rank=0,
        world_size=8,
        hidden_dim=2048,
        max_num_inp_token_per_rank=64,
        num_experts_per_rank=4,
        num_experts_per_token=4,
    )
    with pytest.raises(ValueError, match="only supported without a group"):
        op_mod.FlyDSLDispatchCombineIntraNodeOp(
            cfg, group=object(), **{shared: object()}
        )


def test_group_major_arena_needs_mori_heap():
    """A group-major op on a heap with a group (HIP IPC) rejects the arena."""
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="only supported without a group"):
        op_mod.FlyDSLDispatchGroupMajorOp(
            rank=0,
            world_size=8,
            hidden_dim=2048,
            max_tok_per_rank=64,
            experts_per_rank=4,
            topk=4,
            data_type=torch.float8_e4m3fn,
            unit_size=32,
            scale_dim=64,
            heap=SimpleNamespace(group="ep group"),
            payload_arena=object(),
        )
