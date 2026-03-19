from contextlib import contextmanager
import importlib

import torch

from aiter.ops.triton.attention.unified_attention import (
    _launch_with_optional_amd_buffer_ops_guard,
    _max_kv_offset_elems,
    _needs_disable_amd_buffer_ops,
)


def test_max_kv_offset_elems_matches_strides() -> None:
    cache = torch.empty((4, 64, 8, 128), device="meta")
    s0, s1, s2, s3 = [abs(int(s)) for s in cache.stride()]
    expected = (cache.shape[0] - 1) * s0 + (63 * s1) + (7 * s2) + (127 * s3)

    assert _max_kv_offset_elems(cache, 64, 8, 128) == expected


def test_needs_disable_amd_buffer_ops_false_for_regular_layouts() -> None:
    key_cache = torch.empty((4, 64, 8, 128), device="meta")
    value_cache = torch.empty((4, 64, 8, 128), device="meta")

    assert not _needs_disable_amd_buffer_ops(key_cache, value_cache, 64, 8, 128)


def test_needs_disable_amd_buffer_ops_true_for_large_offsets() -> None:
    large_stride = (1_100_000_000, 1024, 128, 1)
    key_cache = torch.empty_strided((4, 64, 8, 128), large_stride, device="meta")
    value_cache = torch.empty_strided((4, 64, 8, 128), large_stride, device="meta")

    assert _needs_disable_amd_buffer_ops(key_cache, value_cache, 64, 8, 128)


class _FakeAmdKnobs:
    def __init__(self) -> None:
        self.use_buffer_ops = True
        self.scope_entries = 0

    @contextmanager
    def scope(self):
        self.scope_entries += 1
        previous = self.use_buffer_ops
        try:
            yield
        finally:
            self.use_buffer_ops = previous


def test_launch_guard_disables_buffer_ops_when_requested(monkeypatch) -> None:
    module = importlib.import_module("aiter.ops.triton.attention.unified_attention")
    fake_amd = _FakeAmdKnobs()
    seen = []

    monkeypatch.setattr(module.triton.knobs, "amd", fake_amd)

    _launch_with_optional_amd_buffer_ops_guard(
        True, lambda: seen.append(fake_amd.use_buffer_ops)
    )

    assert fake_amd.scope_entries == 1
    assert seen == [False]


def test_launch_guard_leaves_buffer_ops_enabled_when_not_needed(monkeypatch) -> None:
    module = importlib.import_module("aiter.ops.triton.attention.unified_attention")
    fake_amd = _FakeAmdKnobs()
    seen = []

    monkeypatch.setattr(module.triton.knobs, "amd", fake_amd)

    _launch_with_optional_amd_buffer_ops_guard(
        False, lambda: seen.append(fake_amd.use_buffer_ops)
    )

    assert fake_amd.scope_entries == 0
    assert seen == [True]


class _FakeKernel:
    def __init__(self, seen) -> None:
        self.seen = seen

    def __getitem__(self, _grid):
        def _launch(**_kwargs):
            amd_knobs = self.seen["amd"]
            self.seen["values"].append(amd_knobs.use_buffer_ops)

        return _launch


def _dispatch_inputs():
    q = torch.empty((2, 8, 64), device="meta")
    k = torch.empty((4, 64, 1, 64), device="meta")
    v = torch.empty((4, 64, 1, 64), device="meta")
    out = torch.empty_like(q)
    cu_seqlens_q = torch.tensor([0, 2], dtype=torch.int32)
    seqused_k = torch.tensor([32], dtype=torch.int32)
    block_table = torch.zeros((1, 1), dtype=torch.int32)
    return q, k, v, out, cu_seqlens_q, seqused_k, block_table


def _configure_dispatch_case(monkeypatch, use_2d_branch: bool):
    module = importlib.import_module("aiter.ops.triton.attention.unified_attention")
    fake_amd = _FakeAmdKnobs()
    launch_seen = {"amd": fake_amd, "values": []}
    reduce_seen = {"amd": fake_amd, "values": []}

    monkeypatch.setattr(module.triton.knobs, "amd", fake_amd)
    monkeypatch.setattr(module.torch.version, "hip", "mock", raising=False)
    monkeypatch.setattr(module, "get_num_sms", lambda: 1)
    monkeypatch.setattr(module, "_needs_disable_amd_buffer_ops", lambda *args: True)
    monkeypatch.setattr(module, "use_2d_kernel", lambda *args: use_2d_branch)
    monkeypatch.setattr(
        module,
        "select_2d_config",
        lambda *args: {
            "BLOCK_M": 16,
            "BLOCK_Q": 2,
            "TILE_SIZE": 64,
            "num_warps": 2,
            "num_stages": 1,
            "waves_per_eu": 2,
        },
    )
    monkeypatch.setattr(
        module,
        "select_3d_config",
        lambda *args: (
            {
                "TILE_SIZE": 64,
                "NUM_SEGMENTS_PER_SEQ": 2,
                "num_warps": 2,
                "num_stages": 1,
                "waves_per_eu": 2,
            },
            {
                "TILE_SIZE": 64,
                "NUM_SEGMENTS_PER_SEQ": 2,
                "num_warps": 1,
                "num_stages": 1,
                "waves_per_eu": 2,
            },
        ),
    )
    monkeypatch.setattr(module, "kernel_unified_attention_2d", _FakeKernel(launch_seen))
    monkeypatch.setattr(module, "kernel_unified_attention_3d", _FakeKernel(launch_seen))
    monkeypatch.setattr(module, "reduce_segments", _FakeKernel(reduce_seen))

    return module, fake_amd, launch_seen["values"], reduce_seen["values"]


def test_unified_attention_2d_launch_disables_buffer_ops(monkeypatch) -> None:
    module, fake_amd, launch_seen, _ = _configure_dispatch_case(monkeypatch, True)
    q, k, v, out, cu_seqlens_q, seqused_k, block_table = _dispatch_inputs()

    module.unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=2,
        seqused_k=seqused_k,
        max_seqlen_k=32,
        softmax_scale=1.0,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )

    assert fake_amd.scope_entries == 1
    assert launch_seen == [False]


def test_unified_attention_3d_launch_disables_buffer_ops(monkeypatch) -> None:
    module, fake_amd, launch_seen, reduce_seen = _configure_dispatch_case(
        monkeypatch, False
    )
    q, k, v, out, cu_seqlens_q, seqused_k, block_table = _dispatch_inputs()

    module.unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=2,
        seqused_k=seqused_k,
        max_seqlen_k=32,
        softmax_scale=1.0,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )

    assert fake_amd.scope_entries == 1
    assert launch_seen == [False]
    assert reduce_seen == [True]
