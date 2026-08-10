# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness and contract tests for the packed Opus GDN WS path."""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
import torch.nn.functional as F

import aiter.ops.gdn_prefill as adapter
import aiter.ops.opus_gdn_wu_prefill as opus_wu
from aiter.ops.opus_gdn_wu_prefill import (
    _prepare_opus_gdn_varlen_metadata,
    opus_gdn_wu_prefill_fwd,
)
from aiter.ops.triton.gated_delta_net import chunk_gated_delta_rule_opt_vk

_D = 128


def _require_opus_device() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Opus GDN varlen requires a ROCm GPU")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    gfx = properties.gcnArchName.split(":", 1)[0]
    if gfx not in ("gfx942", "gfx950"):
        pytest.skip(f"Opus W/U kernels require gfx942/gfx950, got {gfx}")


def _cu_from_lens(lens: list[int]) -> torch.Tensor:
    values = [0]
    for length in lens:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _make_inputs(
    lens: list[int],
    heads: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    total = sum(lens)
    q = F.normalize(
        torch.randn(1, total, heads, _D, device="cuda", dtype=torch.float32),
        dim=-1,
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn(1, total, heads, _D, device="cuda", dtype=torch.float32),
        dim=-1,
    ).to(torch.bfloat16)
    v = (torch.randn(1, total, heads, _D, device="cuda", dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )
    g = F.logsigmoid(torch.randn(1, total, heads, device="cuda", dtype=torch.float32))
    beta = torch.sigmoid(torch.randn_like(g)).to(torch.bfloat16)
    cu_seqlens = _cu_from_lens(lens)
    initial_state = (
        torch.randn(len(lens), heads, _D, _D, device="cuda", dtype=torch.float32) * 0.01
    )
    return q, k, v, g, beta, cu_seqlens, initial_state


@pytest.mark.parametrize(
    ("lens", "heads"),
    (
        pytest.param([15], 1, id="single-tail"),
        pytest.param([1, 63, 64, 65, 129], 2, id="boundary-mix"),
        pytest.param([15, 85, 200, 900], 4, id="multi-chunk-ragged"),
        pytest.param([64, 128, 256], 4, id="aligned-packed"),
    ),
)
@pytest.mark.parametrize(
    ("with_initial_state", "output_final_state"),
    (
        pytest.param(False, False, id="stateless"),
        pytest.param(False, True, id="final-only"),
        pytest.param(True, False, id="initial-only"),
        pytest.param(True, True, id="state-io"),
    ),
)
def test_opus_gdn_varlen_matches_triton(
    lens: list[int],
    heads: int,
    with_initial_state: bool,
    output_final_state: bool,
) -> None:
    _require_opus_device()
    q, k, v, g, beta, cu_seqlens, state = _make_inputs(
        lens, heads, seed=20260804 + sum(lens) + heads
    )
    initial_state = state if with_initial_state else None

    actual, actual_final = opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        k2_mode=2,
        use_env_overrides=False,
        cu_seqlens=cu_seqlens,
    )
    expected, expected_final = chunk_gated_delta_rule_opt_vk(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    if output_final_state:
        assert actual_final is not None and expected_final is not None
        assert actual_final.dtype == torch.float32
        assert tuple(actual_final.shape) == (
            len(lens),
            heads,
            _D,
            _D,
        )
        torch.testing.assert_close(actual_final, expected_final, rtol=1e-2, atol=2e-3)
    else:
        assert actual_final is None


def test_opus_gdn_varlen_preallocated_output_may_equal_v() -> None:
    _require_opus_device()
    q, k, v, g, beta, cu_seqlens, _ = _make_inputs([1, 63, 64, 65], 2, seed=17)
    expected, _ = opus_gdn_wu_prefill_fwd(
        q,
        k,
        v.clone(),
        g,
        beta,
        cu_seqlens=cu_seqlens,
        use_env_overrides=False,
    )

    aliased = v.clone()
    actual, final_state = opus_gdn_wu_prefill_fwd(
        q,
        k,
        aliased,
        g,
        beta,
        out=aliased,
        cu_seqlens=cu_seqlens,
        use_env_overrides=False,
    )

    assert actual is aliased
    assert final_state is None
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("aliased_input", ("q", "k"))
def test_opus_gdn_varlen_rejects_output_aliasing_q_or_k(
    aliased_input: str,
) -> None:
    _require_opus_device()
    q, k, v, g, beta, cu_seqlens, _ = _make_inputs([65], 1, seed=19)
    out = q if aliased_input == "q" else k

    with pytest.raises(
        ValueError, match=rf"out must not alias {aliased_input} storage"
    ):
        opus_gdn_wu_prefill_fwd(
            q,
            k,
            v,
            g,
            beta,
            out=out,
            cu_seqlens=cu_seqlens,
            use_env_overrides=False,
        )


def test_opus_gdn_varlen_rejects_partial_output_v_overlap() -> None:
    _require_opus_device()
    q, k, source_v, g, beta, cu_seqlens, _ = _make_inputs([65], 1, seed=23)
    storage = torch.empty(
        source_v.numel() + 1, device=source_v.device, dtype=source_v.dtype
    )
    v = storage[:-1].view_as(source_v)
    out = storage[1:].view_as(source_v)
    v.copy_(source_v)

    with pytest.raises(ValueError, match="exactly the same view"):
        opus_gdn_wu_prefill_fwd(
            q,
            k,
            v,
            g,
            beta,
            out=out,
            cu_seqlens=cu_seqlens,
            use_env_overrides=False,
        )


def test_gdn_prefill_auto_rejects_unsafe_output_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_opus_device()
    q, k, v, g, beta, cu_seqlens, _ = _make_inputs([65], 1, seed=25)
    monkeypatch.setattr(adapter, "_runtime_target", lambda _: ("gfx942", 80))
    monkeypatch.setattr(
        adapter,
        "chunk_gated_delta_rule_opt_vk",
        lambda **kwargs: pytest.fail("unsafe alias must not reach the fallback"),
    )

    with pytest.raises(ValueError, match="o must not alias q storage"):
        adapter.gdn_prefill(
            q,
            k,
            v,
            o=q,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
        )


def test_opus_gdn_varlen_matches_independent_dense_ws_sequences() -> None:
    _require_opus_device()
    lens = [1, 63, 64, 65]
    q, k, v, g, beta, cu_seqlens, state = _make_inputs(lens, 2, seed=27)
    packed, packed_final = opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_env_overrides=False,
    )

    dense_outputs = []
    dense_finals = []
    start = 0
    for sequence_id, length in enumerate(lens):
        end = start + length
        dense, dense_final = opus_gdn_wu_prefill_fwd(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            initial_state=state[sequence_id : sequence_id + 1],
            output_final_state=True,
            k2_mode=2,
            use_env_overrides=False,
        )
        dense_outputs.append(dense)
        dense_finals.append(dense_final)
        start = end

    assert torch.equal(packed, torch.cat(dense_outputs, dim=1))
    torch.testing.assert_close(
        packed_final,
        torch.cat(dense_finals, dim=0),
        rtol=0,
        atol=5e-4,
    )


def test_opus_gdn_varlen_metadata_cache_tracks_inplace_mutation() -> None:
    _require_opus_device()
    q, k, v, g, beta, cu_seqlens, state = _make_inputs([64, 64], 2, seed=31)

    # Populate this Tensor's metadata entry, then retain the same identity while
    # changing the sequence/chunk partition.
    opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens=cu_seqlens,
        use_env_overrides=False,
    )
    cu_seqlens.copy_(torch.tensor([0, 1, 128], device="cuda", dtype=torch.int32))

    actual, actual_final = opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_env_overrides=False,
    )
    expected, expected_final = chunk_gated_delta_rule_opt_vk(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens.clone(),
    )

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_final, expected_final, rtol=1e-2, atol=2e-3)


def test_opus_gdn_varlen_metadata_cache_reuses_immutable_inference_tensor() -> None:
    _require_opus_device()
    with torch.inference_mode():
        cu_seqlens = _cu_from_lens([64, 65, 129])
        first = _prepare_opus_gdn_varlen_metadata(cu_seqlens, 64)
        second = _prepare_opus_gdn_varlen_metadata(cu_seqlens, 64)

    assert second is first
    assert first[-1] == 3


def test_opus_gdn_varlen_metadata_cache_tracks_live_tensor_identities() -> None:
    _require_opus_device()
    first_cu_seqlens = _cu_from_lens([64, 64])
    second_cu_seqlens = _cu_from_lens([1, 127])
    first_key = (id(first_cu_seqlens), 64)
    second_key = (id(second_cu_seqlens), 64)

    first = _prepare_opus_gdn_varlen_metadata(first_cu_seqlens, 64)
    second = _prepare_opus_gdn_varlen_metadata(second_cu_seqlens, 64)

    assert _prepare_opus_gdn_varlen_metadata(first_cu_seqlens, 64) is first
    assert _prepare_opus_gdn_varlen_metadata(second_cu_seqlens, 64) is second
    assert opus_wu._VARLEN_METADATA_CACHE[first_key][0]() is first_cu_seqlens
    assert opus_wu._VARLEN_METADATA_CACHE[second_key][0]() is second_cu_seqlens

    first_source_ref = weakref.ref(first_cu_seqlens)
    first_metadata_refs = tuple(weakref.ref(tensor) for tensor in first[1:4])
    del first, first_cu_seqlens
    gc.collect()

    assert first_source_ref() is None
    assert first_key not in opus_wu._VARLEN_METADATA_CACHE
    assert all(ref() is None for ref in first_metadata_refs)
    assert _prepare_opus_gdn_varlen_metadata(second_cu_seqlens, 64) is second
    assert second_key in opus_wu._VARLEN_METADATA_CACHE


def test_opus_gdn_varlen_metadata_cache_keeps_cross_stream_storage_alive() -> None:
    _require_opus_device()
    q, k, v, g, beta, cu_seqlens, _ = _make_inputs([64, 64], 1, seed=35)

    # Populate the cache and finish a reference invocation on its owner stream.
    expected, _ = opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens=cu_seqlens,
        use_env_overrides=False,
    )
    torch.cuda.synchronize()

    cache_key = (id(cu_seqlens), 64)
    cache_entry = opus_wu._VARLEN_METADATA_CACHE[cache_key]
    assert cache_entry[0]() is cu_seqlens
    cached_result = cache_entry[2]
    cached_ptrs = {
        cached_result[1].data_ptr(),
        cached_result[2].data_ptr(),
        cached_result[3].data_ptr(),
    }
    # Do not retain the old metadata through these local inspection variables.
    del cached_result, cache_entry

    # Keep the consumer stream behind a short device-side delay. The Python
    # call returns after launching its kernels, while they still hold raw
    # pointers to the cached metadata.
    delayed_stream = torch.cuda.Stream()
    warmup_stream = torch.cuda.Stream()
    with torch.cuda.stream(warmup_stream):
        torch.cuda._sleep(1)
    warmup_stream.synchronize()

    actual = torch.full_like(v, 7)
    with torch.cuda.stream(delayed_stream):
        torch.cuda._sleep(50_000_000)
        opus_gdn_wu_prefill_fwd(
            q,
            k,
            v,
            g,
            beta,
            out=actual,
            cu_seqlens=cu_seqlens,
            use_env_overrides=False,
        )

    # Mutating this versioned offsets Tensor replaces its per-identity cache
    # entry and drops the old metadata references. Overwrite any old blocks that
    # become reusable before the delayed stream is allowed to consume them.
    cu_seqlens.copy_(torch.tensor([0, 1, 128], device="cuda", dtype=torch.int32))
    _prepare_opus_gdn_varlen_metadata(cu_seqlens, 64)
    reuse_pressure = []
    reused_ptrs: set[int] = set()
    for _ in range(64):
        allocation = torch.zeros(128, device="cuda", dtype=torch.int32)
        reuse_pressure.append(allocation)
        if allocation.data_ptr() in cached_ptrs:
            reused_ptrs.add(allocation.data_ptr())

    torch.cuda.current_stream().synchronize()
    assert not delayed_stream.query(), "device-side delay was too short for the test"
    delayed_stream.synchronize()

    assert torch.equal(actual, expected), (
        "cached metadata storage was reused before its consumer stream "
        f"completed; reused {len(reused_ptrs)}/{len(cached_ptrs)} buffers"
    )


def test_opus_gdn_varlen_metadata_cache_keeps_graph_replay_storage_alive() -> None:
    _require_opus_device()
    q, k, v, g, beta, cu_seqlens, _ = _make_inputs([64, 64], 1, seed=36)
    next_q, next_k, next_v, next_g, next_beta, _, _ = _make_inputs(
        [64, 64], 1, seed=136
    )
    second_cu_seqlens = _cu_from_lens([1, 127])

    # Warm the JIT and metadata cache before capture. A cache miss calls
    # Tensor.tolist(), which is intentionally not part of the captured graph.
    expected, _ = opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens=cu_seqlens,
        use_env_overrides=False,
    )
    next_expected, _ = opus_gdn_wu_prefill_fwd(
        next_q,
        next_k,
        next_v,
        next_g,
        next_beta,
        cu_seqlens=cu_seqlens,
        use_env_overrides=False,
    )
    torch.cuda.synchronize()

    cache_key = (id(cu_seqlens), 64)
    cache_entry = opus_wu._VARLEN_METADATA_CACHE[cache_key]
    assert cache_entry[0]() is cu_seqlens
    cached_result = cache_entry[2]
    cached_result_id = id(cached_result)
    cached_refs = tuple(weakref.ref(tensor) for tensor in cached_result[1:4])
    del cached_result, cache_entry

    # A second live offsets identity must remain hot alongside the one that the
    # graph will capture; a one-entry cache would make the first call miss during
    # capture and attempt the non-capturable metadata build.
    second_result = _prepare_opus_gdn_varlen_metadata(second_cu_seqlens, 64)
    second_key = (id(second_cu_seqlens), 64)
    assert id(_prepare_opus_gdn_varlen_metadata(cu_seqlens, 64)) == cached_result_id
    assert _prepare_opus_gdn_varlen_metadata(second_cu_seqlens, 64) is second_result
    assert cache_key in opus_wu._VARLEN_METADATA_CACHE
    assert second_key in opus_wu._VARLEN_METADATA_CACHE
    assert all(ref() is not None for ref in cached_refs)

    downstream_output = torch.full_like(v, 7)
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, _ = opus_gdn_wu_prefill_fwd(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens=cu_seqlens,
            use_env_overrides=False,
        )
        downstream_output.copy_(captured_output)
    torch.cuda.synchronize()

    # Establish that the captured graph itself is correct before replaying it
    # with new q/k/v/g/beta contents.
    downstream_output.fill_(7)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(downstream_output, expected)

    # A whole-model graph need not retain Python wrappers for intermediate
    # tensors. Keep only the downstream output and the static offsets input,
    # whose lifetimes must therefore keep the private metadata alive.
    captured_output_ref = weakref.ref(captured_output)
    del captured_output
    assert captured_output_ref() is None

    # A graph embeds the metadata pointers for future replays. The exact static
    # offsets input remains alive, so its weak-keyed entry must retain all three
    # derived tensors even with another identity resident in the cache.
    assert all(ref() is not None for ref in cached_refs), (
        "captured metadata tensors were released while their static offsets "
        "input was still alive"
    )
    assert cache_key in opus_wu._VARLEN_METADATA_CACHE
    assert second_key in opus_wu._VARLEN_METADATA_CACHE

    q.copy_(next_q)
    k.copy_(next_k)
    v.copy_(next_v)
    g.copy_(next_g)
    beta.copy_(next_beta)
    downstream_output.fill_(7)
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(downstream_output, next_expected)


def test_opus_gdn_varlen_int64_metadata_mixed_gates_and_custom_scale() -> None:
    _require_opus_device()
    q, k, v, g, beta, cu_seqlens, _ = _make_inputs([63, 65], 2, seed=37)
    g = g.to(torch.bfloat16)
    beta = beta.float()
    cu_seqlens = cu_seqlens.to(torch.int64)
    scale = 0.25

    actual, _ = opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        use_env_overrides=False,
    )
    expected, _ = chunk_gated_delta_rule_opt_vk(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_opus_gdn_varlen_sequences_are_isolated() -> None:
    _require_opus_device()
    q, k, v, g, beta, cu_seqlens, state = _make_inputs([65, 129], 2, seed=29)
    baseline, baseline_final = opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_env_overrides=False,
    )

    split = int(cu_seqlens[1].item())
    q2, k2, v2, g2, beta2, state2 = (
        tensor.clone() for tensor in (q, k, v, g, beta, state)
    )
    q2[:, :split].zero_()
    k2[:, :split].zero_()
    v2[:, :split].zero_()
    g2[:, :split].fill_(-2.0)
    beta2[:, :split].fill_(0.25)
    state2[0].zero_()

    perturbed, perturbed_final = opus_gdn_wu_prefill_fwd(
        q2,
        k2,
        v2,
        g2,
        beta2,
        initial_state=state2,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_env_overrides=False,
    )

    assert torch.equal(perturbed[:, split:], baseline[:, split:])
    assert torch.equal(perturbed_final[1], baseline_final[1])


@pytest.mark.parametrize(
    ("values", "message"),
    (
        pytest.param([1, 65], "endpoints", id="bad-start"),
        pytest.param([0, 64], "endpoints", id="bad-end"),
        pytest.param([0, 32, 32, 65], "strictly increasing", id="empty-seq"),
    ),
)
def test_opus_gdn_varlen_rejects_invalid_cu_seqlens(
    values: list[int], message: str
) -> None:
    _require_opus_device()
    q, k, v, g, beta, _, _ = _make_inputs([65], 1, seed=41)
    cu_seqlens = torch.tensor(values, dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match=message):
        opus_gdn_wu_prefill_fwd(q, k, v, g, beta, cu_seqlens=cu_seqlens)


def test_opus_gdn_varlen_rejects_invalid_metadata_structure() -> None:
    _require_opus_device()
    q, k, v, g, beta, _, _ = _make_inputs([65], 1, seed=43)
    invalid = (
        (
            torch.tensor([[0, 65]], dtype=torch.int32, device="cuda"),
            "shape",
        ),
        (torch.tensor([0, 65], dtype=torch.float32, device="cuda"), "dtype"),
        (
            torch.tensor([0, -1, 65, -1], dtype=torch.int32, device="cuda")[::2],
            "contiguous",
        ),
    )

    for cu_seqlens, message in invalid:
        with pytest.raises(ValueError, match=message):
            opus_gdn_wu_prefill_fwd(q, k, v, g, beta, cu_seqlens=cu_seqlens)


def test_opus_gdn_varlen_rejects_dense_batch_and_zero_bv() -> None:
    _require_opus_device()
    q, k, v, g, beta, cu_seqlens, _ = _make_inputs([65], 1, seed=47)
    with pytest.raises(ValueError, match="batch dimension B=1"):
        opus_gdn_wu_prefill_fwd(
            q.expand(2, -1, -1, -1).contiguous(),
            k.expand(2, -1, -1, -1).contiguous(),
            v.expand(2, -1, -1, -1).contiguous(),
            g.expand(2, -1, -1).contiguous(),
            beta.expand(2, -1, -1).contiguous(),
            cu_seqlens=cu_seqlens,
        )
    with pytest.raises(ValueError, match="Unsupported BV=0"):
        opus_gdn_wu_prefill_fwd(q, k, v, g, beta, BV=0)


def test_varlen_route_bypasses_dense_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_opus_device()
    q, k, v, g, beta, cu_seqlens, _ = _make_inputs([32, 33], 1, seed=53)
    monkeypatch.setattr(adapter, "_runtime_target", lambda _: ("gfx942", 80))
    monkeypatch.setattr(
        adapter,
        "lookup_dense_gfx942_path",
        lambda *args: pytest.fail("varlen must bypass the dense route table"),
    )

    assert (
        adapter.select_gdn_prefill_path(q, k, v, g=g, beta=beta, cu_seqlens=cu_seqlens)
        == "ws"
    )
    with pytest.raises(ValueError, match="W/U split"):
        adapter.select_gdn_prefill_path(
            q,
            k,
            v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            path="wf",
        )


def test_varlen_auto_gfx950_falls_back_without_preparing_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_opus_device()
    q, k, v, g, beta, cu_seqlens, _ = _make_inputs([32, 33], 1, seed=59)
    monkeypatch.setattr(adapter, "_runtime_target", lambda _: ("gfx950", 80))
    monkeypatch.setattr(
        adapter,
        "_prepare_opus_gdn_varlen_metadata",
        lambda *args: pytest.fail("unsupported auto target must not sync metadata"),
    )

    assert (
        adapter.select_gdn_prefill_path(q, k, v, g=g, beta=beta, cu_seqlens=cu_seqlens)
        == "triton"
    )


@pytest.mark.parametrize("path", ("wu", "ws"))
def test_varlen_explicit_wu_paths_select_ws(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_opus_device()
    q, k, v, g, beta, cu_seqlens, _ = _make_inputs([32, 33], 1, seed=60)
    monkeypatch.setattr(adapter, "_runtime_target", lambda _: ("gfx950", 120))

    assert (
        adapter.select_gdn_prefill_path(
            q,
            k,
            v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            path=path,
        )
        == "ws"
    )


def test_gdn_prefill_explicit_ws_runs_packed_kernel() -> None:
    _require_opus_device()
    q, k, v, g, beta, cu_seqlens, state = _make_inputs([15, 65, 129], 2, seed=61)
    expected, expected_final = opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_env_overrides=False,
    )

    out = torch.empty_like(v)
    actual, actual_final = adapter.gdn_prefill(
        q,
        k,
        v,
        o=out,
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        path="ws",
    )

    assert actual is out
    assert torch.equal(actual, expected)
    assert torch.equal(actual_final, expected_final)
