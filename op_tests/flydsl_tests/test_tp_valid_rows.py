"""Bounded gfx1250 TP route/quant/gather validation; never launch invalid IDs unmasked."""

import json

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.grouped_moe_gfx1250 import flydsl_moe_gather_reduce
from aiter.ops.flydsl.moe_kernels import (
    flydsl_moe_fused_quant_preshuffle,
    flydsl_moe_topids_to_rows,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx1250", reason="Requires gfx1250"
)
def test_tp_valid_rows_route_quant_and_graph_replay():
    torch.manual_seed(7)
    M, K, E, TOPK, MAX_M, REP = 80, 5120, 8, 3, 256, 4
    device = "cuda"
    hidden = torch.randn((M, K), device=device, dtype=torch.bfloat16)
    all_ids = torch.stack(
        [torch.randperm(E, device=device)[:TOPK] for _ in range(M)]
    ).int()
    report = []

    def scales_for_rows(scale, rows):
        rows = rows.long().reshape(-1)
        width = K // 32
        base = (rows // (REP * 16)) * (width // 4 * REP * 16) + rows % (REP * 16)
        offsets = base[:, None] + torch.arange(width // 4, device=device)[None, :] * (
            REP * 16
        )
        return (
            scale.reshape(-1).view(torch.int32)[offsets].contiguous().view(torch.uint8)
        )

    def quant(x, rows, nvr=None):
        payload = torch.full((E, MAX_M, K), 0xA5, dtype=torch.uint8, device=device)
        scale = torch.full(
            (E, MAX_M // REP, (K // 32) * REP), 0x5A, dtype=torch.uint8, device=device
        )
        flydsl_moe_fused_quant_preshuffle(
            x.reshape(1, x.shape[0], K),
            E,
            MAX_M,
            wmma_rep=REP,
            quant_mode="fp8",
            topids_to_rows=rows,
            source_topk=TOPK,
            num_valid_routes=nvr,
            out_payload=payload,
            out_scale=scale,
        )
        torch.cuda.synchronize()
        return payload, scale

    for n in [0, 1, 63, 64, 79, 80]:
        ids = all_ids.clone()
        ids[n:] = -1
        nvt = torch.tensor([n], dtype=torch.int32, device=device)
        count, rows = flydsl_moe_topids_to_rows(ids, E, MAX_M, num_local_tokens=nvt)
        torch.cuda.synchronize()
        expected = torch.bincount(ids[:n].reshape(-1).long(), minlength=E)
        assert torch.equal(count.long(), expected), (n, count, expected)
        assert (rows[n:] == -1).all(), (n, rows[n:])
        assert ((rows[:n] >= 0) & (rows[:n] < E * MAX_M)).all()
        if n:
            assert (rows[:n] // MAX_M == ids[:n]).all()
            assert rows[:n].unique().numel() == n * TOPK
        payload, scale = quant(hidden, rows, nvt * TOPK)
        if n == 0:
            assert (payload == 0xA5).all() and (scale == 0x5A).all()
        else:
            # Existing path receives only real rows with valid IDs. Compare every
            # routed payload and scale byte, independent of atomic slot ordering.
            base_count, base_rows = flydsl_moe_topids_to_rows(all_ids[:n], E, MAX_M)
            base_payload, base_scale = quant(hidden[:n], base_rows)
            assert torch.equal(count, base_count)
            assert torch.equal(
                payload.reshape(-1, K)[rows[:n].long()],
                base_payload.reshape(-1, K)[base_rows.long()],
            ), n
            assert torch.equal(
                scales_for_rows(scale, rows[:n]), scales_for_rows(base_scale, base_rows)
            ), n
            grouped = torch.randn((E, MAX_M, K), device=device, dtype=torch.bfloat16)
            weights = torch.rand((M, TOPK), device=device, dtype=torch.float32)
            out = torch.zeros_like(hidden)
            flydsl_moe_gather_reduce(
                grouped, rows, weights, out=out, num_valid_tokens=nvt
            )
            expected_out = (
                (grouped.reshape(-1, K)[rows[:n].long()].float() * weights[:n, :, None])
                .sum(dim=1)
                .to(torch.bfloat16)
            )
            torch.testing.assert_close(out[:n], expected_out, atol=0.03125, rtol=0.02)
        row = {
            "tokens": M,
            "valid": n,
            "topk": TOPK,
            "features": K,
            "route_count": int(count.sum()),
            "passed": True,
        }
        report.append(row)
        print(json.dumps(row), flush=True)

    # Replay must read the device scalar again, not freeze the capture-time count.
    nvt = torch.tensor([64], dtype=torch.int32, device=device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            flydsl_moe_topids_to_rows(all_ids, E, MAX_M, num_local_tokens=nvt)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        graph_count, graph_rows = flydsl_moe_topids_to_rows(
            all_ids, E, MAX_M, num_local_tokens=nvt
        )
    for n in [0, 1, 64, 80, 63]:
        nvt.fill_(n)
        graph.replay()
        torch.cuda.synchronize()
        assert int(graph_count.sum()) == n * TOPK
        assert (graph_rows[n:] == -1).all()
        assert torch.equal(
            graph_count.long(),
            torch.bincount(all_ids[:n].reshape(-1).long(), minlength=E),
        )
        print("GRAPH_REPLAY_PASS", n, flush=True)

    # Invalid IDs inside the live prefix are also guarded before the atomic.
    bad_ids = all_ids.clone()
    bad_ids[0, 0] = -1
    bad_ids[0, 1] = E + 5
    nvt.fill_(M)
    count, rows = flydsl_moe_topids_to_rows(bad_ids, E, MAX_M, num_local_tokens=nvt)
    torch.cuda.synchronize()
    assert int(count.sum()) == M * TOPK - 2
    assert rows[0, 0] == rows[0, 1] == -1
    report.append(
        {"graph_replay_counts": [0, 1, 64, 80, 63], "invalid_live_ids_guarded": True}
    )
    print("TP_VALID_ROWS_PROBE_PASSED", flush=True)


@pytest.mark.parametrize("valid", [None, 0, 1, 64, 80])
def test_grouped_tp_propagates_count_before_quant(monkeypatch, valid):
    from aiter import ActivationType
    from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped
    from aiter.ops.flydsl import moe_kernels

    class ReachedRemap(Exception):
        pass

    calls = []

    def route(ids, experts, max_m, **kwargs):
        calls.append(kwargs["num_valid_routes"])
        return torch.zeros(experts, dtype=torch.int32), torch.zeros_like(ids)

    def remap(*args, **kwargs):
        calls.append(kwargs["num_valid_routes"])
        raise ReachedRemap

    monkeypatch.setattr(grouped, "get_wmma_m_rep", lambda *args: 4)
    monkeypatch.setattr(grouped, "_flydsl_dispatch_context", lambda: None)
    monkeypatch.setattr(moe_kernels, "flydsl_moe_topids_to_rows", route)
    monkeypatch.setattr(grouped, "contiguous_psum_remap", remap)
    with pytest.raises(ReachedRemap):
        grouped._grouped_a8w4_tdm_moe(
            torch.zeros((80, 5120), dtype=torch.bfloat16),
            None,
            None,
            torch.ones((80, 3)),
            torch.zeros((80, 3), dtype=torch.int32),
            E=8,
            model_dim=5120,
            inter_dim=256,
            dtype=torch.bfloat16,
            activation=ActivationType.Silu,
            w1_scale=None,
            w2_scale=None,
            bias1=None,
            bias2=None,
            swiglu_limit=0,
            doweight_stage1=False,
            num_local_tokens=(
                None if valid is None else torch.tensor([valid], dtype=torch.int32)
            ),
        )
    assert len(calls) == 2
    if valid is None:
        assert calls == [None, None]
    else:
        assert calls[0] is calls[1]
        assert calls[0].tolist() == [valid * 3]
