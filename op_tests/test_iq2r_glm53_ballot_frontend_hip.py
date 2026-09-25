# SPDX-License-Identifier: MIT
"""Compare ballot routing with stable CPU ordering and native FP8 quantization."""

import pytest
import torch

from aiter.ops.iq2r import iq2r_route_direct_gather_quant_out

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="requires gfx950",
)


@pytest.mark.parametrize("tokens", range(2, 17))
@pytest.mark.parametrize("pattern", ["hot", "spread", "invalid"])
def test_stable_routing_graph(tokens, pattern, monkeypatch):
    g = torch.Generator(device="cuda").manual_seed(139000 + tokens)
    hidden = torch.randn(
        (tokens, 6144), device="cuda", dtype=torch.bfloat16, generator=g
    )
    ids = torch.empty(tokens * 9, device="cuda", dtype=torch.int32)

    def change(step):
        hidden.normal_(generator=g)
        if pattern == "hot":
            v = torch.arange(9, device="cuda").repeat(tokens)
        else:
            v = torch.randint(
                -2 if pattern == "invalid" else 0,
                260 if pattern == "invalid" else 257,
                (tokens * 9,),
                device="cuda",
                generator=g,
            )
        ids.copy_(v)

    def state():
        return [torch.empty_like(ids) for _ in range(3)] + [
            torch.zeros((tokens * 9, 3), device="cuda", dtype=torch.int32),
            torch.zeros(1, device="cuda", dtype=torch.int32),
            torch.empty((tokens * 9, 6144), device="cuda", dtype=torch.float8_e4m3fn),
            torch.empty((tokens * 9, 192), device="cuda", dtype=torch.uint8),
        ]

    states = {}
    change(0)
    for flag in ["0", "1"]:
        monkeypatch.setenv("IQ2R_GLM53_BALLOT_ROUTER", "1")
        monkeypatch.setenv("IQ2R_GLM53_STATIC_BALLOT", flag)
        s = state()

        def run():
            iq2r_route_direct_gather_quant_out(
                hidden,
                ids,
                *s,
                topk=9,
                expert_count=257,
                expert_map=None,
                expert_start=0,
                expert_stride=1,
                drop_nonlocal_routes=False,
            )

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        states[flag] = (s, graph)
    for step in range(6):
        change(step)
        for s, graph in states.values():
            graph.replay()
        a, b = [v[0] for v in states.values()]
        normalized = [v if 0 <= v < 257 else -1 for v in ids.cpu().tolist()]
        order = sorted(
            range(tokens * 9),
            key=lambda i: (normalized[i] if normalized[i] >= 0 else 257, i),
        )
        assert a[1].cpu().tolist() == order
        assert a[0].cpu().tolist() == [normalized[i] for i in order]
        inverse = [0] * len(order)
        for j, i in enumerate(order):
            inverse[i] = j
        assert a[2].cpu().tolist() == inverse
        expected = []
        offset = 0
        from collections import Counter

        counts = Counter(normalized)
        for expert in sorted(counts, key=lambda v: v if v >= 0 else 257):
            for begin in range(0, counts[expert], 16):
                expected.append(
                    [offset + begin, min(16, counts[expert] - begin), expert]
                )
            offset += counts[expert]
        count = a[4].item()
        assert count == len(expected)
        assert a[3][:count].cpu().tolist() == expected
        assert b[4].item() == count
        for i in [0, 1, 2, 4, 5, 6]:
            assert torch.equal(a[i].view(torch.uint8), b[i].view(torch.uint8)), (
                tokens,
                pattern,
                step,
                i,
            )
        assert torch.equal(a[3][:count], b[3][:count])
