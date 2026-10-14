"""Check the prepared-metadata ABI independently of expert arithmetic."""

import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from aiter.fused_moe import _adaptive_moe_sort
from k3_gluon_hybrid_sort_b8.host import prepare_metadata
from k3_gluon_hybrid_sort_b8.sort_metadata import prepare_routes


@g.jit
def standalone(
    IDs,
    Weights,
    SortIDs,
    SortWeights,
    SortExperts,
    SortCounts,
    SortM,
    SortReverse,
    SortOut,
):
    prepare_routes(
        IDs,
        Weights,
        SortIDs,
        SortWeights,
        SortExperts,
        SortCounts,
        SortM,
        SortReverse,
        8,
    )
    i = gl.arange(0, 32768, layout=gl.BlockedLayout([1], [64], [8], [0]))
    gl.store(SortOut + i, 0, i < 8 * 3584)


def verify(ids, weights, meta):
    cpu_ids, cpu_weights = ids.cpu(), weights.cpu()
    active = sorted(set(cpu_ids.flatten().tolist()))
    counts = meta["SortCounts"].tolist()
    assert counts == [len(active) * 16, 8], counts
    assert meta["SortExperts"][: len(active)].tolist() == active
    packed, ws = meta["SortIDs"].tolist(), meta["SortWeights"].tolist()
    ms, reverse = meta["SortM"].tolist(), meta["SortReverse"].tolist()
    seen = []
    for block, expert in enumerate(active):
        expected = [
            (s, k) for s in range(8) for k in range(16) if cpu_ids[s, k] == expert
        ]
        actual = []
        for row in range(16):
            p = block * 16 + row
            sample, slot = packed[p] & 0xFFFFFF, packed[p] >> 24
            if sample == 8:
                assert ws[p] == 0 and ms[p] == 8
            else:
                assert 0 <= sample < 8 and 0 <= slot < 16
                assert (
                    cpu_ids[sample, slot] == expert
                    and ws[p] == cpu_weights[sample, slot]
                )
                assert ms[p] == sample and reverse[sample * 16 + slot] == p
                actual.append((sample, slot))
                seen.append(sample * 16 + slot)
        assert actual == expected
    assert sorted(seen) == list(range(128))
    assert bool((meta["SortOut"] == 0).all())
    baseline = _adaptive_moe_sort(
        ids, weights, 896, 16, 16, 3584, atomic=True, emit_aux=True
    )
    assert baseline[3].tolist() == counts
    assert baseline[2][: len(active)].tolist() == active
    return len(active)


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(931)
    ids = torch.arange(128, device="cuda").reshape(8, 16).int()
    weights = torch.rand(8, 16, device="cuda")
    meta = prepare_metadata(ids, weights)
    cases = [
        ids.clone(),
        torch.arange(16, device="cuda").expand(8, 16).int().contiguous(),
        (895 - torch.arange(16, device="cuda")).expand(8, 16).int().contiguous(),
    ]
    cases += [
        torch.stack([torch.randperm(896, device="cuda")[:16] for _ in range(8)]).int()
        for _ in range(20)
    ]
    counts = []
    for case in cases:
        ids.copy_(case)
        for tensor in meta.values():
            tensor.fill_(float("nan") if tensor.is_floating_point() else -999)
        k = standalone[(1,)](ids, weights, **meta, num_warps=8)
        counts.append(verify(ids, weights, meta))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        standalone[(1,)](ids, weights, **meta, num_warps=8)
    for case in (cases[1], cases[0], cases[2], cases[-1]):
        ids.copy_(case)
        weights.uniform_()
        meta["SortOut"].fill_(float("nan"))
        graph.replay()
        verify(ids, weights, meta)
    result = dict(
        cases=len(cases),
        graph_replay_cases=4,
        active_experts=counts,
        registers=k.n_regs,
        spills=k.n_spills,
        lds=k.metadata.shared,
    )
    # The caller can redirect the printed result; no absolute output path.
    print(result, flush=True)


if __name__ == "__main__":
    main()
