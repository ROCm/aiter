# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest
import torch


def _require_gpu_and_flydsl():
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    from aiter.ops.flydsl.utils import is_flydsl_available

    if not is_flydsl_available():
        pytest.skip("FlyDSL required")


def test_node_partial_reduce_ep8_bf16_and_fp32():
    """Reduce eight simulated rank slots; this intentionally excludes TP."""

    _require_gpu_and_flydsl()
    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import compile_node_partial_reduce
    from aiter.ops.flydsl.kernels.megamoe_tile.runtime import HierCcoArenaLayout

    torch.manual_seed(71)
    dev = torch.device("cuda", 0)
    stream = torch.cuda.current_stream(dev)
    ranks, capacity, active, hidden = 8, 11, 7, 3584
    partials = (
        torch.randn(ranks, capacity, hidden, device=dev) * 0.05
    ).to(torch.bfloat16)

    # A missing rank slot is represented by a zero-filled payload. The count
    # is only an arrival gate; it is deliberately not interpreted as a prefix
    # mask by the reducer.
    present = (
        (0, 1, 2, 3, 4, 5, 6, 7),
        (3,),
        (0, 2, 5, 7),
        (),
        (1, 6),
        (0, 1, 4, 6, 7),
        (2, 3, 4),
    )
    for source, source_ranks in enumerate(present):
        absent = sorted(set(range(ranks)) - set(source_ranks))
        if absent:
            partials[absent, source] = 0
    partials[:, active:].zero_()

    layout = HierCcoArenaLayout.create(
        max_m_tiles=1,
        max_source_tokens=capacity,
    )
    arena = layout.allocate_local(device=dev)
    generation = 41
    ptrs = layout.epoch_pointers(arena.data_ptr(), generation)
    expected = layout.view(
        arena, "rank_route_expected", parity=ptrs.parity
    )
    ready = layout.view(arena, "rank_route_ready", parity=ptrs.parity)
    node_ready = layout.view(
        arena, "node_partial_ready", parity=ptrs.parity
    )
    counts = torch.tensor(
        [len(ranks_for_source) for ranks_for_source in present],
        dtype=torch.int32,
        device=dev,
    )
    expected[:active].copy_(counts)
    ready[:active].copy_(counts)

    reference = torch.zeros((capacity, hidden), dtype=torch.float32, device=dev)
    for rank in range(ranks):
        reference += partials[rank].float()

    for output_dtype, torch_dtype in (
        ("bf16", torch.bfloat16),
        ("fp32", torch.float32),
    ):
        node_ready.zero_()
        output = torch.zeros((capacity, hidden), dtype=torch_dtype, device=dev)
        reducer = compile_node_partial_reduce(
            D_HIDDEN=hidden,
            NUM_RANKS=ranks,
            output_dtype=output_dtype,
        )
        reducer(
            partials.data_ptr(),
            ptrs.rank_route_ready,
            ptrs.rank_route_expected,
            output.data_ptr(),
            ptrs.node_partial_ready,
            generation,
            capacity,
            active,
            stream=stream,
        )
        torch.cuda.synchronize(dev)

        if output_dtype == "bf16":
            torch.testing.assert_close(
                output[:active],
                reference[:active].to(torch.bfloat16),
                rtol=0.0,
                atol=0.0,
            )
        else:
            torch.testing.assert_close(
                output[:active], reference[:active], rtol=0.0, atol=1.0e-6
            )
        assert torch.count_nonzero(output[active:]).item() == 0
        assert torch.all(node_ready[:active] == generation)
        assert torch.count_nonzero(node_ready[active:]).item() == 0
        assert reducer.num_ranks == 8
        assert "no-tp-reduction" in reducer.output_contract
        assert reducer.requires_zero_filled_missing_rank_slots is True
