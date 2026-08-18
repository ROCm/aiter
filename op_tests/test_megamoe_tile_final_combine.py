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


@pytest.mark.parametrize(
    "local_dtype,remote_dtype,output_dtype",
    [
        ("bf16", "bf16", "bf16"),
        ("bf16", "fp32", "fp32"),
        ("fp32", "bf16", "bf16"),
        ("fp32", "fp32", "fp32"),
    ],
)
def test_final_combine_two_node_partial_multi_epoch(
    local_dtype: str,
    remote_dtype: str,
    output_dtype: str,
):
    """Combine local/remote EP partials; TP all-reduce remains downstream."""

    _require_gpu_and_flydsl()
    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import compile_final_combine
    from aiter.ops.flydsl.kernels.megamoe_tile.runtime import HierCcoArenaLayout

    dev = torch.device("cuda", 0)
    stream = torch.cuda.current_stream(dev)
    capacity, active, hidden = 11, 7, 3584
    layout = HierCcoArenaLayout.create(
        max_m_tiles=1,
        max_source_tokens=capacity,
    )
    arena = layout.allocate_local(device=dev)
    remote_ready = torch.zeros(
        (2, capacity), dtype=torch.int64, device=dev
    )
    torch_local_dtype = (
        torch.bfloat16 if local_dtype == "bf16" else torch.float32
    )
    torch_remote_dtype = (
        torch.bfloat16 if remote_dtype == "bf16" else torch.float32
    )
    torch_output_dtype = (
        torch.bfloat16 if output_dtype == "bf16" else torch.float32
    )

    combine = compile_final_combine(
        D_HIDDEN=hidden,
        local_dtype=local_dtype,
        remote_dtype=remote_dtype,
        output_dtype=output_dtype,
    )
    assert "no-tp-reduction" in combine.output_contract
    assert combine.requires_tp_all_reduce is True

    # Generation 63 reuses generation 61's parity and proves absolute
    # generations, rather than a one-shot flag, control output publication.
    for generation in (61, 62, 63):
        torch.manual_seed(101 + generation)
        local = (torch.randn(capacity, hidden, device=dev) * 0.1).to(
            torch_local_dtype
        )
        remote = (torch.randn(capacity, hidden, device=dev) * 0.1).to(
            torch_remote_dtype
        )
        local[active:].zero_()
        remote[active:].zero_()
        output = torch.zeros(
            (capacity, hidden), dtype=torch_output_dtype, device=dev
        )

        ptrs = layout.epoch_pointers(arena.data_ptr(), generation)
        local_ready = layout.view(
            arena, "node_partial_ready", parity=ptrs.parity
        )
        final_ready = layout.view(
            arena, "final_output_ready", parity=ptrs.parity
        )
        local_ready.zero_()
        if generation != 63:
            final_ready.zero_()
        remote_ready[ptrs.parity].zero_()
        local_ready[:active].fill_(generation)
        remote_ready[ptrs.parity, :active].fill_(generation)

        combine(
            local.data_ptr(),
            remote.data_ptr(),
            ptrs.node_partial_ready,
            remote_ready[ptrs.parity].data_ptr(),
            output.data_ptr(),
            ptrs.final_output_ready,
            generation,
            active,
            stream=stream,
        )
        torch.cuda.synchronize(dev)

        reference = local.float() + remote.float()
        if output_dtype == "bf16":
            reference = reference.to(torch.bfloat16)
        torch.testing.assert_close(output[:active], reference[:active])
        assert torch.count_nonzero(output[active:]).item() == 0
        assert torch.all(final_ready[:active] == generation)
        assert torch.count_nonzero(final_ready[active:]).item() == 0
