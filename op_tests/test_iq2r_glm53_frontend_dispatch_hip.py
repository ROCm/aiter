# SPDX-License-Identifier: MIT
"""Static frontend flags must change the actual launched GPU kernel."""

import pytest
import torch
from aiter.ops.iq2r import iq2r_route_direct_gather_quant_out

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="requires gfx950",
)


@pytest.mark.parametrize(
    "tokens,enabled,static",
    [
        (4, False, False),
        (4, True, True),
        (8, True, True),
        (8, False, False),
        (16, True, False),
    ],
)
def test_static_frontend_dispatch(tokens, enabled, static, monkeypatch):
    monkeypatch.setenv("IQ2R_GLM53_BALLOT_ROUTER", "1")
    monkeypatch.setenv("IQ2R_GLM53_STATIC_BALLOT", "1" if enabled else "0")
    hidden = torch.randn((tokens, 6144), device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(tokens * 9, device="cuda", dtype=torch.int32) % 257
    buffers = [torch.empty_like(ids) for _ in range(3)] + [
        torch.empty((tokens * 9, 3), device="cuda", dtype=torch.int32),
        torch.empty(1, device="cuda", dtype=torch.int32),
        torch.empty((tokens * 9, 6144), device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty((tokens * 9, 192), device="cuda", dtype=torch.uint8),
    ]

    def run():
        iq2r_route_direct_gather_quant_out(
            hidden,
            ids,
            *buffers,
            topk=9,
            expert_count=257,
            expert_map=None,
            expert_start=0,
            expert_stride=1,
            drop_nonlocal_routes=False,
        )

    run()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        run()
        torch.cuda.synchronize()
    kernels = {
        event.name for event in profile.events() if "quant_glm_kernel" in event.name
    }
    selected = [
        name
        for name in kernels
        if "iq2r_route_grouped_static_ballot_quant_glm_kernel" in name
    ]
    generic = [
        name for name in kernels if "iq2r_route_grouped_ballot_quant_glm_kernel" in name
    ]
    assert bool(selected) == static, (tokens, enabled, static, sorted(kernels))
    assert bool(generic) == (not static), (tokens, enabled, static, sorted(kernels))
