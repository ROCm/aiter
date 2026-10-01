# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Marlowe AI
"""Additional exact-M8 native route/layout/reset/graph fixtures.

Constant expert rows are invariant under the native weight shuffle. These
fixtures do not replace the captured-weight independent-oracle campaign.
"""

import torch

from aiter import ActivationType, QuantType
from aiter.fused_moe import GateMode, fused_moe
from aiter.ops.flydsl.mxmoe_tiny_m8 import make_operator


def main():
    if torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx950":
        print("SKIP: exact-M8 fixtures require gfx950")
        return
    device = torch.device("cuda:0")
    w1 = torch.full((257, 512, 3072), 0x22, dtype=torch.uint8, device=device).view(
        torch.float4_e2m1fn_x2
    )
    w2 = torch.empty((257, 6144, 128), dtype=torch.uint8, device=device)
    for expert in range(257):
        nibble = (2, 4, 6)[expert % 3]
        w2[expert].fill_(nibble | (nibble << 4))
    w2 = w2.view(torch.float4_e2m1fn_x2)
    w1.is_shuffled = w2.is_shuffled = True
    weights = {
        "w1": w1,
        "w2": w2,
        "w1_scale": torch.full(
            (257, 512, 192), 120, dtype=torch.uint8, device=device
        ).view(torch.float8_e8m0fnu),
        "w2_scale": torch.full(
            (257, 6144, 8), 120, dtype=torch.uint8, device=device
        ).view(torch.float8_e8m0fnu),
    }
    immutable = {key: value.view(torch.uint8).clone() for key, value in weights.items()}
    operators = [make_operator(weights=weights, rows=8) for _ in range(2)]
    assert all(operator.enabled for operator in operators)
    x = torch.full((8, 6144), 1 / 128, dtype=torch.bfloat16, device=device)
    ids = torch.cat(
        (
            torch.arange(8, dtype=torch.int32, device=device).expand(8, 8),
            torch.full((8, 1), 256, dtype=torch.int32, device=device),
        ),
        1,
    ).contiguous()
    rw = torch.arange(1, 73, dtype=torch.float32, device=device).reshape(8, 9) / 256
    # Deliberately not one: the real supplied shared weight must be preserved.
    rw[:, 8] = 0.5

    def native():
        return fused_moe(
            x,
            **weights,
            topk_weight=rw,
            topk_ids=ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            gate_mode=GateMode.SEPARATED.value,
        )

    def check(operator):
        expected = native().clone()
        actual = operator.run(x, ids, rw)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        if not bool(x.count_nonzero()):
            torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)
        assert operator.state_check()["passed"]

    for operator in operators:
        check(operator)
    ids[:, :8].copy_(torch.arange(64, dtype=torch.int32, device=device).reshape(8, 8))
    for operator in operators:
        check(operator)
    # Expert255 at rawslot63 checks the sign bit of the one-wave ballot.
    ids[7, 7] = 255
    check(operators[0])
    x.zero_()
    for operator in operators:
        check(operator)
    x.fill_(1 / 128)
    torch.cuda.synchronize()
    initial_ids = ids.clone()
    for operator in operators:
        check(operator)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = operator.run(x, ids, rw)
        for index, value in enumerate((0.0, 1 / 128, 1 / 64, 0.0, 1 / 128)):
            x.fill_(value)
            ids[:, :8].copy_((initial_ids[:, :8] + index * 16) % 256)
            rw[:, 8] = (0.5, 1.0)[index % 2]
            graph.replay()
            torch.testing.assert_close(result, native(), atol=0.02, rtol=0.02)
        assert operator.state_check()["passed"]
    for key, before in immutable.items():
        torch.testing.assert_close(
            weights[key].view(torch.uint8), before, atol=0, rtol=0
        )
    print("PASS: native M8 route/reset/shared-weight/two-workspace graph fixtures")


if __name__ == "__main__":
    main()
