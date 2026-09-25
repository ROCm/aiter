"""Exact byte repacking of GLM TP8/TP4 gate records; no requantization."""

import torch


def repack_gate_quad(data: torch.Tensor, *, verify: bool = False) -> torch.Tensor:
    if (
        data.dtype != torch.uint8
        or data.ndim != 2
        or data.shape[1] not in (6 * 48 * 3584, 11 * 48 * 3584)
    ):
        raise ValueError("expected canonical GLM TP8/TP4 gate bytes")
    experts = data.shape[0]
    blocks = 32 if data.shape[1] == 6 * 48 * 3584 else 64
    groups = (blocks + 5) // 6
    source = data.reshape(experts, groups, 48, 3584)
    target = torch.empty(
        (experts, blocks // 4, 48, 2304), dtype=torch.uint8, device=data.device
    )
    for block in range(blocks):
        atom = block % 3
        offset = (block % 6) // 3 * 1792
        third = source[:, block // 6, :, offset + 1024 : offset + 1792].reshape(
            experts, 48, 64, 12
        )
        if atom < 2:
            pair = source[:, block // 6, :, offset : offset + 1024].reshape(
                experts, 48, 64, 16
            )
            record = pair[..., atom * 8 : atom * 8 + 8]
        else:
            record = third[..., :8]
        slot = block % 4
        pair_base = (slot // 2) * 1024
        pair_out = target[:, block // 4, :, pair_base : pair_base + 1024].view(
            experts, 48, 64, 16
        )
        pair_out[..., slot % 2 * 8 : slot % 2 * 8 + 8].copy_(record)
        metadata = target[:, block // 4, :, 2048:2304].view(experts, 48, 64, 4)
        metadata[..., slot].copy_(third[..., 8 + atom])
        if verify:
            assert torch.equal(pair_out[..., slot % 2 * 8 : slot % 2 * 8 + 8], record)
            assert torch.equal(metadata[..., slot], third[..., 8 + atom])
    return target.flatten(1).contiguous()


if __name__ == "__main__":
    torch.manual_seed(69)
    source = torch.randint(0, 256, (2, 6 * 48 * 3584), dtype=torch.uint8)
    target = repack_gate_quad(source, verify=True)
    assert target.shape == (2, 8 * 48 * 2304)
    print(
        {"status": "pass", "source_bytes": source.numel(), "quad_bytes": target.numel()}
    )
