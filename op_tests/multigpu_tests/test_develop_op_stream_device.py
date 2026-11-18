# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""develop=True ops launch on the current stream of their tensors' device.

The pybind bridge (aiter/jit/core.py) used to hand every develop=True op the
current device's stream, while the C++ side guards to the input's device. With
the input on cuda:1 and a side stream current on cuda:0, the launch got cuda:0's
stream on cuda:1 and failed with "invalid resource handle". silu_and_mul stands
for all of them: the stream is chosen in the bridge, not per op.
"""

import sys

import torch
import torch.nn.functional as F

import aiter


def check(call_kwargs: bool) -> bool:
    side0 = torch.cuda.Stream(device=0)
    side1 = torch.cuda.Stream(device=1)
    torch.cuda.set_stream(side1)
    torch.cuda.set_device(0)
    torch.cuda.set_stream(side0)
    assert torch.cuda.current_device() == 0
    assert torch.cuda.current_stream(0) == side0
    assert torch.cuda.current_stream(1) == side1

    src = torch.randn(64, 2 * 4096, dtype=torch.bfloat16, device="cuda:1")
    x = torch.zeros_like(src)
    out = torch.empty(64, 4096, dtype=torch.bfloat16, device="cuda:1")
    torch.cuda.synchronize(1)
    with torch.cuda.stream(side1):
        # Only a launch on side1 is ordered after this copy.
        torch.cuda._sleep(50_000_000)
        x.copy_(src)
    if call_kwargs:
        aiter.silu_and_mul(out=out, input=x)
    else:
        aiter.silu_and_mul(out, x)
    side1.synchronize()

    d = src.shape[-1] // 2
    ref = F.silu(src[:, :d].float()) * src[:, d:].float()
    ok = torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2)
    err = (out.float() - ref).abs().max().item()
    print(
        f"silu_and_mul kwargs={call_kwargs}: max abs err {err:.4f}",
        "ok" if ok else "FAIL",
    )
    return ok


def main() -> int:
    if torch.cuda.device_count() < 2:
        print("skipped: needs 2 GPUs")
        return 0
    results = [check(call_kwargs) for call_kwargs in (False, True)]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
