# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Small, unbuffered checkpoints for PyTorch/ASan runtime qualification."""

import torch

print(f"PyTorch {torch.__version__}, HIP {torch.version.hip}", flush=True)
count = torch.cuda.device_count()
print(f"Visible GPUs: {count}", flush=True)
assert count == 1
print(torch.cuda.get_device_properties(0), flush=True)
print(f"Free/total GPU memory: {torch.cuda.mem_get_info()}", flush=True)
values = torch.arange(64, device="cuda", dtype=torch.float32)
print("GPU tensor allocated", flush=True)
assert (values + 1).sum().item() == 2080
torch.cuda.synchronize()
print("PyTorch smoke test passed", flush=True)
