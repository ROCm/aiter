import time

import torch

y = torch.empty(128 * 1536 * 512, dtype=torch.bfloat16, device="cuda")
t0 = time.time()
n = 0
while time.time() - t0 < 40:
    for _ in range(100):
        y.zero_()
    torch.cuda.synchronize()
    n += 100
print(
    "RESULT sustained us_per_zero_201MB",
    round((time.time() - t0) * 1e6 / n, 1),
    flush=True,
)
