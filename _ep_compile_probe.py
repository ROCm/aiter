"""First full compile (trace) probe for mega_moe_ep.

COMPILE_ONLY via _preload_compiled: traces + codegens the one-launch EP kernel
without dispatching GPU work, so placeholder (null) pointers are safe. This is
bring-up iteration 1 -- the point is to surface device-side trace/codegen errors
from the freshly wired communication + gemm + schedule parts.
"""

import torch

torch.cuda.set_device(0)
from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled  # noqa: E402
from aiter.ops.flydsl.kernels.mega_moe_ep.mega_moe_ep_kernel import (  # noqa: E402
    compile_mega_moe_ep,
)

Z = 0  # placeholder device pointer (null -- never dereferenced under COMPILE_ONLY)
args = [
    Z, Z, Z, Z,              # w1 w1s w2 w2s
    Z, Z, Z, Z, Z,           # x xs ids wts out
    Z,                       # ctrl
    Z, Z, Z, Z, Z, Z, Z, Z,  # p0..p7
    Z, Z, Z, Z, Z, Z,        # off_payload off_pscale off_recv off_part off_flag off_out
    Z, Z, Z, Z, Z, Z, Z, Z,  # inter inters gout rowmap rowidx rowpos ecnt dbg
    0, 8, 48, 8, 32768, 1,   # rank npes epr m mtpr epoch
    304,                     # i32_grid (#CU)
]
stream = torch.cuda.current_stream()
for a8, label in [(False, "a4w4"), (True, "a8w4")]:
    fn = compile_mega_moe_ep(
        H=7168, I=3072, TOPK=6, MT=2, TMAX=262144, E=384, EPR=48, NPES=8,
        act="silu", swiglu_limit=10.0, a8=a8, indexed_payload=True,
    )
    _preload_compiled(fn, *args, stream)
    print(f"COMPILE OK: v4_pro {label} MT=2")
