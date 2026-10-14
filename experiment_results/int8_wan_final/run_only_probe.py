import os, torch, aiter
from aiter.ops.shuffle import shuffle_weight
assert os.environ.get("FLYDSL_RUNTIME_RUN_ONLY") == "1"
print("AITER", aiter.__file__, "CACHE", os.environ["FLYDSL_RUNTIME_CACHE_DIR"])
def call(m, n, k):
    x = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
    w = shuffle_weight(torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda"), layout=(16, 16))
    xs = torch.rand(m, 1, device="cuda"); ws = torch.rand(1, n, device="cuda")
    y = aiter.gemm_a8w8_bpreshuffle(x, w, xs, ws, None, torch.bfloat16)
    torch.cuda.synchronize()
    return y
for shape in [(9450,5120,5120),(9450,13824,5120),(9450,5120,13824)]:
    call(*shape); print("RUN_ONLY_OK", shape, flush=True)
# negative control: K not AOT-compiled must fail loudly (proves run-only is active)
try:
    call(256, 256, 384); print("CONTROL_UNEXPECTED_SUCCESS")
except Exception as e:
    print("CONTROL_EXPECTED_FAILURE", type(e).__name__, str(e)[:160].replace("\n"," "))
