"""GPU time per call of the ATOM-facing kernel (cnt.zero_() + launch + the output allocation) inside a captured graph."""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(__file__))
import vmoe_atom as VA
from safetensors import safe_open
torch.manual_seed(0); dev = "cuda"; D = os.environ.get("CKPT", "/data/Qwen3.6-35B-A3B-MXFP4/"); LY = int(os.environ.get("LAYER", 17))
vm = VA.VMoE(D, torch.device(dev)); vm.load_layer(LY)
with safe_open(D + "outside.safetensors", "pt") as f:
    emb = f.get_tensor("model.language_model.embed_tokens.weight").to(dev)
K = 40
for R in [int(v) for v in os.environ.get("ROWS", "4,8,12,16").split(",")]:
    h = emb[torch.randint(0, 248320, (R,))].float() * 20 + torch.randn(R, 2048, device=dev) * 0.2
    x = (h * torch.rsqrt(h.pow(2).mean(1, keepdim=True) + 1e-6)).to(torch.bfloat16).contiguous()
    vm.run(LY, x); torch.cuda.synchronize()
    def cap(fn):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(K): fn()
        for _ in range(5): g.replay()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(50): g.replay()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / (50 * K) * 1e6
    full = cap(lambda: vm.run(LY, x)); zero = cap(lambda: vm.cnt.zero_())
    print("rows %2d: %.1f us a call in a graph (kernel %.1f + counter zero %.1f)" % (R, full, full - zero, zero))
