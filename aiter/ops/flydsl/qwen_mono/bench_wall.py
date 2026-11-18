"""Whole-launch cost: N launches back to back on one stream, one synchronize at the end (dispatch and teardown included),
next to the in-kernel stamp time."""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(__file__))
import draft_mono as DM
from safetensors import safe_open
dev = "cuda"; D = "/data/Qwen3.6-35B-A3B-MXFP4/"; N = 3000
with safe_open(D + "mtp.safetensors", "pt") as f:
    g = lambda k: f.get_tensor("mtp." + k).to(dev); L = "layers.0."
    pre = None
    Wfc = g("fc.weight"); Wqkv = torch.cat([g(L + "self_attn.%s_proj.weight" % n) for n in "qkv"])
    we, wh, win = g("pre_fc_norm_embedding.weight"), g("pre_fc_norm_hidden.weight"), g(L + "input_layernorm.weight")
with safe_open(D + "outside.safetensors", "pt") as f:
    emb = f.get_tensor("model.language_model.embed_tokens.weight").to(dev)
pre = DM.DraftPre(emb, we, wh, Wfc, win, Wqkv, dev)
post = DM.DraftMono(D, "/data/Qwen3.6-35B-A3B-MXFP4-experts-MXFP8-gemm-lmhead", dev)
T = 3
ids = torch.randint(0, 248320, (T,), device=dev); hid = (emb[ids].float() * 20).to(torch.bfloat16)
x0 = (torch.randn(T, 4096, device=dev) * 0.5).to(torch.bfloat16)
def wall(fn, n=N):
    for _ in range(50): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e6
def stamp(obj, k):
    return float((obj.st[:, k].max() - obj.st[:, 0].min())) / 100
z = torch.zeros(4096, dtype=torch.int32, device=dev)
print("host loop of zero_() only: %.1f us a call" % wall(lambda: z.zero_()))
w = wall(lambda: pre.run(ids, hid)); print("pre : back-to-back %.1f us a launch (with its cnt.zero_()), in-kernel %.1f us" % (w, stamp(pre, 3)))
w = wall(lambda: post.run(x0, hid)); print("post: back-to-back %.1f us a launch (with its cnt.zero_()), in-kernel %.1f us" % (w, stamp(post, 8)))
