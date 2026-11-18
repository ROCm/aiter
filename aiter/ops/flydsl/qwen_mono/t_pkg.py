"""The packaged kernels (atom.models.qwen3_5_draft_mono) against the standalone ones on the same inputs."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import draft_mono as A
from atom.models.qwen3_5_draft_mono import host as B
from safetensors import safe_open
torch.manual_seed(0); dev = "cuda"; D = "/data/Qwen3.6-35B-A3B-MXFP4"; H = "/data/Qwen3.6-35B-A3B-MXFP4-experts-MXFP8-gemm-lmhead"
with safe_open(D + "/outside.safetensors", "pt") as f:
    emb = f.get_tensor("model.language_model.embed_tokens.weight").to(dev)
with safe_open(D + "/mtp.safetensors", "pt") as f:
    g = lambda k: f.get_tensor("mtp." + k).to(dev); L = "layers.0."
    args = (emb, g("pre_fc_norm_embedding.weight"), g("pre_fc_norm_hidden.weight"), g("fc.weight"), g(L + "input_layernorm.weight"),
            torch.cat([g(L + "self_attn.%s_proj.weight" % n) for n in "qkv"]))
for R in (3, 8, 16):
    x0 = (torch.randn(R, 4096, device=dev) * 0.5).to(torch.bfloat16); res = (emb[torch.randint(0, 248320, (4 * R,))].float() * 20).to(torch.bfloat16)
    rows = (torch.arange(R, device=dev) * 4 + 3).contiguous(); ids = torch.randint(0, 248320, (R,), device=dev, dtype=torch.int32)
    out = []
    for M in (A, B):
        if R == 3: M._post, M._pre = M.DraftMono(D, H, dev), M.DraftPre(*args, dev)
        h, i = M._post.run(x0, res, rows); r, q = M._pre.run(ids, res[:R].contiguous()); torch.cuda.synchronize()
        out.append((h.clone(), i.clone(), r.clone(), q.clone()))
    print("R %d: post hidden equal %s, ids equal %s %s | pre residual equal %s, qkv equal %s" % (R, bool((out[0][0] == out[1][0]).all()), bool((out[0][1] == out[1][1]).all()),
          out[1][1][:4].tolist(), bool((out[0][2] == out[1][2]).all()), bool((out[0][3] == out[1][3]).all())))
