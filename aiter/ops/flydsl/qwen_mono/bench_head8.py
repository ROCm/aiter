import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import head8 as H
from head import IMIN
from safetensors import safe_open
torch.manual_seed(0)
E = lambda k, d: int(os.environ.get(k, d))
TW, GW, B = E("TW", 4), E("GW", 2), E("B", 3)
with safe_open("/data/Qwen3.6-35B-A3B-MXFP4-experts-MXFP8-gemm-lmhead/outside.safetensors", "pt") as f:
    w8 = f.get_tensor("lm_head.weight").cuda(); ws = f.get_tensor("lm_head.weight_scale").cuda()
    emb = f.get_tensor("model.language_model.embed_tokens.weight").cuda()
N, K = w8.shape
w8u = w8.view(torch.uint8)
Wf = w8.float() * torch.pow(2.0, ws.float() - 127).repeat_interleave(32, 1)
# x: 16 hidden-like rows (embedding rows + noise), MXFP8 1x32 quantized on the host
x = (emb[torch.randint(0, N, (16,))].float() * 8 + torch.randn(16, K, device="cuda") * 0.05)
am = x.view(16, K // 32, 32).abs().amax(2).clamp_min(1e-30)
code = (torch.floor(torch.log2(am)) - 8 + 127).clamp(1, 254)
xq = (x / torch.pow(2.0, code - 127).repeat_interleave(32, 1)).clamp(-448, 448).to(torch.float8_e4m3fn)
xs = code.to(torch.uint8)
xd = xq.float() * torch.pow(2.0, code - 127).repeat_interleave(32, 1)
ln, G = H.build(N, K, B, TW, GW)
Wp, Sp = H.pack_w(w8u, ws, TW, GW); Xp, XSp = H.pack_x(xq.view(torch.uint8), xs)
out = torch.empty(16, dtype=torch.int64, device="cuda"); st = torch.zeros(256, 4, dtype=torch.int64, device="cuda")
def run():
    out.fill_(IMIN)
    ln(Xp.data_ptr(), XSp.data_ptr(), Wp.data_ptr(), Sp.data_ptr(), out.data_ptr(), st.data_ptr(), stream=torch.cuda.current_stream()); torch.cuda.synchronize()
run()
lg = xd @ Wf.T
ri = lg.to(torch.bfloat16).float().argmax(1); rb = (x.to(torch.bfloat16).float() @ Wf.T).argmax(1)
tok = 0x7FFFFFFF - (out & 0xFFFFFFFF)
ts = []
for _ in range(200):
    run(); ts.append([int(st[:, 1].max() - st[:, 0].min()), float((st[:G["NW"] // 8, 3] - st[:G["NW"] // 8, 2]).float().median()), int((st[:G["NW"] // 8, 3] - st[:G["NW"] // 8, 2]).max())])
t = torch.tensor(ts[20:]).float().median(0).values / 100
print("TW %d GW %d B %d LDS %d KB: tokens vs fp8 reference %d/16 (vs unquantized x %d/16) | in-kernel %.1f us, loop median CTA %.1f, slowest %.1f" % (TW, GW, B, G["LDS"] // 1024, int((tok == ri).sum()), int((tok == rb).sum()), t[0], t[1], t[2]))
