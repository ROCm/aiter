import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import pull as PL
dev = "cuda"; torch.manual_seed(0)
src = torch.randint(0, 255, (400 << 20,), dtype=torch.uint8, device=dev); w32 = src.view(torch.int32)
st = torch.zeros(2048, 4, dtype=torch.int64, device=dev); stride = 181 * 1024
idx = (torch.arange(2048, device=dev)[:, None, None] * stride + torch.arange(128, device=dev)[None, :, None] * 512 + torch.arange(4, device=dev)[None, None, :] * 16) // 4
exp = w32[idx].sum(1, dtype=torch.int32)
for tdm in (0, 1024, 8192, 512):
    ln = PL.build2(8, 3, 0, 0, 0, 1, 0, tdm, 1); st.zero_()
    ln(src.data_ptr(), st.data_ptr(), 256, stride, stream=torch.cuda.current_stream()); torch.cuda.synchronize()
    got = st[:, 2:4].contiguous().view(torch.int32).view(2048, 4)
    if tdm: print("   wave 5 matches expected of waves", (exp == got[5]).all(1).nonzero().flatten().tolist()[:5], "| waves equal to own exp per lane", (got == exp).sum(0).tolist(), "| distinct got rows", len(set(map(tuple, got.tolist()))))
    if tdm: print("   got", got[5].tolist(), "exp", exp[5].tolist(), "nonzero waves", int((got != 0).any(1).sum()))
    print("TDM row bytes %5d: waves with the right checksum %d / 2048" % (tdm, int((got == exp).all(1).sum())))
