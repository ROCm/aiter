import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import stream as S
nb = 248320 * 2048 * 2
W = torch.randint(0, 2**31, (nb // 4,), dtype=torch.int32, device="cuda")
out = torch.zeros(256 * 256, dtype=torch.int32, device="cuda"); st = torch.zeros(256, 2, dtype=torch.int64, device="cuda")
for cfg in sys.argv[1:]:
    CH, D, nt, mode, waves = cfg.split(","); 
    ln, got = S.build(nb, int(CH), int(D), nt == "1", mode, int(waves))
    ts = []
    for _ in range(120):
        ln(W.data_ptr(), out.data_ptr(), st.data_ptr(), stream=torch.cuda.current_stream()); torch.cuda.synchronize()
        ts.append(int(st[:, 1].max() - st[:, 0].min()))
    ts = torch.tensor(ts[20:]).float() / 100
    print("%-16s %.3f GB  median %.1f us  min %.1f  -> %.1f GB/ms" % (cfg, got / 1e9, ts.median(), ts.min(), got / 1e9 / ts.median() * 1e3))
