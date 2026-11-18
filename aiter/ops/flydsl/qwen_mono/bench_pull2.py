import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import pull as PL
dev = "cuda"
src = torch.randint(0, 255, (400 << 20,), dtype=torch.uint8, device=dev)
st = torch.zeros(2048, 4, dtype=torch.int64, device=dev)
for cfg in os.environ.get("CFGS", "8,3,0,0 8,3,0,1 8,3,1,1 8,4,0,1 8,4,1,1 16,2,0,1 16,2,1,1 4,3,0,1 4,8,0,1 4,8,1,1 2,16,0,1 32,1,1,1 8,3,0,1,1").split():
    v = [int(x) for x in cfg.split(",")]
    ln = PL.build3(*v[1:]) if v[0] == 0 else PL.build4(*v[1:]) if v[0] == -1 else PL.build5(*v[1:]) if v[0] == -2 else PL.build2(*v); out = []
    for na in (1, 256):
        res = []
        for it in range(40):
            st.zero_(); ln(src.data_ptr(), st.data_ptr(), na, 181 * 1024, stream=torch.cuda.current_stream()); torch.cuda.synchronize()
            d = (st[:na * 8, 1] - st[:na * 8, 0]).double() / 100
            res.append((float(d.median()), float(d.max()), float((st[:na * 8, 1].max() - st[:na * 8, 0].min())) / 100))
        r = torch.tensor(res[8:]).median(0).values.tolist(); out.append("%d CTAs: wave med %.1f max %.1f, all %.1f us" % (na, r[0], r[1], r[2]))
    print("cfg %-16s | %s" % (cfg, " | ".join(out)))
