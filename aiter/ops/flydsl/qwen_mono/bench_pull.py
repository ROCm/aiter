import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import pull as PL
dev = "cuda"
src = torch.randint(0, 255, (400 << 20,), dtype=torch.uint8, device=dev)      # 400 MB
st = torch.zeros(2048, 4, dtype=torch.int64, device=dev)
for mode in [int(v) for v in os.environ.get('MODES', '5,6,7,8,9').split(',')]:
    ln = PL.build(mode)
    for na in (1, 256):
        for stride in (181 * 1024,):          # contiguous units / units scattered over 380 MB
            res = []
            for it in range(40):
                st.zero_(); ln(src.data_ptr(), st.data_ptr(), na, stride, stream=torch.cuda.current_stream()); torch.cuda.synchronize()
                d = (st[:na * 8, 1] - st[:na * 8, 0]).double() / 100
                res.append((float(d.median()), float(d.max()), float((st[:na * 8, 1].max() - st[:na * 8, 0].min())) / 100))
            r = torch.tensor(res[8:]).median(0).values.tolist()
            print("mode %d  CTAs %3d  stride %3d KB: 64 KB a wave in med %.1f max %.1f us (%.1f KB/us a wave), all waves %.1f us = %.1f MB/us" % (mode, na, stride // 1024, r[0], r[1], 64 / r[0], r[2], na * 8 * 64 / 1024 / r[2]))
