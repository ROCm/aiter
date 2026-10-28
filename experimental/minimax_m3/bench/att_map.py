"""Which CTAs (bids) each ATT run traced: python3 att_map.py DISPATCH LAYER RUN [RUN ..]
(RUN = bench/logs/att/NAME; the hw placement of LAYER from the run's tl_gluon_*_hw.pt)."""
import glob
import json
import os
import re
import sys

import torch

disp, L = sys.argv[1], int(sys.argv[2])
for run in sys.argv[3:]:
    try:
        dd = glob.glob(f"{run}/prof/ui_output_agent_*_dispatch_{disp}")[0]
        hw = torch.load(glob.glob(f"{run}/tl_gluon_*_hw.pt")[0])
    except (IndexError, FileNotFoundError):
        print(f"{os.path.basename(run)}: no trace")
        continue
    h = hw[L, :, 0].int()
    x = hw[L, :, 1].int() & 15
    key2bid = {(int(x[b]) * 4 + int((h[b] >> 13) & 7), int((h[b] >> 8) & 15)): b for b in range(h.shape[0])}
    bids = set()
    for f in os.listdir(dd):
        m = re.match(r"se(\d+)_sm(\d+)_sl(\d+)_wv(\d+)\.json", f)
        if m:
            w = json.load(open(f"{dd}/{f}"))["wave"]
            bids.add(key2bid.get((int(m.group(1)), w["cu"]), -1))
    print(f"{os.path.basename(run)}: {' '.join(str(b) for b in sorted(bids))}")
