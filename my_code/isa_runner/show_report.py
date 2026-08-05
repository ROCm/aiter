#!/usr/bin/env python3
"""Print an adapter report, skipping the verbose capture block."""
import json
import sys

d = json.load(open(sys.argv[1]))
for k, v in d.items():
    if k == "capture":
        c = v
        print("capture.kernel =", c.get("kernel"))
        print("capture.grid   =", c.get("grid"), "block =", c.get("block"))
        print("capture.lds    =", c.get("lds_bytes"), "tiles =", c.get("tiles"))
        print("capture.i32_m  =", c.get("args", {}).get("i32_m"),
              " i32_n =", c.get("args", {}).get("i32_n"))
    else:
        print(f"{k} = {v}")
