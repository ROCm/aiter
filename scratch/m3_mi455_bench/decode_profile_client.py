"""Drive a fixed decode batch on the ATOM server and profile a short window of it.

Sends BS concurrent completions with ~PROMPT tokens each and a long max_tokens, waits
until all are decoding, then /start_profile, sleeps WINDOW s, /stop_profile.
"""

import json
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000"
BS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
PROMPT = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
WINDOW = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
PROFILE = (sys.argv[4] if len(sys.argv) > 4 else "1") == "1"
MAXTOK = int(sys.argv[6]) if len(sys.argv) > 6 else 2048


def post(path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def complete(i, out):
    words = " ".join(f"item{i}_{k}" for k in range(PROMPT // 3))
    t0 = time.time()
    r = json.loads(post("/v1/completions", {
        "model": "/models/MiniMax-M3-MXFP4", "prompt": f"Summarize the list. {words}",
        "max_tokens": MAXTOK, "temperature": 0, "ignore_eos": True}))
    out[i] = (time.time() - t0, r["usage"])


WAIT = float(sys.argv[5]) if len(sys.argv) > 5 else 25.0
out = {}
ths = [threading.Thread(target=complete, args=(i, out)) for i in range(BS)]
for t in ths:
    t.start()
time.sleep(WAIT)  # prefill done, decoding the batch
if PROFILE:
    print("start_profile:", post("/start_profile")[:200], flush=True)
    time.sleep(WINDOW)
    print("stop_profile:", post("/stop_profile", timeout=1200)[:400], flush=True)
for t in ths:
    t.join()
for i, (dt, usage) in sorted(out.items()):
    print(f"req {i}: {dt:.1f} s, usage {usage}")
