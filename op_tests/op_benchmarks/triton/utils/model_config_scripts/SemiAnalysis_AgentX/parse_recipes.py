import copy
import glob
import json
import sys

import yaml

ROOT = sys.argv[1]


def merge(a, b):
    a = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            a[k] = merge(a[k], v)
        else:
            a[k] = v
    return a


def roles_args(cfg):
    out = []
    for rn, r in (cfg.get("roles") or {}).items():
        out.append((rn, r.get("args") or {}, r.get("env") or {}, r.get("gpus")))
    return out


def verify_tokens(a):
    # returns tokens verified per request per decode step (1 + drafts)
    if "speculative-dspark-block-size" in a and "speculative-num-draft-tokens" not in a:
        return 1 + int(a["speculative-dspark-block-size"])
    if "speculative-num-draft-tokens" in a:
        return int(a["speculative-num-draft-tokens"])  # sglang: includes current token
    if "num-speculative-tokens" in a:
        return 1 + int(a["num-speculative-tokens"])
    sc = a.get("speculative-config")
    if sc:
        try:
            return 1 + int(json.loads(sc)["num_speculative_tokens"])
        except (ValueError, KeyError, TypeError):
            pass
    return 1


def ival(a, *ks):
    for k in ks:
        if k in a and a[k] not in (None, ""):
            try:
                return int(a[k])
            except ValueError:
                pass
    return None


rows = []
for f in sorted(glob.glob(ROOT + "/*/*/mi3*/agentic.yaml")):
    with open(f) as fh:
        d = yaml.safe_load(fh)
    base = d.get("base", {})
    model, eng, hw = f.split("/")[-4:-1]
    for k, v in d.items():
        if not k.startswith("override_"):
            continue
        cfg = merge(base, v)
        conc = int(((cfg.get("benchmark") or {}).get("env") or {}).get("CONC", 0) or 0)
        for rn, a, env, gpus in roles_args(cfg):
            tp = ival(a, "tensor-parallel-size", "tp-size", "tp")
            if tp is None:
                tp = int(gpus or 1)
            dp = ival(a, "data-parallel-size", "dp-size", "dp") or 1
            ep = ival(a, "expert-parallel-size", "ep-size", "ep")
            if ep is None:
                ep = tp * dp if a.get("enable-expert-parallel") in (True, "true") else 1
            chunk = ival(
                a,
                "chunked-prefill-size",
                "max-num-batched-tokens",
                "max-prefill-tokens",
                "attn-prefill-chunk-size",
            )
            vt = verify_tokens(a)
            rows.append(
                {
                    "model": model,
                    "engine": eng,
                    "hw": hw,
                    "variant": k,
                    "role": rn,
                    "tp": tp,
                    "dp": dp,
                    "ep": ep,
                    "dpattn": bool(a.get("enable-dp-attention")),
                    "dcp": ival(a, "decode-context-parallel-size"),
                    "gpus": int(gpus or tp * dp),
                    "conc": conc,
                    "verify": vt,
                    "chunk": chunk,
                }
            )
with open(sys.argv[2], "w") as fh:
    json.dump(rows, fh, indent=1)
