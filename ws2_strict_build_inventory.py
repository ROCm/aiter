# SPDX-License-Identifier: MIT
# WS2 driver: build every aiter JIT module with GPU_ARCHS=gfx1250-strict and
# record per-module pass/fail. Run inside a strict-capable ROCm container:
#   GPU_ARCHS=gfx1250-strict python3 ws2_strict_build_inventory.py [--only MOD] [--update-md]
#
# Results land in ws2_build_results.json; --update-md folds them into
# docs/gfx1250-strict-kernel-inventory.md (strict build column).

import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(REPO, "ws2_build_results.json")
MD = os.path.join(REPO, "docs/gfx1250-strict-kernel-inventory.md")

CHILD = r"""
import json, os, sys, traceback
sys.path.insert(0, {repo!r})
os.chdir({repo!r})
from aiter.jit import core
args_list, _ = core.get_args_of_build("all", exclude=[])
name = sys.argv[1]
for a in args_list:
    if a["md_name"] == name:
        try:
            core.build_module(
                md_name=a["md_name"], srcs=a["srcs"],
                flags_extra_cc=a["flags_extra_cc"],
                flags_extra_hip=a["flags_extra_hip"],
                blob_gen_cmd=a["blob_gen_cmd"],
                extra_include=a["extra_include"],
                extra_ldflags=a.get("extra_ldflags"),
                verbose=False, is_python_module=True, is_standalone=False,
                torch_exclude=a.get("torch_exclude", False),
                third_party=a["third_party"],
                flags_extra_hip_per_source=a.get("flags_extra_hip_per_source"),
            )
            print("BUILD_OK")
        except Exception:
            traceback.print_exc()
            print("BUILD_FAIL")
        break
else:
    print("NO_SUCH_MODULE")
"""


def list_modules():
    with open(os.path.join(REPO, "aiter/jit/optCompilerConfig.json")) as f:
        return list(json.load(f).keys())


_SIGNAL_CHILD = r"""
import json, os, re, sys
sys.path.insert(0, {repo!r})
os.chdir({repo!r})
from aiter.jit import core

CSRC = os.path.join({repo!r}, "csrc")
resolved = {{}}
args_list, _ = core.get_args_of_build("all", exclude=[])
for a in args_list:
    srcs = a["srcs"]
    resolved[a["md_name"]] = srcs if isinstance(srcs, list) else [srcs]

with open(os.path.join({repo!r}, "aiter/jit/optCompilerConfig.json")) as f:
    cfg = json.load(f)

# Modules get_args_of_build skips (tune, experimental) still need a signal:
# recover their sources from the raw config expressions instead.
index = {{}}
for root, _dirs, files in os.walk(CSRC):
    for fn in files:
        index.setdefault(fn, []).append(os.path.join(root, fn))
for name, entry in cfg.items():
    if name in resolved:
        continue
    paths = []
    for expr in entry.get("srcs") or []:
        for frag in re.findall(r"[\w/\.\-]+\.(?:cu|cpp|cc|hip)", str(expr)):
            paths.extend(index.get(os.path.basename(frag), [])[:4])
    resolved[name] = paths

cache = {{}}
def text(path):
    if path not in cache:
        try:
            with open(path, errors="ignore") as f:
                cache[path] = f.read()
        except OSError:
            cache[path] = ""
    return cache[path]

out = {{}}
for name in cfg:
    tags = []
    if "gfx1250" in name:
        tags.append("name")
    bodies = [text(p) for p in resolved.get(name, [])]
    if any('"gfx1250"' in b for b in bodies):
        tags.append("str")
    if any("__gfx1250__" in b for b in bodies):
        tags.append("macro")
    out[name] = "+".join(tags) if tags else "-"
print("SIGNALS " + json.dumps(out))
"""


def refresh_signals():
    """Recompute the `gfx1250 signal` column from the resolved sources."""
    p = subprocess.run(
        [sys.executable, "-c", _SIGNAL_CHILD.format(repo=REPO)],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    line = next(
        (ln for ln in (p.stdout + p.stderr).splitlines() if ln.startswith("SIGNALS ")),
        None,
    )
    if line is None:
        print((p.stdout + p.stderr)[-2000:])
        raise SystemExit("could not compute signals")
    signals = json.loads(line[len("SIGNALS ") :])
    with open(MD, encoding="utf-8") as f:
        lines = f.read().splitlines()
    out = []
    changed = 0
    for ln in lines:
        if ln.startswith(("| module_", "| lib")):
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if len(cells) >= 7 and cells[0] in signals:
                if cells[2] != signals[cells[0]]:
                    changed += 1
                cells[2] = signals[cells[0]]
                ln = "| " + " | ".join(cells) + " |"
        out.append(ln)
    with open(MD, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"refreshed {changed} signal cells in {MD}")


def skip_reason(mod, cfg):
    """Modules get_args_of_build("all") never yields, with the reason.

    They are not build failures: *_tune is skipped by core.get_args_of_build,
    and experimental modules are gated behind AITER_ENABLE_EXPERIMENTAL.
    Recording them as "missing" would hide the distinction the inventory needs.
    """
    if mod.endswith("tune"):
        return "skip-tune"
    if cfg.get(mod, {}).get("is_experimental") and not os.environ.get(
        "AITER_ENABLE_EXPERIMENTAL"
    ):
        return "skip-experimental"
    return None


def run_one(mod):
    env = dict(os.environ)
    env.setdefault("GPU_ARCHS", "gfx1250-strict")
    p = subprocess.run(
        [sys.executable, "-c", CHILD.format(repo=REPO), mod],
        capture_output=True,
        text=True,
        env=env,
        timeout=3600,
        check=False,
    )
    out = p.stdout + p.stderr
    if "BUILD_OK" in out:
        return "pass", ""
    tail = "\n".join(out.strip().splitlines()[-15:])
    if "NO_SUCH_MODULE" in out:
        return "missing", tail
    return "fail", tail


def main():
    if "--refresh-signals" in sys.argv:
        refresh_signals()
        return
    only = None
    update_md = "--update-md" in sys.argv
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]

    results = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            results = json.load(f)

    with open(os.path.join(REPO, "aiter/jit/optCompilerConfig.json")) as f:
        cfg = json.load(f)
    mods = [only] if only else list_modules()
    for i, mod in enumerate(mods, 1):
        print(f"[{i}/{len(mods)}] {mod} ...", flush=True)
        reason = skip_reason(mod, cfg)
        if reason:
            status, tail = reason, ""
        else:
            status, tail = run_one(mod)
        results[mod] = {"status": status, "tail": tail}
        with open(RESULTS, "w") as f:
            json.dump(results, f, indent=1)
        print(f"  -> {status}", flush=True)

    if update_md:
        fold_into_md(results)


def fold_into_md(results):
    with open(MD) as f:
        lines = f.read().splitlines()
    out = []
    for ln in lines:
        if ln.startswith(("| module_", "| lib")):
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if len(cells) >= 7 and cells[0] in results:
                cells[3] = results[cells[0]]["status"]
                ln = "| " + " | ".join(cells) + " |"
        out.append(ln)
    with open(MD, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"updated {MD}")


if __name__ == "__main__":
    main()
