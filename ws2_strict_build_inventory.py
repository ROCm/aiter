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
