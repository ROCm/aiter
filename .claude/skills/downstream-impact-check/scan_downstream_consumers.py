#!/usr/bin/env python3
"""Is the code this PR changes reachable from a downstream consumer?

aiter's downstream tests are label-gated and skipped by default, so a PR that changes something
a consumer depends on can pass every aiter check, merge green, and surface as a downstream
incident. That judgement is normally made from memory. It is greppable.

The incident this is built from:
  aiter#4530 changed `aiter/ops/triton/_triton_kernels/moe/moe_routing/topk.py` -- kernel
  internals, no public signature touched. It broke vLLM's gpt-oss MoE routing on MI355 badly
  enough to need a hotfix (vllm#50859), and SGLang reverted its aiter pin the same week
  (sglang#32879, v0.1.19 -> 9127c94).

So the question is NOT "did a public symbol change". An earlier version of this script asked
exactly that and reported "no downstream impact" for that very PR. The question is whether a
module the downstream imports can reach the changed file, so this walks aiter's own imports
upward from the changed files and tests each module in the closure against a downstream
checkout.

Reports the chain, not a verdict -- the reviewer decides whether the label is required.

usage: scan_downstream_consumers.py --diff <diff> [--aiter <root>] [--root <downstream> ...]
"""
import os
import re
import subprocess
import sys

DEFAULT_DOWNSTREAM = ["/sgl-workspace/sglang"]
MAX_DEPTH = 5


def sh(args, timeout=120):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def changed_files(diff_path):
    return [l[6:].strip() for l in open(diff_path)
            if l.startswith("+++ b/") and l[6:].strip().endswith(".py")]


def mod_of(rel):
    return rel[:-3].replace("/", ".")


def _pattern(mod, downstream):
    leaf = mod.rsplit(".", 1)[-1]
    base = rf"(from\s+{re.escape(mod)}\s+import|import\s+{re.escape(mod)}\b"
    if downstream:
        # downstream must be importing it *from aiter*, not defining a same-named local symbol
        return base + rf"|from\s+aiter[\w.]*\s+import\s+[^#\n]*\b{re.escape(leaf)}\b)"
    return base + rf"|from\s+[\w.]*\s+import\s+[^#\n]*\b{re.escape(leaf)}\b)"


def importers(mod, aiter_root):
    out = sh(["grep", "-rlE", _pattern(mod, False), "--include=*.py",
              os.path.join(aiter_root, "aiter")])
    return {os.path.relpath(f, aiter_root) for f in out.split()}


def downstream_hits(mod, roots):
    hits = []
    for r in roots:
        for f in sh(["grep", "-rlE", _pattern(mod, True), "--include=*.py", r]).split():
            if "/.git/" not in f:
                hits.append(f)
    return hits


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] != "--diff":
        print(__doc__)
        return 2
    diff = argv[1]

    def opt(name, default):
        return [argv[i + 1] for i, a in enumerate(argv) if a == name] or default

    aiter_root = opt("--aiter", [os.environ.get("AITER_ROOT", ".")])[0]
    roots = [r for r in opt("--root", DEFAULT_DOWNSTREAM) if os.path.isdir(r)]

    changed = [f for f in changed_files(diff) if f.startswith("aiter/")]
    if not changed:
        print("  diff touches no aiter/*.py -- nothing to trace")
        return 0
    if not roots:
        print("  NO DOWNSTREAM CHECKOUT AVAILABLE -- reachability is unmeasured, not absent.")
        return 2

    print(f"  downstream: {', '.join(roots)}")
    seen = set()
    frontier = {(mod_of(f), (f,)) for f in changed}
    chains = []
    for _ in range(MAX_DEPTH):
        nxt = set()
        for mod, chain in frontier:
            if mod in seen:
                continue
            seen.add(mod)
            hits = downstream_hits(mod, roots)
            if hits:
                chains.append((mod, chain, hits))
                continue                      # boundary found; no need to climb further
            for rel in importers(mod, aiter_root):
                if mod_of(rel) not in seen:
                    nxt.add((mod_of(rel), chain + (rel,)))
        frontier = nxt
        if not frontier:
            break

    if not chains:
        print(f"  {len(changed)} changed aiter file(s); nothing within {MAX_DEPTH} import hops "
              f"is imported by the downstream checkout.")
        print("  That reads as no reachability -- but only for the checkouts above, and only "
              "through static imports. Dispatch by string or env var will not show up here.")
        return 0

    for mod, chain, hits in chains:
        where = sorted({os.path.basename(h) for h in hits})[:4]
        print(f"  REACHED  {mod}")
        print(f"           via: {' <- '.join(chain)}")
        print(f"           {len(hits)} downstream file(s): {', '.join(where)}")
    print(f"\n  {len(chains)} reachable entry point(s). A downstream CI label is warranted; "
          f"confirm the current ci:* definitions in .github/workflows/ before naming one.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
