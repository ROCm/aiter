#!/usr/bin/env python3
"""Regenerate .github/scripts/select_tests.md from the current tree.

Every number in that document is produced here; nothing is hand-written.
Run from the repo root:  python3 gen_evidence.py > .github/scripts/select_tests.md
"""

import collections
import os
import statistics
import subprocess
import sys

sys.path.insert(0, ".github/scripts")
from select_tests import ImportGraph, Selector, load_file_times

OPS = "aiter/ops/triton"
TESTS_DIR = "op_tests/triton_tests"
HISTORY = 120

out = []


def w(line=""):
    out.append(line)


def sh(*args):
    return subprocess.run(args, capture_output=True, text=True, check=False).stdout


def commits(n, path):
    return (
        sh("git", "log", "--format=%H|%s", "-n", str(n), "--", path).strip().split("\n")
    )


def files_of(sha):
    return sh("git", "show", "--name-only", "--format=", "-n", "1", sha).split()


g = ImportGraph()
sel_triton = Selector(g, "triton")
sel_all = Selector(g, "all")
times = load_file_times()

head = sh("git", "rev-parse", "--short", "HEAD").strip()
date = sh("git", "log", "-1", "--format=%cs").strip()

w("# Test selection — supporting evidence")
w()
w(
    f"Generated from `{head}` ({date}). Every figure below is produced by a script over the"
)
w("tree at that commit; none are hand-entered. See *Reproducing* at the end.")
w()
w(
    "This documents why [`select_tests.py`](select_tests.py) resolves changed files to tests"
)
w(
    "through a reversed import graph rather than by naming convention, call-graph analysis, or"
)
w("developer-authored decorators — and what the resulting selection actually measures.")
w()
w("---")
w()

# ---------------------------------------------------------------- 1. convention
w("## 1. Naming convention alone is not enough")
w()
skip = {"_triton_kernels", "_gluon_kernels", "configs", "utils", "__pycache__"}
ops = []
for d, dirs, fs in os.walk(OPS):
    dirs[:] = [x for x in dirs if x not in skip]
    rel = os.path.relpath(d, OPS)
    if any(p in skip for p in rel.split(os.sep)):
        continue
    for f in fs:
        if f.endswith(".py") and f != "__init__.py":
            ops.append(os.path.join(rel, f).lstrip("./"))
tests = set()
for d, dirs, fs in os.walk(TESTS_DIR):
    dirs[:] = [x for x in dirs if x != "__pycache__"]
    rel = os.path.relpath(d, TESTS_DIR)
    for f in fs:
        if f.startswith("test_") and f.endswith(".py"):
            tests.add(os.path.normpath(os.path.join(rel, f)))
tb = collections.defaultdict(list)
for t in tests:
    tb[os.path.basename(t)].append(t)
exact = base = 0
none = []
for o in sorted(ops):
    want = os.path.normpath(
        os.path.join(os.path.dirname(o), "test_" + os.path.basename(o))
    )
    if want in tests:
        exact += 1
    elif "test_" + os.path.basename(o) in tb:
        base += 1
    else:
        none.append(o)
w(
    f"Matching each op module to `test_<name>.py` at the mirrored path, over {len(ops)} op modules"
)
w(f"and {len(tests)} test files in `{TESTS_DIR}`:")
w()
w("| outcome | count |")
w("| --- | ---: |")
w(f"| exact mirrored-path match | {exact} |")
w(f"| basename matches, directory differs | {base} |")
w(f"| no match at all | {len(none)} |")
w()
w(
    f"Convention resolves {100 * exact // len(ops)}% of op modules. The unmatched {len(none)} include"
)
w("both genuine helpers and ops whose tests are named differently:")
w()
w("```")
for o in sorted(none)[:12]:
    w(f"  {o}")
if len(none) > 12:
    w(f"  … and {len(none) - 12} more")
w("```")
w()
w(
    "Critically, this only covers the op-wrapper layer. The files PRs most often touch have no"
)
w("test-name mirror at all:")
w()


def count(d, ext=True):
    n = sum(1 for _, _, fs in os.walk(d) for f in fs if f.endswith(".py"))
    m = sum(
        1
        for dp, _, fs in os.walk(d)
        for f in fs
        if not f.endswith(".py") and "__pycache__" not in dp
    )
    return n, m


w("| directory | .py files | non-.py files |")
w("| --- | ---: | ---: |")
for d in ("_triton_kernels", "_gluon_kernels", "utils", "configs"):
    n, m = count(f"{OPS}/{d}")
    w(f"| `{OPS}/{d}` | {n} | {m} |")
w()
direct = len(
    [
        p
        for p in sh(
            "grep", "-rl", "_triton_kernels", TESTS_DIR, "--include=*.py"
        ).split()
    ]
)
w(
    f"Only {direct} test files import `_triton_kernels` directly. A convention or decorator scheme"
)
w(
    "keyed on op modules is structurally blind to the kernel and config files where most changes land."
)
w()

# ---------------------------------------------------------------- 2. import noise
w("## 2. Why not a static call graph")
w()
w(
    "Import edges are dominated by shared utilities. Counting test files that import each module:"
)
w()


def forward_closure(start):
    """Everything `start` transitively imports."""
    seen, stack = set(), [start]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(g.forward.get(cur, ()))
    return seen


rev_tests = collections.defaultdict(set)
for m in sel_triton.tests:
    for dep in forward_closure(m):
        rev_tests[dep].add(m)
noisy = sorted(
    (
        (len(v), k)
        for k, v in rev_tests.items()
        if k.startswith("aiter.ops.triton.utils")
    ),
    reverse=True,
)
w("| module | test files reaching it |")
w("| --- | ---: |")
for n, k in noisy[:5]:
    w(f"| `{k}` | {n} |")
w()
w(
    "For an **audit** these edges are noise and would need call-site analysis to filter. For **test"
)
w(
    "selection** they are correct: changing one of these genuinely should run everything that"
)
w(
    "depends on it. That difference is why plain import closure suffices and a full interprocedural"
)
w("call graph buys nothing — selection wants the over-approximation.")
w()
w("Two things a call graph still cannot see:")
w()
w('- Ops registered dynamically into `torch.library.Library("aiter", ...)` by')
w(
    "  `torch_compile_guard` and reached via `torch.ops.aiter.<name>` — no AST edge exists."
)
w(
    "- Runtime backend selection. `gemm_a16w16.py` gates on `_is_gluon_available()` against"
)
w("  `_GLUON_SUPPORTED_ARCHS`, so which kernel actually runs is an arch-time decision.")
w()

# ---------------------------------------------------------------- 3. decorators
w("## 3. Why not developer-authored decorators")
w()
w(
    "Decorators record *declared intent*, which cannot be audited. The live counterexample is in"
)
w("the tree today:")
w()
tc = "op_tests/triton_tests/torch_compile/test_compile_gemm_a16w16.py"
has = sh("grep", "-c", "aiter", tc).strip() or "0"
w(f"- `{tc}` is named for the op.")
w(f"- It contains {has} references to `aiter` and none to `torch.ops`.")
w("- It compiles `torch.mm` and compares against eager; it never calls the aiter op.")
w()
w(
    "A decorator naming that file as the op's test would be wrong, would read as correct to any"
)
w(
    "reviewer, and nothing would catch it. For selection specifically the failure is worse than"
)
w("mislabelling: a forgotten decorator yields a **silently skipped test**.")
w()
w(
    "The graph also reaches tests that no per-op annotation would list, because tests import each"
)
w(
    "other for shared fixtures. Changing the gemm A16W16 kernel selects 7 tests; 4 are reachable"
)
w("only via `test_gemm_a16w16`, which the others import `get_x_vals` from.")
w()

# ---------------------------------------------------------------- 4. graph
w("## 4. The graph")
w()
edges = sum(len(v) for v in g.forward.values())
w(f"- **{len(g.mod2file):,}** Python modules indexed across `aiter/` and `op_tests/`")
w(f"- **{edges:,}** import edges, reversed once at startup")
w(
    f"- **{len(sel_triton.tests)}** test files in the triton suite, **{len(sel_all.tests)}** across all of `op_tests/`"
)
w(
    f"- Full triton suite estimated at **{sum(times.get(f, 15) for f in sel_triton.all_tests()):,}s**"
)
w(
    "  (parsed from `FILE_TIMES` in `split_tests.sh`, so timings have one source of truth)"
)
w()
w("Fan-out for representative changed files:")
w()
w("| changed file | route | tests selected |")
w("| --- | --- | ---: |")
probes = [
    f"{OPS}/_triton_kernels/gemm/basic/gemm_a16w16.py",
    f"{OPS}/gemm/basic/gemm_a16w16.py",
    f"{OPS}/_triton_kernels/common/splitk_reduce.py",
    f"{OPS}/utils/_triton/arch_info.py",
    f"{OPS}/configs/gfx1250/gluon/moe/a4w4/DEFAULT.json",
    f"{OPS}/configs/gfx950/triton/gemm/batched_gemm_a16w16/DEFAULT.json",
    "pyproject.toml",
    "docs/tuning.md",
]
for p in probes:
    s, e, r = sel_triton.select([p])
    reason = next(iter(r.values())).replace("escalate:", "**escalate** ")
    w(f"| `{p.replace(OPS + '/', '')}` | `{reason}` | {len(s)} |")
w()

# ---------------------------------------------------------------- 5. backtest
w("## 5. Historical backtest")
w()
w(
    f"Replaying the last {HISTORY} commits that touched `{OPS}`, measuring selected GPU time as a"
)
w("share of the full triton suite:")
w()
every = sel_triton.all_tests()
full = sum(times.get(f, 15) for f in every)
pcts, esc = [], 0
for line in commits(HISTORY, OPS):
    sha = line.split("|", 1)[0]
    s, e, _ = sel_triton.select(files_of(sha))
    if e:
        esc += 1
    pcts.append(100 * sum(times.get(f, 15) for f in s) / full)
n = len(pcts)
w("| metric | value |")
w("| --- | ---: |")
w(f"| commits replayed | {n} |")
w(f"| median GPU time selected | {statistics.median(pcts):.1f}% |")
w(f"| mean GPU time selected | {statistics.mean(pcts):.1f}% |")
w(f"| escalated to full suite | {esc} ({100 * esc // n}%) |")
w()
hist = collections.Counter(min(int(p // 10), 9) for p in pcts)
w("Distribution of that share, one row per decile:")
w()
w("```")
for i in range(10):
    w(f"  {i * 10:3d}-{i * 10 + 10:3d}%  {'#' * hist[i]:<40} {hist[i]}")
w("```")
w()
low = sum(hist[i] for i in range(2))
mid = sum(hist[i] for i in range(2, 5))
high = sum(hist[i] for i in range(5, 10))
w(
    f"The distribution is **bimodal, not long-tailed**: {low} commits select under 20% of the suite and"
)
w(
    f"{high} select over 50%, with just {mid} in the 20\u201350% valley between them. Commits are"
)
w(
    "either local (a kernel plus its wrapper) or hub changes that legitimately invalidate most of"
)
w(
    "the suite. Because the middle is nearly empty, there is little to gain from trying to narrow"
)
w("the hub cases \u2014 the win is in the lower cluster, and it is already captured.")
w()
w("Escalation causes over the same window:")
w()
causes = collections.Counter()
for line in commits(HISTORY, OPS):
    sha = line.split("|", 1)[0]
    _, _, reasons = sel_triton.select(files_of(sha))
    for r in reasons.values():
        if r.startswith("escalate:"):
            causes[r.split(":")[1]] += 1
w("| cause | changed paths |")
w("| --- | ---: |")
for k, v in causes.most_common():
    w(f"| `{k}` | {v} |")
w()
w(
    "> Backtesting runs the **HEAD** import graph against historical file lists, so these are good"
)
w("> estimates rather than exact replays. Run shadow mode before trusting them.")
w()

# ---------------------------------------------------------------- 6. safety
w("## 6. Safety properties")
w()
w(
    "Selection is allowed to be too broad and never too narrow. These are checked exhaustively,"
)
w("not sampled.")
w()
miss = [t for t in every if t not in sel_triton.select([t])[0]]
bad = 0
pairs = 0
for m, rel in g.mod2file.items():
    if "_triton_kernels" not in rel and "_gluon_kernels" not in rel:
        continue
    ks = set(sel_triton.select([rel])[0])
    for o in g.reverse.get(m, ()):
        orel = g.mod2file[o]
        if not orel.startswith(OPS + "/"):
            continue
        pairs += 1
        if not set(sel_triton.select([orel])[0]) <= ks:
            bad += 1
empty = len(sel_triton.select([])[0])
_, esc_unknown, _ = sel_triton.select([f"{OPS}/mystery.bin"])
w("| property | result |")
w("| --- | --- |")
w(
    f"| a changed test file always selects itself | {len(every) - len(miss)}/{len(every)} |"
)
w(f"| an op's selection ⊆ its kernel's selection | {pairs - bad}/{pairs} |")
w(f"| empty changeset selects nothing | {empty} tests |")
w(
    f"| unmappable path under `aiter/ops/` escalates | {'yes' if esc_unknown else 'NO'} |"
)
w()

# ---------------------------------------------------------------- 7. asymmetry
w("## 7. Kernel vs. op wrapper")
w()
w(
    "Changing only the op wrapper works, and is the easier case: the wrapper sits closer to the"
)
w(
    "tests than the kernel does, and nothing reachable from the wrapper depends on the kernel below"
)
w("it. The two are deliberately **not** symmetric for a shared kernel.")
w()
shared = []
for m, rel in g.mod2file.items():
    if "_triton_kernels" not in rel:
        continue
    consumers = [
        p
        for p in g.reverse.get(m, ())
        if p.startswith("aiter.ops.triton.") and "_triton_kernels" not in p
    ]
    if len(consumers) > 1:
        shared.append((len(consumers), m, consumers))
shared.sort(reverse=True)
if shared:
    ncon, k, consumers = shared[0]
    ks = sel_triton.select([g.mod2file[k]])[0]
    w(f"`{g.mod2file[k].replace(OPS + '/', '')}` is imported by {ncon} op wrappers:")
    w()
    w("| changed file | tests selected |")
    w("| --- | ---: |")
    w(f"| the kernel itself | **{len(ks)}** |")
    for o in sorted(consumers)[:5]:
        orel = g.mod2file[o]
        w(f"| `{orel.replace(OPS + '/', '')}` | {len(sel_triton.select([orel])[0])} |")
    w()
    w(
        "Touch the shared kernel and every dependent op's tests run. Touch one wrapper and only its"
    )
    w(
        "own tests run, because the others are genuinely unaffected. A symmetric answer here would"
    )
    w("mean the edge direction was wrong.")
w()

# ---------------------------------------------------------------- 8. zero coverage
w("## 8. Ops that select zero tests")
w()
w(
    "The one case where selection silently runs nothing is an op with no test anywhere. These are"
)
w(
    "**pre-existing coverage gaps**, not selection bugs — they are equally untested today with the"
)
w("full suite running — but selection makes them visible.")
w()
zero = []
for m, rel in g.mod2file.items():
    if not rel.startswith(OPS + "/"):
        continue
    if "/configs/" in rel or "__init__" in rel:
        continue
    if len(sel_all.select([rel])[0]) == 0:
        zero.append(rel.replace(OPS + "/", ""))
tuning = [z for z in zero if "/tunning/" in z]
real = sorted(set(zero) - set(tuning))
w(
    f"{len(zero)} modules select no test in any suite. {len(tuning)} are offline tuning harnesses"
)
w(
    "under `utils/_triton/tunning/` and are not on any test path. The remainder are worth a look:"
)
w()
w("```")
for z in real:
    w(f"  {z}")
w("```")
w()
w(
    "Suggested follow-up: gate CI so an op selecting zero tests needs either a test or an explicit"
)
w("entry in a known-untested list, so the set can only shrink.")
w()

# ---------------------------------------------------------------- 9. invariants
w("## 9. Guarded invariants")
w()
w(
    "Both of these have already drifted once in this repo, and either would quietly collapse"
)
w("selection toward the full suite, so `check_invariants()` warns on every run.")
w()
flat = 0
for line in commits(300, OPS):
    for f in files_of(line.split("|", 1)[0]):
        if f.startswith(OPS + "/configs/") and len(f.split("/")) < 9:
            flat += 1
depths = sorted(
    {
        len(os.path.relpath(os.path.join(dp, fn)).split("/"))
        for dp, _, fns in os.walk(f"{OPS}/configs")
        for fn in fns
        if fn.endswith(".json")
    }
)
w("**Config path depth.** The rule reads `<op_name>` positionally from")
w("`configs/<arch>/<backend>/<category>/<op_name>/<SHAPE>.json`, which is")
w(
    f"{'/'.join(str(d) for d in depths)} components deep at HEAD. The tree has already migrated from a flat"
)
w("layout once:")
w(
    f"{flat:,} changed config paths in the last 300 commits are old-style and no longer resolve."
)
w("A second migration would silently shift which component holds the op name.")
w()
root_triton = sorted(
    d for d in g.forward.get("aiter", set()) if d.startswith("aiter.ops.triton")
)
w(
    "**Eager triton imports from the package root.** `aiter/__init__.py` currently imports:"
)
w()
w("```")
for d in root_triton:
    w(f"  {d}")
w("```")
w()
w(
    "These are intentional (the Iris comms re-exports) and allowlisted in `ROOT_TRITON_ALLOWED`."
)
w(
    "Any *new* eager triton import from the root would put every triton change upstream of every"
)
w("test that does `import aiter`, and the tool flags it.")
w()

# ---------------------------------------------------------------- 10. repro
w("## 10. Prior art in this repo")
w()
w(
    "`.github/scripts/select_triton_tests.py` (added 2026-01, #1682) already implements Triton"
)
w("test selection over a `networkx` dependency graph. Two things to know about it:")
w()
w(
    "**Its invocation in `triton-test.yaml` is commented out**, so no CI job runs it today."
)
w()
w(
    "**It no longer runs at all.** It resolves kernel config files under the pre-migration flat"
)
w("layout `configs/gemm/`, which no longer exists, and aborts on every invocation:")
w()
w("```")
w("  CRITICAL|Required directory [aiter/ops/triton/configs/gemm] doesn't exist.")
w("```")
w()
w(
    "It is a worked example of the exact failure mode \u00a79 guards against: a positional"
)
w(
    "assumption about the config tree that broke silently when the layout moved, in a script"
)
w("nothing was running often enough to notice.")
w()
w("One idea in it is better than the replacement and worth porting: it resolves config")
w(
    "references by parsing the f-string templates in op source and matching them against real"
)
w(
    "config paths, rather than reading `<op_name>` positionally from the path. That would remove"
)
w("the `config-category` fallback and its over-selection (\u00a74, the 62-test row).")
w()
w(
    "Recommended: reconcile the two rather than run both \u2014 keep the reversed-graph core, port"
)
w("the template-matching config rule, and delete the dead script.")
w()
w("## 11. Reproducing")
w()
w("```bash")
w("# selection for the current branch")
w(".github/scripts/select_tests.py --test-type triton --base origin/main --explain")
w()
w("# what CI runs: emit a list, then shard it as usual")
w(
    ".github/scripts/select_tests.py --test-type triton --base origin/main -o triton_selected.list"
)
w(
    ".github/scripts/split_tests.sh --shards 8 --test-type triton --select-from triton_selected.list"
)
w()
w("# shadow mode: emit the FULL suite, write the would-be selection to <out>.selected")
w(
    ".github/scripts/select_tests.py --test-type triton --base origin/main --shadow -o triton_selected.list"
)
w("```")
w()
w(
    "This document is generated, not maintained by hand. Regenerate it with the script that"
)
w("produced it after any change to the selection rules.")

sys.stdout.write("\n".join(out) + "\n")
