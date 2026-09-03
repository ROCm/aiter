#!/usr/bin/env python3
"""Step 1b triage: derive which rule families apply, and collect their evidence.

Replaces Step 3's self-applied prose checklist. Emits only the matching rules so
the model reads ~12 instead of all of them.  Conservative: when a type cannot be decided
structurally it is INCLUDED, never dropped.
"""
import re, sys, pathlib
from collections import OrderedDict

TIER = ("aiter/jit/core.py", "aiter/__init__.py", "aiter/fused_moe.py",
        "aiter/mla.py", "aiter/tuned_gemm.py", "aiter/ops/mha.py",
        "aiter/ops/attention.py", "aiter/ops/gemm_op_a8w8.py",
        "aiter/ops/moe_op.py", "aiter/ops/quant.py")
DOWNSTREAM = ("mla", "fused_moe", "attention", "mha", "quant", "gemm_op_a8w8",
              "moe_op", "jit/core")

def parse(diff):
    files, cur = OrderedDict(), None
    for ln in diff.splitlines():
        m = re.match(r"^diff --git a/(\S+) b/(\S+)", ln)
        if m:
            cur = m.group(2); files[cur] = {"add": [], "del": [], "new": False}
        elif cur:
            if ln.startswith("new file mode"): files[cur]["new"] = True
            elif ln.startswith("+") and not ln.startswith("+++"): files[cur]["add"].append(ln[1:])
            elif ln.startswith("-") and not ln.startswith("---"): files[cur]["del"].append(ln[1:])
    return files


def _calls(src, name):
    """Argument text of each `name(...)` call, paren-balanced.

    A regex with `[^)]*` stops at the first close paren, so `tl.load(p, mask=(a<b),
    other=0.0)` reads as having no `other=` -- 20 of the 195 Triton PRs in the corpus
    contain exactly that shape."""
    out, i, key = [], 0, name + "("
    while True:
        j = src.find(key, i)
        if j < 0:
            return out
        k, depth = j + len(key), 1
        while k < len(src) and depth:
            if src[k] == "(":
                depth += 1
            elif src[k] == ")":
                depth -= 1
            k += 1
        out.append(src[j + len(key):k - 1])
        i = k


def triton_families(add):
    """Triton-specific families. Kept separate from the generic derivation because the
    generic one was tuned on the whole corpus and barely discriminates here: over the 195
    Triton PRs among 600 open aiter PRs, `ops-wrapper` fires on 88% and `modified-kernel`
    on 79%, so a Triton PR arrived carrying a rule set that said little about being Triton.
    Each family below fires on at most 52% of Triton PRs and at most 17% of all of them."""
    fams = []
    loads, stores = _calls(add, "tl.load"), _calls(add, "tl.store")
    unmasked = [c for c in loads + stores if "mask=" not in c]
    no_other = [c for c in loads if "mask=" in c and "other=" not in c]
    if unmasked or no_other:
        fams.append(("triton-mask-bounds", "T1 T2"))
    if re.search(r"\bnum_warps\b|\bnum_stages\b|\bnum_ctas\b|waves_per_eu|"
                 r"matrix_instr|\bkpack\b", add):
        fams.append(("triton-launch-cfg", "T3 T4"))
    if re.search(r"tl\.dot\(|\.to\(tl\.float32\)|allow_tf32|input_precision", add):
        fams.append(("triton-accum-prec", "T5"))
    if re.search(r"grid\s*=\s*lambda|tl\.program_id|tl\.cdiv\(", add):
        fams.append(("triton-grid-map", "T6"))
    return fams


def derive(files, title=""):
    paths = list(files)
    add = "\n".join(l for f in files.values() for l in f["add"])
    dele = "\n".join(l for f in files.values() for l in f["del"])
    t, T = [], title.lower()
    def hit(name, rules): t.append((name, rules))

    INFRA = (".github/", "docs/", "3rdparty/", "third_party/", ".gitignore",
             "requirements", "README", "CONTRIBUTING", ".gitmodules",
             "setup.py", "conftest.py", "pyproject.toml", "hsa/")
    if paths and all(p.startswith(INFRA) or p.endswith((".md", ".txt", ".cfg", ".toml"))
                     or "/docs/" in p for p in paths):
        return [("infra-only", "NONE")]
    # 纯 submodule / 版本指针更新: 只动 subproject commit 行
    if add and all(re.match(r"^(Subproject commit|[0-9a-f]{40}$|.*version.*=)", l.strip())
                   for l in add.splitlines() if l.strip()):
        return [("version-bump", "NONE")]

    if any(f["new"] and (p.startswith("csrc/kernels/") or "/triton/" in p or "_gluon_kernels/" in p)
           for p, f in files.items()):
        hit("new-kernel", "B1 B2 B4 A1 D1 D8 HK6 P6")
    KERNEL_PY = ("aiter/ops/triton/", "_triton_kernels/", "_gluon_kernels/", "/gluon/")
    def is_kernel(p):
        return p.endswith((".cu", ".cuh", ".hip", ".h", ".hpp")) or \
               (p.endswith(".py") and any(k in p for k in KERNEL_PY))
    if any(not f["new"] and is_kernel(p) and (f["add"] or f["del"]) for p, f in files.items()):
        hit("modified-kernel", "A1 B2 D1 D8 P6")   # D9 由 scan_index_width.py 出结果, 不读散文
    if re.search(r"tl\.constexpr", add) or re.search(r"['\"]gfx\d+['\"]", add):
        hit("new-routing-value", "B4 C4")
    if any(p.endswith((".csv", ".yaml", ".yml")) or
           (p.endswith(".json") and ("config" in p or "tuned" in p)) for p in paths):
        hit("tuning-config", "D3 HK4")
    if re.search(r"^\s*(el)?if .*(dtype|arch|layout|quant|backend)", add, re.M):
        hit("dispatch-change", "B1 B3 B4 A3")
    # Tier1 真身: 坏了整个 import aiter 就废, 值得完整 Step4
    if any(p in ("aiter/__init__.py", "aiter/jit/core.py") for p in paths):
        hit("tier1-import-chain", "STEP4 E5")
    # Tier2 骨干: 多模型族共用的 dispatch, 需要 owner 签字
    elif any(p in TIER for p in paths):
        hit("tier2-dispatch", "E5 E4")
    # 其余 op wrapper: 只需确认导出符号没断, 不必走完整 Step4
    elif any(p.startswith(("aiter/ops/", "aiter/jit/")) or
             (p.startswith("aiter/") and p.count("/") == 1 and p.endswith(".py"))
             for p in paths):
        hit("ops-wrapper", "B6")
    if re.search(r"^\s*(def |void |template)", dele, re.M) and re.search(r"^\s*(def |void |template)", add, re.M):
        hit("api-signature", "B6 E1 E5")
    if re.search(r"\b(fp8|e4m3|e5m2|fnuz|mxfp[48]|fp4|int8)\b|\bfp8_max\b|448\.0|240\.0", add, re.I):
        hit("fp8-quant", "C1 C2 D1")
    if any(k in T for k in ("perf", "optimize", "fuse", "faster", "speedup", "%")):
        hit("perf", "P1 P2 P3 P5 P6")
    if paths and all(p.startswith(("op_tests/", "tests/")) for p in paths):
        hit("tests-only", "P2 HK6")
    if re.search(r"cuda\.Stream|stream=|wait_stream", add):
        hit("async-stream", "G1 G1b")
    if any("flydsl" in p.lower() for p in paths):
        hit("flydsl", "D10 D10b")
    if any(d in p for p in paths for d in DOWNSTREAM):
        hit("downstream-op", "E4 E5")
    if any("codegen" in p or p.startswith("csrc/cpp_itfs/") or p.endswith("Makefile")
           for p in paths):
        hit("codegen-buildtool", "A1 D5 B6")
    if re.search(r"@compile_ops|torch\.library\.custom_op", add):
        hit("new-compile-op", "D7 D6")
    if re.search(r"mutates_args|torch_compile_guard|register_fake|_fake\b|gen_fake|abstract_impl", add):
        hit("compile-contract", "D6 D7")
    if re.search(r"(assert |torch\.zeros|\.contiguous\(\)|AITER_CHECK|TORCH_CHECK)", dele):
        hit("invariant-removed", "D4 B7")
    if re.search(r"\bOOB\b|out.of.bounds|overflow|garbage|race|deadlock|leak", T) \
       or re.search(r"__threadfence|__syncthreads|__builtin_amdgcn_s_barrier|atomicAdd|atomic_", add):
        hit("memory-safety", "B2 D1 G1")
    if re.search(r"strided|non.contiguous|contiguous", T) or re.search(r"\.contiguous\(\)|is_contiguous|stride\(", add):
        hit("layout-contiguity", "D8 B7")
    if re.search(r"divisib|non-128|not.*multiple|alignment", T) \
       or re.search(r"%\s*\d+\s*[=!]=|\bceil_div\b|\balign_up\b|\bpad(ded|ding)?_to\b", add):
        hit("alignment", "B7 B2 D8")
    if re.search(r"_preshuffled|_quantized|weight_transform", add):
        hit("weight-variant", "F1")
    if re.search(r'os\.environ\.get\(\s*["\']AITER_', add):
        hit("new-env-var", "HK9 D2")
    if any(p.startswith(("aiter/ops/triton/",)) or "/triton/" in p or
           "_triton_kernels/" in p or "_gluon_kernels/" in p or "/gluon/" in p
           for p in paths):
        t.extend(triton_families(add))
    return t

ALL_RULES = ("A1 A2 A3 B1 B2 B3 B4 B5 B6 B7 C1 C2 C3 C4 D1 D1b D2 D3 D4 D5 D6 D7 D8 "
             "D10 D10b E1 E2 E3 E4 E5 F1 G1 G1b P1 P2 P3 P4 P5 P6 HK1 HK2 HK3 "
             "T1 T2 T3 T4 T5 T6 STEP4")

def deleted_guard_symbols(diff_text):
    """Symbols named by assert / *_CHECK lines the diff removes."""
    syms = set()
    for ln in diff_text.splitlines():
        if not ln.startswith("-") or ln.startswith("---"):
            continue
        body = ln[1:]
        if not re.search(r"\b(assert|AITER_CHECK|TORCH_CHECK)\b", body):
            continue
        for m in re.finditer(r"\b([a-zA-Z_]\w*)\s+is\s+not\s+None", body):
            syms.add(m.group(1))
        for m in re.finditer(r"\b([a-zA-Z_]\w*)->", body):
            syms.add(m.group(1))
    return sorted(syms)

def evidence(sym, files):
    out = []
    for f in files:
        p = pathlib.Path(f)
        if not p.exists():
            continue
        hits = [(i, l.rstrip()) for i, l in enumerate(p.read_text(errors="replace").splitlines(), 1)
                if re.search(rf"\b{re.escape(sym)}\b", l)]
        if hits:
            out.append((f, hits))
    return out


# ---------------------------------------------------------------- symbol sweep
STDLIB_OR_THIRD_PARTY = re.compile(
    r"^(torch|pytest|numpy|np|pandas|einops|triton|typing|dataclasses|functools|"
    r"itertools|pathlib|argparse|logging|os|sys|re|json|math|time|random|"
    r"collections|contextlib|subprocess|warnings|abc|enum|copy|__future__)\b")


def added_imports(diff_text):
    """(module, [names], line) for every import line the diff ADDS."""
    out = []
    for ln in diff_text.splitlines():
        if not ln.startswith("+") or ln.startswith("+++"):
            continue
        body = ln[1:].strip()
        m = re.match(r"from\s+([\w.]+)\s+import\s+(.+)$", body)
        if m:
            mod, names = m.group(1), m.group(2)
            names = names.split("#")[0].replace("(", "").replace(")", "")
            out.append((mod, [n.strip().split(" as ")[0] for n in names.split(",") if n.strip()], body))
            continue
        m = re.match(r"import\s+([\w.]+)", body)
        if m:
            out.append((m.group(1), [], body))
    return out


def resolve_module(mod, root):
    """Path of the .py/package backing a dotted module, or None.

    A directory with no __init__.py is a namespace package and imports fine --
    aiter/utility is one. Requiring __init__.py reported it as hallucinated."""
    rel = mod.replace(".", "/")
    for cand in (root / f"{rel}.py", root / rel / "__init__.py"):
        if cand.exists():
            return cand
    d = root / rel
    if d.is_dir():
        return d          # namespace package: exists, symbols unresolvable here
    return None


def symbol_defined(path, name):
    if path.is_dir():
        return True       # namespace package: cannot disprove without walking it
    # `from pkg import sub` binds a SUBMODULE, which need not be named in
    # __init__.py at all. Searching only the __init__ text called four real
    # imports invented across the 600-PR corpus -- aiter.jit/core,
    # aiter.jit.utils/chip_info, flydsl.kernels/{buffer_ops,vector}.
    if path.name == "__init__.py":
        pkg = path.parent
        if (pkg / f"{name}.py").exists() or (pkg / name).is_dir():
            return True
    try:
        src = path.read_text(errors="replace")
    except Exception:
        return True   # unreadable: do not accuse
    if re.search(rf"^\s*(def|class)\s+{re.escape(name)}\b", src, re.M):
        return True
    if re.search(rf"^\s*{re.escape(name)}\s*[:=]", src, re.M):
        return True
    # re-exported through a star import or __all__; cannot disprove statically
    if "import *" in src or "__all__" in src:
        return True
    return False


def sweep_symbols(diff_text, root):
    """Imports the diff adds that cannot be resolved against `root`. Conservative:
    anything not clearly first-party, or not clearly absent, is left alone.

    `root` MUST be a checkout of the branch this PR will merge INTO, fetched fresh --
    not the PR base, and not whatever the reviewer happens to have on disk. The tree
    you resolve against decides whether this check fires at all, and a stale root
    silently reports nothing. aiter#4994 is the worked example: it added
    `from aiter.ops.flydsl.utils import is_flydsl_available`, and #5116 (3b2a9ce6)
    had deleted that module from main 19 hours earlier. Run against the PR's own base
    the import resolves and this returns clean; run against main-at-merge-time it
    reports. #4994 merged green and was reverted 5.5 hours later (283e1d4b).

    A hit is therefore a REBASE signal, not an accusation of invention -- the import
    may well have been valid when it was written.

    Files the diff ITSELF adds count as existing. The tree available here is the
    base, not head, so without this every PR that adds a module and imports it --
    32% of open aiter PRs, measured -- is reported as unresolved. The check
    that matters is 'this symbol exists nowhere, including in the PR', not 'this
    symbol is missing from base'."""
    bad = []
    added_paths, cur = set(), None
    for ln in diff_text.splitlines():
        m = re.match(r"^diff --git a/(\S+) b/(\S+)", ln)
        if m:
            cur = m.group(2)
        elif ln.startswith("new file mode") and cur:
            added_paths.add(cur)
    added_syms = set()
    for ln in diff_text.splitlines():
        if ln.startswith("+") and not ln.startswith("+++"):
            m = re.match(r"\s*(?:def|class)\s+(\w+)", ln[1:])
            if m:
                added_syms.add(m.group(1))
            m = re.match(r"\s*(\w+)\s*[:=]", ln[1:])
            if m:
                added_syms.add(m.group(1))
    for mod, names, line in added_imports(diff_text):
        if STDLIB_OR_THIRD_PARTY.match(mod) or mod.startswith("."):
            continue
        top = mod.split(".")[0]
        if not (root / top).exists() and not (root / f"{top}.py").exists():
            continue          # not a first-party module -- out of scope
        rel = mod.replace(".", "/")
        if f"{rel}.py" in added_paths or f"{rel}/__init__.py" in added_paths:
            continue          # this PR adds the module
        target = resolve_module(mod, root)
        if target is None:
            bad.append((line, f"module '{mod}' exists neither in the tree nor in this diff"))
            continue
        for n in names:
            if n in added_syms:
                continue      # this PR defines the symbol
            if not symbol_defined(target, n):
                bad.append((line, f"'{n}' is defined neither in {target.relative_to(root)} nor in this diff"))
    return bad


# ------------------------------------------------------------------ rule ledger
ADJUDICATION = re.compile(
    r"^\s*([A-Z]+\d+[a-z]?)\s+(FIRE|CLEAR|N/A)\s*(?:--|—|:)\s*(.+?)\s*$")
MIN_EVIDENCE = 12


def expected_rules(rules_text):
    """Rule ids `rules` mode derived for this diff."""
    ids = []
    for ln in rules_text.splitlines():
        m = re.match(r"\s*\[[\w-]+\s*\]\s*(.+)$", ln)
        if m:
            ids.extend(t for t in m.group(1).split() if re.fullmatch(r"[A-Z]+\d+[a-z]?", t))
    return sorted(set(ids))


CITATION = re.compile(r"([\w./-]+\.(?:py|cu|cuh|hip|h|hpp|cpp|cc|csv|json|yaml|yml))"
                      r"(?::(\d+))?")


def changed_paths(diff_text):
    return set(re.findall(r"^diff --git a/\S+ b/(\S+)", diff_text, re.M))


def check_citations(verdicts_text, diff_text):
    """Verdict lines whose cited file is not in this PR's changed set.

    A reason is free text, so the gate can only check the part of it that refers to
    something outside the model: the path. A FIRE on a file the diff never touched is
    either a stale citation carried over from another review or invented, and both are
    worth stopping.

    Only FIRE is checked. A CLEAR legitimately cites files this PR does not touch --
    "E5 CLEAR: aiter/__init__.py unchanged" is a correct reason precisely because that
    file is absent from the diff, and flagging it would teach the reviewer to stop
    citing anything, which is worse than not checking. Reasons naming no file are not
    flagged either. This narrows the gap; it does not close it."""
    if not diff_text:
        return []
    touched = changed_paths(diff_text)
    bad = []
    for ln in verdicts_text.splitlines():
        m = ADJUDICATION.match(ln)
        if not m:
            continue
        rule, verdict, reason = m.groups()
        if verdict != "FIRE":
            continue
        for path, _line in CITATION.findall(reason):
            base = path.lstrip("./")
            if any(t == base or t.endswith("/" + base) for t in touched):
                continue
            bad.append((rule, verdict, path))
    return bad


def audit_ledger(rules_text, verdicts_text):
    """(missing, thin) -- rules never adjudicated, and rules adjudicated without
    evidence. Empty/empty means the rule pass actually happened.

    This exists because the skill's own history says prose does not survive contact
    with a busy reviewer: the revised D9 text caught 0 of 3 known overflow defects
    across a 14-PR run and the scanner it names was never once invoked. A checklist
    the model marks off to itself degrades under load, and the load is about to be
    large. This is the same move as running the D9 scan inside Step 1 -- make the
    step produce an artifact something else can check."""
    want = expected_rules(rules_text)
    seen = {}
    for ln in verdicts_text.splitlines():
        m = ADJUDICATION.match(ln)
        if m:
            seen[m.group(1)] = (m.group(2), m.group(3))
    missing = [r for r in want if r not in seen]
    thin = [(r, seen[r][0]) for r in want
            if r in seen and len(seen[r][1].strip()) < MIN_EVIDENCE]
    return missing, thin


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("rules", "evidence", "symbols", "ledger"):
        print("usage: triage.py rules <diff> [title]\n"
              "       triage.py evidence <diff> <head-file>...\n"
              "       triage.py symbols <diff> <merge-target-root>\n"
              "       triage.py ledger <rules.txt> <verdicts.txt> [diff]", file=sys.stderr)
        raise SystemExit(2)

    if mode == "ledger":
        rules_text = open(sys.argv[2], errors="replace").read()
        try:
            verdicts_text = open(sys.argv[3], errors="replace").read()
        except OSError:
            verdicts_text = ""
        missing, thin = audit_ledger(rules_text, verdicts_text)
        diff_text = ""
        if len(sys.argv) > 4:
            try:
                diff_text = open(sys.argv[4], errors="replace").read()
            except OSError:
                diff_text = ""
        stale = check_citations(verdicts_text, diff_text)
        want = expected_rules(rules_text)
        if not want:
            # Fail closed. An empty or unreadable rules file is the state produced by a
            # Step 1b that never ran, which is precisely the run that must not reach a
            # verdict -- passing here would make skipping the derivation the cheapest path.
            print("LEDGER UNUSABLE: no rules parsed from "
                  f"{sys.argv[2]}; Step 1b did not produce a rule set", file=sys.stderr)
            raise SystemExit(1)
        for r in missing:
            print(f"UNADJUDICATED: {r} was derived for this diff and has no verdict line")
        for r, v in thin:
            print(f"NO-EVIDENCE: {r} is marked {v} with no reason given")
        for r, v, path in stale:
            print(f"UNTOUCHED-CITATION: {r} is marked {v} citing {path}, "
                  f"which this PR does not change")
        if missing or thin or stale:
            print(f"LEDGER INCOMPLETE: {len(want) - len(missing) - len(thin)}/{len(want)} "
                  f"rules adjudicated with evidence, {len(stale)} citing untouched files",
                  file=sys.stderr)
            raise SystemExit(1)
        print(f"LEDGER COMPLETE: {len(want)}/{len(want)} rules adjudicated with evidence")
        raise SystemExit(0)

    diff = open(sys.argv[2], errors="replace").read()

    if mode == "symbols":
        root = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else ".")
        bad = sweep_symbols(diff, root)
        for line, why in bad:
            print(f"UNRESOLVED-IMPORT: {why}")
            print(f"  added by: {line}")
        raise SystemExit(0)

    if mode == "evidence":
        for s in deleted_guard_symbols(diff):
            ev = evidence(s, sys.argv[3:])
            if not ev:
                continue
            print(f"=== {s} on head ===")
            for f, hits in ev:
                keep = [h for h in hits if re.search(
                    r"has_value|!= *nullptr|== *nullptr|optional|Tensor \| None|: *Tensor", h[1])]
                for i, l in (keep or hits)[:6]:
                    print(f"  {pathlib.Path(f).name}:{i}: {l.strip()[:100]}")
            print()
        raise SystemExit(0)

    title = sys.argv[3] if len(sys.argv) > 3 else ""
    try:
        files = parse(diff)
        types = derive(files, title)
    except Exception as exc:
        # Never take the review down, and never silently narrow it: a diff this cannot
        # parse falls back to the full rule set, which is the pre-derivation behaviour.
        print(f"  DERIVATION FAILED ({type(exc).__name__}: {exc}) -- falling back to all rules",
              file=sys.stderr)
        print(f"    [derivation-failed     ] {ALL_RULES}")
        raise SystemExit(0)
    if not types:
        # GitHub refuses a diff over 20000 lines and returns a JSON error body instead.
        # Falling back to the full rule set is safe but backwards: a 74-file PR is exactly the
        # case that needs narrowing. Say so, so the caller can re-run off the file list.
        if diff.lstrip().startswith("{") and "exceeded the maximum number of lines" in diff:
            print("    [diff-too-large        ] " + ALL_RULES)
            print("  NOTE: GitHub would not serve this diff. Re-run with a path list from\n"
                  "        `gh api repos/OWNER/REPO/pulls/N/files --paginate --jq .[].filename`\n"
                  "        or against a local checkout to get a narrowed rule set.",
                  file=sys.stderr)
            raise SystemExit(0)
        print("    [underivable           ] " + ALL_RULES)
        raise SystemExit(0)
    rules = sorted({r for _, rs in types for r in rs.split()})
    print(f"  files={len(files)}  types={len(types)}  rules={len(rules)}/{len(ALL_RULES.split())}")
    for n, rs in types: print(f"    [{n:20s}] {rs}")
