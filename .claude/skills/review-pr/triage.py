#!/usr/bin/env python3
"""Step 1b triage: derive which rule families apply, and collect their evidence.

Replaces Step 3's self-applied prose checklist. Emits only the matching rules so
the model reads ~10 instead of 44.  Conservative: when a type cannot be decided
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
    return t

ALL_RULES = ("A1 A2 A3 B1 B2 B3 B4 B5 B6 B7 C1 C2 C3 C4 D1 D1b D2 D3 D4 D5 D6 D7 D8 "
             "D10 D10b E1 E2 E3 E4 E5 F1 G1 G1b P1 P2 P3 P4 P5 P6 HK1 HK2 HK3 STEP4")

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

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("rules", "evidence"):
        print("usage: triage.py rules <diff> [title]\n"
              "       triage.py evidence <diff> <head-file>...", file=sys.stderr)
        raise SystemExit(2)

    diff = open(sys.argv[2], errors="replace").read()

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
        # Falling back to all 44 rules is safe but backwards: a 74-file PR is exactly the
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
    print(f"  files={len(files)}  types={len(types)}  rules={len(rules)}/44")
    for n, rs in types: print(f"    [{n:20s}] {rs}")
