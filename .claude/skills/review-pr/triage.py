#!/usr/bin/env python3
"""Step 1b triage: derive which rule families apply, and collect their evidence.

Replaces Step 3's self-applied prose checklist. Emits only the matching rules so
the model reads ~12 instead of all of them.  Conservative: when a type cannot be decided
structurally it is INCLUDED, never dropped.
"""
import re, sys, pathlib, collections
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



def conditional_binding(files):
    """D1b: a name assigned only inside an added `if/elif` and used after the block.

    Matching "an `if` exists and something is indented" fired on 68% of 600 PRs, which
    is not a triage. D1b's own text says: assigned on some branches, referenced
    unconditionally later, and exempt when a definitive `else` also assigns it."""
    for f in files.values():
        lines = f["add"]
        for i, l in enumerate(lines):
            m = re.match(r"(\s*)(el)?if\b.*:\s*$", l)
            if not m:
                continue
            base = len(m.group(1))
            assigned, has_else, j = set(), False, i + 1
            while j < len(lines):
                cur = lines[j]
                if not cur.strip():
                    j += 1
                    continue
                ind = len(cur) - len(cur.lstrip())
                if ind <= base:
                    if re.match(r"\s*(else|elif)\b", cur):
                        has_else = has_else or cur.lstrip().startswith("else")
                        j += 1
                        continue
                    break
                a = re.match(r"\s*([A-Za-z_]\w*)\s*=(?!=)", cur)
                if a:
                    assigned.add(a.group(1))
                j += 1
            if has_else or not assigned:
                continue
            after = "\n".join(lines[j:j + 25])
            for name in assigned:
                if re.search(rf"(?<![\w.]){re.escape(name)}\b", after):
                    return True
    return False



C_LIKE = ("cu", "cuh", "h", "hpp", "cpp", "cc")
NOT_A_FIELD = re.compile(r"^\s*(break|continue|return|goto|else|case|default|using|"
                         r"typedef|friend|public|private|protected)\b")


def struct_field_churn(diff_text):
    """A struct gained or lost a field, and the diff touched no ABI assertion.

    aiter pins the layout of every kargs struct a hand-written code object reads:
    `static_assert(sizeof(pa_sparse_prefill_kargs) == 112)` plus one `offsetof` per
    field. Inserting a field in the middle shifts every offset after it, which is a
    build break at best and a silent ABI mismatch with a prebuilt code object at worst.
    aiter#5220 adds two ints between `stride_qo_h` and `stride_kv_page` in the same
    translation unit that carries the assertions, and updates none of them.

    Fires only when the diff itself does NOT touch an assertion -- a PR that moves the
    fields and fixes the table is the correct shape and needs no finding."""
    if re.search(r"(?m)^[+-].*(offsetof\(|static_assert\(\s*sizeof)", diff_text):
        return False
    for blk in re.split(r"(?m)^diff --git ", diff_text)[1:]:
        head = blk.split("\n", 1)[0]
        m = re.match(r"a/(\S+) b/(\S+)", head)
        if not m or m.group(2).rsplit(".", 1)[-1] not in C_LIKE:
            continue
        in_struct = False
        for ln in blk.split("\n"):
            # git puts the enclosing scope after the second @@, and a field added to an
            # existing struct is exactly the case where the `struct X {` line is context
            # the hunk never shows. aiter#5220's hunk header is
            # `@@ -129,6 +129,10 @@ struct pa_sparse_prefill_kargs`.
            hh = re.match(r"@@ [^@]*@@\s*(.*)$", ln)
            if hh:
                in_struct = bool(re.search(r"\b(struct|class)\s+\w+", hh.group(1)))
                continue
            body = ln[1:] if ln[:1] in "+- " else ln
            if re.search(r"\b(struct|class)\s+\w+.*\{", body):
                in_struct = True
            elif re.match(r"\s*\}\s*;", body):
                in_struct = False
            if not (in_struct and ln[:1] in "+-") or ln.startswith(("+++", "---")):
                continue
            if NOT_A_FIELD.match(body) or "(" in body or body.strip().startswith("//"):
                continue
            if re.match(r"\s*(const\s+)?[\w:<>,\*&]+(\s+[\w:<>\*&]+)+"
                        r"(\s*\[[^\]]*\])?\s*;\s*$", body):
                return True
    return False



COLLECTABLE = re.compile(r"^def test_\w+|^\s+def test_\w+|^class Test\w+", re.M)


def uncollectable_test_file(diff_text):
    """A new test_*.py from which pytest collects nothing, and no other new test.

    HK6 asks a new op to ship `op_tests/test_*.py`. A file named that but written as a
    `main()` script with `_check()` helpers satisfies HK6 by name and contributes nothing
    to CI: pytest imports it and collects zero tests. aiter#4821 ships two.

    aiter already contains such files, so the style is tolerated and this is not a defect
    on its own -- it fires only when the PR ships no collectable test at all, which is the
    case where HK6 reads as satisfied and the code is in fact untested. 3.8% of the 600-PR
    corpus."""
    collectable = False
    candidates = []
    for blk in re.split(r"(?m)^diff --git ", diff_text)[1:]:
        head = blk.split("\n", 1)[0]
        m = re.match(r"a/(\S+) b/(\S+)", head)
        if not m:
            continue
        path = m.group(2)
        base = path.rsplit("/", 1)[-1]
        if not (base.startswith("test_") and base.endswith(".py")):
            continue
        added = "\n".join(l[1:] for l in blk.split("\n")
                           if l.startswith("+") and not l.startswith("+++"))
        if COLLECTABLE.search(added):
            collectable = True
        elif "\nnew file mode" in blk and len(added) > 300:
            candidates.append(path)
    return [] if collectable else candidates



def is_comment_line(line):
    s = line.strip()
    return (not s) or s.startswith(("#", "//", "/*", "*", '"""', "'''"))


def comment_dominated(diff_text, threshold=0.10, floor=60):
    """The non-comment lines of a diff that is almost entirely comments.

    Rare -- 2 of 600 open PRs -- and worth having for the leverage: aiter#4062 changes
    7963 lines across 252 files of which 4 are code, and aiter#4061 changes 6328 of which
    128 are. Those few lines are the entire review; everything else is prose. A reviewer
    facing an 8000-line "condense comments" diff either reads none of it or reads it for
    hours, and both of those miss the four lines."""
    changed = []
    cur = None
    # A `/* ... */` block whose continuation lines carry no leading `*` reads as code
    # line by line. aiter#4061's rewritten quant_utils.cuh header is exactly that, and
    # every one of its prose lines was listed as something to review.
    in_block = {"+": False, "-": False}
    for ln in diff_text.splitlines():
        m = re.match(r"^diff --git a/\S+ b/(\S+)", ln)
        if m:
            cur = m.group(1)
            in_block = {"+": False, "-": False}
            continue
        if ln[:1] not in "+-" or ln.startswith(("+++", "---")):
            continue
        sign, body = ln[0], ln[1:]
        was_in = in_block[sign]
        opens = body.count("/*")
        closes = body.count("*/")
        if opens > closes:
            in_block[sign] = True
        elif closes and was_in:
            in_block[sign] = False
        # Marked as prose, not dropped: the floor and the ratio are about the whole
        # diff, and dropping these left a 6000-line comment rewrite looking like a
        # two-line diff that fell under the floor.
        changed.append((cur, sign, body, was_in))
    if len(changed) < floor:
        return None
    code = [(p_, sg, b) for p_, sg, b, prose in changed
            if not prose and not is_comment_line(b)]
    if len(code) / len(changed) > threshold:
        return None
    # A line whose only change is its trailing comment is not a code change.
    # aiter#4062 reads as four code lines until the trailing comments are stripped,
    # and then as zero, which is the truthful answer for a comment-only PR.
    removed = collections.Counter()
    added = collections.Counter()
    for path, sign, body in code:
        key = (path, strip_trailing_comment(body))
        (removed if sign == "-" else added)[key] += 1
    real = []
    for path, sign, body in code:
        key = (path, strip_trailing_comment(body))
        other = added if sign == "-" else removed
        if other[key]:
            other[key] -= 1
            continue
        real.append((path, sign, body))
    return len(changed), real


def strip_trailing_comment(line):
    out, quote = [], None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                out.append(line[i:i + 2])
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" or line[i:i + 2] == "//":
            break
        out.append(ch)
        i += 1
    return "".join(out).rstrip()



LAUNCH_KNOB = re.compile(r"\b(waves_per_eu|matrix_instr_nonkdim|kpack|num_stages|"
                         r"num_warps|num_ctas)\s*=\s*\d+")
FROM_CONFIG = re.compile(r"=\s*(config|cfg|conf|tune|tuned|kwargs|\*\*|[A-Z_]+\[)")


def hardcoded_launch_knob(files):
    """A launch knob set to a literal in kernel source rather than in the tuned config.

    T4 covers a knob hardcoded ACROSS archs. aiter#5137's revert was a single-arch case
    and a different failure: `waves_per_eu=2` written into pa_decode.py with a six-line
    justification, reverted with "change the config file jsons instead of hardcoding it
    in the code". The config system exists so that a tuning sweep and the code disagree
    loudly; a literal in the source is invisible to the sweep, which will either overwrite
    it or fight it.

    Config files and tests are exempt -- a literal in a JSON/CSV is the config system
    working, and a test pinning a knob is pinning it on purpose."""
    for path, f in files.items():
        if not path.endswith(".py"):
            continue
        if "/configs/" in path or path.startswith(("op_tests/", "tests/")):
            continue
        for line in f["add"]:
            if line.strip().startswith("#"):
                continue
            m = LAUNCH_KNOB.search(line)
            if m and not FROM_CONFIG.search(line):
                return True
    return False


def derive(files, title="", raw_diff=""):
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
        hit("downstream-op", "E4 E5 A2")   # A2 is the same shared-path condition
    if any("codegen" in p or p.startswith("csrc/cpp_itfs/") or p.endswith("Makefile")
           for p in paths):
        hit("codegen-buildtool", "A1 D5 B6")
    if re.search(r"@compile_ops|torch\.library\.custom_op", add):
        hit("new-compile-op", "D7 D6")
    if re.search(r"mutates_args|torch_compile_guard|register_fake|_fake\b|gen_fake|abstract_impl", add):
        hit("compile-contract", "D6 D7")
    # Only code lines. A comment mentioning `.contiguous()` is not a removed invariant;
    # aiter#4062 condenses comments across 252 files and tripped this on prose alone.
    dele_code = "\n".join(l for l in dele.splitlines() if not is_comment_line(l))
    if re.search(r"(assert |torch\.zeros|\.contiguous\(\)|AITER_CHECK|TORCH_CHECK)",
                 dele_code):
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
    # Any new module-level env knob, not only AITER_-prefixed ones. The prefix was in the
    # pattern, so aiter#5157 -- which adds six permanent switches named KR_TAILSPLIT,
    # KR_TS_OCC, KR_TS_MAX_SEG, KR_TS_MIN_SEG, KR_TS_CAP_BLOCKS, KR_TS_TRIM_GRID -- derived
    # no env-var family at all, and HK9 exists precisely because aiter rejects permanent
    # env flags (AITER_MOE_FORCE_BF16_ACT in #3593 was reverted by #4225).
    # HK9 is about a permanent RUNTIME knob. A benchmark or profiling script setting
    # PROF_WARMUP/PROF_ITERS is not one, and firing there teaches the reviewer that the
    # family is noise (aiter#3976 adds three, all in op_tests/flydsl_tests/).
    runtime_env = "\n".join(
        l for p, f in files.items() if not p.startswith(("op_tests/", "tests/"))
        and "bench" not in p and "profile" not in p for l in f["add"])
    if re.search(r'^\w+\s*=.*os\.environ\.(get|\[)|^\s*os\.environ\.get\(',
                 runtime_env, re.M) \
       or re.search(r'os\.environ\.get\(\s*["\']AITER_', runtime_env):
        hit("new-env-var", "HK9 D2")
    # --- rules whose trigger is textual but which no family emitted ---------------
    # Sixteen documented rules were derivable by nothing at all, so every review read
    # past them 100% of the time. Their own trigger text is structural -- "`# TODO` on a
    # `+` line", "develop=True in added code" -- it was simply never wired up.
    if any(p.endswith(".sh") or re.search(r"(^|/)(runperf|test_local_)", p) for p in paths):
        hit("temp-script", "HK1")
    if re.search(r"sys\.path\.(insert|append)\(", add):
        hit("syspath-mutation", "HK3")
    if re.search(r"#\s*(TODO|FIXME)|raise NotImplementedError|^\s*pass\s*$", add, re.M):
        hit("incomplete-code", "HK7")
    if re.search(r"@compile_ops\([^)]*develop\s*=\s*True", add) or "develop=True" in add:
        hit("develop-flag", "HK8")
    if any(p.startswith(("op_tests/", "tests/")) for p in paths) and \
       re.search(r"\.to\(torch\.float32\)|\.double\(\)|\.float\(\)", add):
        hit("test-reference-dtype", "HK10")
    if any(re.search(r"requirements.*\.txt$|setup\.py$|pyproject\.toml$", p) for p in paths):
        hit("new-dependency", "HK11")
    # B5 names the signal itself: a new tl.constexpr bool gating a validity check,
    # e.g. CHECK_NEG_ONE_SENTINEL, CHECK_BOUNDS.
    if re.search(r"\b(CHECK|VALIDATE|VERIFY|ASSERT|GUARD|BOUNDS|SENTINEL|MASK|CLAMP)\w*"
                 r"\s*:\s*tl\.constexpr", add) or \
       re.search(r"\b\w*(CHECK|BOUNDS|SENTINEL|VALID)\w*\s*=\s*(True|False)\b", add):
        hit("constexpr-guard-off", "B5")
    if re.search(r"(torch\.(float16|bfloat16|float8\w*)|dtype\s*=\s*torch\.\w+)", add) and \
       not re.search(r"\.dtype\s*==|is_floating_point|\.dtype\b.*(in|==)", add):
        hit("dtype-assumed", "C3")
    if hardcoded_launch_knob(files):
        hit("hardcoded-launch-knob", "T8")
    if conditional_binding(files):
        hit("conditional-binding", "D1b")
    if re.search(r"def \w+\([^)]*\w+\s*=\s*(None|False|0|1|\"\")", add):
        hit("defaulted-param", "E2")
    if any("bridge" in p or "plugin" in p or "_ext" in p for p in paths):
        hit("plugin-bridge", "E3")
    if re.search(r"num_heads|\btp_size\b|tensor_parallel|\bTP\b", add):
        hit("tp-shapes", "P4")
    if len({p.split("/")[0] for p in paths}) >= 3:
        hit("scattered-diff", "HK2")
    if any(f["new"] and re.search(r"(_v\d+|_opt|_variant|_fast|_new)\.(py|cu|hip)$", p)
           for p, f in files.items()):
        hit("nth-variant", "HK5")
    if uncollectable_test_file(raw_diff):
        hit("uncollectable-test", "HK12b")

    if any(p.startswith(("aiter/ops/triton/",)) or "/triton/" in p or
           "_triton_kernels/" in p or "_gluon_kernels/" in p or "/gluon/" in p
           for p in paths):
        t.extend(triton_families(add))
    return t

# Rules deliberately absent from every derivation, with the reason. A rule not on this
# list and not emitted by any family is unreachable: documented, and read by nobody, ever.
# tests/test_review_skill.py fails on that, because the failure is invisible from inside a
# review -- the ledger only sees rules that were derived.
UNREACHABLE_BY_DESIGN = {
    # scan_index_width.py runs in Step 1 and puts its candidates in front of the reviewer
    # before the rule pass. The prose would be a second, weaker copy of a result already
    # on screen; a 14-PR run showed the prose version catching 0 of 3 known defects.
    "D9": "scanner-backed: scan_index_width.py reports it in Step 1",
    # HK12's trigger is where a file LIVES against what .github/ actually scans, which the
    # diff alone cannot answer. `triage.py citest` computes it in Step 1b and writes
    # ci_coverage.txt; deriving a rule id from the diff would be guessing.
    "HK12": "evidence-backed: triage.py citest writes ci_coverage.txt in Step 1b",
    # Whether a struct's layout is pinned is a fact about the tree. Deriving D11 from the
    # diff alone fired on aiter#5223, six headers with no assertion anywhere near them.
    "D11": "evidence-backed: triage.py structabi writes struct_abi.txt in Step 1b",
}

ALL_RULES = ("A1 A2 A3 B1 B2 B3 B4 B5 B6 B7 C1 C2 C3 C4 D1 D1b D2 D3 D4 D5 D6 D7 D8 "
             "D10 D10b E1 E2 E3 E4 E5 F1 G1 G1b P1 P2 P3 P4 P5 P6 HK1 HK2 HK3 "
             "T1 T2 T3 T4 T5 T6 T8 D11 HK12 HK12b STEP4")

GUARD_NOISE = frozenset("""not is None True False and or in if else self len int float
str bool tuple list dict set all any isinstance type return raise sizeof static_assert
torch aiter np dtypes""".split())


def deleted_guard_symbols(diff_text):
    """Symbols named by a guard the diff removes.

    The `invariant-removed` family fires on assert / *_CHECK / `.contiguous()` /
    `torch.zeros` in deleted lines, but this collected symbols only from the first three.
    A removed `.contiguous()` normalisation -- the most common shape of the finding, and
    the one where the answer is usually "it moved" -- produced no evidence at all, so the
    reviewer was back to grepping by hand, which is what this exists to stop.
    aiter#5235 relocates two of them and the collector said nothing."""
    syms = set()
    for ln in diff_text.splitlines():
        if not ln.startswith("-") or ln.startswith("---"):
            continue
        body = ln[1:]
        if re.search(r"\b(assert|AITER_CHECK|TORCH_CHECK)\b", body):
            # The subject of the guard, whatever shape it takes. Matching only
            # `X is not None` and `X->` extracted nothing from 51 of the 57 corpus PRs
            # that delete a guard: `assert causal, "..."`, `assert WQ.dtype == fp8`,
            # `assert num_head_qo % 16 == 0`, `assert gfx_version in (...)` are all
            # guards whose subject is simply the first identifier after the keyword.
            cond = re.sub(r"^.*?\b(?:assert|AITER_CHECK|TORCH_CHECK)\b\s*\(?", "", body)
            cond = cond.split(",")[0]
            for m in re.finditer(r"\b([a-zA-Z_]\w*)\b", cond):
                name = m.group(1)
                if name in GUARD_NOISE:
                    continue
                syms.add(name)
                break                      # the subject, not every token
            for m in re.finditer(r"\b([a-zA-Z_]\w*)\s+is\s+not\s+None", body):
                syms.add(m.group(1))
            for m in re.finditer(r"\b([a-zA-Z_]\w*)->", body):
                syms.add(m.group(1))
        if re.search(r"\.contiguous\(\)|torch\.zeros|\.zero_\(\)", body):
            # the thing being normalised: `self.w2 = w2 if ... else w2.contiguous()`
            m = re.match(r"\s*(?:self\.)?([a-zA-Z_]\w*)\s*=", body)
            if m:
                syms.add(m.group(1))
            for m in re.finditer(r"\b([a-zA-Z_]\w*)\.(?:contiguous|zero_)\(\)", body):
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
    """(module, [names], line) for every import line the diff ADDS.

    Relative imports are resolved against the file they appear in, so `from .chip_info
    import get_asic_revision` inside aiter/jit/utils/asm_guard.py becomes
    aiter.jit.utils.chip_info. Skipping them entirely left the one case where an
    unresolvable import is worst -- a Tier-1 PR wiring a new module into
    aiter/__init__.py, where a bad relative import breaks `import aiter` outright."""
    out = []
    cur_pkg = None
    for ln in diff_text.splitlines():
        m = re.match(r"^diff --git a/\S+ b/(\S+)", ln)
        if m:
            path = m.group(1)
            # `is_pkg_init` matters: for aiter/ops/flydsl/__init__.py the containing
            # package IS aiter.ops.flydsl, while for aiter/ops/flydsl/x.py it is the
            # parent of x. Stripping `.__init__` and then dropping a component as well
            # resolved `from .kernels.foo import ...` in aiter/ops/flydsl/__init__.py to
            # aiter.ops.kernels.foo -- a module that does not exist -- and reported the
            # PR that wrote it (aiter#4515).
            cur_pkg = path[:-3].replace("/", ".") if path.endswith(".py") else None
            cur_is_pkg_init = bool(cur_pkg) and cur_pkg.endswith(".__init__")
            if cur_is_pkg_init:
                cur_pkg = cur_pkg[: -len(".__init__")]
            continue
        if not ln.startswith("+") or ln.startswith("+++"):
            continue
        if cur_pkg is None:
            # Not a .py file. An `import` line inside docs/tutorials/add_new_op.rst is a
            # worked example, not code, and `from ..jit.core import compile_ops` there
            # was reported as a missing module on two PRs.
            continue
        body = ln[1:].strip()
        rel = re.match(r"from\s+(\.+)([\w.]*)\s+import\s+(.+)$", body)
        if rel:
            dots, tail, names = rel.group(1), rel.group(2), rel.group(3)
            parts = cur_pkg.split(".") if cur_is_pkg_init else cur_pkg.split(".")[:-1]
            up = len(dots) - 1
            if up > len(parts):
                continue                              # escapes the tree; leave alone
            base = parts[: len(parts) - up] if up else parts
            mod = ".".join(base + ([tail] if tail else []))
            names = names.split("#")[0].replace("(", "").replace(")", "")
            out.append((mod, [n.strip().split(" as ")[0]
                              for n in names.split(",") if n.strip()], body))
            continue
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
        elif ln.startswith(("rename to ", "copy to ")):
            # A moved file has no `new file mode`; git writes `rename from`/`rename to`
            # and, at similarity 100%, no hunks at all. Counting only new-file markers
            # made every reorganisation PR report its own destination paths as missing
            # modules -- aiter#5254 moves 74 files under a new linear_attention/ package
            # with 53 renames and 5 new files, and produced 83 UNRESOLVED-IMPORT lines,
            # every one of them pointing at a path the PR itself creates.
            added_paths.add(ln.split(" to ", 1)[1].strip())
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
        # No `mod.startswith(".")` here: added_imports resolves relative imports against
        # the file they appear in, so nothing dotted-relative reaches this point.
        if STDLIB_OR_THIRD_PARTY.match(mod):
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
            # `from pkg import sub` where THIS PR adds pkg/sub.py. The added-module check
            # above only covers the dotted module itself, so a PR that adds a submodule
            # and imports it by name was reported against its own new file -- aiter#5157
            # adds aiter/ops/triton/attention/kr_ua.py and imports `kr_ua` from the
            # package.
            if f"{rel}/{n}.py" in added_paths or f"{rel}/{n}/__init__.py" in added_paths:
                continue
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


def expand_rules(rules_text, rules_md):
    """Full text of exactly the derived rules, each under its category heading.

    The derivation cut which rules the reviewer is TOLD to check from 44 to a median
    of 12, but every rule's text still shipped in SKILL.md and loaded on every review:
    383 lines of rule bodies against a median need of 103. Telling a reader to ignore
    73% of what is in front of them is not the same as not putting it there."""
    want = set(expected_rules(rules_text))
    lines = rules_md.split("\n")
    starts = [(i, m.group(1)) for i, l in enumerate(lines)
              if (m := re.match(r"\*\*([A-Z]+\d+[a-z]?) — ", l))]
    heading, head_at = None, {}
    for i, l in enumerate(lines):
        if l.startswith("### "):
            heading = l
        head_at[i] = heading
    # Housekeeping rules are documented as table rows, not `**HKn — **` blocks, and STEP4
    # names Step 4 rather than a rule body. Treating either as a missing body reported 277
    # of 600 PRs as drifted when nothing had drifted.
    table_rows, table_head = {}, []
    # Both assignments of this flag are equivalent mutants today and that is contingent,
    # not safe: the flag is closed by the first `###` before any table row appears, and
    # nothing after Housekeeping carries a table, so flipping either changes no output.
    # Move a table into the span this scan walks and the equivalence ends -- rows from
    # another section would be emitted as housekeeping rules. No test can distinguish the
    # two today; writing one that passes regardless would only have flattered the score.
    in_hk = False
    for i, l in enumerate(lines):
        if l.startswith("### Housekeeping"):
            in_hk = True
            continue
        if in_hk and l.startswith("### "):
            in_hk = False
        if not in_hk:
            continue
        if l.startswith("|") and len(table_head) < 2:
            table_head.append(l)
        for rid in re.findall(r"\b(HK\d+[a-z]?)\b", l):
            if l.startswith("|"):
                table_rows.setdefault(rid, l)

    out, emitted, last_head = [], [], None
    for (i, rid), (j, _) in zip(starts, starts[1:] + [(len(lines), "")]):
        if rid not in want:
            continue
        k = i
        while k < j and not lines[k].startswith(("### ", "## ")):
            k += 1
        if head_at[i] and head_at[i] != last_head:
            last_head = head_at[i]
            out.append(head_at[i])
            out.append("")
        out.extend(lines[i:k])
        emitted.append(rid)

    hk = [r for r in sorted(want) if r in table_rows]
    if hk:
        out += ["", "### Housekeeping (quick scan)", ""] + table_head
        out += [table_rows[r] for r in hk]
        emitted += hk

    unresolved = sorted(want - set(emitted) - {"STEP4"})
    return "\n".join(out).rstrip() + "\n", emitted, unresolved


# --------------------------------------------------------------- answers ledger
ANSWER_KEY = re.compile(r"^\s*(Q[1-5]|BLIND)\s*[:\-]\s*(.+?)\s*$", re.I)
MIN_ANSWER = 40
SURFACE = re.compile(
    r"^(n/?a|none|no|yes|ok|nothing|same|unchanged|see above|tbd|clean|fine|-+)\.?$", re.I)


DIAGNOSTIC_KEY = re.compile(r"^\s*([1-6])\s*[:\-]\s*(.+?)\s*$")


def audit_diagnostic(text):
    """Step 6's six structural checks, as lines rather than as intentions.

    Check 1 is now a pointer at $WORK/symbols.txt rather than a request to grep every new
    symbol by hand -- the sweep already ran. The other five have no tool behind them, which
    is exactly why they need to leave a trace."""
    seen = {}
    for ln in text.splitlines():
        m = DIAGNOSTIC_KEY.match(ln)
        if m:
            seen[m.group(1)] = m.group(2).strip()
    want = [str(i) for i in range(1, 7)]
    missing = [q for q in want if q not in seen]
    thin = [q for q in want
            if q in seen and (len(seen[q]) < MIN_ANSWER or SURFACE.match(seen[q]))]
    return missing, thin


def audit_answers(text):
    """Step 2's five questions and Step 7.5's one, as an artifact.

    They were prose asking the model to fill in `_Answer:_` inline, which leaves nothing
    behind: a review that skipped them is textually identical to one that did them. This
    is the same failure D9 had before its scan moved into Step 1, and the same one the
    rule pass had before the ledger.

    A bare "no" to the blind-spot question is the answer that costs nothing to give and
    is worth nothing to receive, so short and formulaic answers are rejected. This checks
    that an answer was written, not that it is right -- being wrong is visible in the
    card, being absent is not."""
    seen = {}
    for ln in text.splitlines():
        m = ANSWER_KEY.match(ln)
        if m:
            seen[m.group(1).upper()] = m.group(2).strip()
    want = ["Q1", "Q2", "Q3", "Q4", "Q5", "BLIND"]
    missing = [q for q in want if q not in seen]
    thin = [q for q in want
            if q in seen and (len(seen[q]) < MIN_ANSWER or SURFACE.match(seen[q]))]
    return missing, thin



# ------------------------------------------------------------------- mapping
def family_mapping():
    """(family, rules) for every family `derive` can emit, read out of this file.

    Step 3 carried this table by hand and it drifted: it claimed D9 was derived (D9 is
    scanner-backed and deliberately is not) and omitted 21 rules that are, including all
    six Triton rules. A mapping a reviewer is told not to apply by hand has no business
    being maintained by hand either -- it is generated now, and tests/ fails if the
    committed copy and this function disagree."""
    src = pathlib.Path(__file__).read_text()
    out, seen = [], set()
    for m in re.finditer(r'(?:hit|fams\.append)\(\(?"([\w-]+)",\s*"([^"]+)"\)?\)', src):
        fam, rules = m.group(1), m.group(2)
        if fam in seen:
            continue
        seen.add(fam)
        out.append((fam, rules))
    return out


def render_mapping():
    rows = family_mapping()
    width = max(len(f) for f, _ in rows)
    lines = ["# Family -> rule mapping",
             "",
             "Generated by `triage.py mapping`. Do not edit: `tests/test_review_skill.py`",
             "fails when this file and the deriver disagree. Regenerate with",
             "`python3 triage.py mapping > MAPPING.md`.",
             "",
             f"{len(rows)} families, {len(set(r for _, rs in rows for r in rs.split()))} "
             f"distinct rule ids.",
             "",
             "| family | rules |",
             "|---|---|"]
    for fam, rules in rows:
        lines.append(f"| `{fam}`{' ' * (width - len(fam))} | {rules} |")
    excl = sorted(UNREACHABLE_BY_DESIGN)
    if excl:
        lines += ["", "Documented but deliberately never derived:", ""]
        for rid in excl:
            lines.append(f"- **{rid}** — {UNREACHABLE_BY_DESIGN[rid]}")
    return "\n".join(lines) + "\n"



# ------------------------------------------------------------- test quality
ASSERT_PRIM = re.compile(r"\bassert\b|assert_close|allclose|checkAllclose|"
                         r"pytest\.raises|\.approx|np\.testing|torch\.testing")
TOL = re.compile(r"(?:atol|rtol)\s*=\s*(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
SHAPE = re.compile(r"\b(?:M|N|K|batch|num_tokens|seqlen|seq_len)\s*[=:]\s*(\d+)")


def test_quality(diff_text):
    """Facts about the tests a diff adds, for the reviewer to judge -- not a verdict.

    "This test asserts nothing" cannot be decided from a diff: assertions routinely live
    in a shared helper (`run_fp8(..., verify=True)`), so a rule that fires on a test body
    with no `assert` is wrong more often than right, and over the 600-PR corpus the strict
    version -- a whole new test file with no assertion primitive anywhere -- fires on 2,
    one of which is a benchmark. Neither is a rule worth having.

    What IS decidable is what the test contains, so this prints that: how many assertion
    primitives, which tolerances, which shapes. Rule P2 and Step 6's check 5 are the
    judgement; this is the evidence they need in front of them, the same way evidence.txt
    serves the removed-guard rules."""
    rows = []
    for blk in re.split(r"(?m)^diff --git ", diff_text)[1:]:
        head = blk.split("\n", 1)[0]
        m = re.match(r"a/(\S+) b/(\S+)", head)
        if not m:
            continue
        path = m.group(2)
        if not (path.endswith(".py") and
                (path.startswith(("op_tests/", "tests/")) or "test" in path.rsplit("/", 1)[-1])):
            continue
        added = "\n".join(l[1:] for l in blk.split("\n")
                           if l.startswith("+") and not l.startswith("+++"))
        if not added.strip():
            continue
        new_file = "\nnew file mode" in blk
        base = path.rsplit("/", 1)[-1]
        is_bench = (base.startswith(("bench", "profile")) or "benchmark" in path
                    or "op_benchmarks" in path)
        tests = re.findall(r"^def (test_\w+)", added, re.M)
        asserts = len(ASSERT_PRIM.findall(added))
        tols = sorted({t for t in TOL.findall(added)}, key=lambda x: -float(x))
        shapes = sorted({int(x) for x in SHAPE.findall(added)})
        if not (tests or asserts or tols or shapes):
            continue
        rows.append({"path": path, "new": new_file, "tests": tests, "bench": is_bench,
                     "asserts": asserts, "tols": tols, "shapes": shapes})
    return rows


def render_test_quality(rows):
    out = []
    for r in rows:
        tag = "new file" if r["new"] else "modified"
        # A benchmark with no assertions is a benchmark. Telling the reviewer the check
        # "may be in a helper" about a bench_*.py is misleading, and 68 of 682 rows over
        # the corpus are exactly that.
        if r["bench"]:
            tag += ", benchmark"
        out.append(f"{r['path']}  ({tag})")
        out.append(f"  test functions added : {len(r['tests'])}"
                   f"{'  ' + ', '.join(r['tests'][:4]) if r['tests'] else ''}")
        note = ""
        if r["asserts"] == 0:
            note = ("   <- expected for a benchmark" if r["bench"] else
                    "   <- none in the added lines; the check may be in a helper, or"
                    " there may be none")
        out.append(f"  assertion primitives : {r['asserts']}{note}")
        if r["tols"]:
            worst = float(r["tols"][0])
            note = "   <- loose for a kernel comparison" if worst >= 1e-1 else ""
            out.append(f"  tolerances           : {', '.join(r['tols'])}{note}")
        if r["shapes"]:
            note = ("   <- every shape is small; P2 asks for production sizes"
                    if max(r["shapes"]) <= 16 else "")
            out.append(f"  shapes               : "
                       f"{', '.join(str(x) for x in r['shapes'][:12])}{note}")
        out.append("")
    return "\n".join(out)



# ------------------------------------------------------------------- twins
def near_duplicates(diff_text, root, threshold=0.60):
    """New files that are largely a copy of a file already in the tree.

    Step 6 asks the reviewer to find "mirrored code -- fwd/bwd, v2/v3, prefill/decode,
    gfx942/gfx950 -- and compare it field by field", which is the AI kernel bug signature.
    Finding the twin was left to the reader. Over the 600-PR corpus 3% of PRs add a file
    at least 60% identical to an existing one: fused_gemm_a16w16_copy_x.py against
    _quant_x.py at 68%, test_opus_gmem_gfx1100.cu against _gfx1201.cu at 75%.

    This names the pair and the ratio. It does not judge: a deliberate arch-specific
    variant is the normal shape here, and the finding is an asymmetry BETWEEN the twins,
    which needs a human diff of the two."""
    added = {}
    for blk in re.split(r"(?m)^diff --git ", diff_text)[1:]:
        head = blk.split("\n", 1)[0]
        m = re.match(r"a/(\S+) b/(\S+)", head)
        if not m or "\nnew file mode" not in blk:
            continue
        path = m.group(2)
        if not path.endswith((".py", ".cu", ".cuh", ".h", ".hpp", ".cpp")):
            continue
        lines = {l[1:].strip() for l in blk.split("\n")
                 if l.startswith("+") and not l.startswith("+++") and len(l.strip()) > 21}
        if len(lines) >= 40:
            added[path] = lines
    if not added:
        return []
    out = []
    for path, lines in added.items():
        best, best_path = 0.0, None
        for cand in pathlib.Path(root).rglob("*"):
            if cand.suffix not in (".py", ".cu", ".cuh", ".h", ".hpp", ".cpp"):
                continue
            rel = str(cand.relative_to(root))
            if rel == path or ".git/" in rel or not cand.is_file():
                continue
            try:
                other = {l.strip() for l in cand.read_text(errors="replace").splitlines()
                         if len(l.strip()) > 21}
            except OSError:
                continue
            if len(other) < 40:
                continue
            ratio = len(lines & other) / len(lines)
            if ratio > best:
                best, best_path = ratio, rel
        if best >= threshold:
            out.append((path, best_path, best))
    return out



# --------------------------------------------------------------- ci coverage
def ci_test_scope(root):
    """(patterns, note) describing which test paths a CI job actually scans.

    Derived from .github/, never hardcoded: the shard script is where the truth is and a
    copy of it here would drift the way Step 3's rule table did. Reads the `find` lines in
    split_tests.sh, so a change to its maxdepth or its directories changes this answer."""
    root = pathlib.Path(root)
    scope = []
    sh = root / ".github" / "scripts" / "split_tests.sh"
    if sh.is_file():
        text = sh.read_text(errors="replace")
        for m in re.finditer(r'find\s+"?(\$?\w+|[\w/]+)"?\s+(-maxdepth\s+(\d+)\s+)?'
                             r"-name\s+'([^']+)'", text):
            var, _, depth, pat = m.groups()
            scope.append((var, int(depth) if depth else None, pat))
    dirs = dict(re.findall(r'TEST_DIR="([\w/]+)"', sh.read_text(errors="replace"))
                and [] or [])
    return scope


def uncovered_test_paths(diff_text, root):
    """Test files the diff adds that no CI job will run, and why.

    aiter runs op_tests/*.py at maxdepth 1 plus all of op_tests/triton_tests/. Everything
    else is either label-gated or scanned by nothing: op_tests/multigpu_tests/ needs the
    `multigpu` label and is skipped by default, and op_tests/flydsl_tests/ appears in no
    workflow at all. A test that lands there is not a weak test -- it is not a test, it is
    a script that happens to be committed.

    aiter#4821 ships two files under op_tests/flydsl_tests/. They would not run even if
    every function in them were a `def test_`."""
    root = pathlib.Path(root)
    covered_top, covered_trees, label_gated = set(), [], []
    sh = root / ".github" / "scripts" / "split_tests.sh"
    if sh.is_file():
        t = sh.read_text(errors="replace")
        if re.search(r'TEST_DIR="op_tests"', t) and "-maxdepth 1" in t:
            covered_top.add("op_tests")
        for m in re.finditer(r'TEST_DIR="(op_tests/[\w/]+)"', t):
            covered_trees.append(m.group(1))
    # A path MENTIONED in a workflow is not a path that gets RUN. pr-title-tags.yaml
    # names op_tests/flydsl_tests/ to decide a label, and counting that as coverage made
    # this report aiter#4821 as covered -- the exact case it exists to catch. Only lines
    # that execute something count.
    RUNS = re.compile(r"pytest|python3?\s|run_tests\.sh|split_tests\.sh|\.sh\b")
    wf = root / ".github" / "workflows"
    if wf.is_dir():
        for f in wf.glob("*.y*ml"):
            for line in f.read_text(errors="replace").splitlines():
                if not RUNS.search(line):
                    continue
                for m in re.finditer(r"(op_tests/[\w/]+)", line):
                    d = m.group(1).rstrip("/")
                    if not (root / d).is_dir():
                        d = d.rsplit("/", 1)[0]
                    if (root / d).is_dir() and d not in covered_trees:
                        covered_trees.append(d)
    # Label-gated directories: named in a workflow that also gates on a label.
    for f in (wf.glob("*.y*ml") if wf.is_dir() else []):
        txt = f.read_text(errors="replace")
        if "labels.*.name" not in txt:
            continue
        for m in re.finditer(r"run_(\w+)_tests", txt):
            d = root / "op_tests" / f"{m.group(1)}_tests"
            if d.is_dir():
                rel = str(d.relative_to(root))
                label_gated.append(rel)
                if rel in covered_trees:
                    covered_trees.remove(rel)
    out = []
    for blk in re.split(r"(?m)^diff --git ", diff_text)[1:]:
        head = blk.split("\n", 1)[0]
        m = re.match(r"a/(\S+) b/(\S+)", head)
        if not m or "\nnew file mode" not in blk:
            continue
        path = m.group(2)
        base = path.rsplit("/", 1)[-1]
        if not (base.startswith("test_") and base.endswith(".py")):
            continue
        parent = path.rsplit("/", 1)[0]
        if parent in covered_top:
            continue
        if any(path.startswith(d.rstrip("/") + "/") for d in covered_trees):
            continue
        if any(path.startswith(d.rstrip("/") + "/") for d in label_gated):
            out.append((path, "runs only when the `multigpu` label is set; skipped by "
                              "default, so a merge without the label never runs it"))
            continue
        out.append((path, "no CI job scans this directory -- op_tests is shard-scanned at "
                          "maxdepth 1 and op_tests/triton_tests recursively, nothing else"))
    return out



# --------------------------------------------------------------- perf claims
PERF_NUM = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*"
                      r"(x\b|%|\bus\b|\b[mu]s\b|TFLOPS?\b|GB/s|tok/s)", re.I)
HW_MODEL = re.compile(r"\b(MI\d+\w*|gfx\d+|CDNA\d*|RDNA\d*|fp\d+|bf\d+|int\d+|e\dm\d)",
                      re.I)
# aiter PR descriptions are partly Chinese; an English-only word list marks a table that
# does name its baseline as one that does not.
BASELINE = re.compile(r"\bvs\.?\b|versus|baseline|\bbefore\b|\bafter\b|\bmain\b|torch|"
                      r"\bck\b|hipblas|current|previous|reference|speedup|faster|slower|"
                      r"improv|regress|→|->|"
                      r"对比|基线|之前|之后|原来|改进|提升|加速|优化前|优化后|提速", re.I)


SHARE = re.compile(r"%\s*(of|prefill|decode|busy|utili|occupan|GPU time|end of|elements)"
                   r"|of\s+(GPU|total|kernel)\s+time", re.I)


def perf_claims(body):
    """Numeric performance claims in the PR description, and whether each names what it
    is measured against.

    Scoped to the description on purpose. Kernel comments are full of `4x DS_READ`,
    `<4 x i32>`, `num_tokens x 384 x 7168` and `5% of elements`, so extracting claims from
    code produced more noise than signal over the 600-PR corpus -- 61% of what it called
    unbaselined claims were shapes, vector widths and error bounds. A perf claim lives in
    the description; that is where P1 and P3 ask for it, and that is where this looks."""
    lines = (body or "").splitlines()
    # A markdown table states its baseline in the header: `| batch | before | after |
    # speedup |`. Judging each row alone called every row of aiter#4443's table
    # unbaselined when the column names are the baseline.
    table_header = None
    rows = []
    # `8x MI355X` is a GPU count, not a speedup. Substituting the model name away first
    # left a bare `8x` looking like one.
    COUNT = re.compile(r"\b\d+\s*[x×]\s*(?=MI\d|gfx|GPU|node|card|rank|device|CU\b|"
                       r"warp|wave|stage|shard)", re.I)
    for line in lines:
        line = COUNT.sub(" ", line)
        is_row = line.strip().startswith("|") and line.count("|") >= 3
        if is_row and BASELINE.search(line) and not PERF_NUM.search(HW_MODEL.sub(" ", line)):
            table_header = line
            continue
        if not is_row:
            table_header = None
        stripped = HW_MODEL.sub(" ", line)
        if not PERF_NUM.search(stripped):
            continue
        # A share is not a speedup: "33.07% of GPU time", "81.8% prefill" describe where
        # the time goes, and asking what they are measured against is nonsense.
        if SHARE.search(line) and not re.search(r"\d\s*x\b", stripped):
            continue
        nums = [f"{a}{b}" for a, b in PERF_NUM.findall(stripped)]
        # A signed percentage is a delta, and a delta's other side is "without this
        # change" -- that is a stated baseline in ordinary English. `+8.64% end-to-end`
        # needs no interrogation; a bare `198 TFLOPS` does.
        signed = bool(re.search(r"[+\-−]\s*\**\d+(?:\.\d+)?\s*%", line))
        based = (bool(BASELINE.search(line)) or signed
                 or (is_row and table_header is not None))
        rows.append((line.strip()[:100], nums, based))
    return rows


def render_perf_claims(rows):
    if not rows:
        return "no numeric performance claim in the PR description\n"
    out = []
    bare = [r for r in rows if not r[2]]
    for line, nums, based in rows:
        mark = "ok" if based else "->"
        out.append(f"{mark} {line}")
        if not based:
            out.append(f"     {', '.join(nums)} with nothing said about what it is measured"
                       f" against")
    if bare:
        out.append("")
        out.append(f"{len(bare)} of {len(rows)} claim lines name no baseline. P1 wants the "
                   f"number with its units AND its comparison; P3 wants it reproducible. "
                   f"Ask what the other side of the comparison was -- main, torch, CK, the "
                   f"previous kernel -- and on which shapes.")
    return "\n".join(out) + "\n"



def pinned_struct_churn(diff_text, root):
    """Structs the diff changes that the TREE pins with an offsetof/sizeof assertion.

    D11 originally fired on any struct field churn in C-like source, which over batch five
    reported aiter#5223 -- six headers, no assertion anywhere near them. A layout change is
    only a defect when something asserts the layout, and that fact lives in the tree, not
    in the diff. This finds the struct names whose fields moved, then looks for an
    assertion naming them."""
    root = pathlib.Path(root)
    changed = set()
    for blk in re.split(r"(?m)^diff --git ", diff_text)[1:]:
        head = blk.split("\n", 1)[0]
        m = re.match(r"a/(\S+) b/(\S+)", head)
        if not m or m.group(2).rsplit(".", 1)[-1] not in C_LIKE:
            continue
        scope = None
        for ln in blk.split("\n"):
            hh = re.match(r"@@ [^@]*@@\s*(.*)$", ln)
            if hh:
                sm = re.search(r"\b(?:struct|class)\s+(\w+)", hh.group(1))
                scope = sm.group(1) if sm else None
                continue
            sm = re.search(r"\b(?:struct|class)\s+(\w+)[^;]*\{", ln[1:] if ln[:1] in "+- " else ln)
            if sm:
                scope = sm.group(1)
            if not (scope and ln[:1] in "+-") or ln.startswith(("+++", "---")):
                continue
            body = ln[1:]
            if NOT_A_FIELD.match(body) or "(" in body or body.strip().startswith("//"):
                continue
            if re.match(r"\s*(const\s+)?[\w:<>,\*&]+(\s+[\w:<>\*&]+)+"
                        r"(\s*\[[^\]]*\])?\s*;\s*$", body):
                changed.add(scope)
    if not changed:
        return []
    if re.search(r"(?m)^[+-].*(offsetof\(|static_assert\(\s*sizeof)", diff_text):
        return []                    # the PR updates the assertions: correct shape
    out = []
    for name in sorted(changed):
        pat = re.compile(rf"(offsetof\(\s*{re.escape(name)}\b|"
                         rf"static_assert\(\s*sizeof\(\s*{re.escape(name)}\s*\))")
        for cand in root.rglob("*"):
            if cand.suffix not in (".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc"):
                continue
            try:
                text = cand.read_text(errors="replace")
            except OSError:
                continue
            hits = pat.findall(text)
            if hits:
                out.append((name, str(cand.relative_to(root)), len(hits)))
                break
    return out



# ----------------------------------------------------------------- card gate
FINDING = re.compile(r"^\s*(\U0001F534|\u26A0\uFE0F?|\U0001F4DD)\s*(.+)$")
VALUE = re.compile(r"\b\d+\b|fp8|fp16|bf16|fp4|int8|int32|int64|e4m3|e5m2|gfx\d+|"
                   r"nullptr|None\b|2\^\d+", re.I)
IDENT = re.compile(r"\b[a-z]+_[a-z_0-9]{2,}\b|\b[A-Z][A-Z0-9_]{3,}\b")


def is_concrete(text):
    """Does the finding name something, or is it a feeling?

    The red threshold asks for the input that makes it fire. A value satisfies that; so
    does naming the exact expressions involved -- "row=(tail_blk*num_kv_heads+kv_head_idx)
    *BLOCK_M vs nrows=tail*num_kv_heads*BLOCK_M" is as concrete as a finding gets, and has
    no bare digit in it. "The reduce kernel looks racy" names nothing. Two distinct code
    identifiers is the line between them."""
    return bool(VALUE.search(text)) or len(set(IDENT.findall(text))) >= 2


def audit_card(card_text, verdicts_text, diagnostic_text, answers_text, diff_text):
    """Every finding in the card must trace back to something already adjudicated.

    The three gates before this one check that the work happened. None of them checks
    that the card reports THAT work: a review can adjudicate 27 rules honestly and then
    write a finding that appears in none of them. And the 'before firing any red, write
    down the concrete input that triggers it' threshold was prose with nothing reading it.

    Three things are checkable, and only three. Whether a finding is CORRECT is not one
    of them -- that is what a human reads the card for."""
    touched = set(re.findall(r"^diff --git a/\S+ b/(\S+)", diff_text, re.M))
    # Every FIRE has to reach the card. Checking only that findings trace back to work
    # left the inverse open: a review can adjudicate nine rules FIRE, report none of
    # them, and pass. FIRE means it goes in the card; if it is not going in, the verdict
    # is not FIRE, and saying so is a one-word edit to the ledger. The escape is
    # deliberate and must be written down: `-- not reported: <reason>` on the verdict.
    fired = {}
    for ln in (verdicts_text or "").splitlines():
        m = re.match(r"\s*([A-Z]+\d+[a-z]?)\s+FIRE\b\s*(?:--|—|:)\s*(.*)$", ln)
        if m and "not reported:" not in m.group(2).lower():
            fired[m.group(1)] = m.group(2)[:60]
    backing = (verdicts_text or "") + "\n" + (diagnostic_text or "") + "\n" + (answers_text or "")
    problems = []
    findings = []
    for line in (card_text or "").splitlines():
        m = FINDING.match(line.strip())
        if m:
            findings.append((m.group(1), m.group(2).strip()))
    reported = set()
    for _, text in findings:
        rid = re.match(r"([A-Z]+\d+[a-z]?):", text)
        if rid:
            reported.add(rid.group(1))
    for rule, why in sorted(fired.items()):
        if rule not in reported:
            problems.append(("UNREPORTED-FIRE", f"{rule}: {why}",
                             "adjudicated FIRE and absent from the card -- report it, or "
                             "change the verdict, or append `-- not reported: <reason>`"))
    for sev, text in findings:
        red = sev.startswith("\U0001F534")
        cited = [p for p in re.findall(
            r"([\w./-]+\.(?:py|cu|cuh|hip|h|hpp|cpp|cc|csv|json|yaml|yml|sh|md))", text)]
        untouched = [c for c in cited
                     if not any(t == c.lstrip("./") or t.endswith("/" + c.lstrip("./"))
                                for t in touched)]
        if untouched:
            problems.append(("UNTOUCHED-FINDING", text[:70],
                             f"cites {untouched[0]}, which this PR does not change"))
        # Does anything already adjudicated mention this? Match on the rule id if the
        # finding carries one, else on a distinctive token from the text.
        rid = re.match(r"([A-Z]+\d+[a-z]?):", text)
        if rid:
            if not re.search(rf"(?<![\w]){re.escape(rid.group(1))}\b\s+FIRE", backing):
                problems.append(("UNBACKED-FINDING", text[:70],
                                 f"{rid.group(1)} is reported but was not adjudicated FIRE"))
        elif cited and not any(c.rsplit("/", 1)[-1] in backing for c in cited):
            problems.append(("UNBACKED-FINDING", text[:70],
                             "appears in no verdict, diagnostic or blind-spot line"))
        if red and not is_concrete(text):
            problems.append(("UNPROVEN-RED", text[:70],
                             "names no concrete shape, dtype, arch or value -- the red "
                             "threshold asks for the input that makes it fire"))
    return findings, problems



if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("rules", "evidence", "symbols", "ledger", "expand", "answers",
                        "mapping", "diagnostic", "testquality",
                        "twins", "citest", "perfclaims",
                        "structabi", "commentonly", "card"):
        print("usage: triage.py rules <diff> [title]\n"
              "       triage.py evidence <diff> <head-file>...\n"
              "       triage.py symbols <diff> <merge-target-root>\n"
              "       triage.py ledger <rules.txt> <verdicts.txt> [diff]\n"
              "       triage.py expand <rules.txt> <rules.md>\n"
              "       triage.py answers <answers.txt>\n"
              "       triage.py mapping\n"
              "       triage.py diagnostic <ai_diagnostic.txt>\n"
              "       triage.py testquality <diff>\n"
              "       triage.py twins <diff> <root>\n"
              "       triage.py citest <diff> <root>\n"
              "       triage.py perfclaims <pr_meta.json>\n"
              "       triage.py structabi <diff> <root>\n"
              "       triage.py commentonly <diff>\n"
              "       triage.py card <card.md> <verdicts> <diagnostic> <answers> <diff>", file=sys.stderr)
        raise SystemExit(2)

    if mode == "card":
        def _read(i):
            try:
                return open(sys.argv[i], errors="replace").read()
            except (OSError, IndexError):
                return ""
        card_path = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
        if card_path is None or not card_path.is_file():
            print(f"CARD MISSING: {card_path}. Write the card there before the gate runs "
                  f"-- not writing it was the cheapest way past this check",
                  file=sys.stderr)
            raise SystemExit(1)
        findings, problems = audit_card(_read(2), _read(3), _read(4), _read(5), _read(6))
        if not findings and not problems:
            print("CARD: no findings, and no verdict fired")
            raise SystemExit(0)
        for kind, text, why in problems:
            print(f"{kind}: {text}")
            print(f"  {why}")
        if problems:
            unreported = sum(1 for k, _, _ in problems if k == "UNREPORTED-FIRE")
            about_findings = len({t for k, t, _ in problems if k != "UNREPORTED-FIRE"})
            parts = []
            if about_findings:
                parts.append(f"{about_findings} of {len(findings)} findings cannot be "
                             f"traced")
            if unreported:
                parts.append(f"{unreported} verdict(s) fired and are not on the card")
            print("CARD NOT NAILED DOWN: " + "; ".join(parts), file=sys.stderr)
            raise SystemExit(1)
        print(f"CARD NAILED DOWN: {len(findings)} findings, each traced to an "
              f"adjudication and anchored in a changed file; every FIRE accounted for")
        raise SystemExit(0)

    if mode == "commentonly":
        res = comment_dominated(open(sys.argv[2], errors="replace").read())
        if res is None:
            print("not a comment-dominated diff")
            raise SystemExit(0)
        total, code = res
        print(f"COMMENT-DOMINATED: {total} changed lines, {len(code)} of them code "
              f"({len(code)/total:.1%})")
        print("  Everything else is prose. These are the lines to review:")
        for path, sign, body in code[:40]:
            print(f"    {sign} {path}: {body.strip()[:96]}")
        if len(code) > 40:
            print(f"    ... and {len(code) - 40} more")
        raise SystemExit(0)

    if mode == "structabi":
        root = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else ".")
        rows = pinned_struct_churn(open(sys.argv[2], errors="replace").read(), root)
        if not rows:
            print("no struct with a pinned layout changed shape")
        for name, where, n in rows:
            print(f"PINNED-LAYOUT: {name} gains or loses a field")
            print(f"  {n} assertion(s) in {where} fix its size and field offsets, and this"
                  f" diff changes none of them")
            print(f"  appending at the end shifts nothing; inserting anywhere else shifts"
                  f" every offset after it, and the code objects those assertions guard"
                  f" must be rebuilt")
        raise SystemExit(0)

    if mode == "perfclaims":
        import json as _json
        try:
            body = _json.load(open(sys.argv[2])).get("body") or ""
        except Exception:
            body = ""
        sys.stdout.write(render_perf_claims(perf_claims(body)))
        raise SystemExit(0)

    if mode == "citest":
        root = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else ".")
        rows = uncovered_test_paths(open(sys.argv[2], errors="replace").read(), root)
        if not rows:
            print("every new test file lands where a CI job will run it")
        for path, why in rows:
            print(f"UNRUN-TEST: {path}")
            print(f"  {why}")
        raise SystemExit(0)

    if mode == "twins":
        root = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else ".")
        pairs = near_duplicates(open(sys.argv[2], errors="replace").read(), root)
        if not pairs:
            print("no new file closely mirrors an existing one")
        for new, old, ratio in pairs:
            print(f"TWIN: {new}")
            print(f"  {ratio:.0%} of its substantive lines already appear in {old}")
            print(f"  diff the two and look for the asymmetry: dtype width, a mask on one"
                  f" side only, a flipped stride order, a bound the copy did not adapt")
        raise SystemExit(0)

    if mode == "testquality":
        rows = test_quality(open(sys.argv[2], errors="replace").read())
        if not rows:
            print("no test files touched")
        else:
            sys.stdout.write(render_test_quality(rows))
        raise SystemExit(0)

    if mode == "diagnostic":
        try:
            text = open(sys.argv[2], errors="replace").read()
        except OSError:
            text = ""
        missing, thin = audit_diagnostic(text)
        for q in missing:
            print(f"UNCHECKED: structural check {q} has no line in {sys.argv[2]}")
        for q in thin:
            print(f"NO-SUBSTANCE: check {q} records a verdict but not what was looked at")
        if missing or thin:
            print(f"DIAGNOSTIC INCOMPLETE: {6 - len(missing) - len(thin)}/6",
                  file=sys.stderr)
            raise SystemExit(1)
        print("DIAGNOSTIC COMPLETE: 6/6")
        raise SystemExit(0)

    if mode == "mapping":
        sys.stdout.write(render_mapping())
        raise SystemExit(0)

    if mode == "answers":
        try:
            text = open(sys.argv[2], errors="replace").read()
        except OSError:
            text = ""
        missing, thin = audit_answers(text)
        for q in missing:
            print(f"UNANSWERED: {q} has no line in {sys.argv[2]}")
        for q in thin:
            print(f"NO-SUBSTANCE: {q} is answered with a formula, not an answer")
        if missing or thin:
            print(f"ANSWERS INCOMPLETE: {6 - len(missing) - len(thin)}/6", file=sys.stderr)
            raise SystemExit(1)
        print("ANSWERS COMPLETE: 6/6")
        raise SystemExit(0)

    if mode == "expand":
        rules_text = open(sys.argv[2], errors="replace").read()
        rules_md = open(sys.argv[3], errors="replace").read()
        text, emitted, missing = expand_rules(rules_text, rules_md)
        sys.stdout.write(text)
        if missing:
            # A derived rule with no body is a rules.md that drifted from the deriver.
            # Say so loudly: silently emitting 11 of 12 reads exactly like emitting 12.
            print(f"\nMISSING-RULE-TEXT: {' '.join(missing)} derived but not found in "
                  f"{sys.argv[3]}", file=sys.stderr)
            raise SystemExit(1)
        print(f"\n({len(emitted)} rule bodies)", file=sys.stderr)
        raise SystemExit(0)

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
        # One line per distinct (import, reason). A PR importing a deleted module from
        # four files produced four identical lines, which reads as four problems.
        seen_pairs = set()
        bad = [b for b in bad if not (b in seen_pairs or seen_pairs.add(b))]
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
        types = derive(files, title, diff)
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
