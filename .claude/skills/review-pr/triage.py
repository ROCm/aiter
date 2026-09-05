#!/usr/bin/env python3
"""Step 1b triage: derive which rule families apply, and collect their evidence.

Replaces Step 3's self-applied prose checklist. Emits only the matching rules so
the model reads ~12 instead of all of them.  Conservative: when a type cannot be decided
structurally it is INCLUDED, never dropped.
"""
import difflib, re, sys, pathlib, collections
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
    # `.to(tl.float32)` alone was 20 of the 54 firings and every sample was an upcast on
    # a load, a store or a quantisation max -- the opposite of the accumulator T5 means.
    # The manual reduction it also covers is kept by name instead.
    if re.search(r"tl\.dot\(|allow_tf32|input_precision|"
                 r"acc\w*\s*\+=|acc\w*\s*=\s*tl\.zeros", add):
        fams.append(("triton-accum-prec", "T5"))
    # `tl.cdiv(` is ceiling division, not a grid: of the 7 firings it alone produced, all
    # 7 were a tile count, a block-pointer `shape=`, a mask bound or a loop bound.
    if re.search(r"grid\s*=\s*lambda|tl\.program_id", add):
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



def dropped_parameter(diff_text):
    """Parameters deleted from a Python signature without the name reappearing anywhere.

    api-signature fires only when a `def`/`void`/`template` line is on both sides of the
    diff -- a signature rewritten in place. Deleting one line of a multi-line signature
    never touches the `def` line, so that family stayed silent on 9-11% of open PRs. The
    Python only: the same regex over C++ reads `std::optional<Tensor>& fp8_out,` as a
    parameter named `std`, and telling a type prefix from a parameter name there needs a
    real declaration parser, not a line pattern.
    """
    files, out = parse(diff_text), []
    add_all = "\n".join(l for f in files.values() for l in f["add"])
    cur, old_side = None, []
    for ln in diff_text.splitlines():
        m = re.match(r"^diff --git a/(\S+) b/(\S+)", ln)
        if m:
            cur, old_side = m.group(2), []
            continue
        if ln.startswith("@@"):
            old_side = []
            continue
        if not cur or ln.startswith(("---", "+++")):
            continue
        if ln.startswith("+"):
            continue
        if not ln.startswith(("-", " ")):
            continue
        body = ln[1:]
        if ln.startswith("-") and cur.endswith(".py") \
                and not re.search(r"^(op_tests|tests)/", cur) and "bench" not in cur:
            m = re.match(r"\s*(\w+)\s*(?::[^=]+)?(?:=\s*\S.*)?,\s*(?:#.*)?$", body)
            name = m.group(1) if m else None
            sig = _open_signature(old_side)
            if name and sig and name not in ("return", "if", "else", "for", "while",
                                             "import", "from") \
                    and not re.search(r"\b%s\b" % re.escape(name), add_all):
                out.append((cur, name, sig))
        old_side.append(body)
    return out


SIG_OPEN = re.compile(r"^\s*(def |void |template|__global__|__device__|\w[\w:<>\s\*&]*\s\w+\s*\()")


def _open_signature(old_side):
    """The signature this line sits inside, or None if its parens already closed.

    A hunk header names the ENCLOSING def, not the construct the line belongs to, so
    `transpose_out=True,` passed to a call in the body of `def asm_moe(` read as a dropped
    parameter of asm_moe. Walking the old side back to an unbalanced `(` is what separates
    the signature from everything written inside it.
    """
    depth = 0
    for prev in reversed(old_side[-40:]):
        code = prev.split("#")[0]
        depth += code.count(")") - code.count("(")
        if depth < 0:
            return prev.strip()[:60] if SIG_OPEN.match(prev) else None
    return None


KERNEL_DIRS = ("aiter/ops/triton/", "_triton_kernels/", "_gluon_kernels/", "/gluon/",
               "aiter/ops/flydsl/", "csrc/kernels/", "csrc/py_itfs_cu/")


def untested_new_kernel(diff_text):
    """New kernel files when the PR adds no test pytest will collect.

    HK6 already fires on new-kernel, but it was prose: the reader had to work out whether
    a test existed. Collectibility is decided by FILENAME, the rule pytest actually uses --
    an earlier version looked only under op_tests/ and tests/, and reported aiter#3991 as
    untested while the PR added five test files under aiter/aot/flydsl/tests/. Whether CI
    then runs them is a separate question, and citest is what answers it. A benchmark is
    not a test -- it is the shape these PRs ship instead of one -- so benchmark directories
    and bench_* files do not count. Merely CONTAINING "bench" does not disqualify a file:
    aiter#2889's test_rmsnorm_bench_against_aiter.py holds two collectible tests.
    """
    files = parse(diff_text)
    new = [p for p, f in files.items()
           if f["new"] and p.endswith((".py", ".cu", ".cuh", ".hip"))
           and any(k in p for k in KERNEL_DIRS)
           and not p.startswith(("op_tests/", "tests/"))
           and not re.search(r"tutorial|template|bench", p)]
    if not new:
        return []
    for p, f in files.items():
        base = p.rsplit("/", 1)[-1]
        if not (base.startswith("test_") or base.endswith("_test.py")):
            continue
        if base.startswith("bench") or re.search(r"/(op_)?benchmarks?/", p):
            continue
        if any(re.match(r"\s*(def test_\w+|class Test\w+)", l) for l in f["add"]):
            return []
    return new


def render_untested_kernel(new):
    if not new:
        return "KERNEL TESTS: no new kernel file without a collectible test"
    out = ["NEW KERNEL, NO COLLECTIBLE TEST -- %d file(s). Collectible means a test_*.py" % len(new),
           "or *_test.py anywhere in the tree; benchmark dirs and bench_* files were not",
           "counted. HK6 is the rule; this is its evidence."]
    out += ["  %s" % p for p in new]
    return "\n".join(out)


PYDEF = re.compile(r"^(\s*)def\s+(\w+)\s*\(")
SIB_NOISE = re.compile(r"^\s*(from|import|def |@)")


def _py_functions(text):
    """[(name, body_lines)] -- each def's span runs to the next def at its indent or shallower."""
    lines = text.splitlines()
    starts = [(i, m.group(2), len(m.group(1)))
              for i, l in enumerate(lines) for m in [PYDEF.match(l)] if m]
    out = []
    for k, (i, name, ind) in enumerate(starts):
        end = len(lines)
        for j, _, ind2 in starts[k + 1:]:
            if ind2 <= ind:
                end = j
                break
        out.append((name, lines[i + 1:end]))
    return out


def _variant_pair(a, b):
    """True when two function names are variants: a shared stem, not the same name.

    `_dynamic_per_tensor_quant_fp8_i8_kernel` and `_static_per_tensor_quant_fp8_i8_kernel`
    share 30 characters; `_flydsl_stage1_wrapper` and `_adaptive_moe_sort` share nothing
    that long. Without this the check fires on 23.3% of the corpus, mostly on helpers that
    merely sit in the same file.

    Deliberately not a list of variant suffixes. Adding one (_opt, _v2, _prefill, _decode,
    ...) takes 18.3% to 10.8% and the 43 it drops are mostly real: kernel_unified_attention_2d
    against _3d, three times over, and select_2d_config against select_3d_config. D9's rule
    body records the same lesson from the other direction -- do not narrow it to a name list.
    """
    if not a or not b or a == b:
        return False
    m = difflib.SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
    return m.size >= 8


def sibling_variants(diff_text, root):
    """Lines this PR changed that a variant of the changed function still carries.

    A1 asks whether the sibling kernel has the same bug and has never had an answer: twins
    compares whole FILES, and A1's own example -- aiter#3841, a decode kernel fixed while
    `_prefill_opt` beside it was not -- is two functions in ONE file. 18.3% of 600 open
    PRs. Python only: every hit was, and splitting C bodies by brace depth is a heuristic
    with nothing to show for it.

    Evidence, not a finding. A variant legitimately diverges; what the reader is being
    handed is the pair, so that judgement is made rather than skipped.
    """
    out = []
    for path, blk in _diff_blocks(diff_text):
        if not path.endswith(".py") or "new file mode" in blk:
            continue
        if re.search(r"^(op_tests|tests)/|bench|3rdparty", path):
            continue
        try:
            funcs = _py_functions((pathlib.Path(root) / path).read_text(errors="replace"))
        except OSError:
            continue
        if len(funcs) < 2:
            continue
        scope = ""
        for ln in blk.split("\n"):
            h = re.match(r"^@@ [^@]*@@\s*(.*)$", ln)
            if h:
                scope = h.group(1)
                continue
            if not ln.startswith("-") or ln.startswith("---"):
                continue
            s = ln[1:].strip()
            if len(s) < 25 or TRIVIAL_LINE.match(ln[1:]) or SIB_NOISE.match(s):
                continue
            # a line that computes or decides something -- not a bare rebinding, and not
            # a docstring's `x: [BLOCK_SIZE_M, BLOCK_SIZE_N], fp32`
            if re.match(r"^\w+\s*=\s*[\w.\"\'()]+$", s) or re.match(r"^\w+\s*:\s*\[", s):
                continue
            if not re.search(r"[-+*/%<>]|\w\[|\)\s*\.|\bif\b|\breturn\b|=", s):
                continue
            fm = re.search(r"\b(\w+)\s*\(", scope)
            changed = fm.group(1) if fm else None
            for name, body in funcs:
                if not _variant_pair(changed, name):
                    continue
                if any(x.strip() == s for x in body):
                    out.append((path, changed, name, s[:70]))
    seen, uniq = set(), []
    for o in out:
        if (o[0], o[2]) not in seen:
            seen.add((o[0], o[2]))
            uniq.append(o)
    return uniq[:6]


TRIVIAL_LINE = re.compile(r"^\s*[\}\{\)\(\];,\\]*\s*$|^\s*(#|//|\*)")


def _diff_blocks(diff_text):
    for blk in re.split(r"(?m)^diff --git ", diff_text)[1:]:
        m = re.match(r"a/(\S+) b/(\S+)", blk.split("\n", 1)[0])
        if m:
            yield m.group(2), blk


def render_siblings(rows):
    if not rows:
        return "SIBLINGS: no variant of a changed function still carries a changed line"
    out = ["VARIANT SIBLING STILL CARRIES A CHANGED LINE -- %d pair(s). A1 is the rule;"
           % len(rows),
           "divergence can be correct, so this is the pair to judge, not a finding."]
    for path, changed, sib, line in rows:
        out.append("  %s: %s -> %s" % (path, changed, sib))
        out.append("      %s" % line)
    return "\n".join(out)


OWN_TREE = pathlib.Path(__file__).resolve().parents[3]


def _head_date(root):
    try:
        import subprocess
        r = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%h %ad",
                            "--date=short"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "not a git checkout"
    except Exception:
        return "unreachable"


def tree_root(arg):
    """The tree a forensic reads, with a warning when it is not the one shipping this file.

    Every root-taking mode answers a question about the MERGE TARGET -- is this struct
    pinned, does a CI job scan this path, does a sibling still carry this line. Pointed at
    a second checkout the answers stay well-formed and quietly describe a different repo.

    That is not hypothetical. A sweep and then a follow-up both read a checkout 53 days
    behind, concluded that nothing pinned `pa_sparse_prefill_kargs`, and filed D11's worked
    example as fiction -- confirmed twice over, because the field offsets are pinned through
    a macro and a literal `offsetof(` is absent either way. The rule was right. fetch.sh
    always passes the correct root because it takes it from `git rev-parse --show-toplevel`;
    it is the hand-run measurement that goes wrong, and that is what this line is for.
    """
    root = pathlib.Path(arg).resolve()
    if root != OWN_TREE:
        print("warning: reading %s, but this skill ships from %s.\n"
              "         given: %s\n         skill: %s\n"
              "         Forensics answer questions about the merge target; a second "
              "checkout answers them about itself."
              % (root, OWN_TREE, _head_date(root), _head_date(OWN_TREE)), file=sys.stderr)
    return root


GUARD_LINE = re.compile(r"(\bassert\b|torch\.zeros|\.contiguous\(\)|AITER_CHECK|TORCH_CHECK)")


def _guard_subject(line):
    return set(re.findall(r"\b[a-z_][a-z0-9_]{3,}\b", line.lower())) - GUARD_NOISE


def guard_changes(diff_text):
    """Each guard the diff deletes, and what became of it: moved, changed, or gone.

    `invariant-removed` fires on 69 of 600 open PRs and D4 is red, so the reviewer has to
    establish what happened to every deleted assert by hand -- deleted_guard_symbols says
    itself that the usual answer is "it moved". In 14.5% of them every deleted guard is
    added straight back unchanged and there is nothing to review at all; 58% drop at least
    one outright, and 46% return at least one in changed form.

    The rest is why this reports the pair instead of filtering on it. A guard that returns
    in changed form looked like the same non-event, and is where the finding actually
    lives: aiter#4295 replaces a null check with a size check and the null check is simply
    gone, aiter#4279 halves a TORCH_CHECK bound, aiter#5168 widens an equality to a set
    membership. Suppressing those would have hidden three real weakenings out of seven read.
    """
    files = parse(diff_text)
    dele = [l for f in files.values() for l in f["del"]
            if GUARD_LINE.search(l) and not is_comment_line(l)]
    add = [l for f in files.values() for l in f["add"] if not is_comment_line(l)]
    add_exact = {l.strip() for l in add}
    add_guards = [l.strip() for l in add if GUARD_LINE.search(l)]
    moved, changed, gone = [], [], []
    for d in dele:
        ds = d.strip()
        if ds in add_exact:
            moved.append(ds)
            continue
        best, score = None, 0
        for a in add_guards:
            n = len(_guard_subject(ds) & _guard_subject(a))
            if n > score:
                best, score = a, n
        if score >= 2:
            changed.append((ds, best))
        else:
            gone.append(ds)
    return moved, changed, gone


def render_guard_changes(res):
    moved, changed, gone = res
    if not (moved or changed or gone):
        return "GUARDS: this diff deletes no assert, check, zero-init or contiguous call"
    out = ["GUARDS DELETED -- %d returned unchanged, %d returned in changed form, %d gone."
           % (len(moved), len(changed), len(gone))]
    if changed:
        out.append("The changed ones are where D4 and B7 live: same subject, "
                   "different bound.")
    for a, b in changed[:8]:
        out.append("  changed:")
        out.append("    - %s" % a[:110])
        out.append("    + %s" % b[:110])
    for g in gone[:8]:
        out.append("  gone:    %s" % g[:110])
    if moved:
        out.append("  moved unchanged: %d line(s); the invariant still holds somewhere"
                   % len(moved))
    return "\n".join(out)


# A parenthetical between the verb and the dashes is how a reviewer says the claim was
# narrowed rather than dropped -- `WARN SURVIVED (severity reduced) -- ...`. Rejecting it
# as MALFORMED taught two reviews to re-encode the same fact as prose, which is a worse
# record of what happened.
REFUTATION = re.compile(
    r"^\s*(RED|WARN|NOTE)\s+(SURVIVED|KILLED)\s*(\([^)]{0,60}\))?\s*(?:--|—|:)\s*(.+)$")
REFUTE_SURFACE = re.compile(
    r"^(i )?(re-?)?(checked|verified|confirmed|reviewed|looked)\b.{0,40}$", re.I)


def audit_refutations(text, diff_text, card_text):
    """Step 7.6: every finding the card reports had to survive an attempt to kill it.

    A review that runs all five other gates can still ship a claim whose premise is
    simply false. aiter#5072 reported 125 of 174 config rows as unreachable, marked
    `[verified]`, having reasoned about `getPaddedM`'s padding chain -- and
    `get_CKGEMM_config` loops `for gl in [None, 0, 1]`, so exact M is tried first and
    those rows are reachable. Ten seconds of reading the lookup would have killed it.
    Nothing in the review asked for those ten seconds.

    So the gate asks for the attempt, not for a verdict it cannot judge. What it can
    check is that an attempt was made per reported finding, that it names something
    consulted -- a path, a symbol, a command -- and that the reasoning is not the word
    "checked" with nothing behind it.

    Self-refutation is the weaker tier. It catches a premise never tested; it does not
    catch a reviewer attached to a conclusion. SKILL.md says so and points at the
    independent pass for anyone who wants the stronger one.
    """
    if not (text or "").strip():
        return ["REFUTATIONS MISSING: Step 7.6 writes one line per reported finding "
                "before the card is judged"]
    reported = len([l for l in (card_text or "").splitlines()
                    if l.lstrip().startswith(("\U0001F534", "\u26A0", "\U0001F4DD"))])
    rows, problems = [], []
    for ln in text.splitlines():
        m = REFUTATION.match(ln)
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        if not m:
            problems.append(("MALFORMED", ln.strip()[:70],
                             "expected `RED|WARN|NOTE SURVIVED|KILLED -- what was checked "
                             "and why it did or did not die`"))
            continue
        sev, outcome, _note, why = m.groups()
        rows.append((sev, outcome, why))
        if len(why.strip()) < MIN_CORE_REASON or REFUTE_SURFACE.match(why.strip()):
            problems.append(("NO-ATTEMPT", why.strip()[:70],
                             "names nothing that was consulted; a refutation that cannot "
                             "be repeated is not one"))
            continue
        cited = [c for c, _ in CITATION.findall(why)]
        tokens = set(re.findall(r"\b\w+\b", why))
        if not cited and not (tokens & changed_tokens(diff_text)) \
                and not WORK_ARTIFACT.search(why) \
                and not re.search(r"`[^`]+`|\bgit \w+|\bgrep\b|\bpytest\b|\b\w+/\w+", why):
            problems.append(("UNSOURCED", why.strip()[:70],
                             "no file, symbol or command -- say what you opened"))
    survived = [r for r in rows if r[1] == "SURVIVED"]
    if reported and len(survived) < reported:
        problems.append(("UNREFUTED-FINDING", f"{reported} reported, {len(survived)} survived",
                         "every finding on the card needs a line saying it survived"))
    if problems:
        out = [f"REFUTATION INCOMPLETE: {len(rows)} attempt(s), {len(problems)} problem(s)"]
        for kind, what, why in problems[:8]:
            out.append(f"{kind}: {what}")
            out.append(f"  {why}")
        return out
    return [f"REFUTATIONS COMPLETE: {len(rows)} attempted, {len(survived)} survived, "
            f"{len(rows) - len(survived)} killed before reaching the card"]


INDEP = re.compile(r"^\s*(SURVIVED|KILLED)\s*(\([^)]{0,60}\))?\s*(?:--|—|:)\s*(.+)$")
INDEP_NONE = re.compile(r"^\s*NONE AVAILABLE\s*(?:--|—|:)\s*(.+)$")


def audit_independent(text, card_text):
    """Step 7.7: a reader who has not seen the reasoning judges each reported finding.

    Measured over 200 PRs in four waves: Step 7.6's self-refutation kills 260-280
    candidates per 50 PRs, real work by any measure -- and 28% of what survives it and
    reaches a card is still killed by an independent adversary told the findings are false
    until defended. Four rounds of fixing the tools moved the gate pushbacks (64 to 33) and
    the completion rate (47/50 to 49/50) and did not move that number at all. Refuting your
    own finding catches a premise you never tested. It does not catch a conclusion you are
    committed to, and no amount of instructing yourself harder substitutes for a reader who
    was never committed.

    The step does not require the ability to spawn anything. Hand $WORK/card.md, the diff
    and the merge-target path to a second agent or a person, with no part of your reasoning
    attached, and record what comes back. What this gate can check is that it happened, per
    finding, with something named that a third party could open -- and, when no such reader
    exists, that the card says so instead of implying one looked.
    """
    reported = [l for l in (card_text or "").splitlines()
                if l.lstrip().startswith(("\U0001F534", "\u26A0", "\U0001F4DD"))]
    if not (text or "").strip():
        return ["INDEPENDENT REFUTATION MISSING: Step 7.7 writes one line per reported "
                "finding, or `NONE AVAILABLE -- <reason>` if no independent reader exists"]
    none = INDEP_NONE.match((text or "").strip().splitlines()[0])
    if none:
        if reported and "not independently refuted" not in (card_text or "").lower():
            return ["INDEPENDENT REFUTATION DECLARED UNAVAILABLE, CARD DOES NOT SAY SO: "
                    "add `not independently refuted` to the review line, so a reader knows "
                    "these findings carry only the author's own check"]
        return ["INDEPENDENT REFUTATION: none available (%s); the card says so"
                % none.group(1)[:60]]
    rows, problems = [], []
    for ln in text.splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        m = INDEP.match(ln)
        if not m:
            problems.append(("MALFORMED", ln.strip()[:70],
                             "expected `SURVIVED|KILLED -- what the reader checked`"))
            continue
        outcome, _note, why = m.groups()
        rows.append((outcome, why))
        if len(why.strip()) < MIN_CORE_REASON or REFUTE_SURFACE.match(why.strip()):
            problems.append(("NO-ATTEMPT", why.strip()[:70],
                             "names nothing the reader opened"))
    killed = [r for r in rows if r[0] == "KILLED"]
    survived = [r for r in rows if r[0] == "SURVIVED"]
    if reported and len(survived) < len(reported):
        problems.append(("UNDEFENDED-FINDING",
                         f"{len(reported)} on the card, {len(survived)} defended",
                         "a finding an independent reader did not clear must come off the "
                         "card, not stay on it"))
    if problems:
        out = [f"INDEPENDENT REFUTATION INCOMPLETE: {len(rows)} judged, "
               f"{len(problems)} problem(s)"]
        for kind, what, why in problems[:8]:
            out.append(f"{kind}: {what}")
            out.append(f"  {why}")
        return out
    return [f"INDEPENDENTLY REFUTED: {len(rows)} judged, {len(survived)} defended, "
            f"{len(killed)} killed by a reader who had not seen the reasoning"]


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
    if (re.search(r"^\s*(def |void |template)", dele, re.M)
            and re.search(r"^\s*(def |void |template)", add, re.M)) or \
            dropped_parameter(raw_diff):
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
    # KERNEL_PY lists the triton and gluon paths and stops there, so a FlyDSL kernel was a
    # kernel to `flydsl` (D10, D10b -- compile-result handling) and to nothing else: 95 of
    # 600 open PRs edit aiter/ops/flydsl/kernels/*.py and derive neither kernel family, so
    # the sibling variant, the uninitialised accumulator, the missing contiguous check and
    # the unmeasured cost were never asked about a whole backend. B2 stays out: it is
    # `tl.load`/`tl.store` without a mask, and there is no tl in FlyDSL.
    if any("aiter/ops/flydsl/kernels/" in p and p.endswith(".py") and (f["add"] or f["del"])
           for p, f in files.items()):
        hit("flydsl-kernel", "A1 D1 D8 P6")
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
# The failure is invisible from inside a review -- the ledger only ever sees rules that
# were derived -- so `triage.py mapping` is what surfaces it.
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


# Alternation is longest-first and the tail is guarded: with `cu` before `cuh` and no
# boundary, `gemm_a8w8_blockscale_cktile_common.cuh` matched as `...common.cu`, a file the
# PR changes read as one it does not. `=` belongs in the class because tuned-config names
# carry it -- `gfx1201-GEMM-A8W8_BLOCKSCALE-N=1024-K=1024.json` matched as `1024.json`.
CITATION = re.compile(r"([\w./=-]+\.(?:cuh|hpp|yaml|yml|json|csv|cpp|hip|py|cu|cc|co|h|md|sh))"
                      r"(?![\w])(?::(\d+))?")


def changed_paths(diff_text):
    return set(re.findall(r"^diff --git a/\S+ b/(\S+)", diff_text, re.M))


def check_citations(verdicts_text, diff_text):
    """FIRE verdicts whose citations name no file this PR changes.

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
        paths = [p for p, _ in CITATION.findall(reason)
                 if not WORK_ARTIFACT.search(p)]
        if not paths:
            continue
        # Same anchor rule as the card gate, and for the same reason. E4's own body tells
        # the reviewer to "confirm the current ci:* definitions in .github/workflows/*.yaml"
        # before quoting a label, and D11's tells them to find the assertion table in the
        # tree; doing either put a path here that the diff does not contain, and the gate
        # called the correct move a stale citation. One cited file from the diff is what
        # separates "this PR" from "some other PR".
        def _norm(p):
            return p[2:] if p.startswith("./") else p
        if not any(any(t == _norm(p) or t.endswith("/" + _norm(p))
                       for t in touched) for p in paths):
            bad.append((rule, verdict, paths[0]))
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
             "Generated by `triage.py mapping`. Do not edit — regenerate with",
             "`python3 triage.py mapping > MAPPING.md` and diff, which is how a",
             "disagreement between this file and the deriver is caught.",
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



# ------------------------------------------------------------ core-file ledger
# Step 4's backbone table, as data. Tier 1 is the two files whose failure mode is
# `import aiter` itself; Tier 2 is the dispatch shared by more than one production model
# family. Tier 3 (a single op wrapper, one kernel) is deliberately not required here --
# a Tier-1 rule written to cover `aiter/ops/*.py` fires on 71% of PRs and the assessment
# it mandates then stops being performed at all, which is the hole this gate exists to
# close, not to reopen.
TIER1 = ("aiter/jit/core.py", "aiter/__init__.py")
TIER2 = tuple(p for p in TIER if p not in TIER1)

# $WORK artifacts are evidence, not citations of tree files. P6 is by construction about
# the absence of a measurement, so validation_requirement.json IS where its evidence
# lives; naming it tripped UNTOUCHED-CITATION on four PRs in one wave and the reviews had
# to reword the line to cite source files instead, which made the ledger less accurate
# about where the evidence actually was. SKILL.md tells the reader to ground verdicts in
# these files; the gate must not then punish saying so.
WORK_ARTIFACT = re.compile(r"(?:\$WORK/|\b)(?:rules_expanded|evidence|symbols|twins|"
                           r"test_quality|ci_coverage|perf_claims|struct_abi|comment_only|"
                           r"guards|siblings|kernel_tests|applies|pr_meta|rules|"
                           r"validation_requirement|auto_validation_outcome)\.(?:txt|json)\b")

CORE_LINE = re.compile(
    # `=` belongs here: gfx950-GEMM-A16W16-N=384-K=7168.json could not be written into
    # core_files.txt at all, and the review deleted the line to get past the gate.
    r"^\s*([\w./+=-]+)\s+TIER([123])\s+(COVERED|GAP|N/A)\s*(?:--|—|:)\s*(.+?)\s*$")
MIN_CORE_REASON = 30
# `\b` before a path fragment is useless (`/` is a non-word char), so anchors are matched
# as substrings after normalising `./` -- the reason is prose, not a parseable field.
# The fourth shape is the one aiter is full of: an arch or dtype name, all lower case,
# no underscore -- gfx1250, gfx942, fp8, a8w8, mxfp4, bf16. A one-line dict edit adding
# `18: "gfx1250",` names its subject with a token the first three shapes all reject, so
# the reason that named exactly the right thing was the reason that failed to anchor.
ANCHOR_IDENT = re.compile(r"\b[a-z]+_[a-z_0-9]{2,}\b|\b[A-Z][A-Z0-9_]{3,}\b|"
                          r"\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b|"
                          r"\b(?=[a-z0-9]{3,}\b)[a-z]+\d[a-z0-9]*\b")


def core_files_in(diff_text):
    """(path, tier) for every backbone file this diff touches, from the table only.

    Q2/Q3 of Step 4 ("is this the dispatch for an op used by >1 model family") need
    judgement about a new file and cannot be decided from a diff, so the reviewer may
    add lines for files the table does not know. What a machine can decide is the
    table, and the table is what it demands a line for."""
    touched = changed_paths(diff_text)
    out = []
    for p in sorted(touched):
        if p in TIER1:
            out.append((p, "1"))
        elif p in TIER2:
            out.append((p, "2"))
    return out


def changed_tokens(diff_text):
    """Identifier-shaped tokens on the added/deleted lines of this diff."""
    body = "\n".join(l[1:] for l in diff_text.splitlines()
                     if (l.startswith(("+", "-")) and not l.startswith(("+++", "---"))))
    return set(ANCHOR_IDENT.findall(body))


def audit_core_files(text, diff_text):
    """Step 4 as an artifact: one adjudication per backbone file the diff touches.

    Step 4 was the last step that produced nothing. It could be skipped in silence and
    the review would read identically -- the same hole Step 2 had before answers.txt and
    the rule pass had before the ledger, and the reason both of those exist.

    What is checkable about a risk assessment is not whether the risk was judged right.
    It is that every backbone file in the diff was judged at all, that the judgement
    carries a reason, and that the reason is about THIS PR: a reason naming only the file
    itself, or only code this PR never touched, is a sentence that would be equally true
    of every other PR against that file, which is what "it is a core file, blast radius
    is large" costs nothing to write."""
    want = core_files_in(diff_text)
    tier_of = dict(want)
    touched = changed_paths(diff_text)
    tokens = changed_tokens(diff_text)

    declared_none = any(re.match(r"^\s*NONE\b", ln) for ln in text.splitlines())
    seen, problems = {}, []
    for ln in text.splitlines():
        if not ln.strip() or ln.lstrip().startswith("#") or re.match(r"^\s*NONE\b", ln):
            continue
        m = CORE_LINE.match(ln)
        if not m:
            problems.append(("MALFORMED", ln.strip()[:70],
                             "expected `<path> TIER<1|2|3> COVERED|GAP|N/A -- <reason>`"))
            continue
        path, tier, verdict, reason = m.groups()
        base = path.lstrip("./")
        if not any(t == base or t.endswith("/" + base) for t in touched):
            problems.append(("UNTOUCHED-FILE", base,
                             "assessed but absent from this diff -- a risk assessment of "
                             "a file the PR does not change is a stale or invented line"))
            continue
        seen[base] = (tier, verdict, reason)
        if base in tier_of and tier != tier_of[base]:
            problems.append(("TIER-MISMATCH", base,
                             f"recorded TIER{tier}; the backbone table puts it at "
                             f"TIER{tier_of[base]} -- downgrading a tier is not a way "
                             f"past the checks that tier requires"))
        if len(reason) < MIN_CORE_REASON or SURFACE.match(reason):
            problems.append(("NO-EVIDENCE", base,
                             f"marked {verdict} with no reason -- what breaks, and what "
                             f"in this PR makes you say it does not"))
            continue
        # The anchor: something in the reason has to come from this PR's own change.
        # The file's own path does not count -- restating the subject is what a reason
        # that would fit any PR against this file looks like.
        anchored = any(t != base and (t in reason or t.rsplit("/", 1)[-1] in reason)
                       for t in touched)
        if not anchored:
            anchored = any(tok in tokens for tok in ANCHOR_IDENT.findall(reason))
        if not anchored:
            problems.append(("UNANCHORED", base,
                             "the reason names no file or symbol this PR changes -- it "
                             "would read the same against any PR touching this file"))
    for path, tier in want:
        if path not in seen:
            problems.append(("UNASSESSED", path,
                             f"TIER{tier} backbone file in this diff with no assessment "
                             f"line -- Step 4 was not performed for it"))
    if declared_none and want:
        problems.append(("UNDECLARED-CORE", ", ".join(p for p, _ in want)[:70],
                         "declared NONE while the diff touches backbone files"))
    if not text.strip():
        problems.append(("EMPTY", "core_files.txt",
                         "no assessment and no NONE declaration -- an empty artifact is "
                         "a Step 4 that did not happen, not a Step 4 that found nothing"))
    elif not want and not seen and not declared_none:
        problems.append(("EMPTY", "core_files.txt",
                         "no backbone file in this diff, and no `NONE -- <reason>` line "
                         "saying so"))
    return want, seen, problems


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
    fired, fire_paths, claims = {}, {}, {}
    for ln in (verdicts_text or "").splitlines():
        m = re.match(r"\s*([A-Z]+\d+[a-z]?)\s+FIRE\b\s*(?:--|—|:)\s*(.*)$", ln)
        if m and "not reported:" not in m.group(2).lower():
            fired[m.group(1)] = m.group(2)[:60]
            # The files this verdict cited, so a card that obeys the no-rule-codes rule
            # can still be matched to the FIRE it is reporting.
            fire_paths[m.group(1)] = [pp for pp, _ in CITATION.findall(m.group(2))]
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
        # SKILL.md's output rules say, in as many words, "Do NOT use rule codes (P1, D4,
        # A1...) in output -- they are internal labels only". Claiming a FIRE by its rule
        # id was therefore something the card was forbidden to do, and every correctly
        # written card reported every fired rule as UNREPORTED-FIRE. A finding that names
        # a file the verdict cited is the same claim without the label.
        for rule, paths in fire_paths.items():
            if paths and any(pp in text for pp in paths):
                claims.setdefault(rule, []).append(text[:40])
    # A finding may report more than one rule, and often should: aiter#3165 fired B4 and
    # E5 on the same changed file and one accurate bullet covered both. Requiring N
    # distinct bullets for N rules sharing a path -- which is what this did -- forced the
    # review to annotate E5 `-- not reported: as a separate bullet`, a sentence that was
    # false on its face, since E5 was on the card inside the B4 bullet. Making a reviewer
    # write something untrue to satisfy a gate is worse than the freeloading it prevents,
    # and the freeloading is visible anyway: a card with one bullet and nine FIREs reads
    # as one bullet and nine FIREs.
    for rule, texts in claims.items():
        if texts:
            reported.add(rule)
    for rule, why in sorted(fired.items()):
        if rule not in reported:
            problems.append(("UNREPORTED-FIRE", f"{rule}: {why}",
                             "adjudicated FIRE and absent from the card -- report it, or "
                             "change the verdict, or append `-- not reported: <reason>`"))
    for sev, text in findings:
        red = sev.startswith("\U0001F534")
        cited = [p for p in re.findall(
            # `(?![\w])`: without it `torch.cuda.device(...)` reads as a file called
            # `torch.cu` and `x.contiguous.cuda()` as `x.contiguous.cu`. Those are the
            # two most ordinary words in a HIP review, so the gate produced a citation
            # complaint about a path nobody wrote.
            r"([\w./=-]+\.(?:cuh|hpp|yaml|yml|json|csv|cpp|hip|py|cu|cc|co|h|md|sh)(?![\w]))",
            text)]
        def _is_touched(c):
            # NOT lstrip("./"): it strips every leading dot and slash, so
            # `.github/workflows/perf-parity.yaml` became `github/workflows/...` and a
            # file the PR ADDS was reported as one it does not change.
            c = c[2:] if c.startswith("./") else c
            return any(t == c or t.endswith("/" + c) for t in touched)
        # An ANCHOR is required, not citation purity. Rejecting every path the diff does
        # not contain rejected the forensics the rules ask for: aiter#2478's two strongest
        # findings are contracts stated in csrc/include/moe_sorting_opus.h, which the PR
        # does not touch and which is where the mask convention and the local-id
        # derivation live. The gate made that review delete the filename and describe the
        # header in prose to get through -- it was rewarding vagueness and punishing the
        # reader who went and looked. What it is actually for is stopping a finding that
        # is about some other PR, and one changed file named in the text settles that.
        if cited and not any(_is_touched(c) for c in cited):
            problems.append(("UNANCHORED-FINDING", text[:70],
                             f"cites only {cited[0]} and nothing this PR changes"))
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
                        "structabi", "commentonly", "card", "corefiles",
                        "kerneltest", "siblings", "guards", "refutations",
                        "independent"):
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
              "       triage.py corefiles <core_files.txt> <diff>\n"
              "       triage.py kerneltest <diff>\n"
              "       triage.py siblings <diff> <root>\n"
              "       triage.py guards <diff>\n"
              "       triage.py refutations <refutations.txt> <diff> <card.md>\n"
              "       triage.py independent <independent.txt> <card.md>\n"
              "       triage.py card <card.md> <verdicts> <diagnostic> <answers> <diff>", file=sys.stderr)
        raise SystemExit(2)

    if mode == "independent":
        out = audit_independent(
            open(sys.argv[2], errors="replace").read()
            if pathlib.Path(sys.argv[2]).exists() else "",
            open(sys.argv[3], errors="replace").read()
            if len(sys.argv) > 3 and pathlib.Path(sys.argv[3]).exists() else "")
        print("\n".join(out))
        sys.exit(0 if out[0].startswith(("INDEPENDENTLY REFUTED",
                                         "INDEPENDENT REFUTATION:")) else 1)
    if mode == "refutations":
        out = audit_refutations(open(sys.argv[2], errors="replace").read()
                                if pathlib.Path(sys.argv[2]).exists() else "",
                                open(sys.argv[3], errors="replace").read(),
                                open(sys.argv[4], errors="replace").read()
                                if len(sys.argv) > 4 and pathlib.Path(sys.argv[4]).exists() else "")
        print("\n".join(out))
        sys.exit(0 if out[0].startswith("REFUTATIONS COMPLETE") else 1)
    if mode == "guards":
        print(render_guard_changes(guard_changes(open(sys.argv[2]).read())))
        sys.exit(0)
    if mode == "siblings":
        print(render_siblings(sibling_variants(open(sys.argv[2]).read(), tree_root(sys.argv[3]))))
        sys.exit(0)
    if mode == "kerneltest":
        print(render_untested_kernel(untested_new_kernel(open(sys.argv[2]).read())))
        sys.exit(0)
    if mode == "corefiles":
        core_path = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
        if core_path is None or not core_path.is_file():
            # Fail closed, like the card gate: not writing the file was the cheapest way
            # past every gate that accepted its absence.
            print(f"CORE-FILES MISSING: {core_path}. Step 4 writes its assessment there "
                  f"before this gate runs", file=sys.stderr)
            raise SystemExit(1)
        try:
            diff_text = open(sys.argv[3], errors="replace").read()
        except (OSError, IndexError):
            diff_text = ""
        if not diff_text.strip():
            print(f"CORE-FILES UNUSABLE: no diff read from "
                  f"{sys.argv[3] if len(sys.argv) > 3 else '<missing>'}; without it the "
                  f"backbone set cannot be computed", file=sys.stderr)
            raise SystemExit(1)
        want, seen, problems = audit_core_files(
            core_path.read_text(errors="replace"), diff_text)
        for kind, what, why in problems:
            print(f"{kind}: {what}")
            print(f"  {why}")
        if problems:
            print(f"CORE-FILE ASSESSMENT INCOMPLETE: {len(want) - sum(1 for p, _ in want if p not in seen)}"
                  f"/{len(want)} backbone files assessed, {len(problems)} problem(s)",
                  file=sys.stderr)
            raise SystemExit(1)
        if not want:
            print(f"CORE FILES: none in this diff, declared")
        else:
            print(f"CORE FILES ASSESSED: {len(want)}/{len(want)} backbone files "
                  f"({sum(1 for _, t in want if t == '1')} tier-1), "
                  f"{sum(1 for v in seen.values() if v[1] == 'GAP')} gap(s)")
        raise SystemExit(0)

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
        root = tree_root(sys.argv[3] if len(sys.argv) > 3 else ".")
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
        root = tree_root(sys.argv[3] if len(sys.argv) > 3 else ".")
        diff_src = open(sys.argv[2], errors="replace").read()
        rows = uncovered_test_paths(diff_src, root)
        added = [f for f in changed_paths(diff_src)
                 if re.match(r"test_\w+\.py$|.*_test\.py$", f.rsplit("/", 1)[-1])]
        if not rows:
            # "every new test file lands where a CI job will run it" reads as a green
            # light on a PR that adds no test file at all, which is the opposite of what
            # SKILL.md asks the reader to conclude from silence.
            print("no new test file in this diff, so CI reachability was not exercised"
                  if not added else
                  "every new test file lands where a CI job will run it")
        for path, why in rows:
            print(f"UNRUN-TEST: {path}")
            print(f"  {why}")
        raise SystemExit(0)

    if mode == "twins":
        root = tree_root(sys.argv[3] if len(sys.argv) > 3 else ".")
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
        if not want and re.search(r"^\s*\[[\w-]+\s*\]\s*NONE\s*$", rules_text, re.M):
            # The deriver ran and correctly concluded there is nothing to adjudicate:
            # `[infra-only] NONE` for a diff confined to .github/, `[version-bump] NONE`
            # for a submodule pointer. `NONE` is not an id, so `want` is empty and this
            # gate used to read the right answer as proof the derivation never happened.
            # Three .github-only PRs in a 50-PR run were unpassable without either editing
            # the skill or writing rule ids into rules.txt that the deriver never emitted.
            print("LEDGER N/A: the deriver produced no rules for this diff "
                  "(%s), so there is nothing to adjudicate"
                  % re.search(r"^\s*\[([\w-]+)\s*\]\s*NONE", rules_text, re.M).group(1))
            raise SystemExit(0)
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
        root = tree_root(sys.argv[3] if len(sys.argv) > 3 else ".")
        bad = sweep_symbols(diff, root)
        # One line per distinct (import, reason). A PR importing a deleted module from
        # four files produced four identical lines, which reads as four problems.
        seen_pairs = set()
        bad = [b for b in bad if not (b in seen_pairs or seen_pairs.add(b))]
        if not bad:
            # SKILL.md reads an empty artifact as "that axis was not checked". A sweep
            # that ran and found nothing has to say so, or its silence is indistinguishable
            # from the skip it warns about.
            print("IMPORTS RESOLVED: every first-party import the diff adds exists in "
                  "the merge target or in this diff")
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
