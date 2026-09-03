"""Structural budget for the review-pr skill.

SKILL.md went 487 -> 632 -> 1372 lines between 2026-07-09 and 2026-09-03 while the
instruction content barely moved: at 1210 lines, 825 were inside code fences and only 299
were prose. A skill that long cannot be relied on to be read in full, and nothing stopped
it growing, so it grew. These tests are that stop.

Pure stdlib, no GPU, no network.
"""
import pathlib
import re
import subprocess
import sys
import unittest

SKILL_DIR = pathlib.Path(__file__).resolve().parents[1]
SKILL_MD = SKILL_DIR / "SKILL.md"
RULES_MD = SKILL_DIR / "rules.md"
TRIAGE = SKILL_DIR / "triage.py"
FETCH = SKILL_DIR / "fetch.sh"

# Chosen against the file's own history: the skill did its job at 487 lines, and 500 leaves
# room to edit an instruction without room to paste a program. Raising this number is a
# decision to make the entry file less likely to be read; make it deliberately, in a commit
# that says why, not as a side effect of adding content that belongs in rules.md or a script.
MAX_SKILL_LINES = 500
# A fence longer than this is a program. Programs go in a file next to this one and get
# called; that is how 785 lines of Step 1 bash ended up living in the document.
MAX_FENCE_LINES = 30


def fences(text):
    out, cur, lang = [], None, None
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            if cur is None:
                cur, lang = [], line.strip()[3:] or "plain"
            else:
                out.append((lang, cur))
                cur, lang = None, None
        elif cur is not None:
            cur.append(line)
    return out


class TestEntryBudget(unittest.TestCase):
    def test_skill_md_within_budget(self):
        n = len(SKILL_MD.read_text().split("\n"))
        self.assertLessEqual(
            n, MAX_SKILL_LINES,
            f"SKILL.md is {n} lines, over the {MAX_SKILL_LINES}-line budget. Conditional "
            f"content belongs in rules.md (emitted per-diff by `triage.py expand`); "
            f"executable content belongs in a script called from the doc, like fetch.sh.")

    def test_no_programs_embedded_in_the_document(self):
        long = [(lang, len(body)) for lang, body in fences(SKILL_MD.read_text())
                if len(body) > MAX_FENCE_LINES]
        self.assertEqual(
            long, [],
            f"code fences over {MAX_FENCE_LINES} lines: {long}. Put it in a script next to "
            f"SKILL.md and call it; a reader has to scroll past everything in the document.")


class TestRuleBodiesResolve(unittest.TestCase):
    """Every rule the deriver can emit must have text somewhere, or the reviewer is told to
    check a rule that is documented nowhere. Emitting 11 of 12 bodies reads exactly like
    emitting 12."""

    def _derivable_ids(self):
        src = TRIAGE.read_text()
        ids = set()
        for m in re.finditer(r'hit\("[\w-]+",\s*"([^"]+)"\)', src):
            ids |= set(m.group(1).split())
        for m in re.finditer(r'fams\.append\(\("[\w-]+",\s*"([^"]+)"\)\)', src):
            ids |= set(m.group(1).split())
        return {i for i in ids if re.fullmatch(r"[A-Z]+\d+[a-z]?", i)}

    def test_every_derivable_rule_has_a_body(self):
        rules = RULES_MD.read_text()
        blocks = set(re.findall(r"^\*\*([A-Z]+\d+[a-z]?) — ", rules, re.M))
        rows = set(re.findall(r"\b(HK\d+)\b", rules))
        documented = blocks | rows
        missing = sorted(self._derivable_ids() - documented - {"STEP4"})
        self.assertEqual(missing, [],
                         f"derivable but undocumented in rules.md: {missing}")

    def test_expand_emits_something_for_every_family(self):
        src = TRIAGE.read_text()
        rulesets = [m.group(1) for m in
                    re.finditer(r'hit\("[\w-]+",\s*"([^"]+)"\)', src)]
        rulesets += [m.group(1) for m in
                     re.finditer(r'fams\.append\(\("[\w-]+",\s*"([^"]+)"\)\)', src)]
        for rs in rulesets:
            if rs.strip() == "NONE":
                continue
            fake = f"    [x                   ] {rs}\n"
            tmp = SKILL_DIR / "tests" / ".rules_probe.txt"
            tmp.write_text(fake)
            try:
                r = subprocess.run([sys.executable, str(TRIAGE), "expand", str(tmp),
                                    str(RULES_MD)], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, f"{rs}: {r.stderr.strip()}")
                self.assertTrue(r.stdout.strip(), f"{rs} expanded to nothing")
            finally:
                tmp.unlink(missing_ok=True)


class TestLedgerGate(unittest.TestCase):
    def _ledger(self, rules, verdicts, diff=None):
        d = SKILL_DIR / "tests"
        rp, vp = d / ".r.txt", d / ".v.txt"
        rp.write_text(rules)
        vp.write_text(verdicts)
        argv = [sys.executable, str(TRIAGE), "ledger", str(rp), str(vp)]
        if diff is not None:
            dp = d / ".d.diff"
            dp.write_text(diff)
            argv.append(str(dp))
        try:
            return subprocess.run(argv, capture_output=True, text=True)
        finally:
            for f in (rp, vp, d / ".d.diff"):
                f.unlink(missing_ok=True)

    RULES = "    [async-stream        ] G1 G1b\n"

    def test_empty_rules_fails_closed(self):
        r = self._ledger("", "")
        self.assertEqual(r.returncode, 1)
        self.assertIn("LEDGER UNUSABLE", r.stderr)

    def test_no_verdicts_is_rejected(self):
        r = self._ledger(self.RULES, "")
        self.assertEqual(r.returncode, 1)
        self.assertIn("UNADJUDICATED: G1", r.stdout)

    def test_partial_adjudication_is_rejected(self):
        r = self._ledger(self.RULES, "G1 CLEAR -- no cross-stream handoff in the diff\n")
        self.assertEqual(r.returncode, 1)
        self.assertIn("UNADJUDICATED: G1b", r.stdout)

    def test_reasonless_verdicts_are_rejected(self):
        r = self._ledger(self.RULES, "G1 CLEAR -- ok\nG1b CLEAR -- ok\n")
        self.assertEqual(r.returncode, 1)
        self.assertIn("NO-EVIDENCE", r.stdout)

    def test_complete_ledger_passes(self):
        r = self._ledger(self.RULES,
                         "G1 CLEAR -- producer and consumer are on the default stream\n"
                         "G1b CLEAR -- nothing is captured into a cudagraph here\n")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("LEDGER COMPLETE", r.stdout)

    DIFF = "diff --git a/aiter/mla.py b/aiter/mla.py\n--- a/aiter/mla.py\n+++ b/aiter/mla.py\n"

    def test_fire_citing_an_untouched_file_is_rejected(self):
        r = self._ledger(self.RULES,
                         "G1 FIRE -- aiter/fused_moe.py:88 consumes it on another stream\n"
                         "G1b CLEAR -- nothing is captured into a cudagraph here\n",
                         diff=self.DIFF)
        self.assertEqual(r.returncode, 1)
        self.assertIn("UNTOUCHED-CITATION", r.stdout)

    def test_clear_may_cite_an_untouched_file(self):
        r = self._ledger(self.RULES,
                         "G1 CLEAR -- aiter/fused_moe.py is unchanged, no consumer moved\n"
                         "G1b CLEAR -- nothing is captured into a cudagraph here\n",
                         diff=self.DIFF)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class TestScriptsAreCallable(unittest.TestCase):
    def test_fetch_is_executable_and_parses(self):
        self.assertTrue(FETCH.exists(), "fetch.sh is missing")
        import os
        import stat
        self.assertTrue(os.stat(FETCH).st_mode & stat.S_IXUSR, "fetch.sh is not executable")
        r = subprocess.run(["bash", "-n", str(FETCH)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_skill_md_calls_fetch(self):
        """Calls it, not mentions it. Prose about fetch.sh satisfied a substring check
        while the call site was broken."""
        body = [b for lang, b in fences(SKILL_MD.read_text()) if lang == "bash"]
        invocations = [ln for blk in body for ln in blk
                       if "fetch.sh" in ln and not ln.lstrip().startswith("#")]
        self.assertTrue(invocations, "no bash block in SKILL.md invokes fetch.sh")


if __name__ == "__main__":
    unittest.main()


class TestDerivationIsPinned(unittest.TestCase):
    """Characterization test over 59 real aiter PRs.

    Nothing tested what `derive()` actually produces. Mutation testing made the hole
    explicit: flipping predicates in triage.py -- `and` to `or`, dropping a `not`,
    turning `p.count("/") == 1` into `!=` -- left the structural suite green on 150 of
    170 mutants. Those predicates decide which rules a reviewer is shown, so an edit
    could silently change every review and nothing would fail.

    The sample is stratified: 19 diffs chosen to hit all 31 families at least once, plus
    40 drawn at random with a fixed seed, out of the 600 open PRs collected 2026-09-03.

    When a change to the deriver is intended, the diff of tests/expected.json IS the
    review: it shows exactly which PRs would be triaged differently. Re-bless with
    `python3 tests/bless.py` and put that diff in the commit.
    """

    @classmethod
    def setUpClass(cls):
        import json
        import tarfile
        import tempfile
        cls._tmp = tempfile.TemporaryDirectory()
        with tarfile.open(SKILL_DIR / "tests" / "corpus.tgz") as t:
            t.extractall(cls._tmp.name)
        cls.corpus = pathlib.Path(cls._tmp.name)
        cls.titles = json.loads((cls.corpus / "titles.json").read_text())
        cls.expected = json.loads((SKILL_DIR / "tests" / "expected.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_rule_derivation_matches_the_pinned_corpus(self):
        drifted = []
        for pr, want in sorted(self.expected.items()):
            diff = self.corpus / f"{pr}.diff"
            r = subprocess.run(
                [sys.executable, str(TRIAGE), "rules", str(diff), self.titles.get(pr, "")],
                capture_output=True, text=True)
            self.assertEqual(0, r.returncode, f"PR#{pr}: {r.stderr}")
            if r.stdout != want["rules"]:
                drifted.append(f"PR#{pr}\n  expected:\n{want['rules']}  got:\n{r.stdout}")
        self.assertEqual([], drifted, "\n".join(drifted[:6]) +
                         f"\n({len(drifted)} of {len(self.expected)} PRs changed triage)")

    def test_every_corpus_pr_derives_at_least_one_family(self):
        """The deriver is conservative by design: a diff it cannot classify gets the full
        rule set, never an empty one. An empty result would silently clear a PR."""
        for pr, want in sorted(self.expected.items()):
            ids = set()
            for line in want["rules"].splitlines():
                m = re.match(r"\s*\[[\w-]+\s*\]\s*(.+)$", line)
                if m:
                    ids |= {t for t in m.group(1).split()
                            if re.fullmatch(r"[A-Z]+\d+[a-z]?|STEP4|NONE", t)}
            # The degraded paths (`diff-too-large`, `underivable`) emit the whole rule set
            # with no `rules=N/M` summary line, which is the intended fail-open: a diff
            # that cannot be classified must not come back looking clean.
            self.assertTrue(ids, f"PR#{pr} derived no rules at all")

    def test_expanded_bodies_stay_proportional(self):
        """The point of expansion is that the reviewer reads a fraction of rules.md. If a
        change makes most PRs pull most of the file, the derivation has stopped
        discriminating even if every individual output still looks reasonable."""
        import statistics
        total = len(RULES_MD.read_text().split("\n"))
        sizes = []
        for pr, want in sorted(self.expected.items()):
            rp = pathlib.Path(self._tmp.name) / f".rules-{pr}.txt"
            rp.write_text(want["rules"])
            r = subprocess.run([sys.executable, str(TRIAGE), "expand", str(rp),
                                str(RULES_MD)], capture_output=True, text=True)
            self.assertEqual(0, r.returncode, f"PR#{pr}: {r.stderr}")
            sizes.append(len(r.stdout.rstrip().split("\n")))
        median = statistics.median(sizes)
        self.assertLess(median / total, 0.45,
                        f"median expansion is {median}/{total} lines of rules.md; the "
                        f"derivation is no longer narrowing what gets read")


class TestFailClosedGuards(unittest.TestCase):
    """fetch.sh checks that its required inputs exist and exits 1 when they do not.

    Mutation testing turned every one of those `exit 1`s into `exit 0` and the suite
    stayed green: a missing scanner, schema, or production-scale file would let the run
    continue and report nothing, which is the shape of failure this whole skill keeps
    running into -- silence that reads like a clean result.

    This is a lint over the script's text, not an execution test: running fetch.sh needs
    a GitHub round trip. It pins the guard's shape, which is what the mutants broke.
    """

    GUARD = re.compile(r'^\s*echo "(required[^"]*|.*is missing[^"]*)" >&2\s*$')

    def test_every_required_input_guard_exits_nonzero(self):
        lines = FETCH.read_text().split("\n")
        guards, bad = 0, []
        for i, line in enumerate(lines):
            if not self.GUARD.match(line):
                continue
            guards += 1
            tail = [l.strip() for l in lines[i + 1:i + 4] if l.strip()]
            if not any(re.fullmatch(r"exit [1-9]\d*", t) for t in tail):
                bad.append(f"L{i+1}: {line.strip()} -> {tail[:2]}")
        self.assertGreaterEqual(guards, 4, "the required-input guards have gone missing")
        self.assertEqual([], bad, "guard reports an error then continues: " + str(bad))

    def test_script_aborts_on_error(self):
        """`set -euo pipefail` is what makes an unchecked failure stop the run at all.
        $BASE_SHA/$HEAD_SHA were referenced before assignment for weeks; `set -u` is why
        that was a loud abort rather than two empty strings silently comparing equal."""
        head = "\n".join(FETCH.read_text().split("\n")[:20])
        self.assertIn("set -euo pipefail", head)


class TestSymbolSweep(unittest.TestCase):
    """Every branch of the import sweep, as its own case.

    After the corpus was pinned, mutation testing put a third of the surviving mutants in
    sweep_symbols/symbol_defined/resolve_module: only 6 of the 59 corpus PRs produce any
    UNRESOLVED-IMPORT line, so the paths that decide NOT to report were reached by almost
    nothing. Those are the paths that matter -- this check is silent when it is wrong, and
    silence is indistinguishable from a clean PR.

    Each behaviour below was hand-verified when it was written and then not encoded.
    """

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, rel, text=""):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def sweep(self, diff_text):
        d = self.root / "the.diff"
        d.write_text(diff_text)
        r = subprocess.run([sys.executable, str(TRIAGE), "symbols", str(d), str(self.root)],
                           capture_output=True, text=True)
        self.assertEqual(0, r.returncode, r.stderr)
        return [l for l in r.stdout.splitlines() if l.startswith("UNRESOLVED-IMPORT")]

    @staticmethod
    def diff_adding(path, *added_lines, new_file=False):
        head = f"diff --git a/{path} b/{path}\n"
        if new_file:
            head += "new file mode 100644\n"
        head += f"--- a/{path}\n+++ b/{path}\n@@ -0,0 +1 @@\n"
        return head + "".join(f"+{l}\n" for l in added_lines)

    def test_missing_module_is_reported(self):
        self.write("aiter/__init__.py")
        out = self.sweep(self.diff_adding(
            "aiter/x.py", "from aiter.gone import thing"))
        self.assertEqual(1, len(out), out)
        self.assertIn("aiter.gone", out[0])

    def test_missing_symbol_in_an_existing_module_is_reported(self):
        self.write("aiter/__init__.py")
        self.write("aiter/real.py", "def present():\n    pass\n")
        out = self.sweep(self.diff_adding(
            "aiter/x.py", "from aiter.real import absent"))
        self.assertEqual(1, len(out), out)
        self.assertIn("'absent'", out[0])

    def test_module_added_by_this_very_pr_is_not_reported(self):
        self.write("aiter/__init__.py")
        out = self.sweep(
            self.diff_adding("aiter/brand_new.py", "def f():", new_file=True)
            + self.diff_adding("aiter/x.py", "from aiter.brand_new import f"))
        self.assertEqual([], out)

    def test_symbol_defined_by_this_very_pr_is_not_reported(self):
        self.write("aiter/__init__.py")
        self.write("aiter/real.py", "def old():\n    pass\n")
        out = self.sweep(
            self.diff_adding("aiter/real.py", "def freshly_added():")
            + self.diff_adding("aiter/x.py", "from aiter.real import freshly_added"))
        self.assertEqual([], out)

    def test_namespace_package_is_not_reported(self):
        """A directory with no __init__.py imports fine. Requiring __init__.py called
        aiter/utility -- a real namespace package -- hallucinated."""
        self.write("aiter/__init__.py")
        self.write("aiter/utility/thing.py", "x = 1\n")
        out = self.sweep(self.diff_adding(
            "aiter/x.py", "from aiter.utility import anything"))
        self.assertEqual([], out)

    def test_submodule_binding_is_not_reported(self):
        """`from pkg import sub` binds a SUBMODULE, which need not appear in the package's
        __init__.py. Searching only the __init__ text called aiter.jit/core,
        aiter.jit.utils/chip_info and flydsl.kernels/{buffer_ops,vector} invented."""
        self.write("aiter/__init__.py")
        self.write("aiter/jit/__init__.py", "")
        self.write("aiter/jit/core.py", "def build():\n    pass\n")
        out = self.sweep(self.diff_adding("aiter/x.py", "from aiter.jit import core"))
        self.assertEqual([], out)

    def test_star_import_and_dunder_all_are_not_reported(self):
        self.write("aiter/__init__.py")
        self.write("aiter/reexport.py", "from .other import *\n")
        self.write("aiter/listed.py", "__all__ = ['maybe']\n")
        self.assertEqual([], self.sweep(self.diff_adding(
            "aiter/x.py", "from aiter.reexport import whatever")))
        self.assertEqual([], self.sweep(self.diff_adding(
            "aiter/x.py", "from aiter.listed import maybe")))

    def test_third_party_and_relative_imports_are_skipped(self):
        self.write("aiter/__init__.py")
        out = self.sweep(self.diff_adding(
            "aiter/x.py",
            "import torch", "from numpy import zeros", "import triton",
            "from . import sibling", "from nonexistent_vendor import thing"))
        self.assertEqual([], out)

    def test_removed_lines_are_not_treated_as_added_imports(self):
        """A diff that DELETES an import of a module that is now gone is the fix, not the
        defect. Only `+` lines count."""
        self.write("aiter/__init__.py")
        diff = ("diff --git a/aiter/x.py b/aiter/x.py\n--- a/aiter/x.py\n+++ b/aiter/x.py\n"
                "@@ -1 +0,0 @@\n-from aiter.gone import thing\n")
        self.assertEqual([], self.sweep(diff))


class TestCitationAnchoring(unittest.TestCase):
    """check_citations decides whether a FIRE is anchored to code this PR touches. Seven
    mutants survived in it after the corpus was pinned, because the ledger tests exercised
    one path through it."""

    DIFF = ("diff --git a/aiter/mla.py b/aiter/mla.py\n--- a/aiter/mla.py\n+++ b/aiter/mla.py\n"
            "diff --git a/csrc/kernels/deep/attn.cu b/csrc/kernels/deep/attn.cu\n"
            "--- a/csrc/kernels/deep/attn.cu\n+++ b/csrc/kernels/deep/attn.cu\n")
    RULES = "    [async-stream        ] G1 G1b\n"

    def ledger(self, verdicts):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            (d / "r.txt").write_text(self.RULES)
            (d / "v.txt").write_text(verdicts)
            (d / "d.diff").write_text(self.DIFF)
            return subprocess.run(
                [sys.executable, str(TRIAGE), "ledger", str(d / "r.txt"),
                 str(d / "v.txt"), str(d / "d.diff")], capture_output=True, text=True)

    OK_SECOND = "G1b CLEAR -- nothing is captured into a cudagraph in this diff\n"

    def test_fire_on_a_changed_file_passes(self):
        r = self.ledger("G1 FIRE -- aiter/mla.py:120 hands the buffer across streams\n"
                        + self.OK_SECOND)
        self.assertEqual(0, r.returncode, r.stdout + r.stderr)

    def test_fire_on_a_changed_file_without_a_line_number_passes(self):
        r = self.ledger("G1 FIRE -- aiter/mla.py hands the buffer across streams\n"
                        + self.OK_SECOND)
        self.assertEqual(0, r.returncode, r.stdout + r.stderr)

    def test_fire_on_a_nested_path_passes(self):
        r = self.ledger("G1 FIRE -- csrc/kernels/deep/attn.cu:44 no fence before the read\n"
                        + self.OK_SECOND)
        self.assertEqual(0, r.returncode, r.stdout + r.stderr)

    def test_fire_on_an_untouched_file_is_rejected(self):
        r = self.ledger("G1 FIRE -- aiter/fused_moe.py:88 consumes it on another stream\n"
                        + self.OK_SECOND)
        self.assertEqual(1, r.returncode)
        self.assertIn("UNTOUCHED-CITATION", r.stdout)

    def test_fire_citing_no_file_at_all_passes(self):
        """Not every real defect has a path in its one-line reason. Demanding one would
        push the reviewer to invent a citation, which is the failure being prevented."""
        r = self.ledger("G1 FIRE -- the producer and consumer streams are never joined\n"
                        + self.OK_SECOND)
        self.assertEqual(0, r.returncode, r.stdout + r.stderr)

    def test_clear_may_cite_an_untouched_file(self):
        r = self.ledger("G1 CLEAR -- aiter/fused_moe.py is unchanged, no consumer moved\n"
                        + self.OK_SECOND)
        self.assertEqual(0, r.returncode, r.stdout + r.stderr)

    def test_na_may_cite_an_untouched_file(self):
        r = self.ledger("G1 N/A -- aiter/fused_moe.py holds the only stream handoff\n"
                        + self.OK_SECOND)
        self.assertEqual(0, r.returncode, r.stdout + r.stderr)

    def test_citation_check_is_skipped_without_a_diff(self):
        """Called with two arguments the gate still checks completeness. It must not start
        rejecting citations it has no diff to judge."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            (d / "r.txt").write_text(self.RULES)
            (d / "v.txt").write_text(
                "G1 FIRE -- aiter/fused_moe.py:88 consumes it elsewhere\n" + self.OK_SECOND)
            r = subprocess.run([sys.executable, str(TRIAGE), "ledger",
                                str(d / "r.txt"), str(d / "v.txt")],
                               capture_output=True, text=True)
        self.assertEqual(0, r.returncode, r.stdout + r.stderr)
