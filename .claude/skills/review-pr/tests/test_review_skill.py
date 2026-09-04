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
# fetch.sh took the 785 lines Step 1 used to carry. Moving entropy is not removing it: the
# entry file grew unchecked the first time precisely because nothing counted it. This is
# generous on purpose -- it is a program, and it is not in the reviewer's reading path --
# but it is counted.
MAX_FETCH_LINES = 900


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


    def test_fetch_script_within_budget(self):
        n = len(FETCH.read_text().split("\n"))
        self.assertLessEqual(
            n, MAX_FETCH_LINES,
            f"fetch.sh is {n} lines, over the {MAX_FETCH_LINES}-line budget. Step 1 is a "
            f"pipeline of independent stages; split one out rather than growing this.")


class TestMappingIsGenerated(unittest.TestCase):
    """MAPPING.md must equal what the deriver emits.

    Step 3 used to carry this table by hand in SKILL.md and it lied: D9 listed as derived
    when it is scanner-backed and deliberately is not, and 21 derived rules missing,
    including all six Triton ones. A mapping the reviewer is explicitly told not to apply
    by hand should not be maintained by hand either."""

    def test_committed_mapping_matches_the_deriver(self):
        r = subprocess.run([sys.executable, str(TRIAGE), "mapping"],
                           capture_output=True, text=True)
        self.assertEqual(0, r.returncode, r.stderr)
        committed = (SKILL_DIR / "MAPPING.md").read_text()
        self.assertEqual(
            r.stdout, committed,
            "MAPPING.md is stale. Regenerate: python3 triage.py mapping > MAPPING.md")

    def test_skill_points_at_the_generated_mapping(self):
        body = SKILL_MD.read_text()
        step3 = body[body.index("## Step 3"):body.index("## Step 4")]
        self.assertIn("MAPPING.md", step3)
        self.assertNotRegex(
            step3, r"→\s*[A-Z]+\d",
            "Step 3 carries a hand-written type -> rule mapping again")


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

    def test_no_documented_rule_is_unreachable(self):
        """A rule in rules.md that no family emits is read by nobody, ever.

        Sixteen were: A2, B5, C3, D1b, E2, E3, P4, HK1/2/3/5/7/8/10/11 and D9. Their own
        trigger text is structural -- "`# TODO` on a `+` line", "develop=True in added
        code" -- they were simply never wired to a family. This is invisible from inside a
        review: the ledger can only demand a verdict for rules that were derived, so an
        unreachable rule is a silent hole rather than a missing line.

        Deliberate exclusions go in triage.py's UNREACHABLE_BY_DESIGN with the reason,
        which makes leaving one out a decision rather than an oversight.
        """
        import re as _re
        src = TRIAGE.read_text()
        rules = RULES_MD.read_text()
        documented = set(_re.findall(r"^\*\*([A-Z]+\d+[a-z]?) — ", rules, _re.M)) \
            | set(_re.findall(r"\b(HK\d+)\b", rules))
        block = _re.search(r"UNREACHABLE_BY_DESIGN = \{(.*?)\n\}", src, _re.S)
        self.assertIsNotNone(block, "UNREACHABLE_BY_DESIGN is gone")
        exempt = set(_re.findall(r'"(\w+)":\s*"', block.group(1)))
        unreachable = sorted(documented - self._derivable_ids() - exempt)
        self.assertEqual(
            unreachable, [],
            f"documented but derivable by no family, so never read: {unreachable}. "
            f"Wire each to the family its own Trigger line describes, or add it to "
            f"UNREACHABLE_BY_DESIGN with the reason.")

    def test_exemptions_carry_a_reason(self):
        import re as _re
        block = _re.search(r"UNREACHABLE_BY_DESIGN = \{(.*?)\n\}", TRIAGE.read_text(), _re.S)
        for rid, why in _re.findall(r'"(\w+)":\s*"([^"]*)"', block.group(1)):
            self.assertGreater(len(why), 15, f"{rid} is exempted with no real reason")

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

    def test_submodule_added_by_this_pr_is_not_reported(self):
        """`from pkg import sub` where the PR adds pkg/sub.py.

        The added-module check covered the dotted module itself but not this shape, so a
        PR that ships a new submodule and imports it by name was reported against its own
        new file. Found by running the skill on aiter#5157, which adds
        aiter/ops/triton/attention/kr_ua.py and imports `kr_ua` from the package."""
        self.write("aiter/__init__.py")
        self.write("aiter/ops/__init__.py")
        out = self.sweep(
            self.diff_adding("aiter/ops/fresh.py", '"""new"""', new_file=True)
            + self.diff_adding("aiter/x.py", "from aiter.ops import fresh"))
        self.assertEqual([], out)

    def test_a_renamed_destination_is_not_reported(self):
        """git writes a move as `rename from`/`rename to`, with no `new file mode` and,
        at similarity 100%, no hunks. Detecting only new-file markers made every
        reorganisation PR report its own destination paths: aiter#5254 moves 74 files
        under a new package with 53 renames and produced 83 UNRESOLVED-IMPORT lines, each
        pointing at a path the PR itself creates."""
        self.write("aiter/__init__.py")
        self.write("aiter/old/thing.py", "def f():\n    pass\n")
        moved = ("diff --git a/aiter/old/thing.py b/aiter/new/thing.py\n"
                 "similarity index 100%\n"
                 "rename from aiter/old/thing.py\n"
                 "rename to aiter/new/thing.py\n")
        out = self.sweep(moved + self.diff_adding(
            "aiter/x.py", "from aiter.new.thing import f"))
        self.assertEqual([], out)

    def test_a_copy_destination_is_not_reported(self):
        self.write("aiter/__init__.py")
        self.write("aiter/old/thing.py", "def f():\n    pass\n")
        copied = ("diff --git a/aiter/old/thing.py b/aiter/copy/thing.py\n"
                  "similarity index 100%\n"
                  "copy from aiter/old/thing.py\n"
                  "copy to aiter/copy/thing.py\n")
        self.assertEqual([], self.sweep(copied + self.diff_adding(
            "aiter/x.py", "from aiter.copy.thing import f")))

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

    def test_env_knobs_in_a_benchmark_script_do_not_fire(self):
        """HK9 is about a permanent runtime knob. PROF_WARMUP/PROF_ITERS in
        op_tests/flydsl_tests/profile_flydsl_bwd.py are not one, and firing there teaches
        the reviewer the family is noise (aiter#3976 adds three, all in op_tests/)."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td) / "b.diff"
            d.write_text(self.diff_adding(
                "op_tests/flydsl_tests/profile_flydsl_bwd.py",
                'WARMUP = int(os.environ.get("PROF_WARMUP", 5))', new_file=True))
            r = subprocess.run([sys.executable, str(TRIAGE), "rules", str(d), "x"],
                               capture_output=True, text=True)
        self.assertNotIn("new-env-var", r.stdout)

    def test_env_knobs_without_the_aiter_prefix_are_derived(self):
        """HK9 exists because aiter rejects permanent env flags. The family's pattern
        required an AITER_ prefix, so aiter#5157 -- six permanent switches named
        KR_TAILSPLIT, KR_TS_OCC, KR_TS_MAX_SEG, KR_TS_MIN_SEG, KR_TS_CAP_BLOCKS,
        KR_TS_TRIM_GRID -- derived no env-var family at all. Found by running the skill."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td) / "k.diff"
            d.write_text(self.diff_adding(
                "aiter/ops/triton/attention/kr_ua.py",
                'ENABLED = bool(int(os.environ.get("KR_TAILSPLIT", "0")))',
                '_OCC = int(os.environ.get("KR_TS_OCC", "8"))', new_file=True))
            r = subprocess.run([sys.executable, str(TRIAGE), "rules", str(d), "x"],
                               capture_output=True, text=True)
        self.assertIn("new-env-var", r.stdout,
                      "a new permanent env knob derived no HK9/D2")

    def test_a_broken_relative_import_is_reported(self):
        """`from .gone import x` inside aiter/jit/utils/asm_guard.py resolves to
        aiter.jit.utils.gone. Skipping relative imports left the case where an
        unresolvable one is worst: a Tier-1 PR wiring a new module into
        aiter/__init__.py, where it breaks `import aiter` outright rather than at the
        first call. Found reviewing aiter#5107, whose chain is sound -- but only checked
        by hand, because the sweep said nothing."""
        self.write("aiter/__init__.py")
        self.write("aiter/jit/__init__.py")
        self.write("aiter/jit/utils/__init__.py")
        out = self.sweep(self.diff_adding(
            "aiter/jit/utils/asm_guard.py", "from .gone import thing", new_file=True))
        self.assertEqual(1, len(out), out)
        self.assertIn("aiter.jit.utils.gone", out[0])

    def test_a_relative_import_that_resolves_is_not_reported(self):
        self.write("aiter/__init__.py")
        self.write("aiter/jit/__init__.py")
        self.write("aiter/jit/utils/__init__.py")
        self.write("aiter/jit/utils/chip_info.py", "def get_gfx_runtime():\n    pass\n")
        self.assertEqual([], self.sweep(self.diff_adding(
            "aiter/jit/utils/asm_guard.py",
            "from .chip_info import get_gfx_runtime", new_file=True)))

    def test_a_parent_relative_import_resolves_one_level_up(self):
        self.write("aiter/__init__.py")
        self.write("aiter/jit/__init__.py")
        self.write("aiter/jit/core.py", "def build():\n    pass\n")
        self.write("aiter/jit/utils/__init__.py")
        self.assertEqual([], self.sweep(self.diff_adding(
            "aiter/jit/utils/asm_guard.py", "from ..core import build", new_file=True)))

    def test_a_relative_import_escaping_the_tree_is_left_alone(self):
        self.write("aiter/__init__.py")
        self.assertEqual([], self.sweep(self.diff_adding(
            "aiter/x.py", "from ....way.up import thing")))

    def test_imports_in_a_non_python_file_are_ignored(self):
        """docs/tutorials/add_new_op.rst walks a reader through writing an op, imports
        and all. Two PRs were reported for a module named in a tutorial."""
        self.write("aiter/__init__.py")
        doc = ("diff --git a/docs/tutorials/add_new_op.rst b/docs/tutorials/add_new_op.rst\n"
               "--- a/docs/tutorials/add_new_op.rst\n+++ b/docs/tutorials/add_new_op.rst\n"
               "@@ -0,0 +1,2 @@\n"
               "+   from ..jit.core import compile_ops\n"
               "+   from aiter.ops.nonexistent import thing\n")
        self.assertEqual([], self.sweep(doc))

    def test_third_party_imports_are_skipped(self):
        self.write("aiter/__init__.py")
        out = self.sweep(self.diff_adding(
            "aiter/x.py",
            "import torch", "from numpy import zeros", "import triton",
            "from nonexistent_vendor import thing"))
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


class TestAnswersGate(unittest.TestCase):
    """Step 2's five questions and Step 7.5's blind-spot question, as a checkable artifact.

    They were `_Answer:_` blanks in prose. A review that skipped them left a file
    identical to one that did them, so nothing downstream could tell -- the same shape as
    D9 before its scan moved into Step 1, and as the rule pass before the ledger.
    """

    GOOD = (
        "Q1: splits the trailing q-blocks of the 2-D launch across the KV dimension and "
        "merges the per-segment acc/max/expsum\n"
        "Q2: gfx950 prefill only; bf16 and fp8 KV cache share the path; decode is excluded "
        "by max_seqlen_q == 1\n"
        "Q3: no public change -- unified_attention keeps its signature and the new entry "
        "lives in a separate module\n"
        "Q4: past ~2048 co-resident workgroups the remainder runs as a nearly empty extra "
        "generation that is latency-bound\n"
        "Q5: explains why -- the docstring gives the measured staircase at 256/257/305 "
        "q-blocks and names co-residency\n"
        "BLIND: the buffer cache is keyed by row count and never evicted, so a serving "
        "process accumulates one buffer per distinct prefill length\n")

    def run_gate(self, text):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            f = pathlib.Path(td) / "answers.txt"
            f.write_text(text)
            return subprocess.run([sys.executable, str(TRIAGE), "answers", str(f)],
                                  capture_output=True, text=True)

    def test_complete_answers_pass(self):
        r = self.run_gate(self.GOOD)
        self.assertEqual(0, r.returncode, r.stdout + r.stderr)
        self.assertIn("ANSWERS COMPLETE: 6/6", r.stdout)

    def test_missing_file_is_rejected(self):
        r = self.run_gate("")
        self.assertEqual(1, r.returncode)
        self.assertEqual(6, r.stdout.count("UNANSWERED"))

    def test_a_skipped_question_is_named(self):
        without_q4 = "\n".join(l for l in self.GOOD.split("\n")
                               if not l.startswith("Q4:"))
        r = self.run_gate(without_q4)
        self.assertEqual(1, r.returncode)
        self.assertIn("UNANSWERED: Q4", r.stdout)

    def test_formulaic_answers_are_rejected(self):
        r = self.run_gate("Q1: added a kernel\nQ2: gfx950\nQ3: no\nQ4: faster\n"
                          "Q5: yes\nBLIND: no\n")
        self.assertEqual(1, r.returncode)
        self.assertEqual(6, r.stdout.count("NO-SUBSTANCE"))

    def test_a_bare_no_to_the_blind_spot_question_is_rejected(self):
        """The answer that costs nothing to give and is worth nothing to receive."""
        r = self.run_gate(self.GOOD.replace(
            "BLIND: the buffer cache is keyed by row count and never evicted, so a serving "
            "process accumulates one buffer per distinct prefill length\n",
            "BLIND: none\n"))
        self.assertEqual(1, r.returncode)
        self.assertIn("BLIND", r.stdout)

    def test_step_8_runs_both_gates(self):
        body = SKILL_MD.read_text()
        step8 = body[body.index("## Step 8"):]
        for mode in ("answers", "ledger"):
            self.assertRegex(step8, rf"triage\.py\"?\s+{mode}\b",
                             f"Step 8 does not run the {mode} gate")


class TestSweepSafetyValves(unittest.TestCase):
    """The paths in symbol_defined/sweep_symbols that decide NOT to accuse.

    Mutation flipped each `return True` -- the valves that say "cannot disprove this, so
    say nothing" -- to False and the suite stayed green. Inverted, they turn the sweep
    from a rebase signal into a generator of false accusations against working imports,
    which is the failure mode that gets a whole class of finding dismissed.
    """

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        (self.root / "aiter").mkdir()
        (self.root / "aiter" / "__init__.py").write_text("")

    def tearDown(self):
        import os
        import stat
        for p in self.root.rglob("*"):
            try:
                os.chmod(p, stat.S_IRWXU)
            except OSError:
                pass
        self._tmp.cleanup()

    def sweep(self, diff_text):
        d = self.root / "the.diff"
        d.write_text(diff_text)
        r = subprocess.run([sys.executable, str(TRIAGE), "symbols", str(d), str(self.root)],
                           capture_output=True, text=True)
        self.assertEqual(0, r.returncode, r.stderr)
        return [l for l in r.stdout.splitlines() if l.startswith("UNRESOLVED-IMPORT")]

    IMPORT = ("diff --git a/aiter/x.py b/aiter/x.py\n--- a/aiter/x.py\n+++ b/aiter/x.py\n"
              "@@ -0,0 +1 @@\n+from aiter.real import {name}\n")

    def test_a_function_definition_resolves(self):
        (self.root / "aiter" / "real.py").write_text("def present():\n    pass\n")
        self.assertEqual([], self.sweep(self.IMPORT.format(name="present")))

    def test_a_class_definition_resolves(self):
        (self.root / "aiter" / "real.py").write_text("class Present:\n    pass\n")
        self.assertEqual([], self.sweep(self.IMPORT.format(name="Present")))

    def test_a_module_level_binding_resolves(self):
        """`NAME = ...` and `NAME: T = ...` are importable and are not def/class."""
        (self.root / "aiter" / "real.py").write_text("PRESENT = 3\nTYPED: int = 4\n")
        self.assertEqual([], self.sweep(self.IMPORT.format(name="PRESENT")))
        self.assertEqual([], self.sweep(self.IMPORT.format(name="TYPED")))

    def test_an_unreadable_module_is_not_accused(self):
        import os
        import stat
        f = self.root / "aiter" / "real.py"
        f.write_text("def something():\n    pass\n")
        os.chmod(f, 0)
        try:
            if os.access(f, os.R_OK):
                self.skipTest("running as root; permissions do not deny reads")
            self.assertEqual([], self.sweep(self.IMPORT.format(name="whatever")))
        finally:
            os.chmod(f, stat.S_IRWXU)

    def test_only_added_lines_are_scanned(self):
        """parse() must separate + from - and must not count the +++ header as an added
        line; both were single mutable predicates with nothing covering them."""
        (self.root / "aiter" / "real.py").write_text("x = 1\n")
        removed = ("diff --git a/aiter/x.py b/aiter/x.py\n--- a/aiter/x.py\n"
                   "+++ b/aiter/x.py\n@@ -1 +0,0 @@\n-from aiter.ghost import thing\n")
        self.assertEqual([], self.sweep(removed))

    def test_new_file_marker_is_required_for_the_added_module_exemption(self):
        """A modified file is not an added one. Without `new file mode` the module must
        still resolve against the tree."""
        modified = ("diff --git a/aiter/ghost.py b/aiter/ghost.py\n--- a/aiter/ghost.py\n"
                    "+++ b/aiter/ghost.py\n@@ -1 +1,2 @@\n+def thing():\n"
                    "diff --git a/aiter/x.py b/aiter/x.py\n--- a/aiter/x.py\n"
                    "+++ b/aiter/x.py\n@@ -0,0 +1 @@\n+from aiter.ghost import thing\n")
        # `thing` is added by the diff, so the symbol is exempt either way; the module
        # itself does not exist and is not marked new, so it must be reported.
        out = self.sweep(modified)
        self.assertEqual(1, len(out), out)
        self.assertIn("aiter.ghost", out[0])


class TestFamilyAlternatives(unittest.TestCase):
    """Each alternative in a family's path or pattern test.

    `or` -> `and` inside these left the suite green: the corpus happens to exercise one
    branch of each, so the others were carried by nothing.
    """

    def rules_for(self, *paths, added=("x = 1",), title="t", new=True):
        import tempfile
        body = ""
        for p in paths:
            body += f"diff --git a/{p} b/{p}\n"
            if new:
                body += "new file mode 100644\n"
            body += f"--- a/{p}\n+++ b/{p}\n@@ -0,0 +1 @@\n"
            body += "".join(f"+{l}\n" for l in added)
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td) / "d.diff"
            d.write_text(body)
            r = subprocess.run([sys.executable, str(TRIAGE), "rules", str(d), title],
                               capture_output=True, text=True)
        self.assertEqual(0, r.returncode, r.stderr)
        return r.stdout

    def test_every_triton_path_shape_derives_the_triton_families(self):
        for path in ("aiter/ops/triton/foo.py",
                     "aiter/ops/triton/_triton_kernels/foo.py",
                     "aiter/x/_gluon_kernels/foo.py",
                     "aiter/x/gluon/foo.py"):
            out = self.rules_for(path, added=("tl.load(p)", "num_warps = 4"))
            self.assertIn("triton-", out, f"{path} derived no Triton family")

    def test_every_plugin_bridge_name_derives_e3(self):
        for path in ("aiter/bridge_thing.py", "aiter/plugin_thing.py", "aiter/thing_ext.py"):
            self.assertIn("plugin-bridge", self.rules_for(path), path)

    def test_develop_true_derives_hk8_with_or_without_compile_ops(self):
        self.assertIn("develop-flag", self.rules_for(
            "aiter/ops/x.py", added=("@compile_ops(\"x\", develop=True)",)))
        self.assertIn("develop-flag", self.rules_for(
            "aiter/ops/x.py", added=("register(develop=True)",)))

    def test_docs_directory_counts_as_infra(self):
        self.assertIn("infra-only", self.rules_for("aiter/docs/notes.rst"))


class TestBoundariesAndDegradedPaths(unittest.TestCase):
    """Thresholds at their exact boundary, and the paths taken when input is unusable.

    `<` -> `<=` on MIN_EVIDENCE and MIN_ANSWER survived because every test sat far from
    the threshold. A gate whose boundary nobody pinned drifts by one character at a time.
    """

    def tmpfile(self, td, name, text):
        f = pathlib.Path(td) / name
        f.write_text(text)
        return f

    def ledger(self, verdicts):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            r = self.tmpfile(td, "r.txt", "    [async-stream        ] G1\n")
            v = self.tmpfile(td, "v.txt", verdicts)
            return subprocess.run([sys.executable, str(TRIAGE), "ledger", str(r), str(v)],
                                  capture_output=True, text=True)

    def test_min_evidence_boundary_is_exact(self):
        import re as _re
        n = int(_re.search(r"MIN_EVIDENCE = (\d+)", TRIAGE.read_text()).group(1))
        self.assertEqual(1, self.ledger(f"G1 CLEAR -- {'x' * (n - 1)}\n").returncode,
                         "one character under the threshold was accepted")
        self.assertEqual(0, self.ledger(f"G1 CLEAR -- {'x' * n}\n").returncode,
                         "a reason exactly at the threshold was rejected")

    def answers(self, text):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            f = self.tmpfile(td, "a.txt", text)
            return subprocess.run([sys.executable, str(TRIAGE), "answers", str(f)],
                                  capture_output=True, text=True)

    def test_min_answer_boundary_is_exact(self):
        import re as _re
        n = int(_re.search(r"MIN_ANSWER = (\d+)", TRIAGE.read_text()).group(1))
        def sheet(k):
            return "".join(f"{q}: {'x' * k}\n" for q in ("Q1","Q2","Q3","Q4","Q5","BLIND"))
        self.assertEqual(1, self.answers(sheet(n - 1)).returncode)
        self.assertEqual(0, self.answers(sheet(n)).returncode)

    def test_an_unparseable_diff_falls_back_to_the_whole_rule_set(self):
        """Fail open, loudly: a diff that cannot be classified must not come back looking
        like a small, clean PR."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = self.tmpfile(td, "empty.diff", "")
            r = subprocess.run([sys.executable, str(TRIAGE), "rules", str(d), ""],
                               capture_output=True, text=True)
        self.assertIn("underivable", r.stdout)
        self.assertIn("STEP4", r.stdout)

    def test_a_github_truncation_notice_is_recognised(self):
        """`gh pr diff` on a huge PR returns a JSON error, not a diff. Treating that as an
        empty diff would clear the largest PRs in the queue."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = self.tmpfile(td, "big.diff",
                             '{"message": "exceeded the maximum number of lines"}\n')
            r = subprocess.run([sys.executable, str(TRIAGE), "rules", str(d), ""],
                               capture_output=True, text=True)
        self.assertIn("diff-too-large", r.stdout)
        self.assertIn("STEP4", r.stdout)
        # and it must say how to recover, not just that it gave up
        self.assertIn("pulls/N/files", r.stderr + r.stdout)

    def test_housekeeping_rows_expand_with_their_table_header(self):
        """HK rules are table rows. Emitting the row without the header gives the reviewer
        a bare `| ... | ... |` with no idea which column is the trigger."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            r = self.tmpfile(td, "r.txt", "    [x                   ] HK7 HK8\n")
            out = subprocess.run([sys.executable, str(TRIAGE), "expand", str(r),
                                  str(RULES_MD)], capture_output=True, text=True)
        self.assertEqual(0, out.returncode, out.stderr)
        self.assertIn("### Housekeeping", out.stdout)
        self.assertIn("| Check | Trigger | Flag |", out.stdout)
        self.assertIn("HK7", out.stdout)
        self.assertIn("HK8", out.stdout)

    def test_a_category_heading_is_emitted_once_per_run(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            r = self.tmpfile(td, "r.txt", "    [x                   ] E4 E5\n")
            out = subprocess.run([sys.executable, str(TRIAGE), "expand", str(r),
                                  str(RULES_MD)], capture_output=True, text=True)
        self.assertEqual(1, out.stdout.count("### E — Cross-Repo Sync"))


class TestEvidenceCollector(unittest.TestCase):
    """The cross-file evidence path had no test of its own.

    It exists because prose telling a reviewer to go and grep was read and not acted on,
    producing a withdrawn `q_out is not None` finding on aiter#5143. Every predicate in it
    -- which lines count as deleted, which of those name a guard, which head files exist,
    which hits are worth printing -- was carried by nothing.
    """

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def run_evidence(self, diff_text, head_files):
        d = self.root / "d.diff"
        d.write_text(diff_text)
        paths = []
        for name, body in head_files.items():
            f = self.root / name
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(body)
            paths.append(str(f))
        r = subprocess.run([sys.executable, str(TRIAGE), "evidence", str(d)] + paths,
                           capture_output=True, text=True)
        self.assertEqual(0, r.returncode, r.stderr)
        return r.stdout

    DELETED_GUARD = ("diff --git a/csrc/k.cu b/csrc/k.cu\n--- a/csrc/k.cu\n+++ b/csrc/k.cu\n"
                     "@@ -1 +0,0 @@\n-  TORCH_CHECK(q_out is not None);\n")

    def test_a_removed_guard_is_looked_up_on_head(self):
        out = self.run_evidence(self.DELETED_GUARD,
                                {"k.cu": "if (q_out.has_value()) { use(q_out); }\n"})
        self.assertIn("q_out on head", out)
        self.assertIn("has_value", out)

    def test_a_removed_contiguity_normalisation_is_collected(self):
        """The invariant-removed family fires on a deleted `.contiguous()`, but the
        collector only read assert/*_CHECK lines, so the most common shape of the finding
        produced no evidence and the reviewer was back to grepping. aiter#5235 relocates
        two such lines -- the answer is "it moved", and the collector should show that."""
        moved = ("diff --git a/aiter/x.py b/aiter/x.py\n--- a/aiter/x.py\n+++ b/aiter/x.py\n"
                 "@@ -1,2 +1,2 @@\n"
                 "-        self.w2 = w2 if w2.is_contiguous() else w2.contiguous()\n")
        out = self.run_evidence(moved, {"x.py": "self.w2 = w2.contiguous()  # moved up\n"})
        self.assertIn("w2 on head", out)
        self.assertIn("moved up", out)

    def test_a_removed_zero_init_is_collected(self):
        d = ("diff --git a/aiter/x.py b/aiter/x.py\n--- a/aiter/x.py\n+++ b/aiter/x.py\n"
             "@@ -1 +0,0 @@\n-        acc = torch.zeros(n, device=q.device)\n")
        out = self.run_evidence(d, {"x.py": "acc = torch.empty(n)\nacc.zero_()\n"})
        self.assertIn("acc on head", out)

    def test_the_minus_minus_minus_header_is_not_a_deleted_line(self):
        """`--- a/file` starts with `-`. Counting it as a deleted line makes every diff
        whose header mentions a guard-like path look like a removed guard."""
        header_only = ("diff --git a/assert_helpers.cu b/assert_helpers.cu\n"
                       "--- a/assert_helpers.cu\n+++ b/assert_helpers.cu\n"
                       "@@ -1 +1 @@\n+int x = 1;\n")
        self.assertEqual("", self.run_evidence(header_only, {"k.cu": "x\n"}).strip())

    def test_a_deleted_line_that_is_not_a_guard_is_ignored(self):
        not_a_guard = ("diff --git a/csrc/k.cu b/csrc/k.cu\n--- a/csrc/k.cu\n"
                       "+++ b/csrc/k.cu\n@@ -1 +0,0 @@\n-  int q_out is not None;\n")
        self.assertEqual("", self.run_evidence(not_a_guard, {"k.cu": "q_out\n"}).strip())

    def test_a_head_file_that_does_not_exist_is_skipped_not_fatal(self):
        d = self.root / "d.diff"
        d.write_text(self.DELETED_GUARD)
        r = subprocess.run([sys.executable, str(TRIAGE), "evidence", str(d),
                            str(self.root / "gone.cu")], capture_output=True, text=True)
        self.assertEqual(0, r.returncode, r.stderr)
        self.assertEqual("", r.stdout.strip())

    def test_a_symbol_with_no_hits_prints_nothing(self):
        self.assertEqual("", self.run_evidence(
            self.DELETED_GUARD, {"k.cu": "unrelated code\n"}).strip())

    def test_guard_shaped_hits_are_preferred_over_plain_ones(self):
        """Six lines are printed. If the informative ones are not selected first, the
        reviewer gets six coincidental mentions and none of the answer."""
        body = "\n".join([f"// q_out mention {i}" for i in range(10)]
                          + ["if (q_out != nullptr) { use(q_out); }"])
        out = self.run_evidence(self.DELETED_GUARD, {"k.cu": body + "\n"})
        self.assertIn("nullptr", out)


class TestModuleScopeSkips(unittest.TestCase):
    """`STDLIB_OR_THIRD_PARTY.match(mod) or mod.startswith(".")` -- each alternative alone."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        (self.root / "aiter").mkdir()
        (self.root / "aiter" / "__init__.py").write_text("")

    def tearDown(self):
        self._tmp.cleanup()

    def sweep(self, line):
        d = self.root / "d.diff"
        d.write_text("diff --git a/aiter/x.py b/aiter/x.py\n--- a/aiter/x.py\n"
                     "+++ b/aiter/x.py\n@@ -0,0 +1 @@\n+" + line + "\n")
        r = subprocess.run([sys.executable, str(TRIAGE), "symbols", str(d), str(self.root)],
                           capture_output=True, text=True)
        return [l for l in r.stdout.splitlines() if l.startswith("UNRESOLVED-IMPORT")]

    def test_a_relative_import_is_resolved_not_skipped(self):
        """It used to be skipped outright. `aiter/x.py` importing `.gone` means
        `aiter.gone`, which is checkable and is checked."""
        out = self.sweep("from .gone import thing")
        self.assertEqual(1, len(out), out)
        self.assertIn("aiter.gone", out[0])

    def test_a_stdlib_import_alone_is_skipped(self):
        for mod in ("torch", "numpy", "triton", "pathlib", "dataclasses"):
            self.assertEqual([], self.sweep(f"from {mod} import nothing_real"), mod)

    def test_a_json_body_without_the_truncation_message_is_not_too_large(self):
        """`startswith("{") and "exceeded..." in diff` -- both halves. A diff that merely
        begins with a brace must not be reported as a diff GitHub refused to serve."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            f = pathlib.Path(td) / "d.diff"
            f.write_text('{"message": "Not Found"}\n')
            r = subprocess.run([sys.executable, str(TRIAGE), "rules", str(f), ""],
                               capture_output=True, text=True)
        self.assertIn("underivable", r.stdout)
        self.assertNotIn("diff-too-large", r.stdout)


class TestLastMutants(unittest.TestCase):
    """Predicates that survived until they were looked at individually.

    Each is a single character whose flip changes behaviour in a way no existing case
    reached. Written after classifying the survivors rather than before, which is the
    point of measuring: the untested predicate is not the one you would have guessed.
    """

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        (self.root / "aiter").mkdir()
        (self.root / "aiter" / "__init__.py").write_text("")
        (self.root / "aiter" / "real.py").write_text("KEEP = 1\n")

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_symbol_deleted_by_the_pr_does_not_exempt_an_import_of_it(self):
        """added_syms must come from `+` lines only. Collecting removed lines too means a
        PR that DELETES `def gone()` and still imports `gone` exempts itself from the very
        report that would have caught it."""
        d = self.root / "d.diff"
        d.write_text(
            "diff --git a/aiter/real.py b/aiter/real.py\n--- a/aiter/real.py\n"
            "+++ b/aiter/real.py\n@@ -1,2 +1 @@\n-def gone():\n-    pass\n"
            "diff --git a/aiter/x.py b/aiter/x.py\n--- a/aiter/x.py\n+++ b/aiter/x.py\n"
            "@@ -0,0 +1 @@\n+from aiter.real import gone\n")
        r = subprocess.run([sys.executable, str(TRIAGE), "symbols", str(d), str(self.root)],
                           capture_output=True, text=True)
        out = [l for l in r.stdout.splitlines() if l.startswith("UNRESOLVED-IMPORT")]
        self.assertEqual(1, len(out), out)
        self.assertIn("'gone'", out[0])

    def expand(self, rule_ids):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            f = pathlib.Path(td) / "r.txt"
            f.write_text(f"    [x                   ] {rule_ids}\n")
            r = subprocess.run([sys.executable, str(TRIAGE), "expand", str(f),
                                str(RULES_MD)], capture_output=True, text=True)
        self.assertEqual(0, r.returncode, r.stderr)
        return r.stdout

    def test_exactly_the_header_and_separator_precede_a_housekeeping_row(self):
        """Two header lines, not three: the third would be a real rule row, emitted for a
        rule the diff did not derive."""
        out = self.expand("HK7")
        table = [l for l in out.split("\n") if l.startswith("|")]
        self.assertEqual(3, len(table), table)          # header, separator, HK7
        self.assertIn("Check", table[0])
        self.assertRegex(table[1], r"^\|[\s|:-]+\|$")
        self.assertIn("HK7", table[2])

    def test_rows_outside_the_housekeeping_section_are_not_collected(self):
        """The section flag must start closed and re-close at the next heading; otherwise
        the category tables from A-G leak in as housekeeping."""
        out = self.expand("HK7")
        self.assertNotIn("Core question", out)
        self.assertNotIn("Silent bypass", out)

    def test_a_rule_body_stops_at_the_next_heading(self):
        """The body scan walks until the next `###`/`##`. Without that stop, asking for
        the last rule of a section drags in everything after it."""
        out = self.expand("F1")
        self.assertIn("F1", out)
        self.assertNotIn("### G — Multi-Stream", out)

    def test_no_arguments_prints_usage_rather_than_a_traceback(self):
        r = subprocess.run([sys.executable, str(TRIAGE)], capture_output=True, text=True)
        self.assertEqual(2, r.returncode, r.stderr)
        self.assertIn("usage:", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_symbols_defaults_its_root_and_rules_its_title(self):
        """Both read an optional argv slot behind a length check. Off by one and the tool
        raises IndexError on the two-argument form that Step 1b actually uses."""
        d = self.root / "d.diff"
        d.write_text("diff --git a/aiter/x.py b/aiter/x.py\n--- a/aiter/x.py\n"
                     "+++ b/aiter/x.py\n@@ -0,0 +1 @@\n+import torch\n")
        for argv in ([str(TRIAGE), "symbols", str(d)], [str(TRIAGE), "rules", str(d)]):
            r = subprocess.run([sys.executable] + argv, capture_output=True, text=True,
                               cwd=str(self.root))
            self.assertEqual(0, r.returncode, r.stderr)
            self.assertNotIn("Traceback", r.stderr)


class TestDiagnosticGate(unittest.TestCase):
    """Step 6's six structural checks. Five have no tool behind them, which is exactly why
    they have to leave a trace; the sixth was a duplicate of a sweep already run."""

    GOOD = (
        "1: symbols.txt is clean and the two new kwargs (skip_reduce, sinks) resolve on "
        "head\n"
        "2: compared the split kernel against kernel_unified_attention_2d field by field; "
        "same masks, same int64 strides\n"
        "3: the 6.55 vs 8.26 ms figures appear only in the docstring, no script output is "
        "shipped, marked unverified\n"
        "4: the try/except around the import swallows every exception; it is a real "
        "fallback, not theatre\n"
        "5: no test is added at all, so there is nothing calibrated either way -- that is "
        "the HK6 finding\n"
        "6: _OCC=8 and _MIN_SEGMENTS=8 carry a measured crossover in the docstring; "
        "_MAX_SEGMENTS=64 does not\n")

    def run_gate(self, text):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            f = pathlib.Path(td) / "d.txt"
            f.write_text(text)
            return subprocess.run([sys.executable, str(TRIAGE), "diagnostic", str(f)],
                                  capture_output=True, text=True)

    def test_all_six_recorded_passes(self):
        r = self.run_gate(self.GOOD)
        self.assertEqual(0, r.returncode, r.stdout + r.stderr)
        self.assertIn("DIAGNOSTIC COMPLETE: 6/6", r.stdout)

    def test_nothing_recorded_is_rejected(self):
        r = self.run_gate("")
        self.assertEqual(1, r.returncode)
        self.assertEqual(6, r.stdout.count("UNCHECKED"))

    def test_a_skipped_check_is_named(self):
        r = self.run_gate("\n".join(l for l in self.GOOD.split("\n")
                                    if not l.startswith("4:")))
        self.assertEqual(1, r.returncode)
        self.assertIn("UNCHECKED: structural check 4", r.stdout)

    def test_a_bare_clean_is_not_a_finding(self):
        """"clean" records a verdict without recording what was looked at, which is the
        answer that survives any amount of not looking."""
        r = self.run_gate("".join(f"{i}: clean\n" for i in range(1, 7)))
        self.assertEqual(1, r.returncode)
        self.assertEqual(6, r.stdout.count("NO-SUBSTANCE"))

    def test_step_6_no_longer_asks_for_the_sweep_by_hand(self):
        """Check 1 asked the model to list every new symbol and grep it. triage.py symbols
        had already done that and written $WORK/symbols.txt -- the same duplication D9 had
        before its scan moved into Step 1."""
        body = SKILL_MD.read_text()
        step6 = body[body.index("## Step 6"):body.index("## Step 7 ")]
        self.assertIn("symbols.txt", step6)
        self.assertNotIn("List every symbol NEW to this diff", step6)

    def test_step_8_runs_all_three_gates(self):
        body = SKILL_MD.read_text()
        step8 = body[body.index("## Step 8"):]
        for mode in ("answers", "diagnostic", "ledger"):
            self.assertRegex(step8, rf"triage\.py\"?\s+{mode}\b",
                             f"Step 8 does not run the {mode} gate")


class TestStructAbiRule(unittest.TestCase):
    """D11, added after reviewing aiter#5220.

    The skill derived ten rules for that PR and none of them was about struct layout, so
    the reviewer only reaches the defect by happening to read the file around the change.
    aiter pins every kargs struct a hand-written code object reads: 40 assertions across
    37 files in csrc/, touched by 6% of open PRs.
    """

    HUNK = ("diff --git a/csrc/include/k.h b/csrc/include/k.h\n"
            "--- a/csrc/include/k.h\n+++ b/csrc/include/k.h\n"
            "@@ -129,6 +129,8 @@ struct pa_kargs\n"
            "     int stride_qo_h;\n{added}"
            "     int stride_kv_page;\n")

    def rules(self, diff):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            f = pathlib.Path(td) / "d.diff"
            f.write_text(diff)
            r = subprocess.run([sys.executable, str(TRIAGE), "rules", str(f), "x"],
                               capture_output=True, text=True)
        self.assertEqual(0, r.returncode, r.stderr)
        return r.stdout

    def test_a_field_inserted_mid_struct_fires(self):
        """The struct declaration is in the hunk header's scope context, not in the hunk
        body -- which is the normal shape for adding a field to an existing struct."""
        out = self.rules(self.HUNK.format(added="+    int stride_o_n;\n"))
        self.assertIn("struct-abi", out)
        self.assertIn("D11", out)

    def test_a_removed_field_fires(self):
        self.assertIn("struct-abi",
                      self.rules(self.HUNK.format(added="-    int stride_o_n;\n")))

    def test_updating_the_assertions_in_the_same_diff_does_not_fire(self):
        """A PR that shifts the layout and fixes the table is the correct shape. Firing on
        it would train the reviewer to ignore the rule."""
        fixed = (self.HUNK.format(added="+    int stride_o_n;\n") +
                 "diff --git a/csrc/k.cu b/csrc/k.cu\n--- a/csrc/k.cu\n+++ b/csrc/k.cu\n"
                 "@@ -1 +1 @@\n-static_assert(sizeof(pa_kargs) == 112);\n"
                 "+static_assert(sizeof(pa_kargs) == 120);\n")
        self.assertNotIn("struct-abi", self.rules(fixed))

    def test_control_flow_inside_a_struct_method_is_not_a_field(self):
        """`break;` matched a naive "type name;" shape and fired on a PR that changed no
        layout at all."""
        body = ("diff --git a/csrc/k.h b/csrc/k.h\n--- a/csrc/k.h\n+++ b/csrc/k.h\n"
                "@@ -1,3 +1,4 @@ struct Traits\n     switch (x) {\n"
                "+      break;\n     }\n")
        self.assertNotIn("struct-abi", self.rules(body))

    def test_a_python_file_never_fires(self):
        py = ("diff --git a/aiter/x.py b/aiter/x.py\n--- a/aiter/x.py\n+++ b/aiter/x.py\n"
              "@@ -1,2 +1,3 @@ class Thing\n     a: int\n+    b: int\n     c: int\n")
        self.assertNotIn("struct-abi", self.rules(py))


class TestTestQuality(unittest.TestCase):
    """Evidence about added tests, deliberately not a verdict.

    "This test asserts nothing" is undecidable from a diff -- the check is routinely in a
    shared helper (`run_fp8(..., verify=True)`), so a rule firing on a body with no
    `assert` was wrong on the first two corpus cases inspected. The strict version, a
    whole new test file with no assertion primitive anywhere, fires on 2 of 600 and one of
    those is a benchmark. Neither is worth a rule; both are worth printing.
    """

    def run_tq(self, diff):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            f = pathlib.Path(td) / "d.diff"
            f.write_text(diff)
            r = subprocess.run([sys.executable, str(TRIAGE), "testquality", str(f)],
                               capture_output=True, text=True)
        self.assertEqual(0, r.returncode, r.stderr)
        return r.stdout

    @staticmethod
    def newfile(path, *lines):
        return (f"diff --git a/{path} b/{path}\nnew file mode 100644\n"
                f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n"
                + "".join(f"+{l}\n" for l in lines))

    def test_a_diff_with_no_test_files_says_so(self):
        out = self.run_tq("diff --git a/aiter/x.py b/aiter/x.py\n--- a/aiter/x.py\n"
                          "+++ b/aiter/x.py\n@@ -0,0 +1 @@\n+x = 1\n")
        self.assertIn("no test files touched", out)

    def test_counts_and_names_the_added_tests(self):
        out = self.run_tq(self.newfile(
            "op_tests/test_thing.py",
            "def test_alpha():", "    assert 1 == 1", "def test_beta():",
            "    torch.testing.assert_close(a, b)"))
        self.assertIn("test functions added : 2", out)
        self.assertIn("test_alpha", out)
        self.assertIn("(new file)", out)

    def test_zero_assertions_is_reported_without_accusing(self):
        out = self.run_tq(self.newfile(
            "op_tests/test_thing.py", "def test_alpha():", "    run_it(verify=True)"))
        self.assertIn("assertion primitives : 0", out)
        self.assertIn("may be in a helper", out,
                      "a zero count must carry the reason it is not a finding")

    def test_a_loose_tolerance_is_flagged_in_the_evidence(self):
        out = self.run_tq(self.newfile(
            "op_tests/test_thing.py", "def test_alpha():",
            "    torch.testing.assert_close(a, b, atol=0.5, rtol=0.01)"))
        self.assertIn("0.5", out)
        self.assertIn("loose", out)

    def test_a_tight_tolerance_is_shown_without_a_note(self):
        out = self.run_tq(self.newfile(
            "op_tests/test_thing.py", "def test_a():",
            "    assert_close(a, b, atol=1e-3, rtol=1e-3)"))
        self.assertIn("tolerances", out)
        self.assertNotIn("loose", out)

    def test_toy_only_shapes_are_flagged(self):
        out = self.run_tq(self.newfile(
            "op_tests/test_thing.py", "def test_a():", "    M = 16", "    N = 8",
            "    assert run(M, N)"))
        self.assertIn("every shape is small", out)

    def test_production_shapes_are_not_flagged(self):
        out = self.run_tq(self.newfile(
            "op_tests/test_thing.py", "def test_a():", "    M = 16", "    N = 8192",
            "    assert run(M, N)"))
        self.assertIn("shapes", out)
        self.assertNotIn("every shape is small", out)

    def test_step_1b_writes_the_artifact(self):
        self.assertIn("testquality", FETCH.read_text())
        self.assertIn("test_quality.txt", SKILL_MD.read_text())


class TestTwinDetection(unittest.TestCase):
    """Step 6's check 2 asks for mirrored code and left finding the mirror to the reader."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    BODY = [f"    some_substantive_line_number_{i} = compute(x, y, z, {i})" for i in range(60)]

    def write(self, rel, lines):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n")

    def twins(self, diff):
        d = self.root / "d.diff"
        d.write_text(diff)
        r = subprocess.run([sys.executable, str(TRIAGE), "twins", str(d), str(self.root)],
                           capture_output=True, text=True)
        self.assertEqual(0, r.returncode, r.stderr)
        return r.stdout

    def newfile(self, path, lines):
        return (f"diff --git a/{path} b/{path}\nnew file mode 100644\n"
                f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n"
                + "".join(f"+{l}\n" for l in lines))

    def test_a_copied_kernel_names_its_source(self):
        self.write("aiter/ops/orig.py", self.BODY)
        out = self.twins(self.newfile("aiter/ops/copy_v2.py", self.BODY))
        self.assertIn("TWIN: aiter/ops/copy_v2.py", out)
        self.assertIn("aiter/ops/orig.py", out)
        self.assertIn("100%", out)

    def test_an_unrelated_new_file_is_not_reported(self):
        self.write("aiter/ops/orig.py", self.BODY)
        other = [f"    entirely_different_content_{i} = other(a, b, c, {i})"
                 for i in range(60)]
        self.assertIn("no new file closely mirrors",
                      self.twins(self.newfile("aiter/ops/fresh.py", other)))

    def test_a_short_new_file_is_not_compared(self):
        """Below 40 substantive lines the overlap is noise -- boilerplate imports and a
        license header would match everything."""
        self.write("aiter/ops/orig.py", self.BODY)
        self.assertIn("no new file closely mirrors",
                      self.twins(self.newfile("aiter/ops/tiny.py", self.BODY[:10])))

    def test_a_modified_file_is_not_a_twin(self):
        """Only `new file mode` counts; editing a file is not copying one."""
        self.write("aiter/ops/orig.py", self.BODY)
        edit = ("diff --git a/aiter/ops/orig.py b/aiter/ops/orig.py\n"
                "--- a/aiter/ops/orig.py\n+++ b/aiter/ops/orig.py\n@@ -1 +1 @@\n"
                + "".join(f"+{l}\n" for l in self.BODY))
        self.assertIn("no new file closely mirrors", self.twins(edit))

    def test_step_1b_writes_the_artifact(self):
        self.assertIn("twins", FETCH.read_text())
        self.assertIn("twins.txt", SKILL_MD.read_text())


class TestUncollectableTestFile(unittest.TestCase):
    """HK12, from reviewing aiter#4821.

    HK6 asks a new op to ship op_tests/test_*.py. A `main()` script named that satisfies
    HK6 by name while pytest collects nothing from it. aiter already contains such files,
    so the style is tolerated -- the rule is about the coverage claim, and fires only when
    the PR ships no collectable test at all (3.8% of the 600-PR corpus)."""

    SCRIPT = ["import sys", "def _check(op, cfg):", "    assert op(cfg)",
              "def main():", "    _check(build(), cfg())", "    return 0",
              'if __name__ == "__main__":', "    sys.exit(main())"] + \
             [f"# padding to clear the size floor {i}" for i in range(40)]

    def rules(self, diff):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            f = pathlib.Path(td) / "d.diff"
            f.write_text(diff)
            r = subprocess.run([sys.executable, str(TRIAGE), "rules", str(f), "x"],
                               capture_output=True, text=True)
        self.assertEqual(0, r.returncode, r.stderr)
        return r.stdout

    @staticmethod
    def newfile(path, lines):
        return (f"diff --git a/{path} b/{path}\nnew file mode 100644\n"
                f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n"
                + "".join(f"+{l}\n" for l in lines))

    def test_a_main_driven_script_named_like_a_test_fires(self):
        out = self.rules(self.newfile("op_tests/test_thing.py", self.SCRIPT))
        self.assertIn("uncollectable-test", out)
        self.assertIn("HK12", out)

    def test_a_collectable_test_in_the_same_pr_clears_it(self):
        """The rule is about the coverage claim. A PR shipping both a manual script and a
        real test has made the claim good."""
        diff = (self.newfile("op_tests/test_thing.py", self.SCRIPT) +
                self.newfile("op_tests/test_real.py",
                             ["def test_it():", "    assert compute() == 1"]))
        self.assertNotIn("uncollectable-test", self.rules(diff))

    def test_a_class_based_test_counts_as_collectable(self):
        diff = self.newfile("op_tests/test_thing.py",
                            ["class TestThing:", "    def test_a(self):",
                             "        assert 1"] + self.SCRIPT)
        self.assertNotIn("uncollectable-test", self.rules(diff))

    def test_a_modified_script_does_not_fire(self):
        """Only a new file. Editing an existing manual script is not a new coverage claim."""
        edited = ("diff --git a/op_tests/test_thing.py b/op_tests/test_thing.py\n"
                  "--- a/op_tests/test_thing.py\n+++ b/op_tests/test_thing.py\n"
                  "@@ -1 +1 @@\n" + "".join(f"+{l}\n" for l in self.SCRIPT))
        self.assertNotIn("uncollectable-test", self.rules(edited))

    def test_a_file_not_named_like_a_test_does_not_fire(self):
        self.assertNotIn("uncollectable-test",
                         self.rules(self.newfile("op_tests/bench_thing.py", self.SCRIPT)))
