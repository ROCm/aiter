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
import tempfile
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
        rows = set(re.findall(r"\b(HK\d+[a-z]?)\b", rules))
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
            | set(_re.findall(r"\b(HK\d+[a-z]?)\b", rules))
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

    def test_a_relative_import_inside_a_package_init_resolves_to_that_package(self):
        """For aiter/ops/flydsl/__init__.py the containing package IS aiter.ops.flydsl.
        Stripping `.__init__` and then dropping a component as well resolved
        `from .kernels.foo import ...` to aiter.ops.kernels.foo -- which does not exist --
        and reported aiter#4515 for an import that was fine at that level."""
        self.write("aiter/__init__.py")
        self.write("aiter/ops/__init__.py")
        self.write("aiter/ops/flydsl/kernels/__init__.py")
        self.write("aiter/ops/flydsl/kernels/thing.py", "def go():\n    pass\n")
        out = self.sweep(self.diff_adding(
            "aiter/ops/flydsl/__init__.py", "from .kernels.thing import go"))
        self.assertEqual([], out)

    def test_a_package_init_still_reports_a_module_that_is_gone(self):
        self.write("aiter/__init__.py")
        self.write("aiter/ops/flydsl/kernels/__init__.py")
        out = self.sweep(self.diff_adding(
            "aiter/ops/flydsl/__init__.py", "from .kernels.gone import go"))
        self.assertEqual(1, len(out), out)
        self.assertIn("aiter.ops.flydsl.kernels.gone", out[0])

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

    def test_every_shape_of_removed_guard_yields_its_subject(self):
        """Matching only `X is not None` and `X->` extracted nothing from 51 of the 57
        corpus PRs that delete a guard. The subject of `assert causal, "..."` is `causal`;
        of `assert WQ.dtype == fp8` it is `WQ`; of `assert gfx_version in (...)` it is
        `gfx_version`. Found on aiter#4255, where the assert was REPLACED by a stronger
        named predicate and the reviewer had nothing on screen to see that with."""
        cases = {
            'assert gfx_version in ("gfx942", "gfx950")': "gfx_version",
            'assert causal, "Only causal attention is supported"': "causal",
            "assert WQ.dtype == dtypes.fp8, \"only fp8\"": "WQ",
            "assert num_head_qo % 16 == 0": "num_head_qo",
            'assert q_descale is None, "Q scales not supported"': "q_descale",
            'TORCH_CHECK(out.is_contiguous(), "out must be contiguous");': "out",
        }
        for line, subject in cases.items():
            d = ("diff --git a/aiter/x.py b/aiter/x.py\n--- a/aiter/x.py\n"
                 "+++ b/aiter/x.py\n@@ -1 +0,0 @@\n-    " + line + "\n")
            out = self.run_evidence(d, {"x.py": f"{subject} = compute()  # still here\n"})
            self.assertIn(f"{subject} on head", out, line)

    def test_guard_keywords_are_not_mistaken_for_the_subject(self):
        """`assert not x` must yield `x`, not `not`."""
        d = ("diff --git a/aiter/x.py b/aiter/x.py\n--- a/aiter/x.py\n+++ b/aiter/x.py\n"
             "@@ -1 +0,0 @@\n-    assert not is_fnuz\n")
        out = self.run_evidence(d, {"x.py": "is_fnuz = arch.startswith('gfx94')\n"})
        self.assertIn("is_fnuz on head", out)
        self.assertNotIn("not on head", out)

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


class TestStructAbiEvidence(unittest.TestCase):
    """D11, and the correction batch five forced on it.

    Derived from the diff alone -- "a field moved and no assertion line changed" -- it
    fired on aiter#5223, six headers with no assertion anywhere near them. Whether a
    layout is pinned is a fact about the tree, so the check reads the tree.
    """

    HUNK = ("diff --git a/csrc/include/k.h b/csrc/include/k.h\n"
            "--- a/csrc/include/k.h\n+++ b/csrc/include/k.h\n"
            "@@ -129,6 +129,8 @@ struct pa_kargs\n"
            "     int stride_qo_h;\n{added}"
            "     int stride_kv_page;\n")

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        (self.root / "csrc").mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def pin(self, text):
        (self.root / "csrc" / "k.cu").write_text(text)

    def run_it(self, diff):
        d = self.root / "d.diff"
        d.write_text(diff)
        r = subprocess.run([sys.executable, str(TRIAGE), "structabi", str(d),
                            str(self.root)], capture_output=True, text=True)
        self.assertEqual(0, r.returncode, r.stderr)
        return r.stdout

    def test_a_pinned_struct_is_reported(self):
        self.pin("static_assert(sizeof(pa_kargs) == 112);\n")
        out = self.run_it(self.HUNK.format(added="+    int stride_o_n;\n"))
        self.assertIn("PINNED-LAYOUT: pa_kargs", out)
        self.assertIn("csrc/k.cu", out)

    def test_an_offsetof_assertion_also_counts(self):
        self.pin("PA_ABI(offsetof(pa_kargs, stride_kv_page) == 100);\n")
        self.assertIn("PINNED-LAYOUT",
                      self.run_it(self.HUNK.format(added="+    int stride_o_n;\n")))

    def test_an_unpinned_struct_is_not_reported(self):
        """aiter#5223: the field moved, nothing asserts the layout, nothing to say."""
        self.pin("// no assertions here\n")
        self.assertIn("no struct with a pinned layout",
                      self.run_it(self.HUNK.format(added="+    int stride_o_n;\n")))

    def test_an_assertion_for_a_different_struct_does_not_count(self):
        self.pin("static_assert(sizeof(other_kargs) == 64);\n")
        self.assertIn("no struct with a pinned layout",
                      self.run_it(self.HUNK.format(added="+    int stride_o_n;\n")))

    def test_updating_the_assertion_in_the_same_diff_clears_it(self):
        self.pin("static_assert(sizeof(pa_kargs) == 112);\n")
        fixed = (self.HUNK.format(added="+    int stride_o_n;\n") +
                 "diff --git a/csrc/k.cu b/csrc/k.cu\n--- a/csrc/k.cu\n+++ b/csrc/k.cu\n"
                 "@@ -1 +1 @@\n-static_assert(sizeof(pa_kargs) == 112);\n"
                 "+static_assert(sizeof(pa_kargs) == 120);\n")
        self.assertIn("no struct with a pinned layout", self.run_it(fixed))

    def test_control_flow_is_not_a_field(self):
        self.pin("static_assert(sizeof(Traits) == 8);\n")
        body = ("diff --git a/csrc/k.h b/csrc/k.h\n--- a/csrc/k.h\n+++ b/csrc/k.h\n"
                "@@ -1,3 +1,4 @@ struct Traits\n     switch (x) {\n"
                "+      break;\n     }\n")
        self.assertIn("no struct with a pinned layout", self.run_it(body))

    def test_step_1b_writes_the_artifact(self):
        self.assertIn("structabi", FETCH.read_text())
        self.assertIn("struct_abi.txt", SKILL_MD.read_text())



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

    def test_a_test_file_with_nothing_measurable_is_not_listed(self):
        """No tests, no assertions, no tolerances, no shapes: there is nothing to put in
        front of the reviewer, so the file does not appear."""
        out = self.run_tq(self.newfile("op_tests/test_empty.py", "import torch", "x = 1"))
        self.assertIn("no test files touched", out)

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

    def test_a_benchmark_with_no_assertions_reads_as_expected(self):
        """A benchmark asserts nothing because it measures. Telling the reviewer the check
        "may be in a helper" about a bench_*.py is misleading, and 68 of 682 rows over the
        corpus are exactly that (aiter#4016 ships one beside a real test)."""
        out = self.run_tq(self.newfile(
            "op_tests/triton_tests/bench_thing.py",
            "def graph_time(fn, iters=50):", "    M = 8192", "    return timed(fn, M)"))
        self.assertIn("benchmark", out)
        self.assertIn("expected for a benchmark", out)
        self.assertNotIn("may be in a helper", out)

    def test_a_real_test_with_no_assertions_still_says_helper(self):
        out = self.run_tq(self.newfile(
            "op_tests/test_thing.py", "def test_alpha():", "    run_it(verify=True)"))
        self.assertIn("may be in a helper", out)
        self.assertNotIn("benchmark", out)

    def test_op_benchmarks_directory_counts_as_benchmark(self):
        out = self.run_tq(self.newfile(
            "op_tests/op_benchmarks/triton/test_mha_perf.py",
            "def sweep():", "    M = 4096", "    measure(M)"))
        self.assertIn("expected for a benchmark", out)

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


class TestCiCoverage(unittest.TestCase):
    """Which added tests a CI job will actually run, computed from .github/.

    HK6 asks a new op to ship a test, and the reviewer credits the PR for having one. The
    file's *location* decides whether it runs: aiter shard-scans op_tests at maxdepth 1
    and op_tests/triton_tests recursively, so op_tests/flydsl_tests/ is scanned by nothing
    (5 files in tree) and op_tests/multigpu_tests/ needs a label (29). 9% of the 600-PR
    corpus adds a test that never runs. aiter#4821 ships two.
    """

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        sh = self.root / ".github" / "scripts" / "split_tests.sh"
        sh.parent.mkdir(parents=True)
        sh.write_text('TEST_DIR="op_tests"\n'
                      "find \"$TEST_DIR\" -maxdepth 1 -name 'test_*.py'\n"
                      'TEST_DIR="op_tests/triton_tests"\n'
                      "find \"$TEST_DIR\" -name 'test_*.py'\n")
        for d in ("op_tests", "op_tests/triton_tests", "op_tests/flydsl_tests",
                  "op_tests/multigpu_tests"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self._tmp.cleanup()

    def workflow(self, name, text):
        wf = self.root / ".github" / "workflows"
        wf.mkdir(parents=True, exist_ok=True)
        (wf / name).write_text(text)

    def citest(self, path):
        d = self.root / "d.diff"
        d.write_text(f"diff --git a/{path} b/{path}\nnew file mode 100644\n"
                     f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n"
                     "+def test_a():\n+    assert 1\n")
        r = subprocess.run([sys.executable, str(TRIAGE), "citest", str(d), str(self.root)],
                           capture_output=True, text=True)
        self.assertEqual(0, r.returncode, r.stderr)
        return r.stdout

    def test_a_top_level_op_test_is_covered(self):
        self.assertIn("will run it", self.citest("op_tests/test_thing.py"))

    def test_a_triton_test_at_depth_is_covered(self):
        self.assertIn("will run it",
                      self.citest("op_tests/triton_tests/gemm/test_thing.py"))

    def test_an_unscanned_directory_is_reported(self):
        out = self.citest("op_tests/flydsl_tests/test_thing.py")
        self.assertIn("UNRUN-TEST", out)
        self.assertIn("no CI job scans", out)

    def test_a_nested_path_under_op_tests_is_reported(self):
        """maxdepth 1: op_tests/block/test_x.py is not scanned even though op_tests is."""
        (self.root / "op_tests" / "block").mkdir(parents=True, exist_ok=True)
        self.assertIn("UNRUN-TEST", self.citest("op_tests/block/test_thing.py"))

    def test_a_path_only_named_by_a_labelling_workflow_is_not_covered(self):
        """pr-title-tags.yaml names op_tests/flydsl_tests/ to pick a label. Counting a
        mention as coverage reported aiter#4821 as covered -- the case this exists for."""
        self.workflow("pr-title-tags.yaml",
                      "  const isFlydsl = f.startsWith('op_tests/flydsl_tests/');\n")
        self.assertIn("UNRUN-TEST", self.citest("op_tests/flydsl_tests/test_thing.py"))

    def test_a_label_gated_directory_says_so(self):
        self.workflow("aiter-test.yaml",
                      "        LABEL: ${{ contains(github.event.pull_request.labels.*.name,"
                      " 'multigpu') }}\n          run_multigpu_tests: true\n")
        out = self.citest("op_tests/multigpu_tests/test_thing.py")
        self.assertIn("UNRUN-TEST", out)
        self.assertIn("label", out)

    def test_step_1b_writes_the_artifact(self):
        self.assertIn("citest", FETCH.read_text())
        self.assertIn("ci_coverage.txt", SKILL_MD.read_text())


class TestPerfClaims(unittest.TestCase):
    """Numbers the description claims, and which name a baseline.

    Scoped to the PR body. Reading claims out of kernel comments over the 600-PR corpus
    was 61% noise: `4x DS_READ`, `<4 x i32>`, `num_tokens x 384 x 7168`, `5% of elements`.
    """

    def claims(self, body):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            f = pathlib.Path(td) / "pr_meta.json"
            f.write_text(json.dumps({"body": body}))
            r = subprocess.run([sys.executable, str(TRIAGE), "perfclaims", str(f)],
                               capture_output=True, text=True)
        self.assertEqual(0, r.returncode, r.stderr)
        return r.stdout

    def test_a_bare_absolute_number_is_flagged(self):
        out = self.claims("FlyDSL (direct, no fallback): expect ~198 TFLOPS\n")
        self.assertIn("->", out)
        self.assertIn("198TFLOPS", out)

    def test_a_number_next_to_its_comparison_is_not_flagged(self):
        out = self.claims("reaches 902 TFLOP/s against CK-tile's 838\n")
        self.assertNotIn("->", out)

    def test_a_signed_delta_carries_its_own_baseline(self):
        """`+8.64% end-to-end` compares against not having the change; asking what it is
        measured against is pedantry, and pedantry is what makes a check ignorable."""
        self.assertNotIn("->", self.claims("worth **+8.64% end-to-end** on an 8x server\n"))

    def test_table_rows_inherit_the_header_baseline(self):
        body = ("| batch | before | after | speedup |\n"
                "|---|---|---|---|\n"
                "| 1 | 35.76 | 27.00 | 1.32x |\n"
                "| 2 | 41.61 | 31.30 | 1.33x |\n")
        self.assertNotIn("->", self.claims(body))

    def test_a_chinese_before_after_header_counts_as_a_baseline(self):
        """aiter PR descriptions are partly Chinese. An English-only word list marked
        aiter#5043's table as unbaselined -- correctly, as it happens, since its header is
        `场景 | 输入数据 | 时间 | TFLOPS` and names no comparison; but a table headed
        `优化前 | 优化后 | 提速` does, and would have been marked too."""
        body = ("| 场景 | 优化前 | 优化后 | 提速 |\n|---|---|---|---|\n"
                "| non-causal | 3.0 ms | 2.348 ms | 1.28x |\n")
        self.assertNotIn("->", self.claims(body))

    def test_a_chinese_table_with_no_comparison_column_is_flagged(self):
        body = ("| 场景 | 输入数据 | 时间 | TFLOPS |\n|---|---|---|---|\n"
                "| non-causal | randn | 2.348 ms | 2341 |\n")
        self.assertIn("->", self.claims(body))

    def test_a_share_of_time_is_not_a_speedup(self):
        """`33.07% of GPU time` says where the time goes. Asking what it is measured
        against is nonsense, and the corpus is full of these."""
        out = self.claims("FmhaBatchPrefill was 33.07% of GPU time, and the run is "
                          "81.8% prefill by GPU time\n")
        self.assertNotIn("->", out)

    def test_hardware_model_numbers_are_not_claims(self):
        """MI300X, MI35x, gfx950 all end in something the number matcher wants."""
        out = self.claims("Tested on 8x MI355X (gfx950) and MI308X\n")
        self.assertIn("no numeric performance claim", out)

    def test_a_description_with_no_numbers_says_so(self):
        self.assertIn("no numeric performance claim",
                      self.claims("Refactor the dispatch table.\n"))

    def test_step_1b_writes_the_artifact(self):
        self.assertIn("perfclaims", FETCH.read_text())
        self.assertIn("perf_claims.txt", SKILL_MD.read_text())


class TestCommentDominated(unittest.TestCase):
    """When a diff is almost all comments, the few lines that are not are the review.

    2 of 600 open PRs, and the leverage is the point: aiter#4062 is 7963 changed lines
    across 252 files and reduces to nothing at all.
    """

    def run_it(self, diff):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            f = pathlib.Path(td) / "d.diff"
            f.write_text(diff)
            r = subprocess.run([sys.executable, str(TRIAGE), "commentonly", str(f)],
                               capture_output=True, text=True)
        self.assertEqual(0, r.returncode, r.stderr)
        return r.stdout

    @staticmethod
    def diff_of(path, lines):
        return (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
                "@@ -1 +1 @@\n" + "".join(f"{l}\n" for l in lines))

    def test_a_normal_diff_is_not_reported(self):
        body = [f"+    x{i} = compute({i})" for i in range(70)]
        self.assertIn("not a comment-dominated",
                      self.run_it(self.diff_of("aiter/x.py", body)))

    def test_a_small_diff_is_not_reported(self):
        """Below the floor the ratio means nothing -- two comment lines is not a pattern."""
        self.assertIn("not a comment-dominated",
                      self.run_it(self.diff_of("aiter/x.py", ["+# a", "+# b"])))

    def test_the_code_lines_of_a_comment_diff_are_listed(self):
        body = [f"+# prose line {i}" for i in range(70)] + ["+REVISION = 26"]
        out = self.run_it(self.diff_of("aiter/x.py", body))
        self.assertIn("COMMENT-DOMINATED", out)
        self.assertIn("REVISION = 26", out)

    def test_a_changed_trailing_comment_is_not_a_code_change(self):
        """`X = 25  # old` -> `X = 25  # new` changes no code. aiter#4062 read as four
        code lines until trailing comments were stripped, and as zero after."""
        body = ([f"+# prose {i}" for i in range(70)]
                + ["-REVISION = 25  # rev24, drop the token-major path",
                   "+REVISION = 25  # g layout fixed head-major"])
        out = self.run_it(self.diff_of("aiter/x.py", body))
        self.assertIn("0 of them code", out)

    def test_a_real_code_change_beside_a_comment_change_survives(self):
        body = ([f"+# prose {i}" for i in range(70)]
                + ["-REVISION = 25  # old", "+REVISION = 26  # new"])
        out = self.run_it(self.diff_of("aiter/x.py", body))
        self.assertIn("REVISION = 26", out)

    def test_a_hash_inside_a_string_is_not_a_comment_marker(self):
        body = ([f"+# prose {i}" for i in range(70)]
                + ['-tag = "a#b"', '+tag = "a#c"'])
        self.assertIn('tag = "a#c"', self.run_it(self.diff_of("aiter/x.py", body)))

    def test_c_block_comment_continuations_are_prose(self):
        """A `/* ... */` block whose continuation lines carry no leading `*` reads as code
        line by line; aiter#4061's rewritten header is exactly that."""
        body = ["+/* Convention of the scale in API:"] + \
               [f"+   line {i} of the explanation" for i in range(70)] + ["+*/"]
        self.assertIn("0 of them code", self.run_it(self.diff_of("csrc/x.cuh", body)))

    def test_step_1b_writes_the_artifact(self):
        self.assertIn("commentonly", FETCH.read_text())
        self.assertIn("comment_only.txt", SKILL_MD.read_text())


class TestCardGate(unittest.TestCase):
    """The fourth gate: does the card report the work that was done?

    answers, diagnostic and ledger all check that the work happened. None of them looks
    at the card, so a review could adjudicate 27 rules honestly and then write a finding
    that appears in none of them.
    """

    DIFF = ("diff --git a/aiter/mla.py b/aiter/mla.py\n--- a/aiter/mla.py\n"
            "+++ b/aiter/mla.py\n")
    VERDICTS = ("G1 FIRE -- aiter/mla.py:120 hands the buffer across streams\n"
                "G1b CLEAR -- nothing is captured into a cudagraph here\n")

    def gate(self, card, verdicts=None, blind=None):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            (d / "card.md").write_text(card)
            (d / "v.txt").write_text(self.VERDICTS if verdicts is None else verdicts)
            (d / "diag.txt").write_text("1: symbols.txt clean\n")
            (d / "ans.txt").write_text(blind or "BLIND: nothing further found\n")
            (d / "d.diff").write_text(self.DIFF)
            return subprocess.run(
                [sys.executable, str(TRIAGE), "card", str(d / "card.md"), str(d / "v.txt"),
                 str(d / "diag.txt"), str(d / "ans.txt"), str(d / "d.diff")],
                capture_output=True, text=True)

    def test_a_card_with_no_findings_and_no_fired_verdict_passes(self):
        r = self.gate("Review (advisory): NO FINDINGS\n",
                      verdicts="G1 CLEAR -- no cross-stream handoff in this diff\n")
        self.assertEqual(0, r.returncode, r.stdout + r.stderr)
        self.assertIn("no verdict fired", r.stdout)

    def test_a_backed_anchored_concrete_finding_passes(self):
        r = self.gate("\U0001F534 G1: aiter/mla.py:120 hands the indptr buffer to another "
                      "stream with no record_event, so the consumer reads it at 16384 "
                      "tokens before the producer has written it\n")
        self.assertEqual(0, r.returncode, r.stdout + r.stderr)
        self.assertIn("NAILED DOWN", r.stdout)

    def test_a_finding_anchored_nowhere_in_this_pr_is_rejected(self):
        """The gate's purpose: a finding that is about some other PR."""
        r = self.gate("\u26A0\uFE0F G1: aiter/fused_moe.py:88 consumes it on another "
                      "stream at 4096 tokens\n")
        self.assertEqual(1, r.returncode)
        self.assertIn("UNANCHORED-FINDING", r.stdout)

    def test_a_finding_may_cite_the_tree_as_long_as_it_is_anchored(self):
        """Rewritten deliberately. It used to require that every path in a finding be one
        the diff changes, which rejected the forensics the rules ask for: aiter#2478's two
        strongest findings rest on the mask contract in csrc/include/moe_sorting_opus.h,
        a file the PR does not touch, and the review had to delete the filename and
        describe the header vaguely to get past this gate. Anchored in a changed file and
        citing the tree beside it is the shape of a well-evidenced finding, not a defect."""
        r = self.gate("\U0001F534 D1: aiter/mla.py:200 sizes the buffer from a count "
                      "the contract in csrc/include/moe_sorting_opus.h derives "
                      "differently, so tokens past 96 index another expert\n")
        self.assertNotIn("UNANCHORED-FINDING", r.stdout)

    def test_a_finding_citing_no_file_at_all_is_not_flagged_as_unanchored(self):
        """Concreteness is the vague-red gate's job, not this one's -- two gates reporting
        the same defect under different names sends the reviewer to the wrong fix."""
        r = self.gate("\u26A0\uFE0F D1: the padded buffer is one tile short at 8192 "
                      "tokens on the EP path\n")
        self.assertNotIn("UNANCHORED-FINDING", r.stdout)

    def test_a_finding_whose_rule_was_never_fired_is_rejected(self):
        """G1b was adjudicated CLEAR. Reporting it as a finding contradicts the ledger."""
        r = self.gate("\u26A0\uFE0F G1b: aiter/mla.py:200 captures a host sync inside "
                      "the graph at 8192 tokens\n")
        self.assertEqual(1, r.returncode)
        self.assertIn("UNBACKED-FINDING", r.stdout)

    def test_a_vague_red_is_rejected(self):
        r = self.gate("\U0001F534 the reduce kernel looks racy\n")
        self.assertEqual(1, r.returncode)
        self.assertIn("UNPROVEN-RED", r.stdout)

    def test_a_red_naming_expressions_rather_than_values_passes(self):
        """The threshold asks for the input that makes it fire. Naming the exact
        expressions is concrete even with no digit in the sentence."""
        r = self.gate("\U0001F534 G1: aiter/mla.py hands row=(tail_blk*num_kv_heads+"
                      "kv_head_idx)*BLOCK_M to a consumer decoding nrows=tail*"
                      "num_kv_heads*BLOCK_M\n")
        self.assertEqual(0, r.returncode, r.stdout + r.stderr)

    def test_a_note_is_not_held_to_the_red_threshold(self):
        """A note names no value and no identifiers and that is fine; only red carries
        the threshold. Verdicts with nothing fired, so the note stands alone."""
        r = self.gate("\U0001F4DD aiter/mla.py could use a comment here\n",
                      verdicts="G1 CLEAR -- no cross-stream handoff in this diff\n",
                      blind="BLIND: nothing beyond a readability point in aiter/mla.py\n")
        self.assertEqual(0, r.returncode, r.stdout + r.stderr)

    def test_a_note_from_nowhere_is_still_rejected(self):
        """A free-form finding has to come from somewhere recorded -- Step 7.5 is where
        "things Steps 1-7 did not catch" is written down. A note citing a file that
        appears in no verdict, diagnostic or blind-spot line came from nowhere."""
        r = self.gate("\U0001F4DD aiter/mla.py could use a comment here\n",
                      verdicts="G1 CLEAR -- no cross-stream handoff in this diff\n")
        self.assertEqual(1, r.returncode)
        self.assertIn("UNBACKED-FINDING", r.stdout)

    def test_a_missing_card_file_fails_closed(self):
        """Not writing the card was the cheapest way past the gate: a missing file read
        as an empty card, which read as no findings, which passed."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            (d / "v.txt").write_text(self.VERDICTS)
            (d / "d.diff").write_text(self.DIFF)
            r = subprocess.run(
                [sys.executable, str(TRIAGE), "card", str(d / "absent.md"),
                 str(d / "v.txt"), str(d / "v.txt"), str(d / "v.txt"), str(d / "d.diff")],
                capture_output=True, text=True)
        self.assertEqual(1, r.returncode)
        self.assertIn("CARD MISSING", r.stderr)

    def test_a_fired_verdict_absent_from_the_card_is_rejected(self):
        """The inverse of the trace check, and the one that was open: a review can
        adjudicate nine rules FIRE, report none of them, and pass. FIRE means it goes in
        the card."""
        r = self.gate("Review (advisory): NO FINDINGS\n")
        self.assertEqual(1, r.returncode)
        self.assertIn("UNREPORTED-FIRE", r.stdout)
        self.assertIn("G1", r.stdout)

    def test_a_fire_withdrawn_in_writing_is_accepted(self):
        """The 5-finding cap is real, so there has to be an escape -- and it has to be
        written down rather than silent."""
        withdrawn = ("G1 FIRE -- aiter/mla.py:120 hands the buffer across streams "
                     "-- not reported: below the cap, lower blast radius than T6\n"
                     "G1b CLEAR -- nothing is captured into a cudagraph here\n")
        r = self.gate("Review (advisory): NO FINDINGS\n", verdicts=withdrawn)
        self.assertEqual(0, r.returncode, r.stdout + r.stderr)

    def test_a_card_reporting_its_fire_passes(self):
        r = self.gate("\u26A0\uFE0F G1: aiter/mla.py:120 hands the indptr buffer to "
                      "another stream with no record_event at 16384 tokens\n")
        self.assertEqual(0, r.returncode, r.stdout + r.stderr)
        self.assertIn("every FIRE accounted for", r.stdout)

    def test_step_8_runs_the_card_gate(self):
        body = SKILL_MD.read_text()
        step8 = body[body.index("## Step 8"):]
        self.assertRegex(step8, r"triage\.py\"?\s+card\b",
                         "Step 8 does not run the card gate")


class TestHardcodedLaunchKnob(unittest.TestCase):
    """T8, from the aiter#5137 revert.

    T4 is the cross-arch case. This is the single-arch one: `waves_per_eu=2` in
    pa_decode.py with a six-line justification, reverted on one comment -- "change the
    config file jsons instead of hardcoding it in the code". The justification was not the
    problem; the location was. 11% of the 600-PR corpus.
    """

    def rules(self, path, *lines):
        import tempfile
        body = (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
                "@@ -1 +1 @@\n" + "".join(f"+{l}\n" for l in lines))
        with tempfile.TemporaryDirectory() as td:
            f = pathlib.Path(td) / "d.diff"
            f.write_text(body)
            r = subprocess.run([sys.executable, str(TRIAGE), "rules", str(f), "x"],
                               capture_output=True, text=True)
        self.assertEqual(0, r.returncode, r.stderr)
        return r.stdout

    def test_a_literal_knob_in_kernel_source_fires(self):
        out = self.rules("aiter/ops/triton/attention/pa_decode.py", "        waves_per_eu=2,")
        self.assertIn("hardcoded-launch-knob", out)
        self.assertIn("T8", out)

    def test_every_knob_name_fires(self):
        for knob in ("waves_per_eu", "matrix_instr_nonkdim", "kpack", "num_stages",
                     "num_warps", "num_ctas"):
            self.assertIn("hardcoded-launch-knob",
                          self.rules("aiter/ops/triton/x.py", f"    {knob}=4,"), knob)

    def test_a_knob_read_from_config_does_not_fire(self):
        """`num_warps=config["num_warps"]` is the config system working."""
        for line in ('    num_warps=config["num_warps"],', "    num_warps=cfg.num_warps,",
                     "    **config,", "    num_stages=tuned[key],"):
            self.assertNotIn("hardcoded-launch-knob",
                             self.rules("aiter/ops/triton/x.py", line), line)

    def test_a_literal_in_a_config_file_does_not_fire(self):
        self.assertNotIn("hardcoded-launch-knob",
                         self.rules("aiter/ops/triton/configs/gfx950/x.py",
                                    "    waves_per_eu=2,"))

    def test_a_test_pinning_a_knob_does_not_fire(self):
        """A test pinning a knob pins it on purpose."""
        self.assertNotIn("hardcoded-launch-knob",
                         self.rules("op_tests/test_thing.py", "    waves_per_eu=2,"))

    def test_a_commented_out_knob_does_not_fire(self):
        self.assertNotIn("hardcoded-launch-knob",
                         self.rules("aiter/ops/triton/x.py", "    # waves_per_eu=2,"))


class TestCoreFileGate(unittest.TestCase):
    """The fifth gate: Step 4 produced nothing, so it could be skipped in silence.

    answers/diagnostic/ledger/card all pin a step to an artifact something else reads.
    Step 4 -- the backbone risk assessment, the one step whose subject is the files where
    being wrong costs every consumer -- was still prose, and prose is the failure mode the
    other four gates exist to close.

    What is checkable is not whether the risk was judged right. It is that every backbone
    file in the diff was judged at all, and that the reason is about THIS PR rather than a
    sentence equally true of every PR ever opened against that file."""

    DIFF = (
        "diff --git a/aiter/fused_moe.py b/aiter/fused_moe.py\n"
        "--- a/aiter/fused_moe.py\n+++ b/aiter/fused_moe.py\n@@ -10,6 +10,7 @@\n"
        "-    topk_weights = moe_sorting_fwd(a, b)\n"
        "+    topk_weights = moe_sorting_fwd(a, b, num_local_experts)\n"
        "diff --git a/aiter/__init__.py b/aiter/__init__.py\n"
        "--- a/aiter/__init__.py\n+++ b/aiter/__init__.py\n@@ -3,0 +4 @@\n"
        "+from .ops.gemm_op_a4w4 import *\n"
        "diff --git a/op_tests/test_moe.py b/op_tests/test_moe.py\n"
        "--- a/op_tests/test_moe.py\n+++ b/op_tests/test_moe.py\n@@ -1,0 +2 @@\n"
        "+def test_num_local_experts(): pass\n")

    GOOD = (
        "aiter/fused_moe.py TIER2 COVERED -- num_local_experts is threaded through to "
        "moe_sorting_fwd and op_tests/test_moe.py adds a DSv3 TP=8 decode case for it\n"
        "aiter/__init__.py TIER1 GAP -- the new `from .ops.gemm_op_a4w4 import *` sits "
        "above the rest of the block, so an ImportError inside it truncates the "
        "namespace silently\n")

    # A PR touching only a Tier 3 op wrapper: the gate must still demand an artifact, but
    # must not demand a backbone line there is no backbone file for.
    TIER3_DIFF = (
        "diff --git a/aiter/ops/gemm_op_a4w4.py b/aiter/ops/gemm_op_a4w4.py\n"
        "--- a/aiter/ops/gemm_op_a4w4.py\n+++ b/aiter/ops/gemm_op_a4w4.py\n@@ -1,0 +2 @@\n"
        "+def gemm_a4w4_blockscale(x): pass\n")

    def gate(self, text, diff=None, write=True):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            if write:
                (d / "core_files.txt").write_text(text)
            (d / "d.diff").write_text(self.DIFF if diff is None else diff)
            return subprocess.run(
                [sys.executable, str(TRIAGE), "corefiles", str(d / "core_files.txt"),
                 str(d / "d.diff")], capture_output=True, text=True)

    def test_an_assessment_of_every_backbone_file_passes(self):
        r = self.gate(self.GOOD)
        self.assertEqual(0, r.returncode, r.stdout + r.stderr)
        self.assertIn("CORE FILES ASSESSED: 2/2", r.stdout)

    def test_a_missing_artifact_is_a_hard_failure(self):
        """Not writing the file was the cheapest way past every gate that tolerated it."""
        r = self.gate("", write=False)
        self.assertEqual(1, r.returncode)
        self.assertIn("CORE-FILES MISSING", r.stderr)

    def test_an_empty_artifact_is_rejected(self):
        r = self.gate("")
        self.assertEqual(1, r.returncode)
        self.assertIn("EMPTY", r.stdout)

    def test_a_backbone_file_with_no_line_is_named(self):
        r = self.gate(self.GOOD.split("\n")[0] + "\n")
        self.assertEqual(1, r.returncode)
        self.assertIn("UNASSESSED: aiter/__init__.py", r.stdout)

    def test_a_reason_that_would_fit_any_pr_is_rejected(self):
        """"core file, large blast radius, looks fine" is not an assessment of THIS diff:
        it is a sentence about the file, and it is true whoever wrote the patch."""
        r = self.gate(
            "aiter/fused_moe.py TIER2 COVERED -- this is a core file with a large blast "
            "radius, but the change looks fine to me\n"
            "aiter/__init__.py TIER1 COVERED -- core import chain, seems fine overall and "
            "nothing here looks broken\n")
        self.assertEqual(1, r.returncode)
        self.assertEqual(2, r.stdout.count("UNANCHORED"))

    def test_a_reason_anchored_only_in_untouched_code_is_rejected(self):
        """Citing a caller this PR does not change is legitimate *in addition* -- an
        uncovered caller is exactly what a GAP looks like. What is not legitimate is a
        reason whose only concrete referent is code the PR never touched: nothing in it
        is evidence that this change was looked at."""
        r = self.gate(
            "aiter/fused_moe.py TIER2 COVERED -- callers in aiter/ops/shuffle.py and "
            "aiter/ops/topk.py were checked and are unaffected by any of this\n"
            "aiter/__init__.py TIER1 COVERED -- the import block in aiter/mha.py is "
            "unrelated, so no truncation risk is introduced anywhere\n")
        self.assertEqual(1, r.returncode)
        self.assertEqual(2, r.stdout.count("UNANCHORED"))

    def test_an_untouched_caller_alongside_a_real_anchor_passes(self):
        """The inverse of the test above, and the reason the check is anchoring rather
        than citation-banning: a GAP naming the uncovered caller must survive."""
        r = self.gate(
            "aiter/fused_moe.py TIER2 GAP -- num_local_experts changes moe_sorting_fwd's "
            "arity and aiter/ops/shuffle.py calls it positionally, uncovered by any test\n"
            "aiter/__init__.py TIER1 COVERED -- gemm_op_a4w4 is appended at the end of "
            "the block with no try/except, so nothing below it can be truncated\n")
        self.assertEqual(0, r.returncode, r.stdout + r.stderr)

    def test_a_bare_verdict_with_no_reason_is_rejected(self):
        r = self.gate(
            "aiter/fused_moe.py TIER2 COVERED -- ok\n"
            "aiter/__init__.py TIER1 COVERED -- fine\n")
        self.assertEqual(1, r.returncode)
        self.assertEqual(2, r.stdout.count("NO-EVIDENCE"))

    def test_downgrading_a_tier_is_not_a_way_out(self):
        """The tier decides which checks the reason has to answer, so the cheapest escape
        from a Tier-2 assessment is to call the file Tier 3."""
        r = self.gate(self.GOOD.replace("TIER2", "TIER3"))
        self.assertEqual(1, r.returncode)
        self.assertIn("TIER-MISMATCH: aiter/fused_moe.py", r.stdout)

    def test_a_line_about_a_file_the_pr_does_not_change_is_rejected(self):
        r = self.gate(self.GOOD +
                      "aiter/mla.py TIER2 COVERED -- the mla decode path never calls "
                      "moe_sorting_fwd, so num_local_experts cannot reach it\n")
        self.assertEqual(1, r.returncode)
        self.assertIn("UNTOUCHED-FILE: aiter/mla.py", r.stdout)

    def test_a_malformed_line_is_not_silently_ignored(self):
        r = self.gate("aiter/fused_moe.py is risky\n" + self.GOOD.split("\n", 1)[1])
        self.assertEqual(1, r.returncode)
        self.assertIn("MALFORMED", r.stdout)

    def test_a_tier3_only_diff_may_declare_none(self):
        r = self.gate("NONE -- no Tier 1/2 file here; only aiter/ops/gemm_op_a4w4.py, a "
                      "single op wrapper\n", diff=self.TIER3_DIFF)
        self.assertEqual(0, r.returncode, r.stdout + r.stderr)
        self.assertIn("none in this diff, declared", r.stdout)

    def test_declaring_none_over_a_backbone_diff_is_rejected(self):
        """The one declaration that would let the whole step be skipped."""
        r = self.gate("NONE -- nothing important changed here\n")
        self.assertEqual(1, r.returncode)
        self.assertIn("UNDECLARED-CORE", r.stdout)

    def test_an_extra_line_for_a_tier3_file_in_the_diff_is_allowed(self):
        """Q2/Q3 need judgement about a new file, so the reviewer may assess more than the
        table knows. Only the table is demanded; extra lines are not punished."""
        r = self.gate(self.GOOD +
                      "op_tests/test_moe.py TIER3 COVERED -- adds test_num_local_experts, "
                      "which exercises the new moe_sorting_fwd arity end to end\n")
        self.assertEqual(0, r.returncode, r.stdout + r.stderr)

    def test_a_missing_diff_fails_closed(self):
        """Without the diff the backbone set is empty, and an empty backbone set would
        accept any artifact at all -- including one written for a different PR."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td)
            (d / "core_files.txt").write_text(self.GOOD)
            r = subprocess.run(
                [sys.executable, str(TRIAGE), "corefiles", str(d / "core_files.txt"),
                 str(d / "nope.diff")], capture_output=True, text=True)
        self.assertEqual(1, r.returncode)
        self.assertIn("CORE-FILES UNUSABLE", r.stderr)

    def test_step_4_writes_an_artifact_and_step_8_gates_on_it(self):
        body = SKILL_MD.read_text()
        step4 = body[body.index("## Step 4"):body.index("## Step 5")]
        self.assertIn("core_files.txt", step4,
                      "Step 4 names no artifact, so it can be skipped in silence again")
        step8 = body[body.index("## Step 8"):]
        self.assertRegex(step8, r"triage\.py\"?\s+corefiles\b",
                         "Step 8 does not run the corefiles gate")

    def test_step_8_no_longer_calls_itself_three_gates(self):
        """The count in the prose is the only thing telling a reader one is missing."""
        step8 = SKILL_MD.read_text()
        step8 = step8[step8.index("## Step 8"):]
        self.assertNotIn("The three gates", step8)


def triage_families(added):
    """triton_families() out of triage.py, loaded once, without importing the CLI."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_triage_mod", TRIAGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.triton_families(added)


class TestRecallDetectors(unittest.TestCase):
    """Two gaps found by replaying the 600-PR corpus, and the shapes that must stay silent.

    Both were measured before they were written, and a third candidate was measured and
    dropped: added `except` blocks whose body is pass/continue/return fired on 10.7% of the
    corpus, but 38 of those 64 PRs only ever matched a capability probe, and six sampled
    from the narrowed pass/continue set were all deliberate -- a barrier timeout documented
    in a comment, `int(os.environ[...])` falling through to a default, a ROCm-discovery
    fallback chain. That is the shape of the already-rejected weak no-assertion check, and
    it is not in the tree. Do not add it back without new numbers."""

    def run_triage(self, mode, diff, *rest):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "pr.diff"
            f.write_text(diff)
            r = subprocess.run([sys.executable, str(TRIAGE), mode, str(f), *rest],
                               capture_output=True, text=True)
            return r.stdout

    def sig_diff(self, path, body, hunk_scope="def _kernel("):
        return ("diff --git a/%s b/%s\n--- a/%s\n+++ b/%s\n@@ -1,6 +1,5 @@ %s\n"
                % (path, path, path, path, hunk_scope)) + body

    def new_file(self, path, lines):
        return ("diff --git a/%s b/%s\nnew file mode 100644\n--- /dev/null\n+++ b/%s\n"
                "@@ -0,0 +1,%d @@\n" % (path, path, path, len(lines))
                + "".join("+%s\n" % l for l in lines))

    # -- a parameter dropped from a signature reaches the api-signature family ----
    def test_parameter_deleted_from_signature_derives_api_signature(self):
        """The family required a `def` line on BOTH sides -- a signature rewritten in
        place. Deleting one line of a multi-line signature never touched it, so B6/E1/E5
        were never asked about 3.5% of open PRs."""
        d = self.sig_diff("aiter/ops/triton/k.py",
                          " def _kernel(\n     a_ptr,\n-    q_descale_ptr,\n     b_ptr,\n")
        self.assertIn("api-signature", self.run_triage("rules", d, "t"))

    def test_argument_dropped_from_a_call_is_not_a_parameter(self):
        """A hunk header names the ENCLOSING def, so `transpose_out=True,` passed to a
        call in the body of `def asm_moe(` read as a dropped parameter of asm_moe."""
        d = self.sig_diff("aiter/fused_moe_bf16_asm.py",
                          "     out = ck_moe(\n         hidden,\n"
                          "-                transpose_out=True,\n     )\n",
                          hunk_scope="def asm_moe(")
        self.assertNotIn("api-signature", self.run_triage("rules", d, "t"))

    def test_a_renamed_parameter_is_still_an_api_change(self):
        """Written first as "a rename is not a drop", which was the wrong expectation:
        the old name is gone from the signature either way, and a downstream caller
        passing it as a kwarg breaks. That is what B6/E1/E5 are for."""
        d = self.sig_diff("aiter/ops/triton/k.py",
                          " def _kernel(\n     a_ptr,\n-    q_descale_ptr,\n"
                          "+    q_descale_ptr_v2,\n     b_ptr,\n")
        self.assertIn("api-signature", self.run_triage("rules", d, "t"))

    def test_a_reordered_parameter_is_not_a_drop(self):
        """The name still appears in the signature, so nothing was removed from it."""
        d = self.sig_diff("aiter/ops/triton/k.py",
                          " def _kernel(\n-    q_descale_ptr,\n     a_ptr,\n"
                          "+    q_descale_ptr,\n     b_ptr,\n")
        self.assertNotIn("api-signature", self.run_triage("rules", d, "t"))

    def test_cpp_signatures_are_left_alone(self):
        """The same line pattern reads `std::optional<Tensor>& fp8_out,` as a parameter
        named `std`; telling a type prefix from a name there needs a real parser."""
        d = self.sig_diff("csrc/kernels/attention_ragged.cu",
                          " void launcher(\n     torch::Tensor& out,\n"
                          "-    std::optional<torch::Tensor>& fp8_out,\n     int n,\n",
                          hunk_scope="void launcher(")
        self.assertNotIn("api-signature", self.run_triage("rules", d, "t"))

    # -- kerneltest --------------------------------------------------------------
    KERNEL = ["def k():"] + ["    pass"] * 12

    def test_new_kernel_without_any_test_is_reported(self):
        d = self.new_file("aiter/ops/triton/_triton_kernels/q.py", self.KERNEL)
        self.assertIn("NO COLLECTIBLE TEST", self.run_triage("kerneltest", d))

    def test_a_collectible_test_outside_op_tests_still_counts(self):
        """aiter#3991 adds five test files under aiter/aot/flydsl/tests/; an op_tests-only
        whitelist called it untested, which is a false statement, not a missing one."""
        d = self.new_file("aiter/ops/triton/_triton_kernels/q.py", self.KERNEL) \
            + self.new_file("aiter/aot/flydsl/tests/test_aot.py",
                            ["def test_materializes():", "    assert True"])
        self.assertNotIn("NO COLLECTIBLE TEST", self.run_triage("kerneltest", d))

    def test_a_benchmark_is_not_a_test(self):
        d = self.new_file("aiter/ops/triton/_triton_kernels/q.py", self.KERNEL) \
            + self.new_file("op_tests/op_benchmarks/triton/bench_q.py",
                            ["def test_bench():", "    run()"])
        self.assertIn("NO COLLECTIBLE TEST", self.run_triage("kerneltest", d))

    def test_bench_in_the_name_does_not_disqualify_a_test_file(self):
        """aiter#2889's test_rmsnorm_bench_against_aiter.py holds two collectible tests."""
        d = self.new_file("aiter/ops/triton/_triton_kernels/q.py", self.KERNEL) \
            + self.new_file("aiter/ops/flydsl/test_rmsnorm_bench_against_aiter.py",
                            ["def test_flydsl_rmsnorm_case():", "    assert True"])
        self.assertNotIn("NO COLLECTIBLE TEST", self.run_triage("kerneltest", d))

    def test_a_modified_kernel_is_not_a_new_one(self):
        d = ("diff --git a/aiter/ops/triton/_triton_kernels/q.py "
             "b/aiter/ops/triton/_triton_kernels/q.py\n"
             "--- a/aiter/ops/triton/_triton_kernels/q.py\n"
             "+++ b/aiter/ops/triton/_triton_kernels/q.py\n@@ -1 +1,2 @@\n+    x = 1\n")
        self.assertNotIn("NO COLLECTIBLE TEST", self.run_triage("kerneltest", d))


class TestSiblingVariants(unittest.TestCase):
    """A1's evidence. The rule names the shape -- a decode kernel fixed while `_prefill_opt`
    beside it was not -- and had no forensic answer for it, because twins compares whole
    files and A1's own example is two functions in one.

    The discriminator is the shared name stem. Without it any two functions in the same
    file qualify and the check fires on 23.3% of the corpus, mostly on helpers that merely
    sit together; with it, 18.3%. A list of variant suffixes in place of the stem gives a
    tidier 10.8% and loses kernel_unified_attention_2d against _3d, which is the shape the
    rule exists for -- the same failure D9's body records."""

    FILE = '''
def _dynamic_per_tensor_quant_fp8_i8_kernel(x_in_ptr, offs, mask):
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")
    return x * scale

def _static_per_tensor_quant_fp8_i8_kernel(x_in_ptr, offs, mask):
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")
    return x

def _unrelated_helper_name(a):
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")
    return a
'''

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        (self.root / "aiter/ops/triton").mkdir(parents=True)
        (self.root / "aiter/ops/triton/quant.py").write_text(self.FILE)

    def tearDown(self):
        self._tmp.cleanup()

    def diff(self, deleted, scope="def _dynamic_per_tensor_quant_fp8_i8_kernel(",
             path="aiter/ops/triton/quant.py", new=False):
        head = "diff --git a/%s b/%s\n" % (path, path)
        if new:
            head += "new file mode 100644\n"
        return head + ("--- a/%s\n+++ b/%s\n@@ -1,4 +1,4 @@ %s\n-%s\n+    x = 0\n"
                       % (path, path, scope, deleted))

    def run_siblings(self, diff):
        f = self.root / "pr.diff"
        f.write_text(diff)
        return subprocess.run([sys.executable, str(TRIAGE), "siblings", str(f), str(self.root)],
                              capture_output=True, text=True).stdout

    LINE = '    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")'

    def test_a_variant_still_carrying_the_changed_line_is_reported(self):
        out = self.run_siblings(self.diff(self.LINE))
        self.assertIn("_static_per_tensor_quant_fp8_i8_kernel", out)

    def test_a_function_that_is_not_a_variant_is_not_a_sibling(self):
        """It holds the identical line; it is not a variant of the changed function."""
        out = self.run_siblings(self.diff(self.LINE))
        self.assertNotIn("_unrelated_helper_name", out)

    def test_a_new_file_has_no_siblings_to_diverge_from(self):
        out = self.run_siblings(self.diff(self.LINE, new=True))
        self.assertIn("no variant of a changed function", out)

    def test_a_test_file_is_not_scanned(self):
        (self.root / "op_tests").mkdir()
        (self.root / "op_tests/test_q.py").write_text(self.FILE)
        out = self.run_siblings(self.diff(self.LINE, path="op_tests/test_q.py"))
        self.assertIn("no variant of a changed function", out)

    def test_an_import_line_is_not_evidence(self):
        line = "from aiter.ops.flydsl.moe_kernels import launch_moe_sorting"
        (self.root / "aiter/ops/triton/quant.py").write_text(
            self.FILE + "\ndef _static_per_tensor_quant_fp8_i8_kernel_v2(a):\n    %s\n" % line)
        self.assertIn("no variant of a changed function", self.run_siblings(self.diff(line)))

    def test_a_bare_rebinding_is_not_evidence(self):
        """`dev = torch.device("cpu")` is shared because both variants need a device."""
        line = '    dev = torch.device("cpu")'
        (self.root / "aiter/ops/triton/quant.py").write_text(
            self.FILE.replace('    return x\n', '    return x\n' + line + "\n"))
        self.assertIn("no variant of a changed function", self.run_siblings(self.diff(line)))

    def test_the_gate_is_silent_rather_than_loud_when_the_file_is_gone(self):
        """The tree is the authority; a path it does not have yields nothing, not a crash."""
        out = self.run_siblings(self.diff(self.LINE, path="aiter/ops/triton/absent.py"))
        self.assertIn("no variant of a changed function", out)


class TestD11WorkedExampleIsExecutable(unittest.TestCase):
    """D11's worked example is a claim about the repository, so run it against the
    repository instead of trusting the prose.

    This class exists because the claim was checked against the wrong tree.
    /mnt/raid0/zufa/aiter is a second checkout two months behind main, and in it nothing
    pins `pa_sparse_prefill_kargs` -- so a sweep concluded the example was fiction, that
    D11 could not fire on aiter#5220, and that the rule body needed rewriting. The pin is
    there on the branch under review: `pa_sparse_prefill_opus_kernels.cu` asserts
    `sizeof(pa_sparse_prefill_kargs) == 112` and fixes each field offset through the
    PA_GFX1250_CO_ABI macro, which is also why grepping for a literal `offsetof(` found
    nothing. The rule was right and the measurement was stale."""

    TREE = SKILL_DIR.parents[2]

    def setUp(self):
        if not (self.TREE / "csrc").is_dir():
            self.skipTest("not running inside an aiter checkout")

    def test_the_struct_the_rule_cites_is_pinned_in_this_tree(self):
        hits = [f for f in self.TREE.glob("csrc/**/*.cu")
                if "sizeof(pa_sparse_prefill_kargs)" in f.read_text(errors="replace")]
        self.assertTrue(
            hits,
            "D11's worked example says this struct's size is asserted. Nothing in this "
            "tree asserts it. Either the tree moved and the rule body needs a new example, "
            "or the tree being read is not the merge target.")

    def test_d11_fires_on_its_own_worked_example(self):
        """aiter#5220 adds two ints to the pinned struct and touches no assertion."""
        diff = ("diff --git a/csrc/include/pa_sparse_prefill_opus.h "
                "b/csrc/include/pa_sparse_prefill_opus.h\n"
                "--- a/csrc/include/pa_sparse_prefill_opus.h\n"
                "+++ b/csrc/include/pa_sparse_prefill_opus.h\n"
                "@@ -129,6 +129,8 @@ struct pa_sparse_prefill_kargs\n"
                "     int stride_qo_h;\n+    int stride_o_n;\n+    int stride_o_h;\n"
                "     int stride_kv_page;\n")
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "pr.diff"
            f.write_text(diff)
            out = subprocess.run(
                [sys.executable, str(TRIAGE), "structabi", str(f), str(self.TREE)],
                capture_output=True, text=True).stdout
        self.assertIn("PINNED-LAYOUT", out)
        self.assertIn("pa_sparse_prefill_kargs", out)

    def test_it_stays_silent_when_the_same_diff_updates_the_assertions(self):
        diff = ("diff --git a/csrc/include/pa_sparse_prefill_opus.h "
                "b/csrc/include/pa_sparse_prefill_opus.h\n"
                "--- a/csrc/include/pa_sparse_prefill_opus.h\n"
                "+++ b/csrc/include/pa_sparse_prefill_opus.h\n"
                "@@ -129,6 +129,8 @@ struct pa_sparse_prefill_kargs\n"
                "     int stride_qo_h;\n+    int stride_o_n;\n"
                "-static_assert(sizeof(pa_sparse_prefill_kargs) == 112,\n"
                "+static_assert(sizeof(pa_sparse_prefill_kargs) == 116,\n")
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "pr.diff"
            f.write_text(diff)
            out = subprocess.run(
                [sys.executable, str(TRIAGE), "structabi", str(f), str(self.TREE)],
                capture_output=True, text=True).stdout
        self.assertNotIn("PINNED-LAYOUT", out)


class TestTreeRootGuard(unittest.TestCase):
    """Root-taking forensics answer questions about the merge target; a second checkout
    answers them about itself, in the same shape, with no sign anything is wrong.

    A sweep and the follow-up that checked it both read a checkout 53 days behind and
    agreed that nothing pinned `pa_sparse_prefill_kargs`, so D11's worked example was
    filed as fiction and the rule body was rewritten around the mistake. Two independent
    greps confirmed it because the offsets are pinned through a macro, so a literal
    `offsetof(` is missing from both trees. fetch.sh cannot make this error -- it takes
    the root from `git rev-parse --show-toplevel` -- and the hand-run measurement is
    exactly where it happened."""

    DIFF = ("diff --git a/csrc/x.h b/csrc/x.h\n--- a/csrc/x.h\n+++ b/csrc/x.h\n"
            "@@ -1,2 +1,3 @@ struct foo\n     int a;\n+    int b;\n")

    def run_mode(self, root):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "pr.diff"
            f.write_text(self.DIFF)
            return subprocess.run(
                [sys.executable, str(TRIAGE), "structabi", str(f), str(root)],
                capture_output=True, text=True)

    def test_the_skills_own_tree_is_not_flagged(self):
        r = self.run_mode(SKILL_DIR.parents[2])
        self.assertNotIn("warning:", r.stderr)

    def test_a_different_tree_is_flagged_with_both_heads(self):
        with tempfile.TemporaryDirectory() as other:
            r = self.run_mode(other)
        self.assertIn("warning:", r.stderr)
        self.assertIn("this skill ships from", r.stderr)

    def test_the_warning_never_reaches_stdout(self):
        """fetch.sh tees these modes straight into $WORK/*.txt. A warning on stdout would
        be written into the evidence as though it were a finding."""
        with tempfile.TemporaryDirectory() as other:
            r = self.run_mode(other)
        self.assertNotIn("warning:", r.stdout)
        self.assertNotIn("ships from", r.stdout)

    def test_every_root_taking_mode_goes_through_the_guard(self):
        """symbols, twins, citest, structabi, siblings all read the tree. One that takes
        a root straight from argv is one that can be pointed at a stale checkout in
        silence."""
        src = TRIAGE.read_text()
        block = src[src.index("if __name__"):]
        raw = re.findall(r"^\s+root = pathlib\.Path\(sys\.argv\[3\]", block, re.M)
        self.assertEqual(
            [], raw,
            "a root-taking mode bypasses tree_root(); route it through the guard")


class TestTritonTriggersCarryTheirMaterial(unittest.TestCase):
    """T5 and T6 are both red rules, and both were triggered by tokens that say "this is
    Triton code" rather than "this shape is present".

    Measured over the 232 Triton PRs in the corpus: `tl.cdiv(` on its own produced 7 of
    T6's firings and all 7 were a tile count, a block-pointer `shape=`, a mask bound or a
    loop bound -- ceiling division, never a grid. A bare `.to(tl.float32)` produced 20 of
    T5's 54 and every sampled one was an upcast on a load, a store, or a quantisation max,
    which is the correct direction rather than the precision loss the rule is about.

    Narrowing is not only subtraction: naming the manual accumulator brought in aiter#3613,
    `acc_sq += gl.sum(out_h * out_h, axis=1)` with no `tl.dot` anywhere, which is exactly
    T5's first failure shape and which the old trigger never saw."""

    def fams(self, added):
        return dict(triage_families(added))

    def test_ceiling_division_alone_is_not_a_grid(self):
        for line in ("    num_valid_n_tiles = tl.cdiv(N_o, BLOCK_SIZE_N * 2)",
                     "    mask_n_s = offs_n_s < tl.cdiv(yN, 32)",
                     "    for i_l in range(tl.cdiv(n_res, BL)):",
                     "        shape=(M, tl.cdiv(K, MX_PACK_DIVISOR)),"):
            self.assertNotIn("triton-grid-map", self.fams(line), line)

    def test_a_host_grid_or_a_program_id_still_triggers_t6(self):
        self.assertIn("triton-grid-map", self.fams("grid = lambda META: (  # noqa: E731"))
        self.assertIn("triton-grid-map", self.fams("    pid = tl.program_id(0)"))

    def test_an_upcast_alone_is_not_an_accumulator(self):
        for line in ("    m = tl.max(tl.abs(x)).to(tl.float32)",
                     "    x_tile = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)",
                     "    tl.store(GateScal + pos, w.to(tl.float32), mask=live)"):
            self.assertNotIn("triton-accum-prec", self.fams(line), line)

    def test_a_named_accumulator_triggers_t5_without_a_dot(self):
        """aiter#3613: a running sum of squares in a Gluon kernel, no tl.dot in the diff."""
        self.assertIn("triton-accum-prec",
                      self.fams("        acc_sq += gl.sum(out_h * out_h, axis=1)"))
        self.assertIn("triton-accum-prec",
                      self.fams("    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)"))

    def test_a_dot_still_triggers_t5(self):
        self.assertIn("triton-accum-prec", self.fams("    acc2 = tl.dot(a, b)"))

    def test_the_rule_bodies_quote_the_triggers_that_are_live(self):
        """rules.md states each trigger in prose. Prose drifts; this fails when it has."""
        body = RULES_MD.read_text()
        t6 = body[body.index("**T6 —"):body.index("**T6 —") + 600]
        self.assertNotIn("`tl.cdiv(`, ", t6.split("Trigger:")[1][:120],
                         "T6's stated trigger still lists tl.cdiv as live")
        t5 = body[body.index("**T5 —"):body.index("**T5 —") + 600]
        self.assertIn("accumulator", t5.split("Trigger:")[1][:200])


class TestGuardChanges(unittest.TestCase):
    """D4 is red and `invariant-removed` fires on 69 of 600 open PRs, so what happened to
    each deleted guard is work the reviewer does on every one of them.

    Written first as a filter -- 14.5% re-add every guard verbatim, another quarter return
    them "reworded on the same subject", so suppressing both looked like removing 40% of
    the noise. Reading seven of the reworded ones killed that: aiter#4295 replaces a null
    check with a size check and the null check is gone, aiter#4279 halves a TORCH_CHECK
    bound, aiter#5168 widens an equality to a set membership. Three of seven were real
    weakenings, which is the finding, not the noise. So it reports the pair."""

    def diff(self, path, minus, plus=()):
        head = ("diff --git a/%s b/%s\n--- a/%s\n+++ b/%s\n@@ -1,4 +1,4 @@\n"
                % (path, path, path, path))
        return head + "".join("-%s\n" % m for m in minus) + "".join("+%s\n" % a for a in plus)

    def guards(self, diff):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "pr.diff"
            f.write_text(diff)
            return subprocess.run([sys.executable, str(TRIAGE), "guards", str(f)],
                                  capture_output=True, text=True).stdout

    def test_a_guard_added_back_verbatim_is_a_non_event(self):
        line = "    assert num_head_qo % 16 == 0"
        out = self.guards(self.diff("aiter/ops/mla.py", [line], [line]))
        self.assertIn("moved unchanged", out)
        self.assertIn("0 returned in changed form, 0 gone", out)
        self.assertNotIn("  gone:", out)
        self.assertNotIn("  changed:", out)

    def test_a_guard_that_simply_leaves_is_reported_gone(self):
        out = self.guards(self.diff("aiter/ops/mla.py",
                                    ["    assert num_head_qo % 16 == 0"], ["    pass"]))
        self.assertIn("  gone:", out)
        self.assertIn("num_head_qo", out)

    def test_a_weakened_bound_is_shown_as_a_pair(self):
        """aiter#4279 halves it. Filtering this out as "the check came back" was the
        mistake this class exists to prevent."""
        out = self.guards(self.diff(
            "csrc/k.cu",
            ["  TORCH_CHECK(hidden_size >= (tile_k * split_k) * 2, \"msg\");"],
            ["  TORCH_CHECK(hidden_size >= (tile_k * split_k), \"msg\");"]))
        self.assertIn("  changed:", out)
        self.assertIn("* 2", out)
        self.assertNotIn("moved unchanged", out)

    def test_a_null_check_swapped_for_a_size_check_is_shown_as_a_pair(self):
        """aiter#4295. Same subject, and the null guard is gone."""
        out = self.guards(self.diff(
            "csrc/k.cu",
            ["  AITER_CHECK(valid_split_count != nullptr && valid_split_count->data_ptr(),"],
            ["  AITER_CHECK(valid_split_count->size(0) >= num_seqs,"]))
        self.assertIn("  changed:", out)
        self.assertIn("nullptr", out)

    def test_comments_are_not_guards(self):
        """aiter#4062 condenses comments across 252 files; prose mentioning contiguous is
        not a deleted invariant, and the family already learned this once."""
        out = self.guards(self.diff("aiter/ops/mla.py",
                                    ["    # x must be .contiguous() before the asm call"]))
        self.assertIn("deletes no assert", out)

    def test_a_diff_with_no_guards_says_so_rather_than_nothing(self):
        out = self.guards(self.diff("aiter/ops/mla.py", ["    x = 1"], ["    x = 2"]))
        self.assertIn("deletes no assert", out)

    def test_fetch_runs_it(self):
        self.assertIn('triage.py" guards', FETCH.read_text())


class TestSymbolSweepWorkedExample(unittest.TestCase):
    """The sweep's example is prose about the tree, so run it against the tree.

    D11's example was prose too, and reading it against the wrong checkout produced a
    confident, wrong conclusion that the rule was fiction. The same shape is here:
    aiter#4994 added `from aiter.ops.flydsl.utils import is_flydsl_available` and #5116 had
    deleted that module 19 hours earlier; #4994 merged green and was reverted 5.5 hours
    later. Whether this check fires at all is decided by which tree it resolves against,
    which is exactly the thing prose cannot hold onto."""

    TREE = SKILL_DIR.parents[2]

    def setUp(self):
        if not (self.TREE / "aiter").is_dir():
            self.skipTest("not running inside an aiter checkout")

    def sweep(self, diff):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "pr.diff"
            f.write_text(diff)
            return subprocess.run(
                [sys.executable, str(TRIAGE), "symbols", str(f), str(self.TREE)],
                capture_output=True, text=True).stdout

    def diff_adding(self, line, path="aiter/ops/flydsl/foo.py"):
        return ("diff --git a/%s b/%s\n--- a/%s\n+++ b/%s\n@@ -1 +1,2 @@\n+%s\n"
                % (path, path, path, path, line))

    def test_the_module_the_docstring_names_is_still_gone(self):
        """If it comes back, the worked example stops being one and the docstring lies."""
        out = self.sweep(self.diff_adding(
            "from aiter.ops.flydsl.utils import is_flydsl_available"))
        self.assertIn("UNRESOLVED-IMPORT", out)
        self.assertIn("aiter.ops.flydsl.utils", out)

    def test_a_module_that_exists_is_not_reported(self):
        """The control. Without it the test above passes on a sweep that flags everything."""
        out = self.sweep(self.diff_adding("from aiter.ops.triton.utils.types import torch_to_triton_dtype"))
        self.assertNotIn("UNRESOLVED-IMPORT", out)

    def test_a_module_this_very_diff_adds_is_not_reported(self):
        """`Files the diff ITSELF adds count as existing` -- otherwise every PR that adds a
        module and imports it would be reported for importing its own new file."""
        d = self.diff_adding("from aiter.ops.flydsl.brand_new_thing import go")
        d += ("diff --git a/aiter/ops/flydsl/brand_new_thing.py "
              "b/aiter/ops/flydsl/brand_new_thing.py\nnew file mode 100644\n"
              "--- /dev/null\n+++ b/aiter/ops/flydsl/brand_new_thing.py\n@@ -0,0 +1 @@\n"
              "+def go(): pass\n")
        self.assertNotIn("UNRESOLVED-IMPORT", self.sweep(d))

    def test_a_third_party_import_is_left_alone(self):
        out = self.sweep(self.diff_adding("import numpy as np"))
        self.assertNotIn("UNRESOLVED-IMPORT", out)


class TestFlydslKernelsAreKernels(unittest.TestCase):
    """FlyDSL is a kernel backend and the deriver did not treat it as one.

    KERNEL_PY lists `aiter/ops/triton/`, `_triton_kernels/`, `_gluon_kernels/` and
    `/gluon/`. A FlyDSL kernel matched none of them, so it reached `flydsl` -- D10 and D10b,
    which are about compile-result handling -- and nothing else. 108 of 600 open PRs edit
    `aiter/ops/flydsl/kernels/*.py` and 95 of them derived neither kernel family, so A1's
    sibling variant, D1's uninitialised accumulator, D8's missing contiguous check and P6's
    unmeasured cost were never put to an entire backend. The siblings forensic had already
    been pairing `_flydsl_stage1_wrapper` against `_flydsl_stage2_wrapper` in those files.

    Median rules per PR is unchanged at 15 and the mean moves 14.7 to 15.1: these PRs
    already carried most of what this adds."""

    def diff(self, path, added="    x = compile_flydsl_moe_stage1(a, b)"):
        return ("diff --git a/%s b/%s\n--- a/%s\n+++ b/%s\n@@ -1 +1,2 @@\n+%s\n"
                % (path, path, path, path, added))

    def fams(self, diff, title=""):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "pr.diff"
            f.write_text(diff)
            return subprocess.run([sys.executable, str(TRIAGE), "rules", str(f), title],
                                  capture_output=True, text=True).stdout

    def test_a_flydsl_kernel_derives_the_kernel_rules(self):
        out = self.fams(self.diff("aiter/ops/flydsl/kernels/moe_gemm_2stage.py"))
        self.assertIn("flydsl-kernel", out)
        for rule in ("A1", "D1", "D8", "P6"):
            self.assertRegex(out, r"flydsl-kernel\s*\][^\n]*\b%s\b" % rule)

    def test_it_does_not_claim_the_triton_mask_rule(self):
        """B2 is `tl.load`/`tl.store` without a mask. There is no tl in FlyDSL."""
        out = self.fams(self.diff("aiter/ops/flydsl/kernels/moe_gemm_2stage.py"))
        line = [l for l in out.splitlines() if "flydsl-kernel" in l][0]
        self.assertNotIn("B2", line)

    def test_the_compile_result_family_still_fires_too(self):
        out = self.fams(self.diff("aiter/ops/flydsl/kernels/moe_gemm_2stage.py"))
        self.assertIn("[flydsl ", out)

    def test_a_flydsl_file_outside_kernels_is_not_one(self):
        """`aiter/ops/flydsl/utils.py` is a helper module, not a kernel."""
        out = self.fams(self.diff("aiter/ops/flydsl/utils.py"))
        self.assertNotIn("flydsl-kernel", out)

    def test_a_non_python_file_under_kernels_is_left_to_the_c_path(self):
        """`.cu`/`.cuh` already reach modified-kernel through is_kernel, whatever the path."""
        out = self.fams(self.diff("aiter/ops/flydsl/kernels/foo.cuh", "+  int x = 1;"))
        self.assertIn("modified-kernel", out)

    def test_mapping_lists_the_new_family(self):
        self.assertIn("flydsl-kernel", (SKILL_DIR / "MAPPING.md").read_text())


class TestGatesFoundByRunningTheSkill(unittest.TestCase):
    """Four defects that only surfaced when the skill was run end to end on real PRs.

    Everything before this was measured against the corpus, which exercises the deriver
    and the forensics and never once writes a card. Four full reviews -- aiter#4295,
    #2478, #3836, #5072 -- found these in one pass, and three of the four reported the
    citation conflict independently."""

    def gate(self, card, verdicts, diff):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "card.md").write_text(card)
            (d / "verdicts.txt").write_text(verdicts)
            (d / "pr.diff").write_text(diff)
            (d / "ai_diagnostic.txt").write_text("")
            (d / "answers.txt").write_text("")
            return subprocess.run(
                [sys.executable, str(TRIAGE), "card", str(d / "card.md"),
                 str(d / "verdicts.txt"), str(d / "ai_diagnostic.txt"),
                 str(d / "answers.txt"), str(d / "pr.diff")],
                capture_output=True, text=True)

    DIFF = ("diff --git a/aiter/mla.py b/aiter/mla.py\n--- a/aiter/mla.py\n"
            "+++ b/aiter/mla.py\n@@ -1 +1,2 @@\n+    x = 1\n")

    def test_a_card_obeying_the_no_rule_codes_rule_can_still_claim_its_fire(self):
        """SKILL.md: "Do NOT use rule codes (P1, D4, A1...) in output". The gate matched
        findings to fired rules by a leading `D8:`, so a card that followed the
        instruction reported every fired rule as UNREPORTED-FIRE. The verdict's own cited
        file is the same claim without the label."""
        r = self.gate(
            "\u26A0\uFE0F aiter/mla.py:1 drops the contiguous normalisation on the "
            "EP path at 8192 tokens\n",
            "D8 FIRE -- aiter/mla.py:1 no longer normalises the layout\n",
            self.DIFF)
        self.assertNotIn("UNREPORTED-FIRE", r.stdout)
        self.assertEqual(0, r.returncode, r.stdout)

    def test_an_unreported_fire_is_still_caught(self):
        """The control: the relaxation above must not turn the gate off."""
        r = self.gate(
            "\u26A0\uFE0F aiter/mla.py:1 something unrelated to the verdict\n",
            "D8 FIRE -- csrc/other.cu:9 a file the card never mentions\n",
            self.DIFF)
        self.assertIn("UNREPORTED-FIRE", r.stdout)

    def test_torch_cuda_is_not_a_file_called_torch_cu(self):
        """`torch.cuda.device(...)` and `x.contiguous.cuda()` are the two most ordinary
        expressions in a HIP review; both parsed as citations of files nobody wrote."""
        r = self.gate(
            "\u26A0\uFE0F aiter/mla.py:1 launches without torch.cuda.device(Q.device) "
            "so the module resolves against the wrong context at 2 GPUs\n",
            "D8 FIRE -- aiter/mla.py:1 no device guard\n",
            self.DIFF)
        self.assertNotIn("torch.cu", r.stdout)
        self.assertEqual(0, r.returncode, r.stdout)


class TestSilenceIsDistinguishableFromSkipping(unittest.TestCase):
    """SKILL.md tells the reader an empty artifact means the axis was not checked. Two
    forensics broke that: the symbol sweep wrote 0 bytes when it ran and found nothing,
    and ci_coverage said "every new test file lands where a CI job will run it" on a PR
    that adds no test file, which reads as a pass."""

    def run_mode(self, mode, diff):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "pr.diff"
            f.write_text(diff)
            return subprocess.run(
                [sys.executable, str(TRIAGE), mode, str(f), str(SKILL_DIR.parents[2])],
                capture_output=True, text=True).stdout

    CSV = ("diff --git a/aiter/configs/x.csv b/aiter/configs/x.csv\n"
           "--- a/aiter/configs/x.csv\n+++ b/aiter/configs/x.csv\n@@ -1 +1,2 @@\n+1,2,3\n")

    def test_a_clean_symbol_sweep_says_it_ran(self):
        out = self.run_mode("symbols", self.CSV)
        self.assertTrue(out.strip(), "an empty artifact reads as 'not checked'")
        self.assertIn("IMPORTS RESOLVED", out)

    def test_ci_coverage_does_not_pass_a_diff_with_no_tests(self):
        out = self.run_mode("citest", self.CSV)
        self.assertIn("not exercised", out)
        self.assertNotIn("lands where a CI job will run it", out)

    def test_ci_coverage_still_clears_a_real_test_file(self):
        d = ("diff --git a/op_tests/test_new.py b/op_tests/test_new.py\n"
             "new file mode 100644\n--- /dev/null\n+++ b/op_tests/test_new.py\n"
             "@@ -0,0 +1 @@\n+def test_x(): pass\n")
        self.assertIn("lands where a CI job will run it", self.run_mode("citest", d))


class TestArtifactsListMatchesWhatIsWritten(unittest.TestCase):
    """Three reviews went looking for evidence.txt because fetch.sh's closing line
    promises it unconditionally; it is written only when a guard or signature changed.
    The same line never learned about guards.txt, siblings.txt or kernel_tests.txt, which
    were added later -- by me, three commits apart, without touching it."""

    def test_every_named_artifact_is_one_the_script_writes(self):
        src = FETCH.read_text()
        line = [l for l in src.splitlines() if l.startswith("echo \"artifacts:")]
        self.assertTrue(line, "fetch.sh no longer prints an artifacts line")
        block = src[src.index('echo "artifacts:'):]
        block = block[:block.index('"', block.index("artifacts:") + 40) + 400]
        for name in ("guards.txt", "siblings.txt", "kernel_tests.txt"):
            self.assertIn(name, block, f"{name} is written but never announced")

    def test_the_conditional_artifact_is_announced_conditionally(self):
        src = FETCH.read_text()
        self.assertIn("${HAVE_EVIDENCE:+ evidence.txt}", src,
                      "evidence.txt is written only for guard/signature diffs; "
                      "announcing it unconditionally sends the reader after a file "
                      "that is not there")


class TestFireClaimingIsNotFreeloadable(unittest.TestCase):
    """Relaxing the FIRE claim to "names a file the verdict cited" made it too easy.

    aiter#2478 adjudicated three rules FIRE and every verdict cited aiter/fused_moe.py,
    so one sentence mentioning that path accounted for all three -- the review reported
    three real findings, but the gate could not tell that from one. Where N fired rules
    share a citation, N distinct findings have to name it."""

    DIFF = ("diff --git a/aiter/fused_moe.py b/aiter/fused_moe.py\n"
            "--- a/aiter/fused_moe.py\n+++ b/aiter/fused_moe.py\n@@ -1 +1,2 @@\n+    x = 1\n")
    VERDICTS = ("D1 FIRE -- aiter/fused_moe.py:1 wrong predicate\n"
                "P6 FIRE -- aiter/fused_moe.py:1 two extra launches\n"
                "E4 FIRE -- aiter/fused_moe.py:1 downstream skipped\n")

    def gate(self, card):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "card.md").write_text(card)
            (d / "v.txt").write_text(self.VERDICTS)
            (d / "pr.diff").write_text(self.DIFF)
            (d / "e.txt").write_text("")
            return subprocess.run(
                [sys.executable, str(TRIAGE), "card", str(d / "card.md"), str(d / "v.txt"),
                 str(d / "e.txt"), str(d / "e.txt"), str(d / "pr.diff")],
                capture_output=True, text=True)

    ONE = ("\u26A0\uFE0F aiter/fused_moe.py:1 uses the wrong predicate so the count "
           "never shrinks at 384 experts\n")

    def test_one_finding_cannot_claim_three_fires(self):
        r = self.gate(self.ONE)
        self.assertIn("UNREPORTED-FIRE", r.stdout)
        self.assertEqual(1, r.returncode)

    def test_three_findings_claim_three_fires(self):
        r = self.gate(self.ONE +
                      "\u26A0\uFE0F aiter/fused_moe.py:1 adds two kernel launches per "
                      "call on the non-EP path at 61 layers\n"
                      "\u26A0\uFE0F aiter/fused_moe.py:1 changes a shared contract with "
                      "no downstream CI label set\n")
        self.assertNotIn("UNREPORTED-FIRE", r.stdout)
        self.assertEqual(0, r.returncode, r.stdout)


class TestStalenessIsReported(unittest.TestCase):
    """Whether the diff still applies to the branch it targets.

    aiter#2478 is five months old and both hunks conflict; the review found that by
    thinking to run `git apply --check` itself. Nothing in Step 1b asked, and
    symbols.txt reporting every import resolved reads as evidence of no drift."""

    def test_fetch_writes_an_applies_artifact(self):
        src = FETCH.read_text()
        self.assertIn("applies.txt", src)
        self.assertIn("apply --check", src)

    def test_the_artifact_is_announced(self):
        src = FETCH.read_text()
        block = src[src.index('echo "artifacts:'):]
        self.assertIn("applies.txt", block[:500],
                      "written but not named, which is how evidence.txt got lost")

    def test_both_outcomes_say_which_one_it_is(self):
        src = FETCH.read_text()
        self.assertIn("APPLIES:", src)
        self.assertIn("STALE:", src)


class TestTheToolTextAgreesWithTheInstruction(unittest.TestCase):
    """SKILL.md was amended so that a missing test target is a defect only on a path that
    actually executes; fetch.sh went on printing the unconditional version.

    A control PR run against the amended SKILL.md caught it: the reviewer followed the
    document and overrode the tool, and said so. The next one may follow the tool. A fix
    that lands in the prose and not in the string the program prints is half a fix, and
    this is the exact path the original defect took."""

    def test_fetch_does_not_order_the_missing_target_reported_unconditionally(self):
        src = FETCH.read_text()
        i = src.find("validation REQUIRED but not run")
        self.assertGreater(i, 0, "the message moved; re-point this test")
        msg = src[i:i + 800]
        self.assertIn("EXECUTED at run time", msg)
        self.assertIn("codegen manifest", msg)

    def test_skill_md_carries_the_same_qualifier(self):
        body = (SKILL_DIR / "SKILL.md").read_text()
        i = body.find("no target existed to run")
        self.assertGreater(i, 0)
        self.assertIn("executed at run time", body[i:i + 700])

