"""Execution tests for fetch.sh's fail-closed guards.

TestFailClosedGuards in test_review_skill.py pins the *shape* of these guards by reading
the source: it was written when running fetch.sh meant a GitHub round trip, and it says so.
That leaves the guards themselves never executed -- a guard whose condition is written
backwards, or which is unreachable because an earlier line already aborted, passes that lint
and still lets the run continue in silence. Silence that reads like a clean result is the
failure mode this whole skill exists to prevent, so the guards get run.

The round trip is removed by putting a stub `gh` on PATH (tests/stubs/gh, answering out of a
per-scenario fixture directory) and a stub `git` that is the real git for every local
operation and an instant failure for anything naming a URL. fetch.sh reaches the network
only through `git fetch <https URL>` and both of those calls are already written to tolerate
failure, so this is the offline reviewer's path, not a fabricated one.

The other half is $PROJECT_ROOT: fetch.sh derives it from `git rev-parse --show-toplevel` of
the CWD and resolves every dependency under $PROJECT_ROOT/.claude/skills. So each scenario
builds a throwaway git repo with its own .claude/skills tree and runs the REAL fetch.sh with
CWD inside it. Nothing under the skill's own directory is copied or modified.

Pure stdlib, no GPU, no network.
"""
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

SKILL_DIR = pathlib.Path(__file__).resolve().parents[1]
SKILLS_ROOT = SKILL_DIR.parent
FETCH = SKILL_DIR / "fetch.sh"
TRIAGE = SKILL_DIR / "triage.py"
RULES_MD = SKILL_DIR / "rules.md"
STUBS = pathlib.Path(__file__).resolve().parent / "stubs"

HEAD_SHA = "1" * 40
BASE_SHA = "2" * 40

# A diff and file list that agree with each other, because triage reads the diff and the
# requirement pass reads the metadata's file list.
STATIC_DIFF = """diff --git a/README.md b/README.md
index 1111111..2222222 100644
--- a/README.md
+++ b/README.md
@@ -1,2 +1,2 @@
 title
-old line
+new line
"""

RUNTIME_DIFF = """diff --git a/csrc/kernels/foo.cu b/csrc/kernels/foo.cu
index 1111111..2222222 100644
--- a/csrc/kernels/foo.cu
+++ b/csrc/kernels/foo.cu
@@ -1,4 +1,5 @@
 __global__ void foo(const float* x, float* y, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
-  if (i < n) y[i] = x[i];
+  if (i < n) y[i] = x[i] * 2.0f;
+  __syncthreads();
 }
"""

# Adds exactly one op_tests/test_*.py, which is what makes the requirement pass name a
# target -- the precondition for the auto-validation block that holds the L361 guard.
RUNTIME_WITH_TARGET_DIFF = RUNTIME_DIFF + """diff --git a/op_tests/test_foo.py b/op_tests/test_foo.py
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/op_tests/test_foo.py
@@ -0,0 +1,3 @@
+def test_foo():
+    assert True
+# perftest
"""


def meta(paths, title="a change", body="no linked issue here"):
    return json.dumps({
        "title": title,
        "body": body,
        "number": 4242,
        "labels": [],
        "files": [{"path": p, "additions": 1, "deletions": 1} for p in paths],
        "author": {"login": "someone"},
        "reviews": [],
        "comments": [],
        "baseRefName": "main",
        "headRefOid": HEAD_SHA,
    }, indent=2)


class GuardHarness(unittest.TestCase):
    """Builds a self-contained project root and runs fetch.sh inside it."""

    maxDiff = None

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)
        self.real_git = shutil.which("git")
        self.assertTrue(self.real_git, "git is required to run these tests")

    def tearDown(self):
        self._tmp.cleanup()

    # -- fixtures ---------------------------------------------------------------
    def make_project(
        self,
        diff=STATIC_DIFF,
        paths=("README.md",),
        scanner="ok",          # ok | missing | not_executable | failing
        scale=True,
        schema=True,
        validator="ok",        # ok | missing | not_executable
        rules_md="real",       # real | drifted
        title="a change",
    ):
        root = self.tmp / "proj"
        (root / ".claude" / "skills" / "review-pr").mkdir(parents=True)
        vkp = root / ".claude" / "skills" / "validate-kernel-pr"
        vkp.mkdir(parents=True)
        subprocess.run([self.real_git, "init", "-q", str(root)], check=True)
        for key, value in (("user.email", "t@example.invalid"), ("user.name", "t")):
            subprocess.run([self.real_git, "-C", str(root), "config", key, value], check=True)
        (root / "README.md").write_text("title\nold line\n")
        subprocess.run([self.real_git, "-C", str(root), "add", "-A"], check=True)
        subprocess.run(
            [self.real_git, "-C", str(root), "commit", "-qm", "seed"],
            check=True, capture_output=True,
        )

        review = root / ".claude" / "skills" / "review-pr"
        os.symlink(TRIAGE, review / "triage.py")
        if rules_md == "real":
            os.symlink(RULES_MD, review / "rules.md")
        else:
            # A rules.md that carries none of the bodies the deriver names: the exact drift
            # the L741 guard is worded for.
            (review / "rules.md").write_text("# rules\n\nnothing here yet.\n")

        if scanner != "missing":
            scan = vkp / "scan_index_width.py"
            body = "#!/bin/sh\nexit 0\n" if scanner != "failing" else (
                "#!/bin/sh\necho 'scanner blew up' >&2\nexit 3\n")
            scan.write_text(body)
            scan.chmod(0o755 if scanner != "not_executable" else 0o644)
        if scale:
            (vkp / "production_scale.md").write_text("PRODUCTION SCALE: stub evidence\n")
        if schema:
            (vkp / "report_schema.json").write_text('{"type": "object"}\n')
        if validator != "missing":
            val = vkp / "validate_pr.sh"
            val.write_text("#!/bin/sh\nexit 0\n")
            val.chmod(0o755 if validator != "not_executable" else 0o644)

        gh_dir = self.tmp / "ghfix"
        gh_dir.mkdir(exist_ok=True)
        (gh_dir / "pr_meta.json").write_text(meta(paths, title=title))
        (gh_dir / "pr.diff").write_text(diff)
        (gh_dir / "base_sha.txt").write_text(BASE_SHA + "\n")
        (gh_dir / "comments.json").write_text("[]\n")
        self.gh_dir = gh_dir
        return root

    def run_fetch(self, cwd, args=("4242", "ROCm/aiter"), env=None, timeout=300):
        e = dict(os.environ)
        e["PATH"] = f"{STUBS}{os.pathsep}" + e["PATH"]
        e["GH_STUB_DIR"] = str(getattr(self, "gh_dir", self.tmp))
        e["REVIEW_TEST_REAL_GIT"] = self.real_git
        e.pop("REVIEW_AUTO_VALIDATE", None)
        e.update(env or {})
        return subprocess.run(
            ["bash", str(FETCH), *args],
            cwd=str(cwd), env=e, capture_output=True, text=True, timeout=timeout,
        )

    def assertGuardFired(self, result, needle):
        combined = result.stdout + result.stderr
        self.assertEqual(
            1, result.returncode,
            f"expected exit 1, got {result.returncode}. output:\n{combined[-3000:]}")
        self.assertIn(needle, combined,
                      f"guard fired without saying why. output:\n{combined[-3000:]}")


class TestHarnessIsHonest(GuardHarness):
    """A control run. Without it, every guard test below could be passing because the
    harness is broken rather than because the guard works -- exit 1 is the shape of both."""

    def test_all_inputs_present_completes(self):
        root = self.make_project()
        r = self.run_fetch(root)
        self.assertEqual(0, r.returncode,
                         f"the fully-provisioned control run failed:\n{r.stdout[-2000:]}\n"
                         f"{r.stderr[-2000:]}")
        self.assertIn("WORK=/tmp/review-pr-", r.stdout)
        # And it did so without touching the network or the real GitHub.
        calls = (self.gh_dir / "calls.log").read_text()
        self.assertIn("pr view", calls)
        self.assertIn("pr diff", calls)


class TestGuardsFire(GuardHarness):

    def test_L19_outside_a_git_repo(self):
        """`git rev-parse --show-toplevel` fails, so there is no .claude/skills to resolve."""
        self.make_project()
        outside = self.tmp / "not-a-repo"
        outside.mkdir()
        r = self.run_fetch(outside)
        self.assertGuardFired(r, "review-pr must run inside the repository that owns")

    def test_L89_scanner_missing(self):
        root = self.make_project(scanner="missing")
        r = self.run_fetch(root)
        self.assertGuardFired(r, "required scanner is missing or not executable")
        self.assertIn("scan_index_width.py", r.stderr)

    def test_L89_scanner_present_but_not_executable(self):
        """The guard tests -x, not -e. A scanner that cannot be run is not a scanner."""
        root = self.make_project(scanner="not_executable")
        r = self.run_fetch(root)
        self.assertGuardFired(r, "required scanner is missing or not executable")

    def test_L108_scan_exits_nonzero(self):
        root = self.make_project(scanner="failing")
        r = self.run_fetch(root)
        self.assertGuardFired(r, "required index-width scan failed")
        self.assertIn("do not report an empty candidate list", r.stderr)

    def test_L115_production_scale_missing(self):
        root = self.make_project(scale=False)
        r = self.run_fetch(root)
        self.assertGuardFired(r, "required production-scale evidence is missing")

    def test_L121_report_schema_missing(self):
        root = self.make_project(schema=False)
        r = self.run_fetch(root)
        self.assertGuardFired(r, "required validation report schema is missing")

    def test_L361_validator_not_executable(self):
        """Reached only when the requirement pass says REQUIRED *and* names a target, no
        report was supplied, and auto-validation is on -- so the fixture ships a runtime
        change plus exactly one added op_tests/test_*.py."""
        root = self.make_project(
            diff=RUNTIME_WITH_TARGET_DIFF,
            paths=("csrc/kernels/foo.cu", "op_tests/test_foo.py"),
            validator="not_executable",
        )
        r = self.run_fetch(root)
        self.assertGuardFired(r, "required validator is missing or not executable")
        self.assertIn("validate_pr.sh", r.stderr)

    def test_L361_validator_missing(self):
        root = self.make_project(
            diff=RUNTIME_WITH_TARGET_DIFF,
            paths=("csrc/kernels/foo.cu", "op_tests/test_foo.py"),
            validator="missing",
        )
        r = self.run_fetch(root)
        self.assertGuardFired(r, "required validator is missing or not executable")

    def test_L741_rule_expansion_fails(self):
        """rules.md no longer carries the bodies the deriver names. Shipping the review
        without them is shipping a rule pass over rules the reviewer never sees."""
        root = self.make_project(
            diff=RUNTIME_DIFF, paths=("csrc/kernels/foo.cu",),
            rules_md="drifted", title="optimize foo kernel",
        )
        r = self.run_fetch(root)
        self.assertGuardFired(r, "rule bodies could not be expanded")
        self.assertIn("rules.md has drifted from the deriver", r.stderr)


class TestGuardsDoNotFireSpuriously(GuardHarness):
    """The other half of a fail-closed guard: it must not fail-closed on a healthy input.
    Each case below differs from a firing case above by exactly the guarded condition."""

    def test_disabling_auto_validation_skips_the_validator_guard(self):
        """REVIEW_AUTO_VALIDATE=0 must not turn a missing validator into an abort -- the
        run is opting out of the thing the validator is needed for."""
        root = self.make_project(
            diff=RUNTIME_WITH_TARGET_DIFF,
            paths=("csrc/kernels/foo.cu", "op_tests/test_foo.py"),
            validator="missing",
        )
        r = self.run_fetch(root, env={"REVIEW_AUTO_VALIDATE": "0"})
        self.assertEqual(0, r.returncode,
                         f"opting out of auto-validation aborted the run:\n"
                         f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        self.assertIn("auto-validation is disabled", r.stdout + r.stderr)

    def test_static_only_pr_needs_no_validator(self):
        """No runtime surface -> the requirement pass says NOT REQUIRED, and the L361
        guard is never reached even though the validator is absent."""
        root = self.make_project(validator="missing")
        r = self.run_fetch(root)
        self.assertEqual(0, r.returncode, r.stderr[-2000:])
        self.assertIn("validation not applicable", r.stdout)

    def test_offline_head_fetch_is_a_note_not_an_abort(self):
        """The `git fetch refs/pull/N/head` next to the L89 guard is deliberately NOT a
        guard: an unreachable network degrades the scan, it does not invalidate the run."""
        root = self.make_project()
        r = self.run_fetch(root)
        self.assertEqual(0, r.returncode, r.stderr[-2000:])
        self.assertIn("could not fetch the PR head", r.stderr)


if __name__ == "__main__":
    unittest.main()


class TestDiffMustNotBeEmpty(GuardHarness):
    """`gh pr diff` was unchecked, so a failure wrote an empty file and the run continued.

    An empty diff does not come back looking clean -- the deriver answers `underivable`
    and emits all 52 rules, which is the intended fail-open. It comes back looking like
    52 unanswerable rules instead, with the reason on a stderr line a batch run throws
    away. aiter#4961 is one of the 240 open PRs scanned: 27k lines of diff, refused by the
    API with "could not find pull request diff", which reads as a bad PR number."""

    def test_an_empty_diff_stops_the_run(self):
        root = self.make_project(diff="")
        self.assertGuardFired(self.run_fetch(root), "no diff was fetched")

    def test_a_failing_gh_stops_the_run_and_quotes_it(self):
        root = self.make_project()
        (self.gh_dir / "pr.diff").unlink()
        r = self.run_fetch(root)
        self.assertGuardFired(r, "no diff was fetched")
        self.assertIn("gh:", r.stdout + r.stderr,
                      "gh's own explanation must reach the reader, not just our summary")

    def test_a_diff_over_the_api_cap_says_so_instead_of_not_found(self):
        root = self.make_project()
        (self.gh_dir / "pr_diff_stderr.txt").write_text(
            "could not find pull request diff: HTTP 406: Sorry, the diff exceeded the "
            "maximum number of lines (20000) (https://api.github.com/repos/ROCm/aiter/"
            "pulls/4961)\nPullRequest.diff too_large\n")
        r = self.run_fetch(root)
        self.assertGuardFired(r, "20000-line API cap")
        combined = r.stdout + r.stderr
        self.assertIn("not a missing PR", combined)
        self.assertIn("local checkout", combined,
                      "the reader needs the way forward, not just the refusal")

    def test_a_real_diff_is_not_flagged(self):
        """The control that keeps the three above honest: a present diff must pass."""
        root = self.make_project(diff=RUNTIME_DIFF, paths=("csrc/kernels/foo.cu",))
        self.assertEqual(0, self.run_fetch(root).returncode)


class TestStepOneSurvivesItsOwnHousekeeping(GuardHarness):
    """Six reviews in one wave reported fetch.sh dying before it printed WORK=.

    The stale-worktree cleanup added one commit earlier ended its loop body with an
    `&&` chain. A chain's status is the loop's status, so a worktree NEWER than a day made
    the last iteration return 1, the `while` return 1, and `set -euo pipefail` kill the
    script -- after the diff and most artifacts, before the symbol sweep, the merge-target
    checkout and the `WORK=` line. Any machine that ran a review in the past day hit it,
    which is every machine that uses this. The reviews recovered by hand and said so.

    The test registers a fresh worktree first, because that is the state that triggers it:
    an empty worktree list exits the loop cleanly and proves nothing."""

    def test_a_fresh_worktree_does_not_kill_the_run(self):
        root = self.make_project()
        fresh = self.tmp / "review-pr-fresh"
        subprocess.run([self.real_git, "-C", str(root), "worktree", "add", "--detach",
                        str(fresh), "HEAD"], capture_output=True)
        try:
            r = self.run_fetch(root)
            self.assertEqual(0, r.returncode,
                             f"fetch.sh died with a fresh worktree registered:\n"
                             f"{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
            self.assertIn("WORK=/tmp/review-pr-", r.stdout,
                          "the run ended before announcing its work dir")
        finally:
            subprocess.run([self.real_git, "-C", str(root), "worktree", "remove",
                            "--force", str(fresh)], capture_output=True)

    def test_the_loop_body_is_not_an_and_chain(self):
        """Pinned in source too: the shape is what fails, and it fails only under a state
        a unit test has to go out of its way to create."""
        src = FETCH.read_text()
        i = src.index("worktree list --porcelain")
        block = src[i:i + 700]
        self.assertIn("if [ -d", block,
                      "the cleanup loop is back to an && chain; its status becomes the "
                      "while's, and set -e kills the script on a fresh worktree")


class TestADeletedBaseBranchIsNotFatal(GuardHarness):
    """aiter#4045's base is `dev/randomflow_pr`, deleted after merge. The branch-tip
    lookup 404s, and unguarded `set -e` killed the whole run with one line -- "gh: Branch
    not found (HTTP 404)" -- naming neither the PR nor the ref, after the diff had already
    been fetched successfully. The PR's own metadata carries the base OID."""

    def test_a_404_on_the_base_branch_falls_back_to_the_recorded_oid(self):
        root = self.make_project()
        (self.gh_dir / "base_sha.txt").unlink()          # the stub then errors on `api`
        (self.gh_dir / "pr_meta.json").write_text(
            meta(("README.md",)).replace('"baseRefName": "main"',
                                         '"baseRefName": "dev/gone", '
                                         f'"baseRefOid": "{BASE_SHA}"'))
        r = self.run_fetch(root)
        self.assertEqual(0, r.returncode,
                         f"a deleted base branch should not end the review:\n"
                         f"{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        self.assertIn("has no tip", r.stderr)
        self.assertIn("staleness cannot be judged", r.stderr,
                      "the fallback weakens staleness detection and must say so")

    def test_no_base_at_all_still_fails_closed(self):
        root = self.make_project()
        (self.gh_dir / "base_sha.txt").unlink()
        (self.gh_dir / "pr_meta.json").write_text(
            meta(("README.md",)).replace('"baseRefName": "main"', '"baseRefName": "dev/gone"'))
        r = self.run_fetch(root)
        self.assertEqual(1, r.returncode)
        self.assertIn("no base commit", r.stdout + r.stderr)

