"""Deterministic regression checks for the PR Self-Review gate."""

from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).with_name("pr_self_review.py")
WORKFLOWS = SCRIPT.parent.parent / "workflows"
SPEC = importlib.util.spec_from_file_location("pr_self_review", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def valid_body(sha: str = "abcdef1234567890", changed_tests: str = "None.") -> str:
    return f"""## PR Self-Review
### Reviewed revision
{sha}
### Behavior and intent
This changes a specific review-readiness boundary and explains clearly why the previous behavior was insufficient for reliable human review intake.
### Validation and unverified scope
Ran `python selftest.py`; every deterministic scenario passed. No GPU behavior remains unverified because this workflow reads GitHub metadata only.
### PR-specific question
Only trusted default-branch code executes, while pull request text and patches remain untrusted API data and never become executable input.
### Changed op tests
{changed_tests}
## Test plan
This text must stay outside the self-review answer.
"""


class RequestAPI:
    repo = "ROCm/aiter"

    def __init__(self, responses: dict[str, Any]):
        self.responses = responses
        self.requests: list[tuple[str, str, Any]] = []

    def request(self, path: str, method: str = "GET", body: Any = None) -> Any:
        self.requests.append((path, method, body))
        response = self.responses.get(path)
        if isinstance(response, Exception):
            raise response
        return response

    def pages(self, path: str) -> list[Any]:
        return self.responses.get(path, [])

    def graphql(self, query: str, variables: dict[str, Any]) -> Any:
        response = self.responses.get("graphql", {})
        if isinstance(response, Exception):
            raise response
        return response


class PRSelfReviewSelftest(unittest.TestCase):
    def test_review_events_use_read_only_signal_workflow(self) -> None:
        privileged = (WORKFLOWS / "pr-self-review.yml").read_text()
        signal = (WORKFLOWS / "pr-self-review-signal.yml").read_text()

        self.assertNotRegex(privileged, r"(?m)^  pull_request_review(?:_comment)?:")
        self.assertIn("PR Self-Review Signal", privileged)
        self.assertIn("ref: ${{ github.event.repository.default_branch }}", privileged)
        self.assertRegex(signal, r"(?m)^  pull_request_review:")
        self.assertRegex(signal, r"(?m)^  pull_request_review_comment:")
        self.assertRegex(signal, r"(?ms)^permissions:\s+contents: read\s*$")
        self.assertNotIn("actions/checkout", signal)
        self.assertNotRegex(
            signal, r"(?m)^\s+(?:issues|pull-requests|statuses): write$"
        )

    def test_workflow_gates_drafts_and_passes_event_context(self) -> None:
        privileged = (WORKFLOWS / "pr-self-review.yml").read_text()
        # Draft pull requests stay silent except for the one converted_to_draft
        # run that grays out a stale verdict.
        self.assertIn("github.event.action == 'converted_to_draft'", privileged)
        self.assertIn("github.event.pull_request.draft == false", privileged)
        # The gray-out branch needs the pull-request action to detect the
        # conversion.
        self.assertIn("PR_EVENT_ACTION: ${{ github.event.action }}", privileged)

    def test_sections_stop_at_next_parent_heading(self) -> None:
        issues, answers = gate.validate_author_answers(
            valid_body(), "abcdef1234567890", []
        )
        self.assertEqual(issues, [])
        self.assertEqual(answers["Changed op tests"], "None.")
        self.assertNotIn("Test plan", answers["Changed op tests"])

    def test_untrusted_paths_use_dynamic_code_spans(self) -> None:
        path = "op_tests/[proof](https://evil.example)/`case`.py\nnext"
        rendered = gate.inline_code(path)
        self.assertTrue(rendered.startswith("`` "))
        self.assertTrue(rendered.endswith(" ``"))
        self.assertIn(r"\n", rendered)

        issues, _ = gate.validate_author_answers(
            valid_body(), "abcdef1234567890", [path]
        )
        self.assertIn(rendered, "\n".join(issues))

    def test_ci_requires_success_and_degrades_when_unavailable(self) -> None:
        checks = RequestAPI(
            {
                "commits/head/check-runs?filter=latest&per_page=100": {
                    "check_runs": [
                        {
                            "name": "Black",
                            "status": "completed",
                            "conclusion": "skipped",
                        },
                        {
                            "name": "Ruff",
                            "status": "completed",
                            "conclusion": "success",
                        },
                    ]
                }
            }
        )
        states = gate.inspect_ci(checks, "head")
        self.assertEqual(states["Black"], "failed")
        self.assertEqual(states["Ruff"], "passed")

        unavailable = RequestAPI(
            {
                "pulls/7": {
                    "head": {"sha": "head"},
                    "html_url": "https://example/pr/7",
                    "draft": False,
                    "user": {"login": "alice"},
                    "body": valid_body("head000"),
                },
                "pulls/7/files": [],
                "graphql": {
                    "repository": {
                        "pullRequest": {
                            "headRefOid": "head",
                            "reviews": {"nodes": []},
                            "reviewThreads": {"nodes": []},
                        }
                    }
                },
                "commits/head/check-runs?filter=latest&per_page=100": gate.APIError(
                    "checks unavailable"
                ),
            }
        )
        result = gate.evaluate(unavailable, 7, "shadow")
        self.assertTrue(result.ci_unavailable)
        self.assertTrue(result.ci_pending)
        self.assertEqual(set(result.ci.values()), {"unavailable"})

    def test_gpu_gate_not_required_for_ci_only_changes(self) -> None:
        ci_only = gate.required_checks(
            [".github/workflows/x.yml", "docs/gate.md", "CONTRIBUTE.md"]
        )
        self.assertNotIn("Aiter Test Gate", ci_only)
        self.assertEqual(set(ci_only), {"Black", "Ruff", "Repository dependency"})

        mixed = gate.required_checks([".github/workflows/x.yml", "aiter/ops/x.py"])
        self.assertIn("Aiter Test Gate", mixed)

        self.assertEqual(set(gate.required_checks([])), set(gate.REQUIRED_CHECKS))

    def test_ci_only_pr_becomes_ready_without_gpu_gate(self) -> None:
        api = RequestAPI(
            {
                "pulls/9": {
                    "head": {"sha": "head0000000"},
                    "html_url": "https://example/pr/9",
                    "draft": False,
                    "user": {"login": "alice"},
                    "body": valid_body("head0000000"),
                },
                "pulls/9/files": [
                    {"filename": ".github/workflows/x.yml", "status": "modified"},
                    {"filename": "docs/gate.md", "status": "modified"},
                ],
                "graphql": {
                    "repository": {
                        "pullRequest": {
                            "headRefOid": "head0000000",
                            "reviews": {"nodes": []},
                            "reviewThreads": {"nodes": []},
                        }
                    }
                },
                "commits/head0000000/check-runs?filter=latest&per_page=100": {
                    "check_runs": [
                        {
                            "name": "Black",
                            "status": "completed",
                            "conclusion": "success",
                        },
                        {
                            "name": "Ruff",
                            "status": "completed",
                            "conclusion": "success",
                        },
                        {
                            "name": "Check Repository Dependency",
                            "status": "completed",
                            "conclusion": "success",
                        },
                    ]
                },
            }
        )
        result = gate.evaluate(api, 9, "shadow")
        self.assertEqual(set(result.ci), {"Black", "Ruff", "Repository dependency"})
        self.assertFalse(result.ci_pending)
        self.assertTrue(gate.render(result).startswith("# READY FOR HUMAN REVIEW"))

    def test_gpu_pr_still_waits_for_the_aiter_test_gate(self) -> None:
        api = RequestAPI(
            {
                "pulls/10": {
                    "head": {"sha": "head0000000"},
                    "html_url": "https://example/pr/10",
                    "draft": False,
                    "user": {"login": "alice"},
                    "body": valid_body("head0000000"),
                },
                "pulls/10/files": [
                    {"filename": "aiter/ops/matmul.py", "status": "modified"}
                ],
                "graphql": {
                    "repository": {
                        "pullRequest": {
                            "headRefOid": "head0000000",
                            "reviews": {"nodes": []},
                            "reviewThreads": {"nodes": []},
                        }
                    }
                },
                "commits/head0000000/check-runs?filter=latest&per_page=100": {
                    "check_runs": [
                        {
                            "name": "Black",
                            "status": "completed",
                            "conclusion": "success",
                        },
                        {
                            "name": "Ruff",
                            "status": "completed",
                            "conclusion": "success",
                        },
                        {
                            "name": "Check Repository Dependency",
                            "status": "completed",
                            "conclusion": "success",
                        },
                    ]
                },
            }
        )
        result = gate.evaluate(api, 10, "shadow")
        self.assertEqual(result.ci["Aiter Test Gate"], "missing")
        self.assertTrue(result.ci_pending)
        self.assertTrue(gate.render(result).startswith("# CI IN PROGRESS"))
        # No event may turn incomplete evidence into a verdict; only a manual
        # refresh writes the neutral progress report.
        self.assertFalse(gate.should_write("pull_request_target", result))
        self.assertFalse(gate.should_write("workflow_run", result))
        self.assertTrue(gate.should_write("workflow_dispatch", result))

    def test_titles_separate_author_input_ci_progress_and_verdict(self) -> None:
        common = {
            "head_sha": "a" * 40,
            "pr_url": "https://example/pr/1",
            "draft": False,
            "mode": "enforce",
            "question": "Why?",
        }
        passed = {name: "passed" for name in gate.REQUIRED_CHECKS}

        waiting = gate.Result(
            **common, author_issues=["missing `### Reviewed revision`"], ci=passed
        )
        self.assertTrue(gate.render(waiting).startswith("# WAITING ON AUTHOR"))

        in_progress = gate.Result(**common, ci={**passed, "Aiter Test Gate": "missing"})
        self.assertTrue(gate.render(in_progress).startswith("# CI IN PROGRESS"))

        ready = gate.Result(**common, ci=passed)
        self.assertTrue(gate.render(ready).startswith("# READY FOR HUMAN REVIEW"))

        failed = gate.Result(**common, ci={**passed, "Ruff": "failed"})
        self.assertTrue(gate.render(failed).startswith("# NOT READY FOR HUMAN REVIEW"))

    def test_drafts_stay_silent_except_the_conversion_gray_out(self) -> None:
        note = gate.render_draft_note("b" * 40, "shadow")
        self.assertIn("# DRAFT — evaluation paused", note)
        self.assertIn("no verdict applies", note)
        self.assertIn(gate.MARKER, note)

        common = {
            "head_sha": "a" * 40,
            "pr_url": "https://example/pr/1",
            "mode": "shadow",
            "question": "Why?",
            "ci": {name: "passed" for name in gate.REQUIRED_CHECKS},
        }
        draft = gate.Result(**common, draft=True)
        for event in ("pull_request_target", "workflow_run", "workflow_dispatch"):
            self.assertFalse(gate.should_write(event, draft))

    def test_author_facing_findings_post_before_ci_completes(self) -> None:
        common = {
            "head_sha": "a" * 40,
            "pr_url": "https://example/pr/1",
            "draft": False,
            "mode": "shadow",
            "question": "Why?",
            "ci": {
                "Black": "passed",
                "Ruff": "passed",
                "Repository dependency": "passed",
                "Aiter Test Gate": "missing",
            },
        }
        missing_answers = gate.Result(
            **common, author_issues=["`Behavior and intent` has no substantive answer"]
        )
        self.assertTrue(gate.render(missing_answers).startswith("# WAITING ON AUTHOR"))
        self.assertTrue(gate.should_write("pull_request_target", missing_answers))

        dirty_diff = gate.Result(
            **common,
            hygiene_issues=["generated/debug artifact is committed: `x.pyc`"],
        )
        self.assertTrue(gate.should_write("pull_request_target", dirty_diff))

        # Once the evidence is complete, any event may carry the verdict: an
        # author who fixes their answers after CI finished must not wait for a
        # workflow completion that will never come.
        complete = gate.Result(
            **{**common, "ci": {name: "passed" for name in gate.REQUIRED_CHECKS}}
        )
        self.assertTrue(gate.should_write("pull_request_target", complete))
        self.assertTrue(gate.should_write("workflow_run", complete))

    def test_fork_workflow_run_resolves_by_head_identity(self) -> None:
        api = RequestAPI({})
        api.responses[
            "pulls?state=open&base=main&head=fork-owner%3Afeature%2Fbranch&per_page=100"
        ] = [
            {"number": 41, "head": {"sha": "other"}},
            {"number": 42, "head": {"sha": "fork-sha"}},
        ]
        self.assertEqual(
            gate.find_workflow_pr(api, "fork-owner", "feature/branch", "fork-sha"),
            42,
        )

    def test_copilot_requires_a_substantive_author_reply(self) -> None:
        def inspect(replies: list[dict[str, Any]], outdated: bool = False):
            api = RequestAPI(
                {
                    "graphql": {
                        "repository": {
                            "pullRequest": {
                                "headRefOid": "head",
                                "reviews": {
                                    "nodes": [
                                        {
                                            "author": {
                                                "login": "copilot-reviewer[bot]"
                                            },
                                            "body": "reviewed",
                                            "commit": {"oid": "head"},
                                        }
                                    ]
                                },
                                "reviewThreads": {
                                    "nodes": [
                                        {
                                            "isResolved": False,
                                            "isOutdated": outdated,
                                            "comments": {
                                                "nodes": [
                                                    {
                                                        "author": {
                                                            "login": "copilot-reviewer[bot]"
                                                        },
                                                        "body": "finding",
                                                    },
                                                    *replies,
                                                ]
                                            },
                                        }
                                    ]
                                },
                            }
                        }
                    }
                }
            )
            return gate.inspect_copilot(api, 7, "alice")

        issues, _ = inspect([])
        self.assertEqual(len(issues), 1)
        issues, _ = inspect([{"author": {"login": "alice"}, "body": "done"}])
        self.assertEqual(len(issues), 1)
        issues, advisories = inspect(
            [
                {
                    "author": {"login": "alice"},
                    "body": "Fixed the bounds check and added the missing validation.",
                }
            ]
        )
        self.assertEqual(issues, [])
        self.assertEqual(len(advisories), 1)
        issues, _ = inspect([], outdated=True)
        self.assertEqual(issues, [])

    def test_pagination_covers_githubs_pr_file_limit(self) -> None:
        class PagedAPI(gate.GitHub):
            def __init__(self) -> None:
                super().__init__("ROCm/aiter", "")
                self.calls = 0

            def request(self, path: str, method: str = "GET", body: Any = None) -> Any:
                self.calls += 1
                page = int(re.search(r"[?&]page=(\d+)", path).group(1))
                return [page] * 100 if page <= 30 else []

        api = PagedAPI()
        self.assertEqual(len(api.pages("pulls/1/files")), 3_000)
        self.assertEqual(api.calls, 30)

    def test_draft_and_ci_state_control_readiness(self) -> None:
        common = {
            "head_sha": "a" * 40,
            "pr_url": "https://example/pr/1",
            "mode": "enforce",
            "question": "Why?",
            "answers": {name: "answer" for name in gate.SECTION_NAMES[1:]},
        }
        passed = {name: "passed" for name in gate.REQUIRED_CHECKS}
        draft = gate.Result(**common, draft=True, ci=passed)
        self.assertEqual(draft.state(), "failure")
        self.assertTrue(gate.render(draft).startswith("# NOT READY"))

        ready = gate.Result(**common, draft=False, ci=passed)
        self.assertEqual(ready.state(), "success")
        self.assertTrue(gate.render(ready).startswith("# READY"))

    def test_rendering_is_bounded(self) -> None:
        result = gate.Result(
            head_sha="a" * 40,
            pr_url="https://example/pr/1",
            draft=False,
            mode="shadow",
            question="Why?",
            copilot_issues=[f"finding {index}" for index in range(1_000)],
            hygiene_issues=[f"hygiene {index}" for index in range(1_000)],
            ci={name: "passed" for name in gate.REQUIRED_CHECKS},
            answers={name: "x" * 20_000 for name in gate.SECTION_NAMES[1:]},
        )
        body = gate.render(result)
        self.assertLess(len(body), 65_536)
        self.assertIn("additional finding(s) omitted", body)
        self.assertIn("truncated in readiness report", body)

    def test_status_write_is_idempotent_and_bot_owned(self) -> None:
        result = gate.Result(
            head_sha="head",
            pr_url="https://example/pr/1",
            draft=False,
            mode="shadow",
            question="Why?",
            ci={name: "pending" for name in gate.REQUIRED_CHECKS},
        )
        status = {
            "context": gate.STATUS_CONTEXT,
            "state": "success",
            "description": "shadow: 0 blocker(s), CI pending",
            "target_url": "https://example/comment",
            "creator": {"login": "github-actions[bot]"},
        }
        matching = RequestAPI({"commits/head/statuses?per_page=100": [status]})
        gate.post_status(
            matching, "head", result, "https://example/comment", "github-actions[bot]"
        )
        self.assertEqual(len(matching.requests), 1)

        status["creator"] = {"login": "attacker"}
        spoof = RequestAPI({"commits/head/statuses?per_page=100": [status]})
        gate.post_status(
            spoof, "head", result, "https://example/comment", "github-actions[bot]"
        )
        self.assertEqual(spoof.requests[-1][1], "POST")

    def test_status_write_recognizes_pat_owned_status(self) -> None:
        result = gate.Result(
            head_sha="head",
            pr_url="https://example/pr/1",
            draft=False,
            mode="shadow",
            question="Why?",
            ci={name: "pending" for name in gate.REQUIRED_CHECKS},
        )
        status = {
            "context": gate.STATUS_CONTEXT,
            "state": "success",
            "description": "shadow: 0 blocker(s), CI pending",
            "target_url": "https://example/comment",
            "creator": {"login": "Aiter-Bot"},
        }
        api = RequestAPI({"commits/head/statuses?per_page=100": [status]})
        gate.post_status(api, "head", result, "https://example/comment", "aiter-bot")
        self.assertEqual(len(api.requests), 1)

    def test_sticky_comment_rejects_spoof_and_skips_noop(self) -> None:
        body = "report " + gate.MARKER
        spoof = RequestAPI(
            {
                "issues/1/comments": [
                    {"id": 9, "body": body, "user": {"login": "attacker"}}
                ],
                "issues/1/comments#post": {"html_url": "https://example/new"},
            }
        )

        def spoof_request(path: str, method: str = "GET", body: Any = None) -> Any:
            spoof.requests.append((path, method, body))
            if method == "POST":
                return {"html_url": "https://example/new"}
            return spoof.responses.get(path)

        spoof.request = spoof_request
        self.assertEqual(
            gate.upsert_comment(spoof, 1, body, "github-actions[bot]"),
            "https://example/new",
        )
        self.assertEqual(spoof.requests[-1][1], "POST")

        existing = RequestAPI(
            {
                "issues/1/comments": [
                    {
                        "id": 10,
                        "body": body,
                        "html_url": "https://example/existing",
                        "user": {"login": "github-actions[bot]"},
                    }
                ]
            }
        )
        self.assertEqual(
            gate.upsert_comment(existing, 1, body, "github-actions[bot]"),
            "https://example/existing",
        )
        self.assertEqual(existing.requests, [])

    def test_pat_owned_comment_is_updated_not_reposted(self) -> None:
        body = "report v2 " + gate.MARKER
        api = RequestAPI(
            {
                "issues/1/comments": [
                    {
                        "id": 11,
                        "body": "report v1 " + gate.MARKER,
                        "html_url": "https://example/old",
                        "user": {"login": "aiter-bot"},
                    }
                ],
                "issues/comments/11": {"html_url": "https://example/updated"},
            }
        )
        self.assertEqual(
            gate.upsert_comment(api, 1, body, "aiter-bot"),
            "https://example/updated",
        )
        self.assertEqual(api.requests[-1][1], "PATCH")

    def test_resolve_own_login_falls_back_when_viewer_fails(self) -> None:
        viewer = RequestAPI({"graphql": {"viewer": {"login": "Aiter-Bot"}}})
        self.assertEqual(gate.resolve_own_login(viewer), "aiter-bot")

        failing = RequestAPI({"graphql": gate.APIError("viewer query failed")})
        self.assertEqual(gate.resolve_own_login(failing), "github-actions[bot]")

    def test_writer_token_prefers_pat_but_falls_back_to_github_token(self) -> None:
        self.assertEqual(
            gate.writer_token({"AITER_BOT_TOKEN": "pat", "GH_TOKEN": "gh"}), "pat"
        )
        self.assertEqual(
            gate.writer_token({"AITER_BOT_TOKEN": "", "GH_TOKEN": "gh"}), "gh"
        )
        self.assertEqual(gate.writer_token({"GH_TOKEN": "gh"}), "gh")
        self.assertEqual(gate.writer_token({}), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
