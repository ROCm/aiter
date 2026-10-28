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
        return self.responses.get("graphql", {})


class PRSelfReviewSelftest(unittest.TestCase):
    def test_pr_template_preserves_review_context(self) -> None:
        template = (SCRIPT.parent.parent / "pull_request_template.md").read_text()
        expected = (
            "Summary",
            "Motivation",
            "Changes",
            "Performance (if applicable)",
            "Testing",
            "Documentation",
            "Dependencies",
            "Breaking Changes",
            "PR Self-Review",
        )
        for heading in expected:
            self.assertIn(f"## {heading}", template)

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
        gate.post_status(matching, "head", result, "https://example/comment")
        self.assertEqual(len(matching.requests), 1)

        status["creator"] = {"login": "attacker"}
        spoof = RequestAPI({"commits/head/statuses?per_page=100": [status]})
        gate.post_status(spoof, "head", result, "https://example/comment")
        self.assertEqual(spoof.requests[-1][1], "POST")

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
        self.assertEqual(gate.upsert_comment(spoof, 1, body), "https://example/new")
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
            gate.upsert_comment(existing, 1, body), "https://example/existing"
        )
        self.assertEqual(existing.requests, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
