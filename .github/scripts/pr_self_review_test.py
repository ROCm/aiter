#!/usr/bin/env python3
"""Offline tests for pr_self_review.py."""

import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "pr_self_review", HERE / "pr_self_review.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


HEAD = "abcdef1234567890abcdef1234567890abcdef12"


def body(
    revision="abcdef1",
    behavior=None,
    validation=None,
    question=None,
    tests="None — this PR changes no op test.",
):
    behavior = behavior or (
        "This changes the gfx950 dispatch boundary so unsupported shapes keep the existing path "
        "instead of selecting the new kernel implementation."
    )
    validation = validation or (
        "Ran `python op_tests/test_example.py --dtype bf16` on gfx950; all cases passed. "
        "gfx942 and fp16 remain unverified because this path is gfx950 BF16 only."
    )
    question = question or (
        "The architecture and dtype guards run before shape dispatch, and the final shape table "
        "contains only the supported dimensions used by the new kernel."
    )
    return f"""## PR Self-Review

### Reviewed revision
`{revision}`

### Behavior and intent
{behavior}

### Validation and unverified scope
{validation}

### PR-specific question
{question}

### Changed op tests
{tests}
"""


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"PASS: {name}")


def main():
    issues, telemetry = MODULE.validate_author_answers(body(), HEAD, [])
    check("complete answer passes", not issues)
    check("telemetry has fingerprint", len(telemetry["fingerprint"]) == 12)

    issues, _ = MODULE.validate_author_answers(body(revision="1234567"), HEAD, [])
    check("stale revision fails", any("current head SHA" in issue for issue in issues))

    issues, _ = MODULE.validate_author_answers(body(behavior="done"), HEAD, [])
    check(
        "placeholder answer fails",
        any("Behavior and intent" in issue for issue in issues),
    )

    issues, _ = MODULE.validate_author_answers(
        body(
            validation=(
                "The implementation appears correct across the scenarios considered during development, "
                "and the observed behavior matches the intended design in every case examined so far."
            )
        ),
        HEAD,
        [],
    )
    check(
        "validation needs evidence", any("command, result" in issue for issue in issues)
    )
    check(
        "validation needs unverified scope",
        any("unverified" in issue for issue in issues),
    )

    changed = ["op_tests/test_example.py"]
    issues, _ = MODULE.validate_author_answers(body(), HEAD, changed)
    check(
        "changed op test must be named",
        any("not explained" in issue for issue in issues),
    )
    issues, _ = MODULE.validate_author_answers(
        body(
            tests="`op_tests/test_example.py` covers the new dispatch and fails before this PR."
        ),
        HEAD,
        changed,
    )
    check("named changed op test passes", not issues)

    hard, advisory, tests = MODULE.inspect_hygiene(
        [
            {"filename": "results/debug.log", "status": "added", "patch": "+data"},
            {
                "filename": "op_tests/test_x.py",
                "status": "added",
                "patch": "+print('debug')",
            },
        ]
    )
    check("artifact is hard failure", any("artifact" in issue for issue in hard))
    check(
        "debug print is advisory", any("debug-looking" in issue for issue in advisory)
    )
    check("op test is collected", tests == ["op_tests/test_x.py"])

    check(
        "CI workflow question selected",
        "permissions" in MODULE.select_question([".github/workflows/x.yml"]),
    )
    check(
        "kernel question selected",
        "architecture" in MODULE.select_question(["csrc/x.cu"]),
    )
    check(
        "config question selected",
        "workload keys" in MODULE.select_question(["aiter/configs/x.csv"]),
    )

    class FakeAPI:
        repo = "ROCm/aiter"

        def graphql(self, query, variables):
            del query, variables
            return {
                "repository": {
                    "pullRequest": {
                        "headRefOid": HEAD,
                        "reviews": {
                            "nodes": [
                                {
                                    "author": {"login": "Copilot"},
                                    "body": "Review completed.",
                                    "commit": {"oid": HEAD},
                                }
                            ]
                        },
                        "reviewThreads": {
                            "nodes": [
                                {
                                    "isResolved": True,
                                    "isOutdated": False,
                                    "comments": {
                                        "nodes": [
                                            {
                                                "author": {"login": "Copilot"},
                                                "body": "Finding",
                                            },
                                            {
                                                "author": {"login": "alice"},
                                                "body": "Fixed in the current revision.",
                                            },
                                        ]
                                    },
                                }
                            ]
                        },
                    }
                }
            }

    check(
        "current Copilot review with resolved author reply passes",
        not MODULE.inspect_copilot(FakeAPI(), 7, "alice"),
    )

    class MissingReviewAPI(FakeAPI):
        def graphql(self, query, variables):
            data = super().graphql(query, variables)
            data["repository"]["pullRequest"]["reviews"]["nodes"] = []
            return data

    check(
        "missing current Copilot review is reported",
        any(
            "current head" in issue
            for issue in MODULE.inspect_copilot(MissingReviewAPI(), 7, "alice")
        ),
    )

    class CheckAPI:
        def __init__(self):
            self.path = None

        def request(self, path):
            self.path = path
            return {
                "check_runs": [
                    {
                        "name": "Check Code Style with Black",
                        "status": "completed",
                        "conclusion": "success",
                        "started_at": "2026-10-27T01:00:00Z",
                    },
                    {
                        "name": "Check Code Style with Ruff",
                        "status": "in_progress",
                        "conclusion": None,
                        "started_at": "2026-10-27T01:00:00Z",
                    },
                    {
                        "name": "Check Repository Dependency",
                        "status": "completed",
                        "conclusion": "failure",
                        "started_at": "2026-10-27T01:00:00Z",
                    },
                    {
                        "name": "Aiter Test Gate",
                        "status": "completed",
                        "conclusion": "success",
                        "started_at": "2026-10-27T01:00:00Z",
                    },
                ]
            }

    check_api = CheckAPI()
    ci = MODULE.inspect_ci(check_api, HEAD)
    check(
        "CI aggregator reads the PR head SHA",
        check_api.path == f"commits/{HEAD}/check-runs?filter=latest&per_page=100",
    )
    check("CI aggregator recognizes pass", ci["Black"] == "passed")
    check("CI aggregator recognizes pending", ci["Ruff"] == "pending")
    check("CI aggregator recognizes failure", ci["Repository dependency"] == "failed")
    check("CI aggregator recognizes Aiter gate", ci["Aiter Test Gate"] == "passed")

    class CommentAPI:
        def __init__(self):
            self.path = None

        def pages(self, path):
            del path
            return [
                {
                    "id": 1,
                    "body": MODULE.MARKER,
                    "user": {"login": "malicious-author"},
                },
                {
                    "id": 2,
                    "body": MODULE.MARKER,
                    "user": {"login": "github-actions[bot]"},
                },
            ]

        def request(self, path, method="GET", body=None):
            del method, body
            self.path = path
            return {"html_url": "https://example.test/comment"}

    comment_api = CommentAPI()
    MODULE.upsert_comment(comment_api, 7, "updated")
    check("spoofed sticky marker is ignored", comment_api.path == "issues/comments/2")

    shadow = MODULE.Result(head_sha=HEAD, mode="shadow", question="q")
    shadow.author_issues.append("missing answer")
    check("shadow stays successful", shadow.state() == "success")
    enforce = MODULE.Result(head_sha=HEAD, mode="enforce", question="q")
    enforce.ci = {"Black": "pending"}
    check("enforce waits for CI", enforce.state() == "pending")
    enforce.author_issues.append("missing answer")
    check("author blocker fails", enforce.state() == "failure")

    rendered = MODULE.render(enforce, {})
    check("comment reports required CI", "### Required CI" in rendered)
    check("comment says not ready", "Not ready for human review" in rendered)

    print("All PR Self-Review tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
