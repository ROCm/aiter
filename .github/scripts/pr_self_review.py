#!/usr/bin/env python3
"""Zero-token PR readiness gate.

This script deliberately judges evidence presence and freshness, not whether the code is
correct. It runs from trusted default-branch code, reads GitHub metadata only, updates one
sticky comment, and writes the ``PR Self-Review`` commit-status context.

Default mode is ``shadow``: findings are reported, while the status remains successful. Set
the repository variable ``AITER_PR_SELF_REVIEW_MODE=enforce`` only after shadow calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

MARKER = "<!-- aiter-pr-self-review -->"
STATUS_CONTEXT = "PR Self-Review"
SECTION_NAMES = (
    "Reviewed revision",
    "Behavior and intent",
    "Validation and unverified scope",
    "PR-specific question",
    "Changed op tests",
)
PLACEHOLDER = re.compile(
    r"^(?:n/?a|none|yes|no|done|tested|pass(?:ed)?|ok|same as above|see above)[.!\s]*$",
    re.IGNORECASE,
)
ARTIFACT_PATTERNS = (
    re.compile(r"(?:^|/)(?:__pycache__|\.pytest_cache|\.mypy_cache)(?:/|$)"),
    re.compile(r"\.(?:pyc|pyo|o|obj|so|a|log|tmp|trace|prof|bak)$", re.IGNORECASE),
    re.compile(r"(?:^|/)\.DS_Store$"),
)
CONFLICT_MARKER = re.compile(r"^(?:\+)(?:<<<<<<<|=======|>>>>>>>)", re.MULTILINE)
DEBUG_LINE = re.compile(
    r"^\+.*(?:\bprint\s*\(|\bpdb\.set_trace\s*\(|\bbreakpoint\s*\(|DEBUG_ONLY|debug_printf)",
    re.MULTILINE,
)

# GitHub check-run names are job display names, not workflow names. Aliases keep the gate
# tolerant of the current wording while still making the expected baseline explicit.
REQUIRED_CHECKS = {
    "Black": {"Black", "Check Code Style with Black"},
    "Ruff": {"Ruff", "Check Code Style with Ruff"},
    "Repository dependency": {"Repository dependency", "Check Repository Dependency"},
    "Aiter Test Gate": {"Aiter Test Gate"},
}
PASS_CONCLUSIONS = {"success", "neutral", "skipped"}


class APIError(RuntimeError):
    pass


class GitHub:
    def __init__(self, repo: str, token: str):
        self.repo = repo
        self.token = token

    def request(
        self,
        path: str,
        method: str = "GET",
        body: Any = None,
        accept: str = "application/vnd.github+json",
    ) -> Any:
        url = (
            path
            if path.startswith("https://")
            else f"https://api.github.com/repos/{self.repo}/{path.lstrip('/')}"
        )
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", accept)
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        req.add_header("User-Agent", "aiter-pr-self-review")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                raw = response.read()
                return json.loads(raw or b"null")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise APIError(f"{method} {url}: HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise APIError(f"{method} {url}: {exc}") from exc

    def graphql(self, query: str, variables: dict[str, Any]) -> Any:
        req = urllib.request.Request(
            "https://api.github.com/graphql",
            data=json.dumps({"query": query, "variables": variables}).encode(),
            method="POST",
        )
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("User-Agent", "aiter-pr-self-review")
        req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read() or b"null")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise APIError(f"GraphQL request failed: {exc}") from exc
        if result.get("errors"):
            raise APIError(f"GraphQL errors: {result['errors'][:2]}")
        return result.get("data") or {}

    def pages(self, path: str) -> list[Any]:
        values: list[Any] = []
        separator = "&" if "?" in path else "?"
        for page in range(1, 11):
            batch = self.request(f"{path}{separator}per_page=100&page={page}")
            if not isinstance(batch, list):
                raise APIError(f"expected list from {path}")
            values.extend(batch)
            if len(batch) < 100:
                return values
        return values


@dataclass
class Result:
    head_sha: str
    mode: str
    question: str
    author_issues: list[str] = field(default_factory=list)
    copilot_issues: list[str] = field(default_factory=list)
    hygiene_issues: list[str] = field(default_factory=list)
    hygiene_advisories: list[str] = field(default_factory=list)
    ci: dict[str, str] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)

    @property
    def ci_pending(self) -> bool:
        return any(value in {"pending", "missing"} for value in self.ci.values())

    @property
    def ci_failed(self) -> bool:
        return any(value == "failed" for value in self.ci.values())

    @property
    def blocking(self) -> list[str]:
        failures = self.author_issues + self.copilot_issues + self.hygiene_issues
        if self.ci_failed:
            failures.append("one or more required CI checks failed")
        return failures

    def state(self) -> str:
        if self.mode != "enforce":
            return "success"
        if self.blocking:
            return "failure"
        if self.ci_pending:
            return "pending"
        return "success"


def strip_comments(text: str) -> str:
    return re.sub(r"<!--[\s\S]*?-->", "", text).strip()


def sections(body: str) -> dict[str, str]:
    found: dict[str, str] = {}
    heading = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
    matches = list(heading.finditer(body or ""))
    for index, match in enumerate(matches):
        name = match.group(1).strip()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        found[name] = strip_comments(body[match.end() : end])
    return found


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_./:+-]+|[\u3400-\u9fff]", text))


def answer_is_empty(text: str) -> bool:
    return not text.strip() or bool(PLACEHOLDER.fullmatch(text.strip()))


def select_question(paths: list[str]) -> str:
    joined = "\n".join(paths).lower()
    if any(path.startswith(".github/") for path in paths):
        return "What permissions and untrusted-input boundary does this CI change have, and can fork PR code or text reach a privileged execution path?"
    if any(
        path.startswith(("csrc/", "hsa/"))
        or path.endswith((".cu", ".cuh", ".hip", ".s"))
        for path in paths
    ):
        return "Which architecture, dtype and shape boundaries select the changed kernel path, and what invariant prevents unsupported inputs from reaching it?"
    if "config" in joined or any(path.endswith((".csv", ".json")) for path in paths):
        return "Which exact workload keys consume these configuration changes, and what evidence shows that unrelated shapes keep their previous selection?"
    if any(
        path.startswith("aiter/jit/") or path in {"setup.py", "pyproject.toml"}
        for path in paths
    ):
        return "What changes in dependency, cache or compilation behavior, and how did you verify both a clean build and an existing-cache path?"
    if any(path.startswith("op_tests/") for path in paths):
        return "What behavior would each changed op test fail on before this PR, and why is that coverage not already provided by an existing canonical test?"
    return "What is the most important boundary or failure mode introduced by this change, and what evidence covers it?"


def validate_author_answers(
    body: str, head_sha: str, changed_tests: list[str]
) -> tuple[list[str], dict[str, Any]]:
    parsed = sections(body)
    issues: list[str] = []
    for name in SECTION_NAMES:
        if name not in parsed:
            issues.append(f"missing `### {name}` in the PR description")

    revision = parsed.get("Reviewed revision", "").strip().strip("`")
    if not revision or not head_sha.startswith(revision) or len(revision) < 7:
        issues.append(
            "`Reviewed revision` must contain at least the first 7 characters of the current head SHA"
        )

    minimums = {
        "Behavior and intent": 18,
        "Validation and unverified scope": 18,
        "PR-specific question": 14,
    }
    for name, minimum in minimums.items():
        answer = parsed.get(name, "")
        if answer_is_empty(answer):
            issues.append(f"`{name}` has no substantive answer")
        elif word_count(answer) < minimum:
            issues.append(
                f"`{name}` is too short to show concrete reasoning ({word_count(answer)} units; need {minimum})"
            )

    validation = parsed.get("Validation and unverified scope", "")
    evidence = bool(
        re.search(
            r"`[^`]+`|```|https?://|\b(?:pytest|python|bash|ctest|benchmark|passed|failed)\b",
            validation,
            re.IGNORECASE,
        )
    )
    limitation = bool(
        re.search(
            r"\b(?:unverified|not tested|not run|not covered|remaining|limitation|none)\b|未(?:验证|测试|覆盖)|没有(?:验证|测试)",
            validation,
            re.IGNORECASE,
        )
    )
    if validation and not evidence:
        issues.append("validation answer must name a command, result, or evidence link")
    if validation and not limitation:
        issues.append(
            "validation answer must explicitly state what remains unverified (or say none with a reason)"
        )

    test_answer = parsed.get("Changed op tests", "")
    if changed_tests:
        missing = [
            path
            for path in changed_tests
            if path not in test_answer and path.rsplit("/", 1)[-1] not in test_answer
        ]
        if missing:
            issues.append(
                "changed op tests not explained in `Changed op tests`: "
                + ", ".join(missing[:8])
            )
        if answer_is_empty(test_answer):
            issues.append(
                "`Changed op tests` must explain why each changed op test is necessary and non-redundant"
            )
    elif test_answer and not re.search(r"\bnone\b|没有|无", test_answer, re.IGNORECASE):
        # Informational only; an author may mention tests outside op_tests/.
        pass

    combined = "\n".join(parsed.get(name, "") for name in SECTION_NAMES[1:])
    telemetry = {
        "answer_units": word_count(combined),
        "code_tokens": len(re.findall(r"`[^`]+`", combined)),
        "paths_named": len(
            re.findall(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+", combined)
        ),
        "fingerprint": hashlib.sha256(
            re.sub(r"\s+", " ", combined.lower()).encode()
        ).hexdigest()[:12],
    }
    if (
        telemetry["answer_units"] >= 120
        and telemetry["code_tokens"] == 0
        and telemetry["paths_named"] == 0
    ):
        telemetry["generic_warning"] = (
            "long answer contains no code token or path; audit for generic/AI-generated prose"
        )
    return issues, telemetry


def inspect_hygiene(
    files: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    hard: list[str] = []
    advisory: list[str] = []
    changed_tests: list[str] = []
    for item in files:
        path = item.get("filename") or ""
        patch = item.get("patch") or ""
        if path.startswith("op_tests/"):
            changed_tests.append(path)
        if any(pattern.search(path) for pattern in ARTIFACT_PATTERNS):
            hard.append(f"generated/debug artifact is committed: `{path}`")
        if CONFLICT_MARKER.search(patch):
            hard.append(f"merge-conflict marker appears in `{path}`")
        if DEBUG_LINE.search(patch):
            advisory.append(
                f"new debug-looking statement in `{path}`; confirm it is intentional"
            )
        if (
            item.get("status") == "added"
            and path.startswith("op_tests/")
            and "/op_benchmarks/" not in f"/{path}"
            and not re.search(
                r"(?:assert|checkAllclose|pytest\.raises|torch\.testing)", patch
            )
        ):
            advisory.append(
                f"new op test `{path}` has no obvious assertion/check in the available patch"
            )
    return hard, advisory, changed_tests


COPILOT_QUERY = """
query($owner:String!, $repo:String!, $number:Int!) {
  repository(owner:$owner, name:$repo) {
    pullRequest(number:$number) {
      headRefOid
      reviews(first:100) {
        nodes {
          author { login }
          body
          submittedAt
          commit { oid }
        }
      }
      reviewThreads(first:100) {
        nodes {
          isResolved
          isOutdated
          comments(first:100) {
            nodes { author { login } body createdAt }
          }
        }
      }
    }
  }
}
"""


def inspect_copilot(api: GitHub, number: int, author: str) -> list[str]:
    owner, repo = api.repo.split("/", 1)
    try:
        data = api.graphql(
            COPILOT_QUERY, {"owner": owner, "repo": repo, "number": number}
        )
        pull = (data.get("repository") or {}).get("pullRequest") or {}
        threads = (pull.get("reviewThreads") or {}).get("nodes") or []
        reviews = (pull.get("reviews") or {}).get("nodes") or []
        head_sha = pull.get("headRefOid") or ""
    except APIError as exc:
        return [f"Copilot review threads could not be read: {exc}"]

    issues: list[str] = []
    current_reviews = []
    for review in reviews:
        login = (((review.get("author") or {}).get("login")) or "").lower()
        commit_sha = ((review.get("commit") or {}).get("oid")) or ""
        if "copilot" in login and commit_sha == head_sha:
            current_reviews.append(review)
    if not current_reviews:
        issues.append("Copilot review has not completed for the current head SHA")
    elif any(
        re.search(
            r"unable to review|insufficient.*quota|remaining quota|could not review",
            review.get("body") or "",
            re.IGNORECASE,
        )
        for review in current_reviews
    ):
        issues.append(
            "Copilot could not review the current head (for example, quota or service failure)"
        )

    for thread in threads:
        comments = (thread.get("comments") or {}).get("nodes") or []
        if not comments:
            continue
        first_login = (((comments[0].get("author") or {}).get("login")) or "").lower()
        if "copilot" not in first_login:
            continue
        if thread.get("isOutdated"):
            continue
        author_replied = any(
            (((comment.get("author") or {}).get("login")) or "").lower()
            == author.lower()
            and not answer_is_empty(comment.get("body") or "")
            for comment in comments[1:]
        )
        if not author_replied:
            issues.append(
                "an active Copilot finding has no substantive reply from the PR author"
            )
        elif not thread.get("isResolved"):
            issues.append("an author-replied Copilot finding is still unresolved")
    return issues[:20]


def inspect_ci(api: GitHub, head_sha: str) -> dict[str, str]:
    if not head_sha:
        return {name: "missing" for name in REQUIRED_CHECKS}
    result = api.request(f"commits/{head_sha}/check-runs?filter=latest&per_page=100")
    runs = result.get("check_runs") if isinstance(result, dict) else []
    states: dict[str, str] = {}
    for label, aliases in REQUIRED_CHECKS.items():
        matches = [run for run in runs or [] if run.get("name") in aliases]
        if not matches:
            states[label] = "missing"
            continue
        run = max(
            matches,
            key=lambda value: value.get("started_at") or value.get("created_at") or "",
        )
        if run.get("status") != "completed":
            states[label] = "pending"
        elif run.get("conclusion") in PASS_CONCLUSIONS:
            states[label] = "passed"
        else:
            states[label] = "failed"
    return states


def evaluate(api: GitHub, number: int, mode: str) -> tuple[Result, dict[str, Any]]:
    pr = api.request(f"pulls/{number}")
    files = api.pages(f"pulls/{number}/files")
    head_sha = ((pr.get("head") or {}).get("sha")) or ""
    paths = [item.get("filename") or "" for item in files]
    hard, advisory, changed_tests = inspect_hygiene(files)
    author = ((pr.get("user") or {}).get("login")) or ""
    author_issues, telemetry = validate_author_answers(
        pr.get("body") or "", head_sha, changed_tests
    )
    result = Result(
        head_sha=head_sha,
        mode=mode,
        question=select_question(paths),
        author_issues=author_issues,
        copilot_issues=inspect_copilot(api, number, author),
        hygiene_issues=hard,
        hygiene_advisories=advisory,
        ci=inspect_ci(api, head_sha),
        telemetry=telemetry,
    )
    return result, pr


def icon(state: str) -> str:
    return {
        "passed": "PASS",
        "pending": "WAIT",
        "missing": "WAIT",
        "failed": "FAIL",
    }.get(state, state.upper())


def render(result: Result, pr: dict[str, Any]) -> str:
    ready = not result.blocking and not result.ci_pending
    conclusion = "READY FOR HUMAN REVIEW" if ready else "NOT READY FOR HUMAN REVIEW"

    lines = [
        f"# {conclusion}",
        "",
        "## PR Self-Review",
        "",
        f"Revision `{result.head_sha[:12]}` · mode `{result.mode}`",
        "",
        "Edit the `PR Self-Review` section in the PR description. The check refreshes automatically.",
        "",
        "### Questions",
        "",
        "1. What behavior does this PR change, and why is the existing behavior insufficient?",
        "2. What did you validate, and what remains unverified? Include commands, hardware/configuration and results.",
        f"3. {result.question}",
        "",
        "### Author answers",
    ]
    if result.author_issues:
        lines.extend(f"- FAIL — {issue}" for issue in result.author_issues)
    else:
        lines.append(
            "- PASS — current-SHA answers are present and structurally specific"
        )

    lines.extend(["", "### Copilot findings"])
    if result.copilot_issues:
        lines.extend(f"- FAIL — {issue}" for issue in result.copilot_issues)
    else:
        lines.append(
            "- PASS — no active Copilot thread lacks an author reply and resolution"
        )

    lines.extend(["", "### Code and op-test hygiene"])
    if result.hygiene_issues:
        lines.extend(f"- FAIL — {issue}" for issue in result.hygiene_issues)
    else:
        lines.append("- PASS — no committed artifact or conflict marker detected")
    lines.extend(f"- ADVISORY — {issue}" for issue in result.hygiene_advisories)

    lines.extend(["", "### Required CI"])
    for name, state in result.ci.items():
        lines.append(f"- {icon(state)} — {name}: `{state}`")
    if result.ci_pending:
        lines.append(
            "- Required CI is incomplete; this PR is not ready for human review."
        )
    if result.ci_failed:
        lines.append("- Required CI failed; fix it before requesting human review.")

    lines.extend(["", "### Shadow audit signals"])
    lines.append(f"- Answer size: `{result.telemetry.get('answer_units', 0)}` units")
    lines.append(
        f"- Code tokens: `{result.telemetry.get('code_tokens', 0)}` · paths named: `{result.telemetry.get('paths_named', 0)}`"
    )
    if result.telemetry.get("generic_warning"):
        lines.append(f"- ADVISORY — {result.telemetry['generic_warning']}")
    lines.extend(
        [
            "",
            (
                "<sub>Shadow mode reports what would block but keeps the status successful. "
                "Set `AITER_PR_SELF_REVIEW_MODE=enforce` only after reviewing real false positives.</sub>"
            ),
            "",
            f"<!-- state={result.state()} head={result.head_sha} answer={result.telemetry.get('fingerprint', '')} -->",
            MARKER,
        ]
    )
    return "\n".join(lines)


def upsert_comment(api: GitHub, number: int, body: str) -> str:
    comments = api.pages(f"issues/{number}/comments")
    existing = next(
        (
            comment
            for comment in comments
            if MARKER in (comment.get("body") or "")
            and ((comment.get("user") or {}).get("login") or "").lower()
            == "github-actions[bot]"
        ),
        None,
    )
    if existing:
        response = api.request(
            f"issues/comments/{existing['id']}", method="PATCH", body={"body": body}
        )
    else:
        response = api.request(
            f"issues/{number}/comments", method="POST", body={"body": body}
        )
    return response.get("html_url") or ""


def post_status(api: GitHub, sha: str, result: Result, url: str) -> None:
    state = result.state()
    if result.mode == "shadow":
        description = f"shadow: {len(result.blocking)} blocker(s), CI {'pending' if result.ci_pending else 'complete'}"
    elif state == "success":
        description = "author answers, Copilot findings and required CI are ready"
    elif state == "pending":
        description = "waiting for required CI"
    else:
        description = f"not ready: {len(result.blocking)} blocker(s)"
    api.request(
        f"statuses/{sha}",
        method="POST",
        body={
            "state": state,
            "context": STATUS_CONTEXT,
            "description": description[:140],
            "target_url": url,
        },
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="update the sticky comment and commit status",
    )
    parser.add_argument("--fixture", help="evaluate a local fixture instead of GitHub")
    args = parser.parse_args(argv)

    if args.fixture:
        with open(args.fixture, encoding="utf-8") as fixture_file:
            data = json.load(fixture_file)
        issues, telemetry = validate_author_answers(
            data["body"], data["head_sha"], data.get("changed_tests", [])
        )
        print(json.dumps({"issues": issues, "telemetry": telemetry}, indent=2))
        return 1 if issues else 0

    repo = os.environ.get("GITHUB_REPOSITORY", "ROCm/aiter")
    token = os.environ.get("GH_TOKEN", "")
    number = int(os.environ["PR_NUMBER"])
    mode = os.environ.get("SELF_REVIEW_MODE", "shadow").strip().lower()
    if mode not in {"shadow", "enforce"}:
        raise SystemExit(f"invalid SELF_REVIEW_MODE={mode!r}")

    api = GitHub(repo, token)
    result, pr = evaluate(api, number, mode)
    body = render(result, pr)
    print(body)
    if args.apply:
        url = upsert_comment(api, number, body)
        # PR checks and branch protection are attached to the PR head commit.
        try:
            post_status(api, result.head_sha, result, url or pr.get("html_url") or "")
        except APIError as exc:
            if mode == "enforce":
                raise
            print(
                f"::warning title=PR Self-Review status::shadow mode could not write status: {exc}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
