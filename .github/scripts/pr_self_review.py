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
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

MARKER = "<!-- aiter-pr-self-review -->"
STATUS_CONTEXT = "PR Self-Review"
MAX_PAGES = 30
MAX_RENDERED_ANSWER_CHARS = 4_000
MAX_RENDERED_FINDINGS = 20
MAX_RENDERED_INLINE_CHARS = 320
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
    r"^\+\s*(?:pdb\.set_trace\s*\(|breakpoint\s*\(|DEBUG_ONLY\b|debug_printf\s*\()",
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
PASS_CONCLUSIONS = {"success"}

# Pure CI or documentation changes never trigger the Aiter Test workflow, so its gate
# check-run structurally cannot exist on such PRs. Keeping it required there makes those
# PRs permanently "not ready" -- a false positive with no CI signal to wait for.
CI_ONLY_PREFIXES = (".github/", "docs/")
CI_ONLY_SUFFIXES = (".md", ".rst")


def required_checks(paths: list[str]) -> dict[str, set[str]]:
    """Drop the Aiter Test Gate requirement when all changed paths are CI-only or
    documentation; everything else keeps the full four-check requirement."""
    if paths and all(
        path.startswith(CI_ONLY_PREFIXES) or path.endswith(CI_ONLY_SUFFIXES)
        for path in paths
    ):
        return {
            label: aliases
            for label, aliases in REQUIRED_CHECKS.items()
            if label != "Aiter Test Gate"
        }
    return REQUIRED_CHECKS


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
    ) -> Any:
        url = (
            path
            if path.startswith("https://")
            else f"https://api.github.com/repos/{self.repo}/{path.lstrip('/')}"
        )
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", "application/vnd.github+json")
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
        for page in range(1, MAX_PAGES + 1):
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
    pr_url: str
    draft: bool
    mode: str
    question: str
    author_issues: list[str] = field(default_factory=list)
    copilot_issues: list[str] = field(default_factory=list)
    copilot_advisories: list[str] = field(default_factory=list)
    hygiene_issues: list[str] = field(default_factory=list)
    hygiene_advisories: list[str] = field(default_factory=list)
    ci: dict[str, str] = field(default_factory=dict)
    ci_unavailable: bool = False
    answers: dict[str, str] = field(default_factory=dict)

    @property
    def ci_pending(self) -> bool:
        return any(
            value in {"pending", "missing", "unavailable"} for value in self.ci.values()
        )

    @property
    def ci_failed(self) -> bool:
        return any(value == "failed" for value in self.ci.values())

    @property
    def blocking(self) -> list[str]:
        failures = self.author_issues + self.copilot_issues + self.hygiene_issues
        if self.draft:
            failures.append("pull request is still a draft")
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


def inline_code(value: str) -> str:
    value = value.replace("\r", r"\r").replace("\n", r"\n")
    if len(value) > MAX_RENDERED_INLINE_CHARS:
        value = value[:MAX_RENDERED_INLINE_CHARS].rstrip() + "…"
    longest = max((len(match) for match in re.findall(r"`+", value)), default=0)
    fence = "`" * max(1, longest + 1)
    return f"{fence} {value} {fence}"


def sections(body: str) -> dict[str, str]:
    found: dict[str, str] = {}
    heading = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)
    matches = list(heading.finditer(body or ""))
    for index, match in enumerate(matches):
        if match.group(1) != "###":
            continue
        name = match.group(2).strip()
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
) -> tuple[list[str], dict[str, str]]:
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
            issues.extend(
                "changed op test not explained in `Changed op tests`: "
                + inline_code(path)
                for path in missing[:8]
            )
            if len(missing) > 8:
                issues.append(
                    f"{len(missing) - 8} additional changed op test(s) are not explained"
                )
        if answer_is_empty(test_answer):
            issues.append(
                "`Changed op tests` must explain why each changed op test is necessary and non-redundant"
            )
    answers = {name: parsed.get(name, "").strip() for name in SECTION_NAMES[1:]}
    return issues, answers


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
            hard.append(f"generated/debug artifact is committed: {inline_code(path)}")
        if CONFLICT_MARKER.search(patch):
            hard.append(f"merge-conflict marker appears in {inline_code(path)}")
        if DEBUG_LINE.search(patch):
            advisory.append(
                f"new debug-looking statement in {inline_code(path)}; confirm it is intentional"
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
                f"new op test {inline_code(path)} has no obvious assertion/check in the available patch"
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
          commit { oid }
        }
      }
      reviewThreads(first:100) {
        nodes {
          isResolved
          isOutdated
          comments(first:100) {
            nodes { author { login } body }
          }
        }
      }
    }
  }
}
"""


def inspect_copilot(
    api: GitHub, number: int, author: str
) -> tuple[list[str], list[str]]:
    owner, repo = api.repo.split("/", 1)
    try:
        data = api.graphql(
            COPILOT_QUERY, {"owner": owner, "repo": repo, "number": number}
        )
        pull = (data.get("repository") or {}).get("pullRequest") or {}
        threads = (pull.get("reviewThreads") or {}).get("nodes") or []
        reviews = (pull.get("reviews") or {}).get("nodes") or []
        head_sha = pull.get("headRefOid") or ""
    except APIError:
        return [], [
            "Copilot review threads could not be read; availability is advisory"
        ]

    issues: list[str] = []
    advisories: list[str] = []
    current_reviews = []
    for review in reviews:
        login = (((review.get("author") or {}).get("login")) or "").lower()
        commit_sha = ((review.get("commit") or {}).get("oid")) or ""
        if "copilot" in login and commit_sha == head_sha:
            current_reviews.append(review)
    if not current_reviews:
        advisories.append("Copilot did not review the current head SHA")
    elif any(
        re.search(
            r"unable to review|insufficient.*quota|remaining quota|could not review",
            review.get("body") or "",
            re.IGNORECASE,
        )
        for review in current_reviews
    ):
        advisories.append(
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
            advisories.append(
                "an author-replied Copilot finding is still unresolved; confirm the disposition during human review"
            )
    return issues[:20], advisories[:20]


def inspect_ci(
    api: GitHub, head_sha: str, required: dict[str, set[str]] | None = None
) -> dict[str, str]:
    if required is None:
        required = REQUIRED_CHECKS
    if not head_sha:
        return {name: "missing" for name in required}
    result = api.request(f"commits/{head_sha}/check-runs?filter=latest&per_page=100")
    runs = result.get("check_runs") if isinstance(result, dict) else []
    states: dict[str, str] = {}
    for label, aliases in required.items():
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


def find_workflow_pr(
    api: GitHub, head_owner: str, head_branch: str, head_sha: str
) -> int | None:
    if not head_owner or not head_branch or not head_sha:
        return None
    query = urllib.parse.urlencode(
        {
            "state": "open",
            "base": "main",
            "head": f"{head_owner}:{head_branch}",
            "per_page": 100,
        }
    )
    pulls = api.request(f"pulls?{query}")
    matches = [
        pull
        for pull in pulls
        if ((pull.get("head") or {}).get("sha") or "") == head_sha
    ]
    if len(matches) != 1:
        return None
    return int(matches[0]["number"])


def evaluate(api: GitHub, number: int, mode: str) -> Result:
    pr = api.request(f"pulls/{number}")
    files = api.pages(f"pulls/{number}/files")
    head_sha = ((pr.get("head") or {}).get("sha")) or ""
    paths = [item.get("filename") or "" for item in files]
    hard, advisory, changed_tests = inspect_hygiene(files)
    author = ((pr.get("user") or {}).get("login")) or ""
    author_issues, answers = validate_author_answers(
        pr.get("body") or "", head_sha, changed_tests
    )
    copilot_issues, copilot_advisories = inspect_copilot(api, number, author)
    ci_unavailable = False
    required = required_checks(paths)
    try:
        ci = inspect_ci(api, head_sha, required)
    except APIError:
        ci = {name: "unavailable" for name in required}
        ci_unavailable = True
    result = Result(
        head_sha=head_sha,
        pr_url=pr.get("html_url") or "",
        draft=bool(pr.get("draft")),
        mode=mode,
        question=select_question(paths),
        author_issues=author_issues,
        copilot_issues=copilot_issues,
        copilot_advisories=copilot_advisories,
        hygiene_issues=hard,
        hygiene_advisories=advisory,
        ci=ci,
        ci_unavailable=ci_unavailable,
        answers=answers,
    )
    return result


def icon(state: str) -> str:
    return {
        "passed": "PASS",
        "pending": "WAIT",
        "missing": "WAIT",
        "unavailable": "WAIT",
        "failed": "FAIL",
    }.get(state, state.upper())


def fenced_text(value: str) -> str:
    if len(value) > MAX_RENDERED_ANSWER_CHARS:
        value = (
            value[:MAX_RENDERED_ANSWER_CHARS].rstrip()
            + "\n\n[truncated in readiness report; see the PR description for the full answer]"
        )
    longest = max((len(match) for match in re.findall(r"`+", value)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{value.strip()}\n{fence}"


def append_findings(lines: list[str], level: str, findings: list[str]) -> None:
    for finding in findings[:MAX_RENDERED_FINDINGS]:
        lines.append(f"- {level} — {finding}")
    remaining = len(findings) - MAX_RENDERED_FINDINGS
    if remaining > 0:
        lines.append(f"- {level} — {remaining} additional finding(s) omitted")


def render(result: Result) -> str:
    ready = not result.blocking and not result.ci_pending
    conclusion = "READY FOR HUMAN REVIEW" if ready else "NOT READY FOR HUMAN REVIEW"

    lines = [
        f"# {conclusion}",
        "",
        "## PR Self-Review",
        "",
        f"Revision `{result.head_sha[:12]}` · mode `{result.mode}`",
    ]
    if result.author_issues:
        lines.extend(
            [
                "",
                "**Waiting for the PR author.** The readiness report will appear after the self-review answers are complete for this revision.",
                "",
                "### Questions to answer in the PR description",
                "",
                "1. What behavior does this PR change, and why is the existing behavior insufficient?",
                "2. What did you validate, and what remains unverified? Include commands, hardware/configuration and results.",
                f"3. {result.question}",
                "4. For every changed `op_tests/` file, what behavior does it cover and why is it not redundant? Write `None` if no op test changed.",
                "",
                "Edit the `PR Self-Review` section in the PR description. Include the current revision SHA; this comment refreshes automatically.",
                "",
                "### Missing or invalid answers",
            ]
        )
        append_findings(lines, "FAIL", result.author_issues)
    else:
        lines.extend(
            [
                "",
                "**Author self-review accepted. Readiness report:**",
                "",
                "### Author answers",
                "",
                "- PASS — current-SHA answers are present and structurally specific",
                "",
                "<details>",
                "<summary>Show the author's self-review answers</summary>",
                "",
            ]
        )
        for name in SECTION_NAMES[1:]:
            lines.extend(
                [
                    f"#### {name}",
                    "",
                    fenced_text(result.answers.get(name, "")),
                    "",
                ]
            )
        lines.append("</details>")
        lines.extend(
            [
                "",
                "### PR state",
                (
                    "- WAIT — pull request is still a draft"
                    if result.draft
                    else "- PASS — pull request is marked ready for review"
                ),
            ]
        )
        lines.extend(["", "### Copilot findings"])
        if result.copilot_issues:
            append_findings(lines, "FAIL", result.copilot_issues)
        else:
            lines.append(
                "- PASS — no active Copilot finding lacks a substantive author reply"
            )
        append_findings(lines, "ADVISORY", result.copilot_advisories)

        lines.extend(["", "### Code and op-test hygiene"])
        if result.hygiene_issues:
            append_findings(lines, "FAIL", result.hygiene_issues)
        else:
            lines.append("- PASS — no committed artifact or conflict marker detected")
        append_findings(lines, "ADVISORY", result.hygiene_advisories)

        lines.extend(["", "### Required CI"])
        for name, state in result.ci.items():
            lines.append(f"- {icon(state)} — {name}: `{state}`")
        if result.ci_unavailable:
            lines.append(
                "- ADVISORY — GitHub check runs could not be read; this PR remains not ready."
            )
        if result.ci_pending:
            lines.append(
                "- Required CI is incomplete; this PR is not ready for human review."
            )
        if result.ci_failed:
            lines.append("- Required CI failed; fix it before requesting human review.")

    lines.extend(
        [
            "",
            (
                "<sub>Shadow mode reports what would block but keeps the status successful. "
                "Set `AITER_PR_SELF_REVIEW_MODE=enforce` only after reviewing real false positives.</sub>"
            ),
            "",
            f"<!-- state={result.state()} head={result.head_sha} -->",
            MARKER,
        ]
    )
    return "\n".join(lines)


def upsert_comment(api: GitHub, number: int, body: str, own_login: str) -> str:
    comments = api.pages(f"issues/{number}/comments")
    existing = next(
        (
            comment
            for comment in comments
            if MARKER in (comment.get("body") or "")
            and ((comment.get("user") or {}).get("login") or "").lower() == own_login
        ),
        None,
    )
    if existing:
        if existing.get("body") == body:
            return existing.get("html_url") or ""
        response = api.request(
            f"issues/comments/{existing['id']}", method="PATCH", body={"body": body}
        )
    else:
        response = api.request(
            f"issues/{number}/comments", method="POST", body={"body": body}
        )
    return response.get("html_url") or ""


def post_status(
    api: GitHub, sha: str, result: Result, url: str, own_login: str
) -> None:
    state = result.state()
    if result.mode == "shadow":
        description = f"shadow: {len(result.blocking)} blocker(s), CI {'pending' if result.ci_pending else 'complete'}"
    elif state == "success":
        description = "author answers, Copilot findings and required CI are ready"
    elif state == "pending":
        description = "waiting for required CI"
    else:
        description = f"not ready: {len(result.blocking)} blocker(s)"
    statuses = api.request(f"commits/{sha}/statuses?per_page=100")
    existing = next(
        (
            status
            for status in statuses
            if status.get("context") == STATUS_CONTEXT
            and ((status.get("creator") or {}).get("login") or "").lower() == own_login
        ),
        None,
    )
    if (
        existing
        and existing.get("state") == state
        and existing.get("description") == description[:140]
        and existing.get("target_url") == url
    ):
        return
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


def writer_token(env: Mapping[str, str]) -> str:
    """Write calls prefer the PAT: the org policy keeps GITHUB_TOKEN read-only,
    so comments and commit statuses must go through AITER_BOT_TOKEN."""
    return env.get("AITER_BOT_TOKEN") or env.get("GH_TOKEN", "")


def resolve_own_login(api: GitHub) -> str:
    """Login of the token owner, used to recognize our own sticky comment and
    commit status. Falls back to github-actions[bot] when the query fails."""
    try:
        viewer = api.graphql("query { viewer { login } }", {})
    except APIError:
        return "github-actions[bot]"
    login = (viewer.get("viewer") or {}).get("login") or ""
    return login.strip().lower() or "github-actions[bot]"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="update the sticky comment and commit status",
    )
    args = parser.parse_args(argv)

    repo = os.environ.get("GITHUB_REPOSITORY", "ROCm/aiter")
    token = writer_token(os.environ)
    mode = os.environ.get("SELF_REVIEW_MODE", "shadow").strip().lower()
    if mode not in {"shadow", "enforce"}:
        raise SystemExit(f"invalid SELF_REVIEW_MODE={mode!r}")

    api = GitHub(repo, token)
    raw_number = os.environ.get("PR_NUMBER", "").strip()
    number = (
        int(raw_number)
        if raw_number
        else find_workflow_pr(
            api,
            os.environ.get("WORKFLOW_HEAD_OWNER", ""),
            os.environ.get("WORKFLOW_HEAD_BRANCH", ""),
            os.environ.get("WORKFLOW_HEAD_SHA", ""),
        )
    )
    if number is None:
        print("No open main-targeting pull request matched this workflow run.")
        return 0
    result = evaluate(api, number, mode)
    body = render(result)
    print(body)
    if args.apply:
        own_login = resolve_own_login(api)
        url = ""
        try:
            url = upsert_comment(api, number, body, own_login)
        except APIError as exc:
            if mode == "enforce":
                raise
            print(
                f"::warning title=PR Self-Review sticky comment::shadow mode could not write comment: {exc}"
            )
        # PR checks and branch protection are attached to the PR head commit.
        try:
            post_status(api, result.head_sha, result, url or result.pr_url, own_login)
        except APIError as exc:
            if mode == "enforce":
                raise
            print(
                f"::warning title=PR Self-Review status::shadow mode could not write status: {exc}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
