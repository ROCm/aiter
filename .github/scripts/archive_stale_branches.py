#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Archive stale branches, then delete the archives after notice.

30 days without commits: copy to ``archive/<YYYY-MM-DD>/<name>``, comment, drop
the original. +30 days: comment giving notice. +14: delete the archive. Restore
at any stage with ``git push origin archive/<date>/<name>:<name>``.

No stored state: the archive date is in the ref name, the notice date is the
notice comment's own. Deleting the bot's comment opts a branch out.

Exempt: protected branches, open-PR head and base branches, and
.github/stale-branch-exemptions.txt, checked against the original name at every
stage. Nothing is written without --apply.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.github.com"
ARCHIVE_PREFIX = "archive/"
BOT_LOGIN = "github-actions[bot]"
_ARCHIVED = re.compile(r"^archive/(\d{4}-\d{2}-\d{2})/(.+)$")


def marker(kind: str, ref: str) -> str:
    """Marker naming its archive ref; two branches can share one commit."""
    return f"<!-- stale-branch-{kind}: {ref} -->"


def archive_date(name: str) -> dt.date | None:
    """Date in an archive ref's name, or None: archive/2026-02-30/x is not a date."""
    found = _ARCHIVED.match(name)
    if not found:
        return None
    try:
        return dt.date.fromisoformat(found.group(1))
    except ValueError:
        return None


class Api:
    """The few REST and GraphQL calls this needs, without a dependency."""

    def __init__(self, token: str, repo: str, apply: bool) -> None:
        self.token = token
        self.owner, self.name = repo.split("/", 1)
        self.apply = apply
        self.writes = 0

    def _call(self, method: str, url: str, body: dict | None = None) -> object:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req) as response:
            raw = response.read()
        return json.loads(raw) if raw else None

    def get(self, path: str) -> object:
        return self._call("GET", f"{API}/repos/{self.owner}/{self.name}{path}")

    def write(self, method: str, path: str, body: dict | None = None) -> object:
        """A call that changes the repository. A dry run stops here."""
        self.writes += 1
        if not self.apply:
            return None
        return self._call(method, f"{API}/repos/{self.owner}/{self.name}{path}", body)

    def delete_branch(self, name: str, expect: str) -> bool:
        """Compare-and-swap delete via --force-with-lease; REST has no conditional delete."""
        self.writes += 1
        if not self.apply:
            return True
        ref = f"refs/heads/{name}"
        done = subprocess.run(
            ["git", "push", f"--force-with-lease={ref}:{expect}", "origin", f":{ref}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode == 0:
            return True
        if "stale info" in done.stderr or "[rejected]" in done.stderr:
            return False
        raise RuntimeError(f"git push failed for {name}: {done.stderr.strip()}")

    def graphql(self, query: str, variables: dict) -> dict:
        req = urllib.request.Request(
            f"{API}/graphql",
            data=json.dumps({"query": query, "variables": variables}).encode(),
            method="POST",
        )
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req) as response:
            payload = json.loads(response.read())
        if "errors" in payload:
            raise RuntimeError(f"GraphQL: {payload['errors']}")
        return payload["data"]


def load_exemptions(path: str) -> tuple[set[str], list[re.Pattern]]:
    """Names and ``re:`` patterns from the list; missing or empty is fatal."""
    try:
        lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SystemExit(f"cannot read the exemption list at {path}: {error}")
    names: set[str] = set()
    patterns: list[re.Pattern] = []
    for number, line in enumerate(lines, 1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("re:"):
            try:
                patterns.append(re.compile(entry[3:]))
            except re.error as error:
                raise SystemExit(f"{path}:{number}: bad regex: {error}")
        else:
            names.add(entry)
    if not names and not patterns:
        raise SystemExit(f"{path} lists nothing; refusing to run with no exemptions")
    return names, patterns


def exempt(name: str, names: set[str], patterns: list[re.Pattern]) -> bool:
    return name in names or any(p.search(name) for p in patterns)


_REFS_QUERY = """
query($owner:String!, $name:String!, $cursor:String) {
  repository(owner:$owner, name:$name) {
    refs(refPrefix:"refs/heads/", first:100, after:$cursor) {
      pageInfo { hasNextPage endCursor }
      nodes {
        name
        target { ... on Commit { oid committedDate } }
      }
    }
  }
}
"""


def list_branches(api: Api) -> list[dict]:
    """Every branch with tip SHA and committer date (a rebase keeps the author date)."""
    out: list[dict] = []
    cursor = None
    while True:
        page = api.graphql(
            _REFS_QUERY, {"owner": api.owner, "name": api.name, "cursor": cursor}
        )["repository"]["refs"]
        for node in page["nodes"]:
            target = node.get("target") or {}
            if target.get("oid"):
                out.append(
                    {
                        "name": node["name"],
                        "sha": target["oid"],
                        "date": _parse(target["committedDate"]),
                    }
                )
        if not page["pageInfo"]["hasNextPage"]:
            return out
        cursor = page["pageInfo"]["endCursor"]


def _parse(stamp: str) -> dt.datetime:
    return dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def protected_branches(api: Api) -> set[str]:
    names, page = set(), 1
    while True:
        batch = api.get(f"/branches?protected=true&per_page=100&page={page}")
        if not batch:
            return names
        names.update(b["name"] for b in batch)
        page += 1


def pr_branches(api: Api) -> set[str]:
    """Head and base refs of open PRs; removing a base retargets or closes them."""
    names, page = set(), 1
    while True:
        batch = api.get(f"/pulls?state=open&per_page=100&page={page}")
        if not batch:
            return names
        for pull in batch:
            names.add(pull["head"]["ref"])
            names.add(pull["base"]["ref"])
        page += 1


def ref_sha(api: Api, name: str) -> str | None:
    """Current tip of a branch, or None if it is gone."""
    try:
        ref = api.get(f"/git/ref/heads/{urllib.parse.quote(name, safe='/')}")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
    return ref["object"]["sha"]


def still_exempt(api: Api, name: str) -> str | None:
    """Re-check protection and open-PR membership right before a destructive
    push; the lease only guards the SHA, and both can change mid-run."""
    try:
        branch = api.get(f"/branches/{urllib.parse.quote(name, safe='/')}")
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        return "it no longer exists"
    if (branch or {}).get("protected"):
        return "it is protected"
    quoted = urllib.parse.quote(name, safe="")
    owner = urllib.parse.quote(api.owner, safe="")
    for role, query in (("base", f"base={quoted}"), ("head", f"head={owner}:{quoted}")):
        if api.get(f"/pulls?state=open&per_page=1&{query}"):
            return f"it is the {role} of an open pull request"
    return None


def comments(api: Api, sha: str) -> list[dict]:
    return api.get(f"/commits/{sha}/comments?per_page=100") or []


def find_marker(api: Api, sha: str, marker: str, author: str) -> dict | None:
    """The bot's own marker comment; a pasted look-alike must not start the clock."""
    for comment in comments(api, sha):
        if marker not in (comment.get("body") or ""):
            continue
        if (comment.get("user") or {}).get("login") == author:
            return comment
    return None


def archive(api: Api, branch: dict, today: dt.date, author: str) -> str:
    """Copy under archive/<today>/, comment, delete the original. Idempotent
    across a run that stopped between the three writes."""
    name, sha = branch["name"], branch["sha"]
    if ref_sha(api, name) != sha:
        return f"skipped {name}: moved since the scan"
    reason = still_exempt(api, name)
    if reason:
        return f"skipped {name}: {reason}"
    target = f"{ARCHIVE_PREFIX}{today.isoformat()}/{name}"
    existing = ref_sha(api, target)
    if existing is None:
        try:
            api.write("POST", "/git/refs", {"ref": f"refs/heads/{target}", "sha": sha})
        except urllib.error.HTTPError as error:
            # A 422 is any validation failure, not only already-exists: confirm
            # the archive ref exists before anything is deleted.
            if error.code != 422:
                raise
        if api.apply:
            existing = ref_sha(api, target)
            if existing is None:
                raise RuntimeError(
                    f"{target} does not exist after trying to create it, "
                    f"so {name} was left alone"
                )
    if existing is not None and existing != sha:
        return f"skipped {name}: {target} already exists at another commit"
    if find_marker(api, sha, marker("archived", target), author) is not None:
        return _finish_archive(api, name, sha, target)
    api.write(
        "POST",
        f"/commits/{sha}/comments",
        {
            "body": (
                f"{marker('archived', target)}\n"
                f"`{name}` has had no new commits since "
                f"{branch['date'].date().isoformat()}, so it has been moved to "
                f"`{target}`. Nothing is lost -- this commit is still here, and "
                f"one command puts the branch back:\n\n"
                f"```\ngit push origin {shlex.quote(target + ':' + name)}\n```\n\n"
                f"The archive copy is kept for a while and then removed, with a "
                f"separate comment here giving notice first."
            )
        },
    )
    return _finish_archive(api, name, sha, target)


def _finish_archive(api: Api, name: str, sha: str, target: str) -> str:
    reason = still_exempt(api, name)
    if reason:
        return f"archived {name} -> {target}; original kept, {reason}"
    if api.delete_branch(name, sha):
        return f"archived {name} -> {target}"
    return f"archived {name} -> {target}; original kept, it moved mid-run"


def give_notice(api: Api, branch: dict, original: str, delete_on: dt.date) -> str:
    api.write(
        "POST",
        f"/commits/{branch['sha']}/comments",
        {
            "body": (
                f"{marker('delete-notice', branch['name'])}\n"
                f"`{branch['name']}` is due to be deleted on "
                f"{delete_on.isoformat()}.\n\n"
                f"To keep it, restore the branch:\n\n"
                f"```\ngit push origin {shlex.quote(branch['name'] + ':' + original)}\n```\n\n"
                f"To stop the clock without restoring anything, delete this "
                f"comment -- the deletion only happens while it stands."
            )
        },
    )
    return f"notice on {branch['name']}, deletes {delete_on.isoformat()}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stale-days", type=int, default=30)
    parser.add_argument("--archive-days", type=int, default=30)
    parser.add_argument("--notice-days", type=int, default=14)
    parser.add_argument(
        "--exemptions",
        default=".github/stale-branch-exemptions.txt",
        help="List of exact branch names and re: patterns never to touch.",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=50,
        help="Cap per run, so a first run cannot notify hundreds of people at once.",
    )
    parser.add_argument(
        "--bot-login",
        default=BOT_LOGIN,
        help="Only this account's marker comments are believed.",
    )
    parser.add_argument(
        "--apply", action="store_true", help="Without this, report only."
    )
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        print("GITHUB_TOKEN and GITHUB_REPOSITORY must be set", file=sys.stderr)
        return 2

    api = Api(token, repo, args.apply)
    exempt_names, exempt_patterns = load_exemptions(args.exemptions)
    now = dt.datetime.now(dt.timezone.utc)
    today = now.date()

    skip = protected_branches(api) | pr_branches(api)
    actions: list[str] = []
    failures: list[str] = []

    def act(branch: dict) -> str | None:
        name = branch["name"]
        archived_on = archive_date(name)

        if archived_on is None:
            if (
                name in skip
                or exempt(name, exempt_names, exempt_patterns)
                or name.startswith(ARCHIVE_PREFIX)
            ):
                return None
            if (now - branch["date"]).days < args.stale_days:
                return None
            return archive(api, branch, today, args.bot_login)

        # Only refs this workflow made: the bot's archiving comment naming this
        # ref is the provenance. Deleting it opts the branch out for good.
        original = _ARCHIVED.match(name).group(2)
        if (
            find_marker(api, branch["sha"], marker("archived", name), args.bot_login)
            is None
        ):
            return None
        # Both names: a line added later rescues the original; the archive ref
        # can itself be protected or in a pull request.
        if (
            name in skip
            or original in skip
            or exempt(name, exempt_names, exempt_patterns)
            or exempt(original, exempt_names, exempt_patterns)
        ):
            return None
        notice = find_marker(
            api, branch["sha"], marker("delete-notice", name), args.bot_login
        )

        if notice is None:
            if (today - archived_on).days < args.archive_days:
                return None
            return give_notice(
                api, branch, original, today + dt.timedelta(days=args.notice_days)
            )

        if (now - _parse(notice["created_at"])).days < args.notice_days:
            return None
        reason = still_exempt(api, name)
        if reason:
            return f"skipped {name}: {reason}"
        if not api.delete_branch(name, branch["sha"]):
            return f"skipped {name}: moved since the scan"
        return f"deleted {name}"

    for branch in sorted(list_branches(api), key=lambda b: b["date"]):
        if len(actions) >= args.max_actions:
            break
        # One failing branch must not strand the rest; the job still goes red.
        try:
            done = act(branch)
        except (urllib.error.HTTPError, RuntimeError) as error:
            failures.append(f"{branch['name']}: {error}")
            continue
        if done:
            actions.append(done)

    verb = "did" if args.apply else "would"
    print(f"{verb} act on {len(actions)} branch(es); {api.writes} write call(s)")
    for line in actions:
        print(f"  {line}")
    for line in failures:
        print(f"  FAILED {line}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
