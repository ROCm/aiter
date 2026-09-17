#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Archive stale branches, then delete the archives after notice.

Three stages, so that nothing is removed without somewhere to get it back from:

  day 0    the branch has had no commits for --stale-days
           -> copy it to ``archive/<YYYY-MM-DD>/<name>``, comment on the tip
              commit saying where it went, delete the original ref
  day 30   it has sat in the archive for --archive-days
           -> comment again, this time announcing deletion
  day 74   the notice has stood for --notice-days
           -> delete the archive ref

74 days from the last commit to the ref going away, 44 of them with the commits
still reachable under a name someone can find.

The dates are not stored anywhere. The archive date is the one in the ref name,
and the notice date is the creation date of the notice comment, so a run holds
no state and two runs cannot disagree. Deleting the notice comment resets the
last clock, which is the per-branch opt-out: it needs no admin, and the comment
says so.

Recovering an archived branch is one command, whatever stage it is in:

    git push origin archive/<date>/<name>:<name>

Exempt, and skipped before anything is written: protected branches, the head or
base branch of any open pull request, and everything listed in
.github/stale-branch-exemptions.txt. Base branches matter as much as head
branches -- deleting the base of an open PR retargets or closes it.

The exemption list is checked at every stage against the branch's original
name, so adding a line to it rescues a branch that has already been archived:
the copy stops receiving notices and is never deleted. It does not restore the
ref, which stays a deliberate one-command step.

Nothing is written unless --apply is passed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request

API = "https://api.github.com"
NOTICE_MARKER = "<!-- stale-branch-delete-notice -->"
ARCHIVE_MARKER = "<!-- stale-branch-archived -->"
ARCHIVE_PREFIX = "archive/"
_ARCHIVED = re.compile(r"^archive/(\d{4}-\d{2}-\d{2})/(.+)$")


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
    """Exact names and ``re:`` patterns from the checked-in list.

    A missing file is fatal rather than an empty list. "No exemptions" and "the
    file moved" look identical at the call site, and only one of them should
    let this script near a branch called main.
    """
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
    """Every branch with its tip SHA and commit date, 100 per request.

    The date is the committer date, not the author date: a rebased or
    cherry-picked branch keeps its original author date, so authoring is a
    measure of when the work was written rather than when the branch last
    moved, and a branch someone rebased onto main this morning would read as
    months old.
    """
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
    """Head *and* base refs of every open pull request.

    A base branch is usually the head of another open PR in a stack, but not
    always -- an integration branch several PRs target has no PR of its own,
    and removing it retargets or closes all of them.
    """
    names, page = set(), 1
    while True:
        batch = api.get(f"/pulls?state=open&per_page=100&page={page}")
        if not batch:
            return names
        for pull in batch:
            names.add(pull["head"]["ref"])
            names.add(pull["base"]["ref"])
        page += 1


def comments(api: Api, sha: str) -> list[dict]:
    return api.get(f"/commits/{sha}/comments?per_page=100") or []


def find_marker(api: Api, sha: str, marker: str) -> dict | None:
    for comment in comments(api, sha):
        if marker in (comment.get("body") or ""):
            return comment
    return None


def archive(api: Api, branch: dict, today: dt.date) -> str:
    """Copy the branch under archive/<today>/, say so, then remove the original."""
    target = f"{ARCHIVE_PREFIX}{today.isoformat()}/{branch['name']}"
    api.write(
        "POST", "/git/refs", {"ref": f"refs/heads/{target}", "sha": branch["sha"]}
    )
    api.write(
        "POST",
        f"/commits/{branch['sha']}/comments",
        {
            "body": (
                f"{ARCHIVE_MARKER}\n"
                f"`{branch['name']}` has had no new commits since "
                f"{branch['date'].date().isoformat()}, so it has been moved to "
                f"`{target}`. Nothing is lost -- this commit is still here, and "
                f"one command puts the branch back:\n\n"
                f"```\ngit push origin {target}:{branch['name']}\n```\n\n"
                f"The archive copy is kept for a while and then removed, with a "
                f"separate comment here giving notice first."
            )
        },
    )
    api.write("DELETE", f"/git/refs/heads/{branch['name']}")
    return f"archived {branch['name']} -> {target}"


def give_notice(api: Api, branch: dict, original: str, delete_on: dt.date) -> str:
    api.write(
        "POST",
        f"/commits/{branch['sha']}/comments",
        {
            "body": (
                f"{NOTICE_MARKER}\n"
                f"`{branch['name']}` is due to be deleted on "
                f"{delete_on.isoformat()}.\n\n"
                f"To keep it, restore the branch:\n\n"
                f"```\ngit push origin {branch['name']}:{original}\n```\n\n"
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

    for branch in sorted(list_branches(api), key=lambda b: b["date"]):
        if len(actions) >= args.max_actions:
            break
        name = branch["name"]
        archived = _ARCHIVED.match(name)

        if not archived:
            if (
                name in skip
                or exempt(name, exempt_names, exempt_patterns)
                or name.startswith(ARCHIVE_PREFIX)
            ):
                continue
            if (now - branch["date"]).days >= args.stale_days:
                actions.append(archive(api, branch, today))
            continue

        # An archive ref. Its date is in its name; the original name follows.
        archived_on = dt.date.fromisoformat(archived.group(1))
        original = archived.group(2)
        # Checked again here, against the name the branch had: a line added to
        # the list after the archiving run still rescues the copy.
        if original in skip or exempt(original, exempt_names, exempt_patterns):
            continue
        notice = find_marker(api, branch["sha"], NOTICE_MARKER)

        if notice is None:
            if (today - archived_on).days >= args.archive_days:
                actions.append(
                    give_notice(
                        api,
                        branch,
                        original,
                        today + dt.timedelta(days=args.notice_days),
                    )
                )
            continue

        if (now - _parse(notice["created_at"])).days >= args.notice_days:
            api.write("DELETE", f"/git/refs/heads/{name}")
            actions.append(f"deleted {name}")

    verb = "would" if not args.apply else "did"
    print(f"{verb} act on {len(actions)} branch(es); {api.writes} write call(s)")
    for line in actions:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
