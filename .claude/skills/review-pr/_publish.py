"""Publish gate (phase-2): post a collected report to the PR, as aiter-bot, under the rules.

Rules (after run_one, given 7 green gates):
  - Review line is 🔴 HIGH RISK  → POST too: the reviewer who triggered it must see the high-risk
    findings (still advisory, not a merge gate); a warning annotation also flags it in the checks UI
  - otherwise (✅ / ⚠️ / notes only) → POST: post card.md as a PR comment
  - already posted (same head SHA, by marker) → SKIP: do not repost
  - PR already merged/closed         → SKIP: a post-merge follow-up should not go as a PR review
Publishing identity = the owner of AITER_BOT_TOKEN(_FILE) (aiter-bot). Dry-run by default; --post actually posts.

A marker line is embedded at the end of the comment for followup/dedup:
  <!-- aiter-bot review pr=<pr> head=<sha> -->
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPORTS = HERE / "reports"
REPO = os.environ.get("AITER_BOT_REPO") or os.environ.get("GITHUB_REPOSITORY", "ROCm/aiter")
BOT = os.environ.get("AITER_BOT_NAME", "aiter-bot")


def _token():
    t = os.environ.get("AITER_BOT_TOKEN")
    if t:
        return t.strip()
    tf = os.environ.get("AITER_BOT_TOKEN_FILE")
    if tf:
        try:
            m = re.findall(
                r"(?:ghp_|github_pat_)[A-Za-z0-9_]+",
                pathlib.Path(tf).read_text(encoding="utf-8"),
            )
            if m:
                return m[-1]
        except OSError:
            pass
    try:
        with open(os.path.expanduser("~/.git-credentials"), encoding="utf-8") as fh:
            for line in fh:
                if "github.com" in line and ":" in line:
                    return (
                        line.split("://", 1)[1].split("@")[0].split(":", 1)[1].strip()
                    )
    except OSError:
        pass
    return ""


TOK = _token()


def _api(path, method="GET", body=None):
    cmd = ["curl", "-s", "-m", "25", "-X", method]
    if TOK:
        cmd += ["-H", f"Authorization: token {TOK}"]
    cmd += ["-H", "Accept: application/vnd.github+json"]
    if body is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(body)]
    cmd.append(f"https://api.github.com/repos/{REPO}/{path}")
    try:
        return json.loads(
            subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
            or "null"
        )
    except (json.JSONDecodeError, OSError):
        return None


def head_sha(pr):
    try:
        d = json.loads(
            (REPORTS / f"PR-{pr}" / "pr_meta.json").read_text(encoding="utf-8")
        )
        return d.get("headRefOid") or (d.get("head") or {}).get("sha") or ""
    except (OSError, json.JSONDecodeError):
        return ""


def severity(card):
    """Severity of the Review line: 'high' | 'ok' (anything not 🔴 is ok, i.e. postable)."""
    for ln in card.splitlines():
        if ln.startswith("Review"):
            if "🔴" in ln or "HIGH RISK" in ln.upper():
                return "high"
            return "ok"
    return "ok"


def already_posted(pr, sha):
    """Whether aiter-bot already posted for this head (by marker)."""
    cs = _api(f"issues/{pr}/comments?per_page=100")
    if not isinstance(cs, list):
        return False
    marker = f"aiter-bot review pr={pr} head={sha}"
    return any(marker in (c.get("body") or "") for c in cs)


def decide(pr):
    """-> (action, reason). action in POST | HOLD | SKIP."""
    d = REPORTS / f"PR-{pr}"
    card = d / "card.md"
    if not card.is_file():
        return "SKIP", f"reports/PR-{pr}/card.md does not exist (run run_one first)"
    pull = _api(f"pulls/{pr}")
    if isinstance(pull, dict) and (
        pull.get("merged_at") or pull.get("state") != "open"
    ):
        state = "merged" if pull.get("merged_at") else pull.get("state")
        return (
            "SKIP",
            f"PR already {state} (post-merge follow-up, should not go as a PR review)",
        )
    sha = head_sha(pr)
    if sha and already_posted(pr, sha):
        return "SKIP", f"same head {sha[:9]} already posted (marker hit)"
    # Post the card for EVERY verdict, including 🔴: the reviewer who asked for the review must
    # see it, and 🔴 is exactly what they most need to see. It stays advisory (the Review line
    # says so) and never gates merge; main() adds a warning annotation for the 🔴 ones.
    return "POST", "postable"


def do_post(pr):
    d = REPORTS / f"PR-{pr}"
    body = (d / "card.md").read_text(encoding="utf-8").rstrip()
    sha = head_sha(pr)
    body += f"\n\n<!-- aiter-bot review pr={pr} head={sha} -->"
    r = _api(f"issues/{pr}/comments", method="POST", body={"body": body})
    if isinstance(r, dict) and r.get("html_url"):
        return True, r["html_url"]
    return False, str(r)[:200]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("prs", nargs="+", help="PR number(s)")
    ap.add_argument(
        "--post",
        action="store_true",
        help="actually post; default dry-run prints the decision only",
    )
    a = ap.parse_args(argv)
    print(
        f'=== publish gate  [{"POST" if a.post else "DRY-RUN"}]  identity={BOT}  repo={REPO} ==='
    )
    posted, failed = [], []
    for pr in a.prs:
        action, reason = decide(pr)
        if action == "POST" and a.post:
            ok, info = do_post(pr)
            print(f'  #{pr}: POST -> {"✅ "+info if ok else "❌ "+info}')
            (posted if ok else failed).append(pr)
            if ok and severity((REPORTS / f"PR-{pr}" / "card.md").read_text(encoding="utf-8")) == "high":
                # extra signal in the checks UI; the card itself is already posted as a comment
                print(f"::warning title=aiter-bot HIGH RISK::PR #{pr} — review flags HIGH RISK (advisory, not a merge gate)")
        else:
            print(f"  #{pr}: {action} — {reason}")
    if failed:
        print(f"  ❌ failed to post: {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
