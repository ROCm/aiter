#!/usr/bin/env python
"""Minimal `gh` replacement covering exactly the calls review-pr/fetch.sh makes.

No GitHub CLI is installable in this environment, and `gh pr diff` would in any case
fail on aiter#4961 (diff over GitHub's 20000-line API cap).  `pr diff` here is produced
from a LOCAL checkout with `git diff <merge-base> <head>`, which is what `gh pr diff`
returns and what the skill tells a reviewer to do for an over-cap PR.
"""
import json
import os
import subprocess
import sys
import urllib.request

API = "https://api.github.com"
REPO_DIR = os.environ.get("GH_SHIM_REPO_DIR") or os.getcwd()


def api(path):
    url = path if path.startswith("http") else f"{API}/{path.lstrip('/')}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "gh-shim",
    })
    tok = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def api_paged(path, cap=20):
    out = []
    for page in range(1, cap + 1):
        sep = "&" if "?" in path else "?"
        chunk = api(f"{path}{sep}per_page=100&page={page}")
        if not isinstance(chunk, list) or not chunk:
            break
        out.extend(chunk)
        if len(chunk) < 100:
            break
    return out


def git(*args, check=True):
    env = dict(os.environ)
    if sys.platform == "win32":
        # Windows git needs the SSL backend pinned. Forcing it on Linux breaks every network
        # git op: Ubuntu git is a gnutls build and dies with
        # "fatal: Unsupported SSL backend 'openssl'. Supported SSL backends: gnutls".
        env.update({
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.sslBackend",
            "GIT_CONFIG_VALUE_0": "openssl",
        })
    p = subprocess.run(["git", "-C", REPO_DIR, *args], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env)
    if check and p.returncode != 0:
        sys.stderr.write(p.stderr)
        sys.exit(p.returncode)
    return p


def pr_meta(repo, num):
    pr = api(f"repos/{repo}/pulls/{num}")
    files = api_paged(f"repos/{repo}/pulls/{num}/files")
    reviews = api_paged(f"repos/{repo}/pulls/{num}/reviews")
    comments = api_paged(f"repos/{repo}/issues/{num}/comments")
    return {
        "number": pr["number"],
        "title": pr.get("title") or "",
        "body": pr.get("body") or "",
        "state": pr.get("state"),
        "labels": [{"name": l["name"]} for l in pr.get("labels", [])],
        "author": {"login": (pr.get("user") or {}).get("login", "")},
        "baseRefName": pr["base"]["ref"],
        "baseRefOid": pr["base"]["sha"],
        "headRefOid": pr["head"]["sha"],
        "files": [{"path": f["filename"], "additions": f["additions"],
                   "deletions": f["deletions"]} for f in files],
        "reviews": [{"author": {"login": (r.get("user") or {}).get("login", "")},
                     "body": r.get("body") or "", "state": r.get("state")}
                    for r in reviews],
        "comments": [{"author": {"login": (c.get("user") or {}).get("login", "")},
                      "body": c.get("body") or ""} for c in comments],
    }


def ensure(repo, sha, ref_spec):
    """Make sure `sha` is present locally, fetching ref_spec from the canonical URL."""
    if git("cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode == 0:
        return
    git("fetch", "-q", f"https://github.com/{repo}", ref_spec, check=False)
    if git("cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode != 0:
        git("fetch", "-q", "--unshallow", f"https://github.com/{repo}", ref_spec, check=False)


def pr_diff(repo, num):
    pr = api(f"repos/{repo}/pulls/{num}")
    head, base = pr["head"]["sha"], pr["base"]["sha"]
    ensure(repo, head, f"refs/pull/{num}/head")
    ensure(repo, base, pr["base"]["ref"])
    mb = git("merge-base", base, head, check=False)
    if mb.returncode != 0:
        git("fetch", "-q", "--unshallow", f"https://github.com/{repo}",
            pr["base"]["ref"], check=False)
        mb = git("merge-base", base, head)
    env = dict(os.environ)
    p = subprocess.run(["git", "-C", REPO_DIR, "diff", "--no-color",
                        mb.stdout.strip(), head], capture_output=True, env=env)
    if p.returncode != 0:
        sys.stderr.buffer.write(p.stderr)
        sys.exit(p.returncode)
    sys.stdout.buffer.write(p.stdout)


def jq_path(data, expr):
    cur = data
    for part in expr.strip().lstrip(".").split("."):
        if not part:
            continue
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


def main(argv):
    if not argv:
        sys.exit("gh-shim: no command")
    cmd = argv[0]
    args = argv[1:]

    def opt(name, default=None):
        return args[args.index(name) + 1] if name in args else default

    if cmd == "pr" and args and args[0] == "view":
        print(json.dumps(pr_meta(opt("--repo", "ROCm/aiter"), args[1]), indent=2))
    elif cmd == "pr" and args and args[0] == "diff":
        pr_diff(opt("--repo", "ROCm/aiter"), args[1])
    elif cmd == "issue" and args and args[0] == "view":
        d = api(f"repos/{opt('--repo', 'ROCm/aiter')}/issues/{args[1]}")
        print(json.dumps({"title": d.get("title") or "", "body": d.get("body") or ""}))
    elif cmd == "api":
        path = args[0]
        data = api_paged(path) if path.rstrip("/").endswith(("comments", "reviews", "files")) \
            else api(path)
        jq = opt("--jq")
        print(json.dumps(data, indent=2) if not jq else (jq_path(data, jq) or ""))
    elif cmd == "auth":
        print("gh-shim: unauthenticated REST access")
    else:
        sys.exit(f"gh-shim: unsupported command: {' '.join(argv)}")


if __name__ == "__main__":
    main(sys.argv[1:])
