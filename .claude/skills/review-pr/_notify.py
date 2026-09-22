"""Route a failed review to the owner responsible for that CLASS of failure and post one PR
comment. Single source of truth for the failure-class -> owner map, so run_one.sh (which tags
the failure with `fail <class> ...`) and the workflow (which reports it) cannot drift:
check_prompts.py asserts every class run_one.sh can emit is known here.

Called by the workflow's notify step as `python3 _notify.py <pr>`. Reads `.aiter-review-status`
(written by run_one.sh as "<class>\t<message>"). Owners default below and are overridden by the
repo variable named in each row, so ownership changes without touching code.
"""

import json
import os
import sys
import urllib.request

# class -> (repo-var that overrides the owner, default owner, what is wrong)
CLASSES = {
    "flow":    ("AITER_FLOW_OWNER",    "zufayu",   "the review pipeline itself (bot logic / prompts)"),
    "env":     ("AITER_RUNNER_OWNER",  "zufayu",   "the runner environment (gh, token, git)"),
    "glm":     ("AITER_GLM_OWNER",     "yhl-amd",  "the GLM model service"),
    "atom":    ("AITER_ATOM_OWNER",    "valarLip",         "the ATOM serving framework"),
    "machine": ("AITER_MACHINE_OWNER", "gyohuangxin", "the machine (GPU, disk, network, OS)"),
}


def owner_for(cls):
    var, default, _ = CLASSES[cls]
    return os.environ.get(var) or default or os.environ.get("AITER_FLOW_OWNER") or "zufayu"


def main():
    if len(sys.argv) < 2:
        print("usage: _notify.py <pr>"); return 2
    pr = sys.argv[1]
    path = os.path.join(os.environ.get("GITHUB_WORKSPACE", "."), ".aiter-review-status")
    if not os.path.exists(path):
        return 0  # the review succeeded -> nothing to report
    cls, _, msg = open(path, encoding="utf-8").read().strip().partition("\t")
    if cls not in CLASSES:
        cls = "flow"  # an unknown tag routes to the bot owner rather than nobody
    owner = owner_for(cls)
    domain = CLASSES[cls][2]
    body = (f"⚠️ **aiter-bot** — this review could not complete: a problem with **{domain}**, "
            f"not with this PR.\n\n> {msg}\n\n@{owner} — please take a look.")
    if cls == "glm":
        body += (f"\n\n<sub>Triage: if GLM serves but its output is malformed it may be the ATOM "
                 f"framework (@{owner_for('atom')}); if the box, GPU, disk or network is unhealthy "
                 f"it may be the machine (@{owner_for('machine')}).</sub>")
    body += "\n\nRe-comment `@aiter-bot review` once it is resolved."

    tok = os.environ.get("AITER_BOT_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not tok or not repo:
        print("no AITER_BOT_TOKEN / GITHUB_REPOSITORY -- would post:\n" + body); return 0
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues/{pr}/comments",
        data=json.dumps({"body": body}).encode(),
        headers={"Authorization": f"token {tok}", "Accept": "application/vnd.github+json"},
        method="POST")
    try:
        urllib.request.urlopen(req, timeout=15)
        print(f"reported [{cls}] -> @{owner}")
    except Exception as e:  # noqa: BLE001 - a failed notice must not fail CI
        print(f"notify failed: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
