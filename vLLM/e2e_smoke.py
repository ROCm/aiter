#!/usr/bin/env python3
"""Per-token view of a greedy smoke completion, for e2e.sh.

For every generated token: the greedy pick, the runner-up and the logprob
margin between them. A small margin is a near-tie that rounding differences
(e.g. another all-reduce summation order) can flip. The last line names the
closest call.

    e2e_smoke.py <dir>    # reads <dir>/smoke.json (a /v1/completions
                          # response requested with "logprobs": N)
"""

import json
import sys
from pathlib import Path


def token_rows(logprobs):
    """Yield (pos, token, logprob, runner_up, runner_up_logprob) per token."""
    for pos, (tok, top) in enumerate(
        zip(logprobs["tokens"], logprobs["top_logprobs"])
    ):
        pick = top.get(tok, logprobs["token_logprobs"][pos])
        others = sorted(
            ((lp, other) for other, lp in top.items() if other != tok), reverse=True
        )
        if others:
            alt_lp, alt = others[0]
            yield pos, tok, pick, alt, alt_lp


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    response = json.loads((Path(sys.argv[1]) / "smoke.json").read_text())
    logprobs = response["choices"][0]["logprobs"]
    if not logprobs:
        sys.exit("smoke.json has no logprobs: request them with \"logprobs\": 5")

    print("%4s  %-16s %8s  %-16s %8s  %7s"
          % ("pos", "token", "logprob", "runner-up", "logprob", "margin"))
    closest = None
    for pos, tok, pick, alt, alt_lp in token_rows(logprobs):
        margin = pick - alt_lp
        print("%4d  %-16s %8.3f  %-16s %8.3f  %7.3f"
              % (pos, json.dumps(tok), pick, json.dumps(alt), alt_lp, margin))
        if closest is None or margin < closest[0]:
            closest = (margin, pos, tok, alt)
    if closest:
        margin, pos, tok, alt = closest
        print("closest: pos %d %s over %s by %.3f"
              % (pos, json.dumps(tok), json.dumps(alt), margin))


if __name__ == "__main__":
    main()
