#!/usr/bin/env bash
# Publish gate (phase-2): post the card in reports/PR-<pr>/ to the PR, as aiter-bot,
# under the rules — hold 🔴 for a human / post non-🔴 / skip if already posted for this
# head / skip if merged. Dry-run by default; --post actually posts.
#   publish.sh 5560            # dry-run, print the post/hold decision only
#   publish.sh 5560 --post     # actually post (needs AITER_BOT_TOKEN(_FILE))
# Publishing identity = the owner of the bot token (aiter-bot).
set -euo pipefail
exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_publish.py" "$@"
