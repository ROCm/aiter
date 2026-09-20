#!/usr/bin/env bash
# Independent re-run of the seven gates. Why it exists: an agent will report "I ran them,
# all green." This script trusts no such self-report; it trusts only triage.py's exit code.
# Proven useful: #2510's first round had a red ledger gate and #2409 had a red independent
# gate, both where the agent claimed all green (the latter had run only five gates, with
# independent not among them).
set -euo pipefail
exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_gates.py" "$@"
