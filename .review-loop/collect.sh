#!/usr/bin/env bash
# Collect a finished run's artifacts from the /tmp WORK dir into reports/ — /tmp gets
# wiped, this does not. A missing required artifact is a failure: an incomplete report is
# more dangerous than no report, because it looks complete.
set -euo pipefail
exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_collect.py" "$@"
