#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
python smoke.py
python validate.py
python bench.py
python rotating.py
python trace.py
python profile_groups.py
python report.py
