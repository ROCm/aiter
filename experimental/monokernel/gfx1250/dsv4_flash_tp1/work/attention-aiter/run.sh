#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd "$(dirname "$0")"
python validate.py
python stress.py
python bench.py
python tune.py
python rotating_bench.py
python trace.py
python report.py
