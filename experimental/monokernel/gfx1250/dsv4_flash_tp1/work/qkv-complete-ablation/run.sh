#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd "$(dirname "$0")"
python validate_final.py
python check_api.py
python final_bench.py
python stress.py
