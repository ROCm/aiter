#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd "$(dirname "$0")"
python scratch_validate.py
python scratch_stress.py
python publication_acyclic.py
python wrap.py
python check_api.py
python scratch_bench.py
