#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
python validate_prefix.py
python stress_prefix.py --mode opt
python stress_prefix.py --mode opt_both
python bench_mhc.py
python bench_prefix.py
python rotating_prefix.py
python final_checks.py
python report.py
