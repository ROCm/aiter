#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
python validate.py
python stress.py
python edges.py
python ab.py
python rotating_ab.py
python trace.py
python final_report.py
