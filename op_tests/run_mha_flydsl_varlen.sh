#!/bin/bash
set -e

export FLYDSL_RUNTIME_ENABLE_CACHE=0

cd "$(dirname "$0")"

echo "=== FlyDSL MHA Varlen Test (causal + non-causal) ==="
python3 -u test_mha_flydsl_varlen.py
