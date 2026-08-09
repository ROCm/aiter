#!/bin/bash
# Hardware gate for the buffer_ops / raw-dialect migration work.
# Runs every op_test that covers a migrated kernel, on a cold FlyDSL cache.
PY=/opt/venv/bin/python
OUT=/tmp/validate
rm -rf "$OUT" ~/.flydsl/cache; mkdir -p "$OUT"

pass=0; fail=0
run() {  # run <label> <cmd...>
  local label="$1"; shift
  echo "=== $label"
  if timeout 3600 "$@" >"$OUT/$label.log" 2>&1; then
    echo "    PASS"; pass=$((pass+1))
  else
    echo "    FAIL  (see $OUT/$label.log)"; fail=$((fail+1))
  fi
}

# pytest-collectable
run moe_a8w4      $PY -m pytest op_tests/flydsl_tests/test_flydsl_moe_a8w4.py -q
run moe_a16wfp4   $PY -m pytest op_tests/flydsl_tests/test_flydsl_moe_a16wfp4.py -q
run silu_fq       $PY -m pytest op_tests/flydsl_tests/test_silu_and_mul_fq.py -q
run splitk_hgemm  $PY -m pytest aiter/ops/flydsl/test_flydsl_splitk_hgemm.py -q
run fhmoe         $PY -m pytest op_tests/test_fhmoe.py -q
run causal_conv1d $PY -m pytest op_tests/test_causal_conv1d_prefill_split_qkv.py -q -k flydsl
run gdr_decode    $PY -m pytest aiter/ops/flydsl/test_flydsl_linear_attention.py -q -k gdr_decode
run lin_attn_pref $PY -m pytest op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py -q

# scripts: their test fns take bare positional args, so pytest collection fails.
# That is not a regression -- they must be run as scripts.
run moe_sorting   $PY op_tests/test_moe_sorting.py
run compress_attn $PY op_tests/test_flydsl_compress_attn.py
run qk_norm_rope  $PY op_tests/test_flydsl_qk_norm_rope_quant.py

echo
echo "PASS=$pass FAIL=$fail"
[ "$fail" -eq 0 ]
