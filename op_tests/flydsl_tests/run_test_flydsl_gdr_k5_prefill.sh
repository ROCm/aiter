# bash op_tests/flydsl_tests/run_test_flydsl_gdr_k5_prefill.sh 2>&1 | tee /workspace/perf_flydsl_k5_0715/test_flydsl_gdr_k5_prefill_0713.txt
export GATED_DELTA_RULE_TRITON_AUTOTUNE=1
export HIP_VISIBLE_DEVICES=7
# rm -rf ~/.triton/cache
# FLYDSL_RUNTIME_ENABLE_CACHE=0 HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} python -m pytest op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py::TestPerformance -v -s
rm -rf ~/.triton/cache
FLYDSL_RUNTIME_ENABLE_CACHE=0 HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} python -m pytest op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py::TestPerformance -k "varlen-16k-aws" -v -s
rm -rf ~/.triton/cache
FLYDSL_RUNTIME_ENABLE_CACHE=0 HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} python -m pytest op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py::TestPerformance -k "varlen-32k-aws" -v -s
rm -rf ~/.triton/cache
FLYDSL_RUNTIME_ENABLE_CACHE=0 HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} python -m pytest op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py::TestPerformance -k "varlen-32k-qwen" -v -s
rm -rf ~/.triton/cache
FLYDSL_RUNTIME_ENABLE_CACHE=0 HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} python -m pytest op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py::TestPerformance -k "Qwen3.5-397B" -v -s
rm -rf ~/.triton/cache
FLYDSL_RUNTIME_ENABLE_CACHE=0 HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} python -m pytest op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py::TestPerformance -k "Qwen3.5-35B" -v -s