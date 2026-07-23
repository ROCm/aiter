# bash op_tests/flydsl_tests/run_test_flydsl_gdr_k5_prefill.sh
# mi308: FlyDSL mfma16_hip K5 prefill -- correctness (vs FP32 ref) + perf
# comparison (flydsl-hip vs hip(C++) vs triton(opt_vk)).
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-7}
T=op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py

# correctness vs FP32 reference
FLYDSL_RUNTIME_ENABLE_CACHE=0 python -m pytest \
    ${T}::TestCorrectness::test_correctness_flydsl_mfma16_hip -v

# perf: prints a per-shape us table (need -s); flydsl-hip vs hip vs triton
FLYDSL_RUNTIME_ENABLE_CACHE=0 python -m pytest \
    ${T}::TestPerformance -v -s
