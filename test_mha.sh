AITER_LOG_MORE=1 pytest -vs op_tests/test_mha.py >& test_prebuild_mha.log
AITER_LOG_MORE=1 pytest -vs op_tests/test_mha_varlen.py >& test_mha_varlen_prebuild.log