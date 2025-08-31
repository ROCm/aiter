#!/usr/local/bin

rm -rf aiter/jit/module_rmsnorm_fused.so

rm -rf aiter/jit/build/ck/
rm -rf aiter/jit/build/module_rmsnorm_fused/

python3 op_tests/test_rmsnorm2dFusedAddQuant.py --mode 4
