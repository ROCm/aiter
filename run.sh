#!/usr/local/bin

rm -rf aiter/jit/module_norm.so

rm -rf aiter/jit/build/module_norm/

python3 op_tests/test_layernorm2dFusedAddQuant.py --mode 3
