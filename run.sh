#/usr/local/bin

rm -rf ./aiter/jit/build/ck/
rm -rf ./aiter/jit/build/module_fmla_fwd/
rm -rf ./aiter/jit/module_fmla_fwd.so
ROCM_VISIBLE_DEVICES=1 python3 ./op_tests/test_fmla.py
