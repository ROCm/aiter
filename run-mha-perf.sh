python op_tests/test_mha_flydsl_varlen.py -b 1 -nh 64 -sq 16384 -sk 16384 --random-value false --warmup 5 --repeat 20
python op_tests/test_mha_flydsl_varlen.py -b 1 -nh 64 -sq 16384 -sk 16384 --random-value true --warmup 5 --repeat 20
python op_tests/test_mha_flydsl_varlen.py -d_qk_v 128,128 -b 1 -nh 64 -sq 16384 -sk 16384 --random-value false --warmup 5 --repeat 20
python op_tests/test_mha_flydsl_varlen.py -d_qk_v 128,128 -b 1 -nh 64 -sq 16384 -sk 16384 --random-value true --warmup 5 --repeat 20


# python op_tests/test_mha_flydsl_batch.py -b 1 -nh 64 -sq 16384 -sk 16384 --random-value false --warmup 5 --repeat 20
# python op_tests/test_mha_flydsl_batch.py -b 1 -nh 64 -sq 16384 -sk 16384 --random-value true --warmup 5 --repeat 20


python op_tests/test_mha_flydsl_varlen.py --causal true --return_lse true -b 1 -nh 64 -sq 8192 -sk 8192 --random-value false --warmup 5 --repeat 20
python op_tests/test_mha_flydsl_varlen.py --causal true --return_lse true -d_qk_v 128,128 -b 1 -nh 64 -sq 8192 -sk 8192 --random-value false --warmup 5 --repeat 20
python op_tests/test_mha_flydsl_varlen.py --dtype mxfp8 --causal false --return_lse true -d_qk_v 128,128 -b 1 -nh 64 -sq 8192 -sk 8192 --random-value false --warmup 5 --repeat 20


python op_tests/test_mha_flydsl_varlen.py --causal true --return_lse true -b 1 -nh 32 -sq 8192 -sk 8192 --random-value false --warmup 5 --repeat 20
python op_tests/test_mha_flydsl_varlen.py --causal true --return_lse true -d_qk_v 128,128 -b 1 -nh 32 -sq 8192 -sk 8192 --random-value false --warmup 5 --repeat 20


rocprofv3 --att \
--kernel-include-regex "fmha_fwd_kernel_0" -- \
python profile_mha_flydsl_varlen_minimal.py --causal true --return_lse true -b 1 -nh 32 -sq 1024 -sk 1024 --random-value false --warmup 5 --repeat 20

python op_tests/test_mha_flydsl_varlen.py --causal true --return_lse true -b 1 -nh 32 -sq 1024 -sk 1024 --random-value false --warmup 5 --repeat 20
python op_tests/test_mha_flydsl_varlen.py --causal true --return_lse true -b 1 -nh 32 -sq 8192 -sk 8192 --random-value false --warmup 5 --repeat 20


