
ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 16  -q 21 -d 128 -c
ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 32  -q 21 -d 128 -c
ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 128 -q 21 -d 128 -c

ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 16  -q 64 -d 128 -c
ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 32  -q 64 -d 128 -c
ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 128 -q 64 -d 128 -c

ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 16  -q 256 -d 128 -c
ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 32  -q 256 -d 128 -c
ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 128 -q 256 -d 128 -c

ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 16  -q 512 -d 128 -c
ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 32  -q 512 -d 128 -c
ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 128 -q 512 -d 128 -c

ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 16  -q 1200 -d 128 -c
ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 32  -q 1200 -d 128 -c
ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 128 -q 1200 -d 128 -c

ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 16  -q 3200 -d 128 -c
ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 32  -q 3200 -d 128 -c
ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 128 -q 3200 -d 128 -c

ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 16  -q 5200 -d 128 -c
ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 32  -q 5200 -d 128 -c
ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 128 -q 5200 -d 128 -c

ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 16  -q 8192 -d 128 -c
ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 32  -q 8192 -d 128 -c

ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 16  -q 10000 -d 128 -c
ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 32  -q 10000 -d 128 -c

ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 16  -q 16384 -d 128 -c

ROCR_VISIBLE_DEVICES=3 python ./op_tests/test_mha_fp8.py -b 2 -n 2  -q 90000 -d 128 -c