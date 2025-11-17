python ./matmul_fp16.py
hipcc ./test_matmul.cpp ./matmul_fp16.b3a5a34c_0d1d2d34567c89c10d11c121314.cpp -lgtest -lgtest_main -o ./test_matmul.out
./test_matmul.out
