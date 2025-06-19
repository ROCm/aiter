python ./matmul_fp16.py
hipcc ./test_matmul.cpp ./matmul_fp16.53107118_0d1d2d34567c89c10d11c.cpp -lgtest -lgtest_main -o ./test_matmul.out
./test_matmul.out