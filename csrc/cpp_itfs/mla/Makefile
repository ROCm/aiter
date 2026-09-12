asm_mla_decode_fwd_test:
	hipcc asm_mla_decode_fwd_lib.so asm_mla_decode_fwd_test.cpp -std=c++17 -O0 -g -lgtest -lgtest_main -L/usr/local/lib -lfmt -lcrypto -o asm_mla_decode_fwd_test.out

asm_mla_decode_fwd_lib.so:
	hipcc asm_mla_decode_fwd.cpp -o asm_mla_decode_fwd_lib.so -fPIC -shared -I../include -std=c++17 -O3 -lfmt -lcrypto -ldl

clean:
	rm -f asm_mla_decode_fwd_test.out asm_mla_decode_fwd_lib.so