asm_mla_decode_fwd_torch_test:
	hipcc -L. -Wl,-rpath,. asm_mla_decode_fwd_lib.so asm_mla_decode_fwd_torch.cpp asm_mla_decode_fwd_torch_test.cpp -std=c++17 -O0 -g -I$(shell python3 -c "from torch.utils import cpp_extension; print(cpp_extension.include_paths()[0])") \
	  -I$(shell python3 -c "from torch.utils import cpp_extension; print(cpp_extension.include_paths()[1])") \
	  -Wl,-rpath,$(shell python3 -c "from torch.utils import cpp_extension; print(' '.join(cpp_extension.library_paths()))") \
	  -L$(shell python3 -c "from torch.utils import cpp_extension; print(' '.join(cpp_extension.library_paths()))") -lgtest -lgtest_main -L/usr/local/lib -ltorch -ltorch_cpu -ltorch_hip -lc10 -lc10_hip -lfmt -lcrypto -o asm_mla_decode_fwd_torch_test.out

asm_mla_decode_fwd_test:
	hipcc -L. -Wl,-rpath,. asm_mla_decode_fwd_lib.so asm_mla_decode_fwd_test.cpp -std=c++17 -O0 -g -lgtest -lgtest_main -L/usr/local/lib -lfmt -lcrypto -o asm_mla_decode_fwd_test.out

asm_mla_decode_fwd_lib.so:
	hipcc asm_mla_decode_fwd.cpp -o asm_mla_decode_fwd_lib.so -fPIC -shared -I../include -std=c++17 -O3 -lfmt -lcrypto -ldl

clean:
	rm -f asm_mla_decode_fwd_test.out asm_mla_decode_fwd_lib.so