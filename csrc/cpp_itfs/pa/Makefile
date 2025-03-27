pa_ragged_test:
	hipcc -L. -Wl,-rpath,. pa_ragged_lib.so pa_ragged_torch.cpp pa_ragged_test.cpp -std=c++17 -O0 -g -I$(shell python3 -c "from torch.utils import cpp_extension; print(cpp_extension.include_paths()[0])") \
	  -I$(shell python3 -c "from torch.utils import cpp_extension; print(cpp_extension.include_paths()[1])") \
	  -Wl,-rpath,$(shell python3 -c "from torch.utils import cpp_extension; print(' '.join(cpp_extension.library_paths()))") \
	  -L$(shell python3 -c "from torch.utils import cpp_extension; print(' '.join(cpp_extension.library_paths()))") -lgtest -L/usr/local/lib -ltorch -ltorch_cpu -ltorch_hip -lc10 -lc10_hip -lfmt -lcrypto -o pa_ragged_test.out

pa_ragged_lib.so:
	hipcc pa_ragged.cpp -o pa_ragged_lib.so -fPIC -shared -I../include -std=c++17 -O3 -lfmt -lcrypto -ldl

pa_test:
	hipcc pa.cpp pa_torch.cpp pa_test.cpp -std=c++17 -O3 -I$(shell python3 -c "from torch.utils import cpp_extension; print(cpp_extension.include_paths()[0])") \
	  -I$(shell python3 -c "from torch.utils import cpp_extension; print(cpp_extension.include_paths()[1])") \
	  -Wl,-rpath,$(shell python3 -c "from torch.utils import cpp_extension; print(' '.join(cpp_extension.library_paths()))") \
	  -L$(shell python3 -c "from torch.utils import cpp_extension; print(' '.join(cpp_extension.library_paths()))") -lgtest -L/usr/local/lib -ltorch -ltorch_cpu -ltorch_hip -lc10 -lc10_hip -lfmt -lcrypto -o pa_test.out

clean:
	rm -f pa_test.out pa_ragged_test.out pa_ragged.so