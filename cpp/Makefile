test:
	hipcc pa.cpp pa.h pa_test.cpp -O3 -I$(shell python3 -c "from torch.utils import cpp_extension; print(cpp_extension.include_paths()[0])") \
	  -I$(shell python3 -c "from torch.utils import cpp_extension; print(cpp_extension.include_paths()[1])") \
	  -L$(shell python3 -c "from torch.utils import cpp_extension; print(' '.join(cpp_extension.library_paths()))") -lgtest -L/usr/local/lib -ltorch -ltorch_cpu -ltorch_hip -lc10 -lc10_hip -lfmt -o test

clean:
	rm -f test
