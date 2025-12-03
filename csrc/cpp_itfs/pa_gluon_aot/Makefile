pa_decode_gluon_aot:
	hipcc pa_decode_gluon_aot.cpp \
		-g3 -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-inline \
	    -o pa_decode_gluon_aot.out -std=c++17 \
	    -I/usr/local/lib/python3.10/dist-packages/torch/include \
	    -I/usr/local/lib/python3.10/dist-packages/torch/include/torch/csrc/api/include \
	    -L/usr/local/lib/python3.10/dist-packages/torch/lib \
	    -ltorch -ltorch_cpu -ltorch_hip -lc10 -lc10_hip -lfmt -lcrypto -ldl \
	    -Wl,-rpath,/usr/local/lib/python3.10/dist-packages/torch/lib

pa_decode_gluon_aot_lib.so:
	hipcc pa_decode_gluon_aot.cpp \
	    -shared -fPIC \
	    -o pa_decode_gluon_aot_lib.so -std=c++17 \
	    -I/opt/conda310/lib/python3.10/site-packages/torch/include \
	    -I/opt/conda310/lib/python3.10/site-packages/torch/include/torch/csrc/api/include \
	    -L/opt/conda310/lib/python3.10/site-packages/torch/lib \
	    -ltorch -ltorch_cpu -ltorch_hip -lc10 -lc10_hip -lfmt -lcrypto -ldl \
	    -Wl,-rpath,/opt/conda310/lib/python3.10/site-packages/torch/lib

