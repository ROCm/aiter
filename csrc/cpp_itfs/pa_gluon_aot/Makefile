AITER_ROOT := $(realpath $(dir $(lastword $(MAKEFILE_LIST)))/../../..)
TORCH_PATH := $(shell python3 -c "import torch,os; print(os.path.dirname(torch.__file__))")
PYBIND11_INC := $(shell python3 -m pybind11 --includes)
PYTHON_LDFLAGS := $(shell python3-config --ldflags --embed)

pa_decode_gluon_aot_lib.so:
	hipcc pa_decode_gluon_aot.cpp -o pa_decode_gluon_aot_lib.so -fPIC -shared -std=c++17 -O3 \
		-I$(AITER_ROOT)/csrc/cpp_itfs $(PYBIND11_INC) -I$(TORCH_PATH)/include -I$(TORCH_PATH)/include/torch/csrc/api/include \
		-L$(TORCH_PATH)/lib -ltorch_python -ltorch -ltorch_cpu -ltorch_hip -lc10 -lc10_hip -Wl,-rpath,$(TORCH_PATH)/lib \
		$(PYTHON_LDFLAGS) -lfmt -lcrypto -ldl -lpthread

test_pa_gluon: pa_decode_gluon_aot_lib.so
	hipcc pa_decode_gluon_aot_lib.so test_pa_gluon.cpp -std=c++17 -O0 -g -DTEST_REPO_ROOT=\"$(AITER_ROOT)\" \
		-I$(AITER_ROOT)/csrc/cpp_itfs $(PYBIND11_INC) -I$(TORCH_PATH)/include -I$(TORCH_PATH)/include/torch/csrc/api/include \
		-L$(TORCH_PATH)/lib -ltorch_python -ltorch -ltorch_cpu -ltorch_hip -lc10 -lc10_hip -Wl,-rpath,$(TORCH_PATH)/lib \
		$(PYTHON_LDFLAGS) -lfmt -lcrypto -ldl -lpthread -o test_pa_gluon

run: test_pa_gluon
	PYTHONPATH=$(AITER_ROOT):$$PYTHONPATH ./test_pa_gluon

clean:
	rm -f test_pa_gluon pa_decode_gluon_aot_lib.so
