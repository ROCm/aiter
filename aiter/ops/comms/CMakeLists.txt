cmake_minimum_required(VERSION 3.22.1)

project(rms LANGUAGES CXX)

# -- CUDA Bypass --
set(USE_CUDA OFF CACHE BOOL "" FORCE)
set(uSE_CUDNN OFF CACHE BOOL "" FORCE)
set(AT_CUDA_ENABLED OFF)
set(AT_CUDNN_ENABLED OFF)

set(CAFFE2_USE_CUDA OFF CACHE BOOL "" FORCE)
set(CAFFE2_USE_CUDNN OFF CACHE BOOL "" FORCE)

set(USE_ROCM ON CACHE BOOL "" FORCE)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-std=c++20 -fopenmp -ffast-math")

set(CMAKE_CXX_COMPILER hipcc)

#set(pybind11_DIR
#        ~/.local/lib/python3.10/site-packages/pybind11/share/cmake/pybind11)

set(pybind11_DIR
	/opt/venv/lib/python3.10/site-packages/pybind11/share/cmake/pybind11)
set(Torch_DIR
	/opt/venv/lib/python3.10/site-packages/torch/share/cmake/Torch)

#set(pybind11_DIR
#	/opt/venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11)
#set(Torch_DIR
#	/opt/venv/lib/python3.12/site-packages/torch/share/cmake/Torch)

#set(CMAKE/INSTALL/PREFIX /opt/venu/lib/python3.12/site-packages)

find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)

pybind11_add_module(_rms py_rms_norm.cpp rmsnorm_kernel.cpp)

target_include_directories(_rms PRIVATE
	${Python_INCLUDE_DIRS}
	${pybind11_INCLUDE_DIRS}
	${TORCH_INCLUDE_DIRS})


target_link_directories(_rms PRIVATE
	${TORCH_INSTALL_PREFIX}/lib)

target_link_libraries(_rms PRIVATE
	"-Wl,--no-as-needed"
	torch
	torch_python
	"-Wl,--as-needed"
)

set_target_properties(_rms PROPERTIES
	BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
	INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)


install(TARGETS _rms LIBRARY DESTINATION rms)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py DESTINATION rms)