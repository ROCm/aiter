set -x

# shopt -s expand_aliases


# export HIP_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=1
# export HIP_VISIBLE_DEVICES=6
# export HIP_VISIBLE_DEVICES=7

export root_dir=/mnt/raid0/heyanguang/code


dump_asm() {
    jit_build_dir=/mnt/raid0/heyanguang/code/aiter/aiter/jit/build && /opt/rocm/bin/hipcc \
    -DWITH_HIP -DTORCH_EXTENSION_NAME=module_custom -DTORCH_API_INCLUDE_EXTENSION_H \
    -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" \
    -DPYBIND11_BUILD_ABI=\"_cxxabi1016\" -D_GLIBCXX_USE_CXX11_ABI=1 -I$jit_build_dir/ck/include \
    -I$jit_build_dir/ck/library/include -I$jit_build_dir/module_custom/build/include \
    -isystem /usr/local/lib/python3.12/dist-packages/torch/include -isystem \
    /usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include -isystem \
    /usr/local/lib/python3.12/dist-packages/torch/include/TH -isystem \
    /usr/local/lib/python3.12/dist-packages/torch/include/THC -isystem \
    /usr/local/lib/python3.12/dist-packages/torch/include/THH -isystem \
    /opt/rocm/include -isystem /usr/include/python3.12 -fPIC -std=c++17 \
    -O3 -std=c++17 -fPIC -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 \
    -DCUDA_HAS_FP16=1 -D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1 \
    -DLEGACY_HIPBLAS_DIRECT -DUSE_PROF_API=1 -D__HIP_PLATFORM_HCC__=1 \
    -D__HIP_PLATFORM_AMD__=1 -U__HIP_NO_HALF_CONVERSIONS__ -U__HIP_NO_HALF_OPERATORS__ \
    -mllvm --amdgpu-kernarg-preload-count=16 -Wno-unused-result -Wno-switch-bool \
    -Wno-vla-cxx-extension -Wno-undefined-func-template -Wno-macro-redefined \
    -fgpu-flush-denormals-to-zero -fno-offload-uniform-block \
    -mllvm -enable-post-misched=0 -mllvm -amdgpu-early-inline-all=true \
    -mllvm -amdgpu-function-calls=false -mllvm -amdgpu-coerce-illegal-types=1 \
    --offload-arch=native -fno-gpu-rdc \
    --cuda-device-only \
    -S $jit_build_dir/module_custom/build/srcs/custom_kernels.hip -o custom_kernels.cuda.modi.v2.s
    -S $jit_build_dir/module_custom/build/srcs/custom_kernels.hip -o custom_kernels.cuda.modi.v9.load_a_from_gm.no_async.s
    # -S $jit_build_dir/module_custom/build/srcs/custom_kernels.hip -o custom_kernels.cuda.modi.v9.no_async.s
}


pushd $root_dir/aiter/aiter/jit/build/module_custom/build
# dirs

# dump_asm
ninja && cp ./module_custom.so /mnt/raid0/heyanguang/code/aiter/aiter/jit
# cp ./*.s /mnt/raid0/heyanguang/code/aiter/log_run/

popd




# AITER_LOG_MORE=1 python3 ./op_tests/test_gemm.py > ./log_run/log.alm.x 2>&1
# python3 -u ./op_tests/test_gemm.py > ./log_run/log.2th.x 2>&1
# python3 -u ./op_tests/test_gemm.py > ./log_run/log.51.d6.all_test.wvSplitK_hf_sml_.ori 2>&1
# python3 -u ./op_tests/test_gemm.py > ./log_run/log.51.d6.all_test.wvSplitK_hf_sml_.v9.no_async 2>&1

# python3 -u ./op_tests/test_gemm_a8w8.py > ./log_run/log.quant.x 2>&1
python3 -u ./op_tests/test_gemm_a8w8.py > ./log_run/log.quant.2th.x 2>&1


# AITER_LOG_MORE=1 python3 ./op_tests/test_gemm.py > ./log_run/log.x 2>&1
# HIPCC_VERBOSE=1 python3 ./op_tests/test_gemm.py
# rocprof -d /home/yanguahe/heyanguang/code/aiter/res_rocprof --stats python3 ./op_tests/test_gemm.py
# rocprof --stats python3 ./op_tests/test_gemm.py
# rocprof -i ./prof.txt python3 ./op_tests/test_gemm.py


# python3 ./op_tests/test_gemm.py > ./log_run/log.x 2>&1
# python3 ./op_tests/test_gemm.py > ./log_run/log.d0.modi.v1 2>&1
# python3 ./op_tests/test_gemm.py > ./log_run/log.d0.modi.v2 2>&1


# pytest ./op_tests/triton_tests/test_mla_decode_rope.py::test_op_fwd_rope_integration -v -s

# pytest ./op_tests/triton_tests/test_mla_decode_rope.py::test_op_fwd_rope_integration -v -s -k "False-True-dtype1-8-128-2048-512-127-64"
# pytest ./op_tests/triton_tests/test_mla_decode_rope.py::test_op_fwd_rope_integration -v -s -k "False-True-dtype1-8-128-2050-512-127-64"
# pytest ./op_tests/triton_tests/test_mla_decode_rope.py::test_op_fwd_rope_integration -v -s -k "False-True-dtype1-8-128-2050-512-128-64"


# rocprofv2 -d $root_dir/ttv_dir/ -i $root_dir/att.txt --plugin att auto --mode file,csv python3 ./op_tests/test_gemm.py


set +x
