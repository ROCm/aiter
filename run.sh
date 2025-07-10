set -x

# shopt -s expand_aliases

# export CK_LOGGING=1

# export HIP_VISIBLE_DEVICES=1
# export HIP_VISIBLE_DEVICES=2
# export HIP_VISIBLE_DEVICES=3
export HIP_VISIBLE_DEVICES=6
# export HIP_VISIBLE_DEVICES=7

export root_dir=/mnt/raid0/heyanguang/code/vllm_fa_batch_prefill/aiter


# KERNEL_TYPE=mha_batch_prefill_fp16_logits_nbias_mask_nlse_ndropout_vllm_chunked
# KERNEL_TYPE=mha_batch_prefill_bf16_logits_nbias_mask_nlse_ndropout_sglang_chunked
KERNEL_TYPE=mha_batch_prefill_bf16_logits_nbias_mask_nlse_ndropout_vllm_chunked

export kernel_version=bf16_sglang
# export kernel_version=bf16_vllm

export kernel_name=ck_tile

export trace_file_dir=ttv_$kernel_version
export csv_file_name=$kernel_version
export generated_csv_file_name=${kernel_version}_${kernel_name}_v0.csv




function update_so {
    rm ./aiter/jit/mha_batch_prefill_*so
    # rm /mnt/raid0/heyanguang/code/vllm_fa_batch_prefill/aiter/aiter/jit/build/mha_batch_prefill*/build/fmha_batch_prefill_d*

    BASE_DIR="/mnt/raid0/heyanguang/code/vllm_fa_batch_prefill/aiter/aiter/jit/build"
    # for dir in ${BASE_DIR}/mha_batch_prefill*/build/; do
    # for dir in ${BASE_DIR}/mha_batch_prefill_fp16*/build/; do
    for dir in ${BASE_DIR}/mha_batch_prefill_bf16*/build/; do
    # for dir in ${BASE_DIR}/mha_batch_prefill_bf16*vllm*/build/; do
    # for dir in ${BASE_DIR}/mha_batch_prefill_bf16*sglang*/build/; do
        if [ -d "$dir" ]; then
            pushd $dir
            rm -f ./fmha_batch_prefill_d* && ninja && cp *.so /mnt/raid0/heyanguang/code/vllm_fa_batch_prefill/aiter/aiter/jit/
            # ninja && cp *.so /mnt/raid0/heyanguang/code/vllm_fa_batch_prefill/aiter/aiter/jit/
            popd
        else
            echo "$dir is not exist!"
        fi
    done
}

function dump_asm {
    BASE_DIR="/mnt/raid0/heyanguang/code/vllm_fa_batch_prefill/aiter/aiter/jit/build"

    dump_asm_prefix="/opt/rocm/bin/hipcc  -DWITH_HIP -DTORCH_EXTENSION_NAME=$KERNEL_TYPE \
    -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" \
    -DPYBIND11_BUILD_ABI=\"_cxxabi1018\" -D_GLIBCXX_USE_CXX11_ABI=1 \
    -I/mnt/raid0/heyanguang/code/vllm_fa_batch_prefill/aiter/aiter/jit/build/ck/include \
    -I/mnt/raid0/heyanguang/code/vllm_fa_batch_prefill/aiter/aiter/jit/build/ck/library/include \
    -I/mnt/raid0/heyanguang/code/vllm_fa_batch_prefill/aiter/aiter/jit/build/$KERNEL_TYPE/build/include \
    -isystem /opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/include \
    -isystem /opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/include/torch/csrc/api/include \
    -isystem /opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/include/TH \
    -isystem /opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/include/THC \
    -isystem /opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/include/THH \
    -isystem /opt/rocm/include -isystem /opt/conda/envs/py_3.12/include/python3.12 \
    -fPIC -std=c++17 -O3 -std=c++17 -fPIC -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -DCUDA_HAS_FP16=1 \
    -D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1 -DLEGACY_HIPBLAS_DIRECT -DUSE_PROF_API=1 \
    -D__HIP_PLATFORM_HCC__=1 -D__HIP_PLATFORM_AMD__=1 -U__HIP_NO_HALF_CONVERSIONS__ -U__HIP_NO_HALF_OPERATORS__ \
    -mllvm --amdgpu-kernarg-preload-count=16 -Wno-unused-result -Wno-switch-bool -Wno-vla-cxx-extension -Wno-undefined-func-template \
    -Wno-macro-redefined -fgpu-flush-denormals-to-zero -fno-offload-uniform-block -mllvm -enable-post-misched=0 \
    -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -mllvm -amdgpu-coerce-illegal-types=1 \
    -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=2 -DCK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT=0 \
    --offload-arch=native -fno-gpu-rdc \
    --cuda-device-only"
    for dir in ${BASE_DIR}/$KERNEL_TYPE/build/; do
        if [ -d "$dir" ]; then
            pushd $dir
            hip_file=./srcs/fmha_batch_prefill_d*.hip
            # filename=$(basename "fmha_batch_prefill_d*" .cuda.o)
            # ls -alh $filename
            # $dump_asm_prefix -S $hip_file -o $root_dir/log_run/mha_batch_prefill_bf16_sglang_v2.s
            # $dump_asm_prefix -S $hip_file -o $root_dir/log_run/mha_batch_prefill_bf16_vllm_v2.s
            # $dump_asm_prefix -S $hip_file -o $root_dir/log_run/mha_batch_prefill_bf16_vllm_v4.s
            # $dump_asm_prefix -S $hip_file -o $root_dir/log_run/mha_batch_prefill_bf16_vllm_v4_subv2_page_16.s
            $dump_asm_prefix -S $hip_file -o $root_dir/log_run/mha_batch_prefill_bf16_vllm_v4_subv3_page_16.s
            # $dump_asm_prefix -S $hip_file -o $root_dir/log_run/mha_batch_prefill_bf16_vllm_v5.s
            # $dump_asm_prefix -S $hip_file -o $root_dir/log_run/mha_batch_prefill_bf16_vllm_v5_page_16.s
            # $dump_asm_prefix -S $hip_file -o $root_dir/log_run/mha_batch_prefill_bf16_vllm_v5_subv2_page_16.s
            # $dump_asm_prefix -S $hip_file -o $root_dir/log_run/${KERNEL_TYPE}.s
            # $dump_asm_prefix -S $hip_file -o $root_dir/log_run/$trace_file_dir/mha_bp_${kernel_version}.s
            popd
        else
            echo "$dir is not exist!"
        fi
    done
}

function get_thread_trace {
    # rocprofv3 -i input_att.yaml -- python3 -u ./op_tests/test_gemm.py
    # rocprofv3 -d $root_dir/log_run/$trace_file_dir -i $root_dir/log_run/input_att.yaml -o $root_dir/log_run/$csv_file_name -- \

    # rm -rf ./log_run/$trace_file_dir ./log_run/$trace_file_dir.tar.gz ./log_run/${kernel_version}_*.csv
    # rocprofv2 -d $root_dir/log_run/$trace_file_dir -i $root_dir/log_run/att.txt --plugin att auto --mode file,csv -o $root_dir/log_run/$csv_file_name \
    pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache -v -s -k "19378--5-5--10-10-dtype1-True-30.0-NHD-True-True-128-3-1-1-2048-2048-1-True"
    # pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache -v -s -k "19378--5-5--10-10-dtype1-True-30.0-NHD-True-True-128-3-1-1-2048-2048-1-False"
    # dump_asm

    # pushd ./log_run
    # python3 /mnt/raid0/heyanguang/code/cpp_code/generate_sp3_from_asm.py \
    #     --asm_file $root_dir/log_run/$trace_file_dir/mha_bp_${kernel_version}.s \
    #     --kernel_txt ./$trace_file_dir/kentry_v0_kernel.txt \
    #     --out_sp3_file ./$trace_file_dir/mha_bp_${kernel_version}.sp3
    # tar -zcf ./$trace_file_dir.tar.gz ./$trace_file_dir
    # ls -lah ./$trace_file_dir ./$trace_file_dir.tar.gz ./${kernel_version}_*.csv
    # popd
}

# update_so
# dump_asm
# get_thread_trace

# AITER_LOG_MORE=1 pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache -v -s -k "19378--5-5--10-10-dtype1-False-30.0-NHD-True-True-128-3-1-1-2048-2048-7"



# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache -v -s -k "19378--5-5--10-10-dtype1-False-30.0-NHD-True-True-128-3-1-1-2048-2048-1-True"
# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache -v -s -k "19378--5-5--10-10-dtype1-False-30.0-NHD-True-True-128-3-1-1-2048-2048-7-True"
# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache -v -s -k "19378--5-5--10-10-dtype1-False-30.0-NHD-True-True-128-3-1-1-1023-1025-1-True"
# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache -v -s -k "19378--5-5--10-10-dtype1-False-30.0-NHD-True-True-128-3-1-1-1023-1025-7-True"

# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache -v -s -k "19378--5-5--10-10-dtype1-False-30.0-NHD-True-True-128-3-1-16-2048-2048-1-False"
# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache -v -s -k "19378--5-5--10-10-dtype1-False-30.0-NHD-True-True-128-3-1-16-2048-2048-7-False"
# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache -v -s -k "19378--5-5--10-10-dtype1-False-30.0-NHD-True-True-128-3-1-16-1023-1025-1-False"
# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache -v -s -k "19378--5-5--10-10-dtype1-False-30.0-NHD-True-True-128-3-1-16-1023-1025-7-False"



# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache -v -s -k "19378--5-5--10-10-dtype1-True-30.0-NHD-True-True-128-3-1-1-2048-2048-1-True"
# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache -v -s -k "19378--5-5--10-10-dtype1-True-30.0-NHD-True-True-128-3-1-1-2048-2048-7-True"
# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache -v -s -k "19378--5-5--10-10-dtype1-True-30.0-NHD-True-True-128-3-1-1-1023-1025-1-True"
# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache -v -s -k "19378--5-5--10-10-dtype1-True-30.0-NHD-True-True-128-3-1-1-1023-1025-7-True"

# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache -v -s -k "19378--5-5--10-10-dtype1-True-30.0-NHD-True-True-128-3-1-16-2048-2048-1-False"
# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache -v -s -k "19378--5-5--10-10-dtype1-True-30.0-NHD-True-True-128-3-1-16-2048-2048-7-False"
# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache -v -s -k "19378--5-5--10-10-dtype1-True-30.0-NHD-True-True-128-3-1-16-1023-1025-1-False"
# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache -v -s -k "19378--5-5--10-10-dtype1-True-30.0-NHD-True-True-128-3-1-16-1023-1025-7-False"




export AITER_LOG_MORE=1

# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache_v2 -v -s -k "19378--5-5--10-10-dtype0-True-0.0-NHD-True-True-128-32-8-16-1024-1024-1-False"
# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache_v2 -v -s -k "19378--5-5--10-10-dtype0-True-0.0-NHD-True-True-128-32-8-16-8192-8192-1-False"


# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache_v2 -v -s -k "19378--5-5--10-10-dtype0-True-0.0-NHD-True-True-128-32-8-16-1024-1024-1-False"
# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache_v2 -v -s -k "19378--5-5--10-10-dtype0-True-0.0-NHD-True-True-128-3-1-16-2048-2048-7-False"
# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache_v2 -v -s -k "19378--5-5--10-10-dtype0-False-0.0-NHD-True-True-128-3-1-16-2048-2048-7-False"
# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache_v2 -v -s -k "19378--5-5--10-10-dtype0-True-0.0-NHD-True-True-128-3-1-1-2048-2048-7-True"
# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache_v2 -v -s -k "19378--5-5--10-10-dtype0-False-0.0-NHD-True-True-128-3-1-1-2048-2048-7-True"

# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache_v2 -v -s -k "19378--5-5--10-10-dtype0-True-0.0-NHD-True-True-16-False-True"
# pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache_v2 -v -s -k "19378--5-5--10-10-dtype0-True-0.0-NHD-True-True-16-False-False"

# (pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache_v2 -v -s -k "19378--5-5--10-10-dtype0-True-0.0-NHD-True-True-16-False-True-1" > ./log_run/logx 2>&1 && cat ./log_run/logx | egrep "paged_kv_cache_v2\[|avg_kt_base|PASSED|dump_dir|_ZN7ck_tile|_vllm_layout") > ./log_run/log_vx

# (pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache_v2 -v -s -k "19378--5-5--10-10-dtype0-True-0.0-NHD-True-True-16-False-True" > ./log_run/log1 2>&1 && cat ./log_run/log1 | egrep "paged_kv_cache_v2\[|avg_kt_base|PASSED|dump_dir|_ZN7ck_tile|_vllm_layout") > ./log_run/op_mi300_case_list1_ktprof_perf_ori.txt
# (pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache_v2 -v -s -k "19378--5-5--10-10-dtype0-True-0.0-NHD-True-True-16-False-False" > ./log_run/log1 2>&1 && cat ./log_run/log1 | egrep "paged_kv_cache_v2\[|avg_kt_base|PASSED|dump_dir|_ZN7ck_tile|_vllm_layout") > ./log_run/op_mi300_case_list1_ktprof_perf_v1.txt
# (pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache_v2 -v -s -k "19378--5-5--10-10-dtype0-True-0.0-NHD-True-True-16-False-True" > ./log_run/log1 2>&1 && cat ./log_run/log1 | egrep "paged_kv_cache_v2\[|avg_kt_base|PASSED|dump_dir|_ZN7ck_tile|_vllm_layout") > ./log_run/op_mi300_case_list1_perf_ori.txt
# (pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache_v2 -v -s -k "19378--5-5--10-10-dtype0-True-0.0-NHD-True-True-16-False-False" > ./log_run/log1 2>&1 && cat ./log_run/log1 | egrep "paged_kv_cache_v2\[|avg_kt_base|PASSED|dump_dir|_ZN7ck_tile|_vllm_layout") > ./log_run/op_mi300_case_list1_perf_v1.txt

(pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache_v2 -v -s -k "19378--5-5--10-10-dtype0-True-0.0-NHD-True-True-16-False-True" > ./log_run/log1 2>&1 && cat ./log_run/log1 | egrep "paged_kv_cache_v2\[|avg_kt_base|PASSED|dump_dir|_ZN7ck_tile|_vllm_layout") > ./log_run/op_mi308_case_list1_ktprof_perf_ori.txt
(pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache_v2 -v -s -k "19378--5-5--10-10-dtype0-True-0.0-NHD-True-True-16-False-False" > ./log_run/log1 2>&1 && cat ./log_run/log1 | egrep "paged_kv_cache_v2\[|avg_kt_base|PASSED|dump_dir|_ZN7ck_tile|_vllm_layout") > ./log_run/op_mi308_case_list1_ktprof_perf_v1.txt
# (pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache_v2 -v -s -k "19378--5-5--10-10-dtype0-True-0.0-NHD-True-True-16-False-True" > ./log_run/log1 2>&1 && cat ./log_run/log1 | egrep "paged_kv_cache_v2\[|avg_kt_base|PASSED|dump_dir|_ZN7ck_tile|_vllm_layout") > ./log_run/op_mi308_case_list1_perf_ori.txt
# (pytest ./op_tests/test_batch_prefill.py::test_batch_prefill_with_paged_kv_cache_v2 -v -s -k "19378--5-5--10-10-dtype0-True-0.0-NHD-True-True-16-False-False" > ./log_run/log1 2>&1 && cat ./log_run/log1 | egrep "paged_kv_cache_v2\[|avg_kt_base|PASSED|dump_dir|_ZN7ck_tile|_vllm_layout") > ./log_run/op_mi308_case_list1_perf_v1.txt

# # python3 -u ./op_tests/test_batch_prefill.py
# (python3 -u ./op_tests/test_batch_prefill.py --use_ori_vllm_impl > ./log_run/log2 2>&1 && cat ./log_run/log2 | egrep "paged_kv_cache_v2\[|avg_kt_base|PASSED|dump_dir|_ZN7ck_tile|_vllm_layout") > ./log_run/op_mi300_case_list2_test_shape_ori.txt
# (python3 -u ./op_tests/test_batch_prefill.py --use_ori_qkv_shape --use_ori_vllm_impl > ./log_run/log2 2>&1 && cat ./log_run/log2 | egrep "paged_kv_cache_v2\[|avg_kt_base|PASSED|dump_dir|_ZN7ck_tile|_vllm_layout") > ./log_run/op_mi300_case_list2_real_shape_ori.txt
# (python3 -u ./op_tests/test_batch_prefill.py > ./log_run/log2 2>&1 && cat ./log_run/log2 | egrep "paged_kv_cache_v2\[|avg_kt_base|PASSED|dump_dir|_ZN7ck_tile|_vllm_layout") > ./log_run/op_mi300_case_list2_test_shape_v1.txt
# (python3 -u ./op_tests/test_batch_prefill.py --use_ori_qkv_shape > ./log_run/log2 2>&1 && cat ./log_run/log2 | egrep "paged_kv_cache_v2\[|avg_kt_base|PASSED|dump_dir|_ZN7ck_tile|_vllm_layout") > ./log_run/op_mi300_case_list2_real_shape_v1.txt

# (python3 -u ./op_tests/test_batch_prefill.py --use_ori_vllm_impl > ./log_run/log2 2>&1 && cat ./log_run/log2 | egrep "paged_kv_cache_v2\[|avg_kt_base|PASSED|dump_dir|_ZN7ck_tile|_vllm_layout") > ./log_run/op_mi308_case_list2_test_shape_ori.txt
# (python3 -u ./op_tests/test_batch_prefill.py --use_ori_qkv_shape --use_ori_vllm_impl > ./log_run/log2 2>&1 && cat ./log_run/log2 | egrep "paged_kv_cache_v2\[|avg_kt_base|PASSED|dump_dir|_ZN7ck_tile|_vllm_layout") > ./log_run/op_mi308_case_list2_real_shape_ori.txt
# (python3 -u ./op_tests/test_batch_prefill.py > ./log_run/log2 2>&1 && cat ./log_run/log2 | egrep "paged_kv_cache_v2\[|avg_kt_base|PASSED|dump_dir|_ZN7ck_tile|_vllm_layout") > ./log_run/op_mi308_case_list2_test_shape_v1.txt
# (python3 -u ./op_tests/test_batch_prefill.py --use_ori_qkv_shape > ./log_run/log2 2>&1 && cat ./log_run/log2 | egrep "paged_kv_cache_v2\[|avg_kt_base|PASSED|dump_dir|_ZN7ck_tile|_vllm_layout") > ./log_run/op_mi308_case_list2_real_shape_v1.txt

set +x
