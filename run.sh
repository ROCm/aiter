set -x

shopt -s expand_aliases

alias l.='ls -d .* --color=auto'
alias ll='ls -l --color=auto'
alias ls='ls --color=auto'

# export HIP_VISIBLE_DEVICES=0
# export HIP_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=3
# export HIP_VISIBLE_DEVICES=4
# export HIP_VISIBLE_DEVICES=6
# export HIP_VISIBLE_DEVICES=7


# export LD_LIBRARY_PATH=/mnt/raid0/heyanguang/code/poc_kl/scripts/common:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH
# export PATH=/mnt/raid0/heyanguang/code/poc_kl/scripts/common:$PATH

rocm-smi | egrep "$HIP_VISIBLE_DEVICES    |Device"


function run_aiter_op {
    export AITER_LOG_MORE=1

    python ./op_tests/triton_tests/test_paged_attention_decode_gluon.py --quant_mode per_token -b 128 -q 1 --trans_v
    # python ./op_tests/triton_tests/test_paged_attention_decode_gluon.py --quant_mode per_tensor -b 128 -q 1 --trans_v
    # python ./op_tests/triton_tests/test_paged_attention_decode_gluon.py -b 128 -q 1 --trans_v

    # python ./op_tests/triton_tests/test_paged_attention_decode_gluon.py --quant_mode per_tensor
    # python ./op_tests/triton_tests/test_paged_attention_decode_gluon.py --quant_mode per_tensor --trans_v

    # python ./op_tests/triton_tests/test_paged_attention_decode_gluon.py
    # python ./op_tests/triton_tests/test_paged_attention_decode_gluon.py --trans_v

}


function get_triton_pa_thread_trace {
    rm -rf ~/.triton/cache
    pushd $PWD
    export AITER_LOG_MORE=1

    # KERNEL=_fwd_kernel
    # KERNEL=_attn_fwd
    # KERNEL=_fwd_kernel_stage2
    # KERNEL=matmul_ori_kernel
    # KERNEL=matmul_ori_kernel_v2
    # KERNEL=_triton_mixed_sparse_attn_fwd_kernel_v1
    # KERNEL=_triton_block_sparse_attn_fwd_kernel_v1
    # KERNEL=pa_decode_v2_gluon_fp8
    KERNEL=pa_decode_v2_gluon_big_blk_fp8
    # KERNEL=pa_bf16_pertokenFp8_gqa8_2tg_4w_uhp
    # KERNEL=matmul_kernel
    # KERNEL=_fwd_grouped_kernel_stage1_rope
    # export KERNEL_VERSION="triton_pa_prefill_bf16"
    # export KERNEL_VERSION="triton_mha_fwd_bf16"
    # export KERNEL_VERSION="triton_${KERNEL}_bf16"
    # export KERNEL_VERSION="triton_${KERNEL}_bf16_v2"

    export KERNEL_VERSION="${KERNEL}_v1"
    # export KERNEL_VERSION="${KERNEL}_v1_rm_v_scale"

    # pytest ./test_pa_prefill.py::test_contexted_kv_attention -v -s -k "0-cuda:0-auto-dtype1-128-1-4"
    # pytest ./test_pa_prefill.py::test_mha -v -s -k "False-True-0.0-False-False-128-4-4-1024-2048-2"
    # pytest ./test_pa_prefill.py::test_mha -v -s -k "True-False-0.0-False-False-128-16-1-1024-1024-80"

    # python $aiter_root_dir/op_tests/op_benchmarks/triton/bench_mla_decode.py --model all --seqlen 16384
    # python $aiter_root_dir/op_tests/op_benchmarks/triton/bench_mla_decode.py -b 320 --model all --seqlen 8192
    # python $aiter_root_dir/op_tests/op_benchmarks/triton/bench_mla_decode.py -b 80 --model all --seqlen 8192 -equal_seqlens -print_vgpr
    # python $aiter_root_dir/op_tests/op_benchmarks/triton/bench_mla_decode.py -b 80 --model all --seqlen 8192 -equal_seqlens


    DUMP_TRACE=1
    # DUMP_TRACE=0
    if [ $DUMP_TRACE = 1 ]; then
        trace_dir=./thread_trace/trace_${KERNEL_VERSION}
        rm -rf ./${trace_dir} ./${trace_dir}.tar.gz
        mkdir -p ${trace_dir}

        rocprofv2 -d ${trace_dir} -i ./thread_trace/att.txt --plugin att auto --mode file,csv -o ${trace_dir}/csv_${KERNEL_VERSION} \
        python ./test_pa_mtp.py -n 16,1 -q 1 -c 4096 -b 128 --block_size 1024
        # python ./test_pa_mtp.py -n 16,1 -q 1 -c 4096 -b 128 --block_size 16
        # python ./test_pa_mtp.py -n 8,1 -q 1 -c 4096 -b 80 --block_size 16
        # python ./block_sparse_attn.py
        # python ./mixed_sparse_attn.py
        # python ./00-gemm.py
        # python $aiter_root_dir/op_tests/op_benchmarks/triton/bench_mla_decode.py -b 80 --model all --seqlen 8192 -equal_seqlens
        # pytest ./test_pa_prefill.py::test_mha -v -s -k "False-True-0.0-False-False-128-4-4-1024-2048-2"
        # pytest ./test_pa_prefill.py::test_contexted_kv_attention -v -s -k "0-cuda:0-auto-dtype1-128-1-4"

        cd ./thread_trace
        tar -zcf ./trace_${KERNEL_VERSION}.tar.gz ./trace_${KERNEL_VERSION}
        ls -lah ./trace_${KERNEL_VERSION} ./trace_${KERNEL_VERSION}.tar.gz
        cd -
    fi

    copy_recent_amdgcn_files
    popd
}


# pip install -e .

# install aiter
# python3 setup.py develop

# install triton
# pip install -e python
# pip install -e .


# for i in {1..1..1}; do
# # for i in {1..5..1}; do
# # for i in {1..20..1}; do
#     echo "*******************************iter=$i*******************************"

# done


run_aiter_op
# get_triton_pa_thread_trace


# cat log | egrep "diff.abs.max|max_diff_thr|out_ref_md5|gluon_fp8_output_md5" > out34
# cat log | egrep "perf_fp8_gluon_vs_asm" -A 16 >> out34
# cat log | egrep "paged_attention_decode_v2_gluon_|paged_attention_decode_v2_reduce_kernel" -A 1 >> out34
# cat logtn3.5 | egrep "diff.abs.max|max_diff_thr|out_ref_md5|gluon_fp8_output_md5" > out35
# cat logtn3.5 | egrep "perf_fp8_gluon_vs_asm" -A 16 >> out35
# cat logtn3.5 | egrep "paged_attention_decode_v2_gluon_|paged_attention_decode_v2_reduce_kernel" -A 1 >> out35


# rocprofv2 --version
# cat /etc/os-release


set +x
