set -x

shopt -s expand_aliases

alias l.='ls -d .* --color=auto'
alias ll='ls -l --color=auto'
alias ls='ls --color=auto'
alias python='python3'

# export HIP_VISIBLE_DEVICES=0
# export HIP_VISIBLE_DEVICES=1
# export HIP_VISIBLE_DEVICES=3
# export HIP_VISIBLE_DEVICES=5
export HIP_VISIBLE_DEVICES=6
# export HIP_VISIBLE_DEVICES=7


# export LD_LIBRARY_PATH=/mnt/raid0/heyanguang/code/poc_kl/scripts/common:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH
# export PATH=/mnt/raid0/heyanguang/code/poc_kl/scripts/common:$PATH

rocm-smi | egrep "$HIP_VISIBLE_DEVICES    |Device"
pip show triton
rocprofv3 --version
# triton_cache_dir=~/.triton/cache
# triton_cache_dir=/mnt/raid0/heyanguang/code/fa_triton/aiter/triton_cache1
triton_cache_dir=/mnt/raid0/heyanguang/code/fa_triton/aiter/triton_cache2
export TRITON_CACHE_DIR=${triton_cache_dir}


function copy_recent_amdgcn_files() {
    # dir_name=pa_decode_v2_fp8
    dir_name=pa_decode_v2_gluon_fp8
    # dir_name=pa_decode_v2_gluon_fp8_tn3.5
    # local k=2
    local k=200
    # local dest_dir=$PWD/thread_trace/triton_gen_asm
    local dest_dir=$PWD/thread_trace/triton_gen_asm/$dir_name
    # rm -rf $dest_dir
    # mkdir -p $dest_dir

    # kernel_name=pa_decode_v2_gluon_
    kernel_name=paged_attention_decode_v2_gluon_
    # kernel_name=paged_attention_decode_v2_
    # kernel_name=transpose_*_gluon_kernel

    file_filter="*$kernel_name*"

    amdgcn_filter=${triton_cache_dir}/*/*$kernel_name*.amdgcn
    json_filter=${triton_cache_dir}/*/$kernel_name*.json
    cat $json_filter
    cat $amdgcn_filter | egrep ".sgpr_count|.sgpr_spill_count|.vgpr_count|.vgpr_spill_count|SGPRBlocks|VGPRBlocks|Occupancy"
}


function run_aiter_op {
    rm -rf ${triton_cache_dir}
    # export AITER_LOG_MORE=2
    # export TRITON_INTERPRET=1

    export AITER_LOG_MORE=1
    export MLIR_ASM_VERBOSE=1
    export FLIR_LOG_MORE=1
    export FLIR_DUMP_IR=1
    export FLIR_REBUILD=1
    export FLIR_DUMP_DIR=./flydsl_dump

    # python ./op_tests/triton_tests/test_paged_attention_decode_gluon.py --quant_mode per_token -b 128 -q 1 --trans_v
    # python ./op_tests/triton_tests/test_paged_attention_decode_gluon.py --quant_mode per_tensor -b 128 -q 1 --trans_v
    # python ./op_tests/triton_tests/test_paged_attention_decode_gluon.py -b 4 -q 4 --trans_v
    # python ./op_tests/triton_tests/test_paged_attention_decode_gluon.py -b 128 -q 1 -c 4096 --trans_v 1 --kv_varlen 0
    # python ./op_tests/triton_tests/test_paged_attention_decode_gluon.py -b 128 -q 1 --trans_v 1
    # python ./op_tests/triton_tests/test_paged_attention_decode_gluon.py
    # pytest ./op_tests/triton_tests/test_paged_attention_decode_gluon.py::test_pa_gluon -v -s

    # python ./csrc/cpp_itfs/mla_modi/asm_mla_decode_fwd_test.py

    # python ./op_tests/triton_tests/test_pa_decode_gluon.py -b 128 -q 1 -c 8192 -n 64,8 -d 128 --block_size 16 --compute_type bf16 --quant_mode per_tensor --trans_v false --kv_varlen false --use_aot_impl false --quant_q_and_kv 0,0 --context_partition_size 256

    # python ./op_tests/triton_tests/test_pa_decode_gluon.py -b 64 -q 1 -c 2048 -n 64,8 -d 128 --block_size 64 --compute_type bf16 --quant_mode per_tensor --trans_v false --kv_varlen false --use_aot_impl false --quant_q_and_kv 1,1 --context_partition_size 256
    # python ./op_tests/triton_tests/test_pa_decode_gluon.py -b 64 -q 1 -c 2048 -n 64,8 -d 128 --block_size 16 --compute_type bf16 --quant_mode per_tensor --trans_v false --kv_varlen false --use_aot_impl false --quant_q_and_kv 1,1 --context_partition_size 256
    # python ./op_tests/triton_tests/test_pa_decode_gluon.py -b 64 -q 1 -c 2048 -n 64,8 -d 128 --block_size 16 --compute_type fp8 --quant_mode per_tensor --trans_v false --kv_varlen false --use_aot_impl false --quant_q_and_kv 1,1 --context_partition_size 256
    # python ./op_tests/triton_tests/test_pa_decode_gluon.py -b 64 -q 1 -c 2048 -n 64,8 -d 128 --block_size 16 --compute_type fp8 --quant_mode per_token --trans_v false --kv_varlen false --use_aot_impl false --quant_q_and_kv 1,1 --context_partition_size 256


    # rm -rf aiter/jit/build/ aiter/jit/*.so
    # rm -rf ~/.aiter/build/
    # rm -rf ~/.aiter/build/ && python ./csrc/cpp_itfs/pa_gluon_aot/pa_decode_gluon_aot_prebuild.py
    # python ./csrc/cpp_itfs/pa_gluon_aot/pa_decode_gluon_aot_prebuild.py

    # python ./op_tests/triton_tests/test_pa_decode_gluon.py
    # python ./op_tests/triton_tests/test_pa_decode_gluon.py --use_aot_impl true
    # python ./op_tests/triton_tests/test_pa_decode_gluon.py --use_aot_impl false

    # pytest -v op_tests/triton_tests/test_pa_decode_gluon.py
    # pytest -v -s op_tests/triton_tests/test_pa_decode_gluon.py
    # pytest -v op_tests/triton_tests/test_pa_decode_gluon.py -k normal_accuracy_aot

    # rm -r ~/.aiter/build/ && python ./csrc/cpp_itfs/pa_gluon_aot/pa_decode_gluon_aot_prebuild.py --num_processes 1
    # python ./csrc/cpp_itfs/pa_gluon_aot/pa_decode_gluon_aot_prebuild.py --num_processes 1 -b 3 -q 4 -c 32768 -n 10,1 --block_size 64 --compute_type fp16 --quant_mode per_token --trans_v false --kv_varlen false --use_aot_impl true --quant_q_and_kv 1,1

    # rm -r ~/.aiter/build/pa_decode_reduce_kernel* ~/.aiter/build/pa_decode_attention_kernel* ./pa_decode_reduce_kernel.* ./pa_decode_attention_kernel.*
    # python ./csrc/cpp_itfs/pa_gluon_aot/pa_attention_kernel_test.py --kernel-type direct
    # python ./csrc/cpp_itfs/pa_gluon_aot/pa_attention_kernel_test.py --kernel-type compiled
    # python ./csrc/cpp_itfs/pa_gluon_aot/pa_reduce_kernel_test.py --kernel-type direct
    # python ./csrc/cpp_itfs/pa_gluon_aot/pa_reduce_kernel_test.py --kernel-type compiled

    # python ./op_tests/triton_tests/test_pa_decode_gluon.py -q 1

    # python ./op_tests/triton_tests/test_pa_decode_gluon.py -b 64 -q 1 -c 2048 -n 64,8 -d 128 --block_size 64 --compute_type bf16 --quant_mode per_tensor --trans_v false --kv_varlen false --use_aot_impl false --quant_q_and_kv 0,0 --context_partition_size 256
    # python ./op_tests/triton_tests/test_pa_decode_gluon.py -b 64 -q 4 -c 2048 -n 64,8 -d 128 --block_size 64 --compute_type bf16 --quant_mode per_tensor --trans_v false --kv_varlen false --use_aot_impl false --quant_q_and_kv 0,0 --context_partition_size 256



    # ./amdgcn_edit/test_list.sh


    # python amdgcn_edit/amdgcn_ddg.py amdgcn_edit/pa_dot_kernel.v1.amdgcn -o amdgcn_edit/ddg_output
    # python amdgcn_edit/amdgcn_ddg.py amdgcn_edit/pa_dot_kernel.v1.amdgcn -o amdgcn_edit/ddg_output --stats --inter-deps --waitcnt-deps
    # python amdgcn_edit/amdgcn_ddg.py amdgcn_edit/pa_dot_kernel.v2.amdgcn -o amdgcn_edit/ddg_output --stats --inter-deps --waitcnt-deps --save-json
    # python amdgcn_edit/amdgcn_ddg.py amdgcn_edit/pa_dot_kernel.v2.amdgcn -o amdgcn_edit/ddg_trans_out_v1 --stats --inter-deps --waitcnt-deps --save-json
    # python amdgcn_edit/amdgcn_ddg.py --load-json ./amdgcn_edit/ddg_output/analysis.json -o amdgcn_edit/ddg_output
    # python amdgcn_edit/amdgcn_ddg.py --load-json ./amdgcn_edit/ddg_output/analysis.json -o amdgcn_edit/ddg_output --stats --inter-deps --waitcnt-deps
    # python amdgcn_edit/amdgcn_ddg.py --load-json ./amdgcn_edit/ddg_output/analysis.json -r amdgcn_edit/ddg_output/modi_v0.amdgcn --keep-debug-labels
    # python amdgcn_edit/amdgcn_ddg.py --load-json ./amdgcn_edit/ddg_output/analysis.json -r amdgcn_edit/ddg_output/modi_v0.amdgcn
    # python amdgcn_edit/amdgcn_ddg.py --load-json ./amdgcn_edit/ddg_output/analysis.json --move .LBB0_2 39 -1 -r amdgcn_edit/ddg_output/modi_v0.amdgcn
    # python amdgcn_edit/amdgcn_ddg.py --load-json ./amdgcn_edit/ddg_output/analysis.json --distribute .LBB0_2 global_load_dwordx4 16 -r amdgcn_edit/ddg_output/modi_v0.amdgcn
    # python amdgcn_edit/amdgcn_ddg.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json -r amdgcn_edit/ddg_trans_out_v1/modi_v0.amdgcn
    # python amdgcn_edit/amdgcn_ddg.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json --distribute .LBB0_2 global_load_dwordx4 16 -r amdgcn_edit/ddg_trans_out_v1/modi_v0.amdgcn


    # python amdgcn_edit/amdgcn_ddg.py amdgcn_edit/pa_dot_kernel.v2.amdgcn -o amdgcn_edit/ddg_trans_out_v1 --stats --inter-deps --waitcnt-deps --save-json
    # python amdgcn_edit/amdgcn_ddg.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json -r amdgcn_edit/ddg_trans_out_v1/modi_v0.amdgcn
    # python amdgcn_edit/amdgcn_verify.py amdgcn_edit/ddg_trans_out_v1/modi_v0.amdgcn amdgcn_edit/ddg_trans_out_v1/modi_v0.amdgcn --fingerprint
    # python amdgcn_edit/amdgcn_verify.py --json --fingerprint amdgcn_edit/ddg_output/analysis.json amdgcn_edit/ddg_trans_out_v1/analysis.json


    # python3 amdgcn_edit/amdgcn_register_slice.py amdgcn_edit/pa_dot_kernel.v2.amdgcn --registers v40,v41,v42,v43,v44,v45 --output-dir ./amdgcn_edit/slice_output
    # python3 amdgcn_edit/amdgcn_register_slice.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis_transform.json --registers v40,v41,v42,v43,v44,v45 --output-dir ./amdgcn_edit/slice_output
    # # python amdgcn_edit/amdgcn_register_slice.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis_transform.json --addr-range 1021,1176 --output-dir ./amdgcn_edit/slice_output
    # python3 amdgcn_edit/amdgcn_register_slice.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis_transform.json --registers v40,v41,v42,v43,v44,v45,v216,v217,v218,v219,v220,v221 --output-dir ./amdgcn_edit/slice_output


    # python3 amdgcn_edit/find_related_instructions.py amdgcn_edit/pa_dot_kernel.no_iglp.bf16.kv_blk_64.kv_cmput_blk_128.amdgcn .LBB0_2 global_load_dwordx4
    # python3 amdgcn_edit/find_related_instructions.py amdgcn_edit/pa_dot_kernel.no_iglp.bf16.kv_blk_64.kv_cmput_blk_128.amdgcn .LBB0_10 global_load_dwordx4
    # python3 amdgcn_edit/cross_block_reg_flow.py amdgcn_edit/pa_dot_kernel.no_iglp.bf16.kv_blk_64.kv_cmput_blk_128.amdgcn --bb0 .LBB0_0 --opcode-a global_load_dwordx4 --bb1 .LBB0_2 --opcode-b v_mfma_f32_16x16x16_bf16
    # python3 amdgcn_edit/cross_block_reg_flow.py amdgcn_edit/pa_dot_kernel.hand_opt.amdgcn --bb0 .LBB0_0 --opcode-a global_load_dwordx4 --bb1 .LBB0_2 --opcode-b v_mfma_f32_16x16x16_bf16
    # python3 amdgcn_edit/cross_block_reg_flow.py amdgcn_edit/pa_dot_kernel.hand_opt.amdgcn --bb0 .LBB0_2 --opcode-a global_load_dwordx4 --bb1 .LBB0_10 --opcode-b v_mfma_f32_16x16x16_bf16




    # # # python amdgcn_edit/amdgcn_ddg.py amdgcn_edit/pa_dot_kernel.bf16.kv_blk_64.kv_cmput_blk_256.ps_256.amdgcn -o amdgcn_edit/bf16.kv_blk_64.kv_cmput_blk_256.ps_256 --stats --save-json
    # # # python amdgcn_edit/amdgcn_ddg.py --load-json ./amdgcn_edit/bf16.kv_blk_64.kv_cmput_blk_256.ps_256/analysis.json -r amdgcn_edit/bf16.kv_blk_64.kv_cmput_blk_256.ps_256/modi_v0.amdgcn
    # # python amdgcn_edit/amdgcn_ddg.py amdgcn_edit/pa_dot_kernel.v2.amdgcn -o amdgcn_edit/ddg_trans_out_v1 --stats --save-json
    # # python amdgcn_edit/amdgcn_ddg.py amdgcn_edit/pa_dot_kernel.no_iglp.bf16.kv_blk_64.kv_cmput_blk_128.amdgcn -o amdgcn_edit/ddg_trans_out_v1 --stats --save-json
    # # python amdgcn_edit/amdgcn_ddg.py amdgcn_edit/pa_dot_kernel.no_iglp.bf16.kv_blk_64.kv_cmput_blk_128.ps_256.amdgcn -o amdgcn_edit/ddg_trans_out_v1 --stats --save-json
    # # python amdgcn_edit/amdgcn_ddg.py amdgcn_edit/pa_dot_kernel.no_iglp.bf16.kv_blk_64.kv_cmput_blk_128.with_branch.amdgcn -o amdgcn_edit/ddg_trans_out_v1 --stats --save-json
    # # python amdgcn_edit/amdgcn_ddg.py amdgcn_edit/pa_dot_kernel.opt_mtp_DDD_opt_mtp_tt.bf16.kv_blk_64.kv_cmput_blk_256.amdgcn -o amdgcn_edit/ddg_trans_out_v1 --stats --save-json
    # # python amdgcn_edit/amdgcn_ddg.py amdgcn_edit/pa_dot_kernel.hand_opt.amdgcn.bak -o amdgcn_edit/ddg_trans_out_v1 --stats --save-json
    # python amdgcn_edit/amdgcn_ddg.py amdgcn_edit/pa_dot_kernel.hand_opt.amdgcn -o amdgcn_edit/ddg_trans_out_v1 --stats --save-json
    # # python amdgcn_edit/amdgcn_ddg.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json -t amdgcn_edit/trans_pass_list.json -r amdgcn_edit/ddg_trans_out_v1/modi_v2.amdgcn
    # # python amdgcn_edit/amdgcn_ddg.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json -t amdgcn_edit/trans_pass_list.ps_256.json -r amdgcn_edit/ddg_trans_out_v1/modi_v2.amdgcn
    # # python amdgcn_edit/amdgcn_ddg.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json -t amdgcn_edit/trans_pass_list.json --save-json
    # # python amdgcn_edit/amdgcn_ddg.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json -t amdgcn_edit/trans_pass_list.json -r amdgcn_edit/ddg_trans_out_v1/modi_v2.amdgcn --save-ddg
    # # python amdgcn_edit/amdgcn_ddg.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json -r amdgcn_edit/ddg_trans_out_v1/modi_v2.amdgcn --save-ddg
    # # python amdgcn_edit/amdgcn_ddg.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json -r amdgcn_edit/ddg_trans_out_v1/modi_v0.amdgcn
    # python amdgcn_edit/amdgcn_ddg.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json -r amdgcn_edit/ddg_trans_out_v1/modi_v2.amdgcn



    # python ./generate_random_data.py

    # python op_tests/test_mha.py -b 1 -n 64 -q 8192 -k 8192 -d_qk_v 128,128 -d fp16 -c --no-local -bt no -m mha -det
    # python op_tests/test_mha.py -b 1 -n 64 -q 8192 -k 8192 -d_qk_v 128,128 -d fp16 -c --no-local -bt no -m mha -det --bf16-fwd-backend asm
    python op_tests/test_mha.py -b 1 -n 64 -q 8192 -k 8192 -d_qk_v 128,128 -d bf16 -c --no-local -bt no -m mha -det --bf16-fwd-backend asm
    python op_tests/test_mha.py -b 1 -n 64 -q 8192 -k 8192 -d_qk_v 128,128 -d bf16 -c --no-local -bt no -m mha -det --bf16-fwd-backend ck


    # cp /mnt/raid0/heyanguang/code/fa_triton/latest_triton_py3.12/third_party/amd/backend/compiler.py /usr/local/lib/python3.10/dist-packages/triton/backends/amd/compiler.py
    # # export TRITON_OVERRIDE_AMDGCN_FILE=/mnt/raid0/heyanguang/code/fa_triton/aiter/amdgcn_edit/ddg_trans_out_v1/modi_v2.amdgcn
    # export TRITON_OVERRIDE_AMDGCN_FILE=/mnt/raid0/heyanguang/code/fa_triton/aiter/amdgcn_edit/pa_dot_kernel.hand_opt.amdgcn
    # # export TRITON_OVERRIDE_AMDGCN_FILE=/mnt/raid0/heyanguang/code/fa_triton/aiter/amdgcn_edit/pa_dot_kernel.v4.opt.amdgcn
    # # export TRITON_OVERRIDE_AMDGCN_FILE=/mnt/raid0/heyanguang/code/fa_triton/aiter/amdgcn_edit/debug_test/pass0_level2_step5_move002.amdgcn

    # for ctx_ps in 256 1024; do
    # for ctx_ps in 1024; do
    for ctx_ps in 256; do
        # for b_val in 79 80 81 31 32 33 63 64 65 127 128 129 511 512 513; do
        # for b_val in 120 240 480 512; do
        for b_val in 110; do
            echo "Running with -b ${b_val} --context_partition_size ${ctx_ps}"
            # python ./op_tests/triton_tests/test_pa_decode_gluon.py -b ${b_val} -q 1 -c 65536 -n 10,1 -d 128 --block_size 1024 --compute_type bf16 --quant_mode per_tensor --trans_v true --kv_varlen false --use_aot_impl false --quant_q_and_kv 0,0 --context_partition_size ${ctx_ps}
            # python ./op_tests/triton_tests/test_pa_decode_gluon.py -b ${b_val} -q 1 -c 2048 -n 16,1 -d 128 --block_size 64 --compute_type bf16 --quant_mode per_tensor --trans_v false --kv_varlen false --use_aot_impl false --quant_q_and_kv 0,0 --context_partition_size ${ctx_ps}
            # python ./op_tests/triton_tests/test_pa_decode_gluon.py -b ${b_val} -q 1 -c 2048 -n 8,1 -d 128 --block_size 16 --compute_type bf16 --quant_mode per_tensor --trans_v false --kv_varlen false --use_aot_impl false --quant_q_and_kv 0,0 --context_partition_size ${ctx_ps}
        done
    done


    # python ./op_tests/triton_tests/test_pa_decode_gluon.py -b 80 -q 1 -c 2048 -n 16,1 -d 128 --block_size 64 --compute_type bf16 --quant_mode per_tensor --trans_v false --kv_varlen false --use_aot_impl false --quant_q_and_kv 0,0 --context_partition_size 1024


    # python amdgcn_edit/amdgcn_ddg.py amdgcn_edit/pa_dot_kernel.no_iglp.bf16.kv_blk_64.kv_cmput_blk_128.ps_256.amdgcn -o amdgcn_edit/ddg_trans_out_v1 --stats --save-json
    # python amdgcn_edit/debug_distribute_pass.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json --pass-list amdgcn_edit/trans_pass_list.ps_256.json --output-dir amdgcn_edit/debug_test --detail-pass 3


    # python amdgcn_edit/amdgcn_ddg.py amdgcn_edit/pa_dot_kernel.no_iglp.bf16.kv_blk_64.kv_cmput_blk_128.amdgcn -o amdgcn_edit/ddg_trans_out_v1 --stats --save-json
    # # python amdgcn_edit/amdgcn_ddg.py amdgcn_edit/pa_dot_kernel.no_iglp.bf16.kv_blk_64.kv_cmput_blk_128.ps_256.amdgcn -o amdgcn_edit/ddg_trans_out_v1 --stats --save-json
    # # python amdgcn_edit/debug_distribute_pass.py --start-step 2
    # # python amdgcn_edit/debug_distribute_pass.py --detail-step 3 --start-move 45
    # # python amdgcn_edit/amdgcn_ddg.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json -t amdgcn_edit/trans_pass_list.json --save-json
    # # python amdgcn_edit/debug_distribute_pass.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json --pass-list amdgcn_edit/trans_pass_list.json --output-dir amdgcn_edit/debug_test --skip-test
    # # python amdgcn_edit/debug_distribute_pass.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json --pass-list amdgcn_edit/trans_pass_list.json --output-dir amdgcn_edit/debug_test --skip-test --detail-pass 0 --start-step 0
    # # python amdgcn_edit/debug_distribute_pass.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json --pass-list amdgcn_edit/trans_pass_list.json --output-dir amdgcn_edit/debug_test --skip-test --detail-pass 0 --detail-step 1 --start-move 1
    # python amdgcn_edit/debug_distribute_pass.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json --pass-list amdgcn_edit/trans_pass_list.json --output-dir amdgcn_edit/debug_test --detail-pass 3
    # # python amdgcn_edit/debug_distribute_pass.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json --pass-list amdgcn_edit/trans_pass_list.json --output-dir amdgcn_edit/debug_test --detail-pass 0 --start-step 4
    # # python amdgcn_edit/debug_distribute_pass.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json --pass-list amdgcn_edit/trans_pass_list.json --output-dir amdgcn_edit/debug_test --detail-pass 3 --detail-step 6 --start-move 0


    # python3 ./perf_diff/parse_csv_diff.py > diff_csv.log
    # python3 ./perf_diff/parse_log_diff.py > diff.log

    # cat log | egrep "pid \(0, 0, 0\) idx \(  0,   1\) key_tensor" | uniq
    # cat log | egrep "pid \(0, 0, 0\) idx \(  1,   1\) key_tensor" | uniq

    # log_file=perf_diff/prev_fix_mi350_fp8_aot_DDD_opt_mtp.qlen_1_to_4.log; filter_file=xx0; cat $log_file | egrep "diff.abs.max|max_diff_thr|out_flashattn_ref_md5|out_ref_md5|gluon_output_md5" > $filter_file && cat $log_file | egrep "Selected Columns" -A 8 >> $filter_file
    # log_file=perf_diff/fix_mi350_fp8_aot_DDD_opt_mtp.qlen_1_to_4.log; filter_file=xx1; cat $log_file | egrep "diff.abs.max|max_diff_thr|out_flashattn_ref_md5|out_ref_md5|gluon_output_md5" > $filter_file && cat $log_file | egrep "Selected Columns" -A 8 >> $filter_file

    # log_file=logx.ori2.perf.q1.jit; filter_file=xx0; cat $log_file | egrep "paged_attention_decode_v2_reduce_kernel|paged_attention_decode_v2_gluon_dot_kernel" > $filter_file
    # log_file=logx.modi.perf.q1.jit; filter_file=xx1; cat $log_file | egrep "paged_attention_decode_v2_reduce_kernel|paged_attention_decode_v2_gluon_dot_kernel" > $filter_file
    # log_file=logx; filter_file=xx0; cat $log_file | egrep "paged_attention_decode_v2_reduce_kernel|paged_attention_decode_v2_gluon_dot_kernel" > $filter_file
    # log_file=logx2; filter_file=xx1; cat $log_file | egrep "paged_attention_decode_v2_reduce_kernel|paged_attention_decode_v2_gluon_dot_kernel" > $filter_file
    # filter_file=xx0; awk '{gsub(",","",$5); printf "%-60s exec:%6.1f total:%10.1f avg:%8.2f us\n", $2, $3, $5, $5/$3}' $filter_file > $filter_file.avg
    # filter_file=xx1; awk '{gsub(",","",$5); printf "%-60s exec:%6.1f total:%10.1f avg:%8.2f us\n", $2, $3, $5, $5/$3}' $filter_file > $filter_file.avg

    copy_recent_amdgcn_files
}


function run_test_pipline {

    TRITON_VERSION=$(pip show triton | grep "Version:" | awk '{print $2}')
    ROCM_VERSION=$(rocprofv3 --version | grep "rocm_version:" | awk '{print $2}')
    AITER_BRANCH=fix_mi350_fp8_aot_DDD_opt_mtp
    echo "TRITON_VERSION=$TRITON_VERSION"
    echo "ROCM_VERSION=$ROCM_VERSION"
    echo "AITER_BRANCH=$AITER_BRANCH"

    rm -rf aiter/jit/build/ aiter/jit/*.so
    rm -rf ~/.aiter/build/

    pytest -v -s op_tests/triton_tests/test_pa_decode_gluon.py -k "normal_performance and not aot" 2>&1 | tee ${AITER_BRANCH}.rocm_${ROCM_VERSION}.triton_${TRITON_VERSION}.jit.normal_performance.log
    cp run_pa_gluon_test.main.normal_accuracy_performance.jit.block_size_16.triton.${TRITON_VERSION}.csv ${AITER_BRANCH}.rocm_${ROCM_VERSION}.triton_${TRITON_VERSION}.jit.normal_performance.block_size_16.csv
    cp run_pa_gluon_test.main.normal_accuracy_performance.jit.block_size_64.triton.${TRITON_VERSION}.csv ${AITER_BRANCH}.rocm_${ROCM_VERSION}.triton_${TRITON_VERSION}.jit.normal_performance.block_size_64.csv

    pytest -v -s op_tests/triton_tests/test_pa_decode_gluon.py -k "normal_performance_aot" 2>&1 | tee ${AITER_BRANCH}.rocm_${ROCM_VERSION}.triton_${TRITON_VERSION}.aot.normal_performance.log
    cp run_pa_gluon_test.main.normal_accuracy_performance.jit.block_size_16.triton.${TRITON_VERSION}.csv ${AITER_BRANCH}.rocm_${ROCM_VERSION}.triton_${TRITON_VERSION}.aot.normal_performance.block_size_16.csv
    cp run_pa_gluon_test.main.normal_accuracy_performance.jit.block_size_64.triton.${TRITON_VERSION}.csv ${AITER_BRANCH}.rocm_${ROCM_VERSION}.triton_${TRITON_VERSION}.aot.normal_performance.block_size_64.csv

    pytest -v -s op_tests/triton_tests/test_pa_decode_gluon.py -k "normal_accuracy or normal_accuracy_aot or sliding_window_accuracy or sliding_window_performance" 2>&1 | tee ${AITER_BRANCH}.rocm_${ROCM_VERSION}.triton_${TRITON_VERSION}.norm_swa_accuracy.log
    # pytest -v -s op_tests/triton_tests/test_pa_decode_gluon.py -k "normal_accuracy or normal_accuracy_aot" 2>&1 | tee ${AITER_BRANCH}.rocm_${ROCM_VERSION}.triton_${TRITON_VERSION}.norm_swa_accuracy.log
    python3 ./perf_diff/parse_csv_diff.py > diff_csv.log

}


function get_triton_pa_thread_trace {
    rm -rf ${triton_cache_dir}
    pushd $PWD
    # export AITER_LOG_MORE=1
    # export TRITON_OVERRIDE_AMDGCN_FILE=/mnt/raid0/heyanguang/code/fa_triton/aiter/amdgcn_edit/ddg_trans_out_v1/modi_v2.amdgcn
    export TRITON_OVERRIDE_AMDGCN_FILE=/mnt/raid0/heyanguang/code/fa_triton/aiter/amdgcn_edit/pa_dot_kernel.hand_opt.amdgcn

    # KERNEL_NAME=paged_attention_decode_v2_gluon_fp8
    KERNEL_NAME=paged_attention_decode_v2_gluon_dot_kernel
    # KERNEL_VERSION="${KERNEL_NAME}_fp8_v1"
    # KERNEL_VERSION="${KERNEL_NAME}_fp8_blk_16"
    # KERNEL_VERSION="${KERNEL_NAME}_bf16_blk_16_dot_qk_k_width_16"
    # KERNEL_VERSION="${KERNEL_NAME}_bf16_blk_16"
    # KERNEL_VERSION="${KERNEL_NAME}_bf16_blk_64"
    # KERNEL_VERSION="${KERNEL_NAME}_bf16_blk_64_v2"
    # KERNEL_VERSION="${KERNEL_NAME}_bf16_blk_64_opt"
    KERNEL_VERSION="${KERNEL_NAME}_bf16_blk_64_exp"

    # pytest ./test_pa_prefill.py::test_contexted_kv_attention -v -s -k "0-cuda:0-auto-dtype1-128-1-4"
    # pytest ./test_pa_prefill.py::test_mha -v -s -k "False-True-0.0-False-False-128-4-4-1024-2048-2"
    # pytest ./test_pa_prefill.py::test_mha -v -s -k "True-False-0.0-False-False-128-16-1-1024-1024-80"
    # python $aiter_root_dir/op_tests/op_benchmarks/triton/bench_mla_decode.py -b 80 --model all --seqlen 8192 -equal_seqlens -print_vgpr

    ./pa.exe data_type=1 batch=80 dim=128 kv_seq_lens=2048 num_kv_heads=1 gqa_ratio=8 block_size=16 mtp=0 mask=1 total_loop=100 warm_ups=2 vld=0

    DUMP_TRACE=1
    # DUMP_TRACE=0
    if [ $DUMP_TRACE = 1 ]; then
        rm -rf ./pass_2
        cd ./thread_trace
        trace_dir=./${KERNEL_VERSION}
        rm -rf ./rpf_v3
        rm -rf ./${trace_dir} ./${trace_dir}.tar.gz
        mkdir -p ${trace_dir}
        cd -

        rocprofv3 -i ./input.yaml -- \
        ./pa.exe data_type=1 batch=80 dim=128 kv_seq_lens=2048 num_kv_heads=1 gqa_ratio=8 block_size=16 mtp=0 mask=1 total_loop=100 warm_ups=2 vld=0
        # python ./op_tests/triton_tests/test_pa_decode_gluon.py -b 80 -q 1 -c 2048 -n 16,1 -d 128 --block_size 16 --compute_type bf16 --quant_mode per_tensor --trans_v false --kv_varlen false --use_aot_impl false --quant_q_and_kv 0,0 --context_partition_size 256
        # python ./op_tests/triton_tests/test_pa_decode_gluon.py -b 80 -q 1 -c 2048 -n 16,1 -d 128 --block_size 64 --compute_type bf16 --quant_mode per_tensor --trans_v false --kv_varlen false --use_aot_impl false --quant_q_and_kv 0,0 --context_partition_size 1024
        # python ./op_tests/triton_tests/test_pa_decode_gluon.py -b 16 -q 1 -c 2048 -n 64,4 -d 128 --block_size 64 --compute_type bf16 --quant_mode per_tensor --trans_v false --kv_varlen false --use_aot_impl false --quant_q_and_kv 0,0 --context_partition_size 128
        # python ./op_tests/triton_tests/test_pa_decode_gluon.py -b 16 -q 1 -c 2048 -n 64,4 -d 128 --block_size 16 --compute_type bf16 --quant_mode per_tensor --trans_v false --kv_varlen false --use_aot_impl false --quant_q_and_kv 1,1 --context_partition_size 128
        # python ./op_tests/triton_tests/test_pa_decode_gluon.py -b 16 -q 1 -c 2048 -n 64,4 -d 128 --block_size 16 --compute_type fp8 --quant_mode per_tensor --trans_v false --kv_varlen false --use_aot_impl false --quant_q_and_kv 1,1 --context_partition_size 128
        # python ./op_tests/triton_tests/test_pa_decode_gluon.py -b 16 -q 1 -c 2048 -n 64,4 -d 128 --block_size 64 --compute_type fp8 --quant_mode per_tensor --trans_v false --kv_varlen false --use_aot_impl false --quant_q_and_kv 1,1 --context_partition_size 128
        # python ./op_tests/triton_tests/test_pa_decode_gluon.py -b 16 -q 1 -c 2048 -n 64,4 -d 128 --block_size 64 --compute_type bf16 --quant_mode per_tensor --trans_v false --kv_varlen false --use_aot_impl false --quant_q_and_kv 1,1 --context_partition_size 128

        cd ./thread_trace
        cp -r ./rpf_v3/pass_1/*.att ${trace_dir}
        cp -r ./rpf_v3/pass_1/ui_* ${trace_dir}
        cp -r ./rpf_v3/pass_1/*_agent_info.csv ${trace_dir}
        cp -r ./rpf_v3/pass_1/stats_ui_*.csv ${trace_dir}
        tar -zcf ./${trace_dir}.tar.gz ./${trace_dir}
        ls -lah ./${trace_dir} ./${trace_dir}.tar.gz
        cd -


        # trace_dir=./thread_trace/${KERNEL_VERSION}
        # rm -rf ./${trace_dir} ./${trace_dir}.tar.gz
        # mkdir -p ${trace_dir}
        # rocprofv2 -d ${trace_dir} -i ./thread_trace/att.txt --plugin att auto --mode file,csv -o ${trace_dir}/csv_${KERNEL_VERSION} \
        # python ./test_pa_mtp.py -n 16,1 -q 1 -c 4096 -b 128 --block_size 1024
        # # python ./test_pa_mtp.py -n 16,1 -q 1 -c 4096 -b 128 --block_size 16
        # # python ./test_pa_mtp.py -n 8,1 -q 1 -c 4096 -b 80 --block_size 16
        # # python ./block_sparse_attn.py
        # # python ./mixed_sparse_attn.py
        # # python ./00-gemm.py
        # # python $aiter_root_dir/op_tests/op_benchmarks/triton/bench_mla_decode.py -b 80 --model all --seqlen 8192 -equal_seqlens
        # # pytest ./test_pa_prefill.py::test_mha -v -s -k "False-True-0.0-False-False-128-4-4-1024-2048-2"
        # # pytest ./test_pa_prefill.py::test_contexted_kv_attention -v -s -k "0-cuda:0-auto-dtype1-128-1-4"
        # cd ./thread_trace
        # tar -zcf ./trace_${KERNEL_VERSION}.tar.gz ./trace_${KERNEL_VERSION}
        # ls -lah ./trace_${KERNEL_VERSION} ./trace_${KERNEL_VERSION}.tar.gz
        # cd -
    fi

    copy_recent_amdgcn_files
    popd
}



# install aiter
# python3 setup.py develop

# install triton
# pip install -e python
# pip install -e .


# # Press y then n while install
# ./rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh --prefix=/opt/rocm/
# cd /opt/rocm/
# ll -ah ./opt/rocm/lib/librocprof-trace-decoder.so
# ll -ah ./lib/librocprof-trace-decoder.so
# cp opt/rocm/lib/librocprof-trace-decoder.so ./lib/
# ll -ah ./lib/librocprof-trace-decoder.so


# run_test_pipline
run_aiter_op
# get_triton_pa_thread_trace


set +x
