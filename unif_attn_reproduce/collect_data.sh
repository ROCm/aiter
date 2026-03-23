export PRINT_IRS=1
export TRITON_ALWAYS_COMPILE=1
# if both 0, use bf16, if both 1, fp8 with tensor scale
q_fp8=1
kv_fp8=1
for wpeu in 1; do
for lenn in 2048; do
    seq_q_l=$lenn
    seq_kv_l=$lenn

    num_heads_q=8
    num_heads_k=1
    bs=1
    window_size=0
    block_size=64
    head_size=64
    use_tdm=1
    num_kv_blocks=1
    waves_per_eu=$wpeu
    shuffled_kv_cache=0
    num_warps=4
    BLOCK_M=128

    # pass these arguments to the single_run_am.sh script
    export SEQ_Q_L=$seq_q_l
    export SEQ_KV_L=$seq_kv_l
    export NUM_HEADS_Q=$num_heads_q
    export NUM_HEADS_K=$num_heads_k
    export BS=$bs
    export WINDOW_SIZE=$window_size
    export BLOCK_SIZE=$block_size
    export HEAD_SIZE=$head_size
    export USE_TDM=$use_tdm
    export NUM_KV_BLOCKS=$num_kv_blocks
    export WAVES_PER_EU=$waves_per_eu
    export SHUFFLED_KV_CACHE=$shuffled_kv_cache
    export NUM_WARPS=$num_warps
    export Q_FP8=$q_fp8
    export KV_FP8=$kv_fp8
    export BLOCK_M=$BLOCK_M
    source $TRITON_GFX1250_MODEL_PATH/ffmlite_env.sh
    source $TRITON_GFX1250_MODEL_PATH/am_env.sh

    #export AMD_INSERT_AMDGCN="unified_attention_2d_gluon_block_m_128_tile_size_64_block_size_64_head_size_64_amdgcn_interleaved_4k_4k_1_8.txt"
    #export AMD_INSERT_AMDGCN="exp_reorder_unified_attention_2d_gluon_num_warps_4_block_m_128_tile_size_64_block_size_64_head_size_64_sfl_0_amdgcn.txt"
    #unset AMD_INSERT_AMDGCN
    rm -r ~/.triton/cache/
    folder_name="3_23_shuffle_${SHUFFLED_KV_CACHE}_Q_FP8_${Q_FP8}_K_FP8_${KV_FP8}_BLOCK_M_${BLOCK_M}_NUM_WARPS_${NUM_WARPS}_large_LDS_weu_${WAVES_PER_EU}_sq_${seq_q_l}_sk_${seq_kv_l}_nh_${num_heads_q}_nk_${num_heads_k}_bs_${bs}_ws_${window_size}_bs_${block_size}_hs_${head_size}_tdm_${use_tdm}_nkv_${num_kv_blocks}"
    mkdir -p $folder_name
    cd $folder_name
    # save snapshots
    cp ../single_run_am.sh .
    cp ../run_kernel.py .
    #cp ../${AMD_INSERT_AMDGCN} .
    #capture with FFM
    # Important to do the following! Otherwise will see error:
    #   TCoreAssert FAIL : 0 == (vidmemBase & 0xffffff)
    GPUVM_BASE="${HSA_KMT_MODEL_GPUVM_BASE:-0x200000000}"
    GPUVM_SIZE="${HSA_KMT_MODEL_GPUVM_SIZE:-0xF00000000}"
    #--disp <kernel_name>/0` helps filter out other kernels
    source $TRITON_GFX1250_MODEL_PATH/ffmlite_env.sh
    $TRITON_GFX1250_MODEL_PATH/tools/roccap/bin/roccap capture --loglevel trace --disp "kernel_unified_attention_2d/0" --file gemm.cap bash single_run_am.sh
    echo "After capture"
    echo "--------------------------------"
    LARGEST_CAP=$(ls -S *.cap | head -1)
    echo "Largest cap: ${LARGEST_CAP}"
    #replay with AM
    export DtifFbBaseLocation="${GPUVM_BASE}"
    source $TRITON_GFX1250_MODEL_PATH/am_env.sh
    $TRITON_GFX1250_MODEL_PATH/tools/roccap/bin/roccap play -r "${GPUVM_BASE}-${GPUVM_SIZE}" ./${LARGEST_CAP}
    echo "After replay"
    echo "--------------------------------"
    mkdir -p ATT_trace
    $TRITON_GFX1250_MODEL_PATH/tools/roccap/bin/roccap extract --sp3 0- ${LARGEST_CAP} # whichever .cap file you used earlier for itrace
    $TRITON_GFX1250_MODEL_PATH/ffm-lite/sp3disasm ./roc-dump* ATT_trace.sp3
    $TRITON_GFX1250_MODEL_PATH/tools/rcv/amtool ATT_trace *.mon ATT_trace.sp3

    grep -A1 "WGP00" ./xcc0se0sa0_itrace_emu.mon > ./wgp0.txt
    python3 ../gen_perfetto.py ./wgp0.txt ./${folder_name}.json

    rm *.mon
    cd ../
done
done