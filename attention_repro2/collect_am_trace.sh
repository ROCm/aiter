
export PRINT_IRS=1
for wpeu in 2; do
for lenn in 4096; do
    seq_q_l=$lenn
    seq_kv_l=$lenn

    num_heads_q=8
    num_heads_k=1
    bs=1
    window_size=0
    block_size=128
    tile_size=128
    head_size=64
    use_tdm=1
    num_kv_blocks=1
    waves_per_eu=$wpeu
    num_warps=4
    BLOCK_M=128
    q_fp8=1
    kv_fp8=1
    causal=1
    remove_indirect=0
    num_buffer=3
    loop_variant=2
    shuffled_kv_cache=1
    num_splits=1

    # pass these arguments to the run_am.sh script
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
    export TRITON_ALWAYS_COMPILE=1
    export SHUFFLED_KV_CACHE=$shuffled_kv_cache
    export NUM_WARPS=$num_warps
    export TILE_SIZE=$tile_size
    export Q_FP8=$q_fp8
    export KV_FP8=$kv_fp8
    export BLOCK_M=$BLOCK_M
    export CAUSAL=$causal
    export REMOVE_INDIRECT=$remove_indirect
    export NUM_BUFFER=$num_buffer
    export loop_variant=$loop_variant
    export NUM_SPLITS=$num_splits

    #export AMDGPU_COEXEC_SINGULAR_FLAVOR=0

    # put path here
    #export REPLACE_LLIR_PATH="..."
    # maxnum: v_max_num_f32_e32
    #maximum: v_maximum 
    #export LLIR_TRANSFORM_MAX="maximum 1"
    #export LLIR_UNSINK_ACC_RESCALE=1
    # export LLIR_REMOVE_DS_WAIT_0="all"
    # # # # # # # # If no TDM load wait around
    # export LLIR_REMOVE_BARRIER=1
    
    #unset LLIR_TRANSFORM_MAX
    #export AMD_INSERT_AMDGCN="unified_attention_2d_gluon_block_m_128_tile_size_64_block_size_64_head_size_64_amdgcn_interleaved_4k_4k_1_8.txt"
    #export AMD_INSERT_AMDGCN="exp_reorder_unified_attention_2d_gluon_num_warps_4_block_m_128_tile_size_64_block_size_64_head_size_64_sfl_0_amdgcn.txt"
    #unset AMD_INSERT_AMDGCN
    rm -r ~/.triton/cache/
    folder_name="6_13_triton_up_loop_${loop_variant}_buf_${num_buffer}_page_$((1 - remove_indirect))_causal_${causal}_shuffle_${SHUFFLED_KV_CACHE}_Q_FP8_${Q_FP8}_KV_FP8_${KV_FP8}_triton_BLOCK_M_${BLOCK_M}_NUM_WARPS_${NUM_WARPS}_large_LDS_weu_${WAVES_PER_EU}_sq_${seq_q_l}_sk_${seq_kv_l}_nh_${num_heads_q}_nk_${num_heads_k}_bs_${bs}_ws_${window_size}_bs_${block_size}_hs_${head_size}_tdm_${use_tdm}_nkv_${num_kv_blocks}_splits_${num_splits}"
    mkdir -p $folder_name
    cd $folder_name
    pip show triton > triton_version.txt
    cp ../collect_am_trace.sh .
    cp ../run_am.sh .
    cp ../unified_attention_w_3d.py .
    cp ../run_kernel.py .
    #cp ../${AMD_INSERT_AMDGCN} .
    #capture with FFM
    # Important to do the following! Otherwise will see error:
    #   TCoreAssert FAIL : 0 == (vidmemBase & 0xffffff)
    GPUVM_BASE="${HSA_KMT_MODEL_GPUVM_BASE:-0x200000000}"
    GPUVM_SIZE="${HSA_KMT_MODEL_GPUVM_SIZE:-0xF00000000}"
    #--disp <kernel_name>/0` helps filter out other kernels
    source $TRITON_GFX1250_MODEL_PATH/ffmlite_env.sh
    echo "--------------------------------"
    echo "Before capture"
    echo "--------------------------------"
    $TRITON_GFX1250_MODEL_PATH/tools/roccap/bin/roccap capture --loglevel trace --file attn.cap  --disp "unified_attention_gluon_kernel_2d/0" bash run_am.sh
    echo "--------------------------------"
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

    python3 ../collect_results.py --draw_log ./draw.log --causal $causal --q_fp8 $q_fp8 --kv_fp8 $kv_fp8 --window_size $window_size --head_size $head_size --block_size $block_size --num_heads_q $num_heads_q --num_heads_k $num_heads_k --bs $bs --seq_q_l $seq_q_l --seq_kv_l $seq_kv_l > roofline.txt

    #rm *.mon
    cd ../
done
done
