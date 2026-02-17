# Triton GEMM tunning script

single case testing:
rocprofv3 --kernel-trace -f csv -o res -- python3 ut.py 4 2112 7168 8 32 1024 1 2 1 1 16 0 7

**Running tunning script**

Example 1: Tunning for A16W16 GEMM using default BLOCK_SIZE ranges using GPU 0, see screen.py for deafult range

    python3 screen.py \
        64 8192 3584 0 \
        ut_a16w16_gemm.py \
        > example1.out

Example 2: Background tunning for A8W8 GEMM blockscale using specific `BLOCK_SIZE_K` ranges using GPU 0 ~ 6, because A8W8 blockscale gemm requires only `BLOCK_SIZE_K=128`
    
    N=2112
    K=7168
    for M_G in "8 0" "16 1" "32 2" "64 3" "128 4" "256 5" "8192 6"; do
        set -- $M_G
        M=$1
        G=$2
        nohup python3 screen.py \
            $M $N $K $G \
            ut_a8w8_gemm_blockscale.py \
            --block-size-k-range 128 \
            > example2-M=$M-N=$N-K=$K-G=$G.out &
    done

Example 3: Background tunning for AFP4WFP4 GEMM. In this case `BLOCK_SZIE_M` has to meet the following requirements: `1) BLOCK_SIZE_M < 32 for M < 32, 2) BLOCK_SIZE_M >= 32 for M >= 32`. `BLOCK_SIZE_K` has to meet the following requirements: `BLOCK_SIZE_K >= 256`. If we still use the default settings, GEMM will give assertion errors, screen.py will skip those cases first time it hits assert errors and skip all other cases that shares the same BLOCK_SIZE. See the generated *.log files and termeinal output (example3.out) for more details. It will take a few minutes for screen.py to skip through those failed configs, so if you want to save those few minutes, you have to set dedicated `--block-size-m-range` for each `M` to skip invalid `BLOCK_SZIE_M`. This example also enables verbose printout that shows the pre-pruned cases and the error messages that triggers exclusions of cases on-the-fly.

    N=7168
    K=2048
    G=0
    python3 screen.py \
        64 $N $K $G \
        ut_afp4wfp4_gemm_preshuffle.py \
        --block-size-k-range 256 512 1024 \
        --overwrite \
        --verbose \
        > example3.out
    
**Viewing results and generate JSON config files**

Example 1:

    python view-screen.py ut_a16w16_gemm.py --n-list 8192 --k-list 3584

Example 2: 

    N=2112
    K=7168
    python view-screen.py ut_a8w8_gemm_blockscale.py --n-list $N --k-list $K

Example 3:

    N=2112
    K=7168
    python view-screen.py ut_afp4wfp4_gemm_preshuffle.py --n-list $N --k-list $K

**Verify performance**

To verify that your tunned JSON config files actually is performant and can be correctly picked up by AITER, first you have to copy the generated JSON config files into `<path_to_aiter_root>/aiter/ops/triton/configs/gemm`:

    cp *.json <path_to_aiter_root>/aiter/ops/triton/configs/gemm/

then, you can run, for example,

    python verify-perf.py 32 2112 7168 ut_a8w8_gemm_blockscale_preshuffle.py

and check the kernel name (with config suffix) and runtime to see if both kernel name and runtime match those inside the JSON config files. If the kernel name and runtime do not match, it could be that your JSON file name is wrong. You have to go to the file where the kernel resides and check the `_get_config` function to check the `config_name` arguments.