

N=131072
K=32768
shuffle_weight_scales=1
repeat=1000
warmup=100
num_copies=20
out_dtype="bfloat16"
path="mxfp4_gemm_w_rotating_buff_${N}_${K}_PRESHUFFLED"
override_config=1
skip_reduce=1
export PRINT_IR_TO_FILES=1
export IR_NAME_PREFIX="triton_3_6_TOT_gitd9aa2e65"
export HIP_VISIBLE_DEVICES=1
export TRITON_HIP_USE_ASYNC_COPY=1
export AMDGCN_USE_BUFFER_OPS=1

M_LIST="128"
echo "Running experiments for M = $M_LIST"
echo "Final results will be saved in $path as a csv file (_gemm_afp4wfp4_preshuffle_data.csv)"

for M in $M_LIST; do
    # run accuracy check
    python3 run_kernel_w_rotating_buff.py --acc_check 1 --M $M --N $N --K $K --shuffle_weight_scales $shuffle_weight_scales --warmup $warmup --repeat $repeat --num_copies $num_copies --out_dtype $out_dtype --skip_reduce $skip_reduce --override_config $override_config
    # perf. run
    rocprof --stats python3 run_kernel_w_rotating_buff.py --acc_check 0 --M $M --N $N --K $K --shuffle_weight_scales $shuffle_weight_scales --warmup $warmup --repeat $repeat --num_copies $num_copies --out_dtype $out_dtype --skip_reduce $skip_reduce --override_config $override_config
    python3 collect_res.py --M $M --N $N --K $K --repeat $repeat --kernel_names _gemm_afp4wfp4_preshuffle --path $path --out_dtype $out_dtype --shuffle_weight_scales $shuffle_weight_scales --override_config $override_config
done
echo "Experiments completed"

# clean up
rm results.sysinfo.txt
rm results.json
rm results.csv
rm results.stats.csv
rm results.db

