STAMP="$(date +%Y%m%d_%H%M%S)"
INPUT_CSV="hy3_qmoe_tokens_${STAMP}.csv"
OUTPUT_CSV="hy3_qmoe_results_${STAMP}.csv"
LOG="hy3_qmoe_tokens_${STAMP}.log"

echo "token,model_dim,inter_dim,expert,topk,act_type,dtype,use_g1u1,doweight_stage1,q_type,q_dtype_a,q_dtype_w,q_type2,q_dtype_a2,q_dtype_w2" > "$INPUT_CSV"
for ((t=1; t<=32768; t*=2)); do
  echo "$t,4096,192,193,9,silu,bf16,1,0,QuantType.per_Tensor,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_Tensor,torch.float8_e4m3fnuz,torch.float8_e4m3fnuz" >> "$INPUT_CSV"
done

python test_qmoe_multi.py --csv "$INPUT_CSV" --quant-from-csv --output-csv "$OUTPUT_CSV" 2>&1 | tee -a "$LOG"
