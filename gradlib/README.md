                  _     
                 | |    
   __ _ _ __ __ _| |__  
  / _` | '__/ _` | '_ \ 
 | (_| | | | (_| | |_) |
  \__, |_|  \__,_|_.__/ 
   __/ |                
  |___/ 

grab form vllm

1. to get gemm shapes to be tuned, replace F.linear by tgemm.mm under ater/tuned_gemm.py, 
run
    VLLM_TUNE_GEMM=1 python {workload_tests}
shapes will be captured in ater/configs/untuned_gemm.csv

2. to tune gemms in ater/configs/untuned_gemm.csv,
run 
    python3 gradlib/gradlib/gemm_tuner.py --tuned_file ater/configs/tuned_gemm.csv  --input_file ater/configs/untuned_gemm.csv

3. then run your test as normal~