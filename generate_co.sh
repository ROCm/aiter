#!/bin/bash
set -e
current_dir=$(pwd)
kernel=F8_FMHA_D192_VD128_1TG_8W_32mx8_128nx1

for causal_value in 0 1; do
    echo "Processing with CAUSAL_MASK = $causal_value"
    # delete previous generation
    rm -rf ./*causal${causal_value}*
    
    # Create a copy with causal mask value
    src_sp3=${kernel}_causal${causal_value}
    raw_sp3=${kernel}_raw_causal${causal_value}
    target_s=${kernel}_causal${causal_value}
    fn_name=_ZN5aiter40mla_pfl_qh192_vh128_m32x8_n128x1_causal${causal_value}E
    co_name=mla_pfl_qh192_vh128_m32x8_n128x1_causal${causal_value}.co
    
    # Copy original file and replace CAUSAL_MASK value
    cp ${kernel}.sp3 ${src_sp3}.sp3
    sed -i "s/^var CAUSAL_MASK[ \t]*=.*/var CAUSAL_MASK     = ${causal_value}/" ${src_sp3}.sp3

    # Process the modified file
    sp3 ${src_sp3}.sp3 -hex ${raw_sp3}.hex
    sp3 -hex ${raw_sp3}.hex ${raw_sp3}.sp3    
    python3 ./mla_prefill_cvt.py ${src_sp3}.sp3 ${raw_sp3}.sp3 ${target_s}.s ${fn_name}
    
    echo "Completed compilation for CAUSAL_MASK = $causal_value"
    echo "---"
    
    # cmd="/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa --offload-arch=gfx950 ${target_s}.s -o ${co_name}"
    # echo $cmd
    # eval $cmd
    
    # echo "Generated ${co_name} for CAUSAL_MASK = $causal_value"
    
    # # copy target_s to folder
    # cp ${co_name} ./hsa/gfx950/mla/
done


    