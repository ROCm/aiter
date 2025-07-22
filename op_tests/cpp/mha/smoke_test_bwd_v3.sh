# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#!/bin/sh
EXE="$(find . -name benchmark_mha_bwd -type f | head -n 1)"
KNAME=1

export CK_WARMUP=0
export CK_REPEAT=1

COMMON_ARGS='-v=1'

run_batch_mode_tests() {
    for prec in "fp16" "bf16" ; do
    for perm in 0 1 ; do
    for hdim in 64 72 96 128 144 176 192 ; do
    for v3_atomic_fp32 in 0 1 ; do
    for v3_bf16_cvt in 0 1 2 ; do
    for mask in 0 1 ; do

    if [ $hdim -gt 128 ] && [ $v3_atomic_fp32 -eq 0 ]; then
        echo "skip hdim > 128 & atomic16 cases"
        continue
    fi

    if [ $prec = "fp16" ] && [ $v3_bf16_cvt -gt 0 ]; then
        echo "skip fp16 with bf16_convert cases"
        continue
    fi

    $EXE -prec=bf16 -b=2 -h=4 -h_k=2 -d=$hdim -s=512 -iperm=$perm -operm=$perm -mask=$mask -bwd_v3=1 -v3_atomic_fp32=$v3_atomic_fp32 -v3_bf16_cvt=$v3_bf16_cvt -mode=0 -kname=$KNAME $COMMON_ARGS
    $EXE -prec=bf16 -b=1 -h=3 -h_k=1 -d=$hdim -s=768 -iperm=$perm -operm=$perm -mask=$mask -bwd_v3=1 -v3_atomic_fp32=$v3_atomic_fp32 -v3_bf16_cvt=$v3_bf16_cvt -mode=0 -kname=$KNAME $COMMON_ARGS

    done
    done
    done
    done
    done
    done
}

run_swa_tests() {
    for prec in "bf16" "fp16" ; do
    for perm in 0 1 ; do
    for seqlen_q in 192 301 512 700; do
    for seqlen_k in 192 301 512 700; do
    for hdim in 72 96 128 ; do
    for mask in "t:-1,10" "t:15,-1" "t:15,15" "t:190,187" "b:-1,10" "b:15,-1" "b:15,15" "b:190,187" ; do

    $EXE -prec=$prec -b=2 -h=4 -h_k=2 -d=$hdim -s=$seqlen_q -s_k=$seqlen_k -iperm=$perm -operm=$perm -mask=$mask -bwd_v3=1 -mode=0 -kname=$KNAME $COMMON_ARGS
    $EXE -prec=$prec -b=1 -h=3 -h_k=1 -d=$hdim -s=$seqlen_q -s_k=$seqlen_k -iperm=$perm -operm=$perm -mask=$mask -bwd_v3=1 -mode=0 -kname=$KNAME $COMMON_ARGS
    $EXE -prec=$prec -b=2 -h=2 -d=$hdim -s=$seqlen_q -s_k=$seqlen_k -iperm=$perm -operm=$perm -mask=$mask -bwd_v3=1 -mode=0 -kname=$KNAME $COMMON_ARGS

    done
    done
    done
    done
    done
    done
}

run_group_mode_tests() {
    for seqlen in 63 127 200; do
    for prec in "bf16" "fp16" ; do
    for perm in 0 1 ; do
    for hdim in 64 80 96 120 128 144 160 192; do
    for mask in 0 1 ; do
    for v3_bf16_cvt in 0 1 2 ; do #valid for bf16. Pls set CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT in config.hpp to the corresponding value and re-test if a small number of slight mimatchs occurred

    if [ $prec = "fp16" ] && [ $v3_bf16_cvt -gt 0 ]; then
        echo "skip fp16 with bf16_convert cases"
        continue
    fi

    $EXE -prec=$prec -b=2 -h=3 -d=$hdim -s=$seqlen  -iperm=$perm -operm=$perm -mask=$mask -bwd_v3=1 -v3_bf16_cvt=$v3_bf16_cvt -v3_atomic_fp32=1 -mode=1 -kname=$KNAME $COMMON_ARGS
    $EXE -prec=$prec -b=1 -h=4 -h_k=1 -d=$hdim -s=$seqlen  -iperm=$perm -operm=$perm -mask=$mask -bwd_v3=1 -v3_bf16_cvt=$v3_bf16_cvt -v3_atomic_fp32=1 -mode=1 -kname=$KNAME $COMMON_ARGS

    done
    done
    done
    done
    done
    done
}

# Current native gfx950 kernels has seqlen restriction
run_gfx950_bwd_v3() {
    for prec in "bf16" ; do
    for perm in 0 1 ; do
    for v3_atomic_fp32 in 1 ; do
    for v3_bf16_cvt in 0 1 2 ; do
    for mask in 0 1 ; do
    for batch in 1 3 4 6 8 ; do
    for head in 1 4 7 8 19 32 64 ; do
    for seq in 256 512 1024 ; do

    $EXE -prec=$prec -b=$batch -h=$head -d=128 -s=$seq -iperm=$perm -operm=$perm -mask=$mask -bwd_v3=1 -v3_atomic_fp32=$v3_atomic_fp32 -v3_bf16_cvt=$v3_bf16_cvt -mode=0 -kname=$KNAME $COMMON_ARGS

    done
    done
    done
    done
    done
    done
    done
    done
}

run_batch_mode_tests
run_group_mode_tests
run_swa_tests
# run_gfx950_bwd_v3
