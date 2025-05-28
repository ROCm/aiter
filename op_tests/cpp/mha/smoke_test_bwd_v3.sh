# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#!/bin/sh
EXE="$(find . -name benchmark_mha_bwd -type f | head -n 1)"
KNAME=1

export CK_WARMUP=0
export CK_REPEAT=1

COMMON_ARGS='-v=1'

run_bwd_v3() {
    for prec in "fp16" "bf16" ; do
    for perm in 0 1 ; do
    for hdim in 64 72 128 ; do
    for v3_atomic_fp32 in 0 1 ; do
    for v3_bf16_cvt in 0 1 2 ; do
    for mask in 0 1 ; do

    $EXE -prec=$prec -b=2 -h=4 -h_k=2 -d=$hdim -s=512 -iperm=$perm -operm=$perm -mask=$mask -bwd_v3=1 -v3_atomic_fp32=$v3_atomic_fp32 -v3_bf16_cvt=$v3_bf16_cvt -mode=0 -kname=$KNAME $COMMON_ARGS
    $EXE -prec=$prec -b=1 -h=3 -h_k=1 -d=$hdim -s=768 -iperm=$perm -operm=$perm -mask=$mask -bwd_v3=1 -v3_atomic_fp32=$v3_atomic_fp32 -v3_bf16_cvt=$v3_bf16_cvt -mode=0 -kname=$KNAME $COMMON_ARGS

    done
    done
    done
    done
    done
    done
}

run_hd192() {
    for prec in "fp16" "bf16" ; do
    for perm in 0 1 ; do
    for hdim in 144 160 192 ; do
    for v3_bf16_cvt in 0 1 2 ; do
    for mask in 0 1 ; do

    $EXE -prec=$prec -b=2 -h=4 -h_k=2 -d=$hdim -s=512 -iperm=$perm -operm=$perm -mask=$mask -bwd_v3=1 -v3_atomic_fp32=1 -v3_bf16_cvt=$v3_bf16_cvt -mode=0 -kname=$KNAME $COMMON_ARGS
    $EXE -prec=$prec -b=1 -h=3 -h_k=1 -d=$hdim -s=768 -iperm=$perm -operm=$perm -mask=$mask -bwd_v3=1 -v3_atomic_fp32=1 -v3_bf16_cvt=$v3_bf16_cvt -mode=0 -kname=$KNAME $COMMON_ARGS

    done
    done
    done
    done
    done
}

run_group_mode() {
    for prec in "fp16" "bf16" ; do
    for perm in 0 1 ; do
    for hdim in 64 96 128 ; do
    for v3_bf16_cvt in 0 1 2 ; do
    for mask in 0 1 ; do

    $EXE -prec=$prec -b=2 -h=4 -h_k=2 -d=$hdim -s=512 -iperm=$perm -operm=$perm -mask=$mask -bwd_v3=1 -v3_atomic_fp32=1 -v3_bf16_cvt=$v3_bf16_cvt -mode=1 -kname=$KNAME $COMMON_ARGS
    $EXE -prec=$prec -b=1 -h=3 -h_k=1 -d=$hdim -s=768 -iperm=$perm -operm=$perm -mask=$mask -bwd_v3=1 -v3_atomic_fp32=1 -v3_bf16_cvt=$v3_bf16_cvt -mode=1 -kname=$KNAME $COMMON_ARGS

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

set -x
run_bwd_v3
run_hd192
run_group_mode
# run_gfx950_bwd_v3
set +x
