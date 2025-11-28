# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#!/bin/sh
EXE="$(find . -name benchmark_mha_fwd -type f | head -n 1)"
KNAME=1

export CK_WARMUP=0
export CK_REPEAT=1

COMMON_ARGS='-v=1'

show_help() {
    echo "Usage: $0 -a architecture"
    echo "Current supoorted architecure: gfx942, gfx950"
    exit 0
}

run_gfx950_fwd_v3() {
    echo "Start smoke test for gfx 950"
    for head_dim in 128 192 ; do
    for mode in 0 1 ; do
    for i_perm in 0 1 ; do
    for o_perm in 0 1 ; do
    for mask in 0 2 ; do
    for lse in 0 1 ; do
    for seqlen_q in 127 192 301 512 1024; do
    for seqlen_k in 0 129 512 700 1023 1058; do

    $EXE -prec=bf16 -b=2 -h=4 -h_k=2 -d=$head_dim -d_v=128 -s=$seqlen_q -s_k=$seqlen_k -iperm=$i_perm -operm=$o_perm -mask=$mask -lse=$lse -fwd_v3=1 -mode=$mode -kname=$KNAME $COMMON_ARGS
    $EXE -prec=bf16 -b=1 -h=3 -h_k=1 -d=$head_dim -d_v=128 -s=$seqlen_q -s_k=$seqlen_k -iperm=$i_perm -operm=$o_perm -mask=$mask -lse=$lse -fwd_v3=1 -mode=$mode -kname=$KNAME $COMMON_ARGS
    $EXE -prec=bf16 -b=1 -h=1 -h_k=1 -d=$head_dim -d_v=128 -s=$seqlen_q -s_k=$seqlen_k -iperm=$i_perm -operm=$o_perm -mask=$mask -lse=$lse -fwd_v3=1 -mode=$mode -kname=$KNAME $COMMON_ARGS

    if [[ "$seqlen_q" = "$seqlen_k" ]] && [[ "$mode" = "0" ]]; then
        $EXE -prec=bf16 -b=2 -h=4 -h_k=2 -d=$head_dim -d_v=128 -s=$seqlen_q -s_k=$seqlen_q -iperm=$i_perm -operm=$o_perm -mask=1 -lse=$lse -fwd_v3=1 -mode=$mode -kname=$KNAME $COMMON_ARGS
        $EXE -prec=bf16 -b=1 -h=3 -h_k=1 -d=$head_dim -d_v=128 -s=$seqlen_q -s_k=$seqlen_q -iperm=$i_perm -operm=$o_perm -mask=1 -lse=$lse -fwd_v3=1 -mode=$mode -kname=$KNAME $COMMON_ARGS
        $EXE -prec=bf16 -b=1 -h=1 -h_k=1 -d=$head_dim -d_v=128 -s=$seqlen_q -s_k=$seqlen_q -iperm=$i_perm -operm=$o_perm -mask=1 -lse=$lse -fwd_v3=1 -mode=$mode -kname=$KNAME $COMMON_ARGS
    fi

    if [[ "$mode" = "1" ]]; then
        $EXE -prec=bf16 -b=2 -h=4 -h_k=2 -d=128 -s=$seqlen_q,$seqlen_k -s_k=$seqlen_k,0 -iperm=$i_perm -operm=$o_perm -mask=$mask -lse=$lse -fwd_v3=1 -v3_bf16_cvt=$v3_bf16_cvt -mode=$mode -kname=$KNAME $COMMON_ARGS
    fi

    done
    done
    done
    done
    done
    done
    done
    done
}

run_gfx942_fwd_v3() {
    echo "Start smoke test for gfx 942"
    for mode in 0 1 ; do
    for i_perm in 0 1 ; do
    for o_perm in 0 1 ; do
    for mask in 0 2 ; do
    for lse in 0 1 ; do
    for seqlen_q in 127 192 301 512 1024; do
    for seqlen_k in 0 129 512 700 1023 1058; do
    for v3_bf16_cvt in 0 1 2; do

    $EXE -prec=bf16 -b=2 -h=4 -h_k=2 -d=128 -s=$seqlen_q -s_k=$seqlen_k -iperm=$i_perm -operm=$o_perm -mask=$mask -lse=$lse -fwd_v3=1 -v3_bf16_cvt=$v3_bf16_cvt -mode=$mode -kname=$KNAME $COMMON_ARGS
    $EXE -prec=bf16 -b=1 -h=3 -h_k=1 -d=128 -s=$seqlen_q -s_k=$seqlen_k -iperm=$i_perm -operm=$o_perm -mask=$mask -lse=$lse -fwd_v3=1 -v3_bf16_cvt=$v3_bf16_cvt -mode=$mode -kname=$KNAME $COMMON_ARGS
    $EXE -prec=bf16 -b=1 -h=1 -h_k=1 -d=128 -s=$seqlen_q -s_k=$seqlen_k -iperm=$i_perm -operm=$o_perm -mask=$mask -lse=$lse -fwd_v3=1 -v3_bf16_cvt=$v3_bf16_cvt -mode=$mode -kname=$KNAME $COMMON_ARGS

    if [[ "$seqlen_q" = "$seqlen_k" ]] && [[ "$mode" = "0" ]]; then
        $EXE -prec=bf16 -b=2 -h=4 -h_k=2 -d=128 -s=$seqlen_q -s_k=$seqlen_q -iperm=$i_perm -operm=$o_perm -mask=1 -lse=$lse -fwd_v3=1 -v3_bf16_cvt=$v3_bf16_cvt -mode=$mode -kname=$KNAME $COMMON_ARGS
        $EXE -prec=bf16 -b=1 -h=3 -h_k=1 -d=128 -s=$seqlen_q -s_k=$seqlen_q -iperm=$i_perm -operm=$o_perm -mask=1 -lse=$lse -fwd_v3=1 -v3_bf16_cvt=$v3_bf16_cvt -mode=$mode -kname=$KNAME $COMMON_ARGS
        $EXE -prec=bf16 -b=1 -h=1 -h_k=1 -d=128 -s=$seqlen_q -s_k=$seqlen_q -iperm=$i_perm -operm=$o_perm -mask=1 -lse=$lse -fwd_v3=1 -v3_bf16_cvt=$v3_bf16_cvt -mode=$mode -kname=$KNAME $COMMON_ARGS
    fi

    if [[ "$mode" = "1" ]]; then
        $EXE -prec=bf16 -b=2 -h=4 -h_k=2 -d=128 -s=$seqlen_q,$seqlen_k -s_k=$seqlen_k,0 -iperm=$i_perm -operm=$o_perm -mask=$mask -lse=$lse -fwd_v3=1 -v3_bf16_cvt=$v3_bf16_cvt -mode=$mode -kname=$KNAME $COMMON_ARGS
    fi

    done
    done
    done
    done
    done
    done
    done
    done
}


while getopts ":a:h" opt; do
    case "${opt}" in
        a)
            mode="${OPTARG}"
            ;;
        h )
            show_help
            ;;
        *)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$mode" ]]; then
    echo "Please specify Device Architecture!" >&2
    show_help
    exit 1
fi

case "$mode" in
    "gfx942")
        run_gfx942_fwd_v3
        ;;
    "gfx950")
        run_gfx950_fwd_v3
        ;;
    *)
        echo "Unrecognized arch name: '$mode'" >&2
        echo "Current supoorted architecure: gfx942, gfx950"
        exit 1
        ;;
esac
