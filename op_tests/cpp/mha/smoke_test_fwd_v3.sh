# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#!/bin/sh
EXE="$(find . -name benchmark_mha_fwd -type f | head -n 1)"
KNAME=1

export CK_WARMUP=0
export CK_REPEAT=1

COMMON_ARGS='-v=1'

run_gfx950_fwd_v3() {
    echo "Start smoke test for gfx 950"
    for perm in 0 1 ; do
    for mask in 0 1 ; do
    for seqlen in 384 512 600 783 900 1023 ; do

    $EXE -prec=bf16 -b=2 -h=4 -h_k=2 -d=128 -s=$seqlen -iperm=$perm -operm=$perm -mask=$mask -lse=1 -fwd_v3=1 -mode=0 -kname=$KNAME $COMMON_ARGS
    $EXE -prec=bf16 -b=1 -h=3 -h_k=1 -d=128 -s=$seqlen -iperm=$perm -operm=$perm -mask=$mask -lse=1 -fwd_v3=1 -mode=0 -kname=$KNAME $COMMON_ARGS
    $EXE -prec=bf16 -b=1 -h=1 -h_k=1 -d=128 -s=$seqlen -iperm=$perm -operm=$perm -mask=$mask -lse=1 -fwd_v3=1 -mode=0 -kname=$KNAME $COMMON_ARGS

    done
    done
    done
}

run_gfx942_fwd_v3() {
    echo "Start smoke test for gfx 942"
    for perm in 0 1 ; do
    for mask in 0 1 ; do
    for lse in 0 1 ; do
    for seqlen in 384 512 600 783 900 1023 ; do

    $EXE -prec=bf16 -b=2 -h=4 -h_k=2 -d=128 -s=$seqlen -iperm=$perm -operm=$perm -mask=$mask -lse=$lse -fwd_v3=1 -mode=0 -kname=$KNAME $COMMON_ARGS
    $EXE -prec=bf16 -b=1 -h=3 -h_k=1 -d=128 -s=$seqlen -iperm=$perm -operm=$perm -mask=$mask -lse=$lse -fwd_v3=1 -mode=0 -kname=$KNAME $COMMON_ARGS
    $EXE -prec=bf16 -b=1 -h=1 -h_k=1 -d=128 -s=$seqlen -iperm=$perm -operm=$perm -mask=$mask -lse=$lse -fwd_v3=1 -mode=0 -kname=$KNAME $COMMON_ARGS

    done
    done
    done
    done
}

while getopts ":a:" opt; do
    case "${opt}" in
        a)
            mode="${OPTARG}"
            ;;
        *)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$mode" ]]; then
    echo "Please specify device name by `-a` option" >&2
    exit 1
fi

case "$mode" in
    "942")
        run_gfx942_fwd_v3
        ;;
    "950")
        run_gfx950_fwd_v3
        ;;
    *)
        echo "Unrecognized arch name: '$mode'" >&2
        exit 1
        ;;
esac
