# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#!/bin/sh
EXE="$(find . -name benchmark_mha_bwd -type f | head -n 1)"

export CK_WARMUP=0
export CK_REPEAT=1

for batch in 1 15 ; do
for headnum in 1 9 ; do
for seqlenQ in 16 31 100 ; do
for seqlenK in 127 200 256 ; do
for iperm in 0 1 ; do
for operm in 0 1 ; do

$EXE -prec=fp32 -b=$batch -h=$headnum -d=128 -s=$seqlenQ -s_k=$seqlenK -iperm=$iperm -operm=$operm -bwd_v3=1 -v3_atomic_fp32=1 -v3_bf16_cvt=0 -kname=1 -v=1

done
done
done
done
done
done

# meta case
$EXE -prec=fp32 -b=1792 -h=1 -d=128 -s=32 -s_k=200 -iperm=1 -operm=1 -bwd_v3=1 -v3_atomic_fp32=1 -v3_bf16_cvt=0 -kname=1 -v=1
