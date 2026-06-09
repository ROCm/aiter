# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
#!/bin/bash

if [ $# -ge 1 ] ; then
    FMA_API=$1  # build fwd/bwd/fwd_v3/bwd_v3
else
    FMA_API=""  # build all
fi

echo "######## building mha kernel $FMA_API"
python3 compile.py --api=$FMA_API

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOP_DIR=$(dirname "$SCRIPT_DIR")/../../

if [ x"$FMA_API" = x"fwd" ] || [ x"$FMA_API" = x"" ] ; then
echo "######## linking mha fwd"

/opt/rocm/bin/hipcc  -I$TOP_DIR/3rdparty/composable_kernel/include \
                     -I$TOP_DIR/3rdparty/composable_kernel/example/ck_tile/01_fmha/ \
                     -I$TOP_DIR/csrc/include \
                     -std=c++20 -O3 \
                     -DUSE_ROCM=1 \
                     -DENABLE_CK=1 \
                     -DCK_TILE_FMHA_FWD_SPLITKV_API=1 \
                     --offload-arch=native \
                     -L $SCRIPT_DIR -lmha_fwd \
                     $SCRIPT_DIR/benchmark_mha_fwd.cpp -o fwd.exe
fi

if [ x"$FMA_API" = x"fwd_v3" ] ; then
echo "######## linking mha fwd_v3 (ASM FWD with sink, CK-excluded)"

# fwd_v3 produces libmha_fwd_asm.so with ENABLE_CK=0 (asm-only path +
# fmha_fwd_with_sink_asm dispatcher for gfx1250).  A separate library name
# (libmha_fwd_asm vs libmha_fwd) is used so the JIT blob directory never
# collides with the full CK build of libmha_fwd.
# rpath:
#   $ORIGIN      -> find libmha_fwd_asm.so next to the exe without LD_LIBRARY_PATH
#   /opt/rocm/lib -> libamdhip64.so.7 lives here

hipcc  -I$TOP_DIR/csrc/include \
                     -std=c++20 -O3 \
                     -DUSE_ROCM=1 \
                     -DENABLE_CK=0 \
                     --offload-arch=native \
                     -L $SCRIPT_DIR -lmha_fwd_asm \
                     -Wl,-rpath,'$ORIGIN':/opt/rocm/lib \
                     $SCRIPT_DIR/benchmark_mha_fwd_v3.cpp -o fwd.exe
fi

if [ x"$FMA_API" = x"bwd" ] || [ x"$FMA_API" = x"" ] ; then
echo "######## linking mha bwd"
/opt/rocm/bin/hipcc  -I$TOP_DIR/3rdparty/composable_kernel/include \
                     -I$TOP_DIR/3rdparty/composable_kernel/example/ck_tile/01_fmha/ \
                     -I$TOP_DIR/csrc/include \
                     -std=c++20 -O3 \
                     -DUSE_ROCM=1 \
                     -DENABLE_CK=1 \
                     --offload-arch=native \
                     -L $SCRIPT_DIR -lmha_bwd \
                     $SCRIPT_DIR/benchmark_mha_bwd.cpp -o bwd.exe
fi

if [ x"$FMA_API" = x"bwd_v3" ] ; then
echo "######## linking mha bwd_v3 (CK-excluded host)"
# bwd_v3 produces libmha_bwd.so with ENABLE_CK=0 (asm-only). Pair it with a
# CK-free host so the build doesn't need to compile any CK headers (which
# currently fail on archs like gfx1250 anyway).
# rpath:
#   $ORIGIN     -> find libmha_bwd.so next to the exe without LD_LIBRARY_PATH
#   /opt/rocm/lib -> libamdhip64.so.7 lives here; avoid the user having to
#                    prepend it to LD_LIBRARY_PATH every time
/opt/rocm/bin/hipcc  -I$TOP_DIR/csrc/include \
                     -std=c++20 -O3 \
                     -DUSE_ROCM=1 \
                     -DENABLE_CK=0 \
                     --offload-arch=native \
                     -L $SCRIPT_DIR -lmha_bwd \
                     -Wl,-rpath,'$ORIGIN':/opt/rocm/lib \
                     $SCRIPT_DIR/benchmark_mha_bwd_v3.cpp -o bwd.exe
fi
