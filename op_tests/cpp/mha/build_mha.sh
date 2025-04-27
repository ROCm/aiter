#!/bin/bash

if [ $# -ge 1 ] ; then
    FMA_API=$1  # build fwd/bwd
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
                     -std=c++17 -O3 \
                     -DUSE_ROCM=1 \
                     -DCK_TILE_FMHA_FWD_SPLITKV_API=1 \
                     --offload-arch=native \
                     -L $SCRIPT_DIR -lmha_fwd \
                     $SCRIPT_DIR/benchmark_mha_fwd.cpp -o benchmark_mha_fwd
fi

if [ x"$FMA_API" = x"bwd" ] || [ x"$FMA_API" = x"" ] ; then
echo "######## linking mha bwd"
/opt/rocm/bin/hipcc  -I$TOP_DIR/3rdparty/composable_kernel/include \
                     -I$TOP_DIR/3rdparty/composable_kernel/example/ck_tile/01_fmha/ \
                     -I$TOP_DIR/csrc/include \
                     -std=c++17 -O3 \
                     -DUSE_ROCM=1 \
                     --offload-arch=native \
                     -L $SCRIPT_DIR -lmha_bwd \
                     $SCRIPT_DIR/benchmark_mha_bwd.cpp -o benchmark_mha_bwd
fi
