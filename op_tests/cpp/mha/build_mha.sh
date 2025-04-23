python3 compile.py

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JIT_DIR=$(dirname "$SCRIPT_DIR")/../../aiter/jit/

if [ ! -d "$JIT_DIR" ]; then
    echo "Error: cannot find $JIT_DIR"
    exit 1
fi

/opt/rocm/bin/hipcc  -L/opt/rocm/lib -lamdhip64 -DWITH_HIP \
                     -I$JIT_DIR/build/ck/include \
                     -I$JIT_DIR/build/ck/library/include \
                     -I$JIT_DIR/build/libmha_fwd/build/include \
                     -fPIC -std=c++17 -O3 -std=c++17 \
                     -DUSE_ROCM=1 \
                     -DHIPBLAS_V2 \
                     -DCUDA_HAS_FP16=1 \
                     -D__HIP_PLATFORM_AMD__=1 \
                     -D__HIP_NO_HALF_OPERATORS__=1 \
                     -D__HIP_NO_HALF_CONVERSIONS__=1 \
                     -DLEGACY_HIPBLAS_DIRECT \
                     -DUSE_PROF_API=1 \
                     -D__HIP_PLATFORM_HCC__=1 \
                     -D__HIP_PLATFORM_AMD__=1 \
                     -U__HIP_NO_HALF_CONVERSIONS__ \
                     -U__HIP_NO_HALF_OPERATORS__ \
                     -Wno-unused-result \
                     -Wno-switch-bool \
                     -Wno-vla-cxx-extension \
                     -Wno-undefined-func-template \
                     -Wno-macro-redefined \
                     -fgpu-flush-denormals-to-zero \
                     -mllvm -enable-post-misched=0 \
                     -mllvm -amdgpu-early-inline-all=true \
                     -mllvm -amdgpu-function-calls=false \
                     -mllvm -amdgpu-coerce-illegal-types=1 \
                     -mllvm --amdgpu-kernarg-preload-count=16 \
                     -DCK_TILE_FMHA_FWD_FAST_EXP2=1 \
                     -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=2 \
                     -DCK_TILE_FMHA_FWD_SPLITKV_API=1 \
                     --offload-arch=native \
                     -fno-gpu-rdc \
                     -fno-offload-uniform-block \
                     -L $SCRIPT_DIR \
                     -lmha_fwd $SCRIPT_DIR/benchmark_mha_fwd.cpp -o benchmark_mha_fwd 

/opt/rocm/bin/hipcc  -L/opt/rocm/lib -lamdhip64 -DWITH_HIP \
                     -I$JIT_DIR/build/ck/include \
                     -I$JIT_DIR/build/ck/library/include \
                     -I$JIT_DIR/build/libmha_bwd/build/include \
                     -fPIC -std=c++17 -O3 -std=c++17 \
                     -DUSE_ROCM=1 \
                     -DHIPBLAS_V2 \
                     -DCUDA_HAS_FP16=1 \
                     -D__HIP_PLATFORM_AMD__=1 \
                     -D__HIP_NO_HALF_OPERATORS__=1 \
                     -D__HIP_NO_HALF_CONVERSIONS__=1 \
                     -DLEGACY_HIPBLAS_DIRECT \
                     -DUSE_PROF_API=1 \
                     -D__HIP_PLATFORM_HCC__=1 \
                     -D__HIP_PLATFORM_AMD__=1 \
                     -U__HIP_NO_HALF_CONVERSIONS__ \
                     -U__HIP_NO_HALF_OPERATORS__ \
                     -Wno-unused-result \
                     -Wno-switch-bool \
                     -Wno-vla-cxx-extension \
                     -Wno-undefined-func-template \
                     -Wno-macro-redefined \
                     -fgpu-flush-denormals-to-zero \
                     -mllvm -enable-post-misched=0 \
                     -mllvm -amdgpu-early-inline-all=true \
                     -mllvm -amdgpu-function-calls=false \
                     -mllvm -amdgpu-coerce-illegal-types=1 \
                     -mllvm --amdgpu-kernarg-preload-count=16 \
                     -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=2 \
                     --offload-arch=native \
                     -fno-gpu-rdc \
                     -fno-offload-uniform-block \
                     -L $SCRIPT_DIR \
                     -lmha_bwd $SCRIPT_DIR/benchmark_mha_bwd.cpp -o benchmark_mha_bwd 