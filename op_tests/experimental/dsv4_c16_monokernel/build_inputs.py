"""Build the published device inputs without launching GPU work."""
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent / 'work' / 'swa-fourwave'
rocm = Path(os.environ.get('ROCM_PATH', '/opt/rocm'))
aiter = Path(os.environ.get('AITER_SOURCE', '/app/aiter'))
for name in ('native_act_fold', 'native_pre_four', 'native_qnorm_four', 'pre_stage'):
    subprocess.run([str(rocm / 'llvm/bin/llvm-as'), str(root / (name + '.ll')),
                    '-o', str(root / (name + '.bc'))], check=True)
includes = [aiter / 'csrc/include', aiter / '3rdparty/opus/include',
            aiter / '3rdparty/composable_kernel/include']
for name in ('runtime', 'profile_runtime', 'coeff', 'first_post'):
    subprocess.run([str(rocm / 'llvm/bin/clang++'), '-x', 'hip',
                    '--offload-arch=gfx1250', '--cuda-device-only', '-fgpu-rdc',
                    '-emit-llvm', '-c', '-O3', '-std=c++17',
                    *['-I' + str(p) for p in includes], str(root / (name + '.cpp')),
                    '-o', str(root / (name + '.bc'))], check=True)
subprocess.run([str(rocm / 'bin/hipcc'), '--offload-arch=gfx1250',
                str(root / 'profile_clock.cpp'), '-o', str(root / 'profile_clock')],
               check=True)
(root / 'generated').mkdir(exist_ok=True)
print('Built device inputs; no GPU kernels launched.')
