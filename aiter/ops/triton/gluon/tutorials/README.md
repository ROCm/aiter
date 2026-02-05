# Triton Gluon Extension for AMDGPU

This repo provides some extension to current Gluon to give users more control without breaking the tile level programming model.


## Triton prepare
1. git cherry-pick tip commits from [gluon_ext](https://github.com/ROCm/triton/tree/gluon_ext).
2. build triton from source. (e.g. `pip install -e .`)

common debug: 
cherry-pick should be conflict free since this is add-on.
if your local triton is out-of-date for long time, you may get compile error.
e.g. comment out the warp_id implementation in `python/triton/experimental/gluon/language/amd/cdna3/__init__.py`
if your code hasn't contained the implementation of ttg.warp_id

## Kernel Authoring
1. import gluon
```
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
```
2. using the amd extensions, mostly gl.amd.cdna3/4.xxx

### extensions
1. instruction scheduling related
  - sched_barrier: separate scheduling regions so that instrs can not pass it, strictly respected by llvm compiler
  - sched_group_barrier: specify certain order in the scheduling region. e.g 1 mma followed by 4 ds_read. may not be respected by llvm compiler.
  - iglp_opt: setting scheduling strategy, not effective all the time
2. control related
  - s_barrier: Synchronize threads/workitems within a threadgroup/workgroup
  - s_set_prio: set the priority of warp/wave
  - warp_id: warp/wave id inside a threadgroup/workgroup
for more details, please refer to `python/triton/experimental/gluon/language/amd/cdna3/__init__.py`

### examples
`python bench_matmul.py` will run the matmul examples including triton, gluon, gluon_ext.

- `matmul_triton.py` is the triton version.
- `matmul_gluon_gfx950_256x256x32_3stage.py` and `matmul_gluon_gfx950_256x256x64_2stage.py` are gluon version
- `matmul_gluon_gfx950_256x256x32_3stage_scheduling.py` and `matmul_gluon_gfx950_256x256x64_2stage_scheduling.py` are gluon_ext version, manually control instruction scheduling.
- `matmul_gluon_gfx950_256x256x32_3stage_pingpong.py` and `matmul_gluon_gfx950_256x256x64_2stage_ping.py` are gluon_ext version, implementing a block ping-pong strategy.

