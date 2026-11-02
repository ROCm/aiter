"""Host-side construction of an unchanged native compiled stage adapter."""
import importlib.util, json, math
from pathlib import Path
from bridge.export import export

def runtime_args(entry):
    k=entry['compiled']; fn=entry['function']; vals=dict(zip(fn.arg_names,entry['args'])); vals.update(entry['kwargs'])
    args=[]
    for i,(name,ty) in enumerate(k.src.signature.items()):
        if ty=='constexpr' or (i,) in k.src.constants: continue
        value=vals[name]
        args.append(value.data_ptr() if hasattr(value,'data_ptr') else (0 if value is None else value))
    # Triton appends global and profile scratch pointers even when unused.
    if k.metadata.global_scratch_size or k.metadata.profile_scratch_size:
        raise ValueError('Scratch-using native stages require explicit scratch allocation')
    return args+[0,0]

def build(entry, directory, symbol, physical_warps=9, resident_grid=256):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    k=entry['compiled'];ll=directory/(symbol+'.native.ll');ll.write_text(k.asm['llir'])
    preserve_fp_fusion=bool(getattr(k.metadata,'enable_fp_fusion',False))
    manifest=export(ll,directory/(symbol+'.bc'),symbol,k.metadata.num_warps,physical_warps,preserve_fp_fusion=preserve_fp_fusion)
    manifest['native_enable_fp_fusion']=bool(getattr(k.metadata,'enable_fp_fusion',False))
    values=runtime_args(entry)
    assert len(values)==len(manifest['types'])-4,(len(values),manifest['types'])
    grid=entry['kwargs']['grid']; grid=grid(entry['kwargs']) if callable(grid) else grid
    grid=tuple(grid)+(1,)*(3-len(grid));total=math.prod(grid)
    atypes={'uint64':'Uint64','uint32':'Uint32','float32':'Float32','float64':'Float64'}
    names=[f'a{i}' for i in range(len(values))]
    signature=', '.join(f'{n}:fx.{atypes[t]}' for n,t in zip(names,manifest['types']))
    callargs=', '.join(names)
    code=f'''import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.extern_link import BoundExternFunction
from native_gemm import _named_qa_barrier_pointer
BODY=BoundExternFunction({symbol!r},{manifest['types']!r},'void',bitcode_path={str(directory/(symbol+'.bc'))!r})
@flyc.jit
def body({signature}, logical:fx.Int32):
 BODY({callargs},fx.Uint32(logical%{grid[0]}),fx.Uint32((logical//{grid[0]})%{grid[1]}),fx.Uint32(logical//{grid[0]*grid[1]}),fx.Uint64(0))
@flyc.kernel(known_block_size=[{physical_warps*32},1,1])
def kernel({signature}):
 arena=fx.SharedAllocator(static=False);arena.allocate({max(k.metadata.shared,16)})
 _named_qa_barrier_pointer()
 pid=fx.Int32(fx.block_idx.x);tid=fx.Int32(fx.thread_idx.x)
 if tid<{k.metadata.num_warps*32}:
  for logical in range(pid,{total},{resident_grid}):
   body({callargs},logical)
@flyc.jit
def launch({signature},stream:fx.Stream):
 kernel({callargs}).launch(grid=({resident_grid},1,1),block=({physical_warps*32},1,1),stream=stream)
'''
    code += "\nlaunch.compile_hints['llvm_options']={'amdgpu-expert-scheduling-mode':True}\n"
    path=directory/(symbol+'_adapter.py');path.write_text(code)
    spec=importlib.util.spec_from_file_location(symbol+'_adapter',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    manifest.update(grid=grid,total=total,lds=k.metadata.shared);return module,values,manifest
