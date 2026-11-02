"""Four launches: fused prefix, exact native GU, exact native GD, fused suffix.
The default full-layer schedule remains available for controlled A/B.
All launches share one stream; graph edges replace cross-launch mailbox waits.
"""
import importlib.util
from schedule import build as build_full,ROOT

def build(p,name='swa4_split_moe',profile=None):
 import torch
 from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
 prefix=build_full(p['stages4'],name+'_prefix',limit=27,profile=profile)
 stages=p['stages4'];types=[];values=[];binding={};calls={}
 for i in (29,30):
  names=[]
  for t,v in zip(stages[i]['types'],stages[i].get('values',stages[i].get('args'))):
   key=(t,v)
   if key not in binding:binding[key]=len(values);types.append(t);values.append(v)
   names.append('a'+str(binding[key]))
  calls[i]=','.join(names)
 ann={'uint64':'Uint64','uint32':'Uint32','float32':'Float32','float64':'Float64'}
 sig=','.join(f'a{i}:fx.{ann[t]}' for i,t in enumerate(types));args=','.join(f'a{i}' for i in range(len(values)))
 src=f'''import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.extern_link import BoundExternFunction
from native_gemm import _named_qa_barrier_pointer
L=BoundExternFunction('sw4_local',[],'void',bitcode_path='/work/swa-fourwave/runtime.bc')
@flyc.kernel(name={name+'_suffix'!r},known_block_size=[128,1,1])
def kernel({sig}):
 arena=fx.SharedAllocator(static=False);arena.allocate({max(stages[i]['lds'] for i in (29,30))})
 _named_qa_barrier_pointer()
 row=fx.Int32(fx.block_idx.x)
 B29({calls[29]},row)
 L()
 B30({calls[30]},row)
@flyc.jit
def launch({sig},stream:fx.Stream):
 kernel({args}).launch(grid=(16,1,1),block=(128,1,1),stream=stream)
launch.compile_hints['llvm_options']={{'amdgpu-expert-scheduling-mode':True}}
'''
 path=ROOT/'generated'/f'{name}_suffix.py';path.write_text(src)
 spec=importlib.util.spec_from_file_location(name+'_suffix',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 mod.B29=stages[29]['body'];mod.B30=stages[30]['body']
 entries=[stages[i]['entry'] for i in (27,28)]
 def gu():return entries[0]['function'].run(*entries[0]['args'],**entries[0]['kwargs'])
 def gd():return entries[1]['function'].run(*entries[1]['args'],**entries[1]['kwargs'])
 def suffix():return _run_compiled(mod.launch,*values,torch.cuda.current_stream())
 outputs=(p['add']['output'],p['mid']['outputs'][0],p['mid']['outputs'][1],p['mid']['outputs'][3])
 def run():
  prefix['run']();gu();gd();suffix()
  return outputs
 return dict(run=run,prefix=prefix,gu=gu,gd=gd,suffix=suffix,suffix_module=mod,suffix_args=values,entries=entries,keep=p,outputs=outputs)
