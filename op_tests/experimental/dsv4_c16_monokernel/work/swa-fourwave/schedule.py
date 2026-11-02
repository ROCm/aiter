"""Four-wave resident SWA DAG; independent FFN branches have separate roles."""
import importlib.util,os
from pathlib import Path
ROOT=Path('/work/swa-fourwave')
def build(stages,name='swa4',limit=31,profile=None):
 import torch
 native_tasks=os.getenv('NATIVE_TASK_DESIGN','1')=='1'
 tagged='epoch_arg' in stages[2]
 fine=os.getenv('FINE','1')=='1'
 early=os.getenv('EARLY_COEFF','0')=='1';fuse=os.getenv('FUSE','1')=='1'
 assert not early or fine
 # Dependencies refer to original native-operation identities. Same-CTA stage
 # order is a topological order; a blocked consumer cannot strand its producer.
 deps={i:[i-1] for i in range(1,31)};deps[0]=[]
 deps.update({22:[],21:[15],23:[21,22],24:[23,22],25:[24],26:[15],16:[15],27:[25,26],28:[27],29:[28],30:[29,20]})
 order=list(range(16))+[22,26,21,23,24]+list(range(16,20))+[25,20]+list(range(27,31))
 if stages[2].get('parallel_roles'):deps[14].append(2)
 if tagged:deps[3]=[]
 if stages[0]['total']==0:deps[1]=[]
 order=[i for i in order if i<limit and stages[i]['total']>0]
 # HIP Q input quant and Q norm use hardware pid: their bases stay zero.
 bases={i:0 for i in range(31)}
 bases.update({2:128,15:128,21:192,22:240,23:208,24:208,25:0,26:0,16:160,17:32,18:160,19:160,20:32})
 if native_tasks:bases[25]=192
 types=[];values=[];bindings={};calls={}
 for i in order:
  s=stages[i];vals=s.get('values',s.get('args'));tys=s['types'];names=[]
  for argindex,(t,v) in enumerate(zip(tys,vals)):
   if argindex==s.get('epoch_arg',-1):names.append('ep');continue
   key=(t,v)
   if key not in bindings:bindings[key]=len(values);types.append(t);values.append(v)
   names.append('a'+str(bindings[key]))
  calls[i]=','.join(names)
 ann={'uint64':'Uint64','uint32':'Uint32','float32':'Float32','float64':'Float64'}
 sig=','.join(f'a{i}:fx.{ann[t]}' for i,t in enumerate(types))+',flags:fx.Uint64,epochs:fx.Uint64,health:fx.Uint64'
 call=','.join([f'a{i}' for i in range(len(values))]+['flags','epochs','health'])
 if profile is not None:
  sig+=',trace:fx.Uint64,quiet_ticks:fx.Uint64';call+=',trace,quiet_ticks';profile.order=order
 text=f'''import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.extern_link import BoundExternFunction
from native_gemm import _named_qa_barrier_pointer
BC='/work/swa-fourwave/runtime.bc'
C=BoundExternFunction('sw4_coeff',['uint64']*3+['uint32']*2,'void',bitcode_path=BC)
L=BoundExternFunction('sw4_local',[],'void',bitcode_path=BC)
I=BoundExternFunction('sw4_inverse_coeff',['uint64']*6+['uint32']*3,'void',bitcode_path='/work/swa-fourwave/coeff.bc')
E=BoundExternFunction('sw4_epoch',['uint64'],'uint32',bitcode_path=BC)
P=BoundExternFunction('sw4_put',['uint64','uint32','uint32'],'void',bitcode_path=BC)
G=BoundExternFunction('sw4_get',['uint64','uint64']+['uint32']*5,'void',bitcode_path=BC)
@flyc.kernel(name={name!r},known_block_size=[128,1,1])
def kernel({sig}):
 arena=fx.SharedAllocator(static=False);arena.allocate({max(stages[i]['lds'] for i in order)+(128 if early else 0)})
 _named_qa_barrier_pointer()
 pid=fx.Int32(fx.block_idx.x);tid=fx.Int32(fx.thread_idx.x)
 ep=E(epochs)
'''
 def get(d,start,count,stride=1,indent='  ',site=0):
  return f'{indent}G(flags,health,fx.Uint32({d*512})+fx.Uint32({start}),fx.Uint32({count}),fx.Uint32({stride}),ep,fx.Uint32({site}))\n'
 for i in order:
  s=stages[i];count=32 if native_tasks and i==25 else min(s['total'],256);base=bases[i]
  text+=f' rank=(pid+{256-base})%256\n if rank<{count}:\n'
  if fine and i==7:
   # A QK task owns four tokens of one head (or KV). Publish each logical task.
   text+='  for task in range(rank,260,256):\n   head=task%65\n   if head<64:\n'
   text+=get(6,'head*4',4,indent='    ',site=i)
   text+='   else:\n'+get(4,32,16,indent='    ',site=i)
   text+=f'   B7({calls[7]},task)\n   P(flags,fx.Uint32({7*512})+fx.Uint32(task),ep)\n'
   continue
  elif fine and i==8:
   text+='  row=rank%16;group=(rank//16)%4\n'
   text+=get(7,'(row//4)*65+group*16',16,site=i)+get(7,'(row//4)*65+64',1,site=i)
  elif fine and i==9:
   # Four adjacent heads per CTA; one dependency wait serves all four.
   text+='  row=rank%16;h4=rank//16\n'
   text+=get(8,'row+(h4//4)*16',4,64,site=i)
   if early:
    aa=calls[10].split(',');text+=f'  C({aa[3]},{aa[4]},{aa[5]},fx.Uint32(row),{aa[6]})\n'
   text+=f'  for h in range(4):\n   B9({calls[9]},row+16*(h4*4+h))\n'
   text+=('  L()\n' if fuse else f'  P(flags,fx.Uint32({9*512})+fx.Uint32(rank),ep)\n')
   continue
  elif fine and i==10:
   # Two 1024-value inverse/quant spans consume the same four-head tile.
   text+='  row=rank%16;h4=rank//16\n'
   if not fuse:text+=get(9,'rank',1,site=i)
   if early:
    text+=f'  if tid<32:\n   for q in range(2):\n    I({calls[10]},fx.Uint32(row+16*((h4%2)*2+q)+64*(h4//2)),fx.Uint32(fx.ptrtoint(fx.get_dyn_shared())))\n'
   else:
    text+=f'  if tid<32:\n   for q in range(2):\n    B10({calls[10]},row+16*((h4%2)*2+q)+64*(h4//2))\n'
   text+=f'  P(flags,fx.Uint32({10*512})+fx.Uint32(rank),ep)\n'
   continue
  elif fine and i==28:
   text+=get(27,'(rank//32)*32',32,site=i)
  elif fine and i==30:
   text+=get(20,0,128,site=i)
  elif fine and i==11:
   text+=get(10,'(rank//32)*32',32,site=i)
  else:
   for d in deps[i]:
    assert d in order[:order.index(i)],(i,d)
    text+=get(d,16 if d==2 and i==14 and stages[2].get('parallel_roles') else 0,16 if stages[d].get('parallel_roles') else 32 if fine and i==5 else (32 if native_tasks and d==25 else min(stages[d]['total'],256)),site=i)
  text+=f'  if tid<{s["threads"]}:\n'
  if s['total']<=256:text+=f'   B{i}({calls[i]},rank)\n'
  else:text+=f'   for task in range(rank,{s["total"]},{32 if native_tasks and i==25 else 256}):\n    B{i}({calls[i]},task)\n'
  if fine and i==29:text+='  L()\n'
  elif i!=30 and not(tagged and i==2):text+=f'  P(flags,fx.Uint32({i*512})+fx.Uint32(rank),ep)\n'
 text+=f'''@flyc.jit
def launch({sig},stream:fx.Stream):
 kernel({call}).launch(grid=(256,1,1),block=(128,1,1),stream=stream)
launch.compile_hints['llvm_options']={{'amdgpu-expert-scheduling-mode':True}}
'''
 if profile is not None:text=profile.instrument(text)
 path=ROOT/'generated'/f'{name}.py';path.write_text(text)
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 for i in order:setattr(mod,'B'+str(i),stages[i]['body'])
 flags=torch.zeros(31*512,device='cuda',dtype=torch.int32);epochs=torch.zeros(256,device='cuda',dtype=torch.int32);health=torch.zeros(1,device='cuda',dtype=torch.int32)
 args=values+[flags.data_ptr(),epochs.data_ptr(),health.data_ptr()]
 if profile is not None:args += [profile.trace.data_ptr(),profile.quiet_ticks]
 from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
 def run():return _run_compiled(mod.launch,*args,torch.cuda.current_stream())
 return dict(run=run,health=health,flags=flags,epochs=epochs,module=mod,args=args,stages=stages,order=order,deps=deps,bases=bases)
