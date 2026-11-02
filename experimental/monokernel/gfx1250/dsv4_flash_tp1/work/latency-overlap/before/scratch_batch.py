import json,math
from pathlib import Path
import torch
A_WEIGHTS={3,4,5,6,12,13,15,24,25,34,40,41,42,43}
F_WEIGHTS={2,3,4,5,19,21,22,23,24,25,26,27,28}
class BatchState:
 def __init__(self,m,t=640):
  spec=json.loads(Path('/work/modelwide/scratch-spec.json').read_text())
  def alloc(key,weights,fixed):
   return [torch.zeros((0,) if i in weights else (math.prod(s['shape'])*(1 if i in fixed else m),),device='cuda',dtype=getattr(torch,s['dtype'].split('.')[-1])) for i,s in enumerate(spec[key])]
  self.a=alloc('a',A_WEIGHTS,{17,27,29,30,39})
  self.f=alloc('f',F_WEIGHTS,{11,17,32,33,34,39,42,43})
  self.f[0]=self.a[2]
  def empty(shape,dt=torch.bfloat16):return torch.empty(shape,device='cuda',dtype=dt)
  self.a[31]=empty((m,t,512));self.a[32]=empty((m,t),torch.int32);self.a[33]=empty((m,2),torch.int32)
  self.a[35]=empty((m,64,64),torch.float32);self.a[36]=empty((m,64,64),torch.float32);self.a[37]=empty((m,64,64,512),torch.float32)
  self.xq=empty((m,4096),torch.float8_e4m3fn);self.xs=empty((m,32),torch.uint8)
  self.chosen=empty((m,256),torch.int32);self.weights=empty((m,256))
  self.main_plan=empty((m,8),torch.int32);self.index_plan=empty((m,8),torch.int32)
  self.output=empty((m,4,4096));self.mid32=empty((m,4,4096),torch.float32);self.out32=empty((m,4,4096),torch.float32)
  self.pool=empty((m,4,3,512),torch.float32)
  self.comp_counter=torch.zeros(1,device='cuda',dtype=torch.int32)
  self.mhc=(empty((64,m,32),torch.float32),empty((64,m),torch.float32),empty((m,4,1),torch.float32),empty((m,4,4),torch.float32),torch.zeros(1,device='cuda',dtype=torch.int32))
 def extra(self):return (self.xq,self.xs,self.chosen,self.weights,self.main_plan,self.index_plan,self.output)
