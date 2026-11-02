import torch
import harness as h
from combined import complete
from selected import run as run_two
CONFIGS={'two_launch':None,'grid_atomic':(0,False,4),'grid_load_rt':(1,False,4),'grid_load_ht':(2,False,4),'qready_atomic':(0,True,4),'qready_load_rt':(1,True,4),'qready_load_ht':(2,True,4),'grid_vector_load':(3,False,4),'qready_vector_load':(3,True,4)}
CONFIGS.update({'two_scratch_fp32':('two',0),'two_scratch_fp64':('two',2),'grid_scratch_fp32':(3,False,4,0),'grid_scratch_fp64':(3,False,4,2),'qready_scratch_fp64':(3,True,4,2),'qready_scratch_fp32':(3,True,4,0),'grid_scratch_rt_fp32':(1,False,4,0)})
class State(h.State):
 def __init__(self):
  super().__init__()
  self.ready=torch.zeros(1,device='cuda',dtype=torch.int32)
  self.epoch=torch.zeros(256,device='cuda',dtype=torch.int32)
 def check_state(self):
  d=super().check_state()
  d['epoch_consistent']=bool((self.epoch==self.ready).all().item())
  return d

def run(s,name):
 c=CONFIGS[name]
 if c is None:return run_two(s,'qb_unroll4')
 if c[0]=='two':
  a=h.project_fused[(192,)](h.X,s.xs,h.W,h.WS,s.p,h.NW,s.acc,s.count,s.y,s.lp,s.ls,h.Cos,h.Sin,s.kv,INLINE=True,TAIL=True,**dict(h.cbase,MODE=c[1]),**h.kw)
  b=h.qb_kernel[(256,)](s.lp,s.ls,h.BW,h.BS,s.qp,s.qc,h.Cos,h.Sin,s.q,TAIL=True,BN=128,U=4,**h.kw)
  return a,b
 style,sem,u=c[:3]
 amode=c[3] if len(c)==4 else 1
 return complete[(256,)](h.X,s.xs,h.W,h.WS,s.p,h.NW,s.acc,s.count,s.y,s.lp,s.ls,h.Cos,h.Sin,s.kv,h.BW,h.BS,s.qp,s.qc,s.q,s.bar,s.health,BN=128,GRID=256,STYLE=style,U=u,SEM=sem,Ready=s.ready,Epoch=s.epoch,QAMODE=amode,**h.kw)
