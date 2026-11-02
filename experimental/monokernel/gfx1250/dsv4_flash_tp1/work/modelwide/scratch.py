"""Serving scratch only: no fixture models, RNG changes, or benchmark setup."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
class LayerState:
 def __init__(self):
  specs=json.loads(Path(__file__).with_name('scratch-spec.json').read_text())
  weights={'a':{3,4,5,6,12,13,15,24,25,34,40,41,42,43},'f':{2,3,4,5,19,21,22,23,24,25,26,27,28}}
  arrays={k:[torch.zeros((0,) if i in weights[k] else x['shape'],device='cuda',dtype=getattr(torch,x['dtype'].split('.')[-1])) for i,x in enumerate(v)] for k,v in specs.items()}
  self.av=arrays['a'];self.fv=arrays['f'];self.fv[0]=self.av[2].view(1,4,4096)
  self.a=SimpleNamespace(s=SimpleNamespace(attn=SimpleNamespace(**dict(zip(['cache','indices','indptr','sink','m','l','a','out','counts'],self.av[31:40])))))
  self.f=SimpleNamespace(args=lambda:tuple(self.fv))
 def attn_args(self):
  a=self.av.copy();v=self.a.s.attn
  a[31:40]=[getattr(v,k) for k in ['cache','indices','indptr','sink','m','l','a','out','counts']]
  return tuple(a)
