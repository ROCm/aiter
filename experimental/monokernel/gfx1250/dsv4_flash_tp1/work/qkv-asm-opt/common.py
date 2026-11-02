import json,statistics,re,hashlib
from pathlib import Path
import torch
from safetensors import safe_open
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"baseline"))
from kernels import initialize,project,reduce_norm
from opt import project_opt,reduce_opt
root=Path(__file__).resolve().parent
torch.manual_seed(0)
assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx1250')
fixture=torch.load(root/'fixture.pt',map_location='cpu',weights_only=True)
W=fixture['W_bytes'].cuda().view(torch.float8_e4m3fn)
WS=fixture['WS'].cuda();NW=fixture['NW'].cuda();live=fixture['live'].cuda()
bank=[(x.cuda().view(torch.float8_e4m3fn),s.cuda()) for x,s in fixture['bank']]
X=bank[0][0].clone();XS=bank[0][1].clone()
P=torch.empty((8,1536),device='cuda',dtype=torch.float32)
Acc=torch.empty(1536,device='cuda',dtype=torch.float32)
SS=torch.empty(48,device='cuda',dtype=torch.float32)
Count=torch.empty(50,device='cuda',dtype=torch.int32)
Y=torch.empty(1536,device='cuda',dtype=torch.float32)
kw=dict(num_warps=4,enable_fp_fusion=False)

wf=W.double()*torch.pow(2.,WS.double()-127).repeat_interleave(128,0).repeat_interleave(128,1)
def reference():
 xf=X.double()*torch.pow(2.,XS.double()-127).repeat_interleave(128)
 z=(wf@xf).bfloat16().float()
 return torch.cat([z[:1024]*torch.rsqrt(z[:1024].square().mean()+1.e-6)*NW[:1024],z[1024:]*torch.rsqrt(z[1024:].square().mean()+1.e-6)*NW[1024:]])
def metric(a,b):
 d=a.float()-b.float();return dict(relative_rms=(d.square().mean()/b.float().square().mean().clamp_min(1e-30)).sqrt().item(),max_abs=d.abs().max().item(),exact=torch.equal(a,b),finite=torch.isfinite(a).all().item())
