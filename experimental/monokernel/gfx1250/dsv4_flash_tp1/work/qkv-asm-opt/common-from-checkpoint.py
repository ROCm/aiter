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
p=Path('/data/DeepSeek-V4-Flash');idx=json.loads((p/'model.safetensors.index.json').read_text())['weight_map']
def get(name):
 with safe_open(str(p/idx[name]),framework='pt',device='cpu') as f:return f.get_tensor(name).cuda()
def w(name):return get('layers.0.'+name)
wa=w('attn.wq_a.weight');wk=w('attn.wkv.weight')
W=torch.cat([wa.view(torch.uint8),wk.view(torch.uint8)]).view(wa.dtype)
WS=torch.cat([w('attn.wq_a.scale').view(torch.uint8),w('attn.wkv.scale').view(torch.uint8)])
NW=torch.cat([w('attn.q_norm.weight').float(),w('attn.kv_norm.weight').float()])
# First case is an actual serving projection input, not only synthetic data.
live=torch.load(root/'live-input.pt',map_location='cpu').flatten().cuda()
inputs=[live]
emb=get('embed.weight')
an=w('attn_norm.weight').float()
for token in [0,1,7,42,43,100,1024,8192]:
 x=emb[token].float();inputs.append((x*torch.rsqrt(x.square().mean()+1.e-6)*an).bfloat16())
del emb
bank=[]
for x in inputs:
 z=x.float().reshape(-1,128);amax=z.abs().amax(-1).clamp_min(1.e-10)/448.
 e=(((amax.view(torch.int32)&0x7fffffff)+0x7fffff)>>23).clamp_max(254)
 scale=(e<<23).view(torch.float32)
 bank.append(((z/scale[:,None]).to(torch.float8_e4m3fn).flatten(),e.to(torch.uint8)))
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
