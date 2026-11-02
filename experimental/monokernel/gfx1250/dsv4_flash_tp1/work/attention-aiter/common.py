from pathlib import Path
import torch,json,statistics
from region import AttentionRegion,QKVAttentionRegion
from qkv_region import QKVRegion
root=Path(__file__).resolve().parent
f=torch.load(root/'fixture.pt',weights_only=True)
W=f['W_bytes'].cuda().view(torch.float8_e4m3fn);WS=f['WS'].cuda();NW=f['NW'].cuda();BW=f['QB_bytes'].cuda().view(torch.float8_e4m3fn);BS=f['QBS'].cuda()
X=f['inputs'][0].cuda();Cos=f['cos'].cuda();Sin=f['sin'].cuda()
qregion=QKVRegion(W,WS,NW,BW,BS)
Q,KV=qregion(X,Cos,Sin);torch.cuda.synchronize()
# Recover real layer0 sink when available; persist it for standalone replay.
if (root/'sink.pt').exists():Sink=torch.load(root/'sink.pt',weights_only=True).cuda()
else:
 from safetensors import safe_open
 model=Path('/data/DeepSeek-V4-Flash');index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
 with safe_open(str(model/index['layers.0.attn.attn_sink']),framework='pt',device='cpu') as sf:sn=sf.get_tensor('layers.0.attn.attn_sink').float().contiguous()
 torch.save(sn,root/'sink.pt');Sink=sn.cuda()

def reference(q,cache,idx,sink):
 valid=idx>=0;k=cache[idx.clamp_min(0).long()].double();scores=q.double()@k.T/(512**.5)
 scores=scores.masked_fill(~valid[None,:],-float('inf'));mx=torch.maximum(scores.max(-1).values,sink.double())
 p=(scores-mx[:,None]).exp();den=p.sum(-1)+(sink.double()-mx).exp()
 return (p@k/den[:,None]).bfloat16()
def error(o,r):
 return ((o.float()-r.float()).square().mean()/r.float().square().mean().clamp_min(1e-30)).sqrt().item()

def makecache(t=128):
 gen=torch.Generator(device='cuda').manual_seed(777+t)
 return (torch.randn((t,512),device='cuda',generator=gen)*.2).bfloat16()
