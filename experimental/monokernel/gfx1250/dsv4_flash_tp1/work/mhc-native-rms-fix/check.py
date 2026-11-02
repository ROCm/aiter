import sys,os,json
from pathlib import Path
import torch
root=Path(__file__).resolve().parent
mode=sys.argv[1];src='/work/mhc-optimized' if mode=='before' else str(root/'candidate')
sys.path[:0]=['/work/atom-actual-microbench/qkv','/work/atom-actual-microbench/mhc','/app/aiter','/app/ATOM']
from native import NativeQKV
from native_api import NativeMHC
sys.path.insert(0,src)
import setup as c
import aiter
from atom.model_ops.v4_kernels.paged_decode import sparse_attn_v4_paged_decode
out=root/mode;out.mkdir(exist_ok=True)
s=c.State(128);native_m=c_m=NativeMHC(c.HAF,c.HAS,c.HAB,c.AN)
n=NativeQKV(cache=s.attn.cache.clone(),dest=127)
idx=torch.zeros(16384,device='cuda',dtype=torch.int32);idx[:128]=torch.arange(128,device='cuda',dtype=torch.int32)
ptr=torch.tensor([0,128],device='cuda',dtype=torch.int32)
inputs=[c.f2['inputs'][i].cuda().contiguous() if i<9 else torch.randn((4,4096),device='cuda',generator=torch.Generator(device='cuda').manual_seed(171000+i)).bfloat16() for i in range(137)]
def err(a,b):return c.err(a.flatten(),b.flatten())
allrounds=[]
with torch.inference_mode():
 for rnd in [1,2]:
  rows=[]
  for pos in [1024,1536,2047]:
   n.pos.fill_(pos);c.Cos=n.cos[pos].flatten().float().contiguous();c.Sin=n.sin[pos].flatten().float().contiguous()
   for i,r in enumerate(inputs):
    x,post,comb=native_m(aiter.mhc_res_shuffle(r.view(1,4,4096)))
    q,kv=n(x);ref=sparse_attn_v4_paged_decode(q,n.cache,idx,ptr,s.attn.sink,512**-.5)
    s.x.fill_(float('nan'));s.q.fill_(float('nan'));s.attn.out.fill_(float('nan'))
    actual=s.run(r,'opt_both');torch.cuda.synchronize()
    vals={'x':err(s.x,x),'post':err(s.mix[4:8],post),'comb':err(s.mix[8:],comb),'q':err(s.q,q),'kv':err(s.kv,kv),'attention':err(actual,ref)}
    rows.append({'case':i,'position':pos,**vals,'health':s.check(),'finite':all(torch.isfinite(t).all().item() for t in [s.x,s.q,s.kv,actual])})
   print(mode,rnd,pos,'done',flush=True)
  summary={'round':rnd,'cases':len(rows),'max_errors':{k:max(v[k] for v in rows) for k in vals},'failures':{k:sum(v[k]>.001 for v in rows) for k in vals},'unhealthy':sum(not v['health'] for v in rows),'nonfinite':sum(not v['finite'] for v in rows),'rows':rows}
  allrounds.append(summary);(out/'validation.json').write_text(json.dumps(allrounds,indent=2));print({k:v for k,v in summary.items() if k!='rows'},flush=True)
 k=s.obj
 (out/'resources.json').write_text(json.dumps({'vgpr':k.n_regs,'lds':k.metadata.shared,'spills':k.n_spills},indent=2))
 (out/'prefix.amdgcn').write_text(k.asm['amdgcn'])
