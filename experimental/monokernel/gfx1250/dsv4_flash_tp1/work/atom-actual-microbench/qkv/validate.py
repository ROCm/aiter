from native import *
sys.path.insert(0,'/work/mhc-optimized')
from qkv_region import QKVRegion
n=NativeQKV();f=n.f;r=QKVRegion(f['W_bytes'].cuda().view(torch.float8_e4m3fn),f['WS'].cuda(),f['NW'].cuda(),f['QB_bytes'].cuda().view(torch.float8_e4m3fn),f['QBS'].cuda())
def err(a,b):return ((a.float()-b.float()).square().mean()/b.float().square().mean().clamp_min(1e-30)).sqrt().item()
inputs=[f['live']]+[x for x in f['inputs'] if torch.is_tensor(x) and x.numel()==4096]
for seed in range(12000,12064):
 torch.manual_seed(seed);inputs.append(torch.randn(4096,dtype=torch.bfloat16))
rows=[]
for pos in [1024,1536,2047]:
 n.pos.fill_(pos);c=n.cos[pos].flatten().float().contiguous();s=n.sin[pos].flatten().float().contiguous()
 for i,x in enumerate(inputs):
  x=x.cuda().contiguous();q,k=n(x);fq,fk=r(x,c,s);torch.cuda.synchronize();rows.append({'position':pos,'input':i,'q':err(fq,q),'kv':err(fk,k),'health':r.health.item()})
res={'cases':rows,'count':len(rows),'max_q':max(x['q'] for x in rows),'max_kv':max(x['kv'] for x in rows),'failures':[x for x in rows if max(x['q'],x['kv'])>0.001 or x['health']]}
json.dump(res,open('validation_extended.json','w'),indent=2);print({k:v for k,v in res.items() if k!='cases'},flush=True)
