import sys,torch,json
sys.path[:0]=['/work/batch-full','/work/modelwide','/work/full-attention-native/prefix','/work/ffn-c1/moe']
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import ffn_batch as b, ffn_precise as c
from combined import sync
@g.jit
def batch(X,Q,S,MX,MS,G,LG,W,WS,GU,A,AQ,AS,W2,WS2,Y,H,O,P,MP,GA,SC,GM,BI,CH,WE,COUNT,HEALTH,M:gl.constexpr):
 b.quant_body(X,Q,S,4096,128,8,M);b.quant_body(X,MX,MS,4096,32,1,M);b.router_body(X,G,LG,M)
 sync(COUNT,HEALTH,256,3)
 b.route_body(LG,H,O,P,MP,GA,SC,GM,BI,True,CH,WE,COUNT,HEALTH,M)
 b.shared_body(Q,S,W,WS,GU,4096,64,M)
 sync(COUNT,HEALTH,256,3)
 b.activation_quant(GU,A,AQ,AS,M)
 sync(COUNT,HEALTH,256,3)
 b.shared_body(AQ,AS,W2,WS2,Y,2048,32,M)
@g.jit
def single(X,Q,S,MX,MS,G,LG,W,WS,GU,A,AQ,AS,W2,WS2,Y,H,O,P,MP,GA,SC,GM,BI,COUNT,HEALTH):
 c.quant_body(X,Q,S,4096,128,8);c.quant_body(X,MX,MS,4096,32);c.router_body(X,G,LG)
 sync(COUNT,HEALTH,256,3)
 c.route_body(LG,H,O,P,MP,GA,SC,GM,BI,True)
 c.shared_body(Q,S,W,WS,GU,4096,64)
 sync(COUNT,HEALTH,256,3)
 c.activation_quant(GU,A,AQ,AS)
 sync(COUNT,HEALTH,256,3)
 c.shared_body(AQ,AS,W2,WS2,Y,2048,32)
def e(sh,dt=torch.bfloat16):return torch.empty(sh,device='cuda',dtype=dt)
def z(sh,dt=torch.int32):return torch.zeros(sh,device='cuda',dtype=dt)
def scratch(m):return [e((m,4096),torch.float8_e4m3fn),z((32,8),torch.uint8),e((m,4096),torch.float8_e4m3fn),e((m,128),torch.uint8),e((m,256)),e((m,4096)),e((m,2048)),e((m,2048),torch.float8_e4m3fn),z((16,8),torch.uint8),e((m,4096)),z(256),z(256),z(1),z(m*6),z(m*6),z(m*6),e(m*6),z((m,256)),e((m,256)),z(1),z(1)]
torch.manual_seed(42)
w=torch.randn((4096,4096),device='cuda').to(torch.float8_e4m3fn);w2=torch.randn((4096,2048),device='cuda').to(torch.float8_e4m3fn)
ws=torch.full((4096,32),122,device='cuda',dtype=torch.uint8);ws2=torch.full((4096,16),122,device='cuda',dtype=torch.uint8)
gate=torch.randn((256,4096),device='cuda',dtype=torch.bfloat16)*.02;bias=torch.randn(256,device='cuda')*.05
results=[]
for m in [1,2,4,8]:
 x=torch.randn((m,4096),device='cuda',dtype=torch.bfloat16)
 a=scratch(m)
 def args(a,x):return (x,a[0],a[1],a[2],a[3],gate,a[4],w,ws,a[5],a[6],a[7],a[8],w2,ws2,a[9],*a[10:17],bias)
 obj=batch[(256,)](*args(a,x),*a[17:],m,num_warps=4,enable_fp_fusion=False)
 refs=[]
 for row in range(m):
  r=scratch(1);single[(256,)](*args(r,x[row]),*r[19:],num_warps=4,enable_fp_fusion=False);refs.append(r)
 torch.cuda.synchronize()
 checks={}
 for k in range(10):
  ref=torch.cat([r[k][:,:1] if k in [1,8] else r[k] for r in refs],dim=1 if k in [1,8] else 0)
  v=a[k][:,:m] if k in [1,8] else a[k]
  checks[str(k)]=torch.equal(v.contiguous().view(torch.uint8),ref.contiguous().view(torch.uint8))
 # Canonical sorted expert/token pairs and weights, independent of changed packing.
 hist=a[10].cpu();off=a[11].cpu();nactive=int(a[12]);mapping=a[13].cpu();gather=a[14].cpu();scatter=a[15].cpu();gamma=a[16].cpu()
 checks['hist']=torch.equal(a[10],sum(r[10] for r in refs))
 for row,r in enumerate(refs):
  for j,expert in enumerate(r[13].cpu().tolist()):
   slot=int(scatter[row*6+j]);checks[f'route{row}.{j}']=bool(off[expert]<=slot<off[expert]+hist[expert] and gather[slot]==row*6+j and gamma[slot]==r[16][j].cpu())
 checks['active']=nactive==int((hist>0).sum()) and torch.equal(mapping[:nactive],torch.where(hist>0)[0].int()) and bool((mapping[nactive:]==-1).all())
 result=dict(m=m,checks=checks,health=int(a[-1]),vgpr=obj.n_regs,lds=obj.metadata.shared,spills=obj.n_spills)
 print(result,flush=True);results.append(result)
 assert all(checks.values()) and result['health']==0
open('/work/batch-full/ffn-group-checks.json','w').write(json.dumps(results,indent=2))
