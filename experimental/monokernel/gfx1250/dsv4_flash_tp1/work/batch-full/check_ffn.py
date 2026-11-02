import sys,torch,json,os
sys.path[:0]=['/work/batch-full','/work/modelwide','/work/concurrency-rollout','/work/full-attention-native','/work/full-attention-native/prefix','/work/full-attention-native/output-kernel','/work/ffn-c1/moe']
from scratch_batch import BatchState,A_WEIGHTS,F_WEIGHTS
from model_layer_batch import model_layer as batch
from model_layer_precise import model_layer as single
spec=json.load(open('/work/modelwide/scratch-spec.json'));torch.manual_seed(371)
def weight(k,i):
 s=spec[k][i];dt=getattr(torch,s['dtype'].split('.')[-1]);sh=s['shape']
 if dt==torch.uint8:return torch.full(sh,0x22 if k=='f' and i in [25,27] else 120,device='cuda',dtype=dt)
 v=torch.randn(sh,device='cuda')
 if i in ([3] if k=='a' else [2]):v*=.002
 if i in ([6,15] if k=='a' else [5]):v.fill_(1.)
 if i in ([4] if k=='a' else [3]):v.fill_(1.)
 return v.to(dt)
aw={i:weight('a',i) for i in A_WEIGHTS};fw={i:weight('f',i) for i in F_WEIGHTS}
# Physical scales are fully initialized, random values are finite.
strides=[fw[i].stride() for i in [25,26,27,28]]
def bind(s):
 for i,w in aw.items():s.a[i]=w
 for i,w in fw.items():s.f[i]=w
 return tuple(s.a),tuple(s.f)

from ffn_batch import ffn_body
from ffn_precise import ffn_body as ffn_single
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
@g.jit
def fb(F,CH,W,MI,MO,S1:gl.constexpr,S2:gl.constexpr,S3:gl.constexpr,S4:gl.constexpr,M:gl.constexpr):
 ffn_body(*F,S1,S2,S3,S4,M,CH,W,FP32IN=MI,FP32OUT=MO)
@g.jit
def fs(F,MI,MO,S1:gl.constexpr,S2:gl.constexpr,S3:gl.constexpr,S4:gl.constexpr):
 ffn_single(*F,S1,S2,S3,S4,FP32IN=MI,FP32OUT=MO)
results=[]
for m in [1,2,4,8]:
 s=BatchState(m,128);s.f[0].normal_();s.mid32.copy_(s.f[0].reshape(m,4,4096).float());A,F=bind(s)
 obj=fb[(256,)](F,s.chosen,s.weights,s.mid32,s.out32,*strides,m,num_warps=4,enable_fp_fusion=False)
 torch.cuda.synchronize();print('batch completed',m,flush=True)
 refs=[]
 for row in range(m):
  r=BatchState(1,128);r.f[0].copy_(s.f[0].reshape(m,-1)[row]);r.mid32.copy_(s.mid32[row]);_,FR=bind(r)
  fs[(256,)](FR,r.mid32,r.out32,*strides,num_warps=4,enable_fp_fusion=False);refs.append(r)
 torch.cuda.synchronize()
 check=dict(m=m,equal=torch.equal(s.f[1],torch.cat([r.f[1] for r in refs])),fp32=torch.equal(s.out32,torch.cat([r.out32 for r in refs])),finite=bool(torch.isfinite(s.f[1]).all()),health=int(s.f[-1]),vgpr=obj.n_regs,lds=obj.metadata.shared,spills=obj.n_spills)
 print(check,flush=True);results.append(check);assert check['equal'] and check['fp32'] and check['health']==0
open('/work/batch-full/ffn-checks.json','w').write(json.dumps(results,indent=2))
