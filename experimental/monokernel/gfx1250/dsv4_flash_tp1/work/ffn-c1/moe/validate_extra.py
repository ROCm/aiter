import os,sys,json,statistics,time
os.environ['ATOM_USE_TRITON_MOE']='1';os.environ['ATOM_USE_TRITON_GEMM']='1';os.environ['AITER_TRITON_USE_HERD']='1';os.environ['ATOM_DSV4_FLASH_MOE']='0'
sys.path.insert(0,'/app/ATOM')
import torch
from safetensors import safe_open
from aiter.ops.shuffle import moe_shuffle_weight,moe_shuffle_scale
from aiter.ops.triton.utils.shuffle import moe_weight_decode_view
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.moe.moe_routing.routing import routing
from atom.model_ops.fused_moe_triton import _fused_experts_silu_gugu
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4,get_kernel_config_gluon
from moe_multitoken_prototype import fused
from graphbatch import FlatGraphBatch
M=1;torch.manual_seed(1250)
index=json.load(open('/data/DeepSeek-V4-Flash/model.safetensors.index.json'))['weight_map'];handles={}
def load(k):
 f=index[k]
 if f not in handles:handles[f]=safe_open('/data/DeepSeek-V4-Flash/'+f,framework='pt')
 return handles[f].get_tensor(k).view(torch.uint8).cuda()
# Exact checkpoint gate/up interleave, then native gfx1250 WMMA shuffle.
w13=torch.empty((256,4096,2048),device='cuda',dtype=torch.uint8);sc13=torch.empty((256,4096,128),device='cuda',dtype=torch.uint8);wd=torch.empty((256,4096,1024),device='cuda',dtype=torch.uint8);sd=torch.empty((256,4096,64),device='cuda',dtype=torch.uint8)
for e in range(256):
 p=f'layers.0.ffn.experts.{e}.'
 w13[e,0::2]=load(p+'w1.weight');w13[e,1::2]=load(p+'w3.weight');sc13[e,0::2]=load(p+'w1.scale');sc13[e,1::2]=load(p+'w3.scale');wd[e]=load(p+'w2.weight');sd[e]=load(p+'w2.scale')
 if e%64==0:print('loaded',e,flush=True)
w1=moe_weight_decode_view(moe_shuffle_weight(w13));s1=moe_shuffle_scale(sc13).transpose(-1,-2);w2=moe_weight_decode_view(moe_shuffle_weight(wd));s2=moe_shuffle_scale(sd).transpose(-1,-2)
del w13,sc13,wd,sd;torch.cuda.empty_cache()
a,b,c,d=[t.transpose(1,2) for t in (w1,s1,w2,s2)]
torch.manual_seed(6764);x=torch.randn((M,4096),device='cuda',dtype=torch.bfloat16)
rf=torch.load('/work/ffn-c1/shared-router/fixture.pt',weights_only=False);rw=rf['layers.0.ffn.gate.weight'].cuda();logits=torch.nn.functional.linear(x,rw);torch.save({'x':x.cpu(),'logits':logits.cpu()},'inputs.pt')
rd,gather,scatter=routing(logits,6,score_mode='sqrtsoftplus',renorm=True,routed_scaling_factor=1.5)
xq,xs=downcast_to_mxfp(x,torch.float8_e4m3fn,axis=-1)
print('routing',rd.block_m,rd.n_expts_act,rd.expt_data.hist.tolist(),flush=True)
def native():return _fused_experts_silu_gugu(x,w1,w2,rd,gather,scatter,s1,s2,'GFX1250_SCALE','GFX1250_SCALE',swiglu_limit=10.,preshuffled=True)
i=torch.empty((M*6,2048),device='cuda',dtype=xq.dtype);is_=torch.empty((M*6,64),device='cuda',dtype=torch.uint8);y=torch.empty((M*6,4096),device='cuda',dtype=torch.bfloat16);z=torch.empty((M,4096),device='cuda',dtype=torch.bfloat16);count=torch.zeros(2,device='cuda',dtype=torch.uint32);health=torch.zeros(1,device='cuda',dtype=torch.int32);e=rd.expt_data
args=(a,b,c,d,i,is_,y,gather,e.hist,e.token_offs_raw,e.token_offs_pad[-1:],e.block_pid_map,rd.gate_scal,count,health,z,scatter,a.stride(),b.stride(),c.stride(),d.stride(),10.,M)
obj=None
def candidate():
 global obj
 xx,ss=downcast_to_mxfp(x,torch.float8_e4m3fn,axis=-1)
 obj=fused[(256,)](xx,ss,*args,GRID=256,BUFFERS=3,num_warps=4)
 return z

checks=[]
for seed in range(16):
 torch.manual_seed(81000+seed);x.normal_();ref=native().clone();candidate();torch.cuda.synchronize()
 checks.append({'seed':81000+seed,'neq':int((z!=ref).sum()),'rms':float(((z.float()-ref.float()).square().mean()/ref.float().square().mean()).sqrt()),'health':int(health.item())})
json.dump(checks,open('additional-validation.json','w'),indent=2);print(checks,flush=True)
shared=torch.randn_like(z)
def routing_only():return routing(logits,6,score_mode='sqrtsoftplus',renorm=True,routed_scaling_factor=1.5)
def shared_add():return z+shared
fs={'HERD_routing':routing_only,'shared_add_bf16':shared_add};gs={}
for k,fn in fs.items():
 for _ in range(3):fn()
 torch.cuda.synchronize();gs[k]=FlatGraphBatch(fn,n=128)
rounds=[]
for rnd in [1,2]:
 for g in gs.values():
  for _ in range(5):g.replay()
 samples={k:[] for k in gs}
 for it in range(40):
  for k in list(gs)[::1 if it%2==0 else -1]:
   aa,bb=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);aa.record();gs[k].replay();bb.record();bb.synchronize();samples[k].append(aa.elapsed_time(bb)*1000/128)
 rounds.append({'round':rnd,'medians_us':{k:statistics.median(v) for k,v in samples.items()},'samples_us':samples});print(rounds[-1]['medians_us'],flush=True)
json.dump(rounds,open('routing-add-timings.json','w'),indent=2)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
 routing_only();shared_add();torch.cuda.synchronize()
p.export_chrome_trace('routing-add-trace.json')
json.dump({'vgpr':obj.n_regs,'spills':obj.n_spills,'lds':obj.metadata.shared,'counts':count.tolist(),'health':health.item()},open('candidate-resources.json','w'),indent=2)
