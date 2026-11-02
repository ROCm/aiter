import sys,json,statistics,os
sys.path.insert(0,'/app/ATOM');sys.path.insert(0,'/work/mhc-optimized')
import torch
from atom.model_ops.v4_kernels.paged_decode import sparse_attn_v4_paged_decode
from region import AttentionRegion
N=int(os.environ.get('INDEX_CAP','128'))
torch.manual_seed(420)
def graph(fn):
 for _ in range(4):fn()
 torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):
  for _ in range(128):fn()
 return g
def bench(fs):
 gs={k:graph(v) for k,v in fs.items()};rs=[]
 for r in range(2):
  for g in gs.values():
   for _ in range(5):g.replay()
  raw={k:[] for k in gs}
  for i in range(40):
   for k in list(gs)[::1 if i%2==0 else -1]:
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();gs[k].replay();b.record();b.synchronize();raw[k].append(a.elapsed_time(b)*1000/128)
  rs.append({'median_us':{k:statistics.median(v) for k,v in raw.items()},'samples_us':raw})
 return rs
results={}
for length in [128]:
 q=torch.randn((1,64,512),device='cuda',dtype=torch.bfloat16)*.3
 cache=torch.randn((max(4096,length),512),device='cuda',dtype=torch.bfloat16)*.2
 idx=torch.zeros(max(length,N),device='cuda',dtype=torch.int32);idx[:length]=torch.arange(length,device='cuda',dtype=torch.int32)
 ptr=torch.tensor([0,length],device='cuda',dtype=torch.int32)
 sink=torch.load('/work/mhc-optimized/sink.pt',map_location='cuda',weights_only=False).float().flatten()
 if sink.numel()!=64:raise RuntimeError(sink.shape)
 kv=cache[length-1].clone();ours=AttentionRegion(cache,idx[:length].clone(),sink,splits=4 if length==128 or length>=193 else 2,write_slot=length-1)
 def native():return sparse_attn_v4_paged_decode(q,cache,idx,ptr,sink,512**-.5)
 def fused():return ours(q[0],kv)
 ref=native();out=fused();torch.cuda.synchronize()
 exact=torch.softmax(torch.cat([torch.matmul(q[0].float(),cache[:length].float().T)*512**-.5,sink[:,None]],1),1)[:,:length]@cache[:length].float()
 exact=exact.to(torch.bfloat16).float()
 native_error=((ref[0].float()-exact).square().mean()/exact.square().mean()).sqrt().item()
 fused_error=((out.float()-exact).square().mean()/exact.square().mean()).sqrt().item()
 rel=((out.float()-ref[0].float()).square().mean()/ref.float().square().mean()).sqrt().item()
 r=bench({'native_atom':native,'fused_attention_including_cache_write':fused})
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
  native();torch.cuda.synchronize()
 p.export_chrome_trace(f'envelope-native-{length}-trace.json')
 kernels=[{'name':e.name,'us':e.device_time_total} for e in p.events() if str(e.device_type)=='DeviceType.CUDA']
 results[str(length)]={'q_shape':list(q.shape),'cache_shape':list(cache.shape),'indices_capacity':idx.numel(),'true_length':length,'native_kernels':kernels,'rel_rms_native_vs_fused':rel,'native_vs_fp32_rel_rms':native_error,'fused_vs_fp32_rel_rms':fused_error,'fused_splits':ours.splits,'fused_health':ours.check(),'rounds':r}
 json.dump(results,open('envelope-results.json','w'),indent=2)
 print(length,rel,r[1]['median_us'],flush=True)
