from native import *
import statistics,collections
sys.path.insert(0,'/work/atom-actual-microbench/attention')
from graphbatch import FlatGraphBatch
n=NativeSharedRouter();x=n.x
q,s=linear.per_1x128_e8m0_quant(x,torch.float8_e4m3fn,True)
gu=n.shared.gate_up_proj(q,x_scale=s)
act=torch.empty((64,2048),device='cuda',dtype=torch.bfloat16)
aiter_silu_and_mul(act,gu,10.)
dq,ds=linear.per_1x128_e8m0_quant(act,torch.float8_e4m3fn,True)
funcs={
 'input_quant':lambda:linear.per_1x128_e8m0_quant(x,torch.float8_e4m3fn,True),
 'native_up':lambda:n.shared.gate_up_proj(q,x_scale=s),
 'candidate_up':lambda:n.gemm_candidate(q,s,n.shared.gate_up_proj,'8,4096,4096'),
 'swiglu':lambda:aiter_silu_and_mul(act,gu,10.),
 'down_quant':lambda:linear.per_1x128_e8m0_quant(act,torch.float8_e4m3fn,True),
 'native_down':lambda:n.shared.w2(dq,x_scale=ds),
 'candidate_down':lambda:n.gemm_candidate(dq,ds,n.shared.w2,'8,4096,2048'),
 'native_shared_chain':lambda:n.shared(x),
 'candidate_shared_chain':lambda:n.candidate(x),
 'native_router_layer0':lambda:n.hash_gate(x),
 'native_router_layer3':lambda:n.learned_gate(x),
}
for fn in funcs.values():
 for _ in range(3):fn()
torch.cuda.synchronize()
def err(a,b):return ((a.float()-b.float()).square().mean()/b.float().square().mean().clamp_min(1e-30)).sqrt().item()
validation=[]
for seed in range(6764,6772):
 torch.manual_seed(seed);xx=torch.randn(64,4096,device='cuda',dtype=torch.bfloat16)
 a=n.shared(xx);b=n.candidate(xx);qq,ss=linear.per_1x128_e8m0_quant(xx,torch.float8_e4m3fn,True);u=n.shared.gate_up_proj(qq,x_scale=ss);v=n.gemm_candidate(qq,ss,n.shared.gate_up_proj,'8,4096,4096')
 yy=torch.randn(64,2048,device='cuda',dtype=torch.bfloat16);qq,ss=linear.per_1x128_e8m0_quant(yy,torch.float8_e4m3fn,True);d=n.shared.w2(qq,x_scale=ss);e=n.gemm_candidate(qq,ss,n.shared.w2,'8,4096,2048');torch.cuda.synchronize()
 validation.append({'seed':seed,'up_error':err(v,u),'down_error':err(e,d),'chain_error':err(b,a),'up_exact':torch.equal(u,v),'down_exact':torch.equal(d,e),'chain_exact':torch.equal(a,b)})
json.dump(validation,open('validation.json','w'),indent=2);print('validation',validation,flush=True)
graphs={k:FlatGraphBatch(fn,128) for k,fn in funcs.items()}
for g in graphs.values():
 for _ in range(5):g.replay()
torch.cuda.synchronize();rounds=[]
for rnd in range(2):
 raw={k:[] for k in funcs}
 for i in range(40):
  keys=list(funcs);keys=keys if i%2==0 else keys[::-1]
  for key in keys:
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();graphs[key].replay();b.record();b.synchronize();raw[key].append(a.elapsed_time(b)*1000/128)
 med={k:statistics.median(v) for k,v in raw.items()};rounds.append({'median_us':med,'samples_us':raw});print('round',rnd+1,med,flush=True)
json.dump({'rounds':rounds,'method':'M64 true rows; flat graph128samebuffer repetitions;5warmups40alternatingsamples;secondround','graphs':{k:{'kernel_count':g.kernel_count,'node_types':g.node_types} for k,g in graphs.items()}},open('timings.json','w'),indent=2)
# Single-call profiling for exact kernel names, not primary latency.
for key,fn in funcs.items():
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:fn();torch.cuda.synchronize()
 prof.export_chrome_trace(key+'-trace.json');ev=json.load(open(key+'-trace.json'))['traceEvents'];ks=[{'name':e['name'],'us':e['dur']} for e in ev if e.get('cat')=='kernel'];json.dump(ks,open(key+'-kernels.json','w'),indent=2);print('kernels',key,ks,flush=True)
