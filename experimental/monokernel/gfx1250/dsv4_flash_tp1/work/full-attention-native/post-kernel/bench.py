import sys,json,statistics,torch,aiter
sys.path.insert(0,'/work/atom-actual-microbench/attention')
from graphbatch import FlatGraphBatch
from post import post
torch.manual_seed(905);r=torch.randn((1,4,4096),device='cuda',dtype=torch.bfloat16);x=torch.randn((1,4096),device='cuda',dtype=torch.bfloat16);m=torch.rand(24,device='cuda');m[4:8]*=2
out=torch.empty_like(r);ref=torch.empty_like(r)
def native():aiter.mhc_post(ref,x,r,m[4:8].view(1,4,1),m[8:].view(1,4,4));return ref
def native_shuffled():return aiter.mhc_res_shuffle(native())
objects={}
def run(b,shuffle,grid=256):
 objects[(b,shuffle,grid)]=post[(grid,)](r,x,m,out,B=b,SHUFFLED=shuffle,FMA=2,num_warps=4,enable_fp_fusion=False);return out
fs={'native_plain':native,'native_plus_shuffle':native_shuffled,'post128_plain':lambda:run(128,False),'post128_shuffled':lambda:run(128,True),'post256_shuffled':lambda:run(256,True),'post128_grid32_shuffled':lambda:run(128,True,32)}
validation=[]
for seed in range(512):
 torch.manual_seed(31000+seed);r.normal_();x.normal_();m.uniform_();m[4:8]*=2
 native();expected=aiter.mhc_res_shuffle(ref)
 for b in [128,256]:
  run(b,True);torch.cuda.synchronize()
  if not torch.equal(expected,out):raise AssertionError((seed,b,int((expected!=out).sum())))
validation={'cases':512,'modes':[128,256],'outputs_bitwise_equal':True,'elements_checked':512*2*4*4096}
gs={}
for k,fn in fs.items():
 for _ in range(3):fn()
 torch.cuda.synchronize();gs[k]=FlatGraphBatch(fn)
rounds=[]
for rnd in [1,2]:
 for g in gs.values():
  for _ in range(5):g.replay()
 samples={k:[] for k in gs}
 for i in range(40):
  for k in list(gs)[::1 if i%2==0 else -1]:
   a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();gs[k].replay();b.record();b.synchronize();samples[k].append(a.elapsed_time(b)*1000/128)
 rounds.append({'round':rnd,'medians_us':{k:statistics.median(v) for k,v in samples.items()},'samples_us':samples});print(rounds[-1]['medians_us'],flush=True)
json.dump(rounds,open('timings.json','w'),indent=2);json.dump(validation,open('validation.json','w'),indent=2)
resources={}
for (b,sh,g),obj in objects.items():
 tag=f'b{b}_shuffle{sh}_grid{g}';open(tag+'.amdgcn','w').write(obj.asm['amdgcn']);resources[tag]={'vgpr':obj.n_regs,'spills':obj.n_spills,'lds':obj.metadata.shared}
json.dump(resources,open('resources.json','w'),indent=2)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
 native();native_shuffled();run(128,True);torch.cuda.synchronize()
p.export_chrome_trace('trace.json')
print(validation,resources,flush=True)
