"""GPU0 full-layer split-MoE correctness and graph A/B."""
import os,json,statistics,sys
from pathlib import Path
from types import SimpleNamespace
os.environ.update(FINE='1',FUSE='1',EARLY_COEFF='1',DIRECT_RESIDUAL='1',TAGGED_MHC='0',PACK_ATTENTION='0')
from prepare import prepare,ROOT,torch,full_swa
from schedule import build
from split_moe import build as build_split
from residency import active_workgroups_per_cu
from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled
sys.path.insert(0,'/work/atom-actual-microbench/attention');from graphbatch import GraphBatch
def kernel_count(g):
 import ctypes
 from graphbatch import hip,P,SZ,I,ok
 size=SZ();raw=g.inner.raw_cuda_graph();ok(hip.hipGraphGetNodes(raw,None,ctypes.byref(size)))
 nodes=(P*size.value)();ok(hip.hipGraphGetNodes(raw,nodes,ctypes.byref(size)));count=0
 for node in nodes:
  typ=I();ok(hip.hipGraphNodeGetType(node,ctypes.byref(typ)));count+=typ.value==0
 return count
result={'status':'preparing','scope':'GPU0 C16 TP1 SWA layer0; actual checkpoint, synthetic activations, BF16 KV,128-key window','cases':[],'timings':{}}
path=ROOT/'TASK_NATIVE_FOLDED_RESULT.json'
def save():path.write_text(json.dumps(result,indent=2))
save();p=prepare();ke=p['keep'];original=ke['residual'].clone();split=build_split(p);mono=build(p['stages4'],'swa4_split_control')
def compile(mod,args,name,grid,lds):
 os.environ['FLYDSL_DUMP_IR']='1';_preload_compiled(mod.launch,*args,torch.cuda.current_stream());os.environ.pop('FLYDSL_DUMP_IR',None)
 dr=Path('/root/.flydsl/debug')/name;ir=(dr/'19_gpu_module_to_binary.mlir').read_text();asm=(dr/'21_final_isa.s').read_text();(ROOT/'generated'/f'{name}.s').write_text(asm)
 per=active_workgroups_per_cu(SimpleNamespace(_keepalive=SimpleNamespace(_ir_text=ir)),name,128,lds)
 assert grid<=per*256
 import re
 return {'dynamic_lds':lds,'ctas':grid,'ctas_per_cu':per,**{k:re.findall(r'\.'+k+r':\s*(\d+)',asm) for k in ['vgpr_count','private_segment_fixed_size','group_segment_fixed_size']}}
result['resources']={}
for key,f,name,grid,lds in [('prefix',split['prefix'],'swa4_split_moe_prefix',256,266368),('control',mono,'swa4_split_control',256,266368)]:
 print('COMPILING',key,flush=True);result['resources'][key]=compile(f['module'],f['args'],name,grid,lds)
result['resources']['suffix']=compile(split['suffix_module'],split['suffix_args'],'swa4_split_moe_suffix',16,49152)
result['resources']['native_moe']=[{'grid':e['kwargs']['grid'],'warps':e['compiled'].metadata.num_warps,'lds':e['compiled'].metadata.shared,'vgprs':e['compiled'].n_regs} for e in split['entries']]
actual=split['outputs']
def position(pos):
 ke['pos'].fill_(pos);ke['dest'].copy_(ke['bid']*2048+pos)
 ke['idx'].copy_((ke['bid'][:,None]*2048+torch.arange(pos-127,pos+1,device='cuda',dtype=torch.int32)[None,:]).flatten())
def check(expected):
 errors={n:full_swa.compare(a,b) for n,a,b in zip(('x_prev','post','comb','residual'),actual,expected)}
 assert all(v['finite'] and v['rel_rms']<.002 for v in errors.values()),errors
 assert int(split['prefix']['health'][0])==0
 return errors
for seed in range(4):
 if seed==0:ke['residual'].copy_(original)
 else:torch.manual_seed(1500+seed);ke['residual'].normal_(0,.1)
 for pos in (1024,1535,2047):
  position(pos);expected=tuple(t.clone() for t in p['native']());cache=ke['cache'].clone()
  for e in split['entries']:e['bound']['Y'].fill_(float('nan'))
  p['routed']['output'].fill_(float('nan'));p['add']['output'].fill_(float('nan'))
  split['run']();torch.cuda.synchronize();errors=check(expected);errors['cache']=full_swa.compare(ke['cache'],cache);assert errors['cache']['exact']
  result['cases'].append({'seed':seed,'position':pos,'errors':errors});save();print('VALID',seed,pos,errors['x_prev'],flush=True)
ke['residual'].copy_(original)
nf=ke['nf'];original_call=type(nf).__call__;nf.audit_dual=False;alt=torch.cuda.Stream()
def native_call(self,x=None):
 if not self.audit_dual:return original_call(self,x)
 x=self.x if x is None else x;stream=torch.cuda.current_stream();alt.wait_stream(stream)
 with torch.cuda.stream(alt):shared=self.shared(x)
 routed=self.routed(x,self.route(self.gate(x)));stream.wait_stream(alt)
 return shared+routed
type(nf).__call__=native_call
for pos in (1024,1535,2047):
 position(pos);nf.audit_dual=False;expected=tuple(t.clone() for t in p['native']());graphs={}
 for key,dual in [('native_sequential',False),('native_dual',True)]:
  nf.audit_dual=dual
  for _ in range(2):p['native']()
  torch.cuda.synchronize();graphs[key]=GraphBatch(p['native'],16)
 graphs['monolithic']=GraphBatch(mono['run'],16);graphs['split']=GraphBatch(split['run'],16)
 samples={k:[] for k in graphs}
 for _ in range(5):
  for g in graphs.values():g.replay()
 for j in range(40):
  keys=list(graphs);keys=keys[j%4:]+keys[:j%4]
  if j%2:keys.reverse()
  for k in keys:
   a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();graphs[k].replay();b.record();b.synchronize();samples[k].append(a.elapsed_time(b)*1000/16)
  check(expected);assert int(mono['health'][0])==0
 result['timings'][pos]={'median_us':{k:statistics.median(v) for k,v in samples.items()},'samples_us':samples,'kernels_per_layer':{k:kernel_count(g) for k,g in graphs.items()}};save();print('TIME',pos,result['timings'][pos]['median_us'],flush=True)
 for j in range(50):
  if j%7==0:graphs['native_dual'].replay()
  if j%11==0:torch.empty(4*1024*1024,device='cuda',dtype=torch.uint8).fill_(j)
  graphs['split'].replay();torch.cuda.synchronize();check(expected)
assert kernel_count(graphs['split'])==4
from torch.profiler import profile,ProfilerActivity
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
 for _ in range(4):graphs['split'].replay()
 torch.cuda.synchronize()
prof.export_chrome_trace(str(ROOT/'TASK_NATIVE_FOLDED_TRACE.json'))
groups={}
for e in json.loads((ROOT/'TASK_NATIVE_FOLDED_TRACE.json').read_text())['traceEvents']:
 if e.get('cat')=='kernel':groups.setdefault(e['name'],[]).append(e['dur'])
result['profile_position']=2047;result['profile_kernel_median_us']={k:statistics.median(v) for k,v in groups.items()};assert len(groups)==4
result.update(status='passed',stress_replays_per_position=50);save();print('PASS',flush=True)
