"""Full-kernel profiling: selected stages, deferred stores, graph replay, GPU0."""
import os,json,sys,statistics,re,subprocess
from pathlib import Path
from types import SimpleNamespace
os.environ.update(FINE='1',FUSE='1',EARLY_COEFF='1',TAGGED_MHC='0',DIRECT_RESIDUAL='1',PACK_ATTENTION='0')
from prepare import prepare,ROOT,torch,full_swa
from split_moe import build as build_split

def build(stages,name,profile=None):
 f=build_split(p,name=name,profile=profile)
 return dict(f['prefix'],run=f['run'],keep=f)

from profile_schedule import Profile
from residency import active_workgroups_per_cu
from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled
sys.path.insert(0,'/work/atom-actual-microbench/attention')
from graphbatch import GraphBatch
rate=subprocess.check_output([str(ROOT/'profile_clock')],text=True).split();assert rate[0]=='0';khz=int(rate[1]);ticks_us=khz/1000
outdir=ROOT/os.getenv('PROFILE_DIR',('profiles-immediate' if os.getenv('PROFILE_IMMEDIATE','0')=='1' else 'profiles'));outdir.mkdir(exist_ok=True)
p=prepare();expected=tuple(t.clone() for t in p['native']());torch.cuda.synchronize()
actual=(p['add']['output'],p['mid']['outputs'][0],p['mid']['outputs'][1],p['mid']['outputs'][3])
def check(f):
 torch.cuda.synchronize();assert int(f['health'][0])==0
 c={n:full_swa.compare(a,b) for n,a,b in zip(('x_prev','post','comb','residual'),actual,expected)}
 assert all(x['finite'] and x['rel_rms']<.002 for x in c.values()),c
 return c
def compile(f,name):
 os.environ['FLYDSL_DUMP_IR']='1';_preload_compiled(f['module'].launch,*f['args'],torch.cuda.current_stream());os.environ.pop('FLYDSL_DUMP_IR',None)
 name+='_prefix';dr=Path('/root/.flydsl/debug')/name;ir=(dr/'19_gpu_module_to_binary.mlir').read_text();asm=(dr/'21_final_isa.s').read_text()
 per=active_workgroups_per_cu(SimpleNamespace(_keepalive=SimpleNamespace(_ir_text=ir)),name,128,266368);assert per==1
 (outdir/(name+'.s')).write_text(asm)
 resources={k:re.findall(r'\.'+k+r':\s*(\d+)',asm) for k in ['vgpr_count','sgpr_count','private_segment_fixed_size','group_segment_fixed_size']}
 return resources
base=build(p['stages4'],'swa4_profile_control');resources_base=compile(base,'swa4_profile_control');base['run']();check(base);bg=GraphBatch(base['run'],16)
groups=[[int(x) for x in g.split(',')] for g in os.getenv('PROFILE_GROUPS','1,2,3,4;5,6,7;8,9,10;11,12,13;14,15,16;17,18,19,20;21,22,23,24;25,26').split(';')]
for group in groups:
 name='swa4_profile_'+'_'.join(map(str,group));print('PROFILE COMPILE',name,flush=True)
 prof=Profile(group);f=build(p['stages4'],name,profile=prof);resources=compile(f,name)
 f['run']();checks=check(f);pg=GraphBatch(f['run'],16)
 samples={'baseline':[],'instrumented':[]}
 for _ in range(3):bg.replay();pg.replay()
 for j in range(30):
  for key,g in [('baseline',bg),('instrumented',pg)][::1 if j%2 else -1]:
   a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();samples[key].append(a.elapsed_time(b)*1000/16)
 # During trace collection trace stores wait 5 million shader cycles after
 # completion, at least 1 ms for clocks <=5 GHz. No global clock polling.
 # Warm with the baseline graph before each collection replay.
 f['args'][-1]=0 if os.getenv('PROFILE_IMMEDIATE','0')=='1' else int(ticks_us*1000);tg=GraphBatch(f['run'],16)
 traces=[]
 for j in range(40):
  bg.replay();tg.replay();torch.cuda.synchronize();check(f)
  a=prof.trace.cpu().numpy().copy();origin=int(a[:,:,-2].min());a[a!=0]-=origin
  
  if a[:,:,-1].max()>=ticks_us*800:
   import numpy as np
   np.save(outdir/'clock_debug.npy',a);print('CLOCK DEBUG',khz,origin,a[:,:,-2].min(),a[:,:,-2].max(),a[:,:,-1].min(),a[:,:,-1].max(),samples,flush=True)
   raise RuntimeError('trace stores may overlap active work')
  traces.append(a)
 import numpy as np
 np.savez_compressed(outdir/(name+'.npz'),traces=np.stack(traces),events=np.array(prof.events),ticks_per_us=ticks_us)
 result={'stages':group,'events':prof.events,'clock_khz':khz,'samples_us':samples,'median_us':{k:statistics.median(v) for k,v in samples.items()},'resources':resources,'baseline_resources':resources_base,'checks':checks,'trace_samples':40,'quiet_shader_cycles':f['args'][-1]*50,'profile_kernel_span_us':float(np.median([x[:,:,-1].max()/ticks_us for x in traces]))}
 (outdir/(name+'.json')).write_text(json.dumps(result,indent=2));print('PROFILE DONE',name,result['median_us'],'span',result['profile_kernel_span_us'],'resources',resources,flush=True)
