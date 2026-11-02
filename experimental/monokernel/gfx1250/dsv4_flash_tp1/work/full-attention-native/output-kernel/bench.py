exec(open('/work/full-attention-native/output-kernel/native_check.py').read().split('rows=[]')[0])
sys.path.insert(0,'/work/atom-actual-microbench/attention')
from graphbatch import FlatGraphBatch
import statistics
funcs={'actual_atom':lambda:n(o.view(1,64,512)),'fused':lambda:output_kernel[(256,)](*args,counter,health,**kw)}
def staged():
 for stage in range(5):output_stage[(256,)](*args,STAGE=stage,**kw)
funcs['five_stage_control']=staged
entries=[]
for name,fn in funcs.items():
 for _ in range(3):fn()
 gr=FlatGraphBatch(fn,128)
 for _ in range(5):gr.replay()
 entries.append((name,gr))
results=[]
for rnd in [1,2]:
 samples={name:[] for name,gr in entries}
 for i in range(40):
  for name,gr in entries if i%2 else reversed(entries):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();gr.replay();b.record();b.synchronize();samples[name].append(a.elapsed_time(b)*1000/128)
 row={'round':rnd,'us':{name:statistics.median(v) for name,v in samples.items()},'samples_us':samples};results.append(row);print(row['us'],flush=True)
(root/'timings.json').write_text(json.dumps(results,indent=2))
obj=output_kernel[(256,)](*args,counter,health,**kw)
(root/'output.amdgcn').write_text(obj.asm['amdgcn']);(root/'output.llir').write_text(obj.asm['llir']);(root/'resources.json').write_text(json.dumps({'vgpr':obj.n_regs,'shared':obj.metadata.shared,'scratch':obj.metadata.global_scratch_size,'health':int(health.item())},indent=2))
