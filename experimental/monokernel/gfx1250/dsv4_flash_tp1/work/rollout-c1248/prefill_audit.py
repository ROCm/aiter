"""Opt-in nonblocking GPU event audit. No changes to model dispatch or tensors."""
import functools,os,time,json
from pathlib import Path

def audit_prefill(fn):
 @functools.wraps(fn)
 def wrapped(self,*args,**kwargs):
  prefix=os.environ.get('ATOM_MK_PREFILL_AUDIT')
  if not prefix:return fn(self,*args,**kwargs)
  from atom.utils.forward_context import get_forward_context
  c=get_forward_context().context
  if not c.is_prefill or c.is_dummy_run:return fn(self,*args,**kwargs)
  import torch
  pending=getattr(self,'_mk_prefill_events',[])
  ready=[];remaining=[]
  for start,end,row in pending:
   if end.query():row['gpu_ms']=start.elapsed_time(end);ready.append(row)
   else:remaining.append((start,end,row))
  if ready:
   with open(prefix+'.jsonl','a') as f:
    for row in ready:f.write(json.dumps(row)+'\n')
  self._mk_prefill_events=remaining
  try:tag=Path(prefix+'.tag').read_text().strip()
  except FileNotFoundError:tag='startup'
  layers=getattr(getattr(self.model,'model',None),'layers',[])
  calls_before=sum(getattr(getattr(l,'_experimental_full_layer',None),'calls',0) for l in layers)
  start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  row=dict(tag=tag,wall_start=time.time(),scheduled_bs=c.scheduled_bs,tokens=int(args[0].numel()) if args else None,scheduled_tokens=c.scheduled_tokens,running_bs=c.running_bs,pid=os.getpid())
  batch=args[1] if len(args)>1 else kwargs.get('batch')
  if batch is not None:
   row['req_ids']=[str(x) for x in batch.req_ids]
   row['num_scheduled_tokens']=[int(x) for x in batch.num_scheduled_tokens]
   row['context_lens']=[int(x) for x in batch.context_lens]
  trigger=Path(prefix+'.trace')
  profile=None
  if trigger.exists():
   trigger.unlink();profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]);profile.__enter__()
  start.record();t=time.perf_counter()
  result=fn(self,*args,**kwargs)
  end.record();row['cpu_enqueue_ms']=(time.perf_counter()-t)*1000
  calls_after=sum(getattr(getattr(l,'_experimental_full_layer',None),'calls',0) for l in layers)
  row['monokernel_calls_during_prefill']=calls_after-calls_before
  assert calls_after==calls_before,'Monokernel executed during prefill'
  self._mk_prefill_events.append((start,end,row))
  if profile is not None:
   profile.__exit__(None,None,None);profile.export_chrome_trace(prefix+'-'+str(int(time.time()))+'-prefill-trace.json')
  return result
 return wrapped
