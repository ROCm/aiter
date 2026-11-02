"""Opt-in deferred per-wave timeline; production schedule is unchanged by default."""
import re
class Profile:
 def __init__(self,stages,quiet_ticks=0):
  self.stages=set(stages);self.quiet_ticks=quiet_ticks;self.events=[]
 def instrument(self,text):
  import torch
  def stamp(label,indent):
   n=len(self.events);self.events.append(label)
   return f'{indent}v{n}=T()\n'
  # Make the two possible QK task iterations explicit so each gets its own probe.
  if 7 in self.stages:
   begin=text.index('  for task in range(rank,260,256):')
   end=text.index('\n rank=',begin)
   body=text[begin:end].splitlines()[1:]
   text=text[:begin]+'  task=rank\n'+'\n'.join(x[1:] for x in body)+'\n  if rank<4:\n   task=rank+256\n'+'\n'.join(body)+text[end:]
  start=text.index(' rank=');end=text.index('@flyc.jit')
  chunks=re.split(r'(?=^ rank=)',text[start:end],flags=re.M)
  out='';order=iter(self.order)
  for chunk in chunks:
   if not chunk:continue
   stage=next(order)
   if stage not in self.stages:out+=chunk;continue
   lines=chunk.splitlines(True);out+=''.join(lines[:2]);out+=stamp(f'{stage}:stage_start','  ')
   edge=0;pub=0
   for line in lines[2:]:
    stripped=line.strip();indent=line[:len(line)-len(line.lstrip())]
    if stripped.startswith('G('):
     label=f'{stage}:get{edge}';edge+=1
     out+=stamp(label+':begin',indent)
     n=len(self.events);self.events.append(label+':ready')
     out+=indent+f'v{n}='+stripped.replace('G(','GP(',1)+'\n'
     out+=stamp(label+':end',indent)
    elif stripped.startswith('P(') or stripped=='L()':
     label=f'{stage}:publish{pub}' if stripped.startswith('P(') else f'{stage}:local{pub}';pub+=1
     out+=stamp(label+':begin',indent)+line+stamp(label+':end',indent)
    else:out+=line
   out+=stamp(f'{stage}:stage_end','  ')
  # Epoch start/end permit global-clock interval checks and profiling-overhead checks.
  self.events+=['kernel_start','kernel_end']
  n=len(self.events)
  declarations=''.join(f' v{i}=fx.Uint64(0)\n' for i in range(n-2))
  ep=' ep=E(epochs)\n'
  prefix=text[:start].replace(ep,ep+' trace_start=T()\n'+declarations)
  suffix=f' trace_end=T()\n Q(trace_start,quiet_ticks)\n'
  for i in range(n):
   v=f'v{i}' if i<n-2 else ('trace_start' if i==n-2 else 'trace_end')
   suffix+=f' D(trace,fx.Uint32({n}),fx.Uint32({i}),{v})\n'
  imports="""PBC='/work/swa-fourwave/profile_runtime.bc'
T=BoundExternFunction('sw4_stamp',[],'uint64',bitcode_path=PBC)
GP=BoundExternFunction('sw4_get_profile',['uint64','uint64']+['uint32']*5,'uint64',bitcode_path=PBC)
Q=BoundExternFunction('sw4_trace_quiet',['uint64','uint64'],'void',bitcode_path=PBC)
D=BoundExternFunction('sw4_trace_store',['uint64','uint32','uint32','uint64'],'void',bitcode_path=PBC)
"""
  text=prefix+out+suffix+text[end:]
  text=text.replace('@flyc.kernel',imports+'@flyc.kernel',1)
  self.trace=torch.zeros((256,4,n),device='cuda',dtype=torch.int64)
  return text
