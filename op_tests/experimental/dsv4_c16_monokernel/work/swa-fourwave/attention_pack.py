"""Four independent native one-wave attention tasks per physical CTA."""
from pathlib import Path
import re,subprocess,importlib.util
from bridge.export import export
from bridge.stage import runtime_args

def build(entry):
 p=Path('/work/swa-fourwave/generated');k=entry['compiled']
 assert k.metadata.num_warps==1
 source=p/'attention_pack.native.ll';source.write_text(k.asm['llir'])
 stride=(k.metadata.shared+255)//256*256
 assert stride*4<=266240
 manifests=[]
 for wave in range(4):
  symbol=f'attention_pack_w{wave}';bc=p/(symbol+'.bc')
  m=export(source,bc,symbol,1,4);ll=bc.with_suffix('.ll');s=ll.read_text()
  h=re.search(r'define void @'+symbol+r'\([^\n]*\) alwaysinline \{\n[^\n]*:\n',s)
  a=h.end();b=s.index('\n}',a);body=s[a:b]
  body=re.sub(r'(?:tail )?call i32 @llvm.amdgcn.workitem.id.x\(\)',r'add i32 %pack_tid, 0',body)
  body=re.sub(r'(?:tail )?call i32 @llvm.amdgcn.wave.id\(\)',r'add i32 0, 0',body)
  if wave:body=body.replace('@__dynamic_shared_0',f'getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 {wave*stride})')
  prefix='  %pack_hw_tid = call i32 @llvm.amdgcn.workitem.id.x()\n  %pack_tid = and i32 %pack_hw_tid, 31\n'
  s=s[:a]+prefix+body+s[b:];ll.write_text(s)
  subprocess.run(['/opt/rocm/llvm/bin/llvm-as',str(ll),'-o',str(bc)],check=True);manifests.append(m)
 vals=runtime_args(entry);types=manifests[0]['types'];names=[f'a{i}' for i in range(len(vals))]
 ann={'uint64':'Uint64','uint32':'Uint32','float32':'Float32','float64':'Float64'}
 sig=','.join(f'{n}:fx.{ann[t]}' for n,t in zip(names,types));args=','.join(names)
 grid=tuple(entry['kwargs']['grid']);assert grid==(16,4,16),grid
 code='import flydsl.compiler as flyc\nimport flydsl.expr as fx\nfrom flydsl.compiler.extern_link import BoundExternFunction\n'
 for wave in range(4):code+=f'W{wave}=BoundExternFunction("attention_pack_w{wave}",{types!r},"void",bitcode_path="{p}/attention_pack_w{wave}.bc")\n'
 code+=f'@flyc.jit\ndef body({sig},logical:fx.Int32):\n tid=fx.Int32(fx.thread_idx.x)\n'
 for wave in range(4):
  code+=f' {"if" if wave==0 else "elif"} tid<{(wave+1)*32}:\n  task=logical+{wave*256}\n  W{wave}({args},fx.Uint32(task%16),fx.Uint32((task//16)%4),fx.Uint32(task//64),fx.Uint64(0))\n'
 file=p/'attention_pack_adapter.py';file.write_text(code)
 sp=importlib.util.spec_from_file_location('attention_pack_adapter',file);mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
 return dict(name='attention_four_independent_waves',body=mod.body,module=mod,values=vals,types=types[:-4],threads=128,total=256,lds=stride*4,entry=entry)
