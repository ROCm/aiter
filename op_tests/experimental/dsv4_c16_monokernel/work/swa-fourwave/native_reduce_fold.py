"""Fold native independent MoE-reduction waves; preserve native per-wave TDM."""
from pathlib import Path
import re,subprocess,importlib.util
from bridge.export import export
from bridge.stage import runtime_args
ROOT=Path('/work/swa-fourwave')
def build(entry):
 k=entry['compiled'];assert k.metadata.num_warps==8
 p=ROOT/'generated';source=p/'native_reduce_fold.native.ll';source.write_text(k.asm['llir'])
 symbol='native_reduce_virtual';bc=p/(symbol+'.bc');m=export(source,bc,symbol,8,8,preserve_fp_fusion=bool(k.metadata.enable_fp_fusion))
 ll=bc.with_suffix('.ll');s=ll.read_text();h=re.search(r'define void @'+symbol+r'\([^\n]*\) alwaysinline \{\n[^\n]*:\n',s)
 a=h.end();b=s.index('\n}',a);body=s[a:b]
 body=re.sub(r'(?:tail )?call i32 @llvm.amdgcn.workitem.id.x\(\)', 'add i32 %virtual_tid, 0',body)
 body=re.sub(r'(?:tail )?call i32 @llvm.amdgcn.wave.id\(\)', 'lshr i32 %virtual_tid, 5',body)
 prefix='  %physical_tid = call i32 @llvm.amdgcn.workitem.id.x()\n  %virtual_tid = add i32 %physical_tid, %virtual_offset\n'
 head=s[:a].replace('i64 %lds_addr)', 'i64 %lds_addr, i32 %virtual_offset)')
 s=head+prefix+body+s[b:];ll.write_text(s)
 subprocess.run(['/opt/rocm/llvm/bin/llvm-as',str(ll),'-o',str(bc)],check=True)
 values=runtime_args(entry);types=m['types'][:-4];names=[f'a{i}' for i in range(len(values))];ann={'uint64':'Uint64','uint32':'Uint32','float32':'Float32'}
 sig=','.join(f'{n}:fx.{ann[t]}' for n,t in zip(names,types));args=','.join(names)
 code=f"""import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.extern_link import BoundExternFunction
B=BoundExternFunction({symbol!r},{m['types']+['uint32']!r},'void',bitcode_path={str(bc)!r})
@flyc.jit
def body({sig},logical:fx.Int32):
 B({args},fx.Uint32(logical),fx.Uint32(0),fx.Uint32(0),fx.Uint64(0),fx.Uint32(0))
 B({args},fx.Uint32(logical),fx.Uint32(0),fx.Uint32(0),fx.Uint64(0),fx.Uint32(128))
"""
 out=p/'native_reduce_fold_adapter.py';out.write_text(code);sp=importlib.util.spec_from_file_location('native_reduce_fold_adapter',out);mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
 return dict(name='native_route_reduce_wavefold',body=mod.body,module=mod,values=values,types=types,total=16,threads=128,lds=k.metadata.shared)
