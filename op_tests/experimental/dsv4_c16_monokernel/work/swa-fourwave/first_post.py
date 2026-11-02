from pathlib import Path
import importlib.util

def wrap(stage,ptrs,tagged=False,parallel=False):
 p=Path('/work/swa-fourwave/generated/first_post_wrapper.py')
 types=stage['types']+['uint64']*6;names=[f'a{i}' for i in range(len(types))];n=len(stage['types'])
 anns={'uint64':'Uint64','uint32':'Uint32','float32':'Float32'}
 sig=','.join(f'{k}:fx.{anns[t]}' for k,t in zip(names,types))
 calls=(f" C({','.join(names[n:])},fx.Uint32(logical))\n L()\n B({','.join(names[:n])},logical)" if tagged else f" B({','.join(names[:n])},logical)\n L()\n C({','.join(names[n:])},fx.Uint32(logical))")
 if parallel:
  assert not tagged
  calls=f" if logical<16:\n  B({','.join(names[:n])},logical)\n else:\n  C({','.join(names[n:])},fx.Uint32(logical-16))"
 p.write_text(f'''import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.extern_link import BoundExternFunction
C=BoundExternFunction('sw4_first_post',['uint64']*6+['uint32'],'void',bitcode_path='/work/swa-fourwave/first_post.bc')
L=BoundExternFunction('sw4_local',[],'void',bitcode_path='/work/swa-fourwave/runtime.bc')
@flyc.jit
def body({sig},logical:fx.Int32):
{calls}
''')
 sp=importlib.util.spec_from_file_location('first_post_wrapper',p);mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod);mod.B=stage['body']
 return dict(stage,total=32 if parallel else stage['total'],parallel_roles=parallel,body=mod.body,module=mod,types=types,values=stage['values']+ptrs)
