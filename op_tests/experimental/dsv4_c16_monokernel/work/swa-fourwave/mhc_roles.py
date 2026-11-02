"""Native mHC main and Sinkhorn partitions assigned to independent CTAs."""
from pathlib import Path
import importlib.util
ROOT=Path('/work/swa-fourwave')
def load(name,text):
 p=ROOT/(name+'.py');p.write_text(text)
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def build(entry,capture):
 src=(ROOT/'source/mhc_native.py').read_text()
 start=src.index('    if WS:\n        gl.warp_specialize(')
 end=src.index('        _reduce_apply_main(*main_args, K)',start)+len('        _reduce_apply_main(*main_args, K)')
 main=load('mhc_main_role',src[:start]+'    _reduce_apply_main(*main_args, K)'+src[end:])
 worker=load('mhc_worker_role',src[:start]+'    _post_sinkhorn_ws(post_ptr,comb_ptr,gemm_ptr,sq_ptr,scale_ptr,base_ptr,M,gemm_stride,rms_eps,sk_eps,post_mult,K)'+src[end:])
 parts=[]
 for name,m,nw in [('main',main,4),('worker',worker,1)]:
  kw=dict(entry['kwargs']);grid=kw.pop('grid');kw.pop('warmup',None);kw['num_warps']=nw
  parts.append(capture(m._mhc_pre_reduce_apply_gfx1250_kernel,grid,entry['args'],kw,'mid_'+name+'_role'))
 a,b=parts;types=a['types']+b['types'];names=[f'a{i}' for i in range(len(types))];n=len(a['types']);anns={'uint64':'Uint64','uint32':'Uint32','float32':'Float32','float64':'Float64'}
 sig=','.join(f'{k}:fx.{anns[t]}' for k,t in zip(names,types))
 mod=load('mhc_role_wrapper',f"""import flydsl.compiler as flyc
import flydsl.expr as fx
@flyc.jit
def body({sig},logical:fx.Int32):
 if logical<16:
  A({','.join(names[:n])},logical)
 else:
  if fx.Int32(fx.thread_idx.x)<32:
   B({','.join(names[n:])},logical-16)
""")
 mod.A=a['body'];mod.B=b['body']
 return dict(a,body=mod.body,module=mod,types=types,values=a['values']+b['values'],total=32,threads=128,lds=max(a['lds'],b['lds']),parallel_roles=True,name='mid_parallel_roles')
