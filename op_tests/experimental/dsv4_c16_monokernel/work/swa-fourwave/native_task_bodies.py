"""Task-level native adapters; a large launch grid is not an exclusion."""
import os
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.extern_link import BoundExternFunction
ACT=BoundExternFunction('ffn_act_virtual',['uint64']*2+['uint32']*2,'void',bitcode_path='/work/swa-fourwave/native_act_fold.bc')
@flyc.jit
def act(out:fx.Uint64,x:fx.Uint64,logical:fx.Int32):
 ACT(out,x,fx.Uint32(logical),fx.Uint32(0))
 ACT(out,x,fx.Uint32(logical),fx.Uint32(128))
def apply(st,p):
 if os.getenv('NATIVE_TOPK','1')=='1':
  # Upper four native waves duplicate inputs/LDS writes. Wave0 writes
  # results and atomics; wave bit2 is masked from LDS addressing.
  from native_stage import build as native_build
  entry=next(e for e in p['routed']['capture'].entries if e['function'].__name__=='_topk')
  mod,args,m=native_build(entry,'/work/swa-fourwave/generated','native_topk_four_active',physical_warps=8)
  st[23]=dict(name='native_topk_four_active',body=mod.body,module=mod,values=args,types=m['types'][:-4],total=m['total'],threads=128,lds=m['lds'])
 if os.getenv('FOLD_REDUCE','1')=='1':
  from native_reduce_fold import build
  entry=next(e for e in p['routed']['capture'].entries if e['function'].__name__=='reduce_grouped_gluon')
  st[29]=build(entry)
 if os.getenv('FOLD_ACT','1')!='1':return
 shared=p['shared'];x=shared['entries'][0][0][4];out=shared['acts'][0][0][0]
 st[18]=dict(name='native_vector_silu_folded',body=act,types=['uint64']*2,values=[out.data_ptr(),x.data_ptr()],threads=128,total=16,lds=0)
