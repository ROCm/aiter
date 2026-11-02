from pathlib import Path
s=Path('/work/modelwide/output_scaled.py').read_text()
a=s.index('@g.jit\ndef split_project');b=s.index('@g.jit\ndef reduce_quant',a)
s=s[a:b].replace('GROUPED:gl.constexpr):','GROUPED:gl.constexpr,M:gl.constexpr):')
s=s.replace('[1,128]','[16,128]').replace('[2,1,128]','[2,16,128]')
s=s.replace('shape=[1,K],strides=[K,1]','shape=[M,K],strides=[32768 if GROUPED else K,1]')
a=s.index('        vx=gl.amd.cdna4.async_copy.load_shared_relaxed');b=s.index('        gl.barrier()',a)
s=s[:a]+'        av=gl.amd.cdna4.async_copy.load_shared_relaxed(xsh.index(b%2),AL)\n'+s[b:]
s=s.replace('        ax=gl.load(XS+xb//128+first+b).to(gl.uint32)','        sr=gl.arange(0,16,layout=gl.SliceLayout(1,ASL))\n        sk=gl.arange(0,4,layout=gl.SliceLayout(0,ASL))\n        ax=gl.load(XS+sr[:,None]*(256 if GROUPED else K//128)+xb//128+first+b+sk[None,:]*0,sr[:,None]<M,0).to(gl.uint32)')
s=s.replace('ks*N+on[None,:]+om[:,None]*0,om[:,None]==0','ks*M*N+on[None,:]+om[:,None]*N,om[:,None]<M')
head='''from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from device_scope_helpers import dev_load,dev_store
from combined_scaled import sync
from tail import quant_scale
'''
Path('/work/batch-full/output_batch.py').write_text(head+s)
