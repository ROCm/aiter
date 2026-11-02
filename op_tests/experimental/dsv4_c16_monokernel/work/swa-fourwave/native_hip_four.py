import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.extern_link import BoundExternFunction
N=BoundExternFunction('native_norm',['uint64']*4,'void',bitcode_path='/work/swa-fourwave/native_qnorm_four.bc')
P=BoundExternFunction('mhc_native_pre_gemm_shuffled',['uint64']*4+['uint32']*3,'void',bitcode_path='/work/swa-fourwave/native_pre_four.bc')
@flyc.jit
def norm(out:fx.Uint64,scale:fx.Uint64,x:fx.Uint64,w:fx.Uint64,logical:fx.Int32):
 N(out,scale,x,w)
@flyc.jit
def pre(o:fx.Uint64,sq:fx.Uint64,x:fx.Uint64,w:fx.Uint64,logical:fx.Int32):
 P(o,sq,x,w,fx.Uint32(0),fx.Uint32(logical%2),fx.Uint32(logical//2))
