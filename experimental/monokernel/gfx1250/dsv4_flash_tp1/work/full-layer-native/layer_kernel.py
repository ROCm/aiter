from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from full_kernel import full_attention
from ffn_kernel import ffn_body
from combined import sync
@g.jit
def full_layer(A,F,W1S:gl.constexpr,S1S:gl.constexpr,W2S:gl.constexpr,S2S:gl.constexpr,POST_SHUFFLED:gl.constexpr=False):
 full_attention(*A,T=128,PAGES=128,WRITE_SLOT=127,SPLITS=4,SHUFFLED_INPUT=True,DO_POST=True)
 sync(F[-2],F[-1],256,3)
 ffn_body(*F,W1S,S1S,W2S,S2S,True,POST_SHUFFLED)
