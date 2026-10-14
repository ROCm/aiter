from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def coherent_load(pointer, mask=None, other=None):
    return gl.load(pointer, mask=mask, other=other, cache_modifier=".cv")
