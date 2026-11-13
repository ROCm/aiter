import runpy,sys
from pathlib import Path
from aiter.ops.flydsl import ep32_native80 as impl

calls=0
original=impl.dispatch
def counted(*a,**kw):
    global calls
    calls+=1
    return original(*a,**kw)
impl.dispatch=counted
test=Path(__file__).resolve().parent.parent/'aiter/op_tests/test_moe_ep.py'
sys.path.insert(0,str(test.parent))
sys.argv=[str(test)]+sys.argv[1:]
runpy.run_path(str(test),run_name='__main__')
print('native GEMM2 calls:',calls,flush=True)
assert calls>0,'native path was not exercised'
