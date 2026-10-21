import sys
r=sys.argv[1]
def rd(p):
    t=open(p,encoding='utf-8',newline='').read(); nl='\r\n' if '\r\n' in t else '\n'; return t.replace('\r\n','\n'),nl
def wr(p,t,nl): open(p,'w',encoding='utf-8',newline='').write(t.replace('\n',nl))
p=r+'/host/driver_sm.cpp'; t,nl=rd(p)
old='c->b.T, 0u); }\n}\n}\n'; assert t.count(old)==1
t=t.replace(old,'c->b.T, 0u); }\n}\n'); wr(p,t,nl)
p=r+'/interface/sm_abi_v3.h'; t,nl=rd(p)
old='// Kernarg for sm_ran_v1 (r7): the three scalars are the only kernarg the kernel reads before the first vmem load.'
assert t.count(old)==1
t=t.replace(old,'// (r7 history) the v1 kernarg was three scalars; r19 replaces it by SmRanArgsV3 above.'); wr(p,t,nl); print('ok')