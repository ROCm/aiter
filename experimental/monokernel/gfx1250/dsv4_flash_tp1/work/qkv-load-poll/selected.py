import harness as h
from harness import project_fused,qb_kernel,cbase,kw
CONFIGS={'prior':(1,64,4,1),'qb_unroll2':(1,128,4,2),'qb_unroll4':(1,128,4,4),'qa2_qb_unroll8':(2,128,4,8)}
def run(s,name):
 au,bn,waves,u=CONFIGS[name]
 a=project_fused[(192,)](h.X,s.xs,h.W,h.WS,s.p,h.NW,s.acc,s.count,s.y,s.lp,s.ls,h.Cos,h.Sin,s.kv,INLINE=True,TAIL=True,**dict(cbase,UNROLL=au),**kw)
 b=qb_kernel[(32768//bn,)](s.lp,s.ls,h.BW,h.BS,s.qp,s.qc,h.Cos,h.Sin,s.q,TAIL=True,BN=bn,WAVES=waves,U=u,num_warps=waves,enable_fp_fusion=False)
 return a,b
