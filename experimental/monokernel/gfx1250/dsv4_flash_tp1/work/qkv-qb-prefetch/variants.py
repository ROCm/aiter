import harness as h
from combined import complete
State=h.State
CONFIGS={'baseline':(False,False),'l2':(True,False),'tdm':(False,True),'l2_tdm':(True,True),'l2_nonspec':(2,False),'l2_nonspec_tdm':(2,True)}
def run(s,name):
 pf,tdm=CONFIGS[name]
 return complete[(256,)](h.X,s.xs,h.W,h.WS,s.p,h.NW,s.acc,s.count,s.y,s.lp,s.ls,h.Cos,h.Sin,s.kv,h.BW,h.BS,s.qp,s.qc,s.q,s.bar,s.health,BN=128,GRID=256,STYLE=3,U=4,QAMODE=0,PREFETCH=pf,TDM=tdm,**h.kw)
