import harness as h
from combined import complete
State=h.State
CONFIGS={'global_control':(False,False),'tdm_late':(True,False),'tdm_early':(True,True)}
def run(s,name):
 tdm,early=CONFIGS[name]
 return complete[(256,)](h.X,s.xs,h.W,h.WS,s.p,h.NW,s.acc,s.count,s.y,s.lp,s.ls,h.Cos,h.Sin,s.kv,h.BW,h.BS,s.qp,s.qc,s.q,s.bar,s.health,BN=128,GRID=256,STYLE=3,U=4,QAMODE=0,ALLTDM=tdm,EARLY=early,**h.kw)
