"""Eligible native operation geometries are requirements, not tuning knobs."""
# (logical CTAs, active threads); geometry verified from native HIP graph nodes.
ELIGIBLE={1:(256,128),3:(32,64),4:(48,64),5:(16,128),6:(256,128),11:(256,128),12:(64,64),13:(64,128),14:(64,128),16:(32,64),17:(128,64),19:(16,64),20:(128,64),21:(16,128),24:(16,32)}
# Larger logical grids are scheduled, not excluded. These two native cooperative
# tasks cannot be embedded unchanged with the four-wave parent limit.
EXCEPTIONS={
 2:'Native eight-wave apply/norm has cross-wave reduction; separate one-wave Sinkhorn makes nine total. Retain four-wave apply plus independent coefficient CTA.',
 26:'Native eight-wave MXFP8 task redistributes scales through LDS across both four-wave halves. Folding whole bodies would read scales before the other half writes them. Retain native-source four-wave specialization.'}
def check(stages):
 for i,(ctas,threads) in ELIGIBLE.items():
  s=stages[i];assert (s['total'],s['threads'])==(ctas,threads),(i,s['name'],s['total'],s['threads'])

def check_tasks(stages):
 check(stages)
 expected={0:(0,0),2:(32,128),7:(260,128),8:(256,128),9:(1024,128),10:(512,32),15:(32,128),18:(16,128),22:(1,128),23:(18,128),25:(257,32),26:(128,128),27:(3072,128),28:(3072,128),29:(16,128),30:(16,128)}
 assert set(expected)|set(ELIGIBLE)==set(range(31))
 for i,(tasks,threads) in expected.items():
  s=stages[i];assert (s['total'],s['threads'])==(tasks,threads),(i,s['name'],s['total'],s['threads'])
 assert stages[8]['entry']['compiled'].metadata.num_warps==1
 assert stages[8]['entry']['kwargs']['grid']==(16,4,16)
 assert stages[9]['entry']['compiled'].metadata.num_warps==4
 assert stages[23]['name']=='native_topk_four_active'
 assert stages[29]['name']=='native_route_reduce_wavefold'
 assert all(s['lds']<=266240 for s in stages)
