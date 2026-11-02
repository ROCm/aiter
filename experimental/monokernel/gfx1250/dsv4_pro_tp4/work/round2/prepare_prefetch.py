from pathlib import Path
import json,shutil
r=Path(__file__).resolve().parent
s=(r/'originals/gemm_mxfp8.py').read_text()
s=s.replace('    "CTAS_N",\n','    "CTAS_N",\n    "L2_PREFETCH_DISTANCE",\n',1)
s=s.replace('    NUM_BUFFERS: gl.constexpr,\n','    NUM_BUFFERS: gl.constexpr,\n    L2_PREFETCH_DISTANCE: gl.constexpr,\n')
needle='    # ---------------- Prologue ----------------\n'
pos=s.index(needle,s.index('def _gemm_mxfp8_preshuffle_bandwidth_bound_kernel('))
s=s[:pos]+'''    # Prefetch weights beyond the LDS pipeline without allocating more LDS.
    if L2_PREFETCH_DISTANCE > 0:
        for pf in gl.static_range(NUM_BUFFERS - 1 + L2_PREFETCH_DISTANCE):
            if pf < NUM_K_ITER:
                gl.amd.gfx1250.tdm.prefetch(
                    b_desc, [off_bn_tdm, pf * BLOCK_SIZE_K * 16]
                )

'''+s[pos:]
needle='''    for _ in tl.range(0, NUM_K_ITER - (NUM_BUFFERS - 1), loop_unroll_factor=UNROLL):
'''
assert s.count(needle)==1
s=s.replace(needle,needle+'''        if L2_PREFETCH_DISTANCE > 0:
            pf_next = num_loads + L2_PREFETCH_DISTANCE
            if pf_next < NUM_K_ITER:
                gl.amd.gfx1250.tdm.prefetch(
                    b_desc, [off_bn_tdm, pf_next * BLOCK_SIZE_K * 16]
                )
''',1)
(r/'candidate/gemm_mxfp8.py').write_text(s)
s=(r/'originals/gemm_afp8wfp8.py').read_text().replace('            NUM_BUFFERS=num_buffers,\n','            NUM_BUFFERS=num_buffers,\n            L2_PREFETCH_DISTANCE=config["L2_PREFETCH_DISTANCE"],\n',1)
(r/'candidate/gemm_afp8wfp8.py').write_text(s)
shutil.copytree(r/'originals/configs',r/'candidate/configs',dirs_exist_ok=True)
for p in (r/'candidate/configs').glob('*.json'):
 d=json.loads(p.read_text())
 for k,v in d.items():
  if isinstance(v,dict) and 'NUM_BUFFERS' in v:v['L2_PREFETCH_DISTANCE']=0
 p.write_text(json.dumps(d,indent=4)+'\n')
