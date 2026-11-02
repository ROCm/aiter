from pathlib import Path
root=Path('/home/vigokhal/dsv4-monokernel'); out=root/'c8-qkv-deps'
s=(root/'concurrency-rollout/qa_batch.py').read_text()
finish=s[s.index('@g.jit\ndef finish('):s.index('@g.jit\ndef qa_batch(')]
finish=finish.replace('Positions=None):','Positions=None,SPLITS:gl.constexpr=16,ATOMIC:gl.constexpr=False):').replace('for k in gl.static_range(16):v+=gl.load(P+k*M*1536+ROW*1536+BASE+i)','for k in gl.static_range(1 if ATOMIC else SPLITS):v+=gl.load(P+k*M*1536+ROW*1536+BASE+i)')
header=s[:s.index('@g.jit\ndef finish(')]
(out/'qa_candidate.py').write_text(header+finish)
q=(root/'batch-full/qkv_full_batch.py').read_text().replace('from qa_batch import qa_batch','from qa_candidate import qa_candidate\nfrom combined_scaled import wait_target\nfrom device_scope_helpers import dev_load,dev_store')
q=q.replace('QB_BYTES=33554432):','QB_BYTES=33554432,Ready=None,SPLITS:gl.constexpr=16,BN:gl.constexpr=128,LOCAL:gl.constexpr=True,PRE:gl.constexpr=True,SEM:gl.constexpr=True,ATOMIC:gl.constexpr=False):')
q=q.replace(' qa_batch(', ' qa_candidate(').replace('QB_BYTES=QB_BYTES)','QB_BYTES=QB_BYTES,SPLITS=SPLITS,BN=BN,LOCAL=LOCAL,PRE=PRE,ATOMIC=ATOMIC)')
q=q.replace(' sync(Counter,Health,256,SYNC_STYLE)\n SA:', ''' if SEM:
  target=dev_load(Ready,pid+1)+M
  if pid<M:
   gl.inline_asm_elementwise('s_wait_storecnt 0\\n s_mov_b32 $0, 0',constraints='=s,~{memory}',args=[],dtype=gl.int32,is_pure=False,pack=1)
   gl.barrier()
   gl.atomic_add(Ready,M//M,sem='release',scope='gpu')
  wait_target(Ready,Health,target,SYNC_STYLE)
  dev_store(target,Ready,pid+1)
 else:
  sync(Counter,Health,256,SYNC_STYLE)
 SA:''')
(out/'qkv_candidate.py').write_text(q)
(out/'DESIGN.txt').write_text('C8 QKV dependency ablations. 256 CTAs x 128 threads.\nLOCAL removes B1 using redundant per-producer BF16 TDM quantization. PRE moves bounded QB prefetch before B2. SEM replaces B3 with 8 QA producer arrivals and per-CTA monotonic epochs (Ready[257] int32, initialized once, reused in sequential graph launches). B2/B4 retain global counter. QA scratch: [splits,8,1536] FP32, separate writers. Atomic path clears [8,1536] before a separate initialization barrier, then GPU FP32 atomic adds; B2 ensures completion before norms.\nS16 BN128:192 projection CTAs; S8 BN64:192; S4 BN32:192. Also test S8/4 BN128 (96/48 CTAs). Bounds and FP8 rounding preserved. No serving changes.\nResource gate: zero spills, LDS <=320KiB, resident256 grid on256CUs. Inspect actual compiled assembly before launch.\n')
