from pathlib import Path
import json,hashlib,datetime
from zoneinfo import ZoneInfo
p=Path(__file__).resolve().parent
root=p/'native-c16'
load=lambda sub,name:json.loads((root/sub/(name+'.json')).read_text())
parts=[('Input FP8 quant','quant'),('QA projection','part_qa'),('QA RMSNorm + FP8 quant','norm'),('QB projection','part_qb'),('QK RMSNorm + RoPE + cache','rope'),('Inverse RoPE + FP8 quant','inverse'),('Grouped OA projection','part_oa'),('OB projection','part_ob')]
rows=[]
for label,name in parts:
 d=load('final_parts',name);t=d['rounds'][-1]['median_us'];rows.append((label,name,t,d['validation']))
lines=['# C16 native-design parity investigation','',f'Recorded {datetime.datetime.now(ZoneInfo("America/Chicago")).isoformat(timespec="seconds")} (America/Chicago).','',
'Flash TP1 on gfx1250, C16 decode projection shapes, position 1023 (1k input / 1k output workload setting). These are isolated projection-chain measurements using saved model weights, not model-wide serving results. Four GPUs ran independent experiments. No TP4/e2e TPOT, TTFT, GSM8K, or image publication was performed in this investigation. C16 serving remains disabled.','',
'## Outcome','',
'The standalone native-shaped Gluon kernels now match or beat native for every GEMM. RMSNorm and QK RoPE remain within 10%. Outputs match native exactly on the final fixture; both complete candidate chains also matched every intermediate and output across 64 changing inputs after matching native ISA arithmetic. Literal single-launch composition still regresses. Do not promote these composed candidates over the previously selected implementation.','',
'## Standalone kernels','',
'All times below are microseconds from round 2. Each round uses flat HIP graphs with 32 repetitions, five warmup replays, and 40 alternating paired samples. These are paired native-versus-Gluon measurements, not sums of profiler timestamps.','',
'| Operation | Actual native ATOM | Native-shaped Gluon | Difference |','|---|---:|---:|---:|']
for label,name,t,v in rows:lines.append(f'| {label} | {t["native"]:.3f} | {t["faithful"]:.3f} | {(t["faithful"]/t["native"]-1)*100:+.1f}% |')
lines+=['','Exact native geometry:', '',
'| Stage | Logical CTAs | Waves/CTA | Tile / buffers |','|---|---:|---:|---|',
'| Input quant | 32 | 2 | 16 rows × 128 columns; native HIP global loads |',
'| QA | 48 | 2 | M16 N32 K1024; 4 buffers; no split-K |',
'| QA norm/quant | 16 | 4 | One row, 128 threads × 8 elements |',
'| QB | 256 | 4 | M16 N128 K512; 2 buffers; no split-K |',
'| QK norm/RoPE/cache | 260 | 4 | One wave per row; four rows × one Q head or KV |',
'| Inverse RoPE/quant | 512 | 1 | One row × 1024 columns; 32 elements/thread |',
'| Grouped OA | 256 | 4 | M32 N32 K1024; 4 buffers |',
'| Intermediate quant | 64 | 2 | 16 rows × 128 columns |',
'| OB | 64 | 4 | M16 N64 K512; 4 buffers in faithful control |','',
'Native nonbatched activation scales physically use K-major storage, despite the returned tensor advertising a contiguous logical M×Kgroup shape. Grouped OA uses M-major scales. The corrected harness compares the physical scale layout explicitly. Earlier exploratory elementwise logs showing scale differences were harness comparison errors, not kernel quantization differences; use `final_parts`.','',
'## Progressive composition controls','',
'These keep each native stage’s logical work partition. A fixed grid of 256 CTAs virtualizes 260 QK tasks and 512 inverse-RoPE tasks. Warp specialization preserves the two-wave QA and one-wave inverse stages, but requires eight physical waves for the mixed-stage kernel. This physical composition cost is measured, not claimed identical to a native standalone launch.','',
'| Cumulative region | Actual native | Separate Gluon launches | Single launch |','|---|---:|---:|---:|']
labels={'q2':'Input quant → QA','q3':'Input quant → QA → norm/quant','q4':'Input quant → QA → norm/quant → QB','q5':'Input quant → QA → norm/quant → QB → QK RoPE/cache','o2':'Inverse RoPE/quant → OA','o3':'Inverse RoPE/quant → OA → quant','o4':'Inverse RoPE/quant → OA → quant → OB'}
for name in ['q2','q3','q4','q5','o2','o3','o4']:
 t=load('final_progressive',name)['rounds'][-1]['median_us'];lines.append(f'| {labels[name]} | {t["native"]:.3f} | {t["unfused_gluon"]:.3f} | {t["composed"]:.3f} |')
lines+=['','The controls demonstrate where composition loses parity; they do not isolate every cycle into polling, resource reservation, and instruction-scheduling costs. Adding QB increases the composition gap substantially. For output, the first inverse-RoPE→OA transition already loses. OB also slows when measured after OA, despite competitive standalone warm-weight timing.','',
'## Separate advantage-set trials','',
'After the faithful controls, the advantage trial uses four physical waves, two duplicated logical waves for QA/ordinary quant, groups four independent native inverse-RoPE waves into each of 128 CTAs, uses only required producer groups, performs bounded L2 weight prefetch, stages first QB/OA/OB weight tiles early through TDM, and increases OB buffers to six. QA/QB/OA/OB tile dimensions are unchanged. This is an explicitly different composition variant, not substituted into the faithful baseline.','',
'| Complete chain | Actual native | Separate adapted launches | Single launch with changes |','|---|---:|---:|---:|']
for name in ['q5','o4']:
 t=load('final_advantage',name)['rounds'][-1]['median_us'];lines.append(f'| {labels[name]} | {t["native"]:.3f} | {t["unfused_gluon"]:.3f} | {t["composed"]:.3f} |')
lines+=['','OB buffer sweep (all other trial settings held fixed, paired round-2 results; hardware workers differ):','',
'| OB buffers | Native chain | Separate Gluon chain | Single launch |','|---:|---:|---:|---:|']
for nb in [4,5,6,7]:
 t=load('ob_buffers'+str(nb),'o4')['rounds'][-1]['median_us'];lines.append(f'| {nb} | {t["native"]:.3f} | {t["unfused_gluon"]:.3f} | {t["composed"]:.3f} |')
lines+=['','## Concrete fixes and findings','',
'1. **OA scale mapping:** the original Gluon translation converted through a layout requiring inter-wave LDS exchanges. The assembly had 66 CTA barrier signal/wait pairs. Loading scales directly in the WMMA scale layout removes those exchanges and makes native-sized OA faster than native.','2. **RoPE coefficient loads:** repeated element-index loads generated 64 scalar `global_load_u16` instructions in inverse RoPE. Loading consecutive coefficient pairs and expanding in registers restores vector loads, reducing inverse RoPE/quant from approximately 3.40 to 1.60 µs.','3. **Native arithmetic:** FlyDSL enables LLVM fast math. Its generated QK ISA uses an explicit 8/4/2/1 local reduction tree, descending wave XOR reduction, FMA for RMS epsilon, and `fma(value, cos, signed_mate*sin)` for RoPE. The default generic Gluon sum and a different odd-element FMA produced small BF16 differences. `ROPE_ORDERED=2 ROPE_FMA=7` matches the generated native arithmetic and passed 64 changing inputs exactly.','4. **Masked prefetch SCC clobber:** the shared helper used `s_and_saveexec_b32`, which changes SCC, without declaring `~{scc}`. ISA showed a CTA comparison before the inline assembly and a conditional branch afterward using the overwritten SCC. Some producers were therefore skipped. Added the clobber in the local helper and backported that single-line fix to `/work/concurrency-rollout/masked_prefetch.py`. Patch saved as `masked_prefetch_scc.patch`. Existing parent archives remain unchanged. This is the only backport outside this new experiment directory.','5. **Buffer-load polling failure:** `POLL=3` intermittently timed out even after device-scope TDM and epoch changes. Scalar load controls (`POLL=1/2`), non-temporal global load (`POLL=4`), and atomic-read control (`POLL=0`) passed the tested replay cases. The final scripts use scalar device-scope hot-temporal loads (`POLL=2`). The precise buffer-polling failure cause is not proven; do not reuse its faster timings.','6. **OB cold-chain buffering:** four native buffers were competitive in standalone warm-weight timing but insufficient in this Gluon composition after OA. Six buffers recovered the separate-launch chain to below native. This shows why a standalone kernel timing alone is insufficient.','',
'## Validation and exclusions','',
'- Final single-fixture checks: exact output and scale agreement for all eight standalone operations and all progressive chain intermediates.','- QKV: `native-c16/isa_rounding_2_7/stress_q5.json` has 64 changing inputs, poisoned output buffers, exact agreement for input quant/scales, QA, norm/quant/scales, QB, Q, KV, and cache; zero synchronization-health failures.','- Output: `native-c16/stress_o/stress_o4.json` has 64 changing inputs and exact agreement throughout; zero health failures.','- Progressive harness poisons outputs before the first fused execution and checks twenty individual replays before graph timing. It checks health again after timing. Earlier exploratory comparisons did not poison outputs; final tables use only the strengthened harness.','- Final timing kernels have zero spills; resource metadata and ISA are adjacent to each measurement.','- Reject old `buffer_cg`, `occupancy_check`, `epoch_*`, `diagnostic_*`, original `adv_*` (before SCC fix), and failed `rounding_*` variants. `cooperative_*` additionally used the wrong HIP occupancy API and are invalid. The final harness uses `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`.','- These tests are not model-wide accuracy/GSM8K or serving throughput validation. No C16 serving rollout.','',
'## Reproduction','',
'Container: `vigokhal_dsv4_tp4`; host directory mounts at `/work/c16-native-parity`. Complete parent artifacts provide the fixtures and dependency directories. Model config is `/data/DeepSeek-V4-Flash`.','',
'```bash','docker exec -e HIP_VISIBLE_DEVICES=0 -e EXPERIMENT=recheck vigokhal_dsv4_tp4 /work/c16-native-parity/run_native_parity.sh parts qa','docker exec -e HIP_VISIBLE_DEVICES=1 -e EXPERIMENT=recheck vigokhal_dsv4_tp4 /work/c16-native-parity/run_native_parity.sh elements rope','docker exec -e HIP_VISIBLE_DEVICES=2 -e EXPERIMENT=recheck vigokhal_dsv4_tp4 /work/c16-native-parity/run_native_parity.sh progressive q 5','docker exec -e HIP_VISIBLE_DEVICES=3 -e EXPERIMENT=recheck vigokhal_dsv4_tp4 /work/c16-native-parity/run_native_parity.sh advantage o 4','docker exec -e HIP_VISIBLE_DEVICES=0 -e EXPERIMENT=recheck_stress vigokhal_dsv4_tp4 /work/c16-native-parity/run_native_parity.sh stress q 5','```','',
'Additional part names: `qb oa ob`; elements: `quant norm inverse`; progressive stages: QKV `q 2/3/4/5`, output `o 2/3/4`. Use one process per GPU for timing. The benchmark reports round 2. For future serving, start a server once, run the client twice, and report only the second client.','',
'## Next work','',
'Preserve this native-parity foundation. Optimize boundary scheduling without changing GEMM geometry first. Specifically: remove or duplicate the small input-quant dependency where beneficial; stage QB weights during QA/norm; release KV independently before QB completion; make OB activation TDM wait on the needed K-slice producers rather than all 64 quant CTAs; reduce per-edge polling/invalidation overhead with a proven memory-ordering protocol. Add each change as a separately measured ablation, retain exact intermediate checks, and rerun progressive composition. The single-launch candidates here are slower than both native and the previously selected kernels; no rollout is justified yet.','']
(p/'REPORT.md').write_text('\n'.join(lines))
summary={'parts':[{ 'operation':r[0],'timing_us':r[2],'validation':r[3]} for r in rows], 'progressive':{k:load('final_progressive',k) for k in labels},'advantage':{k:load('final_advantage',k) for k in ['q5','o4']},'validation':{'q':load('isa_rounding_2_7','stress_q5')['maxima'],'o':load('stress_o','stress_o4')['maxima']},'promoted':False}
(p/'RESULTS.json').write_text(json.dumps(summary,indent=2))
files=[f for f in p.rglob('*') if f.is_file() and f.suffix in ['.py','.sh','.cu','.ll','.s'] and '__pycache__' not in f.parts]
(p/'SOURCE_SHA256.json').write_text(json.dumps({str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)},indent=2))
print('Wrote REPORT.md, RESULTS.json, SOURCE_SHA256.json',len(files),'source files')
