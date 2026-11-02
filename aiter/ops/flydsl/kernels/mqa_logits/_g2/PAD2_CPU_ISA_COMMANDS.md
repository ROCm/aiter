# ROUND-PAD2 no-device CPU/ISA verification

Status: **PASS**. 88 authoritative real-ISA compilations; 52 required ISA/dispatch identities; 32 strict selected-ISA checks; 13 CPU gates, all exit 0; 129 unittest tests plus 200 printed script PASS checks. All 37 original non-wrapper functions and all 3 original BF16/FP8/scalar launch branches are source-identical to base.

Final production SHA256: `f10c6cc971ea54595c93f0c395ec0306a10d374adef218b945d938c66eba00bc`.

Full remote evidence SHA256: `59595714d44fd1aca78d36c7d7a51a4c8f508226a383d163ab58c17e71c05580`.
Compact evidence SHA256: `5125ce2e067aabc66ceab0b34a2cd7e7cbe303a5d9f1ad9fc651c27f1ff8d1f5`.

The native suite verifies all 20 benchmark-sized cases, including requests 1/2 prefill and every fp8 scale pointer. The owner PAD suite passed 18 tests (including 20 native counter subcases); the real-source PAD trace passed 1 test spanning 9 shapes. Runtime: Python 3.12.3, FlyDSL 0.3.3.dev886, Torch 2.11.0+rocm10.1.0a20260811-bkc.20260820. The final machine-readable manifest is the authority.

## Scope and safety

- SSH host: `blyu@heliosr-1b114-a07-3.mnb.dcgpu`; Windows key `C:\Users\blyu\.ssh\id_ed25519`.
- Staging: `/data/blyu/qsa/work/pad2-cpu-trace-20261102`.
- Image: `rocm/fw-bringup-mlperf:gfx1250-atom-20261015` (`sha256:d1434db7030a8355a332b2a9237598fadb07a8bf7bb6dbbd081567a13d54686a`).
- Every container uses `--network none`, no `--device` arguments, a read-only source mount, and a read-only frozen public golden mount at `/work/task/MI455_QSA_INDEX_SCORER/golden_baseline`. Every entrypoint and compiler process requires `/dev/kfd` and `/dev/dri` to be absent.
- Direct file import avoids `import aiter`. `COMPILE_ONLY=1` and CPU tensors only. The real production wrapper executes; only `_run` is replaced with `flydsl.compiler.compile` and stream lookup returns `None`. GPU discovery/initialization/synchronization methods are fail-on-call. Outputs are initialized on CPU only to let padding copyback finish; this is not numerical shader execution.
- Base production source was captured before implementation at git `f7c91d5dedf206e41ec13f0daa3fa3e1c82029a8`. Public golden contains reference code, not the optimized kernel's ISA, and is used for CPU correctness rather than optimized-kernel ISA identity.

## Commands on the remote host

```bash
S=/data/blyu/qsa/work/pad2-cpu-trace-20261102
bash "$S/tools/host_pad2_cpu_isa.sh" "$S" base > "$S/base-host.log" 2>&1
bash "$S/tools/host_pad2_cpu_isa.sh" "$S" final3 > "$S/final3-host.log" 2>&1
bash "$S/tools/host_pad2_cpu_isa.sh" "$S" cpu3 > "$S/cpu3-host.log" 2>&1
# Run only after the earlier jobs settle; no duplicate concurrent matrix.
bash "$S/tools/host_pad2_cpu_isa.sh" "$S" native > "$S/native-host.log" 2>&1
python3 -B "$S/tools/verify_pad2_dispatch_isa_cpu.py" --stage "$S" \
  --final-phase final3 --cpu-phase cpu3 \
  --output "$S/evidence/pad2_cpu_isa_evidence.json" \
  --compact-output "$S/evidence/pad2_cpu_isa_summary.json" > "$S/verify-host.log" 2>&1
```

The base archive contains only the captured production source; final archive contains the module and `_g2` scripts/tests, excluding archives and inspection scratch. Source/test-only follow-up syncs are hash-recorded separately.

## Matrix and evidence layout

- Base: 12 fresh-process actual-production dispatch compilations with PAD unset.
- Final: same 12 shapes with PAD unset, `0`, and `1` (36 compilations).
- Exact shapes cover both dtypes, r16/r32 and requests 1/2, including default persistent bf16 prefill and default fast-conversion fp8 prefill.
- Irregular shapes: rows 37, requests 1/3, columns 67, width 17, both dtypes.
- Required identity comparisons: final unset/0 all 12 shapes and final PAD=1 eight exact shapes (32 comparisons). Match final ISA bytes, dispatch name, grid, launch extents, tensor ranks/shapes, and original input/output pointers.
- Full native-size phase: both dtypes, columns 32768/65536, and `(rows,requests)` `(64,16)`, `(16,4)`, `(512,1)`, `(512,2)`, `(512,4)`; width = columns/4. All 20 cases compile against base (PAD unset) and final PAD=1: 40 additional compiles. Match ISA bytes, dispatch/grid/true extents/tensor shapes and every input/output pointer, including fp8 scale.
- Total authoritative compilation count: 88. Required ISA identity comparisons: 52 (32 representative + 20 full native-size). Existing strict ISA checkers run on all 32 real PAD=1 dispatches.
- All original non-wrapper function source segments and all three original BF16/FP8/scalar launch branches are checked for exact source equality against the captured base.
- CPU gates: contract, addressing, Q-LDS, real-source traces r16/r32, K-TDM, routing, fp8 K-TDM, BM64, persistence, fp8 conversion, PAD bucket, PAD bucket trace.
- Each case has `base/results/unset/CASE/dispatch.json` or `final3/results/{unset,0,1}/CASE/dispatch.json`, compiler log, `isa/KERNEL/21_final_isa.s`, and cache. CPU logs/status/version/golden/source manifests are under `cpu3/results`.
- Machine-readable final evidence: `/data/blyu/qsa/work/pad2-cpu-trace-20261102/evidence/pad2_cpu_isa_evidence.json`.

No production wrapper, owner PAD test, or owner PAD documentation was edited by the verifier. No commit/push, device workload, profiler, or GPU runtime was used.
