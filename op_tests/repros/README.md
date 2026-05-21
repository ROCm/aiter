# aiter ASM PA crash — standalone reproducer

A minimal aiter-only reproducer for the HIP illegal-memory crash observed in
production (Kimi-K2.5-MXFP4 + Eagle3 spec-decode on 8x MI355 / gfx950) when
ATOM uses ASM-force paged-attention.

## TL;DR fingerprint

| | value |
|---|---|
| **Kernel** | `pa_bf16_pertokenFp8_gqa8_1tg_4w_mtp_msk1.co` (gfx950) |
| **Trigger** | `batch_size == 128` AND `qlen == 3` (specifically — not a boundary, not `total_qo`) |
| **Shape** | GQA=8 (8 Q heads / 1 KV head), head_size=128, block_size=16 |
| **KV dtype** | **fp8 per-token quant only** — bf16 (`pa_bf16_noquant_gqa8_1tg_4w_mtp_msk1.co`) does NOT crash at the same shape |
| **Failure** | HIP illegal memory access, reported asynchronously. Surfaces at the next `hipModuleLaunchKernel` call (so call N+1 errors when call N was the offender). |
| **Min repeats to crash** | 2–3 invocations of the bad shape. Single call sometimes survives, 2nd or 3rd reliably trips it. |
| **Sequence dependency** | None — pure shape bug. Calling the bad shape in a fresh process triggers it. |
| **Concurrency dependency** | None — single-stream, `AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1` still crashes. |

## How to reproduce in ≤30 seconds (inside the eagle3 container)

```bash
cd /app/aiter-test    # or wherever aiter is importable
AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 \
    python /home/hyi_qle/yhl/project/002-kimi-pa-asm-fix/aiter_repro/pa_asm_fp8_repeat_call.py \
        --bs 128 --ctx 1024 --qlen 3 --kv-dtype fp8 --n-repeat 5
```

Expected output on a buggy build:
```
[aiter] LoadKernel: _ZN5aiter40pa_bf16_pertokenFp8_gqa8_1tg_4w_mtp_msk1E
        hsaco: /app/aiter-test/hsa//gfx950/pa/pa_bf16_pertokenFp8_gqa8_1tg_4w_mtp_msk1.co
[AITER] /app/aiter-test/csrc/include/aiter_hip_common.h:244
        fail to call hipModuleLaunchKernel(...) ---> [HIP error](an illegal memory access)
Aborted (core dumped)
```

Negative controls (all pass on the same build):
```bash
# bf16 KV — same bs=128 qlen=3, different kernel, no crash:
python pa_asm_fp8_repeat_call.py --bs 128 --ctx 1024 --qlen 3 --kv-dtype bf16

# fp8 KV, bs off by one — no crash:
python pa_asm_fp8_repeat_call.py --bs 127 --ctx 1024 --qlen 3 --kv-dtype fp8
python pa_asm_fp8_repeat_call.py --bs 129 --ctx 1024 --qlen 3 --kv-dtype fp8

# fp8 KV, qlen off by one — no crash:
python pa_asm_fp8_repeat_call.py --bs 128 --ctx 1024 --qlen 2 --kv-dtype fp8
python pa_asm_fp8_repeat_call.py --bs 128 --ctx 1024 --qlen 4 --kv-dtype fp8

# same total_qo=384 via other (bs, qlen) — no crash:
python pa_asm_fp8_repeat_call.py --bs 192 --ctx 1024 --qlen 2 --kv-dtype fp8   # 192*2=384
python pa_asm_fp8_repeat_call.py --bs  96 --ctx 1024 --qlen 4 --kv-dtype fp8   # 96*4=384
```

## Sweep data (each cell = 5 repeated calls, fresh process, `AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1`)

KV dtype = fp8, head_size=128, block_size=16, num_blocks=8192, GQA=8 (num_q_heads=8, num_kv_heads=1).

qlen=3 sweep over batch_size (ctx_len=1024, fp8 KV):

| bs | 32 | 64 | 96 | 124 | 125 | 126 | 127 | **128** | 129 | 130 | 144 | 192 | 256 | 512 |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| result | OK | OK | OK | OK | OK | OK | OK | **CRASH** | OK | OK | OK | OK | OK | OK |

bs=128 sweep over qlen (ctx_len=1024, fp8 KV):

| qlen | 1 | 2 | **3** | 4 | 5 | 6 | 7 | 8 |
|------|---|---|---|---|---|---|---|---|
| result | OK | OK | **CRASH** | OK | OK | OK | OK | OK |

bs=128, qlen=3, ctx_len sweep:

| ctx_len | 128 | 1024 | 2048 | 4096 | 6724 | 8192 | 16384 |
|---------|---|---|---|---|---|---|---|
| result | CRASH | CRASH | CRASH | CRASH | CRASH | CRASH | CRASH |

→ ctx_len does not affect; bs=128 ∧ qlen=3 is the entire trigger.

## What's in this directory

| file | purpose |
|---|---|
| `pa_asm_fp8_repeat_call.py` | **The minimal reproducer.** Repeats one `(bs, ctx, qlen)` call N times. Use for bisection. |
| `pa_asm_fp8_min_repro.py` | Single-call variant (does NOT crash by itself — bug needs ≥2 calls). Useful for checking shapes that are individually safe. |
| `pa_asm_fp8_seq_repro.py` | Replays a fixed call sequence from the stress driver and lets you also `--repeat-only-bad` to confirm sequence-independence. |
| `pa_asm_crash_repro.py` | Original stress driver that mimics ATOM's call pattern (random shape mix, multi-stream). Useful for end-to-end "would this build crash under prod-like load?" |
| `pa_asm_fp8_shape_sweep.py` | Sweep wrapper (4 qlens × 9 ctx × 5 bs). Each cell forks a fresh process. |
| `README.md` | This file. |

## Production correlation

- ATOM ASM-force path calls `aiter.pa_fwd_asm` with exactly these shapes during
  the Eagle3 spec-decode target step (Kimi MLA absorbed → 1 KV head, TP=8 →
  8 Q heads per rank → GQA=8). Draft tokens are 3 per step (eagle3 emits 3),
  so `qlen=3`.
- The bench runs at concurrency=128. When the scheduler packs 128 in-flight
  requests into one step, the resulting `attn_metadata` is exactly
  `batch_size=128, max_seqlen_q=3` → triggers this kernel.
- Production crash signature (`event.synchronize() → HIP illegal memory`) is
  the same async-reported error this reproducer surfaces.
- Crash req# varies wildly in production (131, 857, ~900, ~3945) because it
  depends on when the scheduler first assembles a step matching `bs=128 ∧
  qlen=3` — not on cumulative state.

## Build versions used

- aiter: `pr3211-on-main @ aff40475d` (also reproduces on `main @ ee28d47ac`)
- ROCm/HIP: per the `rocm/atom-dev:latest` container
- Target arch: gfx950 (MI355)
- Container: eagle3 (podman)

## Suggested next steps for the ASM team

1. Disassemble `pa_bf16_pertokenFp8_gqa8_1tg_4w_mtp_msk1.co` with
   `roc-obj-extract` / `llvm-objdump -d`, then look at how `batch_size`
   and `max_qlen=3` parameterize the kernarg block. The branch that's hit
   only at `(bs=128, qlen=3)` is the suspect.
2. Compare with the bf16 sibling `pa_bf16_noquant_gqa8_1tg_4w_mtp_msk1.co`
   (which does NOT crash on the same shape) to find the extra fp8 code path.
3. Confirm with `rocm-debug-agent`: run
   ```
   AMD_LOG_LEVEL=4 ROCM_DEBUG_AGENT=on \
       AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 \
       python pa_asm_fp8_repeat_call.py --bs 128 --ctx 1024 --qlen 3 --n-repeat 3
   ```
   to capture the wave dump, faulting PC, and offending V# descriptor.
