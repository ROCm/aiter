# Single-kernel CCO ticket validation

This is a runbook, not an executable launcher. The commands below identify the
landed candidate but must not run until its source has been synchronized to
both WORLD2 hosts.

## Required hand-off

Use these exact values from the implementation:

```bash
DRIVER_REL=op_tests/multigpu_tests/test_megamoe_tile_cco_persistent_h1_world2.py
KERNEL=megamoe_tile_h1_persistent_cco_a4w4_silu_h3584_i384_e56_k16_bm32_ws8_wpe2_world_sc2
```

The driver uses the target contract (`tokens=8`, `H=3584`, `I=384`, `E=56`,
`topk=16`, `workers=240`, work-shards 8, WPE2), prints the exact selected
kernel, and emits `MEGAMOE_CCO_PERSISTENT_H1_PASS` on both WORLD2 ranks. A
substring regex is not acceptable for sign-off.

## 1. Read-only preflight

Use the existing WORLD2 launcher in dry-run mode to check the synchronized
driver/bridge hashes, NIC state, and MORI build without starting a communicator:

```bash
DRY_RUN=1 \
TEST_REL="${DRIVER_REL}" \
MEGAMOE_CCO_PASS_MARKER=MEGAMOE_CCO_PERSISTENT_H1_PASS \
MEGAMOE_CCO_FAIL_MARKER=MEGAMOE_CCO_PERSISTENT_H1_FAIL \
MEGAMOE_CCO_H1_WORKERS=240 \
MEGAMOE_CCO_QP=4 \
MEGAMOE_CCO_BATCH=8 \
MEGAMOE_CCO_CHUNK=65536 \
scripts/megamoe_tile/run_cco_transport_world2.sh
```

Before the real run, check on both nodes:

- the driver hash and relevant AITER source hashes are identical;
- `KERNEL` exactly matches the name printed by the driver;
- GPU 0 is idle enough for profiling;
- each rank uses a new `FLYDSL_RUNTIME_CACHE_DIR` so stale code objects cannot
  supply the result;
- the 64 KiB ticket payload and target H1 inputs are prepared outside timing.

## 2. WORLD2 correctness

After preflight and review, run once with the same variables and `DRY_RUN=0`.
The launcher has backward-compatible configurable PASS/FAIL markers and passes
`MEGAMOE_CCO_H1_WORKERS` into each container. Both ranks must report PASS,
bitwise H1 output, 64 KiB payload integrity, and the same exact kernel name. Do
not profile a failing run.

## 3. Kernel trace

Wrap **both** WORLD2 rank commands with the following rocprof fragment. The
rendezvous/UID and rank commands must still start concurrently. Use distinct
rank output directories and fresh rank-local FlyDSL caches.

```bash
PROFILE_ROOT=/home/hzm/profiles/megamoe_single_ticket_target
RANK="${CCO_RANK}"
export FLYDSL_RUNTIME_CACHE_DIR="/tmp/flydsl_single_ticket_target_rank${RANK}"
export MEGAMOE_CCO_H1_WORKERS=240
export MEGAMOE_CCO_H1_WARMUP=5
export MEGAMOE_CCO_H1_ITERS=50

rocprofv3 \
  --kernel-trace \
  --output-format csv \
  --output-directory "${PROFILE_ROOT}/rank${RANK}" \
  --kernel-include-regex "^${KERNEL}$" \
  -- \
  python "${DRIVER_REL}"
```

`--kernel-include-regex` is an acquisition filter only. The result parser below
still performs exact string equality. The driver takes warmup/iteration counts
from the environment, advances the absolute generation every launch, and
externally credits/reclaims the ring slot before reuse. The final 50 exact rows
are therefore the timed steady samples. Do not use Python/HIP event time as the
authoritative kernel duration.

## 4. Resource and spill metadata

Locate the code object produced by the same fresh-cache run, verify its AMDHSA
`.name` is exactly `KERNEL`, and save its notes:

```bash
LLVM_READELF=/opt/rocm/llvm/bin/llvm-readelf
CODE_OBJECT=/absolute/path/to/the/exact/ticket/code_object
NOTES="${PROFILE_ROOT}/rank${RANK}/exact_kernel_notes.txt"

"${LLVM_READELF}" --notes "${CODE_OBJECT}" > "${NOTES}"
grep -F ".name: ${KERNEL}" "${NOTES}"
```

Sign-off fields are `.vgpr_count`, `.sgpr_count`,
`.group_segment_fixed_size`, `.private_segment_fixed_size`,
`.vgpr_spill_count`, and `.sgpr_spill_count`. The code-object metadata is
authoritative for private/spill; rocprof CSV resource columns are only a
cross-check.

## 5. Exact analysis and baseline comparison

Run once for each rank's steady CSV:

```bash
python scripts/megamoe_tile/analyze_single_kernel_ticket.py \
  --kernel-trace "${PROFILE_ROOT}/rank${RANK}/HOST/PID_kernel_trace.csv" \
  --kernel "${KERNEL}" \
  --last 50 \
  --expect-samples 50 \
  --metadata-notes "${NOTES}" \
  --require-spill-free
```

The analyzer reports exact dispatch durations and compares their median to:

- copy-ticket: `18.80 us`;
- direct-ready H1: `25.68 us`;
- same-stream serial reference: approximately `234 us`.

For the distributed result, report both rank medians and use the slower rank as
the WORLD2 critical-path number. Preserve the absolute profile, CSV, code
object/notes, cache, host, commit, and driver-command paths in the final report.
