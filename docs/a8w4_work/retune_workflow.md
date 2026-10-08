# A8W4 validation and serial retune

Prepare input snapshots after merging and validating the implementation:

```bash
python docs/a8w4_work/run_serial_retune.py --prepare
```

This reads each model's untuned header for shape identity and includes the
untuned/tuned union: DeepSeek-V4 187, Kimi-K3 35, GPT-OSS 16. It saves raw
source snapshots, hashes, tags and input token versus public lookup token.
Repeated timing and flydsl_fallback tags do not add workloads. These outputs
live in this work directory; official model CSVs are not modified.

Run one actual tuning trial at a time through the central execution wrapper,
then validate its new CSV with the public call harness:

```bash
AITER_CONFIG_FMOE=/absolute/path/to/trial_tuned.csv \
  AITER_ONLINE_TUNE=0 AITER_BYPASS_TUNE_CONFIG=0 \
  AITER_FLYDSL_STAGE2_FP8=0 AITER_MXFP4_INTERMEDIATE=0 \
  python op_tests/test_mxfp4_flydsl_public_csv.py \
    --csv /absolute/path/to/trial_tuned.csv --rows 0 1 2 \
    --report docs/a8w4_work/public_trial_report.json
```

Select the GPU centrally and set `HIP_VISIBLE_DEVICES` for that command. The
harness passes explicit FP8 activation/intermediate, INTERLEAVE W1, BF16
output, Situv2 4/25 and the resolved Swiglu limit. It observes and delegates
real stage calls, checks live FP8 payload plus E8M0 scales, output identity,
finite values and amplitude-sensitive error. A skipped/fallback pair is
reported separately and returns nonzero. `--rows` uses zero-based data rows.
Each report preserves the requested row/pair, the same CSV's primary lookup
row/token/pair and actual observed G1/G2. For M=3 the public key is M=4; a
correct row4 execution is counted as padded token coverage and does not claim
that an unused standalone row3 winner was executed. Large-token tiers follow
the same runtime mapping, including smaller-tier lookup when applicable.
For broader modes, a validation CSV may contain selected actual evaluated
profile observations; record that selection and keep final winner CSVs intact.

Once prerequisites and public integration pass, recheck external tune jobs
and idle GPUs and invoke the scheduler directly:

```bash
python docs/a8w4_work/run_serial_retune.py --run --gpus 4 5 6 7 --batch 4
```

GPU IDs above are examples. Use freshly verified physical HIP IDs and unset
inherited `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` for physical selection.
Do not nest the scheduler inside another process holding the same locks.
It holds `/tmp/aiter-independent-tune.lock` plus `/tmp/gpu-<HIP>.lock`, maps
HIP IDs to SMI BDFs and requires three idle samples before each model. Raw
process records remain in the log; only entries whose memory, engines and CU
occupancy are all known finite zero values count as idle contexts. Unknown or
nonzero resources block launch. Samples also require gfx activity within
0–2% and VRAM below 1 GiB with a valid reported capacity.
Other executed Python tuning scripts block launch; formatter file arguments
do not. Each model has its own process group, and all live group members must
end before the next fixed-order model starts. No watchdog, reset, kill or
automatic candidate/shape retry is added.

The fixed order is DeepSeek-V4 → Kimi-K3 → GPT-OSS. Each invokes full search
with `--timeout 300 --errRatio 0.1 --batch 4`, `--mp` equal to selected GPU
count, and default warmup 5/iters 101. Inputs/reference still prepare per
candidate. Large sweeps can take a long time; timeout is per candidate.
The scheduler's `--batch` defaults to 4, following the user's latest request;
this supersedes the earlier batch 6 specification. Actual batch is recorded
in every model's startup/completion JSON.

Every model writes tuned/profile/failed_shapes/log and startup/completion
JSON. Startup JSON exists before the blocking wait and records source hashes,
versions, exact argv/env, PID/PGID, GPU samples and input hashes. The scheduler
checks source consistency between models, drains workers on both successful
and failed exits, checks winners against raw numeric profile error and shared
G1/G2 parsers, and continues subsequent ended models after a failed model.
Partial results retain nonzero aggregate status. Existing output files require
a new output directory; they are not silently overwritten/resumed.

After each complete retune, run the public CSV harness against its final new
CSV in a centrally assigned idle slot. Final acceptance requires that report
as well as scheduler coverage. Source status and full logs remain evidence;
input preparation alone does not claim retune or GPU correctness completion.
