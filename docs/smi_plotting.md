# SMI monitoring plots

Enable automatic reports for benchmarks that emit `AITER_SMI_RESULT` records:

```bash
ENABLE_CK=0 \
AITER_SMI_MONITOR=1 \
AITER_SMI_PLOT=1 \
AITER_SMI_DURATION=2 \
AITER_SMI_SKIP_FUNCTIONS=run_torch \
python3 op_tests/test_gemm_a8w8_blockscale.py \
  --flydsl --bpreshuffle True --apre True --data-init norm \
  -m 512 -nk 6144,7168 7168,3072 8192,1536 2048,7168 65536,1536 7168,16384
```

`AITER_SMI_PLOT=1` is the sole plotting opt-in. The output root defaults to
`./smi_plots`, resolved relative to the working directory when the first result
is emitted. Set `AITER_SMI_PLOT_DIR=/data/my-plots` to override it. An unset,
empty, or `0` plot toggle disables plotting; a directory alone does not enable
it. No records means no report or new directory.

The usual stdout stream or `AITER_SMI_OUTPUT_PATH` file is preserved. Plotting
also copies records into a fresh per-process run directory:

```text
smi_plots/run-<unique-id>/
  smi_results.log
  plotter.log
  report/
    index.html
    all_tests.png / all_tests.svg
    per_test/
    per_case/
    records.csv
    manifest.json
```

At normal Python process exit, one CPU-only subprocess renders all records
captured by that process. Rendering happens after the workload, outside
latency measurements and monitored replay windows. Install the optional
dependency with `python3 -m pip install matplotlib`. Matplotlib is not loaded
by the monitor itself. Plotting failure produces a message on stderr and
keeps the captured input for manual regeneration; it does not change the
benchmark exit status. The report path is also printed on stderr.

Reports from repeated or concurrent processes use separate directories.
Forked children do not inherit their parent's report. For long-lived callers
or multiprocessing workers that bypass Python's normal exit handlers, call
`aiter.smi_monitor.flush_smi_plots()` after the final workload. It returns the
HTML path on success, or `None` for no report or a plotting failure. Repeated
flushes do nothing until more records arrive; new records begin a fresh report.
Forced termination (including `SIGKILL` and `os._exit`) cannot render an exit
report, but already written records remain available.

## Plot existing logs or combine processes

Run the standalone script directly; this requires neither AITER imports nor
a GPU. The CLI also works after a wheel installation by using the installed
`aiter/smi_plot.py` file.

```bash
python3 aiter/smi_plot.py /data/my_run.log -o /data/my_report
python3 aiter/smi_plot.py /data/my-plots --glob smi_results.log -o /data/combined_report
```

Directory scans are recursive. Defaults are `*.log`, `*.jsonl`, and `*.txt`;
use repeatable `--glob` arguments to narrow them. Mixed console logs and bare
JSONL are supported. Malformed SMI records produce a filename/line error.
Repeated results remain separate, with source file/line provenance. When
combining automatic reports, select `smi_results.log` to avoid also reading
copies from your ordinary benchmark logs.

The report includes an all-tests clock overview, a clock view for each test,
and per-case comparisons of clocks, power, activity, and temperature. All
metric summaries are exported to CSV. Unknown/unavailable metrics are not
invented; insufficient or unverified sample quality is labeled explicitly.
Large plots are paginated; use `--rows-per-page` to change the default of 40.
Use `--formats png` for PNG only, and `--title` for an explicit hardware/date
label. Old files from previous CLI configurations are not deleted; the HTML
and manifest reference only the current report.

Markers show means and bars show observed min–max, not confidence intervals.
Current monitor output contains summaries, not time-series samples. The data
includes ramp-up and cannot be used to reconstruct time series or remove warmup
after the fact. Latency is measured before replay; plots do not establish
steady-state GEMM performance.

Within the same plotting environment, identical input bytes, relative input
filenames, and options produce byte-identical report files. Ordering, metric
colors, filenames, fonts, and SVG hashes are fixed; generation timestamps are
omitted. Axis limits come from the whole dataset and are shared across plots.
Rendering may differ across Matplotlib/font versions. The unique outer run
directory does not affect report content.

CPU-only checks:

```bash
python3 -m unittest discover -s op_tests -p 'test_smi_plot*.py' -v
```
