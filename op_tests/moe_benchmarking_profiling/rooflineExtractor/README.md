# Roofline Extractor
Roofline Extractor is a tool that calculates the percent of the empirical peak performance that an application is achieving on a per-kernel basis.

[Example Roofline Plot for rocHPL](http://canofcorn.amd.com/Rooflines/hpl-mi250x-roof-counters.html)

## Install Python Packages
Run this to install all necessary packages:
```
pip install -r requirements.txt
```

## Option 1: Automated Profiling (Recommended)
The easiest way to use rooflineExtractor is with the `profile.py` script, which automates all the profiling steps:

```bash
python3 profile.py -- <app exe> [args...]
```

This script will:
1. Automatically detect your GPU architecture (MI250X, MI300A, MI300X, MI355X)
2. Run `rocprofv3` to collect hardware counters (four runs of the application)
3. Perform post-processing on the counter data
4. Run `rocprofv3` to collect kernel trace data (one run of the application)
5. Run `rooflineExtractor.py` to generate analysis and plots

**Note:** The script uses the `-f csv` flag with rocprofv3, which is only available in ROCm 7 and later. If the flag is not recognized, the script will automatically retry without it.

Optional flags:
* `-o [OUTPUT_DIR]`: Specify output directory (default: `./output/`)
* `--arch [ARCH]`: Specify current GPU architecture (auto-detected if not provided). Options: MI250X, MI300A, MI300X, MI355X
* `--proj-arch [ARCH]`: Specify architecture for runtime projection. Options: MI250X, MI300A, MI300X, MI355X

**Example:**
```bash
python3 profile.py -o my_results --proj-arch MI355X -- ./my_app
```

All output files (counters, traces, plots, analysis) will be saved in the specified output directory.

## Option 2: Manual Profiling
If you prefer to run the profiling steps manually:

### Data Generation
The following two runs of rocprof are needed to use rooflineExtractor:
* Counters
  * This run gathers counters for the application. Pick the input file from this directory that is appropriate for your architecture.
  * `rocprof -i roof-counters-<arch>.txt <app exe>` or `rocprofv3 -i roof-counters-<arch>.txt -f csv -- <app exe>`
    * **Note:** The `-f csv` flag is only recognized in ROCm 7 and later. For older versions, omit this flag.
    * If using rocprofv3, another command is needed to consolidate its output into a single file: `python3 convert-conters-collection-format.py -i <path to rocprofv3 output files> -o <singular output file>`
* Runtime stats
  * This run gathers timing information for the application
  * `rocprof --stats <exe>` or `rocprofv3 --kernel-trace -f csv -- <exe>`
    * **Note:** The `-f csv` flag is only recognized in ROCm 7 and later. For older versions, omit this flag.

### Run rooflineExtractor
`python3 rooflineExtractor.py -c [roof-counter.csv filename] -r [results.csv filename]`

Additional optional flags:
* `--plot`: Generate plots
* `--dump`: Dump pandas dataframe to `logs/<app>_roof_counters_EXTRACTED.csv`
* `--sig-runtime [% runtime]`: Specify what's the minimum runtime for a kernel to be considered "significant" and be included in analysis. Defaults to 10%.
* `--arch [arch]`: Specify GPU architecture (if not provided, rooflineExtractor will guess). Options: MI250X, MI300A, MI300X, MI355X
* `--proj-arch [arch]`: Specify second architecture (future) for runtime projection. Options: MI250X, MI300A, MI300X, MI355X

### Example
Here is an example using nbody-nvidia-mini with **rocprofv3**.
```
# Collect kernel counters with rocprofv3 (using -f csv if supported by your ROCm version)
rocprofv3 -i roof-counters-gfx942.txt -o counters -f csv -- ./nbody-orig 1048576

# Collect runtime stats with rocprofv3 (using -f csv if supported by your ROCm version)
rocprofv3 --kernel-trace -o trace -f csv -- ./nbody-orig 1048576

# Convert the hardware counter collection output to CSV (needed for rooflineExtractor)
python3 convert-counters-collection-format.py -i . -o counters.csv

# Run rooflineExtractor to generate plots and dataframes
python3 rooflineExtractor.py -c counters.csv -r trace_kernel_trace.csv --plot --dump --arch MI300X
```

## Output:
* A guided analysis via the terminal showing per-kernel performance metrics including arithmetic intensity, peak achievable throughput, and distance from the roofline
* An HTML file with an interactive roofline plot showing the performance and arithmetic intensity of each kernel instance
* A CSV file with all of the per-kernel throughput and arithmetic intensity information calculated
