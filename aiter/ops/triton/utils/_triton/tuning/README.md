# Triton GEMM tuning scripts

Run every command from this directory.

| File | What it does |
| --- | --- |
| `sweep_configs.py` | Profiles every candidate config for one `M N K` on one GPU and logs the runtimes to `screen-<harness>-<M>-<N>-<K>.log` |
| `write_best_configs.py` | Picks the fastest config per `M` from those logs and writes `<arch>-<config name>-N=<N>-K=<K>.json` |
| `verify_configs.py` | Profiles a harness with the configs installed in the config tree and prints the kernel name and runtime |
| `parse_kernel_trace.py` | Reduces a `rocprofv3 --kernel-trace` CSV to the median kernel runtime per config; called by the scripts above |
| `harness_<op>.py` | Runs `<op>` with each config given on its command line, or with the installed configs if none are given; the scripts above run it under `rocprofv3` |
| `harness_template.py` | Starting point for a new harness |
| `_utils.py` | Helpers shared by the harnesses and scripts |

Profiling a single config: a harness takes `M N K` followed by the ten config values, in the order of `config_parms_key` in `_utils.py`:

    rocprofv3 --kernel-trace -f csv -o res -- python3 harness_gemm_afp4wfp4.py 4 2112 7168 8 32 1024 1 2 1 1 16 0 7
    python3 parse_kernel_trace.py res_kernel_trace.csv -k gemm

**Running the sweep**

Example 1: Tuning for A16W16 GEMM using default BLOCK_SIZE ranges using GPU 0, see sweep_configs.py for the default ranges

    python3 sweep_configs.py \
        64 8192 3584 0 \
        harness_gemm_a16w16.py \
        > example1.out

Example 2: Background tuning for A8W8 GEMM blockscale using specific `BLOCK_SIZE_K` ranges using GPU 0 ~ 6, because A8W8 blockscale gemm requires only `BLOCK_SIZE_K=128`

    N=2112
    K=7168
    for M_G in "8 0" "16 1" "32 2" "64 3" "128 4" "256 5" "8192 6"; do
        set -- $M_G
        M=$1
        G=$2
        nohup python3 sweep_configs.py \
            $M $N $K $G \
            harness_gemm_a8w8_blockscale.py \
            --block-size-k-range 128 \
            > example2-M=$M-N=$N-K=$K-G=$G.out &
    done

Example 3: Background tuning for AFP4WFP4 GEMM. In this case `BLOCK_SIZE_M` has to meet the following requirements: `1) BLOCK_SIZE_M < 32 for M < 32, 2) BLOCK_SIZE_M >= 32 for M >= 32`. `BLOCK_SIZE_K` has to meet the following requirements: `BLOCK_SIZE_K >= 256`. If we still use the default settings, GEMM will give assertion errors, sweep_configs.py will skip those cases first time it hits assert errors and skip all other cases that shares the same BLOCK_SIZE. See the generated *.log files and terminal output (example3.out) for more details. It will take a few minutes for sweep_configs.py to skip through those failed configs, so if you want to save those few minutes, you have to set dedicated `--block-size-m-range` for each `M` to skip invalid `BLOCK_SIZE_M`. This example also enables verbose printout that shows the pre-pruned cases and the error messages that triggers exclusions of cases on-the-fly.

    N=7168
    K=2048
    G=0
    python3 sweep_configs.py \
        64 $N $K $G \
        harness_gemm_afp4wfp4_preshuffle.py \
        --block-size-k-range 256 512 1024 \
        --overwrite \
        --verbose \
        > example3.out

**Writing the JSON config files**

`write_best_configs.py` prints the fastest config found for each `M` and names the JSON files after the config family the harness tunes, listed in `HARNESS_CONFIG_NAMES` at the top of the script (`--json-prefix` overrides it).

Example 1:

    python3 write_best_configs.py harness_gemm_a16w16.py --n-list 8192 --k-list 3584

Example 2:

    N=2112
    K=7168
    python3 write_best_configs.py harness_gemm_a8w8_blockscale.py --n-list $N --k-list $K

Example 3:

    N=7168
    K=2048
    python3 write_best_configs.py harness_gemm_afp4wfp4_preshuffle.py --n-list $N --k-list $K

**Verifying the configs**

To verify that your tuned JSON config files actually are performant and can be correctly picked up by AITER, first you have to copy the generated JSON config files into the config tree. Every family lives in one nested layout, `configs/<arch>/<backend>/<op>/<d_type>/` (`<path_to_aiter_root>/aiter/ops/triton/configs/CLAUDE.md` is the authoritative rulebook). Files there carry **no arch prefix** — the arch is the directory — and the default file is named exactly `DEFAULT.json`, so drop the arch prefix when copying:

    cp gfx950-GEMM-AFP4WFP4_PRESHUFFLED-N=7168-K=2048.json \
        <path_to_aiter_root>/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_afp4wfp4_preshuffled/GEMM-AFP4WFP4_PRESHUFFLED-N=7168-K=2048.json

`<d_type>` is the config name lowercased with dashes folded to underscores (`GEMM-AFP4WFP4_PRESHUFFLED` → `gemm_afp4wfp4_preshuffled`), and `<backend>` is `triton` unless you tuned the gluon kernel — the two backends read separate directories and never fall back to each other.

Two gotchas: a family's `DEFAULT.json` must be in place before any specialized file resolves, and config reads are cached per path (including missing files), so restart the Python process after copying for the new files to be picked up.

then, you can run, for example,

    python3 verify_configs.py 32 2112 7168 harness_gemm_a8w8_blockscale_preshuffle.py

and check the kernel name (with config suffix) and runtime to see if both kernel name and runtime match those inside the JSON config files. If the kernel name and runtime do not match, it could be that your JSON file name is wrong. You have to go to the file where the kernel resides and check the `_get_config` function to check the `config_name` arguments.

**Adding a harness**

Copy `harness_template.py` to `harness_<op>.py`, fill in its three blocks, and add the harness to `HARNESS_CONFIG_NAMES` in `write_best_configs.py` with the `config_name` that the kernel's `_get_config` passes to `get_gemm_config`.
