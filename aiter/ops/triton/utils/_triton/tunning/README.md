# Triton GEMM tunning script

single case testing:
rocprofv3 --kernel-trace -f csv -o res -- python3 ut.py 4 2112 7168 8 32 1024 1 2 1 1 16 0 7

**Backends and config schemas**

`screen.py`, `view-screen.py` and `verify-perf.py` all take `--backend
triton|gluon`. The default is the arch default: **gluon on gfx1250**, triton
everywhere else. The backend reaches the `ut_*.py` subprocess through the
`AITER_TUNE_BACKEND` env var, not argv, because argv is a positional stream of
config ints chunked by schema length.

The backend selects the **config schema**, which is per-op -- gluon configs are
not one uniform shape. `_utils.SCHEMAS` holds them and `_utils.UT_SCHEMA` maps
each ut script to its schema per backend:

| schema | keys (argv and JSON order) |
| --- | --- |
| `triton_gemm` | BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages, waves_per_eu, matrix_instr_nonkdim, cache_modifier, NUM_KSPLIT |
| `gluon_a16w16` | BLOCK_M/N/K, num_warps, NUM_BUFFERS, kernel_type |
| `gluon_a16w16_persistent` | BLOCK_M/N/K, GROUP_SIZE_M, num_warps, NUM_BUFFERS |
| `gluon_mxfp4_preshuffle` | BLOCK_SIZE_M/N/K, num_warps, NUM_BUFFERS |
| `gluon_batched_a16w16` | BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, waves_per_eu, matrix_instr_nonkdim, cache_modifier, NUM_KSPLIT, NUM_BUFFERS, kernel_type |
| `gluon_afp8wfp8_preshuffle` | BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, waves_per_eu, cache_modifier, NUM_KSPLIT, NUM_BUFFERS, kernel_type, CTAS_M, CTAS_N, B_SCALE_TDM, LOOP_UNROLL_FACTOR |

Some gfx1250 gluon kernels read triton-shaped configs (`gemm_a8w8_blockscale`,
`gemm_a8w8_blockscale_preshuffled`, `gemm_afp4wfp4`), so those map to
`triton_gemm` under both backends. `gemm_a16w16`'s gluon schema deliberately has
no `GROUP_SIZE_M`: the grid is a flat `cdiv(M,BM)*cdiv(N,BN)` and the kernel
never reads it.

Params outside the legacy flags are set with the repeatable generic flag:

    --param NUM_BUFFERS 2 4 6 --param kernel_type 0 1

Encoded params take ints: `kernel_type` 0=`bandwidth_bound` 1=`compute_bound`,
`cache_modifier` 0=`.cg` 1=null, `B_SCALE_TDM` 0/1. The
`--block-size-{m,n,k}-range` flags drive `BLOCK_M/N/K` too, so old invocations
work unchanged on both backends.

The tuner refuses combinations it cannot actually measure, with the reason:

* an op with no gluon kernel (`ut_a8w8_gemm_per_token_scale.py --backend gluon`)
* an op whose gluon kernel is arch-limited (`ut_a8w8_gemm.py` is gluon on gfx950
  only; asking for gluon on gfx1250 is refused)
* an op that picks gluon from the arch with no `backend=` override
  (`ut_afp4wfp4_gemm_preshuffle.py --backend triton` on gfx1250)

Log and JSON names are backend-qualified (`screen-<ut>-<backend>-<M>-<N>-<K>.log`)
so a gluon sweep never overwrites a triton one for the same shape. Pass the same
`--backend` to `view-screen.py` that the sweep ran with -- it picks the schema
used to decode each `screencase` row, and mismatches are caught by a column-count
assert rather than silently emitting a wrong config.

Pre-pruning is schema-aware. For gluon it also drops configs whose tile cannot
hold the requested pipeline depth: `NUM_BUFFERS > num_k_tiles + 1`, and an
estimated LDS footprint over the 320 KB budget. That second rule matters -- a
256x256x128 tile at `NUM_BUFFERS=4` needs ~557 KB, so it silently degrades to
depth 2 and loses 3-4x. The estimate ignores swizzle padding, so it only prunes
configs already over budget before padding, never borderline ones.

Known op-side constraint: the gluon `afp8wfp8_preshuffle` `compute_bound` variant
does not take `CTAS_M`/`CTAS_N`; those combinations error and `screen.py` skips
them as param-specific failures.

**One screencase must have exactly one kernel under it**

If a `screencase` in the log is followed by two indented kernel-name lines, the
number under it is meaningless: `rprof.py` matches every kernel containing
`gemm` and **sums** their p50s. That happens when an op mutates the config dict
it was handed, so the 250 profiled calls do not all launch the same kernel.

This bit `gemm_a8w8_blockscale`, which used to do

    config["NUM_BUFFERS"] = config.pop("num_stages", 1)

in place. Call 1 ran `NUM_BUFFERS=2`, calls 2..250 ran `NUM_BUFFERS=1`, and the
log reported 45.6 us for a kernel that actually takes 8.9 us. Both ends are now
fixed: the op copies the config before normalising it, and every `ut_*.py` also
copies inside `fn()`. Keep that copy if you add a ut script.

**gfx1250 gluon blockscale: num_stages was renamed to NUM_BUFFERS**

The gfx1250 gluon blockscale kernels take `NUM_BUFFERS` (pipeline depth) as a
constexpr and never took `num_stages`; the configs under
`configs/gfx1250/gluon/gemm/gemm_a8w8_blockscale{,_preshuffled}/` now spell it
`NUM_BUFFERS` to match, and the tuner's `gluon_a8w8_blockscale` schema tunes it
directly. It is worth tuning: on M=64 N=4096 K=4096, depth 4 is 8.9 us against
15.0 us at depth 2.

gfx950 gluon blockscale still reads `num_stages` (a different code path and a
different config dir), which is why `UT_SCHEMA` picks the schema per arch. The op
still accepts a legacy `num_stages` spelling so pre-rename configs keep running.

**Running tunning script**

Example 1: Tunning for A16W16 GEMM using default BLOCK_SIZE ranges using GPU 0, see screen.py for deafult range

    python3 screen.py \
        64 8192 3584 0 \
        ut_a16w16_gemm.py \
        > example1.out

Example 2: Background tunning for A8W8 GEMM blockscale using specific `BLOCK_SIZE_K` ranges using GPU 0 ~ 6, because A8W8 blockscale gemm requires only `BLOCK_SIZE_K=128`
    
    N=2112
    K=7168
    for M_G in "8 0" "16 1" "32 2" "64 3" "128 4" "256 5" "8192 6"; do
        set -- $M_G
        M=$1
        G=$2
        nohup python3 screen.py \
            $M $N $K $G \
            ut_a8w8_gemm_blockscale.py \
            --block-size-k-range 128 \
            > example2-M=$M-N=$N-K=$K-G=$G.out &
    done

Example 3: Background tunning for AFP4WFP4 GEMM. In this case `BLOCK_SZIE_M` has to meet the following requirements: `1) BLOCK_SIZE_M < 32 for M < 32, 2) BLOCK_SIZE_M >= 32 for M >= 32`. `BLOCK_SIZE_K` has to meet the following requirements: `BLOCK_SIZE_K >= 256`. If we still use the default settings, GEMM will give assertion errors, screen.py will skip those cases first time it hits assert errors and skip all other cases that shares the same BLOCK_SIZE. See the generated *.log files and termeinal output (example3.out) for more details. It will take a few minutes for screen.py to skip through those failed configs, so if you want to save those few minutes, you have to set dedicated `--block-size-m-range` for each `M` to skip invalid `BLOCK_SZIE_M`. This example also enables verbose printout that shows the pre-pruned cases and the error messages that triggers exclusions of cases on-the-fly.

    N=7168
    K=2048
    G=0
    python3 screen.py \
        64 $N $K $G \
        ut_afp4wfp4_gemm_preshuffle.py \
        --block-size-k-range 256 512 1024 \
        --overwrite \
        --verbose \
        > example3.out
    
**Viewing results and generate JSON config files**

Example 0: a gluon sweep on gfx1250 and the config it produces

    python3 screen.py \
        64 8192 3584 0 \
        ut_a16w16_gemm.py \
        --backend gluon \
        --param NUM_BUFFERS 2 4 --param kernel_type 0 1 \
        > example0.out
    python3 view-screen.py ut_a16w16_gemm.py --backend gluon \
        --n-list 8192 --k-list 3584
    # -> gfx1250-GEMM-A16W16-N=8192-K=3584.json holding BLOCK_M/BLOCK_N/BLOCK_K,
    #    num_warps, NUM_BUFFERS, kernel_type; copy into
    #    configs/gfx1250/gluon/gemm/gemm_a16w16/

Example 1:

    python view-screen.py ut_a16w16_gemm.py --n-list 8192 --k-list 3584

Example 2: 

    N=2112
    K=7168
    python view-screen.py ut_a8w8_gemm_blockscale.py --n-list $N --k-list $K

Example 3:

    N=2112
    K=7168
    python view-screen.py ut_afp4wfp4_gemm_preshuffle.py --n-list $N --k-list $K

**Verify performance**

To verify that your tunned JSON config files actually is performant and can be correctly picked up by AITER, first you have to copy the generated JSON config files into the config tree. Every family lives in one nested layout, `configs/<arch>/<backend>/<op>/<d_type>/` (`<path_to_aiter_root>/aiter/ops/triton/configs/CLAUDE.md` is the authoritative rulebook). Files there carry **no arch prefix** — the arch is the directory — and the default file is named exactly `DEFAULT.json`, so drop the arch prefix when copying:

    cp GEMM-AFP4WFP4_PRESHUFFLED-N=7168-K=2048.json \
        <path_to_aiter_root>/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_afp4wfp4_preshuffled/

`<d_type>` is the config name lowercased with dashes folded to underscores (`GEMM-AFP4WFP4_PRESHUFFLED` → `gemm_afp4wfp4_preshuffled`), and `<backend>` is `triton` unless you tuned the gluon kernel — the two backends read separate directories and never fall back to each other.

Two gotchas: a family's `DEFAULT.json` must be in place before any specialized file resolves, and config reads are cached per path (including missing files), so restart the Python process after copying for the new files to be picked up.

then, you can run, for example,

    python verify-perf.py 32 2112 7168 ut_a8w8_gemm_blockscale_preshuffle.py

and check the kernel name (with config suffix) and runtime to see if both kernel name and runtime match those inside the JSON config files. If the kernel name and runtime do not match, it could be that your JSON file name is wrong. You have to go to the file where the kernel resides and check the `_get_config` function to check the `config_name` arguments.