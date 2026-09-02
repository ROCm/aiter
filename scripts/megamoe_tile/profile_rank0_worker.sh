#!/usr/bin/env bash
# Run as the torchrun --no-python worker entrypoint on both nodes.
# Exactly one global rank is wrapped by rocprofv3; all peers execute the same
# Python benchmark directly so EP16 collectives and CCO progress remain live.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: profile_rank0_worker.sh DRIVER [DRIVER_ARGS...]" >&2
  exit 2
fi

profile_rank="${PROFILE_GLOBAL_RANK:-0}"
python_bin="${PYTHON_BIN:-python3}"

if [[ "${profile_rank}" != "all" && "${RANK:-unset}" != "${profile_rank}" ]]; then
  unset FLYDSL_DEBUG_ENABLE_DEBUG_INFO
  exec "${python_bin}" -u "$@"
fi

# Source-to-ISA DWARF is required only for the rank wrapped by rocprof ATT.
# Enabling it in the parent torchrun environment makes every peer rebuild the
# very large rank-specialized Stage-2 kernel.
export FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1

rocprof_bin="${ROCPROFV3_BIN:-/opt/rocm/bin/rocprofv3}"
profile_root="${PROFILE_ROOT:?PROFILE_ROOT must name the rank0 output directory}"
if [[ "${profile_rank}" == "all" ]]; then
  profile_root="${profile_root}/rank${RANK:?RANK must be set by torchrun}"
fi
profile_mode="${PROFILE_MODE:-trace}"
kernel_regex="${KERNEL_RE:-.*}"
profile_output_formats="${PROFILE_OUTPUT_FORMATS:-csv}"
mkdir -p "${profile_root}"

echo "PROFILE_RANK0 mode=${profile_mode} rank=${RANK} local_rank=${LOCAL_RANK} output=${profile_root} kernel=${kernel_regex}" >&2

case "${profile_mode}" in
  trace)
    read -r -a output_formats <<<"${profile_output_formats}"
    trace_args=(
      --kernel-trace \
      --hip-trace \
      --memory-copy-trace \
      --marker-trace \
      --stats \
      --output-format "${output_formats[@]}" \
      --output-directory "${profile_root}" \
      --kernel-include-regex "${kernel_regex}"
    )
    # rocprofiler-sdk only honors roctxProfilerPause/Resume filtering when
    # selected-region collection is explicitly requested.
    if [[ "${MEGAMOE_TILE_PROFILE_REGIONS:-0}" == "1" ]]; then
      export MEGAMOE_TILE_PROFILER_STARTS_PAUSED=1
      trace_args+=(--selected-regions)
    fi
    exec "${rocprof_bin}" "${trace_args[@]}" -- "${python_bin}" -u "$@"
    ;;
  att)
    att_range="${ATT_ITERATION_RANGE:?ATT_ITERATION_RANGE must be N-N}"
    att_target_cu="${ATT_TARGET_CU:-1}"
    att_gpu_index="${ATT_GPU_INDEX:-0}"
    att_library="${ATT_LIBRARY_PATH:-}"
    att_args=(
      --att
      --att-gpu-index "${att_gpu_index}"
      --att-target-cu "${att_target_cu}"
      --kernel-include-regex "${kernel_regex}"
      --kernel-iteration-range "${att_range}"
      --output-directory "${profile_root}"
    )
    if [[ -n "${att_library}" ]]; then
      att_args+=(--att-library-path "${att_library}")
    fi
    exec "${rocprof_bin}" "${att_args[@]}" -- "${python_bin}" -u "$@"
    ;;
  *)
    echo "unsupported PROFILE_MODE=${profile_mode}; expected trace or att" >&2
    exit 2
    ;;
esac
