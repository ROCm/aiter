#!/usr/bin/env bash
# `sh run_server_test.sh` uses POSIX sh (often dash), which has no pipefail. Re-exec with bash.
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

# Kernel optimisation harness (git worktrees):
#
#   individual    — one worktree per prompt, each branched from BASELINE_REF; Claude CLI runs in that tree only.
#   accumulative  — one combined worktree; prompts run in order on the same tree (combined optimisations).
#
# Phases (RUN_PHASE):
#   build_only           — create worktrees and run Claude CLI per prompt (no benchmarks).
#   benchmarks_only      — read manifest in OUT_DIR, run baseline + benchmarks from each recorded worktree.
#   build_and_benchmarks — build phase then benchmark phase in one invocation.
#
# Env:
#   OUT_DIR         — results root (default: ./kernel_optimization_results/<timestamp>) for logs, benchmarks, manifest
#   WORKTREE_ROOT   — parent directory for git worktrees (default: <repo-parent>/.kernel_opt_worktrees/<timestamp>, or
#                     /tmp/kernel_opt_worktrees/<timestamp> when the repo’s parent is / e.g. /aiter-test — avoids // paths and keeps worktrees outside the repo).
#                     Before each build, any prior worktrees registered under this path are removed (`git worktree remove --force` + rm -rf).
#   PROMPTS_DIR     — glob source dir for *.txt / *.md prompts (default: ./optimization_prompts)
#   OPTIMIZE_MODE   — accumulative | individual (default: individual)
#   RUN_PHASE       — build_only | benchmarks_only | build_and_benchmarks (default: build_and_benchmarks)
#   CLEAN_OUTPUT_BEFORE_BUILD — if 1, before a build: remove git worktrees listed in any existing manifest under OUT_DIR,
#                     run the same worktree cleanup as a fresh WORKTREE_ROOT, then delete OUT_DIR contents (logs, results, manifest).
#                     Refuses if OUT_DIR is / or equals REPO_ROOT. Default: 0
#   BASELINE_REF    — git start point for new worktrees (default: HEAD)
#   SKIP_CLAUDE     — if set, skip Claude CLI (dry run for build phase); SKIP_CURSOR is accepted as an alias
#   ANTHROPIC_API_KEY — API auth for Claude Code (or use `claude auth login` / subscription auth)
#   CLAUDE_MODEL    — optional model for `claude --model` (e.g. sonnet, opus, or full id); CURSOR_AGENT_MODEL is accepted as an alias
#   CLAUDE_BIN      — path to `claude` executable (default: claude)
#   CLAUDE_PERMISSION_FLAGS — optional: space-separated CLI args; if set (even empty), replaces the default permission flags.
#                     Default is `--dangerously-skip-permissions --allowedTools Read,Edit,Bash` as non-root; as root, Claude CLI
#                     forbids dangerously-skip-permissions, so the script uses `--permission-mode acceptEdits --allowedTools Read,Edit,Bash`.
#   KERNEL_FILE     — source file containing the kernels to optimise. Relative to REPO_ROOT, or absolute.
#                     Substituted into {{KERNEL_FILE}} placeholders in prompt files.
#   KERNELS_TO_OPTIMIZE — space- or comma-separated kernel function names to pass to Claude.
#                     Substituted into {{KERNELS_TO_OPTIMIZE}} placeholders in prompt files.
#   PROMPTS_FILTER  — space- or comma-separated list of prompt filenames (basename, with or without extension) to include
#                     in the build phase. Only matching files from PROMPTS_DIR are run. If unset, all prompts are used
#                     and a warning is printed. Example: PROMPTS_FILTER="step1_optimize_bandwidth,step2_tune_occupancy"
#   TRIAL_MODE      — if 1, run only a single ISL/OSL pair (8192,1024) with 1 benchmark iteration per tag instead of
#                     3 pairs × 4 runs. Use this for a quick sanity check that the server starts and produces results
#                     before committing to a full benchmark run. Default: 0.
#   SERVER_READY_TIMEOUT_SEC — wait for OpenAI server log line (default: 600). If the message never appears, that benchmark tag is skipped as failed.
#   SERVER_READY_LINE — substring to wait for in the *server-only* log (default: uvicorn). Do not set to text you also write into that log.
#
# Benchmarks: once per tag (baseline or each worktree) the script starts the OpenAI server in the background, waits
#   for SERVER_READY_LINE (timeout SERVER_READY_TIMEOUT_SEC), then runs benchmark_serving for each ISL/OSL pair.
#   Full mode: 3 pairs (1024/1024, 1024/8192, 8192/1024) × 4 runs = 12 invocations per tag.
#   Trial mode (TRIAL_MODE=1): 1 pair (8192/1024) × 1 run = 1 invocation per tag — useful for a quick sanity check.
#   The server is stopped between tags so GPU VRAM is fully released before the next checkout is tested.
#
# JIT: after each `git worktree add`, `ninja_jit_setup <checkout> <baseline_jit_dir>` seeds the
#   worktree’s .jit_cache from the baseline’s compiled .so files and build trees, then uses ninja
#   depfiles (*.o.d) to detect stale modules. Only modules whose .cu/.cuh sources changed are
#   rebuilt (AITER_REBUILD=2 incremental); unchanged modules load from cache instantly.
#
# Examples:
#   Quick trial (sanity check, 1 size × 1 run):
#     TRIAL_MODE=1 RUN_PHASE=benchmarks_only OUT_DIR=./kernel_optimization_results/20260101_120000 bash run_server_test.sh
#   Clean prior OUT_DIR + worktrees then build:
#     CLEAN_OUTPUT_BEFORE_BUILD=1 OUT_DIR=./kernel_optimization_results/my_run RUN_PHASE=build_only OPTIMIZE_MODE=individual bash run_server_test.sh
#   Build only (individual):   RUN_PHASE=build_only OPTIMIZE_MODE=individual bash run_server_test.sh
#   Build only (combined):     RUN_PHASE=build_only OPTIMIZE_MODE=accumulative bash run_server_test.sh
#   Benchmarks only (reuse):   RUN_PHASE=benchmarks_only OUT_DIR=./kernel_optimization_results/20260101_120000 bash run_server_test.sh
#
# Flow:
#
#   parse env / validate args
#   |
#   |-- [benchmarks_only] -----------------.
#   |                                       |
#   collect_prompts                         |
#   resolve_baseline (git SHA)              |
#   |                                       |
#   build_phase                             |
#   |  |-- [CLEAN=1] --> clean output+worktrees
#   |  |-- [default] --> clear_worktrees_for_build
#   |  |
#   |  |-- [individual] --------.     |-- [accumulative] ---------.
#   |  |  for each prompt:      |     |  git worktree add         |
#   |  |    git worktree add    |     |  ninja_jit_setup          |
#   |  |    ninja_jit_setup     |     |  for each prompt:         |
#   |  |    run_claude_prompt   |     |    run_claude_prompt      |
#   |  |    append_manifest     |     |  append_manifest          |
#   |  '------------------------'     '---------------------------'
#   |
#   |-- [build_only] --> exit
#   |
#   benchmark_phase  <-----------------------------------------'
#   |  for baseline then each worktree in manifest:
#   |    start_openai_server_bg
#   |    wait_for_server_ready
#   |    for each ISL/OSL pair x N runs:
#   |      skip if result JSON already exists
#   |      if server died: restart → wait_for_server_ready
#   |      run benchmark_serving.py
#   |    stop_openai_server
#   |
#   print_results_table
#   |
#   Done
#
set -euo pipefail

# User-local installs (pip --user, Claude Code CLI, etc.)
export PATH="${HOME:-}/.local/bin${PATH:+:$PATH}"

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$REPO_ROOT"

OPTIMIZE_MODE=${OPTIMIZE_MODE:-individual}
RUN_PHASE=${RUN_PHASE:-build_and_benchmarks}
PROMPTS_DIR=${PROMPTS_DIR:-"$REPO_ROOT/optimization_prompts"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=${OUT_DIR:-"$REPO_ROOT/kernel_optimization_results/$TIMESTAMP"}
BASELINE_REF=${BASELINE_REF:-HEAD}
CLEAN_OUTPUT_BEFORE_BUILD=${CLEAN_OUTPUT_BEFORE_BUILD:-0}
TRIAL_MODE=${TRIAL_MODE:-0}  # if 1: single ISL/OSL pair (8192,1024) and 1 run per tag for a quick sanity check
# PROMPTS_FILTER: space- or comma-separated list of prompt filenames (basename, with or without extension) to run.
# Only prompts whose basename matches an entry are included. If unset, all prompts in PROMPTS_DIR are used (with a warning).
# Example: PROMPTS_FILTER="step1_optimize_bandwidth step2_tune_occupancy"
PROMPTS_FILTER=${PROMPTS_FILTER:-}
# KERNEL_FILE: path to the source file containing the kernels to optimise.
# Relative paths are resolved from REPO_ROOT; absolute paths are used as-is.
KERNEL_FILE=${KERNEL_FILE:-}
# KERNELS_TO_OPTIMIZE: space- or comma-separated list of kernel function names to pass to Claude.
# These are substituted into {{KERNELS_TO_OPTIMIZE}} in prompt files.
KERNELS_TO_OPTIMIZE=${KERNELS_TO_OPTIMIZE:-}

MANIFEST="$OUT_DIR/kernel_opt_worktrees.manifest"
REPO_PARENT=$(cd "$REPO_ROOT/.." && pwd)
if [[ -z "${WORKTREE_ROOT:-}" ]]; then
  if [[ "$REPO_PARENT" == "/" ]]; then
    WORKTREE_ROOT="${TMPDIR:-/tmp}/kernel_opt_worktrees/$TIMESTAMP"
  else
    WORKTREE_ROOT="$REPO_PARENT/.kernel_opt_worktrees/$TIMESTAMP"
  fi
fi
BENCH_SCRIPT=${BENCH_SCRIPT:-"$REPO_PARENT/bench_serving/benchmark_serving.py"}
SERVER_READY_TIMEOUT_SEC=${SERVER_READY_TIMEOUT_SEC:-600}
SERVER_READY_LINE=${SERVER_READY_LINE:-"Server started successfully and ready to accept requests!"}
SERVER_MODEL_PATH=${SERVER_MODEL_PATH:-/shared_inference/models/gpt-oss-120b/}
SERVER_TP=${SERVER_TP:-8}
SERVER_PORT=${SERVER_PORT:-8000}

print_help() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Kernel optimization harness for worktree build + benchmark runs.
Configuration is primarily via environment variables.

Options:
  -h, --help    Show this help message and exit.

Common environment variables:
  OPTIMIZE_MODE   accumulative | individual (default: individual)
  RUN_PHASE       build_only | benchmarks_only | build_and_benchmarks (default: build_and_benchmarks)
  OUT_DIR         results root (default: ./kernel_optimization_results/<timestamp>)
  PROMPTS_DIR     prompt files directory (default: ./optimization_prompts)
  WORKTREE_ROOT   parent directory for git worktrees
  BASELINE_REF    git start point for new worktrees (default: HEAD)
  CLEAN_OUTPUT_BEFORE_BUILD
                  0 or 1; if 1, remove prior OUT_DIR contents + related worktrees before build
  SKIP_CLAUDE     set to skip Claude CLI in build phase
  CLAUDE_MODEL    optional model passed to Claude CLI
  CLAUDE_BIN      Claude executable path (default: claude)
  KERNEL_FILE     Source file with kernels to optimise (relative to repo root or absolute).
                  Substituted into {{KERNEL_FILE}} in prompt files.
  KERNELS_TO_OPTIMIZE
                  Space- or comma-separated kernel function names for Claude to target.
                  Substituted into {{KERNELS_TO_OPTIMIZE}} in prompt files.
                  Example: KERNELS_TO_OPTIMIZE="my_kernel_a my_kernel_b"
  PROMPTS_FILTER  Space- or comma-separated prompt filenames (basename, ±extension) to run.
                  If unset, all prompts in PROMPTS_DIR are used (warning printed).
                  Example: PROMPTS_FILTER="step1_bandwidth step2_occupancy"
  TRIAL_MODE      0 (default) or 1; skips to a single ISL/OSL pair (8192,1024) with 1 run per tag.
                  Use before a full run to confirm the server starts and results are produced.
  SERVER_READY_TIMEOUT_SEC
                  server startup wait timeout in seconds (default: 300)
  SERVER_READY_LINE
                  log substring that indicates server readiness
  SERVER_MODEL_PATH
                  model path used by OpenAI server
  SERVER_TP       tensor parallel size used by OpenAI server (default: 8)
  SERVER_PORT     OpenAI server port (default: 8000)

Examples:
  RUN_PHASE=build_only OPTIMIZE_MODE=individual bash $(basename "$0")
  RUN_PHASE=benchmarks_only OUT_DIR=./kernel_optimization_results/20260101_120000 bash $(basename "$0")
  TRIAL_MODE=1 RUN_PHASE=benchmarks_only OUT_DIR=./kernel_optimization_results/20260101_120000 bash $(basename "$0")
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      echo "Run with --help to see usage." >&2
      exit 1
      ;;
  esac
done

mkdir -p "$OUT_DIR/baseline" "$OUT_DIR/optimized/accumulative" "$OUT_DIR/optimized/individual"
# Normalise OUT_DIR to an absolute path. Without this, a relative OUT_DIR (e.g. ./results/...)
# breaks when benchmark subshells do `cd <worktree>` — the relative path then resolves inside
# the worktree instead of the original directory. Also strips any trailing slash.
OUT_DIR=$(cd "$OUT_DIR" && pwd)
MANIFEST="$OUT_DIR/kernel_opt_worktrees.manifest"

# Set up an isolated JIT cache for a worktree and export AITER_JIT_DIR pointing to it.
#
# aiter is installed as a package rooted at REPO_ROOT/aiter/jit/. Because Python
# resolves the package to that fixed install path regardless of cwd, get_user_jit_dir()
# always returns REPO_ROOT/aiter/jit/ — a shared cache used by every worktree.
# Cleaning <worktree>/aiter/jit/ has no effect because the JIT system never writes there.
#
# The fix: set AITER_JIT_DIR to a per-worktree directory. get_user_jit_dir() checks
# this env var first (core.py:383-387) and uses it as the exclusive JIT cache. Starting
# from an empty directory forces all modules to recompile from the worktree's source.

# ninja_jit_setup <checkout_root> [baseline_jit_dir]
#
# Staleness-aware JIT cache setup using ninja depfiles.
#
# Sets AITER_JIT_DIR to <checkout_root>/.jit_cache.
# If baseline_jit_dir is given and non-empty, seeds the worktree cache by
# copying .so files and build/<module>/ trees from the baseline so that ninja
# can do an incremental rebuild instead of recompiling from scratch.
# Then scans *.o.d depfiles in each module's build dir; any module whose .so
# is older than one of its transitive dependencies has its .so removed and
# AITER_REBUILD=2 is exported so the Python JIT does an incremental rebuild.
# Modules with up-to-date .so files are left in place (cache hit).
# Falls back to full-rebuild behaviour if no .so files exist after seeding.
ninja_jit_setup() {
  local checkout_root=$1
  local baseline_jit_dir=${2:-}
  local wt_jit_dir="$checkout_root/.jit_cache"
  mkdir -p "$wt_jit_dir"
  export AITER_JIT_DIR="$wt_jit_dir"

  # Seed from baseline cache if provided.
  if [[ -n "$baseline_jit_dir" && -d "$baseline_jit_dir" ]]; then
    echo "ninja_jit_setup: seeding $wt_jit_dir from baseline $baseline_jit_dir" >&2
    # Copy .so files.
    local f
    for f in "$baseline_jit_dir"/*.so; do
      [[ -f "$f" ]] || continue
      cp -a "$f" "$wt_jit_dir/"
    done
    # Copy build/<module>/ trees (contains object files and depfiles for incremental rebuild).
    if [[ -d "$baseline_jit_dir/build" ]]; then
      cp -a "$baseline_jit_dir/build" "$wt_jit_dir/"
    fi
  fi

  # Collect .so files present in the cache.
  local -a so_files=()
  while IFS= read -r -d '' f; do
    so_files+=("$f")
  done < <(find "$wt_jit_dir" -maxdepth 1 -name '*.so' -print0 2>/dev/null)

  if [[ ${#so_files[@]} -eq 0 ]]; then
    # No prior build and no baseline to seed from — full recompile is unavoidable.
    echo "ninja_jit_setup: no cached .so files in $wt_jit_dir; falling back to clean_jit (full rebuild)." >&2
    rm -rf "$wt_jit_dir"
    mkdir -p "$wt_jit_dir"
    return
  fi

  local stale_count=0
  for so_file in "${so_files[@]}"; do
    local module
    module=$(basename "$so_file" .so)
    local so_mtime
    so_mtime=$(stat -c %Y "$so_file" 2>/dev/null) || continue

    # Gather all depfiles for this module.
    local build_dir="$wt_jit_dir/build/${module}/build"
    local -a depfiles=()
    while IFS= read -r -d '' df; do
      depfiles+=("$df")
    done < <(find "$build_dir" -maxdepth 1 -name '*.d' -print0 2>/dev/null)

    if [[ ${#depfiles[@]} -eq 0 ]]; then
      # No depfiles — can't check staleness; leave .so in place.
      continue
    fi

    local is_stale=0
    local stale_dep=""
    for depfile in "${depfiles[@]}"; do
      # Makefile depfile format: "target.o: dep1 dep2 dep3 ..."
      # Extract dependency paths after the colon; handle backslash-continued lines.
      local dep
      while IFS= read -r dep; do
        [[ -z "$dep" || "$dep" == *: ]] && continue
        dep="${dep%\\}"                      # strip trailing backslash
        dep="${dep#"${dep%%[! ]*}"}"         # ltrim
        dep="${dep%"${dep##*[! ]}"}"         # rtrim
        [[ -z "$dep" ]] && continue
        [[ -f "$dep" ]] || continue
        local dep_mtime
        dep_mtime=$(stat -c %Y "$dep" 2>/dev/null) || continue
        if (( dep_mtime > so_mtime )); then
          is_stale=1
          stale_dep="$dep"
          break
        fi
      done < <(sed 's/^[^:]*: *//' "$depfile" | tr ' ' '\n')
      [[ "$is_stale" -eq 1 ]] && break
    done

    if [[ "$is_stale" -eq 1 ]]; then
      echo "ninja_jit_setup: $module stale (dep newer than .so: $stale_dep) — removing .so for incremental rebuild." >&2
      rm -f "$so_file"
      stale_count=$((stale_count + 1))
    fi
  done

  if [[ "$stale_count" -gt 0 ]]; then
    export AITER_REBUILD=2
    echo "ninja_jit_setup: $stale_count module(s) stale; AITER_REBUILD=2 (incremental ninja rebuild)." >&2
  else
    echo "ninja_jit_setup: all cached modules up-to-date; no rebuild needed." >&2
  fi
}

require_git() {
  if ! git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "ERROR: $REPO_ROOT is not a git repository (worktrees require git)." >&2
    exit 1
  fi
}

resolve_baseline() {
  git -C "$REPO_ROOT" rev-parse "$BASELINE_REF^{commit}"
}

# Start OpenAI server in the background from work_dir; append *only* server stdout/stderr to server_log (so grep for SERVER_READY_LINE is not self-matched).
# Prints server PID on stdout only.
start_openai_server_bg() {
  local work_dir=$1
  local server_log=$2
  mkdir -p "$(dirname "$server_log")"
  (
    cd "$work_dir" && exec env PYTHONUNBUFFERED=1 AITER_JIT_DIR="${AITER_JIT_DIR:-}" python3 -m atom.entrypoints.openai_server \
      --model "$SERVER_MODEL_PATH" \
      -tp "$SERVER_TP" --kv_cache_dtype fp8 \
      --host 0.0.0.0 --port "$SERVER_PORT"
  ) >>"$server_log" 2>&1 &
  echo $!
}

# Wait until server_log contains SERVER_READY_LINE or timeout.
# Returns:
#   0 — server ready
#   1 — timeout
#   2 — server process exited before ready
#   3 — fatal engine error detected in log ("[atom] Engine Core: load model runner failed")
wait_for_server_ready() {
  local server_log=$1
  local server_pid=$2
  local timeout_sec=$3
  local start_ts last_print now elapsed
  start_ts=$(date +%s)
  last_print=$start_ts
  echo "Waiting for server to be ready (pid $server_pid, timeout ${timeout_sec}s). Watching log for: $SERVER_READY_LINE" >&2
  echo "  server log: $server_log" >&2
  while true; do
    if grep -qF "$SERVER_READY_LINE" "$server_log" 2>/dev/null; then
      return 0
    fi
    if grep -qF "[atom] Engine Core: load model runner failed" "$server_log" 2>/dev/null; then
      echo "ERROR: Fatal engine error detected in server log: [atom] Engine Core: load model runner failed" >&2
      return 3
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      return 2
    fi
    now=$(date +%s)
    elapsed=$((now - start_ts))
    if (( elapsed >= timeout_sec )); then
      return 1
    fi
    if (( now - last_print >= 10 )); then
      echo "Still waiting for server to be ready... ${elapsed}s / ${timeout_sec}s (pid $server_pid)." >&2
      last_print=$now
    fi
    sleep 1
  done
}

stop_openai_server() {
  local server_pid=$1
  [[ -z "${server_pid:-}" ]] && return 0

  # Collect ALL descendant PIDs via BFS before killing (they re-parent to PID 1
  # once the parent dies, making them untrackable afterwards).
  # With -tp 8 the server spawns 8 GPU worker processes that may form their own
  # process groups; we must track them explicitly to ensure GPU memory is released.
  local -a all_pids=("$server_pid")
  local -a queue=("$server_pid")
  while [[ ${#queue[@]} -gt 0 ]]; do
    local head=${queue[0]}
    queue=("${queue[@]:1}")
    local -a children
    mapfile -t children < <(pgrep -P "$head" 2>/dev/null || true)
    for child in "${children[@]}"; do
      all_pids+=("$child")
      queue+=("$child")
    done
  done

  # SIGTERM first so Python/C++ destructors run and GPU resources are released cleanly.
  # SIGKILL bypasses destructors and leaves mappings alive in the KFD,
  # permanently holding VRAM until the driver session is reset.
  local pgid
  pgid=$(ps -o pgid= -p "$server_pid" 2>/dev/null | tr -d ' ') || true
  kill -TERM "$server_pid" 2>/dev/null || true
  [[ -n "$pgid" && "$pgid" != "$$" ]] && kill -TERM -- "-$pgid" 2>/dev/null || true
  for pid in "${all_pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done

  # Wait up to 60s for graceful exit (destructors release HIP IPC handles).
  # Only SIGKILL if the process tree hasn't exited by then.
  local grace_deadline=$(( $(date +%s) + 60 ))
  local -a grace_remaining=("${all_pids[@]}")
  while [[ ${#grace_remaining[@]} -gt 0 && $(date +%s) -lt $grace_deadline ]]; do
    local -a still_alive=()
    for pid in "${grace_remaining[@]}"; do
      kill -0 "$pid" 2>/dev/null && still_alive+=("$pid")
    done
    grace_remaining=("${still_alive[@]}")
    [[ ${#grace_remaining[@]} -gt 0 ]] && sleep 1
  done
  if [[ ${#grace_remaining[@]} -gt 0 ]]; then
    echo "  Graceful shutdown incomplete after 60s; sending SIGKILL to ${#grace_remaining[@]} process(es)." >&2
    [[ -n "$pgid" && "$pgid" != "$$" ]] && kill -KILL -- "-$pgid" 2>/dev/null || true
    for pid in "${grace_remaining[@]}"; do
      kill -KILL "$pid" 2>/dev/null || true
    done
  fi

  # After SIGKILL, wait up to 10s for all remaining processes to fully exit
  # so the KFD releases their HIP contexts before the next server starts.
  local deadline=$(( $(date +%s) + 10 ))
  local -a remaining=("${grace_remaining[@]}")
  while [[ ${#remaining[@]} -gt 0 && $(date +%s) -lt $deadline ]]; do
    local -a still_alive=()
    for pid in "${remaining[@]}"; do
      kill -0 "$pid" 2>/dev/null && still_alive+=("$pid")
    done
    remaining=("${still_alive[@]}")
    [[ ${#remaining[@]} -gt 0 ]] && sleep 1
  done
  if [[ ${#remaining[@]} -gt 0 ]]; then
    echo "WARN: ${#remaining[@]} server process(es) still alive after SIGKILL; next server start may OOM. PIDs: ${remaining[*]}" >&2
  fi

  # Wait for the port to be free so the next server can bind.
  local elapsed=0
  while ss -tlnp 2>/dev/null | grep -q ":${SERVER_PORT} " && [[ $elapsed -lt 30 ]]; do
    sleep 1
    elapsed=$((elapsed + 1))
  done
  if [[ $elapsed -ge 30 ]]; then
    echo "WARN: port $SERVER_PORT still in use after 30s — next server start may fail." >&2
  fi
}

# Run benchmark workload; optional third arg is checkout directory (worktree) for cwd when invoking the bench script.
run_benchmarks() {
  local tag=$1
  local log_dir=$2
  local work_dir=${3:-$REPO_ROOT}
  mkdir -p "$log_dir"
  local log="$log_dir/benchmark_${tag}.log"
  # Use a per-worktree JIT cache so each checkout compiles its own modules.
  # Always derive from work_dir so successive benchmark tags each get the right dir
  # (build_phase exports AITER_JIT_DIR per worktree via ninja_jit_setup, but benchmark_phase
  # may run standalone or iterate over multiple worktrees in a loop).
  export AITER_JIT_DIR="$work_dir/.jit_cache"
  mkdir -p "$AITER_JIT_DIR"
  local model="$SERVER_MODEL_PATH"
  local -a isl_osl_pairs=("1024,1024" "1024,8192" "8192,1024")
  local bench_runs=4
  if [[ "${TRIAL_MODE:-0}" == "1" ]]; then
    isl_osl_pairs=("8192,1024")
    bench_runs=1
    echo "TRIAL_MODE=1: single ISL/OSL pair (8192,1024), 1 run" >>"$log"
  fi
  local conc=8 port=$SERVER_PORT
  local server_log="$log_dir/openai_server_${tag}.log"
  local server_pid

  if [[ ! -f "$BENCH_SCRIPT" ]]; then
    echo "ERROR: Benchmark script not found: $BENCH_SCRIPT" >&2
    exit 1
  fi

  # Recovery: if all result files for this tag already exist, skip server startup entirely.
  local _done=0 _total=0
  for _pair in "${isl_osl_pairs[@]}"; do
    local _isl=${_pair%%,*} _osl=${_pair##*,}
    for _i in $(seq 1 "$bench_runs"); do
      _total=$((_total + 1))
      [[ -f "$log_dir/result_${tag}_isl${_isl}_osl${_osl}_conc${conc}_run${_i}.json" ]] && _done=$((_done + 1))
    done
  done
  if [[ "$_done" -eq "$_total" ]]; then
    echo "All ${_total} result(s) for tag=${tag} already exist in $log_dir — skipping server start." | tee -a "$log"
    return 0
  fi
  [[ "$_done" -gt 0 ]] && echo "${_done}/${_total} result(s) for tag=${tag} already exist — those runs will be skipped." | tee -a "$log"

  : >"$server_log"
  local checkout_kind
  if [[ "$(cd "$work_dir" && pwd)" == "$(cd "$REPO_ROOT" && pwd)" ]]; then
    checkout_kind="baseline (repo root)"
  else
    checkout_kind="worktree"
  fi
  {
    echo "===== OpenAI server (background) — $checkout_kind ====="
    echo "cwd: $work_dir"
    echo "Server-only log (grep target must appear only from the server, not this file): $server_log"
    echo "Wait up to ${SERVER_READY_TIMEOUT_SEC}s for log substring:"
    printf '%s\n' "$SERVER_READY_LINE"
    echo "command:"
    printf 'cd %q && PYTHONUNBUFFERED=1 python3 -m atom.entrypoints.openai_server \\\n' "$work_dir"
    printf '  --model %q -tp %s --kv_cache_dtype fp8 --host 0.0.0.0 --port %s\n' \
      "$SERVER_MODEL_PATH" "$SERVER_TP" "$SERVER_PORT"
    echo "===== benchmark ${tag}: OpenAI server pid will be recorded below (shared across ${#isl_osl_pairs[@]} ISL/OSL pairs × ${bench_runs} runs each) ====="
  } >>"$log"

  server_pid=$(start_openai_server_bg "$work_dir" "$server_log")
  {
    echo "OpenAI server pid ${server_pid}"
  } >>"$log"

  set +e
  wait_for_server_ready "$server_log" "$server_pid" "$SERVER_READY_TIMEOUT_SEC"
  local wr=$?
  set -e
  if [[ "$wr" -ne 0 ]]; then
    stop_openai_server "$server_pid"
    local wr_reason
    case "$wr" in
      1) wr_reason="timeout after ${SERVER_READY_TIMEOUT_SEC}s" ;;
      2) wr_reason="server process exited before ready" ;;
      3) wr_reason="fatal engine error: [atom] Engine Core: load model runner failed" ;;
      *) wr_reason="unknown (exit ${wr})" ;;
    esac
    {
      echo "ERROR: OpenAI server did not become ready for tag=${tag} (${wr_reason}). Skipping all benchmark runs for this tag. See: $server_log"
    } | tee -a "$log" >&2
    return 1
  fi

  {
    echo "Server ready (${SERVER_READY_LINE}); running benchmark_serving ${#isl_osl_pairs[@]} ISL/OSL pairs × ${bench_runs} runs each"
  } >>"$log"

  local bench_rc=0
  for pair in "${isl_osl_pairs[@]}"; do
    local isl=${pair%%,*}
    local osl=${pair##*,}
    for i in $(seq 1 "$bench_runs"); do
      local result_fn="result_${tag}_isl${isl}_osl${osl}_conc${conc}_run${i}.json"
      # Recovery: skip this run if the result file already exists from a prior attempt.
      if [[ -f "$log_dir/$result_fn" ]]; then
        echo "Skipping isl=${isl} osl=${osl} run ${i}/${bench_runs} ($tag) — result already exists." | tee -a "$log"
        continue
      fi
      # Recovery: if the server died since the last run, restart it before continuing.
      if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "WARN: Server (pid $server_pid) died unexpectedly; restarting for tag=${tag}." | tee -a "$log" >&2
        # Truncate the server log so wait_for_server_ready's grep doesn't match the old ready line.
        : >"$server_log"
        server_pid=$(start_openai_server_bg "$work_dir" "$server_log")
        echo "Restarted server (new pid: $server_pid)" >>"$log"
        set +e
        wait_for_server_ready "$server_log" "$server_pid" "$SERVER_READY_TIMEOUT_SEC"
        local restart_rc=$?
        set -e
        if [[ "$restart_rc" -ne 0 ]]; then
          stop_openai_server "$server_pid"
          local restart_reason
          case "$restart_rc" in
            1) restart_reason="timeout after ${SERVER_READY_TIMEOUT_SEC}s" ;;
            2) restart_reason="server process exited before ready" ;;
            3) restart_reason="fatal engine error" ;;
            *) restart_reason="unknown (exit ${restart_rc})" ;;
          esac
          echo "ERROR: Restarted server failed to become ready (${restart_reason}). Aborting remaining benchmarks for tag=${tag}." | tee -a "$log" >&2
          return 1
        fi
        echo "Restarted server is ready; resuming benchmarks." | tee -a "$log"
      fi
      {
        echo "===== benchmark ${tag} isl=${isl} osl=${osl} run ${i}/${bench_runs} ====="
        echo "cwd: $work_dir"
        echo "test command line:"
        printf 'cd %q && \\\n' "$work_dir"
        printf 'python %q \\\n' "$BENCH_SCRIPT"
        printf '  --backend=vllm --base-url=%q --endpoint=/v1/completions \\\n' "http://localhost:$port"
        printf '  --model=%q \\\n' "$model"
        echo '  --dataset-name=random \'
        echo "  --random-input-len=$isl --random-output-len=$osl \\"
        echo "  --num-prompts=$((conc * 4)) \\"
        echo "  --max-concurrency=$conc \\"
        echo '  --random-range-ratio 1.0 \'
        echo '  --request-rate=inf --ignore-eos \'
        echo '  --save-result --percentile-metrics=ttft,tpot,itl,e2el \'
        printf '  --result-dir=%q \\\n' "$log_dir"
        printf '  --result-filename=%q\n' "$result_fn"
        echo "single line (paths shell-escaped):"
        printf 'cd %q && python %q --backend=vllm --base-url=%q --endpoint=/v1/completions --model=%q --dataset-name=random --random-input-len=%s --random-output-len=%s --num-prompts=%s --max-concurrency=%s --random-range-ratio 1.0 --request-rate=inf --ignore-eos --save-result --percentile-metrics=ttft,tpot,itl,e2el --result-dir=%q --result-filename=%q\n' \
          "$work_dir" "$BENCH_SCRIPT" "http://localhost:$port" "$model" "$isl" "$osl" "$((conc * 4))" "$conc" "$log_dir" "$result_fn"
        echo "--- output ---"
      } >>"$log"
      echo "Running isl=${isl} osl=${osl} run ${i}/${bench_runs} ($tag) cwd=$work_dir → $log"
      set +e
      (cd "$work_dir" && python "$BENCH_SCRIPT" \
        --backend=vllm --base-url="http://localhost:$port" --endpoint=/v1/completions \
        --model="$model" \
        --dataset-name=random \
        --random-input-len="$isl" --random-output-len="$osl" \
        --num-prompts=$((conc * 4)) \
        --max-concurrency="$conc" \
        --random-range-ratio 1.0 \
        --request-rate=inf --ignore-eos \
        --save-result --percentile-metrics="ttft,tpot,itl,e2el" \
        --result-dir="$log_dir" --result-filename="$result_fn" \
        >>"$log" 2>&1)
      local st=$?
      set -e
      if [[ $st -ne 0 ]]; then
        bench_rc=$st
        echo "ERROR: benchmark_serving exited $st for ${tag} isl=${isl} osl=${osl} run ${i} (see $log)" | tee -a "$log" >&2
      fi
    done
  done

  stop_openai_server "$server_pid"
  echo "Stopped OpenAI server (pid was ${server_pid}) for tag=${tag}" >>"$log"

  if [[ "$bench_rc" -ne 0 ]]; then
    return "$bench_rc"
  fi
  return 0
}

# Run Claude Code CLI (`claude -p`) from the given workspace (worktree or repo root).
# Without permission bypass, print mode stops at "approve the file edit" and writes no changes — see CLAUDE_PERMISSION_FLAGS.
run_claude_prompt() {
  local prompt_file=$1
  local agent_log=$2
  local workspace=$3
  mkdir -p "$(dirname "$agent_log")"
  if [[ -n "${SKIP_CLAUDE:-}" || -n "${SKIP_CURSOR:-}" ]]; then
    echo "SKIP_CLAUDE/SKIP_CURSOR set — skipping Claude CLI for $(basename "$prompt_file")" | tee -a "$agent_log"
    return 0
  fi
  local prompt_text
  # Strip YAML frontmatter (--- ... ---) if present — it is harness metadata, not prompt content.
  # Claude CLI's argument parser misreads '---' at the start of a -p value as an option name.
  if IFS= read -r _first_line < "$prompt_file" && [[ "$_first_line" == "---" ]]; then
    prompt_text=$(awk 'NR==1{next} /^---$/{if(!found){found=1;next}} found' "$prompt_file")
  else
    prompt_text=$(cat "$prompt_file")
  fi
  # Substitute {{KERNEL_FILE}} and {{KERNELS_TO_OPTIMIZE}} placeholders so prompt files
  # remain kernel-agnostic; the actual targets are driven by env vars.
  # Fail fast if a placeholder is present but the corresponding variable is unset.
  if [[ "$prompt_text" == *'{{KERNEL_FILE}}'* && -z "${KERNEL_FILE:-}" ]]; then
    echo "ERROR: prompt $(basename "$prompt_file") contains {{KERNEL_FILE}} but KERNEL_FILE is not set." >&2
    exit 1
  fi
  if [[ "$prompt_text" == *'{{KERNELS_TO_OPTIMIZE}}'* && -z "${KERNELS_TO_OPTIMIZE:-}" ]]; then
    echo "ERROR: prompt $(basename "$prompt_file") contains {{KERNELS_TO_OPTIMIZE}} but KERNELS_TO_OPTIMIZE is not set." >&2
    exit 1
  fi
  local kernel_file_abs
  if [[ -n "${KERNEL_FILE:-}" ]]; then
    if [[ "${KERNEL_FILE}" == /* ]]; then
      # Absolute path: re-anchor to the worktree by replacing the repo root prefix.
      # e.g. /aiter-test/csrc/... → <worktree>/csrc/...
      kernel_file_abs="${KERNEL_FILE/#$REPO_ROOT/$workspace}"
    else
      # Relative path: resolve relative to the workspace root so Claude always
      # receives an absolute path pointing into the worktree, not the baseline repo.
      # Without this, Claude may resolve the relative path from the main repo root
      # (via the worktree's .git pointer) and edit the baseline instead of the worktree copy.
      kernel_file_abs="$workspace/$KERNEL_FILE"
    fi
  else
    kernel_file_abs=""
  fi
  local kernels_formatted="" k
  # Normalise commas to spaces so both "a,b" and "a b" work as separators.
  for k in ${KERNELS_TO_OPTIMIZE//,/ }; do
    [[ -z "$k" ]] && continue
    kernels_formatted+=$'\n'"    ${k}"
  done
  prompt_text="${prompt_text//\{\{REPO_ROOT\}\}/$workspace}"
  prompt_text="${prompt_text//\{\{KERNEL_FILE\}\}/$kernel_file_abs}"
  prompt_text="${prompt_text//\{\{KERNELS_TO_OPTIMIZE\}\}/$kernels_formatted}"

  local claude_bin=${CLAUDE_BIN:-claude}
  local model_flag=()
  if [[ -n "${CLAUDE_MODEL:-}" ]]; then
    model_flag=(--model "$CLAUDE_MODEL")
  elif [[ -n "${CURSOR_AGENT_MODEL:-}" ]]; then
    model_flag=(--model "$CURSOR_AGENT_MODEL")
  fi
  # Default: unattended edits in worktrees. Root cannot use --dangerously-skip-permissions (Claude CLI exits 1). Override: CLAUDE_PERMISSION_FLAGS.
  local perm_flags
  if [[ -n "${CLAUDE_PERMISSION_FLAGS+_}" ]]; then
    read -r -a perm_flags <<<"${CLAUDE_PERMISSION_FLAGS}"
  elif [[ "$(id -u)" -eq 0 ]]; then
    perm_flags=(--permission-mode acceptEdits --allowedTools "Read,Edit,Bash")
  else
    perm_flags=(--dangerously-skip-permissions --allowedTools "Read,Edit,Bash")
  fi
  echo "[$(date '+%T')] Claude CLI started: $(basename "$prompt_file") (workspace: $workspace)" >&2
  echo "[$(date '+%T')] Live output below — also logged to: $agent_log" >&2
  set +e
  # stdbuf -oL forces line-buffered output through the pipe so each line appears
  # immediately on the terminal rather than being held until the buffer fills.
  # Falls back gracefully if stdbuf is not installed.
  local _stdbuf=()
  command -v stdbuf >/dev/null 2>&1 && _stdbuf=(stdbuf -oL)
  (cd "$workspace" && "${_stdbuf[@]}" "$claude_bin" -p "$prompt_text" \
    "${perm_flags[@]}" \
    "${model_flag[@]}" \
    2>&1) | tee -a "$agent_log"
  local st=${PIPESTATUS[0]}
  set -e
  echo "[$(date '+%T')] Claude CLI finished (exit $st): $(basename "$prompt_file")" >&2
  if [[ $st -ne 0 ]]; then
    echo "ERROR: Claude CLI exited $st (see $agent_log). Check API auth; as root, ensure the log does not show a permissions-flag error (use CLAUDE_PERMISSION_FLAGS if needed)." >&2
    exit $st
  fi
}

collect_prompts() {
  if [[ ! -d "$PROMPTS_DIR" ]]; then
    echo "ERROR: PROMPTS_DIR is not a directory: $PROMPTS_DIR" >&2
    exit 1
  fi

  local -a all_prompts
  mapfile -t all_prompts < <(
    find "$PROMPTS_DIR" -maxdepth 1 -type f \( -name '*.txt' -o -name '*.md' \) | LC_ALL=C sort
  )
  if [[ ${#all_prompts[@]} -eq 0 ]]; then
    echo "ERROR: No prompt files in $PROMPTS_DIR (*.txt or *.md)." >&2
    exit 1
  fi

  if [[ -z "${PROMPTS_FILTER:-}" ]]; then
    echo "WARN: PROMPTS_FILTER not set — running all ${#all_prompts[@]} prompt(s) in $PROMPTS_DIR." \
         "Set PROMPTS_FILTER to a space/comma-separated list of filenames to restrict which prompts run." >&2
    _prompts=("${all_prompts[@]}")
    return
  fi

  # Normalise filter: replace commas with spaces, split into array.
  local -a filter_names
  IFS=' ,' read -r -a filter_names <<<"${PROMPTS_FILTER}"

  _prompts=()
  local f name base
  for f in "${all_prompts[@]}"; do
    base=$(basename "$f")
    for name in "${filter_names[@]}"; do
      # Match against full basename OR basename-without-extension.
      if [[ "$base" == "$name" || "${base%.*}" == "$name" || "${base%.*}" == "${name%.*}" ]]; then
        _prompts+=("$f")
        break
      fi
    done
  done

  if [[ ${#_prompts[@]} -eq 0 ]]; then
    echo "ERROR: PROMPTS_FILTER='${PROMPTS_FILTER}' matched no files in $PROMPTS_DIR." >&2
    echo "       Available: $(IFS=', '; echo "${all_prompts[*]/#$PROMPTS_DIR\//}")" >&2
    exit 1
  fi

  # Warn about any filter entries that matched nothing.
  local matched name base f
  for name in "${filter_names[@]}"; do
    matched=0
    for f in "${_prompts[@]}"; do
      base=$(basename "$f")
      [[ "$base" == "$name" || "${base%.*}" == "$name" || "${base%.*}" == "${name%.*}" ]] && matched=1 && break
    done
    if [[ "$matched" -eq 0 ]]; then
      echo "WARN: PROMPTS_FILTER entry '${name}' matched no file in $PROMPTS_DIR — skipped." >&2
    fi
  done
}

slugify() {
  basename "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/_$//;s/^_//'
}

# Remove any git worktrees whose paths live under WORKTREE_ROOT, then delete WORKTREE_ROOT (fresh build).
clear_worktrees_for_build() {
  local wr_abs=$WORKTREE_ROOT
  if [[ -d "$WORKTREE_ROOT" ]]; then
    wr_abs=$(cd "$WORKTREE_ROOT" && pwd)
  elif command -v realpath >/dev/null 2>&1; then
    wr_abs=$(realpath -m "$WORKTREE_ROOT")
  fi

  echo "Clearing prior kernel-opt worktrees under: $wr_abs"
  local -a to_remove=()
  local line path
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == worktree\ * ]] || continue
    path=${line#worktree }
    path=${path//$'\r'/}
    path=${path%% }
    [[ -z "$path" ]] && continue
    if [[ "$path" == "$REPO_ROOT" ]]; then
      continue
    fi
    if [[ "$path" == "$wr_abs" || "$path" == "$wr_abs/"* ]]; then
      to_remove+=("$path")
    fi
  done < <(git -C "$REPO_ROOT" worktree list --porcelain 2>/dev/null || true)

  for path in "${to_remove[@]}"; do
    if [[ -d "$path" ]]; then
      echo "  git worktree remove --force $path"
      git -C "$REPO_ROOT" worktree remove --force "$path" 2>/dev/null || true
    fi
  done

  rm -rf "$WORKTREE_ROOT"
}

# Remove git worktrees recorded as tag<TAB>path lines after --- in an existing manifest (orphan cleanup).
remove_git_worktrees_from_manifest() {
  local manifest_path=$1
  [[ ! -f "$manifest_path" ]] && return 0
  local past_header=0 line tag path
  while IFS= read -r line || [[ -n "$line" ]]; do
    line=${line//$'\r'/}
    if [[ "$past_header" == "0" ]]; then
      [[ "$line" == "---" ]] && past_header=1
      continue
    fi
    [[ -z "$line" || "$line" == \#* ]] && continue
    IFS=$'\t' read -r tag path <<<"$line"
    [[ -z "${path:-}" || "$path" == "$tag" || "$path" == "$REPO_ROOT" ]] && continue
    if [[ -d "$path" ]]; then
      echo "  git worktree remove --force (prior manifest) $path"
      git -C "$REPO_ROOT" worktree remove --force "$path" 2>/dev/null || true
    fi
  done <"$manifest_path"
}

# If CLEAN_OUTPUT_BEFORE_BUILD=1: unregister worktrees from old manifest, clear WORKTREE_ROOT worktrees, wipe OUT_DIR, recreate dirs.
clean_output_and_worktrees_before_build() {
  if [[ "$OUT_DIR" == "/" || "$OUT_DIR" == "$REPO_ROOT" ]]; then
    echo "ERROR: CLEAN_OUTPUT_BEFORE_BUILD=1 refuses OUT_DIR=$OUT_DIR (must not be / or the repo root)." >&2
    exit 1
  fi
  echo "CLEAN_OUTPUT_BEFORE_BUILD=1 — removing prior manifest worktrees, clearing WORKTREE_ROOT, wiping $OUT_DIR"
  remove_git_worktrees_from_manifest "$MANIFEST"
  clear_worktrees_for_build
  if [[ -d "$OUT_DIR" ]]; then
    find "$OUT_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  fi
  mkdir -p "$OUT_DIR/baseline" "$OUT_DIR/optimized/accumulative" "$OUT_DIR/optimized/individual"
}

write_manifest_header() {
  local baseline_sha=$1
  mkdir -p "$(dirname "$MANIFEST")"
  mkdir -p "$WORKTREE_ROOT"
  {
    echo "v1"
    echo "REPO_ROOT=$REPO_ROOT"
    echo "WORKTREE_ROOT=$WORKTREE_ROOT"
    echo "OPTIMIZE_MODE=$OPTIMIZE_MODE"
    echo "BASELINE_REF=$BASELINE_REF"
    echo "BASELINE_SHA=$baseline_sha"
    echo "TIMESTAMP=$TIMESTAMP"
    echo "---"
  } >"$MANIFEST"
}

append_manifest_entry() {
  local tag=$1
  local path=$2
  printf '%s\t%s\n' "$tag" "$(cd "$path" && pwd)" >>"$MANIFEST"
}

# Create worktrees and run Claude CLI for each prompt. Writes $MANIFEST (tag<TAB>absolute_path per line after ---).
build_phase() {
  local baseline_sha=$1
  local opt_subdir=$OPTIMIZE_MODE
  if [[ "$CLEAN_OUTPUT_BEFORE_BUILD" == "1" ]]; then
    clean_output_and_worktrees_before_build
  else
    clear_worktrees_for_build
  fi
  write_manifest_header "$baseline_sha"

  if [[ "$OPTIMIZE_MODE" == "individual" ]]; then
    mkdir -p "$WORKTREE_ROOT/individual"
    local step=0
    for prompt_file in "${PROMPTS[@]}"; do
      step=$((step + 1))
      local slug
      slug=$(slugify "$prompt_file")
      local tag="step${step}_${slug}"
      local branch="kernel-opt/${TIMESTAMP}/individual/${step}/${slug}"
      local wt_path="$WORKTREE_ROOT/individual/${tag}"
      echo "=== Worktree (individual): $wt_path branch $branch ==="
      git -C "$REPO_ROOT" worktree add -b "$branch" "$wt_path" "$baseline_sha"
      ninja_jit_setup "$wt_path" "$REPO_ROOT/.jit_cache"
      local agent_log="$OUT_DIR/optimized/$opt_subdir/${tag}_claude_cli.log"
      echo "=== Claude CLI: $prompt_file → $agent_log (cwd=$wt_path) ==="
      run_claude_prompt "$prompt_file" "$agent_log" "$wt_path"
      append_manifest_entry "$tag" "$wt_path"
    done
  else
    mkdir -p "$WORKTREE_ROOT/accumulative"
    local branch="kernel-opt/${TIMESTAMP}/accumulative/combined"
    local wt_path="$WORKTREE_ROOT/accumulative/combined"
    echo "=== Worktree (accumulative): $wt_path branch $branch ==="
    git -C "$REPO_ROOT" worktree add -b "$branch" "$wt_path" "$baseline_sha"
    ninja_jit_setup "$wt_path" "$REPO_ROOT/.jit_cache"
    local step=0
    for prompt_file in "${PROMPTS[@]}"; do
      step=$((step + 1))
      local slug
      slug=$(slugify "$prompt_file")
      local tag="step${step}_${slug}"
      local agent_log="$OUT_DIR/optimized/$opt_subdir/${tag}_claude_cli.log"
      echo "=== Claude CLI: $prompt_file → $agent_log (cwd=$wt_path) ==="
      run_claude_prompt "$prompt_file" "$agent_log" "$wt_path"
    done
    append_manifest_entry "accumulative_combined" "$wt_path"
  fi
  echo "Build phase done. Manifest: $MANIFEST"
}

# Run benchmarks from repo root (baseline) then from each worktree listed in the manifest.
benchmark_phase() {
  if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: Manifest not found: $MANIFEST (set OUT_DIR to a prior run that completed build_only or build_and_benchmarks)." >&2
    exit 1
  fi
  local opt_subdir
  opt_subdir=$(grep '^OPTIMIZE_MODE=' "$MANIFEST" | head -1 | cut -d= -f2-)
  if [[ -z "$opt_subdir" ]]; then
    opt_subdir=$OPTIMIZE_MODE
  fi
  local bench_dir="$OUT_DIR/optimized/$opt_subdir"

  echo "=== Baseline benchmarks (cwd=$REPO_ROOT) ==="
  set +e
  run_benchmarks "baseline" "$OUT_DIR/baseline" "$REPO_ROOT"
  local baseline_rc=$?
  set -e
  if [[ "$baseline_rc" -ne 0 ]]; then
    echo "WARN: baseline benchmarks finished with exit $baseline_rc (server timeout, skipped tag, or benchmark_serving failure). Continuing with manifest entries." >&2
  fi

  local past_header=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line=${line//$'\r'/}
    if [[ "$past_header" == "0" ]]; then
      [[ "$line" == "---" ]] && past_header=1
      continue
    fi
    [[ -z "$line" || "$line" == \#* ]] && continue
    local tag path
    IFS=$'\t' read -r tag path <<<"$line"
    if [[ -z "${path:-}" || "$path" == "$tag" ]]; then
      echo "WARN: skip malformed manifest line: $line" >&2
      continue
    fi
    if [[ ! -d "$path" ]]; then
      echo "ERROR: worktree path missing: $path" >&2
      exit 1
    fi
    echo "=== Benchmarks for $tag (cwd=$path) ==="
    set +e
    run_benchmarks "$tag" "$bench_dir" "$path"
    local tag_rc=$?
    set -e
    if [[ "$tag_rc" -ne 0 ]]; then
      echo "WARN: benchmarks for $tag finished with exit $tag_rc (server timeout, skipped tag, or benchmark_serving failure). Continuing." >&2
    fi
  done <"$MANIFEST"
}

# Read OPTIMIZE_MODE from a manifest file (fallback: $OPTIMIZE_MODE global).
read_opt_subdir_from_manifest() {
  local manifest_path=$1
  local val
  val=$(grep '^OPTIMIZE_MODE=' "$manifest_path" 2>/dev/null | head -1 | cut -d= -f2-)
  echo "${val:-$OPTIMIZE_MODE}"
}

# Print a side-by-side comparison table of all benchmark tags (baseline + worktrees).
# Averages runs 1-4 per tag; cells show N/A when no JSON results exist.
# Output goes to stdout (terminal) and $out_dir/results_summary.txt.
print_results_table() {
  local out_dir=$1
  local opt_subdir=$2
  echo ""
  echo "=== Benchmark Results Summary ==="
  python3 - "$out_dir" "$opt_subdir" <<'PYEOF'
import glob, json, os, re, sys

out_dir    = sys.argv[1]
opt_subdir = sys.argv[2]

# Tuple: (json_key, column_label, format_str, higher_is_better)
METRICS = [
    ("output_throughput",      "Out tok/s",    "{:.1f}",  True),
    ("total_token_throughput", "Total tok/s",  "{:.1f}",  True),
    ("mean_ttft_ms",           "TTFT mean ms", "{:.1f}",  False),
    ("median_ttft_ms",         "TTFT p50 ms",  "{:.1f}",  False),
    ("mean_tpot_ms",           "TPOT mean ms", "{:.2f}",  False),
    ("median_tpot_ms",         "TPOT p50 ms",  "{:.2f}",  False),
    ("mean_itl_ms",            "ITL mean ms",  "{:.2f}",  False),
    ("p99_itl_ms",             "ITL p99 ms",   "{:.2f}",  False),
    ("mean_e2el_ms",           "E2EL mean ms", "{:.1f}",  False),
]

def load_tag(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    rows = [json.load(open(f)) for f in files]
    result = {}
    for key, _, _, higher_is_better in METRICS:
        vals = [r[key] for r in rows if key in r and r[key] is not None]
        if len(vals) > 1:
            # Drop the slowest sample: lowest value for throughput, highest for latency.
            vals = sorted(vals)[:-1] if higher_is_better else sorted(vals)[1:]
        result[key] = sum(vals) / len(vals) if vals else None
    result["_runs"] = len(files)
    result["_completed"] = sum(r.get("completed", 0) for r in rows)
    return result

# Discover ISL/OSL pairs from result filenames across all directories.
ISL_OSL_RE = re.compile(r"_isl(\d+)_osl(\d+)_")
def find_isl_osl_pairs(search_dirs):
    pairs = []
    seen = set()
    for d in search_dirs:
        for f in glob.glob(os.path.join(d, "result_*.json")):
            m = ISL_OSL_RE.search(os.path.basename(f))
            if m:
                pair = (int(m.group(1)), int(m.group(2)))
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
    pairs.sort()
    return pairs

search_dirs = [
    os.path.join(out_dir, "baseline"),
    os.path.join(out_dir, "optimized", opt_subdir),
]
isl_osl_pairs = find_isl_osl_pairs(search_dirs)
# Fallback for old-format results (no osl in filename).
if not isl_osl_pairs:
    isl_osl_pairs = [(8192, 1024)]

# Collect ordered tag names from manifest.
tag_names = ["baseline"]
manifest = os.path.join(out_dir, "kernel_opt_worktrees.manifest")
past_header = False
if os.path.exists(manifest):
    for line in open(manifest):
        line = line.rstrip("\r\n")
        if not past_header:
            if line == "---":
                past_header = True
            continue
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            tag_names.append(parts[0])

col_w    = max(16, max(len(t) for t in tag_names))
metric_w = 14

def build_subtable(isl, osl):
    # Load results for each tag for this ISL/OSL pair.
    data = {}
    data["baseline"] = load_tag(
        os.path.join(out_dir, "baseline", f"result_baseline_isl{isl}_osl{osl}_*.json")
    )
    # Fallback: old filename format without osl field.
    if data["baseline"] is None:
        data["baseline"] = load_tag(
            os.path.join(out_dir, "baseline", f"result_baseline_isl{isl}_*.json")
        )
    for tag in tag_names[1:]:
        data[tag] = load_tag(
            os.path.join(out_dir, "optimized", opt_subdir, f"result_{tag}_isl{isl}_osl{osl}_*.json")
        )
        if data[tag] is None:
            data[tag] = load_tag(
                os.path.join(out_dir, "optimized", opt_subdir, f"result_{tag}_isl{isl}_*.json")
            )

    header_row = f"{'Metric':<{metric_w}}" + "".join(f"  {t:>{col_w}}" for t in tag_names)
    sep = "-" * len(header_row)
    lines = [f"  ISL={isl}  OSL={osl}", sep, header_row, sep]
    for key, label, fmt, _ in METRICS:
        row = f"{label:<{metric_w}}"
        for tag in tag_names:
            d = data[tag]
            if d is None or d.get(key) is None:
                row += f"  {'N/A':>{col_w}}"
            else:
                row += f"  {fmt.format(d[key]):>{col_w}}"
        lines.append(row)
    lines.append(sep)
    row = f"{'Runs/Completed':<{metric_w}}"
    for tag in tag_names:
        d = data[tag]
        if d is None:
            row += f"  {'N/A':>{col_w}}"
        else:
            row += f"  {str(d['_runs'])+'r/'+str(d['_completed'])+'req':>{col_w}}"
    lines.append(row)
    lines.append(sep)
    return lines

all_lines = []
for isl, osl in isl_osl_pairs:
    all_lines.extend(build_subtable(isl, osl))
    all_lines.append("")

table = "\n".join(all_lines)
print(table)

summary = os.path.join(out_dir, "results_summary.txt")
with open(summary, "w") as f:
    f.write(table + "\n")
print(f"Results summary written to: {summary}", file=sys.stderr)
PYEOF
}

# --- main ---
case "$OPTIMIZE_MODE" in
  accumulative|individual) ;;
  *)
    echo "ERROR: OPTIMIZE_MODE must be accumulative or individual (got: $OPTIMIZE_MODE)" >&2
    exit 1
    ;;
esac

case "$RUN_PHASE" in
  build_only|benchmarks_only|build_and_benchmarks) ;;
  *)
    echo "ERROR: RUN_PHASE must be build_only, benchmarks_only, or build_and_benchmarks (got: $RUN_PHASE)" >&2
    exit 1
    ;;
esac

case "$CLEAN_OUTPUT_BEFORE_BUILD" in
  0|1) ;;
  *)
    echo "ERROR: CLEAN_OUTPUT_BEFORE_BUILD must be 0 or 1 (got: $CLEAN_OUTPUT_BEFORE_BUILD)" >&2
    exit 1
    ;;
esac

require_git
echo "Results: $OUT_DIR"
echo "Mode: $OPTIMIZE_MODE  Run phase: $RUN_PHASE"

if [[ "$RUN_PHASE" == "benchmarks_only" ]]; then
  benchmark_phase
  print_results_table "$OUT_DIR" "$(read_opt_subdir_from_manifest "$MANIFEST")"
  echo "Done. Benchmarks under: $OUT_DIR"
  exit 0
fi

collect_prompts
PROMPTS=("${_prompts[@]}")
echo "Prompts: ${#PROMPTS[@]} from $PROMPTS_DIR"

BASELINE_SHA=$(resolve_baseline)
build_phase "$BASELINE_SHA"

if [[ "$RUN_PHASE" == "build_only" ]]; then
  echo "Done. Worktrees and manifest under: $OUT_DIR (no benchmarks run)."
  exit 0
fi

benchmark_phase
print_results_table "$OUT_DIR" "$OPTIMIZE_MODE"
echo "Done. Benchmarks, Claude CLI logs, and manifest under: $OUT_DIR"
