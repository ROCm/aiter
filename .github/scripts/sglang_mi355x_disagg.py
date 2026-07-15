#!/usr/bin/env python3
"""Helpers for the SGLang MI355X disaggregation workflow.

The workflow intentionally reuses SGLang's launcher from the upstream PR and
keeps the aiter-side delta small. This helper patches the launcher for aiter's
Spur/Slurm environment and narrows the recipe to the requested smoke surface.
"""

from __future__ import annotations

import argparse
from pathlib import Path

RECIPE_PATH = Path("scripts/ci/slurm/recipes/mi355x-fp8/dsv4pro/1k1k/1p1d.yaml")
LAUNCHER_PATH = Path("scripts/ci/slurm/launch_mi355x.sh")


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one launcher match, got {count}: {old!r}")
    return text.replace(old, new, 1)


def replace_all(text: str, old: str, new: str, min_count: int = 1) -> str:
    count = text.count(old)
    if count < min_count:
        raise SystemExit(
            f"expected at least {min_count} launcher match(es), got {count}: {old!r}"
        )
    return text.replace(old, new)


def patch_launcher(args: argparse.Namespace) -> None:
    root = Path(args.sglang_workspace)
    launcher = root / LAUNCHER_PATH
    text = launcher.read_text(encoding="utf-8")

    text = replace_once(
        text,
        'SLURM_PARTITION="${SLURM_PARTITION:-amd-sglang}"',
        'SLURM_PARTITION="${SLURM_PARTITION-}"',
    )
    text = replace_once(
        text,
        'WORKDIR="$HOME/.mi355x_ci/${MATRIX_CONFIG_NAME}"',
        'WORKDIR="${SHARED_WORKDIR_ROOT:-$HOME/.mi355x_ci}/${MATRIX_CONFIG_NAME}"',
    )
    text = replace_once(
        text,
        """DOCKER_COMMON="--rm --network host --ipc host --shm-size 32g --privileged \\
--security-opt seccomp=unconfined \\
--device /dev/kfd --device /dev/dri --device /dev/infiniband \\
-v /it-share:/it-share:ro -v $HOME:/host_home $CHECKOUT_DOCKER_ARGS"
""",
        """MODEL_MOUNT_ARGS=""
for mount_root in ${MODEL_MOUNT_ROOTS:-/it-share /data /models}; do
    if [[ -d "$mount_root" ]]; then
        MODEL_MOUNT_ARGS+=" -v $mount_root:$mount_root:ro"
    fi
done

AITER_MOUNT_ARGS=""
if [[ -n "${AITER_SOURCE_DIR:-}" ]]; then
    AITER_MOUNT_ARGS=" -v $AITER_SOURCE_DIR:/aiter-under-test:ro"
fi

DOCKER_COMMON="--rm --network host --ipc host --shm-size 32g --privileged \\
--security-opt seccomp=unconfined \\
--device /dev/kfd --device /dev/dri --device /dev/infiniband \\
-v $WORKDIR:/ci_workdir -v $HOME:/host_home $MODEL_MOUNT_ARGS $AITER_MOUNT_ARGS $CHECKOUT_DOCKER_ARGS"
""",
    )
    text = replace_once(
        text,
        'CHECKOUT_DOCKER_ARGS="-e SGLANG_USE_CHECKOUT_RUNTIME=$SGLANG_USE_CHECKOUT_RUNTIME"',
        'SGLANG_RUNTIME_WORKSPACE="${SGLANG_RUNTIME_WORKSPACE:-$GITHUB_WORKSPACE}"\n'
        'CHECKOUT_DOCKER_ARGS="-e SGLANG_USE_CHECKOUT_RUNTIME=$SGLANG_USE_CHECKOUT_RUNTIME"',
    )
    text = replace_once(
        text,
        'CHECKOUT_SHA="$(git -C "$GITHUB_WORKSPACE" rev-parse HEAD)"',
        'CHECKOUT_SHA="$(git -C "$SGLANG_RUNTIME_WORKSPACE" rev-parse HEAD)"',
    )
    text = replace_once(
        text,
        '-C "$GITHUB_WORKSPACE" -cf - . | tar -C "$CHECKOUT_STAGE" -xf -',
        '-C "$SGLANG_RUNTIME_WORKSPACE" -cf - . | tar -C "$CHECKOUT_STAGE" -xf -',
    )
    text = replace_once(
        text,
        'cat > "$WORKDIR/prefill_entry.sh" <<EOF',
        """cat > "$WORKDIR/install_checkout_aiter.sh" <<'EOF'
#!/bin/bash
set -euo pipefail

if [[ ! -d /aiter-under-test ]]; then
  echo "[checkout-aiter] /aiter-under-test not mounted; using image-baked aiter"
  exit 0
fi

RUNTIME_AITER="${RUNTIME_AITER:-/tmp/aiter-under-test-runtime}"
echo "[checkout-aiter] reinstalling aiter from /aiter-under-test"
rm -rf "$RUNTIME_AITER"
mkdir -p "$RUNTIME_AITER"
tar --exclude='__pycache__' --exclude='*.pyc' \
  -C /aiter-under-test -cf - . | tar -C "$RUNTIME_AITER" -xf -

python3 -m pip uninstall -y amd-aiter aiter || true
MAX_JOBS="${AITER_MAX_JOBS:-64}" PREBUILD_KERNELS="${AITER_PREBUILD_KERNELS:-0}" GPU_ARCHS="${GPU_ARCHS:-gfx950}" \
  python3 -m pip install --no-build-isolation -e "$RUNTIME_AITER"
python3 -m pip show amd-aiter || python3 -m pip show aiter || true
EOF

cat > "$WORKDIR/prefill_entry.sh" <<EOF
""",
    )
    text = replace_all(
        text,
        'bash "\\$CIDIR/install_checkout_sglang.sh"\n',
        'bash "\\$CIDIR/install_checkout_sglang.sh"\n'
        'bash "\\$CIDIR/install_checkout_aiter.sh"\n',
        min_count=2,
    )
    text = replace_once(
        text,
        "    bash \\$CIDIR/install_checkout_sglang.sh\n",
        "    bash \\$CIDIR/install_checkout_sglang.sh\n"
        "    bash \\$CIDIR/install_checkout_aiter.sh\n",
    )
    text = replace_all(
        text,
        "/host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}",
        "/ci_workdir",
    )
    text = replace_once(
        text,
        """NODELIST_ARG=()
[[ -n "${SLURM_NODELIST:-}" ]] && NODELIST_ARG=(--nodelist="$SLURM_NODELIST")
""",
        """NODELIST_ARG=()
if [[ -n "${SLURM_RESERVATION:-}" ]]; then
    NODELIST_ARG=()
elif [[ -n "${SLURM_NODELIST:-}" ]]; then
    NODELIST_ARG=()
fi

RESERVATION_ARG=()
[[ -n "${SLURM_RESERVATION:-}" ]] && RESERVATION_ARG=(--reservation "$SLURM_RESERVATION")

PARTITION_ARG=()
[[ -n "${SLURM_PARTITION:-}" ]] && PARTITION_ARG=(-p "$SLURM_PARTITION")
""",
    )
    text = replace_once(
        text,
        """WORKDIR="$1"; PW="${2:-1}"; DW="${3:-1}"
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
PNODES=("${NODES[@]:0:PW}")
DNODES=("${NODES[@]:PW:DW}")
PNODE="${PNODES[0]}"; DNODE="${DNODES[0]}"
PIP=$(getent ahostsv4 "$PNODE" | head -1 | awk '{print $1}')
DIP=$(getent ahostsv4 "$DNODE" | head -1 | awk '{print $1}')
""",
        """WORKDIR="$1"; PW="${2:-1}"; DW="${3:-1}"

DRIVE_LOCK="$WORKDIR/drive.lock"
if ! mkdir "$DRIVE_LOCK" 2>/dev/null; then
  echo "[drive] another launcher instance already owns $DRIVE_LOCK; exiting duplicate task"
  exit 0
fi
trap 'rmdir "$DRIVE_LOCK" 2>/dev/null || true' EXIT

expand_nodelist() {
  local raw="$1"
  if command -v scontrol >/dev/null 2>&1; then
    scontrol show hostnames "$raw" 2>/dev/null && return 0
  fi
  python3 - "$raw" <<'PY'
import re
import sys

raw = sys.argv[1]

def split_top(value: str) -> list[str]:
    parts, buf, depth = [], [], 0
    for ch in value:
        if ch == "," and depth == 0:
            if buf:
                parts.append("".join(buf))
                buf = []
            continue
        if ch == "[":
            depth += 1
        elif ch == "]" and depth:
            depth -= 1
        buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts

def expand_one(item: str) -> list[str]:
    match = re.fullmatch(r"([^\\[]*)\\[([^\\]]+)\\](.*)", item)
    if not match:
        return [item] if item else []
    prefix, body, suffix = match.groups()
    expanded = []
    for piece in body.split(","):
        range_match = re.fullmatch(r"(\\d+)-(\\d+)", piece)
        if range_match:
            start_s, end_s = range_match.groups()
            width = max(len(start_s), len(end_s))
            start, end = int(start_s), int(end_s)
            step = 1 if end >= start else -1
            expanded.extend(
                f"{prefix}{value:0{width}d}{suffix}"
                for value in range(start, end + step, step)
            )
        else:
            expanded.append(f"{prefix}{piece}{suffix}")
    return expanded

for part in split_top(raw):
    for node in expand_one(part.strip()):
        print(node)
PY
}

RAW_NODELIST="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-${SPUR_JOB_NODELIST:-${SPUR_NODELIST:-}}}}"
if [[ -z "$RAW_NODELIST" ]]; then
  echo "[drive] ERROR: no Slurm/SPUR nodelist env found" >&2
  env | sort | grep -E '^(SLURM|SPUR)_' >&2 || true
  exit 1
fi
mapfile -t NODES < <(expand_nodelist "$RAW_NODELIST")
if (( ${#NODES[@]} < PW + DW )); then
  echo "[drive] ERROR: need $((PW + DW)) nodes, got ${#NODES[@]} from $RAW_NODELIST: ${NODES[*]}" >&2
  exit 1
fi
PNODES=("${NODES[@]:0:PW}")
DNODES=("${NODES[@]:PW:DW}")
PNODE="${PNODES[0]}"; DNODE="${DNODES[0]}"
PIP=$(getent ahostsv4 "$PNODE" | head -1 | awk '{print $1}')
DIP=$(getent ahostsv4 "$DNODE" | head -1 | awk '{print $1}')
PIP="${PIP:-$PNODE}"
DIP="${DIP:-$DNODE}"

run_on_node() {
  local node="$1"
  shift
  local self_full self_short
  self_full="$(hostname)"
  self_short="$(hostname -s 2>/dev/null || hostname)"
  if [[ "$node" == "$self_full" || "$node" == "$self_short" ]]; then
    "$@"
  else
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o ConnectTimeout=30 "$node" "$@"
  fi
}
""",
    )
    text = replace_once(
        text,
        """for n in "${PNODES[@]}"; do
  ( srun --overlap -N1 --nodelist="$n" bash "$WORKDIR/prefill.sh" > "$WORKDIR/prefill_$n.log" 2>&1
    echo "prefill@$n rc=$?" > "$WORKDIR/server_exit_prefill_$n" ) &
done
for n in "${DNODES[@]}"; do
  ( srun --overlap -N1 --nodelist="$n" bash "$WORKDIR/decode.sh" > "$WORKDIR/decode_$n.log" 2>&1
    echo "decode@$n rc=$?" > "$WORKDIR/server_exit_decode_$n" ) &
done
sleep 5
""",
        """for n in "${PNODES[@]}"; do
  ( run_on_node "$n" bash "$WORKDIR/prefill.sh" > "$WORKDIR/prefill_$n.log" 2>&1
    echo "prefill@$n rc=$?" > "$WORKDIR/server_exit_prefill_$n" ) &
done
for n in "${DNODES[@]}"; do
  ( run_on_node "$n" bash "$WORKDIR/decode.sh" > "$WORKDIR/decode_$n.log" 2>&1
    echo "decode@$n rc=$?" > "$WORKDIR/server_exit_decode_$n" ) &
done
sleep 5
""",
    )
    text = replace_once(
        text,
        """# Each server's srun runs here on the login node and returns exactly when its
# compute-node container exits. Wrap it so the return code lands in a marker
# file on shared NFS. The monitor then watches for markers instead of polling
# PIDs -- unambiguous (no zombie/kill -0 guesswork) and it records which role
# died and with what code. (A hung-but-alive server is NOT caught here; that is
# bounded by bench.sh's health-wait timeout.)
""",
        """# SPUR does not fully match Slurm's nested srun/--overlap behavior, so dispatch
# per-node work through SSH after the outer srun reserves the nodes. Wrap each
# command so the return code lands in a marker file on shared NFS. The monitor
# then watches for markers instead of polling PIDs and records which role died.
""",
    )
    text = replace_once(
        text,
        """( srun --overlap -N1 --nodelist="$PNODE" bash "$WORKDIR/bench.sh" "$PIP" "$DIP" > "$WORKDIR/bench.log" 2>&1
  echo $? > "$WORKDIR/bench_exit" ) &
""",
        """( run_on_node "$PNODE" bash "$WORKDIR/bench.sh" "$PIP" "$DIP" > "$WORKDIR/bench.log" 2>&1
  echo $? > "$WORKDIR/bench_exit" ) &
""",
    )
    text = replace_once(
        text,
        """for n in "${PNODES[@]}"; do srun --overlap -N1 --nodelist="$n" docker kill mi355x_prefill >/dev/null 2>&1 || true; done
for n in "${DNODES[@]}"; do srun --overlap -N1 --nodelist="$n" docker kill mi355x_decode  >/dev/null 2>&1 || true; done
""",
        """for n in "${PNODES[@]}"; do run_on_node "$n" docker kill mi355x_prefill >/dev/null 2>&1 || true; done
for n in "${DNODES[@]}"; do run_on_node "$n" docker kill mi355x_decode  >/dev/null 2>&1 || true; done
""",
    )
    text = replace_once(
        text,
        """salloc -p "$SLURM_PARTITION" -N"$TOTAL_NODES" "${NODELIST_ARG[@]}" "${EXCLUDE_ARG[@]}" "${EXCLUSIVE_ARG[@]}" \\
    --job-name "$JOB_NAME" -t "$TIME_LIMIT" \\
""",
        """SBATCH_SCRIPT="$WORKDIR/submit.sh"
BATCH_EXIT="$WORKDIR/batch_exit"
rm -f "$BATCH_EXIT"

cat > "$SBATCH_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --nodes=$TOTAL_NODES
#SBATCH --ntasks=$TOTAL_NODES
#SBATCH --ntasks-per-node=1
#SBATCH --time=$TIME_LIMIT
#SBATCH --output=$WORKDIR/slurm-%j.out
#SBATCH --error=$WORKDIR/slurm-%j.err
EOF
if [[ -n "${SLURM_PARTITION:-}" ]]; then
    printf '#SBATCH --partition=%s\\n' "$SLURM_PARTITION" >> "$SBATCH_SCRIPT"
fi
if [[ -n "${SLURM_RESERVATION:-}" ]]; then
    printf '#SBATCH --reservation=%s\\n' "$SLURM_RESERVATION" >> "$SBATCH_SCRIPT"
elif [[ -n "${SLURM_NODELIST:-}" ]]; then
    printf '#SBATCH --nodelist=%s\\n' "$SLURM_NODELIST" >> "$SBATCH_SCRIPT"
fi
if [[ "${SLURM_EXCLUSIVE:-1}" == "1" ]]; then
    printf '#SBATCH --exclusive\\n' >> "$SBATCH_SCRIPT"
fi
if [[ -n "${SLURM_EXCLUDE:-}" ]]; then
    printf '#SBATCH --exclude=%s\\n' "$SLURM_EXCLUDE" >> "$SBATCH_SCRIPT"
fi
cat >> "$SBATCH_SCRIPT" <<EOF

set -euo pipefail
set +e
bash "$WORKDIR/drive.sh" "$WORKDIR" "$PW" "$DW"
rc=\\$?
echo "\\$rc" > "$BATCH_EXIT"
exit "\\$rc"
EOF
chmod +x "$SBATCH_SCRIPT"

parse_batch_job_id() {
    local output="$1"
    output="${output//$'\\r'/}"
    if [[ "$output" =~ Submitted[[:space:]]+batch[[:space:]]+job[[:space:]]+([0-9]+) ]]; then
        printf '%s\\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    if [[ "$output" =~ ^[[:space:]]*([0-9]+)(\\;.*)?[[:space:]]*$ ]]; then
        printf '%s\\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    return 1
}

echo "=== submitting batch script ==="
sed -n '1,80p' "$SBATCH_SCRIPT"
SBATCH_OUTPUT="$(sbatch "$SBATCH_SCRIPT")"
SALLOC_RC=$?
echo "$SBATCH_OUTPUT"
if [[ "$SALLOC_RC" -eq 0 ]]; then
    if ! BATCH_JOB_ID="$(parse_batch_job_id "$SBATCH_OUTPUT")"; then
        echo "ERROR: unable to parse batch job id from sbatch output: $SBATCH_OUTPUT" >&2
        SALLOC_RC=1
    else
        echo "batch_job_id=$BATCH_JOB_ID"
        SLURM_OUT="$WORKDIR/slurm-${BATCH_JOB_ID}.out"
        SLURM_ERR="$WORKDIR/slurm-${BATCH_JOB_ID}.err"
        while [[ ! -f "$BATCH_EXIT" ]]; do
            queue_line="$(squeue -j "$BATCH_JOB_ID" -h 2>/dev/null || true)"
            if [[ -n "$queue_line" ]]; then
                echo "[batch] $queue_line"
                sleep 30
            else
                sleep 5
                [[ -f "$BATCH_EXIT" ]] || { echo "ERROR: batch job $BATCH_JOB_ID ended without $BATCH_EXIT" >&2; SALLOC_RC=1; break; }
            fi
        done
        if [[ -f "$BATCH_EXIT" ]]; then
            SALLOC_RC="$(cat "$BATCH_EXIT" 2>/dev/null || echo 1)"
        fi
        echo "--- slurm stdout ---"
        cat "$SLURM_OUT" 2>/dev/null || true
        echo "--- slurm stderr ---"
        cat "$SLURM_ERR" 2>/dev/null || true
    fi
fi
""",
    )
    text = replace_once(
        text,
        """    bash "$WORKDIR/drive.sh" "$WORKDIR" "$PW" "$DW"
SALLOC_RC=$?
""",
        "",
    )

    launcher.write_text(text, encoding="utf-8")
    print(f"patched {launcher}")


def parse_concurrency_list(raw: str) -> list[int]:
    values = [item for item in raw.replace(",", " ").split() if item]
    if not values:
        raise SystemExit("concurrency list cannot be empty")
    try:
        parsed = [int(item) for item in values]
    except ValueError as exc:
        raise SystemExit(f"invalid concurrency list {raw!r}: {exc}") from exc
    if any(value <= 0 for value in parsed):
        raise SystemExit(f"concurrency values must be positive: {parsed}")
    return parsed


def parse_bool(raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"invalid boolean value: {raw!r}")


def configure_recipe(args: argparse.Namespace) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required: python3 -m pip install pyyaml") from exc

    root = Path(args.sglang_workspace)
    recipe = root / RECIPE_PATH
    payload = yaml.safe_load(recipe.read_text(encoding="utf-8"))

    payload["bench"]["concurrencies"] = parse_concurrency_list(args.concurrency_list)
    payload["bench"].setdefault("accuracy", {})["enabled"] = parse_bool(
        args.accuracy_enabled
    )

    recipe.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"configured {recipe}")
    print(f"concurrencies={payload['bench']['concurrencies']}")
    print(f"accuracy_enabled={payload['bench']['accuracy']['enabled']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    patch = subparsers.add_parser("patch-launcher")
    patch.add_argument("sglang_workspace")
    patch.set_defaults(func=patch_launcher)

    recipe = subparsers.add_parser("configure-recipe")
    recipe.add_argument("sglang_workspace")
    recipe.add_argument("--concurrency-list", required=True)
    recipe.add_argument("--accuracy-enabled", required=True)
    recipe.set_defaults(func=configure_recipe)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
