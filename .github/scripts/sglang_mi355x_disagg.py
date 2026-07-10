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
-v /it-share:/it-share:ro -v $HOME:/host_home"
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
-v $WORKDIR:/ci_workdir $MODEL_MOUNT_ARGS $AITER_MOUNT_ARGS"
""",
    )
    text = replace_once(
        text,
        "# ---------------------------------------------------------------------------\n"
        "# Write per-role scripts that srun dispatches to each compute node.\n"
        "# ---------------------------------------------------------------------------\n",
        """# Install the aiter checkout under test inside each serving container. The
# checkout is mounted read-only from shared storage; copy it to /tmp because
# editable installs write metadata into the source tree.
AITER_INSTALL_SNIPPET='
if [ -d /aiter-under-test ]; then
  rm -rf /tmp/aiter-under-test
  cp -a /aiter-under-test /tmp/aiter-under-test
  cd /tmp/aiter-under-test
  python3 -m pip uninstall -y amd-aiter aiter || true
  MAX_JOBS="${AITER_MAX_JOBS:-64}" PREBUILD_KERNELS="${AITER_PREBUILD_KERNELS:-0}" GPU_ARCHS="${GPU_ARCHS:-gfx950}" python3 -m pip install --no-build-isolation -e .
  python3 -m pip show amd-aiter || python3 -m pip show aiter || true
  cd - >/dev/null
fi
'

# ---------------------------------------------------------------------------
# Write per-role scripts that srun dispatches to each compute node.
# ---------------------------------------------------------------------------
""",
    )
    text = replace_once(
        text,
        """  $IMAGE python3 -m sglang.launch_server \\
  --model-path $MODEL_PATH --host 0.0.0.0 --port $PPORT \\
  $COMMON_FLAGS --disaggregation-mode prefill --disaggregation-bootstrap-port $PBOOT
""",
        """  $IMAGE bash -lc '
    set -ex
    $AITER_INSTALL_SNIPPET
    exec python3 -m sglang.launch_server \\
      --model-path $MODEL_PATH --host 0.0.0.0 --port $PPORT \\
      $COMMON_FLAGS --disaggregation-mode prefill --disaggregation-bootstrap-port $PBOOT
  '
""",
    )
    text = replace_once(
        text,
        """  $IMAGE python3 -m sglang.launch_server \\
  --model-path $MODEL_PATH --host 0.0.0.0 --port $DPORT \\
  $COMMON_FLAGS --disaggregation-mode decode --disaggregation-bootstrap-port $DBOOT
""",
        """  $IMAGE bash -lc '
    set -ex
    $AITER_INSTALL_SNIPPET
    exec python3 -m sglang.launch_server \\
      --model-path $MODEL_PATH --host 0.0.0.0 --port $DPORT \\
      $COMMON_FLAGS --disaggregation-mode decode --disaggregation-bootstrap-port $DBOOT
  '
""",
    )
    text = replace_once(
        text,
        "CIDIR=/host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}",
        "CIDIR=/ci_workdir",
    )
    text = replace_once(
        text,
        "OUT=/host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}/raw_conc\\${C}.json",
        "OUT=/ci_workdir/raw_conc\\${C}.json",
    )
    text = replace_once(
        text,
        """NODELIST_ARG=()
[[ -n "${SLURM_NODELIST:-}" ]] && NODELIST_ARG=(--nodelist="$SLURM_NODELIST")
""",
        """NODELIST_ARG=()
[[ -n "${SLURM_NODELIST:-}" ]] && NODELIST_ARG=(--nodelist="$SLURM_NODELIST")

PARTITION_ARG=()
[[ -n "${SLURM_PARTITION:-}" ]] && PARTITION_ARG=(-p "$SLURM_PARTITION")
""",
    )
    text = replace_once(
        text,
        """salloc -p "$SLURM_PARTITION" -N"$TOTAL_NODES" "${NODELIST_ARG[@]}" "${EXCLUSIVE_ARG[@]}" \\
    --job-name "$JOB_NAME" -t "$TIME_LIMIT" \\
""",
        """salloc "${PARTITION_ARG[@]}" -N"$TOTAL_NODES" "${NODELIST_ARG[@]}" "${EXCLUSIVE_ARG[@]}" \\
    --job-name "$JOB_NAME" -t "$TIME_LIMIT" \\
""",
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
