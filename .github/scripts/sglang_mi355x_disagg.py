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
    NODELIST_ARG=(--nodelist="$SLURM_NODELIST")
fi

RESERVATION_ARG=()
[[ -n "${SLURM_RESERVATION:-}" ]] && RESERVATION_ARG=(--reservation "$SLURM_RESERVATION")

PARTITION_ARG=()
[[ -n "${SLURM_PARTITION:-}" ]] && PARTITION_ARG=(-p "$SLURM_PARTITION")
""",
    )
    text = replace_once(
        text,
        """salloc -p "$SLURM_PARTITION" -N"$TOTAL_NODES" "${NODELIST_ARG[@]}" "${EXCLUDE_ARG[@]}" "${EXCLUSIVE_ARG[@]}" \\
    --job-name "$JOB_NAME" -t "$TIME_LIMIT" \\
""",
        """salloc "${PARTITION_ARG[@]}" -N"$TOTAL_NODES" "${NODELIST_ARG[@]}" "${RESERVATION_ARG[@]}" "${EXCLUDE_ARG[@]}" "${EXCLUSIVE_ARG[@]}" \\
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
