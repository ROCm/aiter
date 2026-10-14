"""Launch one of the frozen B8 implementations on eight gfx950 GPUs."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=["monokernel", "hybrid5", "hybrid4"], required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--blocks", type=int, choices=[1, 8], default=1)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    command = [
        sys.executable,
        str(directory / f"bench_{args.variant}.py"),
        "--world",
        "8",
        "--batches",
        "8",
        "--count",
        str(args.count),
        "--repeats",
        str(args.repeats),
        "--blocks",
        str(args.blocks),
        "--variants",
        "moe_fused",
        "projection_attnres_moe_fused",
        "--tuned",
        "--buffer",
        "--lds",
        "--word-zero-guard",
        "--down-lds",
        "--down-chunk",
        "128",
        "--down-unroll",
        "3",
        "--batch-tail",
        "--dense-lds",
        "--projection-unroll",
        "4",
        "--projection-buffer",
        "--coherent-payloads",
        "--normal-launch",
        "--mfma-lanes",
        "--grid",
        "512",
        "--front32",
        "--front-unroll",
        "14",
        "--wave-publish",
        "--tail-rows",
        "32",
        "--output",
        str(args.output.resolve()),
    ]
    print(shlex.join(command), flush=True)
    if args.print_command:
        return
    if not (args.model / "model.safetensors.index.json").is_file():
        parser.error("--model must point to the Kimi K3 checkpoint directory")
    env = os.environ.copy()
    env["KIMI_K3_MODEL"] = str(args.model.resolve())
    env["PYTHONPATH"] = os.pathsep.join(
        [str(directory.parent), str(directory), env.get("PYTHONPATH", "")]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".command.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
