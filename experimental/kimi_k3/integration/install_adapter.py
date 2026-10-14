"""Select an adapter after applying vllm-kimi-k3.patch to the AMD K3 model."""

import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=["monokernel", "hybrid5", "hybrid4"], required=True
    )
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check patch applicability without writing the adapter",
    )
    args = parser.parse_args()
    amd = args.vllm_source / "vllm/models/kimi_k3/amd"
    linear = (amd / "linear.py").read_text()
    kda = (amd / "kda.py").read_text()
    if "CompleteMoE(self," not in linear or "defer_output_projection" not in kda:
        parser.error("Apply integration/vllm-kimi-k3.patch to this vLLM checkout first")
    source = Path(__file__).resolve().parent / f"{args.variant}.py"
    if not args.check:
        shutil.copyfile(source, amd / "monokernel.py")
    print(f'{args.variant}: {source} -> {amd / "monokernel.py"}')


if __name__ == "__main__":
    main()
