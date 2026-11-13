"""Generate/temporarily install one sorted-A candidate; no tile sweep."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OPS = ROOT.parent / "aiter/aiter/ops/flydsl"
SOURCE = ROOT / "source"
TARGETS = {"kernel": OPS / "kernels/mxfp4_gemm1_ep32.py", "wrapper": OPS / "mxfp4_gemm1_kernels.py"}
EXTRAS = {"ep32_sorted_a.py": OPS / "ep32_sorted_a.py", "ep32_sorted_a_kernel.py": OPS / "kernels/ep32_sorted_a.py"}


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def generate():
    SOURCE.mkdir(exist_ok=True)
    for key, path in TARGETS.items():
        frozen = SOURCE / f"baseline_{key}.py"
        if frozen.exists():
            assert frozen.read_bytes() == path.read_bytes(), path
        else:
            frozen.write_bytes(path.read_bytes())
    kernel = (SOURCE / "baseline_kernel.py").read_text()
    kernel = replace_once(kernel, "aq_num_records = fx.Int64(i32_ntok * fx.Int32(A_ROW_BYTES))",
        "aq_num_records = fx.Int64(i32_total_m_blocks) * fx.Int64(80 * A_ROW_BYTES)")
    start = kernel.index("    cached_actual_row = []")
    stop = kernel.index("    N0_HALF = N_OUT // 32", start)
    kernel = kernel[:start] + (
        "    # Sorted-A experiment: addresses are contiguous; no token gather.\n"
        "    cached_actual_row = []\n"
        "    for sub in range_constexpr(kSubBlocks):\n"
        "        cached_actual_row.append([m_row + wave_n * fx.Int32(BM // 4)\n"
        "            + fx.Int32(sub * 8 + row_group * 4) + lane_div_16\n"
        "            for row_group in range_constexpr(2)])\n"
        "    tail_row = m_row + fx.Int32(BM) + wave_n * fx.Int32(4) + lane_div_16\n\n"
    ) + kernel[stop:]
    kernel = replace_once(kernel, 'variant_tag = "nt" if use_nt else "cached"',
        'variant_tag = "nt_asorted" if use_nt else "cached_asorted"')
    (SOURCE / "sorted_kernel.py").write_text(kernel)
    wrapper = (SOURCE / "baseline_wrapper.py").read_text()
    wrapper = replace_once(wrapper,
        "    grid = gemm1_grid(n_tokens, BM, NE=NE, TOPK=topk, INTER=D_INTER, BN=BN)",
        "    if ep32:\n"
        "        assert (BM,D_HIDDEN,D_INTER,NE,n_tokens) == (80,3584,3072,28,4096), 'sorted-A experiment only supports this bench shape'\n"
        "        from .ep32_sorted_a import prepare\n"
        "        a_quant = prepare(a_quant,m_indices,cumsum_tensor,n_tokens)\n"
        "    grid = gemm1_grid(n_tokens, BM, NE=NE, TOPK=topk, INTER=D_INTER, BN=BN)")
    (SOURCE / "sorted_wrapper.py").write_text(wrapper)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob("*.py")}
    (ROOT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def mutate(restore):
    for key, path in TARGETS.items():
        frozen, candidate = SOURCE / f"baseline_{key}.py", SOURCE / f"sorted_{key}.py"
        expected, replacement = (candidate, frozen) if restore else (frozen, candidate)
        assert path.read_bytes() == expected.read_bytes(), f"source changed: {path}"
    for src, path in EXTRAS.items():
        if restore:
            assert path.read_bytes() == (SOURCE / src).read_bytes(), f"helper changed: {path}"
        else:
            assert not path.exists(), path
    for key, path in TARGETS.items():
        path.write_bytes((SOURCE / f"{'baseline' if restore else 'sorted'}_{key}.py").read_bytes())
    for src, path in EXTRAS.items():
        if restore:
            path.unlink()
        else:
            path.write_bytes((SOURCE / src).read_bytes())
    print("restored sorted-A" if restore else "installed sorted-A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("generate", "install", "restore"))
    args = parser.parse_args()
    if args.action == "generate":
        generate()
    else:
        mutate(args.action == "restore")
