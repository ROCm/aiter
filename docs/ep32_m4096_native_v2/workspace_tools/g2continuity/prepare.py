"""Small, reversible probes for default-entry GEMM2 timing.

The official bench and its parser are used unchanged. Only one production
source is replaced for each probe, and its exact original bytes are restored.
"""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WT = ROOT.parent / "aiter"
TARGETS = {
    "kernel": WT / "aiter/ops/flydsl/kernels/mxmoe_gemm2_ep32_bm80var.py",
    "wrapper": WT / "aiter/ops/flydsl/ep32_native80.py",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def generate():
    source = ROOT / "source"
    source.mkdir(exist_ok=True)
    original = {}
    for key, path in TARGETS.items():
        data = path.read_bytes()
        frozen = source / f"baseline_{key}.py"
        if frozen.exists():
            assert frozen.read_bytes() == data, f"baseline moved: {path}"
        else:
            frozen.write_bytes(data)
        original[key] = data.decode()

    nt = replace_once(
        original["kernel"],
        "store = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), Float32)",
        "# CDNA4 NT store changes cache replacement only; waits stay intact.\n"
        "            store = fx.make_copy_atom(fx.rocdl.BufferCopy128b(2), Float32)",
    )
    nt = replace_once(nt, "name += f'_bn{BN}w{NW}_bs{BSTAGES}'", "name += f'_cnt_bn{BN}w{NW}_bs{BSTAGES}'")
    (source / "nt_kernel.py").write_text(nt)

    grow = replace_once(original["wrapper"], "_buffers={}\n", "")
    grow = replace_once(
        grow,
        "        key=(out.device,torch.cuda.current_stream(out.device).cuda_stream,max_sorted)\n"
        "        if key not in _buffers:\n"
        "            _buffers[key]=torch.empty((max_sorted,3584),dtype=torch.float32,device=out.device)\n"
        "        target=_buffers[key]",
        "        from aiter.fused_moe import _stage2_scratch\n"
        "        target=_stage2_scratch('target',(max_sorted,3584),torch.float32,out.device)",
    )
    (source / "grow_wrapper.py").write_text(grow)

    pingpong = replace_once(
        original["wrapper"],
        "        if key not in _buffers:\n"
        "            _buffers[key]=torch.empty((max_sorted,3584),dtype=torch.float32,device=out.device)\n"
        "        target=_buffers[key]",
        "        if key not in _buffers:\n"
        "            _buffers[key]=([torch.empty((max_sorted,3584),dtype=torch.float32,device=out.device) for _ in range(2)],0)\n"
        "        pair,index=_buffers[key]\n"
        "        target=pair[index]\n"
        "        _buffers[key]=(pair,1-index)",
    )
    (source / "pingpong_wrapper.py").write_text(pingpong)
    manifest = {
        "baseline": {key: sha(path.read_bytes()) for key, path in TARGETS.items()},
        "variants": {p.name: sha(p.read_bytes()) for p in source.glob("*.py")},
    }
    (ROOT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def mutate(case, restore):
    key = "kernel" if case == "nt" else "wrapper"
    variant = ROOT / "source" / f"{case}_{key}.py"
    frozen = ROOT / "source" / f"baseline_{key}.py"
    expected, replacement = (variant, frozen) if restore else (frozen, variant)
    assert TARGETS[key].read_bytes() == expected.read_bytes(), f"source changed: {TARGETS[key]}"
    TARGETS[key].write_bytes(replacement.read_bytes())
    print(f"{'restored' if restore else 'installed'} {case}: {sha(replacement.read_bytes())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("generate", "install", "restore"))
    parser.add_argument("case", nargs="?", choices=("nt", "grow", "pingpong"))
    args = parser.parse_args()
    if args.action == "generate":
        generate()
    else:
        assert args.case
        mutate(args.case, args.action == "restore")
