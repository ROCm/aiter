"""Dump ISA for stage2 baseline (CShuffle) and swap_ab (direct epilog).

Sets FLYDSL_DUMP_IR=1 to trigger ISA generation during JIT compilation,
then locates the *_final_isa.s files under the dump directories.

Usage (inside docker/atom1):
    python dump_isa.py
    python dump_isa.py --dump-dir /tmp/isa
"""

import argparse
import glob
import os
import shutil

import torch

torch.set_default_device("cuda")

from bench_stage2 import setup_stage2_data, call_stage2


TOKEN = 1024
MODEL_DIM = 7168
INTER_DIM = 256
EXPERT = 257
TOPK = 9
BLOCK_M = 64
TILE_N = 128
TILE_K = 256


def find_isa_files(dump_dir):
    pattern = os.path.join(dump_dir, "**", "*final_isa*")
    return sorted(glob.glob(pattern, recursive=True))


def compile_and_dump(d, dump_dir, swap_ab, use_async_copy, label):
    if os.path.exists(dump_dir):
        shutil.rmtree(dump_dir)
    os.makedirs(dump_dir, exist_ok=True)

    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_DUMP_DIR"] = dump_dir

    print(f"\n{'='*60}")
    print(f"  Compiling {label}  (swap_ab={swap_ab})")
    print(f"  tile = t{BLOCK_M}x{TILE_N}x{TILE_K}")
    print(f"  dump dir = {dump_dir}")
    print(f"{'='*60}")

    call_stage2(d, TOPK, BLOCK_M, tile_n=TILE_N, swap_ab=swap_ab, use_async_copy=use_async_copy)
    torch.cuda.synchronize()

    del os.environ["FLYDSL_DUMP_IR"]
    if "FLYDSL_DUMP_DIR" in os.environ:
        del os.environ["FLYDSL_DUMP_DIR"]

    isa_files = find_isa_files(dump_dir)
    if isa_files:
        print(f"\n  ISA files for {label}:")
        for f in isa_files:
            sz = os.path.getsize(f)
            print(f"    {f}  ({sz} bytes)")
    else:
        print(f"\n  WARNING: no *final_isa* files found under {dump_dir}")
        all_files = []
        for root, dirs, files in os.walk(dump_dir):
            for fn in files:
                all_files.append(os.path.join(root, fn))
        if all_files:
            print(f"  Files present ({len(all_files)}):")
            for f in all_files[:20]:
                print(f"    {f}")
        else:
            print(f"  Directory is empty.")

    return isa_files


def main():
    parser = argparse.ArgumentParser(description="Dump stage2 ISA")
    parser.add_argument("--dump-dir", type=str, default="/tmp/isa",
                        help="Root directory for ISA dumps")
    args = parser.parse_args()

    baseline_dir = os.path.join(args.dump_dir, "baseline")
    swap_dir = os.path.join(args.dump_dir, "swap_ab")

    print(f"Setting up data: token={TOKEN}, model_dim={MODEL_DIM}, "
          f"inter_dim={INTER_DIM}, E={EXPERT}, topk={TOPK}, block_m={BLOCK_M}")
    d = setup_stage2_data(TOKEN, MODEL_DIM, INTER_DIM, EXPERT, TOPK, BLOCK_M)

    baseline_files = compile_and_dump(d, baseline_dir, swap_ab=False, use_async_copy=True,
                                      label="baseline (CShuffle)")
    swap_files = compile_and_dump(d, swap_dir, swap_ab=True, use_async_copy=True,
                                  label="swap_ab (direct epilog)")

    print(f"\n{'='*60}")
    print("Done. ISA files:")
    print(f"  Baseline: {baseline_files}")
    print(f"  Swap_ab:  {swap_files}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
