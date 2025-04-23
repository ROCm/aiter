# mha benchmark

This folder contains benchmark scripts for mha_fwd and mha_bwd.

Current unsupported features:
* `benchmark_mha_fwd` vlayout col_major
* `benchmark_mha_fwd` appendkv

## build and run
This build method is independent from aiter and torch. By running
```
bash build_mha.sh
```
This will result in executables `benchmark_mha_fwd` and `benchmark_mha_bwd` in this folder. You can type `./benchmark_mha_fwd -?` to list all the arguments.

To validate the integrity of the executable, we also provide `smoke_test` scripts.

One possible way to integrate aiter::mha into your project is to 
```
python3 compile.py
```
to build `libmha_fwd.so` and `libmha_bwd.so` and link them into your project.