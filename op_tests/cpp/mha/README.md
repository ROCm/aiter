# mha benchmark

This folder contains benchmark scripts for mha_fwd and mha_bwd. The implementation are ported from ck.

Current unsupported features:
* `bench_mha_fwd` vlayout col_major
* `bench_mha_fwd` appendkv

## build
Make sure `aiter` has been installed, then run this command under this folder:
```
bash build_mha.sh
```
This will result in executables `bench_mha_fwd` and `bench_mha_bwd` in this folder, or you can just run
```
python3 -c "import aiter; aiter.compile_bench_mha_fwd()"
python3 -c "import aiter; aiter.compile_bench_mha_bwd()"
```
to build the executable separately.

## run
You can type `./bench_mha_fwd -?` to list all the arguments.
Or you can just run the smoke test to check the integrity of your executable by `bash smoke_test_fwd.sh`