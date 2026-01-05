# Triton GEMM tunning script

1. modify `ut.py` to the GEMM you would like to tune
2. modify `screen.py` to set the parameter space you would like to screen
3. run `nohup python screen.py 8 2112 7168 0 > out`
4. modify `view-screen.py` to set the shapes you would like to collect data for
5. 
single case testing:
rocprofv3 --kernel-trace -f csv -o res -- python3 ut.py 4 2112 7168 8 32 1024 1 2 1 1 16 0 7