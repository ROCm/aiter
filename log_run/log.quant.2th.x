[aiter] WARNING: NUMA balancing is enabled, which may cause errors. It is recommended to disable NUMA balancing by running "sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'" for more details: https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#disable-numa-auto-balancing
cu_count=80
len(boundary_mnk_list)=8
len(mnk_list)=54
total valid (m, n, k) tuples with k divisible by 16: 230400
total test case count: 2
[W608 04:50:05.508289327 collection.cpp:1100] Warning: ROCTracer produced duplicate flow start: 1 (function operator())
[aiter] type hints mismatch, override to --> wvSplitKQ(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor, arg4: torch.Tensor, arg5: int) -> None
[aiter] a,b: [perf] dim: (4, 32, 8192)        dtype: torch.bfloat16, quantDtype: torch.float8_e4m3fnuz, torch avg: 39.61    us, skinny_gemm avg: 7.70     us, uplift: 414.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] a,b: [perf] dim: (4, 32, 8192)        dtype: torch.float16, quantDtype: torch.float8_e4m3fnuz, torch avg: 37.41    us, skinny_gemm avg: 7.62     us, uplift: 391.2%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
