# Build and Run the Aiter Container as a Non-root User


## Build a container with Aiter installed and add a non-root user using the Dockerfile provided below:

```
ARG BASE_DOCKER="rocm/pytorch:latest"
FROM $BASE_DOCKER
RUN pip install pandas zmq einops && \
    pip install numpy==1.26.2
# Create a new user
RUN useradd -m newuser
# Switch to the new user
USER newuser
# Set the working directory
WORKDIR /home/newuser
RUN rm -rf aiter && \
    git clone --recursive https://github.com/ROCm/aiter.git && \
    cd aiter && \
    python3 setup.py develop
```

## Build the container image:

```
docker build --no-cache -t rocm-aiter:test -f Dockerfile.aiter .
```


## Run the container:

This command runs a Docker container interactively with access to GPU devices, adding the current user's render and video group IDs to ensure proper permissions for accessing /dev/dri and /dev/kfd.


```
docker run -it --device=/dev/dri --device=/dev/kfd  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) rocm-aiter:test /bin/bash
```


## Check the permission to GPU access in the container:


```
newuser@0d2817135822:~$ rocminfo
ROCk module version 6.12.12 is loaded
=====================
HSA System Attributes
=====================
Runtime Version:         1.15
Runtime Ext Version:     1.7
System Timestamp Freq.:  1000.000000MHz
Sig. Max Wait Duration:  18446744073709551615 (0xFFFFFFFFFFFFFFFF) (timestamp count)
Machine Model:           LARGE
System Endianness:       LITTLE
Mwaitx:                  DISABLED
XNACK enabled:           NO
DMAbuf Support:          YES
VMM Support:             YES
==========
HSA Agents
==========
*******
Agent 1
*******
  Name:                    INTEL(R) XEON(R) PLATINUM 8568Y+
  Uuid:                    CPU-XX
  Marketing Name:          INTEL(R) XEON(R) PLATINUM 8568Y+
  Vendor Name:             CPU
  Feature:                 None specified
......
```

## Run the aiter tests to check the permission:

```
newuser@0d2817135822:~/aiter$ python3 op_tests/test_gemm_a8w8_blockscale.py
[aiter] WARNING: NUMA balancing is enabled, which may cause errors. It is recommended to disable NUMA balancing by running "sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'" for more details: https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#disable-numa-auto-balancing
[aiter] start build [module_aiter_enum] under /home/newuser/aiter/aiter/jit/build/module_aiter_enum
Successfully preprocessed all matching files.
[aiter] finish build [module_aiter_enum], cost 19.54064721s
[aiter]
calling test_gemm(dtype                        = torch.bfloat16,
                  m                            = 16,
                  n                            = 1536,
                  k                            = 7168)
/opt/conda/envs/py_3.12/lib/python3.12/site-packages/redis/connection.py:77: UserWarning: redis-py works best with hiredis. Please consider installing
  warnings.warn(msg)
[W801 06:34:09.405641462 collection.cpp:1098] Warning: ROCTracer produced duplicate flow start: 1 (function operator())
[aiter] shape is M:16, N:1536, K:7168, found padded_M: 16, N:1536, K:7168 is tuned on cu_num = 304 in CKGEMM , kernel name is a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1!
[aiter] start build [module_gemm_a8w8_blockscale] under /home/newuser/aiter/aiter/jit/build/module_gemm_a8w8_blockscale
Successfully preprocessed all matching files.
......
```
