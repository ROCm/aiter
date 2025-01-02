       _            
      | |           
  __ _| |_ ___ _ __ 
 / _` | __/ _ \ '__|
| (_| | ||  __/ |   
 \__,_|\__\___|_|   
                    
AI Tensor Engine for ROCm
ATER is a unified platform for operator level requests of all the customers, and matches different customers' needs. Developers can focus on operators, and let the customers integrate this op collection into their own private/public/whatever framework.

Some summary of the features:
    C++ level API
    Python level API
    The underneath kernel could from triton/ck/asm
    Not only inference kernels, but also training kernels, and gemm+comm kerenls (so we can do any kerne+framework dirty WAs for any arch limiit)


## clone
`git clone --recursive https://github.com/ROCm/ater.git`
or
`git submodule sync ; git submodule update --init --recursive`

## install into python
under ater root dir run: `python3 setup.py develop`

## run operators supported by ater
there are number of op test, you can run them like this: `python3 op_tests/test_layernorm2d.py`
|  **Ops**   | **Description**                                                                             |
|------------|---------------------------------------------------------------------------------------------|
|GEMM        | D=AxB+C                                                                                     |
|FusedMoE    | bf16 balabala                                                                               |
|WIP         | coming soon...                                                                              |
