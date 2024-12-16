# ater
AI Tensor Engine for ROCm

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
