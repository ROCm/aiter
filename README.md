# aiter
![image](https://github.com/user-attachments/assets/9457804f-77cd-44b0-a088-992e4b9971c6)


AITER is AMD’s centralized repository that support various of high performance AI operators for AI workloads acceleration, where a good unified place for all the customer operator-level requests, which can match different customers' needs. Developers can focus on operators, and let the customers integrate this op collection into their own private/public/whatever framework.
 

Some summary of the features:
* C++ level API
* Python level API
* The underneath kernel could come from triton/ck/asm
* Not just inference kernels, but also training kernels and GEMM+communication kernels—allowing for workarounds in any kernel-framework combination for any architecture limitation.



## Installation
```
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop
```

If you happen to forget the `--recursive` during `clone`, you can use the following command after `cd aiter`
```
git submodule sync && git submodule update --init --recursive
```

## Run operators supported by aiter

There are number of op test, you can run them with: `python3 op_tests/test_layernorm2d.py`
|  **Ops**                      | **Description**                                                                                                                                                   |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|ELEMENT WISE                   | ops: + - * /                                                                                                                                                      |
|SIGMOID                        | (x) = 1 / (1 + e^-x)                                                                                                                                              |
|AllREDUCE                      | Reduce + Broadcast                                                                                                                                                |
|KVCACHE                        | W_K W_V                                                                                                                                                           |
|MHA                            | Multi-Head Attention                                                                                                                                              |
|MLA                            | Multi-head Latent Attention with [KV-Cache layout](https://docs.flashinfer.ai/tutorials/kv_layout.html#page-table-layout )                                        |
|PA                             | Paged Attention                                                                                                                                                   |
|FusedMoe                       | Mixture of Experts                                                                                                                                                |
|QUANT                          | BF16/FP16 -> FP8/INT4                                                                                                                                             |
|RMSNORM                        | root mean square                                                                                                                                                  |
|LAYERNORM                      | x = (x - u) / (σ2 + ϵ) e*0.5                                                                                                                                      |
|ROPE                           | Rotary Position Embedding                                                                                                                                         |
|GEMM                           | D=αAβB+C                                                                                                                                                          |

