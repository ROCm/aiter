# DSR1 Serving 中 MoE token 数的计算方法

本文说明在以下两个脚本的当前配置下，如何估算 Serving benchmark 中最常见的 MoE 输入 token 数 `T`：

- `/app/scripts/dsr1/serve_dsr1fp4_a8w8_tp1.sh`
- `/app/scripts/dsr1/bench_dsr1fp4.sh`

当前 trace 中观测到：

| phase | 大部分 MoE 调用的 `T` |
|---|---:|
| decode | 64 |
| prefill | 15375 |

这里的 `T` 是一次 model forward 中进入 MoE layer 的 hidden-state row 数，不是 top-k route 数。DeepSeek-R1 使用 `topk=8` 时，对应的 route 数是 `T × 8`。

## 1. 两个脚本中的相关配置

### 1.1 Server 配置

`serve_dsr1fp4_a8w8_tp1.sh` 最终执行：

```bash
python -m atom.entrypoints.openai_server \
  --model "$MODEL_PATH" \
  --kv_cache_dtype fp8 \
  --block-size 64 \
  --no-enable_prefix_caching \
  --no-enable-chunked-prefill
```

与本计算直接相关的是：

- `--no-enable-chunked-prefill`：一个 prompt 的 prefill 不会被拆成多个 token chunk；scheduler 只能把若干个完整 prompt 装入一次 prefill forward。
- 脚本没有显式传入 `--max-num-batched-tokens`，因此当前 ATOM 版本采用默认值 `16384`。
- 脚本没有显式传入 `--max-num-seqs`，因此当前 ATOM 版本采用默认值 `512`。
- 脚本名及当前启动方式为 TP1；没有 data parallel 切分，因此单 rank 看到的 `T` 就是本次 forward 的总 token 数。

当前 ATOM 默认值来自 `/app/ATOM/atom/model_engine/arg_utils.py`：

```python
max_num_batched_tokens: int = 16384
max_num_seqs: int = 512
```

因此后续记为：

```text
B = max_num_batched_tokens = 16384
S = max_num_seqs           = 512
```

注意：`B=16384` 和 `S=512` 并不是这两个 shell 脚本显式写出的值，而是脚本未覆盖参数时采用的当前 ATOM 默认值。若 ATOM 默认配置变化，必须重新代入计算。

### 1.2 Benchmark 配置

`bench_dsr1fp4.sh` 中的相关参数为：

```bash
CONC=${2:-64}
ISL_LIST=(1024)
OSL=1024

python -m atom.benchmarks.benchmark_serving \
  --dataset-name=random \
  --random-input-len="$ISL" \
  --random-output-len="$OSL" \
  --random-range-ratio 1.0 \
  --num-prompts=$(( CONC * 4 )) \
  --max-concurrency="$CONC" \
  --request-rate=inf \
  --ignore-eos
```

使用默认参数运行时：

```text
C = max_concurrency = 64
P = num_prompts     = 64 × 4 = 256
Lclient             = random_input_len = 1024
O                    = random_output_len = 1024
```

其中：

- `--request-rate=inf` 会一次性制造高 backlog，scheduler 通常有足够请求填满 batch。
- `--ignore-eos` 和 `OSL=1024` 使请求不会因为提前生成 EOS 而过早退出，稳态 decode 中通常能保持 64 个 active requests。
- `--random-range-ratio=1.0` 使 benchmark 生成的每个输入都以 `1024` 为目标长度，而不是在一个长度区间内随机采样。

## 2. Decode 阶段大部分 MoE `T` 的计算

普通 autoregressive decode 的一次 forward 对每个 active sequence 处理一个新 token。当前启动命令没有启用 speculative decode，因此：

```text
每个 active sequence 每次 decode forward 贡献 1 个 token
```

一次 decode forward 可容纳的 sequence 数受以下三者限制：

1. benchmark 并发上限 `C=64`；
2. server sequence 上限 `S=512`；
3. token budget `B=16384`，单 sequence 每步只占 1 个 token。

所以稳态 decode 的最大 batch 为：

```text
Tdecode
  = min(C, S, floor(B / 1)) × 1
  = min(64, 512, 16384)
  = 64
```

因此：

```text
decode 阶段大部分 MoE T = 64
```

这不是仅由 `OSL=1024` 直接得出的。`OSL=1024`、`--ignore-eos`、`P=256` 和 `--request-rate=inf` 的作用是让服务长时间处于有 64 个 active requests 的稳态；真正决定单次 decode forward 上界的是 `--max-concurrency=64`。

Trace 中出现 `4096` 次 `T=64` 的 decode forward，也与下面的数量关系一致：

```text
总输出 token 数 = 256 requests × 1024 tokens/request = 262144
满 batch forward 数 = 262144 / 64 = 4096
```

## 3. Prefill 阶段大部分 MoE `T` 的计算

### 3.1 为什么每个请求实际是 1025 个 prefill token

benchmark 在客户端构造目标长度为 `1024` 的随机 prompt 时，使用 `add_special_tokens=False` 校准长度；server 收到文本后通过 `tokenizer.encode(prompt)` 使用默认 special-token 行为。当前模型的 `/data/models/DeepSeek-R1-0528-MXFP4/tokenizer_config.json` 配置了：

```json
{
  "add_bos_token": true,
  "bos_token": "<｜begin▁of▁sentence｜>",
  "add_eos_token": false
}
```

因此 server 会在客户端校准的 1024 个 token 前再加入 1 个 BOS token，进入模型的实际 prompt 长度为：

```text
Lserver = Lclient + 1 = 1024 + 1 = 1025
```

这也得到 Serving trace 的总量验证。256 个请求的 prefill token 总数为：

```text
256 × 1025 = 262400
```

两个版本 trace 中全部 prefill shape 按调用次数求和都恰好是 `262400`：

```text
baseline:
4×1025 + 4×3075 + 16×15375 = 262400

optimized:
3×1025 + 3×3075 + 1×4100 + 16×15375 = 262400
```

如果更换模型或 tokenizer，应先重新确认 server 是否仍会增加这 1 个 special token，不能固定假设所有模型都是 `ISL+1`。

### 3.2 一次 prefill forward 能装入多少个完整请求

server 的 token budget 是：

```text
B = 16384
```

又因为 `--no-enable-chunked-prefill`，每个长度为 `1025` 的 prompt 必须作为整体调度。一次 forward 最多能放入：

```text
Nprefill
  = floor(B / Lserver)
  = floor(16384 / 1025)
  = 15 requests
```

边界检查：

```text
15 × 1025 = 15375 <= 16384
16 × 1025 = 16400 > 16384
```

因此高 backlog 下，一个接近填满 token budget 的 prefill forward 为：

```text
Tprefill
  = Nprefill × Lserver
  = 15 × 1025
  = 15375
```

所以：

```text
prefill 阶段大部分 MoE T = 15375
```

`max_num_seqs=512` 和 `max_concurrency=64` 在这里都不是主要限制，因为一次只装入 15 个 prompt：

```text
15 < 64 < 512
```

## 4. 为什么还会出现较小的 Prefill `T`

Trace 中还出现了：

```text
1025 = 1 × 1025
3075 = 3 × 1025
4100 = 4 × 1025
```

这些 shape 表示相应 forward 只调度了 1、3 或 4 个完整 prompt。它们通常发生在启动、prefill/decode 交错、请求完成补位或尾部阶段。

`--request-rate=inf` 能制造 backlog，但异步请求进入 server、scheduler 取队列和请求状态转换仍有时序，因此不能保证每一次 prefill 都严格包含 15 个请求。脚本参数能给出 `15375` 这个高 backlog 下的主要 shape，但不能仅凭静态参数预测每个较小 shape 的准确出现次数。

## 5. 通用计算公式

对同类 benchmark，可使用以下方法估算最常见的 MoE `T`。

### Decode

设：

- `C`：benchmark 的 `max_concurrency`；
- `S`：server 的 `max_num_seqs`；
- `B`：server 的 `max_num_batched_tokens`；
- `q`：每个 active sequence 在一次 decode forward 中贡献的 token 数；普通 decode 时 `q=1`。

则：

```text
Tdecode = min(C, S, floor(B / q)) × q
```

当前配置：

```text
Tdecode = min(64, 512, floor(16384 / 1)) × 1 = 64
```

### Prefill

设：

- `Lclient`：benchmark 生成的 prompt token 长度；
- `delta_special`：server tokenizer 相比客户端长度增加的 special token 数；
- `Lserver = Lclient + delta_special`；
- `B`：server 的 `max_num_batched_tokens`；
- prefill 未启用 chunking。

则高 backlog 下：

```text
Nprefill = min(C, S, floor(B / Lserver))
Tprefill = Nprefill × Lserver
```

当前配置：

```text
Lserver  = 1024 + 1 = 1025
Nprefill = min(64, 512, floor(16384 / 1025)) = 15
Tprefill = 15 × 1025 = 15375
```

## 6. 结论与适用边界

当前两个脚本、当前 ATOM 默认配置和当前 DeepSeek-R1 tokenizer 共同决定：

```text
decode 大部分 MoE T  = 64
prefill 大部分 MoE T = 15375
```

其中：

- `T=64` 可以主要从 benchmark 的 `CONC=64` 推出，但仍以 server 的 `max_num_seqs` 和 token budget 足够大为前提。
- `T=15375` 不能只看两个 shell 文件的显式参数就完整推出；还需要知道当前 ATOM 默认 `max_num_batched_tokens=16384`，以及当前 server tokenization 将客户端的 `1024` token prompt 变为 `1025` token。
- 这些结果表示高 backlog 下的主要 shape，不保证每个 forward 都相同。请求到达和 scheduler 时序会产生较小的 prefill batch；active request 不足时也会产生小于 `64` 的 decode batch。
- 若修改 `CONC`、`ISL`、tokenizer special-token 行为、`max_num_batched_tokens`、`max_num_seqs`、chunked prefill、speculative decode、TP/DP 配置，必须重新计算并用 trace 验证。
