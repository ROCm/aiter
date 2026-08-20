# Comm-Fused MoE Full-Width Reduce 代码导读

本文只解释 M=2048 full-width KID 的 reduce 热路径，对应当前工作树：

```text
aiter/ops/flydsl/comm_fused_moe_host.py
aiter/ops/flydsl/kernels/comm_fused_moe/full_width.py
aiter/ops/flydsl/kernels/comm_fused_moe/collectives.py
```

这里的 reduce 分为两步：

1. `local_reduce`：本 rank 内对 TOPK route 和 shared partial 求和。
2. `tp_reduce_scatter`：读取 TP8 八个 rank 的 local partial，求和后按 token 行分片。

整体调度是：

```text
Stage2 GEMM
    → local_reduce
    → partial barrier
    → tp_reduce_scatter
    → reduced-payload barrier
    → tp_all_gather
```

## 1. Host 中的实际调用顺序

`_FullWidthRunner.__call__()` 中的相关代码：

```python
_run_compiled(
    k.compile_stage2_compute(),
    (ptr_arg(self.route), *common, stream),
)
_run_compiled(
    k.compile_stage2_local_reduce(),
    (
        ptr_arg(self.route),
        ptr_arg(self.partial),
        ptr_arg(shared_partial),
        stream,
    ),
)
_barrier(
    self.partial,
    self.partial_flat_base,
    self.partial_ready,
    stream,
)
_run_compiled(
    k.compile_stage2_tp_reduce_scatter(),
    (
        fx.Int64(self.partial_flat_base),
        ptr_arg(self.reduced_shard),
        ptr_arg(self.reduced_payload),
        ptr_arg(self.reduced_scale),
        self.rank,
        stream,
    ),
)
_barrier(
    self.reduced_payload,
    self.reduced_payload_base,
    self.reduced_ready,
    stream,
)
_run_compiled(
    k.compile_stage2_tp_all_gather(),
    (
        fx.Int64(self.reduced_payload_base),
        fx.Int64(self.reduced_scale_base),
        ptr_arg(self.output),
        self.rank,
        stream,
    ),
)
```

这些调用使用同一 stream，因此顺序是严格的。`local_reduce` 不会与
`tp_reduce_scatter` 并发；中间的 barrier 保证所有 TP rank 的 partial 都已
发布。

## 2. 固定 Shape 与并行常量

`full_width.py` 中的常量：

```python
M = 2048
H = 7168
TOPK = 6
TP = 8

SHARD_ROWS = M // TP
BLOCK = 256
REDUCE_SCATTER_GRID = 128
LOCAL_WORKERS = 640
VECTOR_WIDTH = 8
GROUPS_PER_ROW = H // 32
LOCAL_COLUMN_TILES = (H + BLOCK * VECTOR_WIDTH - 1) // (
    BLOCK * VECTOR_WIDTH
)
```

展开后：

```text
SHARD_ROWS        = 2048 / 8 = 256
GROUPS_PER_ROW    = 7168 / 32 = 224
BLOCK * VECTOR    = 256 * 8 = 2048 columns
LOCAL_COLUMN_TILES = ceil(7168 / 2048) = 4
```

## 3. Reduce 前的三种数据布局

### 3.1 Stage2 GEMM 的 route

Host 分配：

```python
self.route = torch.empty(
    (M, TOPK, H + H // 8),
    dtype=torch.uint8,
    device=device,
)
```

每个 token 有六条 route，每条 route 的布局是：

```text
[H bytes FP8 payload][H/8 bytes E8M0 scales]
```

因此：

```python
route_row_bytes = H + H // 8
```

route 的 scale group 大小是 8：每 8 个 FP8 值共用一个 E8M0 scale。

### 3.2 Local Reduce 的 partial

`partial` 是一块连续 workspace：

```text
[M * H bytes FP8 payload]
[M * (H/32) bytes E8M0 scales]
[8-byte epoch]
```

partial 的 scale group 大小是 32：每 32 个 FP8 值共用一个 scale。

### 3.3 TP Reduce-Scatter 的输出

Reduce-scatter 有三个输出 pointer：

```text
output  = 本 rank 的 [SHARD_ROWS, H] BF16 reduced shard
payload = 本 rank 的 [SHARD_ROWS, H] FP8 reduced payload
scales  = 本 rank 的 [SHARD_ROWS, H/32] E8M0 scales
```

`payload` 尾部另有 epoch；`scales` 是独立 symmetric tensor。

## 4. Local Reduce 的 Launcher

原代码：

```python
@functools.cache
def compile_stage2_local_reduce():
    @flyc.kernel(
        name="comm_fused_moe_local",
        known_block_size=[BLOCK, 1, 1],
    )
    def kernel(
        route: fx.Pointer,
        partial: fx.Pointer,
        shared: fx.Pointer,
    ):
        _emit_local_worker(
            route,
            partial,
            shared,
            fx.Int32(gpu.block_idx.x),
        )

    @flyc.jit
    def launch(route, partial, shared, stream):
        kernel(route, partial, shared).launch(
            grid=(LOCAL_WORKERS, 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    return launch
```

这里启动：

```text
640 CTAs
× 256 threads/CTA
```

`gpu.block_idx.x` 是 local worker id。真正的 token/tile 分工在
`_emit_local_worker()` 里完成。

## 5. Local Worker 如何遍历 token 和 column tile

原代码：

```python
def _emit_local_worker(route, partial, shared, worker):
    work = scf.ForOp(
        arith.index_cast(T.index, worker),
        arith.constant(M * LOCAL_COLUMN_TILES, index=True),
        arith.constant(LOCAL_WORKERS, index=True),
    )
    with ir.InsertionPoint(work.body):
        item = arith.index_cast(T.i32, work.induction_variable)
        token = item // fx.Int32(LOCAL_COLUMN_TILES)
        tile = item - token * fx.Int32(LOCAL_COLUMN_TILES)
        _emit_local_reduce(
            route,
            partial,
            shared,
            token,
            tile * fx.Int32(BLOCK * VECTOR_WIDTH),
        )
        scf.YieldOp([])
```

逻辑 work item 总数是：

```text
M * LOCAL_COLUMN_TILES
= 2048 * 4
= 8192
```

但只启动 640 个 CTA。第 `worker` 个 CTA 处理：

```text
item = worker
item = worker + 640
item = worker + 1280
...
```

每个 `item` 解码成：

```text
token = item / 4
tile  = item % 4
```

四个 tile 对应：

```text
tile 0: columns    0..2047
tile 1: columns 2048..4095
tile 2: columns 4096..6143
tile 3: columns 6144..7167
```

最后一个 tile 只有 1024 列有效，后半 CTA 线程由后面的 `column < H` 屏蔽。

## 6. Local Reduce 中的线程到 column 映射

原代码：

```python
tid = fx.Int32(gpu.thread_idx.x)
column = tid * fx.Int32(VECTOR_WIDTH) + column_base
active = scf.IfOp(
    arith.cmpi(CmpIPredicate.ult, column, fx.Int32(H))
)
```

`VECTOR_WIDTH=8`，所以一个线程负责 8 个连续 hidden values：

```text
thread 0 -> column_base + 0..7
thread 1 -> column_base + 8..15
thread 2 -> column_base + 16..23
...
thread 255 -> column_base + 2040..2047
```

## 7. 定位当前 token 的六条 route

原代码：

```python
route_row_bytes = H + H // 8
route_addr = (
    fx.Int64(ptrtoint(route))
    + fx.Int64(token) * fx.Int64(TOPK * route_row_bytes)
)
route_row = buffer_ops.create_buffer_resource_from_addr(
    route_addr,
    num_records_bytes=TOPK * route_row_bytes,
)
```

`route_addr` 跳过之前所有 token，直接指向当前 token 的 route 0：

```text
route base
+ token * 6 * route_row_bytes
```

`route_row` 的 resource 范围包含当前 token 的全部六条 route。

## 8. 解码并累加六条 route

原代码：

```python
acc = fx.Vector.filled(VECTOR_WIDTH, 0.0, fx.Float32)
for route_slot in range_constexpr(TOPK):
    words = load_fp8_words(
        route_row,
        fx.Int32(route_slot * (route_row_bytes // 4))
        + column // fx.Int32(4),
        word_count=2,
        load_width=2,
        cache_modifier=2,
    )
    values = decode_fp8_f32(words)
    scale = load_e8m0_scale(
        route_row,
        fx.Int32(route_slot * route_row_bytes + H)
        + column // fx.Int32(8),
        2,
    )
    acc = acc + fx.Vector.from_elements(
        [value * scale for value in values],
        fx.Float32,
    )
```

### 8.1 FP8 payload 地址

`load_fp8_words()` 按 `i32` 读取，因此：

```text
route_slot * (route_row_bytes / 4)
```

是当前 route 在 `i32` 索引中的起点，`column / 4` 是当前 8 个 FP8
元素的起点。

```text
word_count=2
2 * int32 = 8 bytes = 8 FP8 values
```

### 8.2 E8M0 scale 地址

```text
route_slot * route_row_bytes
+ H
+ column / 8
```

其中 `+H` 跳过当前 route 的 FP8 payload，`column/8` 选择当前 8-value
group 的 scale。

### 8.3 这段代码的数学语义

```text
acc[0:8] = 0

for route_slot in 0..5:
    acc += decode_fp8(route[token, route_slot, column:column+8])
```

Stage2 GEMM 配置了 `doweight_stage2=True`，所以 route 已经包含 routing
weight，这里不再乘一次 weight。

## 9. 读取并累加 shared partial

原代码：

```python
shared_addr = (
    fx.Int64(ptrtoint(shared))
    + fx.Int64(token) * fx.Int64(H * 2)
)
shared_row = buffer_ops.create_buffer_resource_from_addr(
    shared_addr,
    num_records_bytes=H * 2,
)
shared_values = fx.Vector(
    buffer_ops.buffer_load(
        shared_row,
        column,
        vec_width=VECTOR_WIDTH,
        dtype=T.bf16,
        cache_modifier=2,
    )
).extf(T.vec(VECTOR_WIDTH, T.f32))
acc = acc + shared_values
```

`H*2` 是一行 BF16 的字节数。该线程读取和 route 相同的 8 个 column，
将 BF16 扩展到 FP32 后加入 `acc`。

到这里，`acc` 的准确含义是：

```text
acc[0:8]
= sum(route[token, 0:6, column:column+8])
+ shared_partial[token, column:column+8]
```

## 10. 四个 lane 组成一个 group32 量化组

每个线程只持有 8 个 FP32，但 partial 要求 32 个值共用一个 E8M0 scale。

原代码：

```python
lane = tid & fx.Int32(63)
local_max = fx.Float32(1e-10).maximumf(
    fmath.absf(acc).reduce(ReductionOp.MAX)
)
max_bits = local_max.bitcast(fx.Int32)
for xor_lane in (1, 2):
    remote_bits = fx.rocdl.ds_bpermute(
        T.i32,
        (lane ^ fx.Int32(xor_lane)) * fx.Int32(4),
        max_bits,
    )
    local_max = local_max.maximumf(
        fx.Int32(remote_bits).bitcast(fx.Float32)
    )
    max_bits = local_max.bitcast(fx.Int32)
```

### 10.1 每个 lane 先算自己的 8-value max

```text
lane 0: max(abs(column  0.. 7))
lane 1: max(abs(column  8..15))
lane 2: max(abs(column 16..23))
lane 3: max(abs(column 24..31))
```

### 10.2 `xor 1` 和 `xor 2` 组成四 lane reduce

```text
xor 1: lane 0 <-> 1, lane 2 <-> 3
xor 2: lane 0/1 <-> 2/3
```

两轮后，连续四个 lane 都拥有同一个 32-value max。`lane=tid&63` 使该操作
严格限制在 AMD wave64 内。

## 11. 将 Local Reduce 结果量化为 MXFP8

原代码：

```python
e8m0, quant_scale = e8m0_scale(local_max)
packed_words = pack_fp8_words(acc, quant_scale, 2)
```

`pack_fp8_words(..., 2)` 将当前线程的 8 个 FP32 值打包为：

```text
2 * int32 = 8 FP8 values
```

四个 lane 合起来生成：

```text
32 FP8 values + 1 E8M0 scale
```

## 12. 写入 partial payload 和 scale

### 12.1 每个线程写 8 个 FP8

原代码：

```python
payload_addr = (
    fx.Int64(ptrtoint(partial))
    + fx.Int64(token) * fx.Int64(H)
)
payload_row = buffer_ops.create_buffer_resource_from_addr(
    payload_addr,
    num_records_bytes=H,
)
store_fp8_words(payload_row, column, packed_words, 2)
```

因为 FP8 每个元素 1 byte，所以一行 payload 正好是 `H` bytes。

### 12.2 每四个 lane 只有一个 lane 写 scale

原代码：

```python
scale_leader = scf.IfOp(
    arith.cmpi(
        CmpIPredicate.eq,
        lane & fx.Int32(3),
        fx.Int32(0),
    )
)
with ir.InsertionPoint(scale_leader.then_block):
    scale_addr = (
        fx.Int64(ptrtoint(partial))
        + fx.Int64(M * H)
        + fx.Int64(token) * fx.Int64(GROUPS_PER_ROW)
    )
    scale_row = buffer_ops.create_buffer_resource_from_addr(
        scale_addr,
        num_records_bytes=GROUPS_PER_ROW,
    )
    buffer_ops.buffer_store(
        e8m0.to(fx.Int8),
        scale_row,
        column // fx.Int32(32),
        offset_is_bytes=True,
    )
```

`lane&3==0` 选中每个四 lane group 的第一个 lane，避免四个 lane 重复写同一个
scale。

scale 区域位于完整 FP8 payload 之后：

```text
partial + M*H + token*(H/32) + column/32
```

Local Reduce 结束后，每个 rank 都有一份完整 `[M,H]` local partial。

## 13. Partial Barrier 的作用

Host 紧接着调用：

```python
_barrier(
    self.partial,
    self.partial_flat_base,
    self.partial_ready,
    stream,
)
```

这个 barrier 先 system-release 本 rank 的 partial epoch，再等待全部 TP8 rank 达到
相同 epoch，最后 system-acquire。

没有这一步，`tp_reduce_scatter` 可能在 peer 还没写完 partial 时就开始远程
load。

## 14. TP Reduce-Scatter 的 Launcher

原代码：

```python
@functools.cache
def compile_stage2_tp_reduce_scatter():
    @flyc.kernel(
        name="comm_fused_moe_tp_reduce_scatter",
        known_block_size=[BLOCK, 1, 1],
    )
    def kernel(
        flat_base: fx.Int64,
        output: fx.Pointer,
        payload: fx.Pointer,
        scales: fx.Pointer,
        rank: fx.Int32,
    ):
        emit_tp_reduce_scatter(
            flat_base,
            output,
            payload,
            scales,
            rank,
            fx.Int32(gpu.block_idx.x),
            tokens=M,
            output_width=H,
            payload_width=H,
            shard_rows=SHARD_ROWS,
            tp=TP,
            block=BLOCK,
            reduce_scatter_grid=REDUCE_SCATTER_GRID,
        )

    @flyc.jit
    def launch(flat_base, output, payload, scales, rank, stream):
        kernel(flat_base, output, payload, scales, rank).launch(
            grid=(REDUCE_SCATTER_GRID, 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    return launch
```

这里启动：

```text
128 CTAs * 256 threads = 32768 threads
```

每个线程的基本 work item 不再是 8 个元素，而是完整一个 32-value MXFP8
group。

## 15. Reduce-Scatter 的线程到 token/group 映射

原代码：

```python
groups_per_row = payload_width // 32
start = arith.index_cast(
    T.index,
    worker * fx.Int32(block) + fx.Int32(gpu.thread_idx.x),
)
loop = scf.ForOp(
    start,
    arith.constant(shard_rows * groups_per_row, index=True),
    arith.constant(reduce_scatter_grid * block, index=True),
)
with ir.InsertionPoint(loop.body):
    pack = arith.index_cast(T.i32, loop.induction_variable)
    local_token = pack // fx.Int32(groups_per_row)
    group = pack - local_token * fx.Int32(groups_per_row)
    column = group * fx.Int32(32)
    global_token = rank * fx.Int32(shard_rows) + local_token
```

当前：

```text
groups_per_row = 7168 / 32 = 224
shard_rows * groups_per_row = 256 * 224 = 57344 work items
```

全网格只有 32768 个线程，所以每个线程通过 stride=32768 的循环处理一个或
两个 group。

work item 映射是：

```text
pack
  -> local_token in [0, 255]
  -> group in [0, 223]
  -> column = group * 32
  -> global_token = rank * 256 + local_token
```

例如 rank 3 的 `local_token=10` 对应：

```text
global_token = 3 * 256 + 10 = 778
```

所以每个 rank 只 reduce 自己负责的 256 个 token rows，这就是 scatter 部分。

## 16. 对一个 group32 读取全部 TP rank

原代码：

```python
acc = fx.Vector.filled(32, 0.0, fx.Float32)

for source_round in range_constexpr(tp):
    source = (
        rank + local_token + fx.Int32(source_round)
    ) % fx.Int32(tp)
    base = peer_base(flat_base, source)
```

`source_round=0..7` 保证所有 TP rank 都被访问一次。

```python
source = (rank + local_token + source_round) % TP
```

只是对 peer 读取顺序做轮转，不改变求和结果。不同 token 的第一个 source 可能不同。

`peer_base()` 的计算是：

```text
peer partial base
= canonical flat base
+ source rank * FLAT_VA_RANK_STRIDE
```

## 17. 读取 peer partial payload 和 scale

原代码：

```python
source_row = buffer_ops.create_buffer_resource_from_addr(
    base + fx.Int64(global_token) * fx.Int64(payload_width),
    num_records_bytes=payload_width,
)
words = load_fp8_words(
    source_row,
    column // fx.Int32(4),
    word_count=8,
    load_width=4,
    cache_modifier=2,
)
```

`word_count=8` 表示：

```text
8 * int32 = 32 bytes = 32 FP8 values
```

scale 地址：

```python
scale_row = buffer_ops.create_buffer_resource_from_addr(
    base
    + fx.Int64(tokens * payload_width)
    + fx.Int64(global_token) * fx.Int64(groups_per_row),
    num_records_bytes=groups_per_row,
)
scale = load_e8m0_scale(scale_row, group, 2)
```

地址分解：

```text
base
+ M*H                         # 跳过全部 partial FP8 payload
+ global_token*(H/32)         # 跳到当前 token 的 scale row
+ group                       # 当前 32-value group 的 scale
```

## 18. 解码并跨 TP8 累加

原代码：

```python
values = decode_fp8_f32(words)
acc = acc + fx.Vector.from_elements(
    [value * scale for value in values],
    fx.Float32,
)
```

该代码在八轮 source loop 中重复，所以最终：

```text
acc[0:32]
= partial_rank0[global_token, column:column+32]
+ partial_rank1[global_token, column:column+32]
+ ...
+ partial_rank7[global_token, column:column+32]
```

这是 TP reduce 本体。没有跨线程 atomic，因为一个 `(global_token, group)` 完全由一个
线程负责，该线程自己遍历全部 8 个 rank。

## 19. 将 TP 求和结果重新量化

原代码：

```python
e8m0, packed = quantize_group32(acc)
```

`quantize_group32()` 的代码：

```python
def quantize_group32(acc):
    local_max = fx.Float32(1e-10).maximumf(
        fmath.absf(acc).reduce(ReductionOp.MAX)
    )
    e8m0, scale = e8m0_scale(local_max)
    return e8m0, pack_fp8_words(acc, scale, 8)
```

这次一个线程自己已经持有完整 32-value vector，所以不需要 Local Reduce 里的
`ds_bpermute` 四 lane 归约。

输出是：

```text
8 * int32 packed words = 32 FP8 values
1 * E8M0 scale
```

## 20. 写 reduced payload 和 scale

原代码：

```python
payload_row = buffer_ops.create_buffer_resource_from_addr(
    fx.Int64(ptrtoint(payload))
    + fx.Int64(local_token) * fx.Int64(payload_width),
    num_records_bytes=payload_width,
)
store_fp8_words(payload_row, column, packed, 4)

scale_row = buffer_ops.create_buffer_resource_from_addr(
    fx.Int64(ptrtoint(scales))
    + fx.Int64(local_token) * fx.Int64(groups_per_row),
    num_records_bytes=groups_per_row,
)
buffer_ops.buffer_store(
    e8m0.to(fx.Int8),
    scale_row,
    group,
    offset_is_bytes=True,
)
```

这里使用 `local_token`，不是 `global_token`，因为每个 rank 的 reduced workspace 只保存
自己的 256 行 shard：

```text
rank-local reduced payload row 0
<-> global token rank*256 + 0
```

## 21. 为什么本 rank BF16 也从 MXFP8 解码

原代码：

```python
decoded = decode_group32(e8m0, packed)
output_row = buffer_ops.create_buffer_resource_from_addr(
    fx.Int64(ptrtoint(output))
    + fx.Int64(local_token) * fx.Int64(output_width * 2),
    num_records_bytes=output_width * 2,
)
_store_bf16_group32(output_row, column, decoded)
```

这里没有直接将 FP32 `acc` 转 BF16，而是：

```text
FP32 TP sum
    -> quantize to MXFP8
    -> decode the same MXFP8 to BF16
    -> write local reduced shard
```

后续其他 rank 在 all-gather 中也会解码同一份 MXFP8 payload。因此本 rank 和其他
rank 看到的 BF16 结果来自完全相同的量化数据。

如果本 rank 直接将 FP32 `acc` 转 BF16，而其他 rank 从 MXFP8 解码，同一个 token row
在不同 rank 上可能不再 bitwise 一致。

## 22. Reduced-Payload Barrier 与 All-Gather 的衔接

Reduce-scatter 完成后，host 执行：

```python
_barrier(
    self.reduced_payload,
    self.reduced_payload_base,
    self.reduced_ready,
    stream,
)
```

这个 barrier 的 epoch 存在 `reduced_payload` 尾部。`reduced_scale` 没有独立 epoch，
因为 payload 和 scale 由同一个 reduce-scatter kernel 写入，barrier 的 system
release/acquire 一起保证两者可见。

All-gather 按 remote rank 读取：

```python
payload = create_resource(peer_base(payload_base, source))
scales = create_resource(peer_base(scale_base, source))

words = load_fp8_words(payload, ...)
scale = buffer_load(scales, ...)
values = decode_group32(scale, words)
store_bf16(output, values)
```

因此 reduce-scatter 里写入的 `payload/scales` 是 all-gather 的通信格式，不是临时无用
中间结果。

## 23. 完整 Reduce 的等价伪代码

### 23.1 Local Reduce

```python
for token in range(M):
    for column_group8 in range(H // 8):
        acc8 = fp32_zeros(8)

        for route_slot in range(TOPK):
            acc8 += decode_route_fp8(
                route[token, route_slot, column_group8]
            )

        acc8 += bf16_to_fp32(
            shared[token, column_group8]
        )

        group32_scale = reduce_max_across_four_lanes(acc8)
        partial[token, column_group8] = quantize_fp8(
            acc8,
            group32_scale,
        )
```

### 23.2 TP Reduce-Scatter

```python
shard_begin = rank * SHARD_ROWS

for local_token in range(SHARD_ROWS):
    global_token = shard_begin + local_token

    for group in range(H // 32):
        acc32 = fp32_zeros(32)

        for source_rank in rotated_all_tp_ranks:
            acc32 += decode_partial_mxfp8(
                partial[source_rank][global_token, group]
            )

        e8m0, packed32 = quantize_group32(acc32)

        reduced_payload[local_token, group] = packed32
        reduced_scale[local_token, group] = e8m0
        reduced_shard_bf16[local_token, group] = decode_group32(
            e8m0,
            packed32,
        )
```

## 24. 两个 Reduce Kernel 的运算结构对比

| 属性 | Local Reduce | TP Reduce-Scatter |
| --- | --- | --- |
| 数据范围 | 本 rank | TP8 所有 rank |
| 归约维度 | TOPK=6 route + shared | TP=8 local partial |
| 单线程持有 | 8 FP32 values | 32 FP32 values |
| group32 max | 4 lanes 通过 `ds_bpermute` 合作 | 单线程 vector reduce |
| CTA/grid | 640 CTAs × 256 threads | 128 CTAs × 256 threads |
| 输入类型 | route FP8 + shared BF16 | remote partial MXFP8 |
| 累加类型 | FP32 | FP32 |
| 输出 | 完整 `[M,H]` local MXFP8 partial | `[M/TP,H]` MXFP8 + BF16 shard |
| 通信 | 无 | flat-VA remote load |

Local Reduce 是本 rank 的 route/shared 合并和通信压缩；TP Reduce-Scatter 才是真正
的跨 rank TP 求和。
