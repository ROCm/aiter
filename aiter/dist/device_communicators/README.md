# Custom Communication Group User Guide

## Overview

Custom communication groups allow users to define arbitrary GPU groupings for collective operations (e.g., allreduce, allgather, reduce_scatter), instead of relying on the default TP/DP/PP/EP layout. This is useful when different stages of a model require different parallelism configurations — for example, attention computed with TP4 (two independent groups) while MLP communication uses TP8.

The system supports **single group** and **multi-group** modes. Multiple groups can be initialized upfront and selected by name at runtime, avoiding expensive destroy/reinit during inference.

## API Reference

### Key Imports

```python
from aiter.dist.parallel_state import (
    CustomGroupConfig,
    ensure_model_parallel_initialized,
    get_custom_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.communication_op import custom_all_reduce, custom_all_gather, custom_reduce_scatter
```

### `CustomGroupConfig` Class

A builder for constructing the config dict passed to `ensure_model_parallel_initialized`.

| Method | Description |
|--------|-------------|
| `__init__()` | Create an empty config |
| `add_group(name, ranks)` | Add a named group with a rank list (1D or 2D) |
| `data() -> dict` | Return the config dict for passing to init functions |

### `get_custom_group(name=None)`

Retrieve initialized `GroupCoordinator` instances.

| Scenario | Behavior |
|----------|----------|
| Single group, `name=None` | Returns the `GroupCoordinator` instance directly |
| Multiple groups, `name=None` | Returns a `dict` of all `GroupCoordinator` instances |
| Any, `name="xxx"` | Returns the `GroupCoordinator` for that specific group |

### `custom_all_reduce(input_, ..., group=None)`

Perform allreduce on a custom group. When only one group exists, `group` can be omitted. When multiple groups exist, pass the group name.

### `custom_all_gather(input_, ..., group=None)`

Perform allgather on a custom group. When only one group exists, `group` can be omitted. When multiple groups exist, pass the group name.

### `custom_reduce_scatter(input_, ..., group=None)`

Perform reduce-scatter on a custom group. When only one group exists, `group` can be omitted. When multiple groups exist, pass the group name.

## Config Dict Format

Whether built via `CustomGroupConfig` or written manually, the config dict has the following structure:

```python
{
    "group_name": <List[int] or List[List[int]]>,
    ...
}
```

Each value can be:
- **1D `List[int]`**: all ranks form a single communication group, e.g. `[0,1,2,3,4,5,6,7]` → one TP8 group
- **2D `List[List[int]]`**: multiple independent subgroups, e.g. `[[0,1,2,3],[4,5,6,7]]` → two independent TP4 groups that operate without interfering with each other

## Usage Examples

### Example 1: Single Custom Group (TP8)

```python
config = {"tp_group": [0, 1, 2, 3, 4, 5, 6, 7]}

ensure_model_parallel_initialized(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    custom_group_config=config,
)

# use
out = custom_all_reduce(x)  # group name can be omitted for single group
```

### Example 2: Two Independent TP4 Groups

```python
config = {"tp_group": [[0, 1, 2, 3], [4, 5, 6, 7]]}

ensure_model_parallel_initialized(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    custom_group_config=config,
)

# Devices 0-3 allreduce among themselves, devices 4-7 allreduce among themselves
out = custom_all_reduce(x)
```

### Example 3: Multi-Group (TP4x2 + DP2x4)

Using `CustomGroupConfig` builder:

```python
config = CustomGroupConfig()
config.add_group("tp_group", [[0, 1, 2, 3], [4, 5, 6, 7]])
config.add_group("dp_group", [[0, 4], [1, 5], [2, 6], [3, 7]])

ensure_model_parallel_initialized(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    custom_group_config=config.data(),
)

# Phase 1: TP allreduce within subgroups of 4
out_tp = custom_all_reduce(x, group="tp_group")

# Phase 2: DP allreduce within subgroups of 2
out_dp = custom_all_reduce(out_tp, group="dp_group")
```

Or equivalently, using a raw dict:

```python
config = {
    "tp_group": [[0, 1, 2, 3], [4, 5, 6, 7]],
    "dp_group": [[0, 4], [1, 5], [2, 6], [3, 7]],
}

ensure_model_parallel_initialized(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    custom_group_config=config,
)
```

### Example 4: Using `get_custom_group` Directly

```python
# Single group — returns GroupCoordinator directly
group = get_custom_group()
dist.all_reduce(tensor, group=group.device_group)

# Multiple groups — get by name
tp_group = get_custom_group("tp_group")
dp_group = get_custom_group("dp_group")

# Or get the full dict
all_groups = get_custom_group()  # returns {"tp_group": ..., "dp_group": ...}
```

### Example 5: CUDA Graph Capture with Custom Groups

```python
tp_group = get_custom_group("tp_group")

graph = torch.cuda.CUDAGraph()
with tp_group.graph_capture() as gc:
    with torch.cuda.graph(graph, stream=gc.stream):
        out = custom_all_reduce(x, group="tp_group")
```

## Important Notes

### Mutual Exclusion with Standard Interfaces

Custom groups and standard parallel group interfaces are **mutually exclusive**:

| `custom_group_config` | Standard ops (`tensor_model_parallel_all_reduce`, etc.) | Custom ops (`custom_all_reduce`) |
|---|---|---|
| `None` (not set) | Available | **AssertionError** |
| Set (has groups) | **AssertionError** | Available |

- When `custom_group_config` is **not set**, use `tensor_model_parallel_all_reduce()`, `data_parallel_all_reduce()`, and other standard interfaces. Calling `custom_all_reduce()` will raise an error.
- When `custom_group_config` **is set**, only `custom_all_reduce()` is available. Calling any standard interface (e.g., `tensor_model_parallel_all_reduce()`) will raise an error.

### Automatic Buffer Management

When `custom_group_config` is provided, the standard TP/PP/DP/EP groups **automatically skip** `CudaCommunicator` creation. This means:

- **No redundant memory allocation**: only the custom groups allocate communication buffers (~3 GB per group). Standard groups exist only as lightweight `GroupCoordinator` wrappers without GPU buffers.
- **`tp_size` / `dp_size` do not affect GPU memory**: you can pass any valid values (e.g., `tp_size=1` or `tp_size=8`) — the standard groups will never allocate expensive buffers when custom groups are present.
- **Recommended**: set `tensor_model_parallel_size=1` and `pipeline_model_parallel_size=1` when all communication is handled by custom groups. This is the simplest and most explicit configuration.

### Validation Rules

The following checks are enforced during initialization:

1. **Rank coverage**: Every rank `0..world_size-1` must appear exactly once across all subgroups. No duplicates, no missing ranks.
2. **Uniform subgroup size** (for 2D lists): All subgroups must have the same size.
3. **No duplicate group names**: Each group name must be unique within the config. `CustomGroupConfig.add_group()` enforces this automatically.

### Initialization and Lifecycle

```python
# 1. Initialize distributed environment
set_custom_all_reduce(True)
init_distributed_environment(world_size=8, rank=rank_id, ...)

# 2. Initialize model parallel with custom groups
# tp_size and pp_size can be set to 1 when all communication uses custom groups
ensure_model_parallel_initialized(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    custom_group_config=config,
)

# 3. Use custom_all_reduce() during training/inference
out = custom_all_reduce(x, group="tp_group")

# 4. Cleanup
destroy_model_parallel()
destroy_distributed_environment()
```

- All groups are created during `ensure_model_parallel_initialized` and persist until `destroy_model_parallel`.
- No need to destroy/reinitialize between phases — switch groups by passing different `group=` names.
