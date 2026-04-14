# Custom Communication Group User Guide

## Overview

Custom communication groups allow users to define arbitrary GPU groupings for collective operations (e.g., allreduce), instead of relying on the default TP/DP/PP/EP layout. This is useful when different stages of a model require different parallelism configurations — for example, attention computed with tp4dp2 while MLP communication uses tp8.

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
from aiter.dist.communication_op import custom_all_reduce, custom_all_gather
```

### `CustomGroupConfig` Class

A builder for constructing the config dict passed to `ensure_model_parallel_initialized`.

| Method | Description |
|--------|-------------|
| `__init__()` | Create an empty config |
| `add_group(name, tp_group=, dp_group=)` | Add a named group with TP and/or DP rank lists |
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

> **Note:** For multi-group functionality, only `allreduce` and `allgather` operations are currently supported.

## Config Dict Format

Whether built via `CustomGroupConfig` or written manually, the config dict has the following structure:

```python
{
    "group_name": {
        "tp_group": <List[int] or List[List[int]]>,
        "dp_group": <List[int] or List[List[int]]>   # optional, can be omitted or []
    },
    ...
}
```

- `tp_group` (1D list): all ranks form a single TP group, e.g. `[0,1,2,3,4,5,6,7]`
- `tp_group` (2D list): multiple TP subgroups, e.g. `[[0,1,2,3],[4,5,6,7]]`
- `dp_group` (1D list): all ranks form a single DP group
- `dp_group` (2D list): multiple DP subgroups
- When both `tp_group` and `dp_group` are provided (both 2D), EP groups are derived automatically

## Usage Examples

### Example 1: Single Custom TP Group (tp4 on GPUs [0,2,4,6])

```python
config = {"default": {"tp_group": [0, 1, 2, 3]}}

ensure_model_parallel_initialized(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=1,
    custom_group_config=config,
)

# use
out = custom_all_reduce(x)  # group name can be omitted for single group
```

### Example 2: Custom TP+DP (tp4dp2, derives EP group)

```python
config = {
    "default": {
        "tp_group": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "dp_group": [[0, 4], [1, 5], [2, 6], [3, 7]],
    }
}

ensure_model_parallel_initialized(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=1,
    custom_group_config=config,
)

out = custom_all_reduce(x)
```

### Example 3: Multi-Group (attention tp4dp2 + communication tp8)

Using `CustomGroupConfig` builder:

```python
config = CustomGroupConfig()
config.add_group(
    "attn",
    tp_group=[[0, 1, 2, 3], [4, 5, 6, 7]],
    dp_group=[[0, 4], [1, 5], [2, 6], [3, 7]],
)
config.add_group(
    "comm",
    tp_group=[0, 1, 2, 3, 4, 5, 6, 7],
)

ensure_model_parallel_initialized(
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=1,
    custom_group_config=config.data(),
)

# Phase 1: attention — allreduce on the "attn" group
out_attn = custom_all_reduce(x, group="attn")

# Phase 2: communication — allreduce on the "comm" group
out_comm = custom_all_reduce(out_attn, group="comm")
```

Or equivalently, using a raw dict:

```python
config = {
    "attn": {
        "tp_group": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "dp_group": [[0, 4], [1, 5], [2, 6], [3, 7]],
    },
    "comm": {
        "tp_group": [0, 1, 2, 3, 4, 5, 6, 7],
    },
}

ensure_model_parallel_initialized(
    tensor_model_parallel_size=8,
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
attn_group = get_custom_group("attn")
comm_group = get_custom_group("comm")

# Or get the full dict
all_groups = get_custom_group()  # returns {"attn": ..., "comm": ...}
```

### Example 5: CUDA Graph Capture with Custom Groups

```python
attn_group = get_custom_group("attn")

graph = torch.cuda.CUDAGraph()
with attn_group.graph_capture() as gc:
    with torch.cuda.graph(graph, stream=gc.stream):
        out = custom_all_reduce(x, group="attn")
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

### Validation Rules

The following checks are enforced during initialization:

1. **Consistent GPU count**: All groups in the config must use the same total number of GPUs (equal to `world_size`).
2. **No duplicate group names**: Each group name must be unique within the config. `CustomGroupConfig.add_group()` enforces this automatically.
3. **Rank coverage**: Every rank `0..world_size-1` must appear exactly once in `tp_group` (and `dp_group` if provided). No duplicates, no missing ranks.
4. **TP/DP grid consistency** (when both provided): Ranks within the same TP subgroup must be in different DP subgroups — i.e., TP and DP groupings must form a valid grid.
5. **At least one of `tp_group` or `dp_group`** must be provided for each group entry.

### Initialization and Lifecycle

```python
# 1. Initialize distributed environment
set_custom_all_reduce(True)
init_distributed_environment(world_size=8, rank=rank_id, ...)

# 2. Initialize model parallel with custom groups
ensure_model_parallel_initialized(tp_size, pp_size, custom_group_config=config)

# 3. Use custom_all_reduce() during training/inference
out = custom_all_reduce(x, group="attn")

# 4. Cleanup
destroy_model_parallel()
destroy_distributed_environment()
```

- All groups are created during `ensure_model_parallel_initialized` and persist until `destroy_model_parallel`.
- No need to destroy/reinitialize between phases — switch groups by passing different `group=` names.
