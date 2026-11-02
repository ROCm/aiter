"""Device-buffer accounting of an engine: walk every CUDA tensor it holds, dedup by
storage, and report the total split into weights / buffers (used by _ep_bench.py when
EP_MEMACCT=1, after the bench forwards so lazily allocated workspaces are included)."""

import re

import torch

_WEIGHT = re.compile(r"(^|[._\[])(w1|w2|w1s|w2s)([._\]]|_scale|$)|weight")


def walk(root):
    seen, stor = set(), {}

    def visit(o, path, depth):
        if depth > 8 or id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, torch.Tensor):
            if o.device.type != "cuda":
                return
            s = o.untyped_storage()
            key = s.data_ptr()
            if key not in stor or s.nbytes() > stor[key][0]:
                stor[key] = (s.nbytes(), path)
            return
        if isinstance(o, dict):
            for k, v in o.items():
                visit(v, f"{path}[{k}]", depth + 1)
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                visit(v, f"{path}[{i}]", depth + 1)
        elif hasattr(o, "__dict__") and type(o).__module__.split(".")[0] in ("aiter", "mori"):
            for k, v in vars(o).items():
                visit(v, f"{path}.{k}", depth + 1)

    visit(root, "eng", 0)
    return stor


def report(tag, root, top=14):
    stor = walk(root)
    w = sum(n for n, p in stor.values() if _WEIGHT.search(p.split(".")[-1]))
    tot = sum(n for n, _ in stor.values())
    print(f"[MEM {tag}] buffers={(tot - w) / 2**30:.3f} GiB weights={w / 2**30:.3f} GiB "
          f"torch_alloc={torch.cuda.memory_allocated() / 2**30:.3f} GiB", flush=True)
    for n, p in sorted(stor.values(), reverse=True)[:top]:
        print(f"[MEM {tag}]   {n / 2**20:10.1f} MiB  {p}", flush=True)
