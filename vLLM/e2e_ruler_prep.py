#!/usr/bin/env python3
"""Generate the RULER prompts of an `e2e.sh eval` run once, for every arm.

    e2e_ruler_prep.py <dir> <tokenizer> <tasks> <lengths> <seed>

Runs in an image with lm-eval[ruler] (/opt/lm-eval/bin/python), on the CPU.
lm-eval's RULER tasks build their 500 prompts per task and length at load time,
tokenizing every haystack while it is sized: minutes per lm-eval run at 64k,
redone for every arm while its server holds the GPUs. Here each (task, length)
is generated once, in parallel, with lm-eval's own generator and seeded on its
own (so it does not depend on which other tasks run), into
<dir>/<length>/<task>.jsonl. Next to it goes <task>.yaml: the built-in task with
the dataset read from that file, under the same name, so that
`lm_eval --include_path <dir>/<length>` runs it in place of the built-in one.
--limit then keeps the first prompts, as it would of the generated ones. The
yaml points at the file as <dir> is seen here; mount it at the same path in the
server's container. A (task, length) whose files exist is not regenerated.
"""

import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def builtin_yaml(task):
    """Path of a built-in lm-eval task's yaml."""
    from lm_eval.tasks import TaskManager

    entry = TaskManager().task_index.get(task)
    if entry is None or not entry.yaml_path:
        sys.exit("%s is not an lm-eval task" % task)
    return str(entry.yaml_path)


def generate(task, yaml_path, length, tokenizer, seed, out):
    import numpy as np
    from lm_eval.tasks._yaml_loader import load_yaml

    make = load_yaml(yaml_path).get("custom_dataset")
    if not callable(make):
        sys.exit("%s builds no dataset of its own: not a RULER task" % task)
    random.seed(seed)
    np.random.seed(seed)
    ds = make(max_seq_lengths=[length], tokenizer=tokenizer)["test"]
    tmp = out.with_suffix(".tmp")
    ds.to_json(tmp, lines=True, force_ascii=False)
    os.replace(tmp, out)
    return "%s at %d: %d prompts" % (task, length, len(ds))


def main():
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    root, tokenizer = Path(sys.argv[1]), sys.argv[2]
    tasks = sys.argv[3].split(",")
    lengths = [int(n) for n in sys.argv[4].split()]
    seed = int(sys.argv[5])

    jobs = {}
    for task in tasks:
        yaml_path = builtin_yaml(task)
        for length in lengths:
            d = root / str(length)
            d.mkdir(parents=True, exist_ok=True)
            data = d / ("%s.jsonl" % task)
            # JSON is valid YAML; the include's own keys are overridden.
            (d / ("%s.yaml" % task)).write_text(json.dumps({
                "include": yaml_path,
                "task": task,
                "custom_dataset": None,
                "dataset_path": "json",
                "dataset_name": None,
                "dataset_kwargs": {"data_files": {"test": str(data)}},
            }, indent=2) + "\n")
            if data.exists():
                print("%s at %d: cached" % (task, length), flush=True)
            else:
                jobs[(task, length)] = (task, yaml_path, length, tokenizer, seed, data)
    with ProcessPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        for done in [pool.submit(generate, *j) for j in jobs.values()]:
            print(done.result(), flush=True)


if __name__ == "__main__":
    main()
