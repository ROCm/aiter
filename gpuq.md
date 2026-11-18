---
name: gpuq
description: "Launch GPU work, including a kernel, through the gpuq client inside a container. The host daemon is already running. Set GPUQ_USER to the job owner and run gpuq --num-gpus=1 <command>. One GPU unless they ask for more via --num-gpus=N."
user-invocable: false
---

# gpuq client

The host daemon is already running, with eviction on. Do not install or restart it. Reaching it from a container is the `gpuq-container` skill.

A job that skips `gpuq` can be SIGTERMed while eviction is on.

Set `GPUQ_USER` to the person who owns the job. One GPU unless they ask for more. `<command>` is whatever runs the kernel: a Python bench, or a binary.

```bash
export GPUQ_USER="<owner>"
gpuq --num-gpus=1 <command>
```

More GPUs: `gpuq --num-gpus=N <command>`. `docker exec` sees `GPUQ_USER` when the container was created with it, or when that exec passes `-e GPUQ_USER=<owner>`.
