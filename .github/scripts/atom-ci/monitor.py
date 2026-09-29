#!/usr/bin/env python3
"""Record compiler progress without writing to the watchdog's progress logs."""

import json
import os
import sys
import time
from pathlib import Path

import psutil


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def snapshot():
    processes = []
    for proc in psutil.process_iter(
        ["pid", "ppid", "name", "status", "cpu_times", "memory_info"]
    ):
        try:
            info = proc.info
            if not any(
                name in info["name"]
                for name in (
                    "clang",
                    "hipcc",
                    "ninja",
                    "lld",
                    "python",
                    "cc1",
                    "lm_eval",
                )
            ):
                continue
            processes.append(
                {
                    "pid": info["pid"],
                    "ppid": info["ppid"],
                    "name": info["name"],
                    "status": info["status"],
                    "cpu_seconds": sum(info["cpu_times"][:2]),
                    "rss": info["memory_info"].rss,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            continue
    builds = {}
    for path in Path("/app/aiter-test/aiter/jit/build").glob("*/build/.ninja_log"):
        try:
            stat = path.stat()
            builds[str(path)] = {"bytes": stat.st_size, "mtime": stat.st_mtime}
        except OSError:
            continue
    return {
        "time": time.time(),
        "cpu_count": os.cpu_count(),
        "load": os.getloadavg(),
        "memory_available": psutil.virtual_memory().available,
        "processes": processes,
        "ninja_logs": builds,
        "cgroup": {
            name: read_optional("/sys/fs/cgroup/" + name)
            for name in (
                "cpu.max",
                "cpu.stat",
                "cpu.pressure",
                "memory.max",
                "memory.current",
                "memory.events",
                "memory.pressure",
                "cpuset.cpus.effective",
            )
        },
    }


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "resources.jsonl").open("a", buffering=1) as stream:
        while not (output / "stop-monitor").exists():
            stream.write(json.dumps(snapshot()) + "\n")
            time.sleep(10)


if __name__ == "__main__":
    main()
