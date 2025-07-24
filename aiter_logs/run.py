# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import json
import argparse
import os
import csv
import sys
import gzip


def get_num_gpu():
    return 8


def load_json_files(directory):
    json_data = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith("json"):
                with open(directory + file, "r") as f:
                    data = json.load(f)
                    json_data.setdefault("traceEvents", []).append(data["traceEvents"])
            elif file.endswith("json.gz"):
                with gzip.open(directory + file, "rt") as f:
                    data = json.load(f)
                    json_data.setdefault("traceEvents", []).append(data["traceEvents"])
    return json_data


def parse(json_file_path, output_file_name, function_name):

    data_all = load_json_files(json_file_path)
    data = data_all["traceEvents"]

    kernels = []
    found = False
    for entries in data:
        for entry in entries:
            # if 'dur' in entry:
            # kernels.append(entry)
            # found = True
            if (
                "name" in entry
                and "cat" in entry
                and (entry["cat"] in ["kernel", "gpu_memcpy", "gpu_user_annotation"])
            ):
                if function_name == "all":
                    kernels.append(entry)
                    found = True
                elif function_name in entry["name"]:
                    kernels.append(entry)
                    found = True
            elif "cat" in entry and (
                entry["cat"]
                not in [
                    "ac2g",
                    "cuda_runtime",
                    "python_function",
                    "cpu_instant_event",
                    "cpu_op",
                    "user_annotation",
                    "Trace",
                ]
            ):
                print(entry["cat"])
    if not found:
        print("There is no " + function_name + " in this log")
        return

    sorted_kernels = sorted(kernels, key=lambda x: (x["ts"], x["pid"]))

    csv_file_name = output_file_name + ".csv"
    json_file_out = output_file_name + ".json"

    json_data_out = {}
    json_data_out.setdefault("traceEvents", []).append({})

    with open(csv_file_name, "w", newline="") as csvfile:
        fieldnames = [
            "pid",
            "dur",
            "ts",
            "min_dur",
            "max_dur",
            "min_start",
            "max_start",
            "latency_before_first_gpu",
            "max_dur - min_dur",
            "duration_from_last_arrival",
            "first_gpu",
            "last_gpu",
            "shortest_gpu",
            "longest_gpu",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        i = 1
        min_dur = sys.float_info.max
        max_dur = sys.float_info.min
        min_start = sys.float_info.max
        max_start = sys.float_info.min
        first_gpu = 0
        last_gpu = 0
        longest_gpu = 0
        shortest_gpu = 0
        for entry in sorted_kernels:
            record = {"pid": entry["pid"], "dur": entry["dur"], "ts": entry["ts"]}
            json_data_out.setdefault("traceEvents", []).append(entry)
            if entry["dur"] < min_dur:
                min_dur = min(min_dur, entry["dur"])
                shortest_gpu = entry["pid"]

            if entry["dur"] > max_dur:
                max_dur = max(max_dur, entry["dur"])
                longest_gpu = entry["pid"]

            if entry["ts"] < min_start:
                min_start = min(min_start, entry["ts"])
                first_gpu = entry["pid"]

            if entry["ts"] > max_start:
                max_start = max(max_start, entry["ts"])
                duration_from_last_arrival = entry["dur"]
                last_gpu = entry["pid"]

            writer.writerow(record)
            if (i) % get_num_gpu() == 0:
                record = {
                    "min_dur": min_dur,
                    "max_dur": max_dur,
                    "min_start": min_start,
                    "max_start": max_start,
                    "latency_before_first_gpu": max_start - min_start,
                    "max_dur - min_dur": max_dur - min_dur,
                    "duration_from_last_arrival": duration_from_last_arrival,
                    "first_gpu": first_gpu,
                    "last_gpu": last_gpu,
                    "shortest_gpu": shortest_gpu,
                    "longest_gpu": longest_gpu,
                }
                writer.writerow(record)
                csvfile.write("\n")
                min_dur = sys.float_info.max
                max_dur = sys.float_info.min
                min_start = sys.float_info.max
                max_start = sys.float_info.min
                first_gpu = 0
                last_gpu = 0
                longest_gpu = 0
                shortest_gpu = 0

                i = 0
            i = i + 1

        with open(json_file_out, "w") as jsonfileout:
            json.dump(json_data_out, jsonfileout, indent=4)

    print(f"Data successfully written to {csv_file_name} and {json_file_out}.")


def main():
    parser = argparse.ArgumentParser(description="Json file and the function to parse.")

    parser.add_argument(
        "json_file_path",
        metavar="file_path",
        type=str,
        help="Path to the JSON file to process",
    )
    parser.add_argument("output_file_name", type=str, help="Output File Name")
    parser.add_argument(
        "function_name",
        type=str,
        help="Kernel Function Name, e.g., oneShotAllReduce, ncclDevKernel_Generic, mscclKernel",
    )

    args = parser.parse_args()
    parse(args.json_file_path, args.output_file_name, args.function_name)


if __name__ == "__main__":
    main()
