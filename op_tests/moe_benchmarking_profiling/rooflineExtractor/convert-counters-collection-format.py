#!/usr/bin/env python3
import os
import pandas as pd
import argparse
import logging


def read_csv(file_path):
    df = pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.info(f"Error reading {file_path}: {e}")
        raise
    return df


def get_counter_collection_files(root_path):
    file_paths = []
    for root, _, files in os.walk(root_path):
        if "pmc_" in root:
            for file in files:
                if file.endswith("counter_collection.csv"):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
    return file_paths


def get_combined_df(args):
    files_list = []
    for input in args.input:
        if os.path.isfile(input):
            files_list.append(input)
        elif os.path.isdir(input):
            files_list.extend(get_counter_collection_files(input))
    if not files_list:
        raise ValueError("Valid Input files not found")
    logging.info(f"Processing files: {files_list}")
    combined_df = pd.DataFrame()
    for file in files_list:
        combined_df = pd.concat([combined_df, read_csv(file)], ignore_index=True)
    return combined_df


def write_to_file(df, args):
    logging.info(f"Saving output file to : {args.output}")
    directory, file_path = os.path.split(args.output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_csv(args.output, index=False)


def main(args):
    logging.basicConfig(level=args.loglevel)
    input_df = get_combined_df(args)
    # Validate
    columns = [
        "Correlation_Id",
        "Dispatch_Id",
        "Agent_Id",
        "Queue_Id",
        "Process_Id",
        "Thread_Id",
        "Grid_Size",
        "Kernel_Id",
        "Kernel_Name",
        "Workgroup_Size",
        "LDS_Block_Size",
        "Scratch_Size",
        "VGPR_Count",
        "SGPR_Count",
        "Counter_Name",
        "Counter_Value",
        "Start_Timestamp",
        "End_Timestamp",
    ]
    for col in input_df.columns:
        if col not in columns:
            logging.debug(f"Unexpected column {col} found in rocprofv3 input file")

    non_index_columns = [
        "Correlation_Id",
        "Start_Timestamp",
        "End_Timestamp",
        "Process_Id",
        "Thread_Id",
        "Kernel_Id",
    ]

    # Convert
    indexes = [
        "Dispatch_Id",
        "Agent_Id",
        "Grid_Size",
        "Kernel_Name",
        "LDS_Block_Size",
        "Queue_Id",
        "SGPR_Count",
        "Scratch_Size",
        "VGPR_Count",
        "Workgroup_Size",
    ]

    # Drop duplicate counters in multiple PMC lines
    input_df.drop_duplicates(
        subset=indexes + ["Counter_Name"], keep="first", inplace=True
    )

    pivoted_data = input_df.pivot_table(
        index=indexes, columns="Counter_Name", values="Counter_Value", aggfunc="sum"
    ).reset_index()

    # Save
    write_to_file(pivoted_data, args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Rocprofv3 Counter Collection input files and/or directories containing `*counter_collection.csv` files",
        nargs="+",
        default=[],
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Rocprofv1 formatted output file name",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Debug Logs",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose Logs",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
