import pdb
#!/usr/bin/env python3
"""
Script to run rocprofv3 with both counter collection and kernel tracing,
then analyze the results with rooflineExtractor.

Usage:
    python profile.py [-o OUTPUT_DIR] [--arch ARCH] [--proj-arch PROJ_ARCH] -- <run_command> [args...]
    
Example:
    python profile.py -- ./my_app arg1 arg2
    python profile.py -o ./results -- ./my_app -v --debug
    python profile.py --arch MI300X -- ./my_app
    python profile.py --arch MI300A --proj-arch MI300X -- ./my_app
    
Default output directory is ./data/output/
If --arch is not provided, the GPU architecture will be auto-detected.

Note:
    The script attempts to use the -f csv flag with rocprofv3. This flag is not
    recognized in versions of ROCm older than ROCm 7. If the flag is not recognized,
    the script will automatically retry without it.
"""

import sys
import subprocess
import os
import argparse
import re
import glob
import shutil
from pathlib import Path
from datetime import datetime


class LogWriter:
    """Helper class to write output to both console and log file."""
    def __init__(self, log_file_path):
        self.log_file = open(log_file_path, 'w')
        self.log_file_path = log_file_path
    
    def write(self, message, end='\n', file=None):
        """Write message to both console and log file."""
        # Write to console
        if file == sys.stderr:
            print(message, end=end, file=sys.stderr)
        else:
            print(message, end=end)
        
        # Write to log file
        self.log_file.write(message + end)
        self.log_file.flush()
    
    def close(self):
        """Close the log file."""
        self.log_file.close()


def run_rocprofv3_with_retry(cmd_with_csv, cmd_without_csv, cwd, logger=None):
    """
    Run rocprofv3 command with -f csv flag. If it fails due to unrecognized flag,
    retry without the -f csv flag.
    
    Args:
        cmd_with_csv: Command list with -f csv included
        cmd_without_csv: Command list without -f csv
        cwd: Working directory for the command
        logger: LogWriter instance for logging output
        
    Returns:
        subprocess.CompletedProcess result
    """
    def stream_output(cmd, cwd):
        """Helper to stream output in real-time and capture it for error checking."""
        stdout_lines = []
        stderr_lines = []
        
        # Start process with pipes for stdout and stderr
        process = subprocess.Popen(
            cmd, 
            cwd=cwd,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        # Use select or threads to handle both stdout and stderr
        import selectors
        import io
        
        sel = selectors.DefaultSelector()
        sel.register(process.stdout, selectors.EVENT_READ)
        sel.register(process.stderr, selectors.EVENT_READ)
        
        # Track which streams are still open
        streams_open = {process.stdout, process.stderr}
        
        while streams_open:
            # Wait for data to be available
            for key, _ in sel.select(timeout=0.1):
                stream = key.fileobj
                line = stream.readline()
                
                if line:
                    if stream == process.stdout:
                        stdout_lines.append(line)
                        if logger:
                            logger.write(line, end='')
                        else:
                            print(line, end='')
                        sys.stdout.flush()
                    else:  # stderr
                        stderr_lines.append(line)
                        if logger:
                            logger.write(line, end='', file=sys.stderr)
                        else:
                            print(line, end='', file=sys.stderr)
                        sys.stderr.flush()
                else:
                    # Stream closed
                    sel.unregister(stream)
                    streams_open.discard(stream)
        
        # Wait for process to finish
        returncode = process.wait()
        
        # Create a result object similar to subprocess.CompletedProcess
        class Result:
            def __init__(self, returncode, stdout, stderr):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr
        
        return Result(returncode, ''.join(stdout_lines), ''.join(stderr_lines))
    
    # Try with -f csv first - stream output in real-time
    result = stream_output(cmd_with_csv, cwd)
    
    # Check if the command failed due to unrecognized -f flag
    if result.returncode != 0 and result.stderr:
        stderr_lower = result.stderr.lower()
        # Check for common error messages indicating the flag is not recognized
        flag_not_recognized = any(phrase in stderr_lower for phrase in [
            'unrecognized option',
            'invalid option',
            'unknown option',
            'unrecognized argument',
            'invalid argument',
            'unknown argument',
            '-f',
            'usage:',
        ])
        
        if flag_not_recognized and '-f' in result.stderr:
            # Print retry message
            if logger:
                logger.write("Note: -f csv flag not recognized. Retrying with pre-ROCm 7 syntax...")
                logger.write(f"Retry command: {' '.join(cmd_without_csv)}")
            else:
                print(f"Note: -f csv flag not recognized. Retrying with pre-ROCm 7 syntax...")
                print(f"Retry command: {' '.join(cmd_without_csv)}")
            
            # Retry without -f csv - stream output in real-time
            result = stream_output(cmd_without_csv, cwd)
    
    return result


def detect_gpu():
    """
    Detect the AMD GPU model running on the system.
    Returns one of: 'MI250X', 'MI300A', 'MI300X', 'MI355X', or None if detection fails.
    """
    try:
        # Try using rocm-smi to get GPU information
        result = subprocess.run(['rocm-smi', '--showproductname'], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False)
        
        if result.returncode == 0:
            output = result.stdout.lower()
            
            # Check for specific GPU models
            if 'mi355x' in output or 'instinct mi355x' in output:
                return 'MI355X'
            elif 'mi300x' in output or 'instinct mi300x' in output:
                return 'MI300X'
            elif 'mi300a' in output or 'instinct mi300a' in output:
                return 'MI300A'
            elif 'mi250x' in output or 'instinct mi250x' in output:
                return 'MI250X'
            elif 'mi250' in output or 'instinct mi250' in output:
                return 'MI250X'
        
        # Try alternative method: check GPU device name from rocminfo
        result = subprocess.run(['rocminfo'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False)
        
        if result.returncode == 0:
            output = result.stdout.lower()
            
            # Look for marketing name or device name
            for line in output.split('\n'):
                if 'marketing name' in line or 'name:' in line:
                    if 'mi355x' in line:
                        return 'MI355X'
                    elif 'mi300x' in line:
                        return 'MI300X'
                    elif 'mi300a' in line:
                        return 'MI300A'
                    elif 'mi250x' in line or 'mi250' in line:
                        return 'MI250X'
        
        # Try checking gfx architecture ID
        result = subprocess.run(['rocminfo'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False)
        
        if result.returncode == 0:
            output = result.stdout.lower()
            
            # Match gfx architecture to GPU model
            if 'gfx950' in output:
                return 'MI355X'
            elif 'gfx942' in output:
                # gfx942 can be MI300A or MI300X, need to check further
                if 'mi300a' in output:
                    return 'MI300A'
                else:
                    return 'MI300X'
            elif 'gfx90a' in output:
                return 'MI250X'
    
    except FileNotFoundError:
        print("Warning: rocm-smi or rocminfo not found. Unable to detect GPU automatically.")
    except Exception as e:
        print(f"Warning: Error detecting GPU: {e}")
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Run rocprofv3 with counter collection and kernel tracing',
        usage='%(prog)s [-o OUTPUT_DIR] [--arch ARCH] [--proj-arch PROJ_ARCH] -- run_command [args...]'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='data/output',
        help='Directory for output files (default: ./data/output)'
    )
    parser.add_argument(
        '--arch',
        required=False,
        help='Supply architecture (to aid in guided analysis). Options: MI250X, MI300A, MI300X, MI355X. If not provided, will auto-detect.'
    )
    parser.add_argument(
        '--proj-arch',
        required=False,
        help='Supply second architecture for runtime projection. Options: MI250X, MI300A, MI300X, MI355X'
    )
    parser.add_argument(
        'run_command',
        nargs=argparse.REMAINDER,
        help='Command to profile and its arguments (use -- before command)'
    )
    
    args = parser.parse_args()
    output_dir = args.output_dir
    run_command = args.run_command
    
    # Prepend ./data/ to output_dir if it's not already there and not an absolute path
    if not os.path.isabs(output_dir) and not output_dir.startswith('data/') and not output_dir.startswith('./data/'):
        output_dir = os.path.join('data', output_dir)
    
    # Remove the '--' separator if present
    if run_command and run_command[0] == '--':
        run_command = run_command[1:]
    
    # Validate that a command was provided
    if not run_command:
        parser.error("No run command provided. Use: profile.py [-o OUTPUT_DIR] -- run_command [args...]")
    
    # Map GPU model to gfx architecture for counter file selection
    gpu_to_gfx = {
        'MI250X': 'gfx90a',
        'MI300A': 'gfx942',
        'MI300X': 'gfx942',
        'MI355X': 'gfx950'
    }
    
    # Use user-provided arch if available, otherwise auto-detect
    if args.arch:
        detected_gpu = args.arch.upper()
        gfx_arch = gpu_to_gfx.get(detected_gpu, 'gfx90a')
        gpu_message = f"Using user-specified architecture: {detected_gpu} ({gfx_arch})"
    else:
        # Detect GPU (before creating logger, so store messages)
        detected_gpu = detect_gpu()
        
        if detected_gpu:
            gfx_arch = gpu_to_gfx.get(detected_gpu, 'gfx90a')
            gpu_message = f"Detected GPU: {detected_gpu} ({gfx_arch})"
        else:
            gfx_arch = 'gfx90a'
            detected_gpu = 'MI250X'  # Default fallback
            gpu_message = f"Warning: Could not detect GPU. Defaulting to {detected_gpu} ({gfx_arch})"
    
    # Save original working directory
    original_dir = os.getcwd()
    
    # Convert output_dir to absolute path
    output_path = Path(__file__).parent / output_dir
    output_dir = os.path.abspath(output_path)
    
    # Convert input counter file to absolute path (use detected architecture)
    roof_path = Path(__file__).parent / f'roof-counters-{gfx_arch}.txt'
    counter_input_file = os.path.abspath(roof_path)
    
    # Convert conversion script to absolute path
    convert_path = Path(__file__).parent / f'convert-counters-collection-format.py'
    conversion_script = os.path.abspath(convert_path)
    
    # Convert rooflineExtractor script to absolute path
    roofline_path = Path(__file__).parent / f'rooflineExtractor.py'
    roofline_script = os.path.abspath(roofline_path)
    
    # Convert all relative file paths in run_command to absolute paths
    # This ensures that when we change directory, the paths still work
    for i in range(len(run_command)):
        # Check if this argument looks like a file/directory that exists
        if os.path.exists(run_command[i]) and not os.path.isabs(run_command[i]):
            run_command[i] = os.path.abspath(run_command[i])
    
    run_command_str = ' '.join(run_command)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(output_dir, f'profile_{timestamp}.log')
    logger = LogWriter(log_file_path)
    
    logger.write(f"Profiling session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.write(gpu_message)
    logger.write(f"Run command: {run_command_str}")
    logger.write(f"Output directory: {output_dir}")
    logger.write(f"Log file: {log_file_path}")
    logger.write("=" * 80)
    
    # Run rocprofv3 with counter collection
    logger.write("\n[1/4] Running rocprofv3 with counter collection (four runs of the application)...")
    logger.write(f"Command: rocprofv3 -i {counter_input_file} -o counters -f csv -- {run_command_str}")
    logger.write("-" * 80)
    
    counter_cmd_with_csv = ['rocprofv3', '-i', counter_input_file, '-o', 'counters', '-f', 'csv', '--'] + run_command
    counter_cmd_without_csv = ['rocprofv3', '-i', counter_input_file, '-o', 'counters', '--'] + run_command
    try:
        result1 = run_rocprofv3_with_retry(counter_cmd_with_csv, counter_cmd_without_csv, None, logger)
        if result1.returncode != 0:
            logger.write(f"Error: Counter collection failed with exit code {result1.returncode}")
            logger.close()
            sys.exit(1)
        else:
            logger.write("Counter collection completed successfully")
            # Move counter output directories to output_dir
            for counter_dir in glob.glob('pmc_*'):
                logger.write(f"Moving {counter_dir} to {output_dir}")
                try:
                    shutil.move(counter_dir, output_dir)
                except shutil.Error as e:
                    # If the destination exists, move files individually
                    if "already exists" in str(e):
                        logger.write(f"Warning: {counter_dir} already exists in {output_dir}, moving files individually.")
                        dest_dir = os.path.join(output_dir, os.path.basename(counter_dir))
                        for item in os.listdir(counter_dir):
                            src_item = os.path.join(counter_dir, item)
                            dest_item = os.path.join(dest_dir, item)
                            logger.write(f"  Moving {src_item} to {dest_item}")
                            shutil.move(src_item, dest_item)
                        # Remove the now-empty source directory
                        os.rmdir(counter_dir)
                    else:
                        raise
            # Also move any other counters* files that may exist
            for counter_file in glob.glob('counters*'):
                dest = os.path.join(output_dir, counter_file)
                logger.write(f"Moving {counter_file} to {dest}")
                shutil.move(counter_file, dest)
    except FileNotFoundError:
        logger.write("Error: rocprofv3 not found. Make sure it's installed and in your PATH.")
        logger.close()
        sys.exit(1)
    except Exception as e:
        logger.write(f"Error running counter collection: {e}")
        logger.close()
        sys.exit(1)
    
    logger.write("=" * 80)
    
    # Convert counter collection format
    logger.write("\n[2/4] Converting counter collection format...")
    logger.write(f"Command: cd {output_dir} && python3 {conversion_script} -i . -o counters.csv")
    logger.write("-" * 80)
    
    convert_cmd = ['python3', conversion_script, '-i', '.', '-o', 'counters.csv']
    try:
        result2 = subprocess.run(convert_cmd, cwd=output_dir, check=False,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                universal_newlines=True)
        
        # Log output
        if result2.stdout:
            logger.write(result2.stdout, end='')
        if result2.stderr:
            logger.write(result2.stderr, end='', file=sys.stderr)
        
        if result2.returncode != 0:
            logger.write(f"Error: Counter conversion failed with exit code {result2.returncode}")
            logger.close()
            sys.exit(1)
        else:
            logger.write("Counter conversion completed successfully")
    except FileNotFoundError:
        logger.write(f"Error: convert_counters_collection_format.py not found. Expected it to be in the same directory as profile.py (path: {conversion_script}).")
        logger.close()
        sys.exit(1)
    except Exception as e:
        logger.write(f"Error running counter conversion: {e}")
        logger.close()
        sys.exit(1)
    
    logger.write("=" * 80)
    
    # Run rocprofv3 with kernel tracing
    logger.write("\n[3/4] Running rocprofv3 with kernel tracing (one run of the application)...")
    logger.write(f"Command: rocprofv3 --kernel-trace -o trace -f csv -- {run_command_str}")
    logger.write("-" * 80)
    
    trace_cmd_with_csv = ['rocprofv3', '--kernel-trace', '-o', 'trace', '-f', 'csv', '--'] + run_command
    trace_cmd_without_csv = ['rocprofv3', '--kernel-trace', '-o', 'trace', '--'] + run_command
    try:
        result3 = run_rocprofv3_with_retry(trace_cmd_with_csv, trace_cmd_without_csv, None, logger)
        if result3.returncode != 0:
            logger.write(f"Error: Kernel tracing failed with exit code {result3.returncode}")
            logger.close()
            sys.exit(1)
        else:
            logger.write("Kernel tracing completed successfully")
            # Move trace output files to output_dir
            for trace_file in glob.glob('trace*'):
                dest = os.path.join(output_dir, trace_file)
                logger.write(f"Moving {trace_file} to {dest}")
                shutil.move(trace_file, dest)
    except Exception as e:
        logger.write(f"Error running kernel tracing: {e}")
        logger.close()
        sys.exit(1)
    
    logger.write("=" * 80)
    
    # Run rooflineExtractor
    logger.write("\n[4/4] Running rooflineExtractor...")
    
    # Build roofline command with arch flags
    roofline_cmd = ['python3', roofline_script, '-c', os.path.join(output_dir, 'counters.csv'), '-r', os.path.join(output_dir, 'trace_kernel_trace.csv'), '-p', '-d', '--arch', detected_gpu]
    
    # Add --proj-arch if provided
    if args.proj_arch:
        roofline_cmd.extend(['--proj-arch', args.proj_arch.upper()])
    
    logger.write(f"Command: cd {original_dir} && python3 {' '.join(roofline_cmd[1:])}")
    logger.write("-" * 80)
    try:
        result4 = subprocess.run(roofline_cmd, cwd=original_dir, check=False,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                universal_newlines=True)
        
        # Log output
        if result4.stdout:
            logger.write(result4.stdout, end='')
        if result4.stderr:
            logger.write(result4.stderr, end='', file=sys.stderr)
        
        if result4.returncode != 0:
            logger.write(f"Error: rooflineExtractor failed with exit code {result4.returncode}")
            logger.close()
            sys.exit(1)
        else:
            logger.write("rooflineExtractor completed successfully")
    except FileNotFoundError:
        logger.write("Error: rooflineExtractor.py not found in current directory.")
        logger.close()
        sys.exit(1)
    except Exception as e:
        logger.write(f"Error running rooflineExtractor: {e}")
        logger.close()
        sys.exit(1)
    
    logger.write("=" * 80)
    logger.write("\nAll profiling and conversion steps completed!")
    logger.write(f"Output files in {output_dir}:")
    
    # List of output files to check
    output_files = [
        ("Kernel runtime trace", os.path.join(output_dir, 'trace_kernel_trace.csv')),
        ("Hardware counters", os.path.join(output_dir, 'counters.csv')),
        ("Roofline statistics per kernel dispatch", os.path.join(output_dir, 'counters_EXTRACTED.csv')),
        ("Roofline statistics (aggregated dispatches)", os.path.join(output_dir, 'counters_EXTRACTED_AGG.csv')),
        ("Roofline plot", os.path.join(output_dir, 'counters.html')),
        ("Full console log", log_file_path)
    ]
    
    # Only print files that exist
    for description, file_path in output_files:
        if os.path.exists(file_path):
            logger.write(f"  - {description:<45}: {file_path}")
    
    logger.write(f"\nProfiling session ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.close()


if __name__ == "__main__":
    main()


