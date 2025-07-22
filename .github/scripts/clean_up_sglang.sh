#!/bin/bash

# Uninstall the Python package sgl-kernel
echo "Uninstalling sgl-kernel..."
python3 -m pip uninstall -y sgl-kernel

# Check and stop the Docker container ci_sglang
echo "Stopping Docker container ci_sglang..."
docker stop ci_sglang

# Remove the Docker container ci_sglang
echo "Removing Docker container ci_sglang..."
docker rm ci_sglang

echo "Clean up completed."
