# RMS (Root Mean Square) for Multi-GPU Systems

RMS is a high-performance library providing efficient Root Mean Square (RMS) operations for multi-GPU setups on a single node. In distributed training scenarios (e.g., tensor parallelism or pipeline parallelism), each GPU holds partial results from operations like GEMM. The RMS implementation performs an **all-gather** to collect partial sums of squares, followed by an **all-reduce** to compute the global sum, enabling accurate RMS calculation across all GPUs. Finally, an **all-scatter** (or broadcast of the RMS value) distributes the normalized tensor identically to each GPU.

This is particularly useful for implementing distributed RMSNorm (Root Mean Square Layer Normalization) in large language models.

The library supports two programming models:
- **Multithreaded Shared Memory**: Uses multiple threads within a single process to manage GPUs via shared memory.
- **Distributed Multiprocessing**: Uses multiple processes (e.g., via `multiprocessing.spawn` or `torchrun`) for explicit distributed control.

---

## Installation & Integration

To integrate RMS with PyTorch, use the provided ROCm-based container environment:

1. **Pull the Docker Image**:

   docker pull rocm/pytorch-private:vllm_91_rocm7.0_aiter_1005_7ed998_latest

2. **Run the Container**:
   Mount your source code directory (containing the RMS library) to `/workspace`:

   docker run -it --rm --device=/dev/kfd --device=/dev/dri --shm-size=16g -v $(pwd):/workspace rocm/pytorch-private:vllm_91_rocm7.0_aiter_1005_7ed998_latest

3. **Install the Package**:
   Inside the container:

   cd /workspace
   pip install .

   This installs the RMS library and integrates it with the container's PyTorch installation.

---

## Running Tests and Examples

The examples demonstrate mixed-precision RMS operations on a node with 8 GPUs.

Navigate to the examples directory:

cd /workspace/examples

- **Multithreaded Shared Memory Model**:
  Run on a single process using 8 threads:

  python3 shared-rms-mixed.py

- **Distributed Multiprocessing Model**:
  - Using `multiprocessing.spawn` to launch 8 processes:

    python3 dist-spawn-rms-mixed.py

  - Using `torchrun` to launch 8 processes (recommended for PyTorch distributed):

    torchrun --nproc_per_node=8 dist-torch-rms-mixed.py

These scripts test the RMS functionality in mixed precision and validate correctness across GPUs.

