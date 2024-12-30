#pragma once

class Kernel_AR
{
private:
    hipModule_t module;
    hipFunction_t kernel_func;

public:
    Kernel_AR(const char *name, const char *hsaco);

    void launch_kernel(const void *gpu_buf_in,
                       const void *gpu_sig_in[],
                       uint32_t gpu_id,
                       uint32_t buf_size,
                       uint32_t world_size);
};