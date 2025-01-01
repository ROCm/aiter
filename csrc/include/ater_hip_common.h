#include <hip/hip_runtime.h>

#define HIP_CALL(call)                                                                  \
    do                                                                                  \
    {                                                                                   \
        hipError_t err = call;                                                          \
        if (err != hipSuccess)                                                          \
        {                                                                               \
            printf("[HIP error](%s) fail to call %s\n", hipGetErrorString(err), #call); \
            exit(0);                                                                    \
        }                                                                               \
    } while (0)

struct p3
{
    unsigned int _p0;
    unsigned int _p1;
    unsigned int _p2;
};
struct p2
{
    unsigned int _p0;
    unsigned int _p1;
};
struct AterAsmKernelArgs
{
    void *args_ptr;
    void *arg_size_ptr;
    int gdx;
    int gdy;
    int gdz;
    int bdx;
    int bdy;
    int bdz;
    const hipStream_t stream;
};

class AterAsmKernel
{
private:
    hipModule_t module;
    hipFunction_t kernel_func;

public:
    AterAsmKernel(const char *name, const char *hsaco)
    {
        HIP_CALL(hipModuleLoad(&module, (std::string(ATER_ASM_DIR) + hsaco).c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, name));
    };

    void launch_kernel(const AterAsmKernelArgs &kargs)
    {
        void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kargs.args_ptr,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, kargs.arg_size_ptr,
                          HIP_LAUNCH_PARAM_END};

        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       kargs.gdx, kargs.gdy, kargs.gdz,
                                       kargs.bdx, kargs.bdy, kargs.bdz,
                                       0, kargs.stream, nullptr, (void **)&config));
    };
};