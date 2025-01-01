#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "ater_hip_common.h"

struct __attribute__((packed)) KernelArgs
{
    void *ptr_O;
    p2 _p0;
    void *ptr_Q;
    p2 _p1;
    void *ptr_K;
    p2 _p2;
    void *ptr_V;
    p2 _p3;
    void *ptr_BT;
    p2 _p4;
    void *ptr_CL;
    p2 _p5;
    void *ptr_KQ;
    p2 _p6;
    void *ptr_VQ;
    p2 _p7;
    float sclg2e;
    p3 _p12;
    unsigned int mblk;
    p3 _p13;
    unsigned int batch;
    p3 _p14;
    unsigned int Qs;
    p3 _p15;
    unsigned int Bs;
    p3 _p16;
    unsigned int KVs;
    p3 _p17;
};

const float f_log2E = log2f(expf(1));
class Kernel
{
private:
    hipModule_t module;
    hipFunction_t kernel_func;

public:
    Kernel(const char *name, const char *hsaco)
    {
        HIP_CALL(hipModuleLoad(&module, (std::string(ATER_ASM_DIR) + hsaco).c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, name));
    };

    template <typename T, typename T_O>
    void launch_kernel(torch::Tensor &O,            //   [num_seqs, num_heads, head_size]
                       torch::Tensor &Q,            //   [num_seqs, num_heads, head_size]
                       torch::Tensor &K,            //   [num_blocks, num_kv_heads, head_size/x, block_size, x]
                       torch::Tensor &V,            //   [num_blocks, num_kv_heads, block_size/X, head_size, X]
                       torch::Tensor &block_tables, //   [num_seqs, max_num_blocks_per_seq]
                       torch::Tensor &context_lens, //   [num_seqs]
                       std::optional<torch::Tensor> K_QScale = std::nullopt,
                       std::optional<torch::Tensor> V_QScale = std::nullopt)
    {
        int batch = block_tables.size(0);
        int max_num_blocks = block_tables.size(1);
        int num_heads = Q.size(1);
        int head_size = Q.size(2);
        int num_kv_heads = K.size(1);
        int block_size = K.size(3);
        const int gqa_ratio = num_heads / num_kv_heads;

        int dim = head_size;
        int stride_Q = gqa_ratio * dim * sizeof(uint16_t);
        int stride_KV_head = block_size * dim * sizeof(T);
        int stride_KV_blk = stride_KV_head * num_kv_heads;
        float k_log2e = f_log2E;
        float k_scalar = sqrt(dim);
        k_scalar = (float)((double)k_log2e / (double)k_scalar);

        KernelArgs args;
        size_t arg_size = sizeof(args);
        args.ptr_O = O.data_ptr();
        args.ptr_Q = Q.data_ptr();
        args.ptr_K = K.data_ptr();
        args.ptr_V = V.data_ptr();
        args.ptr_BT = block_tables.data_ptr();
        args.ptr_CL = context_lens.data_ptr();
        if constexpr (std::is_same<T, uint8_t>::value)
        {
            args.ptr_KQ = K_QScale.value().data_ptr();
            args.ptr_VQ = V_QScale.value().data_ptr();
        }
        else
        {
            args.ptr_KQ = nullptr;
            args.ptr_VQ = nullptr;
        }
        args.sclg2e = k_scalar;
        args.mblk = max_num_blocks;
        args.batch = batch;
        args.Qs = stride_Q;
        args.Bs = stride_KV_blk;
        args.KVs = stride_KV_head;

        void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          &arg_size, HIP_LAUNCH_PARAM_END};

        int bdx = 256;
        int gdx = 1;
        int gdy = batch; // sub_X_cnt;
        int gdz = 1;
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       gdx, gdy, gdz,
                                       bdx, 1, 1,
                                       0, stream, nullptr, (void **)&config));
    };
};

torch::Tensor pa_fwd(torch::Tensor &Q,            //   [num_seqs, num_heads, head_size]
                     torch::Tensor &K,            //   [num_blocks, num_kv_heads, head_size/x, block_size, x]
                     torch::Tensor &V,            //   [num_blocks, num_kv_heads, block_size/X, head_size, X]
                     torch::Tensor &block_tables, //   [num_seqs, max_num_blocks_per_seq]
                     torch::Tensor &context_lens, //   [num_seqs]
                     std::optional<torch::Tensor> K_QScale = std::nullopt,
                     std::optional<torch::Tensor> V_QScale = std::nullopt)
{
    torch::Tensor output = torch::empty_like(Q);
    static Kernel impl("pa_kernel_func", "pa_a16w16.co");
    impl.launch_kernel<uint16_t, uint16_t>(output,
                                           Q,
                                           K,
                                           V,
                                           block_tables,
                                           context_lens,
                                           K_QScale,
                                           V_QScale);
    return output;
}