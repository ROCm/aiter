#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "communication_asm.h"

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
struct __attribute__((packed)) KernelArgs
{
    void *ptr_gpu0_data;
    p2 _p0;
    void *ptr_gpu0_sig;
    p2 _p8;
    void *ptr_gpu1_sig;
    p2 _p9;
    void *ptr_gpu2_sig;
    p2 _p10;
    void *ptr_gpu3_sig;
    p2 _p11;
    void *ptr_gpu4_sig;
    p2 _p12;
    void *ptr_gpu5_sig;
    p2 _p13;
    void *ptr_gpu6_sig;
    p2 _p14;
    void *ptr_gpu7_sig;
    p2 _p15;
    unsigned int gpuId;
    p3 _p16;
    unsigned int stride_gpu;
    p3 _p17;
    unsigned int stride_tg;
    p3 _p18;
    unsigned int stride_wave;
    p3 _p19;
    unsigned int loopcnt;
    p3 _p20;
};

#define HIP_CALL(call)                                                                 \
    do                                                                                 \
    {                                                                                  \
        hipError_t err = call;                                                         \
        if (err != hipSuccess)                                                         \
        {                                                                              \
            printf("[hiperror](%s) fail to call %s\n", hipGetErrorString(err), #call); \
            exit(0);                                                                   \
        }                                                                              \
    } while (0)

Kernel_AR::Kernel_AR(const char *name, const char *hsaco)
{
    HIP_CALL(hipModuleLoad(&module, (std::string(ATER_ASM_DIR) + hsaco).c_str()));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, name));
};

void Kernel_AR::launch_kernel(const void *gpu_buf_in,
                              const void *gpu_sig_in[],
                              uint32_t gpu_id,
                              uint32_t buf_size,
                              uint32_t world_size)
{
    int bdx = 256;
    int gdx = 64;
    int gdy = 1;
    int gdz = 1;
    int stride_GPU = buf_size / world_size; // stride base on the pass in GPU id; gpu0 focus on 0~15; gpu1 focus on 16~31
    int stride_TG = stride_GPU / gdx;       // stride base on TG id; 64 TGs, every TG focus on 16*8192/64=2048 elements
    int stride_WV = stride_TG / (bdx / 64); // stride base on Wave id, 4 waves, every wave focus on 512 elements; 1024 bytes

    KernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_gpu0_data = const_cast<void *>(gpu_buf_in);
    args.ptr_gpu0_sig = const_cast<void *>(gpu_sig_in[0]);
    args.ptr_gpu1_sig = const_cast<void *>(gpu_sig_in[1]);
    args.ptr_gpu2_sig = const_cast<void *>(gpu_sig_in[2]);
    args.ptr_gpu3_sig = const_cast<void *>(gpu_sig_in[3]);
    args.ptr_gpu4_sig = const_cast<void *>(gpu_sig_in[4]);
    args.ptr_gpu5_sig = const_cast<void *>(gpu_sig_in[5]);
    args.ptr_gpu6_sig = const_cast<void *>(gpu_sig_in[6]);
    args.ptr_gpu7_sig = const_cast<void *>(gpu_sig_in[7]);
    args.gpuId = gpu_id;
    args.stride_gpu = stride_GPU;
    args.stride_tg = stride_TG;
    args.stride_wave = stride_WV;
    args.loopcnt = 10;

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &arg_size, HIP_LAUNCH_PARAM_END};

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                   gdx, gdy, gdz,
                                   bdx, 1, 1,
                                   0, stream, nullptr, (void **)&config));
};

// void open_ipc_handles(int device, const std::vector<std::string> &gpu_handles, std::vector<void *> &gpu_ptr)
// {
//     HIP_CALL(hipSetDevice(device));
//     for (size_t i = 0; i < gpu_handles.size(); ++i)
//     {
//         char *ipc_ptr;
//         HIP_CALL(hipIpcOpenMemHandle((void **)&ipc_ptr,
//                                      *((const hipIpcMemHandle_t *)gpu_handles[i].data()),
//                                      hipIpcMemLazyEnablePeerAccess));
//         gpu_ptr[i] = ipc_ptr;
//     }
// }
// void enable_peer_access(int rank_id, int world_size)
// {
//     for (int i = 0; i < world_size; ++i)
//     {
//         if (i != rank_id)
//         {
//             int canAccessPeer;
//             HIP_CALL(hipDeviceCanAccessPeer(&canAccessPeer, rank_id, i));
//             if (canAccessPeer)
//             {
//                 HIP_CALL(hipDeviceEnablePeerAccess(i, 0));
//             }
//             else
//             {
//                 std::cerr << "Device " << rank_id << " cannot access device " << i << std::endl;
//             }
//         }
//     }
// }
// torch::Tensor allreduce_fwd(torch::Tensor &input,
//                             uint32_t rank_id,
//                             const std::vector<std::string> &gpu_sig_handles,
//                             const std::vector<std::string> &gpu_buf_handles)
// {
//     int world_size = gpu_sig_handles.size();
//     if (rank_id < 0 || rank_id >= world_size)
//         throw std::invalid_argument("invalid rank passed in");
//     enable_peer_access(rank_id, world_size);

//     std::vector<void *> gpu_sig_in(world_size);
//     std::vector<void *> gpu_buf_in(world_size);

//     open_ipc_handles(rank_id, gpu_sig_handles, gpu_sig_in);
//     open_ipc_handles(rank_id, gpu_buf_handles, gpu_buf_in);

//     static Kernel impl("allreduce_kernel_func", "all_reduce.co");

//     auto input_size = input.numel() * input.element_size();
//     AT_CUDA_CHECK(hipMemcpy((void *)gpu_buf_in[rank_id], input.data_ptr(),
//                             // AT_CUDA_CHECK(hipMemcpy((void *)gpu_buf_in[(rank_id + 1) % world_size], input.data_ptr(),
//                             input_size, cudaMemcpyDeviceToDevice));
//     // impl.launch_kernel<uint16_t, uint16_t>(gpu_buf_in,
//     //                                        gpu_sig_in,
//     //                                        rank_id,
//     //                                        input_size,
//     //                                        world_size);

//     auto options = torch::TensorOptions()
//                        .dtype(input.dtype())
//                        .device(input.device());
//     return torch::from_blob((void *)gpu_buf_in[rank_id], {input.sizes()}, options);
// }

// uint64_t HIP_Malloc(uint32_t sizeInByte)
// {
//     void *buf;
//     HIP_CALL(hipMalloc(&buf, sizeInByte));
//     uint64_t buf_addr = reinterpret_cast<uint64_t>(buf);

//     return buf_addr;
// }