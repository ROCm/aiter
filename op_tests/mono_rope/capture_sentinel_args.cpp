// Capture A07's actual public-wrapper launch; only used for correctness probes.
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
static unsigned char captured[120];
static char captured_name[1024];
static bool ready = false;
extern "C" int sentinel_a07_args(void* output, char* name) {
    if (!ready) return 0;
    std::memcpy(output, captured, 120);
    std::strcpy(name, captured_name);
    return 1;
}
extern "C" hipError_t hipLaunchKernel(const void* f, dim3 grid, dim3 block,
                                      void** args, size_t lds, hipStream_t stream) {
    static auto real = reinterpret_cast<decltype(&hipLaunchKernel)>(dlsym(RTLD_NEXT,"hipLaunchKernel"));
    static auto name_of = reinterpret_cast<decltype(&hipKernelNameRefByPtr)>(dlsym(RTLD_NEXT,"hipKernelNameRefByPtr"));
    if (!real || !name_of) std::abort();
    const char* name = name_of(f,stream);
    if (name && std::strstr(name,"inverse_rope_group_quant_persistent_kernel")) {
        if (grid.x!=16 || grid.y!=16 || grid.z!=1 || block.x!=128 || block.y!=1 || block.z!=1 || lds!=8192) std::abort();
        constexpr int offsets[] = {0,8,16,24,32,40,48,52,56,60,64,68,72,80,88,96,100,104,108,109,112,116};
        constexpr int sizes[]   = {8,8,8,8,8,8,4,4,4,4,4,4,8,8,8,4,4,4,1,1,4,4};
        std::memset(captured,0,120);
        for (int i=0;i<22;++i) std::memcpy(captured+offsets[i],args[i],sizes[i]);
        if (std::strlen(name)>=sizeof(captured_name)) std::abort();
        std::strcpy(captured_name,name);
        ready=true;
    }
    return real(f,grid,block,args,lds,stream);
}
