// ualoe_allreduce.cpp -- cross-node custom all-reduce on gfx1250 over IFOE.
// Lifts aiter custom_all_reduce's 2-stage naive kernel + start_sync/end_sync,
// but shares peer buffers via HIP FABRIC handles (works cross-node, per ubench 07)
// instead of IPC. One process per rank; rank 0 is the TCP coordinator that
// all-gathers + broadcasts the fabric handles. TP4 = 4 ranks 1 node; TP8 = 8 ranks
// across 2 nodes.
//
// Build: hipcc -std=c++17 -O3 ualoe_allreduce.cpp -o ualoe_allreduce
// Run  : ./ualoe_allreduce --rank R --world N --gpu G --coord IP --port P --mb MB
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <vector>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"[r%d] ERR %s:%d %s\n",g_rank,__FILE__,__LINE__,hipGetErrorString(e));exit(2);}}while(0)
static int g_rank=0;
constexpr int kMaxBlocks=304;   // allow high occupancy (>= CU count)
struct Signal { alignas(128) uint32_t start[kMaxBlocks][8]; alignas(128) uint32_t end[kMaxBlocks][8]; alignas(128) uint32_t _flag[kMaxBlocks]; };
struct __align__(16) RankData { const void* ptrs[8]; };
struct __align__(16) RankSignals { Signal* signals[8]; };
#define DINLINE __device__ __forceinline__

template<int ngpus>
DINLINE void start_sync(const RankSignals& sg, Signal* self_sg, int rank){
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if(threadIdx.x < ngpus){
        __scoped_atomic_store_n(&sg.signals[threadIdx.x]->start[blockIdx.x][rank], flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
        while(__scoped_atomic_load_n(&self_sg->start[blockIdx.x][threadIdx.x], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) < flag);
    }
    __syncthreads();
    if(threadIdx.x==0) self_sg->_flag[blockIdx.x]=flag;
}
template<int ngpus, bool final_sync=false>
DINLINE void end_sync(const RankSignals& sg, Signal* self_sg, int rank){
    __syncthreads();
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if(threadIdx.x < ngpus){
        __scoped_atomic_store_n(&sg.signals[threadIdx.x]->end[blockIdx.x][rank], flag, final_sync?__ATOMIC_RELAXED:__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        while(__scoped_atomic_load_n(&self_sg->end[blockIdx.x][threadIdx.x], final_sync?__ATOMIC_RELAXED:__ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) < flag);
    }
    __syncthreads();
    if(threadIdx.x==0) self_sg->_flag[blockIdx.x]=flag;
}
template<int ngpus> DINLINE float4 preduce(const float4* const ptrs[], int idx){
    float4 t=ptrs[0][idx];
    #pragma unroll
    for(int i=1;i<ngpus;i++){ float4 v=ptrs[i][idx]; t.x+=v.x; t.y+=v.y; t.z+=v.z; t.w+=v.w; }
    return t;
}
template<typename P> DINLINE P* tmp_buf(Signal* s){ return (P*)(s+1); }

template<int ngpus>
__global__ void __launch_bounds__(512,1) allreduce2(RankData* in_dp, RankSignals sg, Signal* self_sg, float* result, int rank, int size){
    int tid=blockIdx.x*blockDim.x+threadIdx.x, stride=gridDim.x*blockDim.x;
    int part=size/ngpus, start=rank*part, end=(rank==ngpus-1)?size:start+part, largest=part+size%ngpus;
    const float4* ptrs[ngpus]; float4* tmps[ngpus];
    #pragma unroll
    for(int i=0;i<ngpus;i++){ int t=(rank+i)%ngpus; ptrs[i]=(const float4*)in_dp->ptrs[t]; tmps[i]=tmp_buf<float4>(sg.signals[t]); }
    start_sync<ngpus>(sg,self_sg,rank);
    for(int idx=start+tid; idx<end; idx+=stride) tmps[0][idx-start]=preduce<ngpus>(ptrs,idx);
    end_sync<ngpus>(sg,self_sg,rank);
    for(int idx=tid; idx<largest; idx+=stride){
        #pragma unroll
        for(int i=0;i<ngpus;i++){ int gr=(rank+i)%ngpus; if(gr==ngpus-1||idx<part){ ((float4*)result)[gr*part+idx]=tmps[i][idx]; } }
    }
}

// instrumented copy: block 0 records per-phase cycles [start_sync, reduce, end_sync, allgather]
template<int ngpus>
__global__ void __launch_bounds__(512,1) allreduce2_prof(RankData* in_dp, RankSignals sg, Signal* self_sg, float* result, int rank, int size, unsigned long long* prof){
    int tid=blockIdx.x*blockDim.x+threadIdx.x, stride=gridDim.x*blockDim.x;
    int part=size/ngpus, start=rank*part, end=(rank==ngpus-1)?size:start+part, largest=part+size%ngpus;
    const float4* ptrs[ngpus]; float4* tmps[ngpus];
    #pragma unroll
    for(int i=0;i<ngpus;i++){ int t=(rank+i)%ngpus; ptrs[i]=(const float4*)in_dp->ptrs[t]; tmps[i]=tmp_buf<float4>(sg.signals[t]); }
    long long c0=clock64();
    start_sync<ngpus>(sg,self_sg,rank);
    long long c1=clock64();
    for(int idx=start+tid; idx<end; idx+=stride) tmps[0][idx-start]=preduce<ngpus>(ptrs,idx);
    long long c2=clock64();
    end_sync<ngpus>(sg,self_sg,rank);
    long long c3=clock64();
    for(int idx=tid; idx<largest; idx+=stride){
        #pragma unroll
        for(int i=0;i<ngpus;i++){ int gr=(rank+i)%ngpus; if(gr==ngpus-1||idx<part){ ((float4*)result)[gr*part+idx]=tmps[i][idx]; } }
    }
    long long c4=clock64();
    if(threadIdx.x==0 && blockIdx.x==0){ prof[0]=c1-c0; prof[1]=c2-c1; prof[2]=c3-c2; prof[3]=c4-c3; }
}

// MLP-optimized: U-way unrolled reduce-scatter + per-peer contiguous allgather.
// Raises outstanding fabric loads per thread (ubench-02 HBM lesson) to saturate IFOE.
template<int ngpus,int U>
__global__ void __launch_bounds__(512,1) allreduce2_opt(RankData* in_dp, RankSignals sg, Signal* self_sg, float* result, int rank, int size){
    int tid=blockIdx.x*blockDim.x+threadIdx.x, stride=gridDim.x*blockDim.x;
    int part=size/ngpus, start=rank*part, end=(rank==ngpus-1)?size:start+part, largest=part+size%ngpus;
    const float4* ptrs[ngpus]; float4* tmps[ngpus];
    #pragma unroll
    for(int i=0;i<ngpus;i++){ int t=(rank+i)%ngpus; ptrs[i]=(const float4*)in_dp->ptrs[t]; tmps[i]=tmp_buf<float4>(sg.signals[t]); }
    start_sync<ngpus>(sg,self_sg,rank);
    // reduce-scatter: U float4 in flight per thread across the whole peer set
    for(int i0=start+tid; i0<end; i0+=stride*U){
        float4 acc[U];
        #pragma unroll
        for(int u=0;u<U;u++){ int idx=i0+u*stride; acc[u]= idx<end ? ptrs[0][idx] : float4{0,0,0,0}; }
        #pragma unroll
        for(int p=1;p<ngpus;p++){
            #pragma unroll
            for(int u=0;u<U;u++){ int idx=i0+u*stride; if(idx<end){ float4 v=ptrs[p][idx]; acc[u].x+=v.x;acc[u].y+=v.y;acc[u].z+=v.z;acc[u].w+=v.w; } }
        }
        #pragma unroll
        for(int u=0;u<U;u++){ int idx=i0+u*stride; if(idx<end) tmps[0][idx-start]=acc[u]; }
    }
    end_sync<ngpus>(sg,self_sg,rank);
    // allgather: each peer's reduced shard copied contiguously (ubench-07 read pattern)
    #pragma unroll
    for(int i=0;i<ngpus;i++){
        int gr=(rank+i)%ngpus; int cnt=(gr==ngpus-1)?largest:part;
        float4* dst=(float4*)result+(size_t)gr*part; const float4* src=tmps[i];
        for(int i0=tid; i0<cnt; i0+=stride*U){
            #pragma unroll
            for(int u=0;u<U;u++){ int idx=i0+u*stride; if(idx<cnt) dst[idx]=src[idx]; }
        }
    }
}

// ---- TDM (gfx1250 tensor_load_to_lds) for the reduce-scatter read path ----
using sg0v=int __attribute__((ext_vector_type(4)));
using sg1v=int __attribute__((ext_vector_type(8)));
using sg2v=int __attribute__((ext_vector_type(4)));
using sg3v=int __attribute__((ext_vector_type(4)));
using sgxv=int __attribute__((ext_vector_type(8)));
__device__ void s_wait_tensorcnt(short) __asm("llvm.amdgcn.s.wait.tensorcnt");
static constexpr int TDM_CPOL=(0)|(2<<3);
__device__ inline void tdm_load(uint32_t lds,const void* g,uint32_t td0){ // 1-row tile of td0 floats
    uint64_t ga=reinterpret_cast<uint64_t>(g); uint32_t sg0[4],sg1[8];
    sg0[0]=1u; sg0[1]=lds; sg0[2]=uint32_t(ga); sg0[3]=(1u<<31)|uint32_t((ga>>32)&0x01FFFFFFu);
    sg1[0]=(2u&0x3u)<<16; sg1[1]=(td0<<16); sg1[2]=(td0>>16)|(1u<<16);
    sg1[3]=(1u>>16)|((td0&0xFFFFu)<<16); sg1[4]=1u; sg1[5]=td0; sg1[6]=0; sg1[7]=0;
    __builtin_amdgcn_tensor_load_to_lds(__builtin_bit_cast(sg0v,sg0),__builtin_bit_cast(sg1v,sg1),
        sg2v{0,0,0,0},sg3v{0,0,0,0},sgxv{0,0,0,0,0,0,0,0},TDM_CPOL);
}
// TILE = float4 count per tile per peer; LDS = ngpus*TILE*16 bytes
template<int ngpus, int TILE>
__global__ void __launch_bounds__(512,1) allreduce2_tdm(RankData* in_dp, RankSignals sg, Signal* self_sg, float* result, int rank, int size){
    extern __shared__ char smem[];
    int part=size/ngpus, start=rank*part, end=(rank==ngpus-1)?size:start+part, largest=part+size%ngpus;
    const float4* ptrs[ngpus]; float4* tmps[ngpus];
    #pragma unroll
    for(int i=0;i<ngpus;i++){ int t=(rank+i)%ngpus; ptrs[i]=(const float4*)in_dp->ptrs[t]; tmps[i]=tmp_buf<float4>(sg.signals[t]); }
    start_sync<ngpus>(sg,self_sg,rank);
    // reduce-scatter: own shard [start,end), tiled; TDM-load each peer's tile -> LDS, sum
    for(int base=start+blockIdx.x*TILE; base<end; base+=gridDim.x*TILE){
        int n = end-base < TILE ? end-base : TILE;      // float4 count this tile
        if(threadIdx.x==0){
            #pragma unroll
            for(int p=0;p<ngpus;p++) tdm_load(p*TILE*16u, ptrs[p]+base, (uint32_t)n*4u); // n*4 floats
        }
        s_wait_tensorcnt(0); __syncthreads();
        for(int k=threadIdx.x;k<n;k+=blockDim.x){
            float4 acc=((float4*)(smem+0))[k];
            #pragma unroll
            for(int p=1;p<ngpus;p++){ float4 v=((float4*)(smem+(size_t)p*TILE*16))[k]; acc.x+=v.x;acc.y+=v.y;acc.z+=v.z;acc.w+=v.w; }
            tmps[0][(base-start)+k]=acc;
        }
        __syncthreads();
    }
    end_sync<ngpus>(sg,self_sg,rank);
    // stage2 allgather (direct)
    int tid=blockIdx.x*blockDim.x+threadIdx.x, stride=gridDim.x*blockDim.x;
    for(int idx=tid; idx<largest; idx+=stride){
        #pragma unroll
        for(int i=0;i<ngpus;i++){ int gr=(rank+i)%ngpus; if(gr==ngpus-1||idx<part){ ((float4*)result)[gr*part+idx]=tmps[i][idx]; } }
    }
}

static void sa(int fd,const void*b,size_t n){const char*p=(const char*)b;while(n){ssize_t w=send(fd,p,n,0);if(w<=0)exit(3);p+=w;n-=w;}}
static void ra(int fd,void*b,size_t n){char*p=(char*)b;while(n){ssize_t r=recv(fd,p,n,0);if(r<=0)exit(3);p+=r;n-=r;}}
static void* fab(size_t bytes,int dev,size_t& tot,hipMemFabricHandle_t& fh){
    hipMemAllocationProp pr={}; pr.type=hipMemAllocationTypePinned; pr.requestedHandleTypes=hipMemHandleTypeFabric;
    pr.location.type=hipMemLocationTypeDevice; pr.location.id=dev;
    size_t g=0; CK(hipMemGetAllocationGranularity(&g,&pr,hipMemAllocationGranularityRecommended));
    tot=((bytes+g-1)/g)*g; hipMemGenericAllocationHandle_t h; CK(hipMemCreate(&h,tot,&pr,0));
    void*p=0; CK(hipMemAddressReserve(&p,tot,0,0,0)); CK(hipMemMap(p,tot,0,h,0));
    hipMemAccessDesc ad={}; ad.location.type=hipMemLocationTypeDevice; ad.location.id=dev; ad.flags=hipMemAccessFlagsProtReadWrite;
    CK(hipMemSetAccess(p,tot,&ad,1)); CK(hipMemExportToShareableHandle(&fh,h,hipMemHandleTypeFabric,0)); return p;
}
static void* imp(const hipMemFabricHandle_t& fh,size_t tot,int dev){
    hipMemGenericAllocationHandle_t h; CK(hipMemImportFromShareableHandle(&h,(void*)&fh,hipMemHandleTypeFabric));
    void*p=0; CK(hipMemAddressReserve(&p,tot,0,0,0)); CK(hipMemMap(p,tot,0,h,0));
    hipMemAccessDesc ad={}; ad.location.type=hipMemLocationTypeDevice; ad.location.id=dev; ad.flags=hipMemAccessFlagsProtReadWrite;
    CK(hipMemSetAccess(p,tot,&ad,1)); return p;
}
struct HPair{ hipMemFabricHandle_t in_h, sg_h; };  // per-rank handles

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    // opt (MLP-unrolled) kernel is the default: saturates the bidirectional fabric wall (~355 GB/s/dir).
    int rank=0,world=0,gpu=0,port=55570,mb=64,use_tdm=0,ublocks=0,uthreads=512,use_prof=0,use_opt=1,uunroll=8; const char* coord="127.0.0.1";
    for(int i=1;i<argc;i++){ auto A=[&](const char*k){return !strcmp(argv[i],k);};
        if(A("--rank"))rank=atoi(argv[++i]); else if(A("--world"))world=atoi(argv[++i]); else if(A("--gpu"))gpu=atoi(argv[++i]);
        else if(A("--coord"))coord=argv[++i]; else if(A("--port"))port=atoi(argv[++i]); else if(A("--mb"))mb=atoi(argv[++i]); else if(A("--tdm")){use_tdm=1;use_opt=0;} else if(A("--naive"))use_opt=0; else if(A("--blocks"))ublocks=atoi(argv[++i]); else if(A("--threads"))uthreads=atoi(argv[++i]); else if(A("--prof")){use_prof=1;use_opt=0;} else if(A("--opt"))use_opt=1; else if(A("--unroll"))uunroll=atoi(argv[++i]); }
    g_rank=rank; CK(hipSetDevice(gpu));
    size_t bytes=(size_t)mb<<20; size_t nfloat=bytes/4; int nvec=(int)(nfloat/4);

    size_t itot,stot; hipMemFabricHandle_t in_fh,sg_fh;
    void* in_buf=fab(bytes,gpu,itot,in_fh);                          // my input (peers read)
    void* sg_buf=fab(sizeof(Signal)+bytes,gpu,stot,sg_fh);          // my Signal + tmp (peers read/write)
    CK(hipMemset(sg_buf,0,sizeof(Signal)));
    { std::vector<float> h(nfloat,(float)(rank+1)); CK(hipMemcpy(in_buf,h.data(),bytes,hipMemcpyHostToDevice)); }
    float* result; CK(hipMalloc(&result,bytes));
    CK(hipDeviceSynchronize());

    // ---- coordinator all-gather of fabric handles ----
    std::vector<HPair> tab(world);
    tab[rank].in_h=in_fh; tab[rank].sg_h=sg_fh;
    std::vector<int> cfd(world,-1);   // rank0: client fds
    if(rank==0){
        int s=socket(AF_INET,SOCK_STREAM,0); int one=1; setsockopt(s,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
        sockaddr_in a={}; a.sin_family=AF_INET; a.sin_addr.s_addr=INADDR_ANY; a.sin_port=htons(port);
        if(bind(s,(sockaddr*)&a,sizeof(a))<0){perror("bind");exit(3);} listen(s,world);
        for(int c=0;c<world-1;c++){ int fd=accept(s,0,0); int cr; ra(fd,&cr,4); ra(fd,&tab[cr],sizeof(HPair)); cfd[cr]=fd; }
        close(s);
        for(int r=1;r<world;r++) sa(cfd[r],tab.data(),sizeof(HPair)*world);   // broadcast full table
    } else {
        int fd=socket(AF_INET,SOCK_STREAM,0); sockaddr_in a={}; a.sin_family=AF_INET; a.sin_port=htons(port); inet_pton(AF_INET,coord,&a.sin_addr);
        int ok=0; for(int i=0;i<120&&!ok;i++){ if(connect(fd,(sockaddr*)&a,sizeof(a))==0){ok=1;break;} usleep(500000); close(fd); fd=socket(AF_INET,SOCK_STREAM,0);} if(!ok)exit(3);
        int one=1; setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
        sa(fd,&rank,4); sa(fd,&tab[rank],sizeof(HPair)); ra(fd,tab.data(),sizeof(HPair)*world); cfd[0]=fd;
    }
    // import peers' input + signal buffers
    RankData h_rd; RankSignals sg;
    for(int i=0;i<world;i++){
        if(i==rank){ h_rd.ptrs[i]=in_buf; sg.signals[i]=(Signal*)sg_buf; }
        else       { h_rd.ptrs[i]=imp(tab[i].in_h,itot,gpu); sg.signals[i]=(Signal*)imp(tab[i].sg_h,stot,gpu); }
    }
    RankData* in_dp; CK(hipMalloc(&in_dp,sizeof(RankData))); CK(hipMemcpy(in_dp,&h_rd,sizeof(RankData),hipMemcpyHostToDevice));
    auto barrier=[&](){ char b=1; if(rank==0){ for(int r=1;r<world;r++) ra(cfd[r],&b,1); for(int r=1;r<world;r++) sa(cfd[r],&b,1);} else { sa(cfd[0],&b,1); ra(cfd[0],&b,1);} };

    int TH=uthreads; int nb=ublocks>0?ublocks:(nvec/world+TH-1)/TH; if(ublocks==0 && nb>208)nb=208; if(nb>kMaxBlocks)nb=kMaxBlocks; if(nb<1)nb=1;  // ~208 blocks: best occupancy vs barrier cost for the opt kernel
    constexpr int TILE=512; size_t tdm_lds=(size_t)8*TILE*16;   // up to 8 peers * 8KB
    #define OPTARGS in_dp,sg,(Signal*)sg_buf,result,rank,nvec
    #define LOPT(NG) switch(uunroll){ case 1: allreduce2_opt<NG,1><<<nb,TH>>>(OPTARGS); break; case 2: allreduce2_opt<NG,2><<<nb,TH>>>(OPTARGS); break; case 8: allreduce2_opt<NG,8><<<nb,TH>>>(OPTARGS); break; default: allreduce2_opt<NG,4><<<nb,TH>>>(OPTARGS);}
    auto launch=[&](){
        if(use_opt){ switch(world){ case 2: LOPT(2); break; case 4: LOPT(4); break; case 8: LOPT(8); break; default: if(rank==0) printf("unsupported world=%d\n",world);} return; }
        if(use_tdm){ switch(world){
            case 2: allreduce2_tdm<2,TILE><<<nb,TH,2*TILE*16>>>(in_dp,sg,(Signal*)sg_buf,result,rank,nvec); break;
            case 4: allreduce2_tdm<4,TILE><<<nb,TH,4*TILE*16>>>(in_dp,sg,(Signal*)sg_buf,result,rank,nvec); break;
            case 8: allreduce2_tdm<8,TILE><<<nb,TH,8*TILE*16>>>(in_dp,sg,(Signal*)sg_buf,result,rank,nvec); break;
            default: if(rank==0) printf("unsupported world=%d\n",world);
        } } else { switch(world){
            case 2: allreduce2<2><<<nb,TH>>>(in_dp,sg,(Signal*)sg_buf,result,rank,nvec); break;
            case 4: allreduce2<4><<<nb,TH>>>(in_dp,sg,(Signal*)sg_buf,result,rank,nvec); break;
            case 8: allreduce2<8><<<nb,TH>>>(in_dp,sg,(Signal*)sg_buf,result,rank,nvec); break;
            default: if(rank==0) printf("unsupported world=%d\n",world);
        } }
    };
    (void)tdm_lds;
    barrier(); launch(); CK(hipDeviceSynchronize()); barrier();   // correctness run
    { std::vector<float> h(nfloat); CK(hipMemcpy(h.data(),result,bytes,hipMemcpyDeviceToHost));
      float exp=(float)world*(world+1)/2.0f; size_t bad=0; for(size_t j=0;j<nfloat;j++) if(h[j]!=exp)++bad;
      printf("[rank %d] allreduce correctness: %s (expect %.0f, mism=%zu/%zu)\n",rank,bad?"FAIL":"PASS",exp,bad,nfloat); }

    // ---- benchmark ----
    int WARM=20,LOOP=100;
    for(int i=0;i<WARM;i++) launch(); CK(hipDeviceSynchronize()); barrier();
    auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<LOOP;i++) launch(); CK(hipDeviceSynchronize());
    auto t1=std::chrono::high_resolution_clock::now(); barrier();
    double us=std::chrono::duration<double,std::micro>(t1-t0).count()/LOOP;
    // busbw convention (nccl): 2*(N-1)/N * size / time
    double busbw=2.0*(world-1)/world*(double)bytes/(us/1e6)/1e9;
    double algbw=(double)bytes/(us/1e6)/1e9;
    const char* mode=use_opt?"opt":(use_tdm?"TDM":"naive");
    if(rank==0) printf("\n=== allreduce world=%d, %dMB, %s(U=%d), blk=%d th=%d: %.1f us/iter, algbw %.1f GB/s, busbw %.1f GB/s ===\n",world,mb,mode,use_opt?uunroll:0,nb,TH,us,algbw,busbw);
    barrier();

    if(use_prof){
        unsigned long long* d_prof; CK(hipMalloc(&d_prof,4*sizeof(unsigned long long))); CK(hipMemset(d_prof,0,4*sizeof(unsigned long long)));
        barrier();
        switch(world){
            case 2: allreduce2_prof<2><<<nb,TH>>>(in_dp,sg,(Signal*)sg_buf,result,rank,nvec,d_prof); break;
            case 4: allreduce2_prof<4><<<nb,TH>>>(in_dp,sg,(Signal*)sg_buf,result,rank,nvec,d_prof); break;
            case 8: allreduce2_prof<8><<<nb,TH>>>(in_dp,sg,(Signal*)sg_buf,result,rank,nvec,d_prof); break;
        }
        CK(hipDeviceSynchronize()); barrier();
        unsigned long long h_prof[4]; CK(hipMemcpy(h_prof,d_prof,4*sizeof(unsigned long long),hipMemcpyDeviceToHost));
        double tot=h_prof[0]+h_prof[1]+h_prof[2]+h_prof[3];
        printf("[rank %d] PROF cycles: start_sync=%llu (%.0f%%) reduce=%llu (%.0f%%) end_sync=%llu (%.0f%%) allgather=%llu (%.0f%%)\n",
            rank,h_prof[0],100*h_prof[0]/tot,h_prof[1],100*h_prof[1]/tot,h_prof[2],100*h_prof[2]/tot,h_prof[3],100*h_prof[3]/tot);
    }
    return 0;
}
