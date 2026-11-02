#include <hip/hip_runtime.h>
#include <stdint.h>
extern __shared__ unsigned char __dynamic_shared_0[];
extern "C" __device__ __attribute__((always_inline,used)) unsigned sw4_epoch(uint64_t address){
 auto *p=(unsigned*)address+blockIdx.x;
 unsigned e=p[0]+1; __syncthreads(); if(threadIdx.x==0)p[0]=e; return __builtin_amdgcn_readfirstlane(e);
}
extern "C" __device__ __attribute__((always_inline,used)) void sw4_put(uint64_t address,unsigned index,unsigned epoch){
 asm volatile("s_wait_loadcnt 0\n s_wait_storecnt 0\n s_wait_dscnt 0\n s_wait_asynccnt 0\n s_wait_tensorcnt 0" ::: "memory");
 __syncthreads();
 if(threadIdx.x==0){auto*p=(unsigned*)address+index;asm volatile("global_store_b32 %0, %1, off scope:SCOPE_DEV"::"v"(p),"v"(epoch):"memory");}
}
extern "C" __device__ __attribute__((always_inline,used)) void sw4_get(uint64_t address,uint64_t health,unsigned start,unsigned count,unsigned stride,unsigned epoch,unsigned site){
 auto*p=(unsigned*)address;
 for(unsigned i=threadIdx.x;i<count;i+=128){
  unsigned seen,n=0;
  do{
   asm volatile("global_load_b32 %0, %1, off scope:SCOPE_DEV th:TH_LOAD_NT\n s_wait_loadcnt 0":"=v"(seen):"v"(p+start+i*stride):"memory");
   if(seen==epoch)break;
   asm volatile("s_sleep 1");
   if(++n==1000000u){atomicExch((unsigned*)health,site+1);break;}
  }while(true);
 }
 __syncthreads();
 asm volatile("s_wait_loadcnt 0\n global_inv scope:SCOPE_DEV\n s_wait_loadcnt 0" ::: "memory");
}
extern "C" __device__ __attribute__((always_inline,used)) void sw4_local(){
 asm volatile("s_wait_loadcnt 0\n s_wait_storecnt 0\n s_wait_dscnt 0\n s_wait_tensorcnt 0" ::: "memory");
 __syncthreads();
}
extern "C" __device__ __attribute__((always_inline,used)) void sw4_coeff(uint64_t positions,uint64_t cos,uint64_t sin,unsigned row,unsigned maxpos){
 if(threadIdx.x<32){
  int64_t pos=((int64_t*)positions)[row];pos=pos<0?0:pos;pos=pos>=maxpos?maxpos-1:pos;
  unsigned lane=threadIdx.x;
  uint64_t addr=(lane<16?cos:sin)+pos*64+(lane%16)*4;
  unsigned dst=(unsigned)(uintptr_t)__dynamic_shared_0+266240+lane*4;
  asm volatile("global_load_async_to_lds_b32 %0, %1, off"::"v"(dst),"v"(addr):"memory");
 }
}
