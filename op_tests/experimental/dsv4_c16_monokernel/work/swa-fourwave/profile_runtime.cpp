#include "runtime.cpp"
extern "C" __device__ __attribute__((always_inline,used)) uint64_t sw4_stamp(){
 asm volatile("" ::: "memory");
 uint64_t t=wall_clock64();
 asm volatile("" ::: "memory"); return t;
}
extern "C" __device__ __attribute__((always_inline,used)) uint64_t sw4_get_profile(uint64_t address,uint64_t health,unsigned start,unsigned count,unsigned stride,unsigned epoch,unsigned site){
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
 uint64_t ready=sw4_stamp();
 asm volatile("s_wait_loadcnt 0\n global_inv scope:SCOPE_DEV\n s_wait_loadcnt 0" ::: "memory");
 return ready;
}
extern "C" __device__ __attribute__((always_inline,used)) void sw4_trace_quiet(uint64_t start,uint64_t ticks){
 // Poll only the local cycle counter: repeated GET_REALTIME messages can
 // themselves congest the global timer service and delay measured timestamps.
 uint64_t c=clock64();
 while(clock64()-c<ticks*50)asm volatile("s_sleep 1");
}
extern "C" __device__ __attribute__((always_inline,used)) void sw4_trace_store(uint64_t ptr,unsigned n,unsigned index,uint64_t value){
 if((threadIdx.x&31)==0)((uint64_t*)ptr)[(blockIdx.x*4+threadIdx.x/32)*n+index]=value;
}
