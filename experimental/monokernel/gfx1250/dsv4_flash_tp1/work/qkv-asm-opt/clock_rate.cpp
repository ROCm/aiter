#include <hip/hip_runtime_api.h>
#include <cstdio>
int main(){int khz=0;auto e=hipDeviceGetAttribute(&khz,hipDeviceAttributeWallClockRate,0);printf("error=%d wall_clock_khz=%d attribute=%d\n",int(e),khz,int(hipDeviceAttributeWallClockRate));return int(e);}
