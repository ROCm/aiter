#include <hip/hip_runtime.h>
#include <cstdio>
int main(){int rate=0;auto e=hipDeviceGetAttribute(&rate,hipDeviceAttributeWallClockRate,0);printf("%d %d\n",int(e),rate);return int(e);}
