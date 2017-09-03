#include <stdio.h>
#include <cuda.h>

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("Device name: %s\n", prop.name);
    printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("Shared memory available per block in bytes: %d", prop.sharedMemPerBlock);
    printf("\nMaximum size of each dimension of a grid: %ld\n", prop.maxGridSize);
    printf("Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Maximum size of each dimension of a block: %ld\n", prop.maxThreadsDim);
    printf("Constant memory available on device in bytes: %d\n", prop.totalConstMem);
    printf("Global memory available on device in bytes: %ld\n", prop.totalGlobalMem);
    printf("Major compute capability: %d\n",prop.major);
    printf("Minor compute capability: %d\n",prop.minor);

  }
}
