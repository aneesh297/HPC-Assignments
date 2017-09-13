//Execution order ./a.out input.raw output.raw
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include "wb.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

#define BLOCK_SIZE 512 //@@ You can change this
#define THREADS_PER_BLOCK 512.0

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)


/*
__global__ void total(float *input, float *output, int len) {

  __shared__ float partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
    if (start + t < len)
       partialSum[t] = input[start + t];
    else
       partialSum[t] = 0;
    if (start + BLOCK_SIZE + t < len)
       partialSum[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
    else
       partialSum[BLOCK_SIZE + t] = 0;
    //@@ Traverse the reduction tree
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          partialSum[t] += partialSum[t+stride];
    }
    //@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
    if (t == 0)
       output[blockIdx.x] = partialSum[0];

}
*/

  __global__ void total(float *inp,float* output, int N)
{
    __shared__ float intraBlock[512];

    int lindex = threadIdx.x;
    int gindex = blockIdx.x*blockDim.x + threadIdx.x;

    if(gindex<N) intraBlock[lindex]=inp[gindex];

    __syncthreads();

    if(lindex<256 && gindex+256<N) intraBlock[lindex]+=intraBlock[lindex+256];
    __syncthreads();

    if(lindex<128 && gindex+128<N) intraBlock[lindex]+=intraBlock[lindex+128];
    __syncthreads();

    if(lindex<64 && gindex+64<N) intraBlock[lindex]+=intraBlock[lindex+64];
    __syncthreads();

    if(lindex<32 && gindex+32<N) intraBlock[lindex]+=intraBlock[lindex+32];
    __syncthreads();

    if(lindex<16 && gindex+16<N) intraBlock[lindex]+=intraBlock[lindex+16];
    __syncthreads();

    if(lindex<8 && gindex+8<N) intraBlock[lindex]+=intraBlock[lindex+8];
    __syncthreads();

    if(lindex<4 && gindex+4<N) intraBlock[lindex]+=intraBlock[lindex+4];
    __syncthreads();

    if(lindex<2 && gindex+2<N) intraBlock[lindex]+=intraBlock[lindex+2];
    __syncthreads();

    if(lindex<1 && gindex+1<N) intraBlock[lindex]+=intraBlock[lindex+1];
    __syncthreads();

    if(lindex==0) output[blockIdx.x]=intraBlock[0];
}


  

int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list


  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
int no_of_blocks = ceil(numInputElements/THREADS_PER_BLOCK);

  //numOutputElements = 100;    

  hostOutput = (float *)calloc(no_of_blocks, sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numInputElements);
  wbLog(TRACE, "The number of output elements in the input is ",
        numOutputElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**)&deviceInput,sizeof(float) * numInputElements);
  cudaMalloc((void**)&deviceOutput,sizeof(float) * no_of_blocks);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceInput,hostInput,sizeof(float) * numInputElements, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceOutput,hostOutput,sizeof(float) * no_of_blocks, cudaMemcpyHostToDevice);
  //@@ Copy memory to the GPU here
  wbTime_stop(GPU, "Copying input memory to the GPU.");
  //@@ Initialize the grid and block dimensions here
  dim3 DimBlock(BLOCK_SIZE);
  dim3 DimGrid((numInputElements - 1)/BLOCK_SIZE + 1);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  total<<<no_of_blocks,THREADS_PER_BLOCK>>>(deviceInput, deviceOutput, numInputElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput,deviceOutput,sizeof(float) * no_of_blocks, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  /********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input. For simplicity, we do not
   * require that for this lab.
   ********************************************************************/

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  float finSum = 0;
  //wbSolution(args, hostOutput, 1);
  for(int i=0;i<no_of_blocks;i++) finSum+=hostOutput[i];

  cout<<"Output: "<<finSum<<endl;
  string output(argv[2]);
  FILE *handle = fopen(output.c_str(),"r");
  float x,y;
  fscanf(handle, "%f %f",&x,&y);
  cout<<"Expected Output: "<<y<<endl;


  free(hostInput);
  free(hostOutput);

  return 0;
}
