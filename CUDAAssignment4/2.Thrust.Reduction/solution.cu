#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "wb.h"

float* readInput(char* fileName,int *len){
    FILE *fp = fopen(fileName,"r");
    fscanf(fp,"%d",len);

    float* inp = (float*)malloc(sizeof(float)*(*len));


    for(int i=0;i<(*len);i++) fscanf(fp,"%f",&inp[i]);
    fclose(fp);

    return inp;
}
float readOutput(char* fileName){
    FILE *fp = fopen(fileName,"r");
    int len;
    fscanf(fp,"%d",&len);

    float out;
    fscanf(fp,"%f",&out);

    return out;
}
int main(int argc, char *argv[]) {
  wbArg_t args;
  float *hostInput;
  float expectedOutput;
  int inputLength;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  // Import host input data
  wbTime_start(Generic, "Importing data to host");
  hostInput = readInput(wbArg_getInputFile(args, 0), &inputLength);
  expectedOutput = readOutput(wbArg_getInputFile(args,1));

  thrust::host_vector<float> hostVectorInput(hostInput,hostInput+inputLength);
  wbTime_stop(Generic, "Importing data to host");

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
  // Declare and allocate thrust device input and output vectors
  wbTime_start(GPU, "Doing GPU memory allocation");
  thrust::device_vector<float> deviceVectorInput(inputLength,0.0f);
  wbTime_stop(GPU, "Doing GPU memory allocation");

  // Copy to device
  wbTime_start(Copy, "Copying data to the GPU");
  thrust::copy(hostVectorInput.begin(),hostVectorInput.end(),deviceVectorInput.begin());
  wbTime_stop(Copy, "Copying data to the GPU");

  // Execute vector addition
  wbTime_start(Compute, "Doing the computation on the GPU");
  float sum = thrust::reduce(deviceVectorInput.begin(), deviceVectorInput.end(),
                            0.0f, thrust::plus<float>());
  wbTime_stop(Compute, "Doing the computation on the GPU");
  /////////////////////////////////////////////////////////

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  printf("Expected Output = %0.2f\n",expectedOutput);
  printf("Obtained Output = %0.2f\n",sum);

  free(hostInput);
  return 0;
}
