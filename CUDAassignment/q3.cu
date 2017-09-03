#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N (16*1024*1024)
#define THREADS_PER_BLOCK 512.0

__global__ void add(float *a,float *b,float *c){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N) c[i]=a[i]+b[i];
}

void random_floats(float *a,int n){
    int i;
    float maxVal = 5.0;
    for(i=0;i<n;i++)
        a[i] = ((float)rand()/(float)RAND_MAX)*maxVal;
}
int checkAdd(float *a,float *b,float *c)
{
    int i;
    for(i=0;i<N;i++){
        if(c[i]!=a[i]+b[i]) return 0;
    }
    return 1;
}
int main(){
    float *a,*b,*c;
    int size = sizeof(float)*N;

    a=(float*)malloc(size);
    b=(float*)malloc(size);
    c=(float*)malloc(size);

    random_floats(a,N);
    random_floats(b,N);

    float *d_a,*d_b,*d_c;
    cudaMalloc((void**)&d_a,size);
    cudaMalloc((void**)&d_b,size);
    cudaMalloc((void**)&d_c,size);

    cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,size,cudaMemcpyHostToDevice);

    add<<<ceil(N/THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(d_a,d_b,d_c);

    cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost);

    if(checkAdd(a,b,c)) printf("Addition Verified\n");
    else printf("Something went wrong!");

    free(a);free(b);free(c);
    cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);

    return 1;
}
