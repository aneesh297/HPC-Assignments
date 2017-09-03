#include <stdio.h>
#include <math.h>

#define N (16*1024)
#define THREADS_PER_BLOCK 512.0

void random_floats(float *a,int n){
    int i;
    float maxVal = 5.0;
    for(i=0;i<n;i++){
        a[i] = ((float)rand()/(float)RAND_MAX)*maxVal;
    }
}

__global__ void sum(float *inp,float* blockSums)
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

    if(lindex==0) blockSums[blockIdx.x]=intraBlock[0];
}

int checkSum(float *a,float calcSum)
{
    float s=0;
    int i;
    for(i=0;i<N;i++) s+=a[i];
    printf("Sequential Sum = %f\n",s);
    if(abs(s-calcSum)>0.1) return 0;
    else return 1;
}

int main(){
    float *a,*blockSums;

    int size = N*sizeof(float);
    int no_of_blocks = ceil(N/THREADS_PER_BLOCK);

    a=(float*)malloc(size);
    blockSums = (float*)malloc(no_of_blocks*sizeof(float));
    random_floats(a,N);

    float *d_a,*d_blockSums;
    
    cudaMalloc((void**)&d_a,size);
    cudaMalloc((void**)&d_blockSums,no_of_blocks*sizeof(float));

    cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
    
    sum<<<no_of_blocks,THREADS_PER_BLOCK>>>(d_a,d_blockSums);

    cudaMemcpy(blockSums,d_blockSums,no_of_blocks*sizeof(float),cudaMemcpyDeviceToHost);

    float finSum=0;
    int i;
    for(i=0;i<no_of_blocks;i++) finSum+=blockSums[i];

    printf("Final Sum = %f\n",finSum);

    checkSum(a,finSum);
}

        

    

