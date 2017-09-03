#include <cuda.h>
#include <iostream>
#include <stdio.h>

#define N 1024
int a[N][N],b[N][N],c[N][N];
using namespace std;

__global__ void addMatrix(int a[][N], int b[][N], int c[][N], int n){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row<n && col<n)
		c[row][col] = a[row][col] + b[row][col];
}

void random_int(int a[][N], int n)
{
   int i,j;
   for (i = 0; i < n; ++i)
   	for (j = 0; j < n; ++j)
    	a[i][j] = rand() % 101;

}

int main(void)
{
	
	int (*pA)[N], (*pB)[N], (*pC)[N];

	random_int(a,N);
	random_int(b,N);

	cudaMalloc((void**)&pA, (N*N)*sizeof(int));
	cudaMalloc((void**)&pB, (N*N)*sizeof(int));
	cudaMalloc((void**)&pC, (N*N)*sizeof(int));

	cudaMemcpy(pA, a, (N*N)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(pB, b, (N*N)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(pC, c, (N*N)*sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimBlock(64, 64);
    dim3 dimGrid(N/dimBlock.x, N/dimBlock.y);

    addMatrix<<<dimGrid,dimBlock>>>(pA,pB,pC,N);

    cudaMemcpy(c, pC, (N*N)*sizeof(int), cudaMemcpyDeviceToHost);

	int i, j;
	/*
	printf("C = \n");
	for(i=0;i<N;i++){
	    for(j=0;j<N;j++){
	        printf("%d ", c[i][j]);
	    }
	    printf("\n");
	}
	*/

	cudaFree(pA); 
	cudaFree(pB); 
	cudaFree(pC);

	printf("\n");

	return 0;
}
