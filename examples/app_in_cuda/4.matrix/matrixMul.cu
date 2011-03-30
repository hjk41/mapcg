// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "inc.h"

static float *GenMatrix(int M_ROW_COUNT, int M_COL_COUNT)
{
        float *matrix = (float*)malloc(sizeof(float)*M_ROW_COUNT*M_COL_COUNT);

        srand(10);
        for (int i = 0; i < M_ROW_COUNT; i++)
                for (int j = 0; j < M_COL_COUNT; j++)
                        matrix[i*M_COL_COUNT+j] = (float)(rand() % 100);

        return matrix;
}

__global__ void
matrixMul( float* C, float* A, float* B, int dim)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int dx = blockIdx.x;
  int dy = blockIdx.y;

  C[(dy*8+ty) * dim + dx*8 + tx] = 0;
  for (int i=0; i < dim; i++) {
    C[(dy*8+ty) * dim + dx*8 + tx] += A[ (dy*8+ty) * dim + i] * B[i * dim + dx * 8 +tx];
  }
}
int
main(int argc, char** argv)
{
        cudaSetDevice( 0 );

	double t1=get_time();

        int dim;
        if(!get_opt(argc,argv,"dim",dim)){
                return 1;
        }

        unsigned int mem_size_A = dim * dim * sizeof(float);
        unsigned int mem_size_B = dim * dim * sizeof(float);

        float * matrixA=GenMatrix(dim,dim);
        float * matrixB=GenMatrix(dim,dim);
    	float* d_A;
    	cutilSafeCall(cudaMalloc((void**) &d_A, mem_size_A));
    	float* d_B;
    	cutilSafeCall(cudaMalloc((void**) &d_B, mem_size_B));

    	cutilSafeCall(cudaMemcpy(d_A, matrixA, mem_size_A,
                              cudaMemcpyHostToDevice) );
    	cutilSafeCall(cudaMemcpy(d_B, matrixB, mem_size_B,
                              cudaMemcpyHostToDevice) );

    	unsigned int size_C = dim * dim;
    	unsigned int mem_size_C = sizeof(float) * size_C;
    	float* d_C;
    	cutilSafeCall(cudaMalloc((void**) &d_C, mem_size_C));

    	float* h_C = (float*) malloc(mem_size_C);

	double t2=get_time();
	printf("init: %lf\n", t2-t1);

    	dim3 threads(8, 8);
    	dim3 grid(dim / 8, dim / 8);

    	matrixMul<<< grid, threads >>>(d_C, d_A, d_B, dim);

    	cutilSafeCall(cudaMemcpy(h_C, d_C, mem_size_C,
                              cudaMemcpyDeviceToHost) );
        printf("%f\n", h_C[0]);

	double t3=get_time();
	printf("computation time: %lf\n", t3-t2);
        return (int)h_C[0];
}

