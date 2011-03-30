// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "inc.h"

float * GenVectors(int numVectors, int numDims){
        srand(10);
        float * buf=(float*)malloc(sizeof(float)*numVectors*numDims);
        for(int i=0;i<numVectors*numDims;i++)
                buf[i]=rand()%100;
        return buf;
}

__global__ void
similarity_kernel( float* src, float* tgt, int num, int num_dims)
{
        int tx = threadIdx.x;
	int bx = blockIdx.x;
	int bd = blockDim.x;
	for (int i = tx; i < num; i+=bd) {
		int idx = bx * num + i;
                const float * a=src + bx*num_dims;
                const float * b=src + i*num_dims;
                float ab=0;
                float aa=0;
                float bb=0;
                float ai,bi;
                for(int i=0;i<num_dims;i++){
                        ai=a[i];
                        bi=b[i];
                        ab+=ai*bi;
                        aa+=ai*ai;
                        bb+=bi*bi;
                }
                tgt[idx]=sqrt((ab*ab)/(aa*bb));
        }
}
int
main(int argc, char** argv)
{
        cudaSetDevice( 0 );

	double t1=get_time();

        int num_points;
        int num_dims;

        if(!get_opt(argc,argv,"np", num_points) || !get_opt(argc,argv,"nd",num_dims)){
                return 1;
        }

        float * points=GenVectors(num_points, num_dims);

        unsigned int mem_size = sizeof(float)*num_points*num_dims;
	unsigned int mem_size_tgt = sizeof(float) * num_points * num_points;

    	float* d_src;
    	cutilSafeCall(cudaMalloc((void**) &d_src, mem_size));

    	cutilSafeCall(cudaMemcpy(d_src, points, mem_size,
                              cudaMemcpyHostToDevice) );
	float* result = (float *)malloc(mem_size_tgt);

        float* d_tgt;
        cutilSafeCall(cudaMalloc((void**) &d_tgt, mem_size_tgt));


	double t2=get_time();
	printf("init: %lf\n", t2-t1);

    	dim3 threads(128);
   	dim3 grid(num_points);
	similarity_kernel<<< grid, threads >>>(d_src, d_tgt, num_points, num_dims);
        double t3=get_time();
        printf("computation time: %lf\n", t3-t2);

    	cutilSafeCall(cudaMemcpy(result, d_tgt, mem_size_tgt,
                              cudaMemcpyDeviceToHost) );

        return 0;
}

