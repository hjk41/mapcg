// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "inc.h"
#define VECTOR_SPACE    1000

int *GenPoints(int num_points, int dim)
{
        srand(1024);
        int *buf = (int*)malloc(sizeof(int)*num_points*dim);

        for (int i = 0; i < num_points; i++)
                for (int j = 0; j < dim; j++)
                        buf[i*dim+j] = rand()%VECTOR_SPACE;
//                      buf[i*dim+j] = (i*dim+j)%VECTOR_SPACE;

        return buf;
}

int *GenMeans(int num_means, int dim)
{
        srand(1024);
        int *buf = (int*)malloc(dim*num_means*sizeof(int));
        for (int i = 0; i < num_means; i++)
                for (int j = 0; j < dim; j++)
//                      buf[i*dim+j] = (i*dim+j)%VECTOR_SPACE;
                        buf[i*dim + j] = rand()%VECTOR_SPACE;
        return buf;
}

__device__ unsigned int get_sq_dist(int* point, int* mean, int dim)
{
        int i;
        unsigned int sum = 0;
        for(i = 0; i < dim; i++)
        {
                sum += (point[i] - mean[i])*(point[i] - mean[i]);
        }
        return sum;
}


__global__ void
kmeans_kernel(int* points, int* means, int *o_means, int *o_count, int *o_points, int num, int dim)
{
	int tx = threadIdx.x;
	int dx = blockIdx.x;
	int idx = dx * blockDim.x + tx;
	int *point = points + dim * idx;
        unsigned int min_dist, cur_dist;
        min_dist = get_sq_dist(point, means, dim);
        int min_idx = 0;
        for(int j = 1; j < num; j++)
        {
                cur_dist = get_sq_dist(point, means + j * dim, dim);
                if(cur_dist < min_dist)
                {
                        min_dist = cur_dist;
                        min_idx = j;
                }
        }
	o_points[idx] = min_idx;
	for (int j = 0; j < dim; j++)
	{
		atomicAdd(o_means + min_idx * dim + j, point[j]);
	}
	atomicAdd(o_count + min_idx, 1);
}

__global__ void
kmeans_set_zero(int* means)
{
	means[blockIdx.x * blockDim.x + threadIdx.x] = 0;
}

__global__ void
kmeans_set_zero_count(int *counts)
{
	counts[blockIdx.x] = 0;
}

__global__ void
kmeans_average(int* means, int* counts)
{
	if (counts[blockIdx.x] == 0)
		means[blockIdx.x * blockDim.x + threadIdx.x] = 0;
	else
		means[blockIdx.x * blockDim.x + threadIdx.x] /= counts[blockIdx.x];
}

int
main(int argc, char** argv)
{
        cudaSetDevice( 0 );

	double t1=get_time();

        int num_points, num_means, dim, num_iters;
        if (!get_opt(argc, argv, "vn", num_points) ||
         !get_opt(argc, argv, "mn", num_means) ||
         !get_opt(argc, argv, "dim", dim) ||
        !get_opt(argc,argv, "ni", num_iters)){
                 printf("Usage: %s -vn INT -mn INT -dim INT\n"
                        "\t-vn vector number\n"
                        "\t-mn mean number\n"
                        "\t-dim vector dimension\n"
                        "\t-ni number iteration\n", argv[0]);
                return 1;
        }
        int * points=GenPoints(num_points, dim);
        int * means=GenMeans(num_means, dim);

        unsigned int mem_size_points = sizeof(int)*num_points*dim;
	unsigned int mem_size_means = dim*num_means*sizeof(int);
	unsigned int mem_size_out_points = num_points * sizeof(int);

    	int* d_points;
    	cutilSafeCall(cudaMalloc((void**) &d_points, mem_size_points));
        int* d_means;
        cutilSafeCall(cudaMalloc((void**) &d_means, mem_size_means));

    	cutilSafeCall(cudaMemcpy(d_points, points, mem_size_points,
                              cudaMemcpyHostToDevice) );
        cutilSafeCall(cudaMemcpy(d_means, means, mem_size_means,
                              cudaMemcpyHostToDevice) );

	int* d_out_points;
        cutilSafeCall(cudaMalloc((void**) &d_out_points, mem_size_out_points));
        int* d_out_counts;
        cutilSafeCall(cudaMalloc((void**) &d_out_counts, sizeof(int) * num_means));

	int* out_points = (int *)malloc(mem_size_out_points);
	int* out_means = (int *)malloc(mem_size_means);


	int* d_out_means;
        cutilSafeCall(cudaMalloc((void**) &d_out_means, mem_size_means));

	double t2=get_time();
	printf("init: %lf\n", t2-t1);

	int* counters = (int *)malloc (sizeof(int) * num_means);

	int changed = 1, i = 0;
	while (changed && i < num_iters) {
		dim3 threads(128);
		dim3 grid(num_points / 128);
		kmeans_set_zero_count<<< num_means, 1>>> (d_out_counts);

		if (i / 2 * 2 == i) {
			kmeans_set_zero<<< num_means, dim>>> (d_out_means);
		    	kmeans_kernel<<< grid, threads >>>(d_points, d_means, d_out_means, d_out_counts, d_out_points, num_means, dim);
	                kmeans_average<<< num_means, dim >>>(d_out_means, d_out_counts);
	                cutilSafeCall(cudaMemcpy(out_means, d_out_means, mem_size_means, cudaMemcpyDeviceToHost) );
		}
		else {
			kmeans_set_zero<<< num_means, dim>>> (d_means);
                        kmeans_kernel<<< grid, threads >>>(d_points, d_out_means, d_means, d_out_counts, d_out_points, num_means, dim);
	                kmeans_average<<< num_means, dim >>>(d_means, d_out_counts);
                        cutilSafeCall(cudaMemcpy(means, d_means, mem_size_means, cudaMemcpyDeviceToHost) );
		}
/*		cutilSafeCall(cudaMemcpy(counters, d_out_counts, sizeof(int)*num_means, cudaMemcpyDeviceToHost) );
		for (int j = 0; j < num_means; j++)
		{	printf ("%d(", counters[j]);
			for (int k = 0; k < dim; k++)
				printf ("%d ", i / 2 * 2 == i?means[j * dim + k]:out_means[j * dim + k]);
			printf (")\t");
		}
		printf ("\n");*/
		changed = memcmp (means, out_means, mem_size_means);
		i++;
	}

	cutilSafeCall(cudaMemcpy(out_points, d_out_points, mem_size_out_points,
                              cudaMemcpyDeviceToHost) );

	double t3=get_time();
	printf("computation time: %lf\n", t3-t2);
        return 0;
}

