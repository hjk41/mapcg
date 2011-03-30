// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "inc.h"

const int mode=1024;
struct body_t{
        float pos_x;
        float pos_y;
        float pos_z;
        float vel_x;
        float vel_y;
        float vel_z;
        float mass;
};

typedef float3 acc_t;

typedef int task_t;

body_t * gen_body(unsigned int n){
        srand(1000);
        body_t * b=new body_t[n];
        for(int i=0;i<n;i++){
                b[i].pos_x=rand()%mode;
                b[i].pos_y=rand()%mode;
                b[i].pos_z=rand()%mode;
                b[i].mass=rand();
                b[i].vel_x=0;
                b[i].vel_y=0;
                b[i].vel_z=0;
        }
        return b;
}


__global__ void
nbody_kernel( body_t* src, body_t* tgt, int num)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
		const float G=9.8;
		const float max_dist=10000;
		const float soften=0.0001;
		const float delta=0.0001;

                body_t body1=src[idx];
                acc_t total_acc;
                total_acc.x=0;
                total_acc.y=0;
                total_acc.z=0;
                for(int y=0;y<num;y++){
                        if(idx==y)
                                continue;
                        const body_t body2=src[y];
                        // calculate r/(r^2+soften)^1.5
                        float r1=body1.pos_x-body2.pos_x;
                        float r2=body1.pos_y-body2.pos_y;
                        float r3=body1.pos_z-body2.pos_z;

                        float rr=r1*r1+r2*r2+r3*r3;
                        // ignore bodies that are too far
                        if(rr>=max_dist*max_dist)
                                continue;
                        float r_soften=rr+soften;
                        r_soften=sqrt(r_soften);
                        r_soften=r_soften*r_soften*r_soften;
                        float invert=1/r_soften;

                        r1*=invert;
                        r2*=invert;
                        r3*=invert;
                        total_acc.x+=0-r1*body2.mass;
                        total_acc.y+=0-r2*body2.mass;
                        total_acc.z+=0-r3*body2.mass;
                }

                // update pos and speed
                total_acc.x*=G;
                total_acc.y*=G;
                total_acc.z*=G;
                body1.vel_x+=total_acc.x*delta;
                body1.vel_y+=total_acc.y*delta;
                body1.vel_z+=total_acc.z*delta;

                body1.pos_x+=delta*body1.vel_x;
                body1.pos_y+=delta*body1.vel_y;
                body1.pos_z+=delta*body1.vel_z;

		tgt[idx] = body1;
}
int
main(int argc, char** argv)
{
        cudaSetDevice( 0 );

	double t1=get_time();

        int num_body;
        if(!get_opt(argc,argv,"n",num_body)){
                return 1;
        }

        unsigned int mem_size_A = num_body * sizeof(body_t);

	body_t * bodies=gen_body(num_body);

    	body_t* d_A;
    	cutilSafeCall(cudaMalloc((void**) &d_A, mem_size_A));
        body_t* d_B;
        cutilSafeCall(cudaMalloc((void**) &d_B, mem_size_A));

    	cutilSafeCall(cudaMemcpy(d_A, bodies, mem_size_A,
                              cudaMemcpyHostToDevice) );


	double t2=get_time();
	printf("init: %lf\n", t2-t1);

	int total = 1;
	for (int i = 0; i < total; i++) {
	    	dim3 threads(128);
    		dim3 grid(num_body / 128);

		if (i / 2 * 2 == i)
		    	nbody_kernel<<< grid, threads >>>(d_A, d_B, num_body);
		else
			nbody_kernel<<< grid, threads >>>(d_B, d_A, num_body);
	}

	if (total / 2 * 2 == total)
	    	cutilSafeCall(cudaMemcpy(bodies, d_A, mem_size_A,
                              cudaMemcpyDeviceToHost) );
	else
                cutilSafeCall(cudaMemcpy(bodies, d_B, mem_size_A,
                              cudaMemcpyDeviceToHost) );

	double t3=get_time();
	printf("computation time: %lf\n", t3-t2);

	for(int i=0;i<10 && i<num_body;i++){
		printf("%f,%f,%f\n", bodies[i].pos_x, bodies[i].pos_y, bodies[i].pos_z);
	}

        return 0;
}

