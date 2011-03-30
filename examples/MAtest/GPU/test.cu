#include <cuda.h>
#include <sys/time.h>
#include <iostream>
#include "HMMSMA.h"
//#include "MA_builtin.h"
using namespace std;
using namespace HMM_GPU;

double get_time(){
        cudaThreadSynchronize();
        timeval t;
        gettimeofday(&t, NULL);
        return (double)t.tv_sec+(double)t.tv_usec/1000000;
}

__global__ void foo(int num_alloc, int blocksize){
	SMA_Start_Kernel();
	for(int i=0;i<num_alloc;i++){
		void * ptr=SMA_Malloc(blocksize);
		if(!ptr){
			if(threadIdx.x==0 && blockIdx.x==0)
				printf("failed, i=%d\n", i);
			break;
		}
	}
}

int main(int argc, char ** argv){
	if(argc!=5){
		cout<<"usage: "<<argv[0]<<" n_blocks blocksize griddim blockdim"<<endl;
		return 1;
	}
	int n_blocks=atoi(argv[1]);
	int blocksize=atoi(argv[2]);
	int griddim=atoi(argv[3]);
	int blockdim=atoi(argv[4]);
	int totalsize=griddim*blockdim*blocksize*n_blocks;	

	cudaSetDevice(0);

	double t1=get_time();
	SMA_Init(totalsize+10*1024*1024);
	foo<<<griddim,blockdim>>>(n_blocks, blocksize);
	cudaThreadSynchronize();	
	CUT_CHECK_ERROR("foo");
	double t2=get_time();

	cout<<"time: "<<t2-t1<<endl;

	return 0;
}
