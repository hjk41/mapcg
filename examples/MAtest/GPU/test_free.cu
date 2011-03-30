#include <cuda.h>
#include <sys/time.h>
#include <iostream>
//#include "HMMSMA.h"
#include "MA_builtin.h"
using namespace std;
using namespace HMM_GPU;

double get_time(){
        cudaThreadSynchronize();
        timeval t;
        gettimeofday(&t, NULL);
        return (double)t.tv_sec+(double)t.tv_usec/1000000;
}

__global__ void foo(int num_alloc, int blocksize, void ** ptr_array){
	SMA_Start_Kernel();
	void ** pa=ptr_array+blockDim.x*blockIdx.x+threadIdx.x;
	for(int i=0;i<num_alloc;i++){
		void * ptr=SMA_Malloc(blocksize);
		*pa=ptr;
		pa=(void**)ptr;
		if(!ptr){
			if(threadIdx.x==0 && blockIdx.x==0)
				printf("failed, i=%d\n", i);
			break;
		}
	}
}

__global__ void bar(int num_alloc, void ** ptr_array){
	int threadID=blockDim.x*blockIdx.x+threadIdx.x;
	void * p=ptr_array[threadID];
	void * pnext=*(void**)p;
	for(int i=0;i<num_alloc-1;i++){
		SMA_Free(p);
		p=pnext;
		pnext=*(void**)p;
	}
	SMA_Free(p);
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
	int totalsize=griddim*blockdim*n_blocks*blocksize;

	cudaSetDevice(0);

	SMA_Init(totalsize+10*1024*1024);
	// allocate the pointer array
	void ** ptr_array;
	cudaMalloc((void**)&ptr_array, griddim*blockdim*sizeof(void*));
	double t1=get_time();
	foo<<<griddim,blockdim>>>(n_blocks, blocksize, ptr_array);
	cudaThreadSynchronize();	
	CUT_CHECK_ERROR("foo");
	double t2=get_time();
	bar<<<griddim,blockdim>>>(n_blocks,ptr_array);
	SMA_Destroy();
	CUT_CHECK_ERROR("bar");
	double t3=get_time();
	cout<<"    malloc: "<<t2-t1<<"\t free: "<<t3-t2<<endl;

	return 0;
}
