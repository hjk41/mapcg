#include <omp.h>
#include <iostream>
#include <sys/time.h>
#include "MA_Builtin.cxx"
#include <stdlib.h>
using namespace std;
using namespace HMM_CPU;

double get_time(){
        timeval t;
        gettimeofday(&t, NULL);
        return (double)t.tv_sec+(double)t.tv_usec/1000000;
}

int main(int argc, char ** argv){

	if(argc!=4){
		cout<<"usage: "<<argv[0]<<" n_blocks blocksize n_threads"<<endl;
		return 1;
	}
	int n_blocks=atoi(argv[1]);
	int blocksize=atoi(argv[2]);
	int n_threads=atoi(argv[3]);

	omp_set_num_threads(n_threads);
	SMA_Init(n_threads, n_blocks);

	double t1=get_time();
	#pragma omp parallel
	{
		for(int i=0;i<n_blocks;i++){
			void * p=SMA_Malloc(blocksize);
			if(!p){
				if(omp_get_thread_num()==0)
					printf("failed\n");
				break;
			}
		}
	}
	double t2=get_time();
	SMA_Destroy();
	double t3=get_time();
	cout<<"malloc: "<<t2-t1<<"\t"<<"free: "<<t3-t2<<endl;	

	return 0;
}
