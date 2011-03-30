/***************************************************************************

    MapCG: MapReduce Framework for CPU & GPU

    Copyright (C) 2010, Chuntao HONG (Chuntao.Hong@gmail.com).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

**************************************************************************/

#include "HMMCommUtil.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <cuda.h>

//=============================
// CUDA related code
//=============================
int get_num_gpus(){
	int n=1;
	CE(cudaGetDeviceCount(&n));
	return n;
}

void set_gpu_num(int n){
	CE(cudaSetDevice(n));
}


//=====================================
// commonly used functions
//=====================================
void map_file(const char * filename, void * & buf, unsigned int & size){
	int fp=open(filename,O_RDONLY);
	if(fp){
		struct stat filestat;
		fstat(fp, &filestat);
		size=filestat.st_size;
		buf=mmap(0,size,PROT_READ,MAP_PRIVATE,fp,0);
	}
	else
		buf=0;
}

double last_t=get_time();
double get_time(){
	cudaThreadSynchronize();
	timeval t;
	gettimeofday(&t, NULL);
	return (double)t.tv_sec+(double)t.tv_usec/1000000;
}
void init_timer(){
	last_t=get_time();
}
double time_elapsed(){
	double new_t=get_time();
	double t=new_t-last_t;
	last_t=new_t;
	return t;
}

#ifdef __TIMING__
	#define printtime(...) printf(__VA_ARGS__)
#else
	#define printtime(...)
#endif

#ifdef __LOG__
	#define printlog(...) printf(__VA_ARGS__)
#else
	#define printlog(...)
#endif


