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

#ifndef MA_BUILTIN_H
#define MA_BUILTIN_H

#include "HMMUtilGPU.h"

namespace HMM_GPU{
const unsigned int ALIGN_SIZE=8;

__device__ unsigned int SMA_RoundToAlign(unsigned int size){
	return (size+ALIGN_SIZE-1)&(~(ALIGN_SIZE-1));
}

__host__ void SMA_Init(unsigned int size){
	cudaThreadSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024);
}

__host__ void SMA_Destroy(){
}

__device__ int SMA_Start_Kernel(){
	return 0;
}

__device__ void * SMA_Malloc(unsigned int size){
	return malloc( SMA_RoundToAlign(size) );
}

__device__ void SMA_Free(void * p){
	free(p);
}
};
#endif
