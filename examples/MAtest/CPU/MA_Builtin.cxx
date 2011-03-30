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

#include "MA_Builtin.h"
#include <omp.h>

namespace HMM_CPU{

using namespace std;

SMA_CPU::SMA_CPU():index(0){
}

void SMA_CPU::Clear(){
	for(int i=0;i<allocated_blocks.size();i++){
		delete[] allocated_blocks[i];
	}
	allocated_blocks.clear();
}

SMA_CPU::~SMA_CPU(){
}

void * SMA_CPU::SMA_Malloc(unsigned int size){
	size=RoundToAlign(size);
	char * p=new char[size];
	allocated_blocks[index++]=p;
	return p;
}

void SMA_Init(unsigned int num_threads, int n_blocks){
	SMA_Pool=new SMA_CPU[num_threads];
	for(int i=0;i<num_threads;i++){
		SMA_Pool[i].Reserve(n_blocks);
	}
}

void SMA_Destroy(){
	#pragma omp parallel
	{
		SMA_Pool[omp_get_thread_num()].Clear();
	}
	delete[] SMA_Pool;
}

void * SMA_Malloc(unsigned int size){
	return SMA_Pool[omp_get_thread_num()].SMA_Malloc(size);
}

};
