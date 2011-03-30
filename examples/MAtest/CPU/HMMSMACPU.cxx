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

#include "HMMSMACPU.h"
#include <omp.h>

namespace HMM_CPU{

using namespace std;

SMA_CPU::SMA_CPU(){
	current_page=MallocLargeBlock(PAGE_SIZE);
	current_offset=0;
}

SMA_CPU::~SMA_CPU(){
	for(int i=0;i<allocated_blocks.size();i++){
		delete[] allocated_blocks[i];
	}
}

void * SMA_CPU::SMA_Malloc(unsigned int size){
	size=RoundToAlign(size);
	if(size<PAGE_SIZE/2){
		// malloc from local page
		if(current_offset+size<PAGE_SIZE){
			unsigned int old_offset=current_offset;
			current_offset+=size;
			return current_page+old_offset;
		}
		else{
			current_page=MallocLargeBlock(PAGE_SIZE);
			current_offset=size;
			return current_page;
		}
	}
	else{
		// malloc from global 
		return MallocLargeBlock(size);
	}
}

void SMA_Init(unsigned int num_threads, int n_blocks){
	SMA_Pool=new SMA_CPU[num_threads];
}

void SMA_Destroy(){
	delete[] SMA_Pool;
}

void * SMA_Malloc(unsigned int size){
	return SMA_Pool[omp_get_thread_num()].SMA_Malloc(size);
}

};
