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

#ifndef CPU_SMA_H
#define CPU_SMA_H

#include <vector>
#include "HMMUtilCPU.cxx"

namespace HMM_CPU{

class SMA_CPU{
private:
	std::vector<char *> allocated_blocks;
	char * current_page;
	unsigned int current_offset;
private:
	static const unsigned int PAGE_SIZE=1024*1024;
	static const unsigned int ALIGN_SIZE=4;
	unsigned int RoundToAlign(unsigned int size){
		return (size+ALIGN_SIZE-1)&(~(ALIGN_SIZE-1));
	}
	char * MallocLargeBlock(unsigned int size){
		char * block=new char[size];
		allocated_blocks.push_back(block);
		return block;
	}
public:
	SMA_CPU();
	~SMA_CPU();

	void * SMA_Malloc(unsigned int size);
	void SMA_Free(void * p);
};

static SMA_CPU * SMA_Pool;

void SMA_Init(unsigned int num_threads);
void SMA_Destroy();
void * SMA_Malloc(unsigned int size);

};// namespace HMM_CPU

#endif
