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

#include "HMMGPUScheduler.h"
#include "GPULib/HMMScheduler.h"
#include <cuda.h>
#include <stdio.h>

/*
class HMMGPUScheduler:public HMMScheduler{
public:
	~HMMScheduler();

	void init(HMMSchedulerSpec & args);
	void do_map_reduce();
	OutputChunk get_output();
};
*/
HMMGPUScheduler::HMMGPUScheduler(int device_num){
/*	cudaSetDevice(device_num);
	int i;
	cudaGetDevice(&i);
	printf("thread %d is now in device %d\n", device_num+1, i);
*/
}

HMMGPUScheduler::~HMMGPUScheduler(){
	scheduler.finish_scheduler();
}

void HMMGPUScheduler::init(const HMMSchedulerSpec & args){
	scheduler.init_scheduler(args);
}

void HMMGPUScheduler::do_map_reduce(){
	scheduler.do_map_reduce();
}

OutputChunk HMMGPUScheduler::get_output(){
	return scheduler.get_output();
}

