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

#ifndef HMMCPUSCHEDULER_H
#define HMMCPUSCHEDULER_H

#include "HMMScheduler.h"
#include "HMMSchedulerSpec.h"
#include "HMMDS.h"
#include "CPULib/HMMSchedulerCPU.h"

class HMMCPUScheduler:public HMMScheduler{
public:
	~HMMCPUScheduler();

	void init(const HMMSchedulerSpec & args);
	void do_map_reduce();
	OutputChunk get_output();
private:
	HMM_CPU::HMMSchedulerCPU scheduler;
};


#endif
