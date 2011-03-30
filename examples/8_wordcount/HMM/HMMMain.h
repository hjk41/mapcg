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

#ifndef HMMMAIN_H
#define HMMMAIN_H

//=============================
// entry of the HMM MapReduce scheduler
//=============================

#include <string.h>
#include "HMMSchedulerSpec.h"
#include "UtilLib/HMMTSQueue.h"
#include "HMMDS.h"
#include "HMMScheduler.h"
#include "HMMGlobalData.h"

class HMMMainScheduler:public HMMScheduler{
private:
	HMMMainScheduler(){};	// disallow creation of the scheduler without args
private:
	HMMSchedulerSpec args;
	OutputChunk output;
public:
	struct Job{
		Job():input(NULL),input_size(0){};
		Job(const void * i, unsigned int s):input(i),input_size(s){};
	
		const void * input;
		unsigned int input_size;
	};
	typedef TSQueue<Job> JobQueue;
	typedef TSQueue<OutputChunk> OutputQueue;

public:
	//=============================
	// defined in HMMScheduler
	//=============================
	// void set_input(const void *, const unsigned int);	
	// void set_unit_size(unsigned int);	

	// constructor and destructor
	HMMMainScheduler(const HMMSchedulerSpec & spec){memset(&output,0,sizeof(output));init(spec);};
	~HMMMainScheduler();

	void init(const HMMSchedulerSpec & args);
	void do_map_reduce();
	OutputChunk get_output();
private:
	OutputChunk merge_output(OutputQueue & queue);
};


#endif
