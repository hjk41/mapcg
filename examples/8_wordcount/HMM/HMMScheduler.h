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

#ifndef HMMSCHEDULER_H
#define HMMSCHEDULER_H

/*********************************
 * HMMScheduler.h
 *    base class for schedulers
 *
 *
 ********************************/

#include "HMMSchedulerSpec.h"
#include "HMMDS.h"

class HMMScheduler{
private:
	const void * input_data;
	unsigned int input_size;
	unsigned int unit_size;
public:
	HMMScheduler():input_data(0),input_size(0){};
	virtual ~HMMScheduler(){};

	//===================================
	// functions defined in derived class
	//===================================
	virtual void do_map_reduce()=0;
	virtual OutputChunk get_output()=0;
	virtual void init(const HMMSchedulerSpec &)=0;
};

#endif
