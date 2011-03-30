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

#ifndef MAPREDUCEFUNC_CU
#define MAPREDUCEFUNC_CU

#include "DS.h"
#include <stdio.h>
#include <math.h>

#if defined(HMMHASHTABLECPU_H)
#define HMM_FUNC HMMSchedulerCPU:: 
namespace HMM_CPU{
#elif defined(HMMHASHTABLEGPU_H)
#define HMM_FUNC __device__
namespace HMM_GPU{
#endif

void HMM_FUNC slice(const void * input_array, const unsigned int data_size, 
                                const unsigned int unit_size, const unsigned int index, 
                                const void * &ret_ptr, unsigned int & ret_size){
	default_slice(input_array, data_size, unit_size, index, ret_ptr, ret_size);
}

void HMM_FUNC map(const void * ptr, const unsigned int size, const global_data_t & global_data){
	point_pair_t * pairs=(point_pair_t *)ptr;
	unsigned int num_dims=global_data.num_dims;
	unsigned int num_points=global_data.num_points;
	const float * points=global_data.points;
	for(int i=0;i<size/sizeof(point_pair_t);i++){
		point_pair_t * p=pairs+i;
		const float * a=points+p->x*num_dims;
		const float * b=points+p->y*num_dims;
		float ab=0;
		float aa=0;
		float bb=0;
		float ai,bi;
		for(int i=0;i<num_dims;i++){
			ai=a[i];
			bi=b[i];
			ab+=ai*bi;
			aa+=ai*ai;
			bb+=bi*bi;
		}
		int x=p->x;
		int y=p->y;
		float result=sqrt((ab*ab)/(aa*bb));
		unsigned int idx=x*num_points+y;
		emit_intermediate(&idx, sizeof(unsigned int), &result, sizeof(float));
	}
}

void HMM_FUNC reduce(HMMKVs_t & kv, const global_data_t & global_data){
	const void * val;
	unsigned int valsize;
	kv.get_val(val,valsize);
	emit(kv, val, valsize);
}

void HMM_FUNC combine(void * old_val, const void * new_val){

}

};
#endif
