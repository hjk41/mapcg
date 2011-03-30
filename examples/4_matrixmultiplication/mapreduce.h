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

#ifndef MAPREDUCE_H
#define MAPREDUCE_H

#include <stdio.h>

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
	const pos_t * p_pos=(const pos_t *)ptr;
	for(int i=0;i<size/sizeof(pos_t);i++){
		pos_t p=p_pos[i];
		float sum=0;
		float * a=global_data.A+p.x*global_data.dim;
		float * b=global_data.B;
		for(int j=0;j<global_data.dim;j++){
			sum+=a[j]*b[j*global_data.dim+p.y];
		}
		unsigned int index=p.x*global_data.dim+p.y;

		emit_intermediate(&index, sizeof(unsigned int), &sum, sizeof(float));
	}
}

void HMM_FUNC reduce(HMMKVs_t & kv, const global_data_t & global_data){
	const void * val;
	unsigned int valsize;
	kv.get_val(val, valsize);
	emit(kv, val, valsize);
}

void HMM_FUNC combine(void * old_val, const void * new_val){

}
};
#endif
