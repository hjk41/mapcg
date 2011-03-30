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
#define UTIL_FUNC
namespace HMM_CPU{
#elif defined(HMMHASHTABLEGPU_H)
#define HMM_FUNC __device__
#define UTIL_FUNC __device__
namespace HMM_GPU{
#endif

inline unsigned int UTIL_FUNC get_sq_dist(int* point, int* mean, int dim)
{
	int i;
	unsigned int sum = 0;
	for(i = 0; i < dim; i++)
	{
		sum += (point[i] - mean[i])*(point[i] - mean[i]);
	}
	return sum;
}

void HMM_FUNC slice(const void * input_array, const unsigned int data_size, 
                                const unsigned int unit_size, const unsigned int index, 
                                const void * &ret_ptr, unsigned int & ret_size){
	default_slice(input_array, data_size, unit_size, index, ret_ptr, ret_size);
}

void HMM_FUNC map(const void * ptr, const unsigned int size, const global_data_t & global_data){
	int * point_index=(int*)ptr;
	for(int i=0;i<size/sizeof(int);i++){
		int index = point_index[i];
		int dim = global_data.dim;
		int * point=global_data.points+dim*index;
		int num_means = global_data.num_means;
		int* means = global_data.means;

		unsigned int min_dist, cur_dist;
		min_dist = get_sq_dist(point, means, dim);
		int min_idx = 0;
		for(int j = 1; j < num_means; j++)
		{
			cur_dist = get_sq_dist(point, &means[j*dim], dim);
			if(cur_dist < min_dist)
			{
				min_dist = cur_dist;
				min_idx = j;
			}
		}
		emit_intermediate(&min_idx, sizeof(int), &index, sizeof(int));	// emit to the new cluster
		index+=num_means;
		emit_intermediate(&index,sizeof(int), &min_idx, sizeof(int));	// emit new position
	}
}

void HMM_FUNC reduce(HMMKVs_t & kv, const global_data_t & global_data){
	const void * key;
	unsigned int keysize;
	kv.get_key(key,keysize);
	unsigned int index = *((int*)key);
	if(index>=global_data.num_means){	// this is a position pair
		const void * val;
		unsigned int valsize;
		kv.get_val(val,valsize);	
		emit(kv, val,valsize);
		return;
	}
	else{
		int dim = global_data.dim;
		int num_means = global_data.num_means;
		int* means = &(global_data.means[index*dim]);
		unsigned int val_num = 0;

		for(int i = 0; i < dim; i++)
		{
			means[i] = 0;
		}
		while(!kv.end()){
			const void * val;
			unsigned int valsize;
			kv.get_val(val,valsize);
			val_num++;
			int pIdx = *(int*) val;
			int * point=global_data.points+pIdx*dim;
			for(int i = 0; i < dim; i++){
				means[i] += point[i];
			}
			kv.next_val();
		}
		for(int i = 0; i < dim; i++)
		{
			means[i] /= val_num;
		}
		emit(kv, means, sizeof(int)*dim);
	}
}

void HMM_FUNC combine(void * old_val, const void * new_val){

}

};
#endif
