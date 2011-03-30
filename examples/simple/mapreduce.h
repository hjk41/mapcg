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

#ifdef HMMSCHEDULERCPU_H
namespace HMM_CPU{

void HMMSchedulerCPU::slice(const void * input_array, const unsigned int data_size, const unsigned int unit_size, const unsigned int index, const void * &ret_ptr, unsigned int & ret_size){
	default_slice(input_array,data_size,unit_size,index,ret_ptr,ret_size);
}

void HMMSchedulerCPU::map(const void * ptr, const unsigned int size, const global_data_t & gd){
	int * p=(int*)ptr;
	int val=1;
	for(int i=0;i<size;i+=sizeof(int)){
//printf("emit_inter: %d, %d\n", *(p+i), val);
		emit_intermediate(p+i, sizeof(int), &val, sizeof(int));
	}
}

void HMMSchedulerCPU::reduce(HMMKVs_t & kv, const global_data_t & gd){
	const void * key, *val;
	unsigned keysize, valsize;
	kv.get_key(key,keysize);

	int i=0;
	while(!kv.end()){
		kv.get_val(val,valsize);
		i+=*(int*)val;
		kv.next_val();
	}
//printf("emit: %d, %d\n", *(int *)key, i);
	emit(kv, &i, sizeof(int));
}

void HMMSchedulerCPU::combine(void * old_val, const void *new_val){
	*(int*)old_val+=*(int*)new_val;
}

};
#endif

#ifdef HMMSCHEDULERGPU_H
namespace HMM_GPU{

__device__ void slice(const void * input_array, const unsigned int data_size, const unsigned int unit_size, const unsigned int index, const void * &ret_ptr, unsigned int & ret_size){
	default_slice(input_array,data_size,unit_size,index,ret_ptr,ret_size);
}

__device__ void map(const void * ptr, const unsigned int size, const global_data_t & gd){
	int * p=(int*)ptr;
	int val=1;
	for(int i=0;i<size;i+=sizeof(int)){
//printf("emit_inter: %d, %d\n", *(p+i), val);
		emit_intermediate(p+i, sizeof(int), &val, sizeof(int));
	}
}

__device__ void reduce(HMMKVs_t & kv, const global_data_t & gd){
	const void * key, *val;
	unsigned keysize, valsize;
	kv.get_key(key,keysize);

	int i=0;
	while(!kv.end()){
		kv.get_val(val,valsize);
		i+=*(int*)val;
		kv.next_val();
	}
//printf("emit: %d, %d\n", *(int *)key, i);
	emit(kv, &i, sizeof(int));
}

__device__ void combine(void * old_val, const void *new_val){
//printf("combine: %d, %d\n", *(int*)old_val, *(int*)new_val);
	*(int*)old_val+=*(int*)new_val;
}
};
#endif

#endif
