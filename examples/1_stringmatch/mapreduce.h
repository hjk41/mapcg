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
	int * offsets=(int*)ptr;
	int num_lines=size/sizeof(int);
	for(int i=0;i<num_lines;i++){
		char * line=global_data.content+offsets[i];
		char * pWord=line;
		char * keyword=global_data.keyword;
		while(*pWord!='\0'){
			char * curr=pWord;
			char * pkeyword=keyword;
			while(*curr==*pkeyword && *curr!='\0' && *pkeyword!='\0'){
				curr++;
				pkeyword++;
			}
			if(*pkeyword=='\0'){
				int pos=pWord-line;
				emit_intermediate(&offsets[i], sizeof(int), &pos, sizeof(int));
				break;
			}
			pWord++;
		} 
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
