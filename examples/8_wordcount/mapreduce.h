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
	char * base_ptr=global_data.content;
	int * offsets=(int*)ptr;
	for(int i=0;i<size/sizeof(int);i++){
		char *p = base_ptr + offsets[i];
		int wordSize = 0;
		char * word_start=p;
		bool in_word=false;
		while(1)
		{
			if(*p>='a' && *p<='z'	   // a character from a-z, A-Z or 0-9, or '
				|| *p>='0' && *p<='9'
				|| *p>='A' && *p<='Z'
				|| *p=='\''){
				wordSize++;
				if(!in_word){
					in_word=true;
					word_start=p;
				}
			}
			else{
				if(in_word){
					in_word=false;
					int val=1;
					emit_intermediate(word_start, wordSize, &val, sizeof(int));
//					printf("emit_inter(word_start=%d, wordSize=%d, &val=%p)\n",word_start,wordSize,&val);
					wordSize=0;
				}
				if(*p=='\0')
					break;
			}
			p++;
		};
	}
}	

void HMM_FUNC reduce(HMMKVs_t & kv, const global_data_t & global_data){
	int sum=0;
	const void * val;
	unsigned int valsize;
	while(!kv.end()){
		kv.get_val(val,valsize);
		kv.next_val();
		sum++;
	}
	emit(kv, &sum, sizeof(int));
//	printf("emit(sum=%d)\n",sum);
}

void HMM_FUNC combine(void * old_val, const void * new_val){

}

};
#endif
