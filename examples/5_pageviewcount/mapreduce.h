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

int UTIL_FUNC StrHash(char *str, int strLen)
{
        int     hash = strLen;

        for (int i=0; i < strLen; i++)
                hash = (hash<<4)^(hash>>28)^str[i];

        return hash;
}

void HMM_FUNC slice(const void * input_array, const unsigned int data_size, 
				const unsigned int unit_size, const unsigned int index, 
				const void * &ret_ptr, unsigned int & ret_size){
	default_slice(input_array, data_size, unit_size, index, ret_ptr, ret_size);
}

void HMM_FUNC map(const void * ptr, const unsigned int size, const global_data_t & global_data){
	if(global_data.pass==1){
		int offset=*(int*)ptr;
		char * line=global_data.content+offset;
		char * curr=line;
		while(*curr!='\0'){
			curr++;
		}
		int len=curr-line;
		int2 k;
		k.x=StrHash(line,len);
		k.y=offset;
		emit_intermediate(&k, sizeof(int2), &len, sizeof(int));
	}
	else{
		// find url
		int2 k=*(int2*)ptr;
		char * line=global_data.content+k.y;
		char * curr=line;
		while(*curr!='\t'){
			curr++;			
		}
		int len=curr-line;
		k.x=StrHash(line, len);
		emit_intermediate(&k, sizeof(int2), &len, sizeof(int));
	}
}

void HMM_FUNC reduce(HMMKVs_t & kv, const global_data_t & global_data){
	if(global_data.pass==1){
		const void * val;
		unsigned int valsize;
		kv.get_val(val,valsize);
		emit(kv, val, valsize);
	}
	else{
		int sum=0;
		while(!kv.end()){
			sum++;
			kv.next_val();
		}
		emit(kv,&sum,sizeof(int));
	}
}

void HMM_FUNC combine(void * old_val, const void * new_val){

}

};
#endif
