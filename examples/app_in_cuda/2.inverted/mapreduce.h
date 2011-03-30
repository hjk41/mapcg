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




void HMM_FUNC slice(const void * input_array, const unsigned int data_size, 
				const unsigned int unit_size, const unsigned int index, 
				const void * &ret_ptr, unsigned int & ret_size){
	default_slice(input_array, data_size, unit_size, index, ret_ptr, ret_size);
}

#define START		0x00
#define IN_TAG		0x01
#define IN_ATAG		 0x02
#define FOUND_HREF	  0x03
#define START_LINK	  0x04

int UTIL_FUNC StrCmp(char * p1, char * p2, int size){
	for(int i=0;i<size;i++){
		if(p1[i]<p2[i])
			return -1;
		else if(p1[i]>p2[i])
			return 1;
	}
	return 0;
} 

char * UTIL_FUNC StrChr(char * p, char c){
	while(*p!='\0' && *p!=c)
		p++;
	if(*p==c)
		return p;
	return NULL;
}

void HMM_FUNC map(const void * ptr, const unsigned int size, const global_data_t & global_data){
	char *linebuf = (char*)(global_data.content+*(int*)ptr);
	int val=1;
	char * link_start;
	char *link_end;
	int state = START;
	char href[5];
	href[0] = 'h';
	href[1] = 'r';
	href[2] = 'e';
	href[3] = 'f';
	href[4] = '\0';

	for (char *p = linebuf; *p != '\0'; p++){
		switch (state){
			case START:
				if (*p == '<') state = IN_TAG;
				break;
			case IN_TAG:
				if (*p == 'a') state = IN_ATAG;
				else if (*p == ' ') state = IN_TAG;
				else state = START;
				break;
			case IN_ATAG:
				if (*p == 'h'){
					if (StrCmp(p, href, 4) == 0){
						state = FOUND_HREF;
						p += 4;
					}
					else state = START;
				}
				else if (*p == ' ') state = IN_ATAG;
				else state = START;
				break;
			case FOUND_HREF:
				if (*p == ' ') state = FOUND_HREF;
				else if (*p == '=') state = FOUND_HREF;
				else if (*p == '\"') state  = START_LINK;
				else state = START;
				break;
			case START_LINK:
				link_start=p;
				link_end = StrChr(p, '\"');
				if (link_end != NULL)
				{
					*link_end='\0';
//printf("emit_inter(%s,%d)\n",link_start,val);
					emit_intermediate(link_start, link_end-link_start+1, &val, sizeof(int));
					p =link_end;
				}
				state = START;
				break;
		}//switch
	}//for
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
}

void HMM_FUNC combine(void * old_val, const void * new_val){

}

};
#endif
