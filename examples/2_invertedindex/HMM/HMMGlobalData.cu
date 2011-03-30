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

#include "HMMGlobalData.h"
#include <cuda.h>
#include <map>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
using namespace std;

#include "UtilLib/HMMCommUtil.h"

#  define CUT_CHECK_ERROR(errorMessage) do {			     \
    cudaError_t err = cudaGetLastError();				   \
    if( cudaSuccess != err) {					   \
	fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
		errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
	exit(EXIT_FAILURE);					       \
    }								   \
    err = cudaThreadSynchronize();					 \
    if( cudaSuccess != err) {					   \
	fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
		errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
	exit(EXIT_FAILURE);					       \
    } } while (0)

#define FIELD_PTR(base, offset) ((void*)((char*)(base)+offset))

struct GlobalDataListTuple{
	unsigned int size;
	bool dirty;
	GlobalDataListTuple():size(0),dirty(true){};
	GlobalDataListTuple(unsigned int s):size(s),dirty(true){};
};
typedef std::map<unsigned int, GlobalDataListTuple> GlobalDataList;
typedef	std::map<int,global_data_t *> GlobalDataDevicePointers; 

GlobalDataList global_data_settings;
GlobalDataDevicePointers global_data_pointers;

void declare_global_array(unsigned int offset, unsigned int size){
	global_data_settings[offset]=GlobalDataListTuple(size);
}

void mark_array_as_dirty(unsigned int offset){
	global_data_settings[offset].dirty=true;
}

void mark_all_as_clean(){
	for(GlobalDataList::iterator it=global_data_settings.begin(); it!=global_data_settings.end();++it){
		it->second.dirty=false;
	}
}


//----------------------
// device functions
void sync_global_array(global_data_t * h_global_data, global_data_t * d_global_data, unsigned int offset, unsigned int size){
	assert(d_global_data);
	// copy data
	void * d_ptr=FIELD_PTR(d_global_data,offset);
	void * h_ptr=FIELD_PTR(h_global_data,offset);
	void * h_array=*(void**)h_ptr;
	void * d_array;
	cudaMemcpy((void*)(&d_array),d_ptr,sizeof(void*),cudaMemcpyDeviceToHost);
	CUT_CHECK_ERROR("sync_global_array");
	cudaMemcpy(d_array,h_array,size,cudaMemcpyHostToDevice);
	CUT_CHECK_ERROR("sync_global_array");
}

void sync_global_data(global_data_t * h_global_data, global_data_t * d_global_data){
double t1=get_time();
	global_data_t old_global_data,new_global_data;
	// updated scala variables
	memcpy(&new_global_data,h_global_data,sizeof(global_data_t));
	cudaMemcpy(&old_global_data, d_global_data, sizeof(global_data_t), cudaMemcpyDeviceToHost);
	// update arrays
	for(GlobalDataList::iterator it=global_data_settings.begin(); it!=global_data_settings.end(); ++it){
		// copy device pointers to tmp
		int offset=it->first;
		void * p1=FIELD_PTR(&old_global_data,offset);
		void * p2=FIELD_PTR(&new_global_data,offset);
		memcpy(p2,p1,sizeof(void*));
		// update arrays
		if(it->second.dirty){
			sync_global_array(h_global_data, d_global_data, it->first, it->second.size);
			it->second.dirty=false;
		}
	}
	// ok, now new_global_data has been updated, copy back to device
	cudaMemcpy(d_global_data, &new_global_data, sizeof(global_data_t), cudaMemcpyHostToDevice);
double t2=get_time();
cout<<"sync_gpu_global_data: "<<t2-t1<<endl;
}

global_data_t * malloc_gpu_global_data(global_data_t * h_global_data){
double t1,t2;
t1=get_time();
	int gpu_num;
	cudaGetDevice(&gpu_num);
	global_data_t * d_global_data=global_data_pointers[gpu_num];
/*	if(d_global_data!=NULL){
		sync_global_data(h_global_data, d_global_data);
	}
	else {
*/		global_data_t tmp_data;
		memcpy(&tmp_data, h_global_data, sizeof(global_data_t));
		for(GlobalDataList::iterator it=global_data_settings.begin(); it!=global_data_settings.end(); ++it){
			int offset=it->first;
			int size=it->second.size;
			void * d_ptr=FIELD_PTR(&tmp_data,offset);
			void * h_ptr=FIELD_PTR(h_global_data,offset);
			cudaMalloc((void**)d_ptr, size);
			CUT_CHECK_ERROR("malloc_gpu_global_data");
			void * h_array=*(void**)h_ptr;
			void * d_array=*(void**)d_ptr;
			cudaMemcpy(d_array,h_array,size,cudaMemcpyHostToDevice);
		}
		cudaMalloc((void**)&d_global_data, sizeof(global_data_t));
		cudaMemcpy(d_global_data, &tmp_data, sizeof(global_data_t),cudaMemcpyHostToDevice);
		CUT_CHECK_ERROR("malloc_gpu_global_data");
		global_data_pointers[gpu_num]=d_global_data;
//	}
t2=get_time();
cout<<"malloc_gpu_global_data: "<<t2-t1<<endl;
	return d_global_data;
}

void free_gpu_global_data(global_data_t * d_global_data){
	global_data_t tmp_data;
	cudaMemcpy(&tmp_data, d_global_data, sizeof(global_data_t), cudaMemcpyDeviceToHost);
	for(GlobalDataList::iterator it=global_data_settings.begin(); it!=global_data_settings.end(); ++it){
		int offset=it->first;
		int size=it->second.size;
		void * d_ptr=FIELD_PTR(&tmp_data,offset);
		void * d_array=*(void**)d_ptr;
		cudaFree(d_array);
	}	
	cudaFree(d_global_data);
}

unsigned int get_global_data_size(){
	unsigned int size=0;
	for(GlobalDataList::iterator it=global_data_settings.begin(); it!=global_data_settings.end(); ++it){
		size+=it->second.size;
	}
	size+=sizeof(global_data_t);
	return size;
}
