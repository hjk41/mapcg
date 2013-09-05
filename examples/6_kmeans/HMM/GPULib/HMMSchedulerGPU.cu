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


#define __TIMING__
//#define __DEBUG__


#include "HMMHashTable.h"
#include "HMMScheduler.h"
#include "HMMSMA.h"
#include "HMMScan.h"
#include "HMMSort.h"
#include "HMMHashTable.cu"
#include "../HMMDS.h"
#include "../HMMGlobalData.h"

#include <cuda.h>
#include <assert.h>
#include <iostream>
#include <vector>
using namespace std;
#ifdef __TIMING__
#include <sys/time.h>
#endif
namespace HMM_GPU{

__device__ GlobalDeviceState g_device_state;

__device__ void emit_intermediate(const void * key, const unsigned int keysize, const void * val, const unsigned int valsize);
__device__ void emit(HMMKVs_t & kvlist, const void * val, const unsigned int valsize);
__device__ void default_slice(const void * input_array, const unsigned int data_size, const unsigned int unit_size, const unsigned int index, const void * &ret_ptr, unsigned int & ret_size);

/*
//---------------------------------------------------
// Forward declaration
//---------------------------------------------------
__device__ void slice(const void * input_array, const unsigned int data_size, const unsigned int unit_size, const unsigned int index, const void * &ret_ptr, unsigned int & ret_size);
__device__ void map(const void * ptr, const unsigned int size, const global_data_t & gd);
__device__ void reduce(HMMKVs_t & kv, const global_data_t & gd);
__device__ void emit_intermediate(const void * key, const unsigned int keysize, const void * val, const unsigned int valsize);
__device__ void emit(HMMKVs_t & kvlist, const void * val, const unsigned int valsize);
__device__ void default_slice(const void * input_array, const unsigned int data_size, const unsigned int unit_size, const unsigned int index, const void * &ret_ptr, unsigned int & ret_size);
*/

__device__ void emit_intermediate(const void * key, const unsigned int keysize, const void * val, const unsigned int valsize){
	g_device_state.hash_table->insert(key, keysize, val, valsize);
}

// emit a key/value in reduce, should be called only once per reduce
__device__ void emit(HMMKVs_t & kvlist, const void * val, const unsigned int valsize){
        kvlist.kit.setValue(val, valsize);
}

__device__ void default_slice(const void * input_array, const unsigned int data_size, const unsigned int unit_size, const unsigned int index, const void * &ret_ptr, unsigned int & ret_size){
	unsigned int offset=index*unit_size;
	ret_ptr=(char*)input_array+offset;
	ret_size= offset+unit_size>data_size? data_size-offset:unit_size;
}
};


#include "../../mapreduce.h"

namespace HMM_GPU{

//------------------------------------------
// timing and logging
//------------------------------------------
#ifdef __TIMING__
double HMMSchedulerGPU::my_time(){
	cudaThreadSynchronize();
	timeval t;
	gettimeofday(&t,NULL);
	return (double)t.tv_sec+(double)t.tv_usec/1000000;
}
double HMMSchedulerGPU::time_elapsed(){
	double new_t=my_time();
	double t=new_t-last_t;
	last_t=new_t;
	return t;
}
#define printtime(...) printf(__VA_ARGS__)
#else
#define printtime(...)
#endif

#ifdef __DEBUG__
#define printlog(...) printf(__VA_ARGS__)
#else
#define printlog(...) 
#endif

/*****************************************************
  function implementation

******************************************************/
// -----------------------
// setup g_state and device
//------------------------
void HMMSchedulerGPU::init_scheduler(const HMMSchedulerSpec & args){
#ifdef __TIMING__
	last_t=my_time();
#endif
	// copy the arguments
	assert(args.input);
	g_state.input=args.input;

	assert(args.input_size>0);
	g_state.input_size=args.input_size;

	assert(args.unit_size>0);
	g_state.unit_size=args.unit_size;

	g_state.local_combine=args.local_combine;

	assert(args.num_hash_buckets>0);
	g_state.num_hash_buckets=args.num_hash_buckets;

	g_state.map_grid_dim=args.gpu_map_grid;
	g_state.map_block_dim=args.gpu_map_block;
	g_state.reduce_grid_dim=args.gpu_reduce_grid;
	g_state.reduce_block_dim=args.gpu_reduce_block;

	g_state.sort_output=args.sort_output;

	g_state.global_data=args.global_data;
	// set state as initialized
	g_state.initialized=true;
}

//------------------------------------------
// do a map_reduce iteration
//------------------------------------------
void HMMSchedulerGPU::do_map_reduce(){
	assert(g_state.initialized);
	// clear result from last run
	if(g_state.output_keys){
		free(g_state.output_keys);
		g_state.output_keys=NULL;
	}
	if(g_state.output_vals){
		free(g_state.output_vals);
		g_state.output_vals=NULL;
	}
	if(g_state.output_index){
		free(g_state.output_index);
		g_state.output_index=NULL;
	}
	if(g_state.output_buckets){
		free(g_state.output_buckets);
		g_state.output_buckets=NULL;
	}
	// do map/reduce
	schedule_tasks();
}

//------------------------------------------
// workers
//------------------------------------------
__global__ void map_worker(const void * input, unsigned int input_size, unsigned int unit_size){
	// setup SMA
	SMA_Start_Kernel();

	int threadID=getThreadID();
	int numThreads=getNumThreads();
	
	// split data and invoke map
	unsigned int numChunks=(input_size+unit_size-1)/unit_size;
	for(int i=threadID; i<numChunks; i+=numThreads){
		const void * ptr;
		unsigned int size;
		HMM_GPU::slice(input, input_size, unit_size, i, ptr, size);
		map(ptr,size,*(g_device_state.global_data));
	}
}

__global__ void reduce_worker(){
	// setup SMA
	SMA_Start_Kernel();

	int threadID=getThreadID();
	int numThreads=getNumThreads();
	// invoke reduce
	HashMultiMap * hashTable=g_device_state.hash_table;
	unsigned int numChunks=hashTable->num_buckets;
	for(int i=threadID; i<numChunks; i+=numThreads){
		int key_size=0;
		int val_size=0;
		int num_keys=0;
		for(KeysIter kit=hashTable->getBucket(i); kit; ++kit){
			HMMKVs_t kvlist(kit);
			reduce(kvlist,*(g_device_state.global_data));
			const void * ptr;
			unsigned int size;
			kit.getKey(ptr, size);
			key_size+=size;
			kit.getValues().getValue(ptr,size);
			val_size+=size;
			num_keys+=1;
		}
		g_device_state.key_size_per_bucket[i]=key_size;
		g_device_state.val_size_per_bucket[i]=val_size;
		g_device_state.num_pairs_per_bucket[i]=num_keys;
	}
}

__global__ void copy_hash_to_array(const unsigned int * key_start, const unsigned int * val_start, const unsigned int * index_start){
	int threadID=getThreadID();
	int numThreads=getNumThreads();
	HashMultiMap * hashTable=g_device_state.hash_table;
	unsigned int numChunks=hashTable->num_buckets;
	for(int i=threadID; i<numChunks; i+=numThreads){
		int keyIndex=key_start[i];
		int valIndex=val_start[i];
		int idxIndex=index_start[i];
		g_device_state.output_buckets[i].x=idxIndex;
		for(KeysIter kit=hashTable->getBucket(i); kit; ++kit){
			const void *key, *val;
			unsigned int keysize, valsize;
			kit.getKey(key, keysize);
			kit.getValues().getValue(val, valsize);
			copyVal(g_device_state.output_keys+keyIndex, key, keysize);
			copyVal(g_device_state.output_vals+valIndex, val, valsize);
			g_device_state.output_index[idxIndex].x=keyIndex;
			g_device_state.output_index[idxIndex].y=keysize;
			g_device_state.output_index[idxIndex].z=valIndex;
			g_device_state.output_index[idxIndex].w=valsize;
			keyIndex+=keysize;
			valIndex+=valsize;
			idxIndex++;
		}
		g_device_state.output_buckets[i].y=idxIndex-g_device_state.output_buckets[i].x;
	}
}

void dump_device_memory(void * d_ptr, unsigned int size){
#ifdef __DEBUG__
//	printf("mem dump of address: %d\n",(unsigned int)d_ptr);
	unsigned int * tmp=(unsigned int*)malloc(size);
	memcpyD2H(tmp, d_ptr, size);
	for(int i=0;i<size/sizeof(int);i++){
		printf("%d ",tmp[i]);
	}
	printf("\n");
	free(tmp);
#endif
}

// -----------------------
// schedule the tasks
//------------------------
void HMMSchedulerGPU::schedule_tasks()
{
	// setup global data
	if(g_state.global_data && !g_state.gd_initialized){
		GlobalDeviceState h_device_state;
		memcpyFromSymbol(&h_device_state, g_device_state);
		h_device_state.global_data=::malloc_gpu_global_data(g_state.global_data);
		memcpyToSymbol(g_device_state, &h_device_state);
		g_state.gd_initialized=true;
	}

printtime("init time: %f \n", time_elapsed());
// assume no streaming for now
	// get GPU num
	int gpu_num=0;
	CE(cudaGetDevice(&gpu_num));

	GlobalDeviceState h_device_state;
	memcpyFromSymbol(&h_device_state, g_device_state);
	::sync_global_data(g_state.global_data, h_device_state.global_data);
	// initialize memory allocator
	unsigned int memory_pool_size=getAllocatableMemSize(gpu_num);
	unsigned int reserved_mem=RESERVED_MEM_SIZE+::get_global_data_size();
	assert(memory_pool_size>reserved_mem);
	SMA_Init(memory_pool_size-reserved_mem);	//reserve some space for input data
	// create hash map
	h_device_state.local_combine=g_state.local_combine;
	h_device_state.hash_table=newHashMultiMap(g_state.num_hash_buckets);
	memcpyToSymbol(g_device_state, &h_device_state);

	//----------
	// map
	//----------
	// copy input to GPU memory
	void * d_input=(void*)SMA_Malloc_From_Host(g_state.input_size);
	memcpyH2D(d_input, g_state.input, g_state.input_size);
	// start map
	map_worker<<<g_state.map_grid_dim, g_state.map_block_dim>>>(d_input, g_state.input_size, g_state.unit_size);
	cudaThreadSynchronize();
	CUT_CHECK_ERROR("map worker.");
printlog("map done\n");
printtime("map time: %f \n", time_elapsed());
	// get intermediate data offset, the result of reduce will be put after this offset
	void * d_mem_pool;
	unsigned int pool_size;
	SMA_Get_Status(d_mem_pool, pool_size);

	//----------
	// reduce
	//----------
	// allocate array for counting reduce space
	int num_hash_buckets=g_state.num_hash_buckets;
	unsigned int * d_key_size_per_bucket=(unsigned int*)SMA_Malloc_From_Host(num_hash_buckets*sizeof(unsigned int));
	unsigned int * d_val_size_per_bucket=(unsigned int*)SMA_Malloc_From_Host(num_hash_buckets*sizeof(unsigned int));
	unsigned int * d_num_pairs_per_bucket=(unsigned int*)SMA_Malloc_From_Host(num_hash_buckets*sizeof(unsigned int));
	dMemset(d_key_size_per_bucket, 0, num_hash_buckets*sizeof(unsigned int));	
	dMemset(d_val_size_per_bucket, 0, num_hash_buckets*sizeof(unsigned int));	
	dMemset(d_num_pairs_per_bucket, 0, num_hash_buckets*sizeof(unsigned int));	
	h_device_state.key_size_per_bucket=d_key_size_per_bucket;
	h_device_state.val_size_per_bucket=d_val_size_per_bucket;
	h_device_state.num_pairs_per_bucket=d_num_pairs_per_bucket;
	memcpyToSymbol(g_device_state, &h_device_state);	

	// invoke reduce and count reduce space at the same time
	reduce_worker<<<g_state.reduce_grid_dim, g_state.reduce_block_dim>>>();
printlog("reduce done\n");
printtime("reduce time: %f \n", time_elapsed());
	CUT_CHECK_ERROR("reduce worker.");
	// delete the intermediate data
	// prefix sum the arrays
	unsigned int * d_key_start_per_bucket=(unsigned int*)SMA_Malloc_From_Host(num_hash_buckets*sizeof(unsigned int));
	unsigned int * d_val_start_per_bucket=(unsigned int*)SMA_Malloc_From_Host(num_hash_buckets*sizeof(unsigned int));
	unsigned int * d_index_start_per_bucket=(unsigned int*)SMA_Malloc_From_Host(num_hash_buckets*sizeof(unsigned int));
	unsigned int total_key_size=prefix_sum(d_key_size_per_bucket, d_key_start_per_bucket, num_hash_buckets);
	unsigned int total_val_size=prefix_sum(d_val_size_per_bucket, d_val_start_per_bucket, num_hash_buckets);
	unsigned int total_num_pairs=prefix_sum(d_num_pairs_per_bucket, d_index_start_per_bucket, num_hash_buckets);
printlog("total_key_size=%d\n",total_key_size);
printlog("total_val_size=%d\n",total_val_size);
printlog("total_num_pairs=%d\n",total_num_pairs);
	// allocate the output array
	char * d_output_keys=(char *)SMA_Malloc_From_Host(total_key_size);
	char * d_output_vals=(char *)SMA_Malloc_From_Host(total_val_size);
	int4 * d_output_index=(int4 *)SMA_Malloc_From_Host(total_num_pairs*sizeof(int4));
	int2 * d_output_buckets=(int2 *)SMA_Malloc_From_Host(g_state.num_hash_buckets*sizeof(int2));
	h_device_state.output_keys=d_output_keys;
	h_device_state.output_vals=d_output_vals;
	h_device_state.output_index=d_output_index;
	h_device_state.output_buckets=d_output_buckets;
	memcpyToSymbol(g_device_state, &h_device_state);
	// copy output data into array
	copy_hash_to_array<<<g_state.reduce_grid_dim, g_state.reduce_block_dim>>>(d_key_start_per_bucket, d_val_start_per_bucket, d_index_start_per_bucket);
printlog("copy hash to array done\n");
printtime("hash to array: %f \n", time_elapsed());
	CUT_CHECK_ERROR("copy hash to array.");
	// copy output to CPU memory, and clear the entire memory pool
	char * h_output_keys=(char *)malloc(total_key_size);
	char * h_output_vals=(char *)malloc(total_val_size);
	int4 * h_output_index=(int4 *)malloc(total_num_pairs*sizeof(int4));
	int2 * h_output_buckets=(int2 *)malloc(g_state.num_hash_buckets*sizeof(int2));
	memcpyD2H(h_output_keys, d_output_keys, total_key_size);
	memcpyD2H(h_output_vals, d_output_vals, total_val_size);
	memcpyD2H(h_output_index, d_output_index, total_num_pairs*sizeof(int4));
	memcpyD2H(h_output_buckets, d_output_buckets, g_state.num_hash_buckets*sizeof(int2));
	// destroy memory pool
	SMA_Destroy();
/*
	if(g_state.sort_output){
		//----------
		// sort/merge
		//----------
		// copy the output key/valus back
		d_output_keys=(char *)dMalloc(total_key_size);
		d_output_vals=(char *)dMalloc(total_val_size);
		d_output_index=(int4 *)dMalloc(total_num_pairs*sizeof(int4));
		memcpyH2D(d_output_keys, h_output_keys, total_key_size);
		memcpyH2D(d_output_vals, h_output_vals, total_val_size);
		memcpyH2D(d_output_index, h_output_index, total_num_pairs*sizeof(int4));
		// malloc space for sorted result
		char * d_sorted_keys=(char *)dMalloc(total_key_size);
		char * d_sorted_vals=(char *)dMalloc(total_val_size);
		int4 * d_sorted_index=(int4 *)dMalloc(total_num_pairs*sizeof(int4));

		// start the sorting 
		GPUBitonicSortMem(d_output_keys, total_key_size, d_output_vals, total_val_size, d_output_index, total_num_pairs,
				d_sorted_keys, d_sorted_vals, d_sorted_index);
	
		// copy the output to CPU memory
		memcpyD2H(h_output_keys, d_sorted_keys, total_key_size);
		memcpyD2H(h_output_vals, d_sorted_vals, total_val_size);
		memcpyD2H(h_output_index, d_sorted_index, total_num_pairs*sizeof(int4));

		dFree(d_output_keys);
		dFree(d_output_vals);
		dFree(d_output_index);
		dFree(d_sorted_keys);
		dFree(d_sorted_vals);
		dFree(d_sorted_index);
	}
*/
	g_state.output_keys=h_output_keys;
	g_state.output_vals=h_output_vals;
	g_state.output_index=h_output_index;
	g_state.output_buckets=h_output_buckets;
	g_state.output_num_pairs=total_num_pairs;
printtime("finish: %f \n", time_elapsed());
}

__global__ void copy_hash_to_array_with_offset(const unsigned int * key_start, const unsigned int * val_start, const unsigned int * index_start,
					const unsigned int key_offset, const unsigned int val_offset, const unsigned int index_offset){
	int threadID=getThreadID();
	int numThreads=getNumThreads();
	HashMultiMap * hashTable=g_device_state.hash_table;
	unsigned int numChunks=hashTable->num_buckets;
	for(int i=threadID; i<numChunks; i+=numThreads){
		int keyIndex=key_start[i];
		int valIndex=val_start[i];
		int idxIndex=index_start[i];
		g_device_state.output_buckets[i].x=idxIndex+index_offset;
		int idxStart=idxIndex;
		for(KeysIter kit=hashTable->getBucket(i); kit; ++kit){
			const void *key, *val;
			unsigned int keysize, valsize;
			kit.getKey(key, keysize);
			kit.getValues().getValue(val, valsize);
			copyVal(g_device_state.output_keys+keyIndex, key, keysize);
			copyVal(g_device_state.output_vals+valIndex, val, valsize);
			g_device_state.output_index[idxIndex].x=keyIndex+key_offset;
			g_device_state.output_index[idxIndex].y=keysize;
			g_device_state.output_index[idxIndex].z=valIndex+val_offset;
			g_device_state.output_index[idxIndex].w=valsize;
			keyIndex+=keysize;
			valIndex+=valsize;
			idxIndex++;
		}
		g_device_state.output_buckets[i].y=idxIndex-idxStart;
	}
}

__global__ void copy_chunk_to_hash(const char * keys, const char * vals, const int4* index, 
		const int2* buckets, const unsigned int num_buckets){
	// setup SMA
	SMA_Start_Kernel();

	int index_offset=buckets[0].x;
	int key_offset=index[0].x;
	int val_offset=index[0].z;

	int threadID=getThreadID();
	int numThreads=getNumThreads();
	HashMultiMap * hash_table=g_device_state.hash_table;
	for(int i=threadID; i<num_buckets; i+=numThreads){
		int bucket_start=buckets[i].x;
		int bucket_end=bucket_start+buckets[i].y;
		for(int j=bucket_start;j<bucket_end;j++){
			int4 idx=index[j-index_offset];
			const char * key=keys+idx.x-key_offset;
			unsigned int keysize=idx.y;
			// get corresponding valuelist
			ValueList & vlist=hash_table->buckets[i].getValueList(key, keysize);
			const char * val=vals+idx.z-val_offset;
			unsigned int valsize=idx.w;
			// insert into the list
			vlist.insert(val, valsize);
		}
	}
}

void HMMSchedulerGPU::do_merge(const OutputChunk * outputs, const unsigned int output_size){
	printlog("------------- merging data in GPU -----------------\n");
	//------------------------------------------------
	// calculate how much data to copy at one time
	//------------------------------------------------
	// assume evenly distributed keys for now
	large_size_t total_data_size=0;
	for(int i=0;i<output_size;i++){
		OutputChunk output=outputs[i];
		int4 index=output.index[output.count-1];
		total_data_size+= index.x+index.y;	// keys
		total_data_size+= index.z+index.w;	// vals
		total_data_size+= output.count*sizeof(int4);	// index
		total_data_size+= output.num_buckets*sizeof(int2);	// buckets
	}
	printlog("do_merge(): total_data_size=%d\n",total_data_size);
	unsigned int num_hash_buckets=g_state.num_hash_buckets;
	// get GPU num
	int gpu_num=0;
	CE(cudaGetDevice(&gpu_num));
	unsigned int global_mem_size=getGlobalMemSize(gpu_num);
	unsigned int data_per_piece=global_mem_size/4;
	unsigned int num_pieces=(total_data_size+data_per_piece-1)/data_per_piece;
	unsigned int buckets_per_piece=(num_hash_buckets+num_pieces)/num_pieces;
	vector<OutputChunk> final_outputs;
	unsigned int key_offset=0;
	unsigned int val_offset=0;
	unsigned int index_offset=0;
	//------------------------------------------------
	// process the pieces one by one
	//------------------------------------------------
	for(unsigned int bucket_start=0; bucket_start<num_hash_buckets; bucket_start+=buckets_per_piece){
		// calculate number of actual buckets to process
		unsigned int bucket_end=bucket_start+buckets_per_piece-1;
		if(bucket_end>=num_hash_buckets)
			bucket_end=num_hash_buckets-1;
		unsigned int num_buckets_to_process=bucket_end-bucket_start+1;
		// setup memory allocator and hash table
		GlobalDeviceState h_device_state;
		memcpyFromSymbol(&h_device_state, g_device_state);
		// initialize memory allocator
		unsigned int memory_pool_size=getAllocatableMemSize(gpu_num);
		SMA_Init(memory_pool_size-RESERVED_MEM_SIZE-::get_global_data_size());	//reserve some space for input data
		// create hash map
		h_device_state.local_combine=g_state.local_combine;
		h_device_state.hash_table=newHashMultiMap(num_buckets_to_process);
		memcpyToSymbol(g_device_state, &h_device_state);

		//----------------------
		// copy chunk onto GPU
		//----------------------
		for(int i=0;i<output_size;i++){
			OutputChunk output=outputs[i];
			unsigned int index_start=output.buckets[bucket_start].x;
			unsigned int index_end=output.buckets[bucket_end].x+output.buckets[bucket_end].y-1;
			unsigned int index_size=index_end-index_start+1;
			if(index_size!=0){	// copy only when there is data
				unsigned int key_start=output.index[index_start].x;
				unsigned int key_size=output.index[index_end].x+output.index[index_end].y-key_start;
				unsigned int val_start=output.index[index_start].z;
				unsigned int val_size=output.index[index_end].z+output.index[index_end].w-val_start;
				// copy data
				char * d_keys=(char*)dMalloc(key_size);
				char * d_vals=(char*)dMalloc(val_size);
				int4 * d_index=(int4*)dMalloc(sizeof(int4)*index_size);
				int2 * d_buckets=(int2*)dMalloc(sizeof(int2)*num_buckets_to_process);
				memcpyH2D(d_keys, output.keys+key_start, key_size);
				memcpyH2D(d_vals, output.vals+val_start, val_size);
				memcpyH2D(d_index, output.index+index_start, sizeof(int4)*index_size);
				memcpyH2D(d_buckets, output.buckets+bucket_start, sizeof(int2)*num_buckets_to_process);
				// insert into hash
				copy_chunk_to_hash<<<g_state.reduce_grid_dim, g_state.reduce_block_dim>>>(d_keys,
							d_vals, d_index, d_buckets, num_buckets_to_process);
				printlog("output num %d copied\n",i);
				CUT_CHECK_ERROR("copy chunk to hash.");
				// delete data
				dFree(d_keys, key_size);
				dFree(d_vals, val_size);
				dFree(d_index, sizeof(int4)*index_size);
				dFree(d_buckets, sizeof(int2)*num_buckets_to_process);
			}
		}
		printlog("copy chunk to hash done\n");
		printtime("copy time: %f \n", time_elapsed());
		CUT_CHECK_ERROR("copy chunk to hash.");

		//--------------------
		// reduce
		//--------------------
		// allocate array for counting reduce space
		int num_hash_buckets=num_buckets_to_process;
		unsigned int * d_key_size_per_bucket=(unsigned int*)SMA_Malloc_From_Host(num_hash_buckets*sizeof(unsigned int));
		unsigned int * d_val_size_per_bucket=(unsigned int*)SMA_Malloc_From_Host(num_hash_buckets*sizeof(unsigned int));
		unsigned int * d_num_pairs_per_bucket=(unsigned int*)SMA_Malloc_From_Host(num_hash_buckets*sizeof(unsigned int));
		dMemset(d_key_size_per_bucket, 0, num_hash_buckets*sizeof(unsigned int));	
		dMemset(d_val_size_per_bucket, 0, num_hash_buckets*sizeof(unsigned int));	
		dMemset(d_num_pairs_per_bucket, 0, num_hash_buckets*sizeof(unsigned int));	
		h_device_state.key_size_per_bucket=d_key_size_per_bucket;
		h_device_state.val_size_per_bucket=d_val_size_per_bucket;
		h_device_state.num_pairs_per_bucket=d_num_pairs_per_bucket;
		memcpyToSymbol(g_device_state, &h_device_state);	
		// invoke reduce and count reduce space at the same time
		reduce_worker<<<g_state.reduce_grid_dim, g_state.reduce_block_dim>>>();
		printlog("reduce done\n");
		printtime("reduce time: %f \n", time_elapsed());
		CUT_CHECK_ERROR("reduce worker.");

		//----------------------------
		// copy output data into array
		//----------------------------
		// prefix sum the arrays
		unsigned int * d_key_start_per_bucket=(unsigned int*)SMA_Malloc_From_Host(num_hash_buckets*sizeof(unsigned int));
		unsigned int * d_val_start_per_bucket=(unsigned int*)SMA_Malloc_From_Host(num_hash_buckets*sizeof(unsigned int));
		unsigned int * d_index_start_per_bucket=(unsigned int*)SMA_Malloc_From_Host(num_hash_buckets*sizeof(unsigned int));
		unsigned int total_key_size=prefix_sum(d_key_size_per_bucket, d_key_start_per_bucket, num_hash_buckets);
		unsigned int total_val_size=prefix_sum(d_val_size_per_bucket, d_val_start_per_bucket, num_hash_buckets);
		unsigned int total_num_pairs=prefix_sum(d_num_pairs_per_bucket, d_index_start_per_bucket, num_hash_buckets);
		printlog("total_key_size=%d\n",total_key_size);
		printlog("total_val_size=%d\n",total_val_size);
		printlog("total_num_pairs=%d\n",total_num_pairs);
		// allocate the output array
		char * d_output_keys=(char *)SMA_Malloc_From_Host(total_key_size);
		char * d_output_vals=(char *)SMA_Malloc_From_Host(total_val_size);
		int4 * d_output_index=(int4 *)SMA_Malloc_From_Host(total_num_pairs*sizeof(int4));
		int2 * d_output_buckets=(int2 *)SMA_Malloc_From_Host(num_buckets_to_process*sizeof(int2));
		h_device_state.output_keys=d_output_keys;
		h_device_state.output_vals=d_output_vals;
		h_device_state.output_index=d_output_index;
		h_device_state.output_buckets=d_output_buckets;
		memcpyToSymbol(g_device_state, &h_device_state);
		// copy hash to array
		copy_hash_to_array_with_offset<<<g_state.reduce_grid_dim, g_state.reduce_block_dim>>>(d_key_start_per_bucket, d_val_start_per_bucket, d_index_start_per_bucket, key_offset, val_offset, index_offset);
		// ajust offset
		key_offset+=total_key_size;
		val_offset+=total_val_size;
		index_offset+=total_num_pairs;
		printlog("copy hash to array done\n");
		printtime("hash to array: %f \n", time_elapsed());
		CUT_CHECK_ERROR("copy hash to array.");
		// copy output to CPU memory, and clear the entire memory pool
		char * h_output_keys=(char *)malloc(total_key_size);
		char * h_output_vals=(char *)malloc(total_val_size);
		int4 * h_output_index=(int4 *)malloc(total_num_pairs*sizeof(int4));
		int2 * h_output_buckets=(int2 *)malloc(num_buckets_to_process*sizeof(int2));
		memcpyD2H(h_output_keys, d_output_keys, total_key_size);
		memcpyD2H(h_output_vals, d_output_vals, total_val_size);
		memcpyD2H(h_output_index, d_output_index, total_num_pairs*sizeof(int4));
		memcpyD2H(h_output_buckets, d_output_buckets, num_buckets_to_process*sizeof(int2));
		// destroy memory pool
		SMA_Destroy();

		// record the output
		OutputChunk final_out;
		final_out.keys=h_output_keys;
		final_out.vals=h_output_vals;
		final_out.index=h_output_index;
		final_out.buckets=h_output_buckets;
		final_out.count=total_num_pairs;
		final_out.num_buckets=num_buckets_to_process;
		final_outputs.push_back(final_out);
		printtime("finish: %f \n", time_elapsed());
	}
	//------------------------------------------------
	// put the pieces together
	//------------------------------------------------
	if(final_outputs.size()<=1){
		OutputChunk output=final_outputs[0];
		g_state.output_keys=output.keys;
		g_state.output_vals=output.vals;
		g_state.output_index=output.index;
		g_state.output_buckets=output.buckets;
		g_state.output_num_pairs=output.count;
	}
	else{
		char * final_keys=(char*)malloc(key_offset);
		char * final_vals=(char*)malloc(val_offset);
		int4 * final_index=(int4*)malloc(index_offset*sizeof(int4));
		int2 * final_buckets=(int2*)malloc(num_hash_buckets*sizeof(int2));
		key_offset=0;
		val_offset=0;
		index_offset=0;
		unsigned int bucket_offset=0;
		for(int i=0;i<final_outputs.size();i++){
			OutputChunk output=final_outputs[i];
			int4 last_index=output.index[output.count-1];
			unsigned int keysize=last_index.x+last_index.y;
			unsigned int valsize=last_index.z+last_index.w;
			memcpy(final_keys+key_offset, output.keys, keysize);
			memcpy(final_vals+val_offset, output.vals, valsize);
			memcpy(final_index+index_offset, output.index, sizeof(int4)*output.count);
			memcpy(final_buckets+bucket_offset, output.buckets, sizeof(int2)*output.num_buckets);
			key_offset+=keysize;
			val_offset+=valsize;
			index_offset+=output.count;
			bucket_offset+=output.num_buckets;
			free(output.keys);
			free(output.vals);
			free(output.index);
			free(output.buckets);
		}
		g_state.output_keys=final_keys;
		g_state.output_vals=final_vals;
		g_state.output_index=final_index;
		g_state.output_buckets=final_buckets;
		g_state.output_num_pairs=index_offset;
		assert(bucket_offset==g_state.num_hash_buckets);
		printtime("put pieces together: %f\n", time_elapsed());
	}
}

OutputChunk HMMSchedulerGPU::get_output(){
	OutputChunk output;
	output.keys=g_state.output_keys;
	output.vals=g_state.output_vals;
	output.index=g_state.output_index;
	output.count=g_state.output_num_pairs;
	output.buckets=g_state.output_buckets;
	output.num_buckets=g_state.num_hash_buckets;

	g_state.output_keys=NULL;
	g_state.output_vals=NULL;
	g_state.output_index=NULL;
	g_state.output_buckets=NULL;
	return output;
}

void HMMSchedulerGPU::finish_scheduler(){
	free(g_state.output_keys);
	g_state.output_keys=NULL;
	free(g_state.output_vals);
	g_state.output_vals=NULL;
	free(g_state.output_index);
	g_state.output_index=NULL;
	free(g_state.output_buckets);
	g_state.output_buckets=NULL;

	// delete device global data
	GlobalDeviceState h_device_state;
	memcpyFromSymbol(&h_device_state, g_device_state);
	if(g_state.gd_initialized){
//		::free_gpu_global_data(h_device_state.global_data);
	}
}
};

