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

#include <string.h>
#include "HMMSchedulerCPU.h"
#include "HMMHashTableCPU.h"
#include "HMMSchedulerCPU.h"
#include "HMMUtilCPU.h"
#include "HMMSMACPU.h"
#include "HMMSortCPU.h"
#include "../../mapreduce.h"

#include <assert.h>
#include <iostream>
#include <vector>

//#define __TIMING__
//#define __DEBUG__

#ifdef __TIMING__
#include <sys/time.h>
#endif


namespace HMM_CPU{


/*****************************************************
  timing and logging mechanism
*****************************************************/
#ifdef __TIMING__
double HMMSchedulerCPU::my_time(){
	timeval t;
	gettimeofday(&t,NULL);
	return (double)t.tv_sec+(double)t.tv_usec/1000000;
}

double HMMSchedulerCPU::time_elapsed(){
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
// setup 
//------------------------
void HMMSchedulerCPU::init_scheduler(const HMMSchedulerSpec & args) 
{
#ifdef __TIMING
	last_t=my_time();
#endif
	// copy the arguments
	assert(args.input);
	input=args.input;

	assert(args.input_size>0);
	input_size=args.input_size;

	assert(args.unit_size>0);
	unit_size=args.unit_size;

	local_combine=args.local_combine;

	assert(args.num_hash_buckets>0);
	num_hash_buckets=args.num_hash_buckets;

	assert(args.cpu_threads>0);
	num_threads=args.cpu_threads;

	assert(args.mi_ratio>0);
	mi_ratio=args.mi_ratio;

	sort_output=args.sort_output;

	global_data=args.global_data;

	// setup pointers
	output_keys=NULL;
	output_vals=NULL;
	output_index=NULL;
	output_buckets=NULL;

	// set state as initialized
	initialized=true;
}

//------------------------------------------
// do a map_reduce iteration
//------------------------------------------
void HMMSchedulerCPU::do_map_reduce(){
	assert(initialized);
	// clear result from last run
	if(output_keys){
		delete[](output_keys);
		output_keys=NULL;
	}
	if(output_vals){
		delete[](output_vals);
		output_vals=NULL;
	}
	if(output_index){
		delete[](output_index);
		output_index=NULL;
	}
	if(output_buckets){
		delete[](output_buckets);
		output_index=NULL;
	}
	// do map/reduce
	schedule_tasks();
}

// -----------------------
// schedule the tasks
//------------------------
void HMMSchedulerCPU::schedule_tasks()
{
printtime("init time: %f \n", time_elapsed());
	// set number of threads to use
	omp_set_num_threads(num_threads);
	SMA_Init(num_threads);
	// create hash maps
	hash_tables=new HashMultiMap*[num_threads];
	for(int i=0;i<num_threads;i++)
		hash_tables[i]=newHashMultiMap(num_hash_buckets);

	//----------
	// map
	//----------
	// start map
	unsigned int num_units=(input_size+unit_size-1)/unit_size;
	#pragma omp parallel for
	for(int i=0;i<num_units;i++){
		const void * ptr;
		unsigned int size;
		slice(input, input_size, unit_size, i, ptr, size);
		map(ptr, size, *(global_data));
	}
printlog("map done\n");
printtime("map time: %f \n", time_elapsed());

	//----------
	// reduce
	//----------
	// allocate array for counting reduce space
	key_size_per_bucket=new unsigned int[num_hash_buckets];
	val_size_per_bucket=new unsigned int[num_hash_buckets];
	num_pairs_per_bucket=new unsigned int[num_hash_buckets];
	memset(key_size_per_bucket, 0, num_hash_buckets*sizeof(unsigned int));
	memset(val_size_per_bucket, 0, num_hash_buckets*sizeof(unsigned int));
	memset(num_pairs_per_bucket, 0, num_hash_buckets*sizeof(unsigned int));	
	if(!local_combine){
		// invoke reduce and count reduce space at the same time
		#pragma omp parallel for 
		for(int i=0;i<num_hash_buckets;i++){
			HashMultiMap & merge_table=*hash_tables[0];
			for(int thread=1;thread<num_threads;thread++){
				HashMultiMap & curr_table=*hash_tables[thread];
				KeyListNode * first_new_key=NULL;
				KeyListNode * last_new_key=NULL;
				for(KeyListNode * curr_key=curr_table.buckets[i].head; curr_key!=NULL; curr_key=curr_key->next){
					// merge the key_vals list together
					KeyListNode * merge_node=NULL;
					for(merge_node=merge_table.buckets[i].head; merge_node!=NULL; merge_node=merge_node->next){
						if(inter_key_equal(curr_key->data.key.val, curr_key->data.key.size, merge_node->data.key.val, merge_node->data.key.size)){
							// merge the lists, add the current list to the head of merge list
							ValueListNode * v_node=curr_key->data.vlist.head;
							while(v_node->next!=NULL){
								v_node=v_node->next;
							}
							v_node->next=merge_node->data.vlist.head;
							merge_node->data.vlist.head=curr_key->data.vlist.head;
							// remove this key from the list
							if(first_new_key!=NULL){
								last_new_key->next=curr_key->next;
							}
							break;
						}
					}
					if(merge_node==NULL){
						// not found, this is a new key
						if(first_new_key==NULL)
							first_new_key=curr_key;
						last_new_key=curr_key;
					}
				}
				if(first_new_key!=NULL){
					last_new_key->next=merge_table.buckets[i].head;
					merge_table.buckets[i].head=first_new_key;
				}
			}
			// invoke reduce
			for(KeysIter kit=merge_table.getBucket(i); kit; ++kit){
				HMMKVs_t kvlist(kit);
				reduce(kvlist, *(global_data));
			}
			// count keysize and valsize
			for(KeyListNode * node=merge_table.buckets[i].head; node!=NULL; node=node->next){
				key_size_per_bucket[i]+=node->data.key.size;
				val_size_per_bucket[i]+=node->data.vlist.head->data.size;
				num_pairs_per_bucket[i]+=1;
			}
		}
	}
	else{
		// invoke local_combine
		#pragma omp parallel for 
		for(int i=0;i<num_hash_buckets;i++){
			HashMultiMap * merge_table=hash_tables[0];
			for(int thread=1;thread<num_threads;thread++){
				HashMultiMap * curr_table=hash_tables[thread];
				KeyListNode * first_new_key=NULL;
				KeyListNode * last_new_key=NULL;
				for(KeyListNode * curr_key=curr_table->buckets[i].head; curr_key!=NULL; curr_key=curr_key->next){
					// merge the key_vals list together
					KeyListNode * merge_node=NULL;
					for(merge_node=merge_table->buckets[i].head; merge_node!=NULL; merge_node=merge_node->next){
						if(inter_key_equal(curr_key->data.key.val, curr_key->data.key.size, merge_node->data.key.val, merge_node->data.key.size)){
							// local_combine
							combine(&(merge_node->data.vlist.head->data.val[0]), &(curr_key->data.vlist.head->data.val)[0]);
							// remove this key from the list
							if(first_new_key!=NULL){
								last_new_key->next=curr_key->next;
							}
							break;
						}
					}
					if(merge_node==NULL){
						// not found, this is a new key
						if(first_new_key==NULL)
							first_new_key=curr_key;
						last_new_key=curr_key;
					}
				}
				if(first_new_key!=NULL){
					KeyListNode *&h=merge_table->buckets[i].head;
					last_new_key->next=h;
					h=last_new_key;
				}
			}
			// count keysize and valsize
			for(KeyListNode * node=merge_table->buckets[i].head; node!=NULL; node=node->next){
				key_size_per_bucket[i]+=node->data.key.size;
				val_size_per_bucket[i]+=node->data.vlist.head->data.size;
				num_pairs_per_bucket[i]+=1;
			}
		}
	}
printlog("reduce done\n");
printtime("reduce time: %f \n", time_elapsed());
	// prefix sum the arrays
	unsigned int * key_starts=new unsigned int[num_hash_buckets];
	unsigned int * val_starts=new unsigned int[num_hash_buckets];
	unsigned int * index_starts=new unsigned int[num_hash_buckets];
	unsigned int key_start=0;
	unsigned int val_start=0;
	unsigned int index_start=0;
	for(int i=0;i<num_hash_buckets;i++){
		key_starts[i]=key_start;
		val_starts[i]=val_start;
		index_starts[i]=index_start;
		key_start+=key_size_per_bucket[i];
		val_start+=val_size_per_bucket[i];
		index_start+=num_pairs_per_bucket[i];
	}
	num_output_pairs=index_start;
	delete[] key_size_per_bucket;
	delete[] val_size_per_bucket;
	delete[] num_pairs_per_bucket;
	printlog("total_key_size=%d\n",key_start);
	printlog("total_val_size=%d\n",val_start);
	printlog("total_num_pairs=%d\n",index_start);

	// allocate the output array
	output_keys=new char[key_start];
	output_vals=new char[val_start];
	output_index=new int4[index_start];
	output_buckets=new int2[num_hash_buckets];
	// copy output data into array
	#pragma omp parallel for 
	for(int i=0;i<num_hash_buckets;i++){
		int keyIndex=key_starts[i];
		int valIndex=val_starts[i];
		int idxIndex=index_starts[i];
		output_buckets[i].x=idxIndex;
		for(KeysIter kit=hash_tables[0]->getBucket(i); kit; ++kit){
			const void *key, *val;
			unsigned int keysize, valsize;
			kit.getKey(key, keysize);
			kit.getValues().getValue(val, valsize);
			copyVal(output_keys+keyIndex, key, keysize);
			copyVal(output_vals+valIndex, val, valsize);
			output_index[idxIndex].x=keyIndex;
			output_index[idxIndex].y=keysize;
			output_index[idxIndex].z=valIndex;
			output_index[idxIndex].w=valsize;
			keyIndex+=keysize;
			valIndex+=valsize;
			idxIndex++;
		}
		output_buckets[i].y=idxIndex-output_buckets[i].x;
	}
	delete[] key_starts;
	delete[] val_starts;
	delete[] index_starts;
	SMA_Destroy();
	for(int i=0;i<num_threads;i++)
		delHashMultiMap(hash_tables[i]);
	delete[] hash_tables;
	printlog("copy hash to array done\n");
	printtime("hash to array: %f \n", time_elapsed());

/*	if(sort_output){
		sort_chunk(output_keys, output_vals, output_index, num_output_pairs);
	}
printtime("sort: %f \n", time_elapsed());*/
};

//--------------------------------------
// do merge
//--------------------------------------
void HMMSchedulerCPU::do_merge(const OutputChunk * outputs, const unsigned int output_size)
{
printtime("init time: %f \n", time_elapsed());
	// set number of threads to use
	omp_set_num_threads(num_threads);
	SMA_Init(num_threads);
	// create hash map
	hash_tables=new HashMultiMap*[1];
	hash_tables[0]=newHashMultiMap(num_hash_buckets);
	HashMultiMap * hash_table=hash_tables[0];

	if(!local_combine){
		//---------------------------
		// copy output to hash table
		//---------------------------
		#pragma omp parallel for
		for(int i=0;i<num_hash_buckets;i++){
			for(int j=0;j<output_size;j++){
				const OutputChunk & output=outputs[j];
				for(int k=output.buckets[i].x; k<output.buckets[i].x+output.buckets[i].y; k++){
					int4 index=output.index[k];
					printlog("inserting %d, %d\n", *(int*)(output.keys+index.x), *(int*)(output.vals+index.z));
					hash_table->insert(output.keys+index.x, index.y, output.vals+index.z, index.w);
				}
			}
		}
		printlog("copy output to hash done\n");
		printtime("copy time: %f \n", time_elapsed());

		//----------
		// reduce
		//----------
		// allocate array for counting reduce space
		key_size_per_bucket=new unsigned int[num_threads];
		val_size_per_bucket=new unsigned int[num_threads];
		num_pairs_per_bucket=new unsigned int[num_threads];
		memset(key_size_per_bucket, 0, num_threads*sizeof(unsigned int));
		memset(val_size_per_bucket, 0, num_threads*sizeof(unsigned int));
		memset(num_pairs_per_bucket, 0, num_threads*sizeof(unsigned int));	
		// invoke reduce and count reduce space at the same time
		#pragma omp parallel for 
		for(int i=0;i<hash_table->num_buckets;i++){
			int threadID=omp_get_thread_num();
			for(KeysIter kit=hash_table->getBucket(i); kit; ++kit){
				HMMKVs_t kvlist(kit);
				reduce(kvlist, *(global_data));
			}
			for(KeyListNode * node=hash_table->buckets[i].head; node!=NULL; node=node->next){
				key_size_per_bucket[threadID]+=node->data.key.size;
				val_size_per_bucket[threadID]+=node->data.vlist.head->data.size;
				num_pairs_per_bucket[threadID]+=1;
			}
		}
		printlog("reduce done\n");
		printtime("reduce time: %f \n", time_elapsed());
	}
	else{
		key_size_per_bucket=new unsigned int[num_threads];
		val_size_per_bucket=new unsigned int[num_threads];
		num_pairs_per_bucket=new unsigned int[num_threads];
		memset(key_size_per_bucket, 0, num_threads*sizeof(unsigned int));
		memset(val_size_per_bucket, 0, num_threads*sizeof(unsigned int));
		memset(num_pairs_per_bucket, 0, num_threads*sizeof(unsigned int));	
		#pragma omp parallel for 
		for(int i=0;i<num_hash_buckets;i++){
			int threadID=omp_get_thread_num();
			for(int j=0;j<output_size;j++){
				const OutputChunk & output=outputs[j];
				for(int k=output.buckets[i].x; k<output.buckets[i].x+output.buckets[i].y; k++){
					int4 index=output.index[k];
					printlog("inserting %d, %d\n", *(int*)(output.keys+index.x), *(int*)(output.vals+index.z));
//					hash_table->insert(output.keys+index.x, index.y, output.vals+index.z, index.w);
					ValueList & vlist=hash_table->buckets[i].getValueList(output.keys+index.x, index.y);
					if(vlist.head!=NULL){
						combine(vlist.head->data.val, output.vals+index.z);
					}
					else{
						vlist.insert(output.vals+index.z, index.w);
						key_size_per_bucket[threadID]+=index.y;
						val_size_per_bucket[threadID]+=index.w;
						num_pairs_per_bucket[threadID]+=1;
					}
				}
			}
		}
		printlog("copy output to hash done\n");
		printtime("copy time: %f \n", time_elapsed());
	}
	// prefix sum the arrays
	unsigned int * key_starts=new unsigned int[num_threads];
	unsigned int * val_starts=new unsigned int[num_threads];
	unsigned int * index_starts=new unsigned int[num_threads];
	unsigned int key_start=0;
	unsigned int val_start=0;
	unsigned int index_start=0;
	for(int i=0;i<num_threads;i++){
		key_starts[i]=key_start;
		val_starts[i]=val_start;
		index_starts[i]=index_start;
		key_start+=key_size_per_bucket[i];
		val_start+=val_size_per_bucket[i];
		index_start+=num_pairs_per_bucket[i];
	}
	num_output_pairs=index_start;
	delete[] key_size_per_bucket;
	delete[] val_size_per_bucket;
	delete[] num_pairs_per_bucket;
printlog("total_key_size=%d\n",key_start);
printlog("total_val_size=%d\n",val_start);
printlog("total_num_pairs=%d\n",index_start);
	// allocate the output array
	output_keys=new char[key_start];
	output_vals=new char[val_start];
	output_index=new int4[index_start];
	output_buckets=new int2[num_hash_buckets];
	// copy output data into array
	#pragma omp parallel for 
	for(int i=0;i<hash_table->num_buckets;i++){
		int threadID=omp_get_thread_num();
		int keyIndex=key_starts[threadID];
		int valIndex=val_starts[threadID];
		int idxIndex=index_starts[threadID];
		output_buckets[i].x=idxIndex;
		for(KeysIter kit=hash_table->getBucket(i); kit; ++kit){
			const void *key, *val;
			unsigned int keysize, valsize;
			kit.getKey(key, keysize);
			kit.getValues().getValue(val, valsize);
			copyVal(output_keys+keyIndex, key, keysize);
			copyVal(output_vals+valIndex, val, valsize);
			output_index[idxIndex].x=keyIndex;
			output_index[idxIndex].y=keysize;
			output_index[idxIndex].z=valIndex;
			output_index[idxIndex].w=valsize;
			keyIndex+=keysize;
			valIndex+=valsize;
			idxIndex++;
		}
		output_buckets[i].y=idxIndex-output_buckets[i].x;
		key_starts[threadID]=keyIndex;
		val_starts[threadID]=valIndex;
		index_starts[threadID]=idxIndex;
	}
	delete[] key_starts;
	delete[] val_starts;
	delete[] index_starts;

	SMA_Destroy();
	delHashMultiMap(hash_tables[0]);
	delete[] hash_tables;
printlog("copy hash to array done\n");
printtime("hash to array: %f \n", time_elapsed());

	if(sort_output){
		sort_chunk(output_keys, output_vals, output_index, num_output_pairs);
	}
printtime("sort: %f \n", time_elapsed());
}

OutputChunk HMMSchedulerCPU::get_output(){
	OutputChunk output;
	output.keys=output_keys;
	output.vals=output_vals;
	output.index=output_index;
	output.count=num_output_pairs;
	output.buckets=output_buckets;
	output.num_buckets=num_hash_buckets;
	
	output_keys=NULL;
	output_vals=NULL;
	output_index=NULL;
	output_buckets=NULL;
	num_output_pairs=0;
	return output;
}

void HMMSchedulerCPU::finish_scheduler(){
	delete[](output_keys);
	output_keys=NULL;
	delete[](output_vals);
	output_vals=NULL;
	delete[](output_index);
	output_index=NULL;
	delete[](output_buckets);
	output_buckets=NULL;
}

void HMMSchedulerCPU::default_slice(const void * input, const unsigned input_size, const unsigned unit_size, const unsigned index, const void *& ret_ptr, unsigned int & ret_size){
	ret_ptr=(const char *)input+unit_size*index;
	unsigned next_start=(index+1)*unit_size;
	if(next_start>input_size)
		next_start=input_size;
	ret_size=next_start-unit_size*index;
}

void HMMSchedulerCPU::emit_intermediate(const void * key, const unsigned int keysize, const void * val, const unsigned int valsize){
	int threadID=omp_get_thread_num();
	if(!local_combine){
		hash_tables[threadID]->insert(key, keysize, val, valsize);
	}
	else{
	// local_combine the value 
		const void * tmp=val;
		// find if there is already a value
		if(hash_tables[threadID]->findVal(key,keysize,tmp,valsize)){
			combine((void*)tmp, val);
		}
	}
}

// emit a key/value in reduce, should be called only once per reduce
void HMMSchedulerCPU::emit(HMMKVs_t & kvlist, const void * val, const unsigned int valsize){
	kvlist.kit.setValue(val, valsize);
}

};	// namespace HMM_CPU
