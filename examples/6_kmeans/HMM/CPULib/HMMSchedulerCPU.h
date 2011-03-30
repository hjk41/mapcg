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

#ifndef HMMSCHEDULERCPU_H
#define HMMSCHEDULERCPU_H

#include "../HMMDS.h"
#include "../HMMSchedulerSpec.h"
#include "HMMHashTableCPU.h"
#include "../../DS.h"
#include <omp.h>
#include <vector>

namespace HMM_CPU{

class HMMSchedulerCPU{
private:
	const void * input;		// input data, should be an array
	unsigned int input_size;	// Total # of bytes of data
	unsigned int unit_size;		// # of bytes per split
	unsigned int num_threads;	// number of threads to use
	unsigned int num_hash_buckets;	// number of buckets in hash table
	float mi_ratio;			// memory-input ratio 
	bool sort_output;		// whether to sort output
	bool local_combine;			
	global_data_t * global_data;
	bool initialized;

	// hash table
	HashMultiMap ** hash_tables;
	// dump hash into array
	unsigned int * key_size_per_bucket;
	unsigned int * val_size_per_bucket;
	unsigned int * num_pairs_per_bucket;
	// output
	char * output_keys;
	char * output_vals;
	int4 * output_index;
	unsigned int num_output_pairs;
	int2 * output_buckets;

	// timing
	double last_t;
	
private:
	// timer, used to log
	double my_time();
	double time_elapsed();

	// implementation
	void schedule_tasks();

	// worker functions
	void * map_worker(void*);
	void * reduce_worker(void*);
	void * copy_hash_to_array(void*);

	// utility
	static void default_slice(const void * data, const unsigned int size, const unsigned unit_size, const unsigned index, const void * & ret, unsigned int & ret_size);
	void emit_intermediate(const void * key, const unsigned int keysize, const void * val, const unsigned int valsize);
	void emit(HMMKVs_t & kvlist, const void * val, const unsigned int valsize);
	
	// functions defined by user	
	void map(const void * ptr, const unsigned int size, const global_data_t & gd);
	void reduce(HMMKVs_t & kv, const global_data_t & gd);
	void combine(void * old_val, const void * new_val);
public:
	// interface
	void init_scheduler(const HMMSchedulerSpec & args);
	void do_map_reduce();
	OutputChunk get_output();
	void finish_scheduler();

	void do_merge(const OutputChunk * outputs, const unsigned int size);

	// global data manipulation
	void set_global_data(const global_data_t *);
	static void slice(const void * input_array, const unsigned int data_size, const unsigned int unit_size, const unsigned int index, const void * &ret_ptr, unsigned int & ret_size);
};

};	// namespace HMM_CPU
#endif	// #define HMMSCHEDULERCPU_H
