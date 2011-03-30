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

#ifndef HMMSCHEDULERGPU_H
#define HMMSCHEDULERGPU_H

#include <list>
#include <map>
#include "../HMMSchedulerSpec.h"
#include "../HMMDS.h"
#include "../../DS.h"

namespace HMM_GPU{
class HashMultiMap;
// device side state
struct GlobalDeviceState{
	bool local_combine;
	HashMultiMap * hash_table;

	unsigned int * key_size_per_bucket;
	unsigned int * val_size_per_bucket;
	unsigned int * num_pairs_per_bucket;

	char * output_keys;
	char * output_vals;
	int4 * output_index;
	int2 * output_buckets;

	global_data_t * global_data;
};

class HMMSchedulerGPU{
private:
	static const int RESERVED_MEM_SIZE=1*1024*1024;		// magic number. reserved memory size when allocating the memory pool
	struct GlobalState{
		GlobalState():input(NULL),input_size(0),unit_size(0),
			map_grid_dim(256),map_block_dim(256),reduce_grid_dim(256),reduce_block_dim(256),
			num_hash_buckets(3367),sort_output(false),
			output_keys(NULL),output_vals(NULL),output_index(NULL),output_num_pairs(0),output_buckets(NULL),
			global_data(NULL),
			initialized(false),gd_initialized(false){}
		const void * input;		// input data, should be an array
		unsigned int input_size;		// Total # of bytes of data
		unsigned int unit_size;		// # of bytes per split
		bool local_combine;

		unsigned int map_grid_dim;	// grid dim in map
		unsigned int map_block_dim;	// block dim in map

		unsigned int num_hash_buckets;	// number of chunks in hash table

		unsigned int reduce_grid_dim;	// grid dim in reduce
		unsigned int reduce_block_dim;	// block dim in reduce

		bool sort_output;		  // whether we should od merge

		char * output_keys;      // array to final keys
		char * output_vals;      // value
		int4 * output_index;     // final index
		unsigned int output_num_pairs;     // final number of keys
		int2 * output_buckets;

		global_data_t * global_data;
		bool initialized;
		bool gd_initialized;
	};

private:
	GlobalState g_state;
	double last_t;

private:
	// timer, used to log
	double my_time();
	double time_elapsed();

	void schedule_tasks();
public:
	// interface
	void init_scheduler(const HMMSchedulerSpec & args);
	void do_map_reduce();
	void do_merge(const OutputChunk * outputs, const unsigned int size);
	OutputChunk get_output();
	void finish_scheduler();
};

};
#endif
