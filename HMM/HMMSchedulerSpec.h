#ifndef HMMSCHEDULERSPEC_H
#define HMMSCHEDULERSPEC_H

//================================
// scheduler specification
//================================

#include "../DS.h"

// execution mode
enum ExeMode{
	CPU,
	GPU,
	CPU_GPU
};

// scheduler specification
struct HMMSchedulerSpec{
public:
	HMMSchedulerSpec();
	HMMSchedulerSpec(const HMMSchedulerSpec & rhs);
	HMMSchedulerSpec(int argc, char * argv[]);     // parse spec from command line arguments
	const char * AvailOptions();
	void PrintArgs();

	//=============================
	// for main scheduler
	unsigned int slice_num;
	// input
	const void * input;
	unsigned int input_size;
	unsigned int unit_size;

	// global data
	global_data_t * global_data;

	// exe mode, use CPU or GPU, or both
	ExeMode exe_mode;
	ExeMode merge_mode;
	// how many gpu to use
	unsigned int num_gpus;

	// do local combine
	bool local_combine;

	// sorting related
	bool sort_output;
	ExeMode sort_mode;

	// memory-input ratio. how many bytes do we need when processing a byte of input?
	float mi_ratio;

	// number of hash buckets
	unsigned int num_hash_buckets;

	//=============================
	// for CPU scheduler
	unsigned int cpu_threads;

	//=============================
	// for GPU scheduler
	unsigned int gpu_map_grid;
	unsigned int gpu_map_block;
	unsigned int gpu_reduce_grid;
	unsigned int gpu_reduce_block;
};


#endif
