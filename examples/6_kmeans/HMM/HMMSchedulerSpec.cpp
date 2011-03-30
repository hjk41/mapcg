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

#include "HMMSchedulerSpec.h"
#include "UtilLib/HMMCommUtilTemplate.h"
#include "UtilLib/HMMCommUtil.h"
#include <stdlib.h>
#include <string.h>
#include <string>
using namespace std;

//=================================
// Scheduler Specification Defaults
//=================================
HMMSchedulerSpec::HMMSchedulerSpec(){
	slice_num=1;
	input=NULL;
	input_size=0;
	unit_size=0;
	global_data=NULL;

	exe_mode=GPU;
	merge_mode=GPU;
	num_gpus=1;

	local_combine=false;

	sort_output=false;
	sort_mode=CPU;

	mi_ratio=1;

	num_hash_buckets=3079;

	cpu_threads=4;

	gpu_map_grid=128;
	gpu_map_block=128;
	gpu_reduce_grid=60;
	gpu_reduce_block=64;
}

HMMSchedulerSpec::HMMSchedulerSpec(const HMMSchedulerSpec &rhs){
	memcpy(this, &rhs, sizeof(HMMSchedulerSpec));
}

void HMMSchedulerSpec::PrintArgs(){
	printf("num_slice\t=%d\n"
		"unit_size\t=%d\n"
		"exe_mode\t=%d\n"
		"num_gpus\t=%d\n"
		"num_hash_buckets\t=%d\n"
		"cpu_threads\t=%d\n",slice_num, unit_size,exe_mode,num_gpus,num_hash_buckets,cpu_threads);
}

HMMSchedulerSpec::HMMSchedulerSpec(int argc, char * argv[]){
	// input cannot be set in command line
	input=NULL;
	input_size=0;
	global_data=NULL;
	local_combine=false;

	// slice size
	slice_num=1;
	get_opt(argc,argv,"num_slice",slice_num);

	// unit size
	unit_size=0;
	get_opt(argc,argv,"unit_size",unit_size);

	// exe mode
	exe_mode=GPU;
	string exe_mode_str="GPU";
	get_opt(argc,argv,"exe_mode",exe_mode_str);
	if(exe_mode_str=="GPU"){
		exe_mode=GPU;
	}
	else if(exe_mode_str=="CPU"){
		exe_mode=CPU;
	}
	else if(exe_mode_str=="CPU_GPU" || exe_mode_str=="GPU_CPU"){
		exe_mode=CPU_GPU;
	}
	else{
		// illegal parameter
		print_error("illegal parameter for -exe_mode, use 'GPU', 'CPU', or 'CPU_GPU'\n");
	}

	// merge mode
	merge_mode=GPU;
	string merge_mode_str="GPU";
	get_opt(argc,argv,"merge_mode",merge_mode_str);
	if(merge_mode_str=="GPU"){
		merge_mode=GPU;
	}
	else if(merge_mode_str=="CPU"){
		merge_mode=CPU;
	}
	else{
		// illegal parameter
		print_error("illegal parameter for -merge_mode, use 'GPU' or 'CPU'\n");
	}


	// number of gpus to use
	num_gpus=1;
	get_opt(argc,argv,"num_gpus",num_gpus);

	// sorting
	sort_output=false;
	get_opt(argc,argv,"sort_output",sort_output);

	sort_mode=CPU;
	string sort_mode_str="CPU";
	get_opt(argc,argv,"sort_mode",sort_mode_str);
	if(sort_mode_str=="GPU"){
		sort_mode=GPU;
	}
	else if(sort_mode_str=="CPU"){
		sort_mode=CPU;
	}
	else{
		// illegal parameter
		print_error("illegal parameter for -sort_mode, use 'GPU' or 'CPU'\n");
	}

	// memory-input ratio
	mi_ratio=1;
	get_opt(argc,argv,"mi_ratio",mi_ratio);

	// number of hash buckets
	num_hash_buckets=3079;
	get_opt(argc,argv,"num_hash_buckets",num_hash_buckets);

	// number of CPU worker threads to spawn
	cpu_threads=4;
	get_opt(argc,argv,"cpu_threads",cpu_threads);

	// GPU thread setting
	gpu_map_grid=128;
	get_opt(argc,argv,"gpu_map_grid",gpu_map_grid);
	gpu_map_block=128;
	get_opt(argc,argv,"gpu_map_block",gpu_map_block);
	gpu_reduce_grid=60;
	get_opt(argc,argv,"gpu_reduce_grid",gpu_reduce_grid);
	gpu_reduce_block=64;
	get_opt(argc,argv,"gpu_reduce_block",gpu_reduce_block);
}

const char * HMMSchedulerSpec::AvailOptions(){
	return "-unit_size <int>\n"
		"-exe_mode <string[CPU|GPU|CPU_GPU]>\n"
		"-num_gpus <int>\n"
		"-sort_output\n"
		"-sort_mode <string[CPU|GPU]>\n"
		"-mi_ratio <float>\n"
		"-num_hash_buckets <int>\n"
		"-cpu_threads <int>\n"
		"-gpu_map_grid <int>\n"
		"-gpu_map_block <int>\n"
		"-gpu_reduce_grid <int>\n"
		"-gpu_reduce_block <int>\n";
}

