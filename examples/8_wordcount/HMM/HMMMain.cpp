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

#include "HMMMain.h"
#include "HMMDS.h"
#include "UtilLib/HMMTSQueue.h"
#include "UtilLib/HMMCommUtil.h"
#include "HMMCPUScheduler.h"
#include "HMMGPUScheduler.h"
#include "HMMGlobalData.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <pthread.h>
#include <stdio.h>
using namespace std;

//=============================
// Scheduler Implementation
//=============================
/*
class HMMMainScheduler:public HMMScheduler{
public:
	//=============================
	// defined in HMMScheduler
	//=============================
	// void set_input(const void *, const unsigned int);
	// void set_unit_size(unsigned int);

	void init(const HMMSchedulerSpec &);
	void do_map_reduce();
	OutputChunk & get_output();
};
*/

void HMMMainScheduler::init(const HMMSchedulerSpec & a){
	args=a;
	// check if arguments are properly set
	assert(args.input);
	assert(args.input_size>0);
	assert(args.unit_size>0);
	assert(args.mi_ratio>0);
	assert(args.num_hash_buckets>0);

	if(args.exe_mode==GPU || args.exe_mode==CPU_GPU){
		assert(args.num_gpus>0);
		assert(args.gpu_map_grid>0);
		assert(args.gpu_map_block>0);
		assert(args.gpu_reduce_grid>0);
		assert(args.gpu_reduce_block>0);
	}

	if(args.exe_mode==CPU || args.exe_mode==CPU_GPU){
		assert(args.cpu_threads>0);
	}

	// see if there is enough GPU
	int n=get_num_gpus();
	args.num_gpus= args.num_gpus<=n ? args.num_gpus : n;

	// reset output
	delete[] output.keys;
	output.keys=NULL;
	delete[] output.vals;
	output.vals=NULL;
	delete[] output.index;
	output.index=NULL;
}

enum SchedulerType{
	SCHEDULER_CPU,
	SCHEDULER_GPU
};

struct Param{
	HMMMainScheduler::JobQueue * job_queue;
	HMMMainScheduler::OutputQueue * output_queue;
	HMMSchedulerSpec * args;
	SchedulerType type;
	int gpu_num;
};

void * scheduler_worker(void * p){
	Param * param=(Param*)p;
	HMMScheduler * scheduler=NULL;
	if(param->type==SCHEDULER_CPU)
		scheduler=new HMMCPUScheduler;
	else
		scheduler=new HMMGPUScheduler(param->gpu_num);
	if(scheduler){
		HMMSchedulerSpec local_args=*(param->args);
		local_args.sort_output=true;
		HMMMainScheduler::Job job;
		while(param->job_queue->pop(job)){
if(param->type==SCHEDULER_CPU)
	printlog("CPU thread get a block\n");
else
	printlog("GPU thread %d get a block\n", param->gpu_num);
sleep(0);
			// process this job
			local_args.input=job.input;
			local_args.input_size=job.input_size;
			scheduler->init(local_args);
			scheduler->do_map_reduce();
			// push the output into the output queue
			param->output_queue->push(scheduler->get_output());
		}
		delete scheduler;
	}
	return 0;
}

void HMMMainScheduler::do_map_reduce(){

init_timer();
	// setup the input job_queue
	JobQueue job_queue;
	OutputQueue output_queue;
	int num_steps=args.slice_num;
	int step=args.unit_size;
	if(num_steps*args.unit_size >= args.input_size){
		step=args.unit_size;
	}
	else{
		step=((args.input_size/args.unit_size)/num_steps)*args.unit_size;
	}
	for(int i=0;;i++){
		if(i*step>=args.input_size)
			break;
		const void * ptr=NULL;
		unsigned int slice_size=0;
		HMM_CPU::HMMSchedulerCPU::slice(args.input, args.input_size, step, i, ptr, slice_size);
		if(slice_size>0){
			Job job;
			job.input=ptr;
			job.input_size=slice_size;
			job_queue.push(job);
		}
	}

printlog("init_time: %f\n",time_elapsed());	
	// spawn CPUScheduler and GPUScheduler threads

	Param para;
	para.job_queue=&job_queue;
	para.output_queue=&output_queue;
	para.args=&args;
	Param * gpu_params=NULL;
	Param * cpu_params=NULL;
	pthread_t * gpu_threads=NULL;
	pthread_t * cpu_threads=NULL;
	if(args.exe_mode==GPU || args.exe_mode==CPU_GPU){
		gpu_params=new Param[args.num_gpus];
		gpu_threads=new pthread_t[args.num_gpus];
		for(int i=0;i<args.num_gpus;i++){
			gpu_params[i]=para;
			gpu_params[i].type=SCHEDULER_GPU;
			gpu_params[i].gpu_num=i;
			pthread_create(&gpu_threads[i], NULL, scheduler_worker, &gpu_params[i]);
 //  		scheduler_worker(&gpu_params[i]);
		}
	}
	if(args.exe_mode==CPU || args.exe_mode==CPU_GPU){
		cpu_params=new Param;
		cpu_threads=new pthread_t;
		*cpu_params=para;
		cpu_params->type=SCHEDULER_CPU;
		pthread_create(cpu_threads, NULL, scheduler_worker, cpu_params);
	}

	// wait till work done
	if(gpu_threads){
		for(int i=0;i<args.num_gpus;i++)
			pthread_join(gpu_threads[i],NULL);
	}
	if(cpu_threads)
		pthread_join(cpu_threads[0],NULL);
	delete[] gpu_params;
	delete[] cpu_params;
	delete[] gpu_threads;
	delete[] cpu_threads;

printlog("mapreduce_time: %f\n",time_elapsed());	
	// merge output
	output=merge_output(output_queue);
printlog("merge_time: %f\n",time_elapsed());	
	// mark all global data as synchronized
	::mark_all_as_clean();
}

OutputChunk HMMMainScheduler::get_output(){
	return output;
}

HMMMainScheduler::~HMMMainScheduler(){
	// destroy the output chunk
	delete[] output.keys;
	delete[] output.vals;
	delete[] output.index;
	output.count=0;
}

OutputChunk HMMMainScheduler::merge_output(OutputQueue & q){
	vector<OutputChunk> outputs;
	OutputChunk output;
	while(q.pop(output)){
		outputs.push_back(output);
	}
	if(outputs.size()==1)
		return outputs[0];
	if(args.merge_mode==CPU){
		printf("merging with CPU...\n");
		HMM_CPU::HMMSchedulerCPU merger;
		merger.init_scheduler(args);
		merger.do_merge(&outputs[0], outputs.size());
		return merger.get_output();
	}
	else{
		printf("merging with GPU...\n");
		HMM_GPU::HMMSchedulerGPU merger;
		merger.init_scheduler(args);
		merger.do_merge(&outputs[0], outputs.size());
		return merger.get_output();
	}
}
