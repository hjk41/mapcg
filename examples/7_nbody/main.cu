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

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include "time.h"
#include "math.h"
#include <vector>
#include <assert.h>
using namespace std;

#include "HMM/HMMMain.h"
#include "HMM/UtilLib/HMMCommUtil.h"
#include "HMM/UtilLib/HMMCommUtilTemplate.h"
#include "DS.h"

const int mode=1024;

body_t * gen_body(unsigned int n){
	srand(1000);
	body_t * b=new body_t[n];
	for(int i=0;i<n;i++){
		b[i].pos_x=rand()%mode;
		b[i].pos_y=rand()%mode;
		b[i].pos_z=rand()%mode;
		b[i].mass=rand();
		b[i].vel_x=0;
		b[i].vel_y=0;
		b[i].vel_z=0;
	}
	return b;
}

int main(int argc, char **argv)
{   
	//----------------------------------------------
	//configuration
	//----------------------------------------------
	unsigned int num_body;
	if(!get_opt(argc,argv, "n", num_body)){
		cout<<"usage: "<<argv[0]<<" -n num_bodies"<<endl;
		return 1;
	}	

	// generate input array
	task_t * tasks=new task_t[num_body];
	for(int i=0;i<num_body;i++){
		tasks[i]=i;
	}

	HMMSchedulerSpec args(argc,argv);
	args.input=tasks;
	args.input_size=sizeof(task_t)*num_body;
	if(args.unit_size>0)
		args.unit_size*=sizeof(task_t);
	else
		args.unit_size=sizeof(task_t);
	args.num_hash_buckets=num_body;
	args.PrintArgs();

	// generate initial status
	body_t * bodies=gen_body(num_body);
/*	printf("init pos: ======================\n");
	for(int i=0;i<num_body && i<10;i++){
		printf("(%f, %f, %f)\n", bodies[i].pos_x, bodies[i].pos_y, bodies[i].pos_z);
	}
	printf("init pos: ======================\n");
*/
	global_data_t global_data;
	global_data.bodies=bodies;
	global_data.delta=0.0001;
	global_data.soften=0.0001;
	global_data.num_body=num_body;
	args.global_data=&global_data;

	DECLARE_GLOBAL_ARRAY(bodies, num_body*sizeof(body_t));

	HMMMainScheduler scheduler(args);
	double time=0;
	for(int i=0;i<1;i++){
		scheduler.init(args);
		double t1=get_time();
		scheduler.do_map_reduce();
		double t2=get_time();
		time+=t2-t1;
		OutputChunk output=scheduler.get_output();
		if(output.count!=0){
			memcpy(bodies, output.vals, sizeof(body_t)*num_body);
			UPDATE_GLOBAL_ARRAY(bodies);
		}
		else{
			cout<<"no more moves, stop"<<endl;
			break;
		}
	}
	cout<<"total time: "<<time<<endl;

	OutputChunk output=scheduler.get_output();
	cout<<"num outputs: "<<output.count<<endl;
	for(int i=0;i<num_body && i<10;i++){
		printf("(%f, %f, %f)\n", bodies[i].pos_x, bodies[i].pos_y, bodies[i].pos_z);
	}

	return 0;
}
