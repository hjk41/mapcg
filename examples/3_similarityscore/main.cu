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
#include <assert.h>
using namespace std;

#include "HMM/HMMMain.h"
#include "HMM/UtilLib/HMMCommUtil.h"
#include "HMM/UtilLib/HMMCommUtilTemplate.h"
#include "DS.h"

float * GenVectors(int numVectors, int numDims){
	srand(10);
	float * buf=(float*)malloc(sizeof(float)*numVectors*numDims);
	for(int i=0;i<numVectors*numDims;i++)
		buf[i]=rand()%100;
	return buf;
}

int main(int argc, char * argv[]){
double t1=get_time();

	unsigned int num_points;
	unsigned int num_dims;
	if(!get_opt(argc,argv,"np", num_points) || !get_opt(argc,argv,"nd",num_dims)){
		cout<<"usage: "<<argv[0]<<" -np num_points -nd num_dims"<<endl;
		return 1;
	}

	float * points=GenVectors(num_points, num_dims);

	HMMSchedulerSpec args(argc,argv);
	unsigned int num_inputs=num_points*num_points;
	point_pair_t * input=new point_pair_t[num_inputs];
	unsigned int index=0;
	for(int i=0;i<num_points;i++){
		for(int j=0;j<num_points;j++){
			input[index].x=i;
			input[index].y=j;
			index++;
		}
	}	args.input=input;
	args.input_size=num_inputs*sizeof(point_pair_t);
        if(args.unit_size>0)
		args.unit_size*=sizeof(point_pair_t);
	else
		args.unit_size=sizeof(point_pair_t);
	args.num_hash_buckets=num_points*num_points;
	args.PrintArgs();

	global_data_t global_data;
	global_data.points=points;
	global_data.num_points=num_points;
	global_data.num_dims=num_dims;
	args.global_data=&global_data;
	DECLARE_GLOBAL_ARRAY(points, sizeof(float)*num_points*num_dims);


	HMMMainScheduler scheduler(args);
double t2=get_time();
cout<<"init: "<<t2-t1<<endl;

	scheduler.do_map_reduce();
	
double t3=get_time();
cout<<"==== total time: "<<t3-t2<<endl;

	OutputChunk output=scheduler.get_output();
	cout<<"total output: "<<output.count<<endl;
	for(int i=0;i<output.count && i<10;i++){
		int keyIdx=output.index[i].x;
		int valIdx=output.index[i].z;
		unsigned int key=*(unsigned int*)((char*)(output.keys)+keyIdx);
		float val=*(float*)((char*)(output.vals)+valIdx);
		cout<<"("<<key<<","<<val<<")"<<endl;
	}

	return 0;
}
