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
using namespace std;

#include "HMM/HMMMain.h"
#include "HMM/UtilLib/HMMCommUtil.h"
#include "HMM/UtilLib/HMMCommUtilTemplate.h"
#include "DS.h"

#define VECTOR_SPACE    1000

//--------------------------------------------------
//generate data
//--------------------------------------------------
int *GenPoints(int num_points, int dim)
{
	srand(1024);
	int *buf = (int*)malloc(sizeof(int)*num_points*dim);

	for (int i = 0; i < num_points; i++)
		for (int j = 0; j < dim; j++)
			buf[i*dim+j] = rand()%VECTOR_SPACE;
//			buf[i*dim+j] = (i*dim+j)%VECTOR_SPACE;

	return buf;
}

int *GenMeans(int num_means, int dim)
{
	srand(1024);
	int *buf = (int*)malloc(dim*num_means*sizeof(int));
	for (int i = 0; i < num_means; i++)
		for (int j = 0; j < dim; j++)
//			buf[i*dim+j] = (i*dim+j)%VECTOR_SPACE;
			buf[i*dim + j] = rand()%VECTOR_SPACE;
	return buf;
}


int main(int argc, char * argv[]){
double t1=get_time();
	int num_points;
	int dim;
	int num_means;
	int num_iters;
	if (!get_opt(argc, argv, "vn", num_points) ||
	 !get_opt(argc, argv, "mn", num_means) ||
	 !get_opt(argc, argv, "dim", dim) ||
	!get_opt(argc,argv, "ni", num_iters)){
		 printf("Usage: %s -vn INT -mn INT -dim INT\n"
			"\t-vn vector number\n"
			"\t-mn mean number\n"
			"\t-dim vector dimension\n"
			"\t-ni number iteration\n", argv[0]);
		return 1;
	}


	int * points=GenPoints(num_points, dim);
	int * means=GenMeans(num_means, dim);
	int * index=new int[num_points];
	for(int i=0;i<num_points;i++)
		index[i]=i;
	int * pos=new int[num_points];
	for(int i=0;i<num_points;i++)
		pos[i]=-1;

	HMMSchedulerSpec args(argc,argv);
	args.input=index;
	args.input_size=num_points*sizeof(int);
	args.unit_size=sizeof(int);
	args.num_hash_buckets=num_means+num_points;
	args.PrintArgs();

	global_data_t global_data;
	global_data.dim = dim;
	global_data.num_means = num_means;
	global_data.points=points;
	global_data.means = means;
	args.global_data = &global_data;

	DECLARE_GLOBAL_ARRAY(points, sizeof(int)*num_points*dim);
	DECLARE_GLOBAL_ARRAY(means, dim*num_means*sizeof(int));

	HMMMainScheduler scheduler(args);
	double t2,t3;
	double mapreduce_time=0;
	bool changed=true;
	int iter=0;
	while(changed){
		cout<<"------------ iter "<<iter++<<endl;	
		if(iter>num_iters)
			break;
		changed=false;
		scheduler.init(args);
		t2=get_time();
		scheduler.do_map_reduce();
		t3=get_time();
		mapreduce_time+=t3-t2;
		OutputChunk output=scheduler.get_output();

		printf("recCount: %d\n",output.count);
		printf("valSize: %d\n",output.index[output.count-1].z+output.index[output.count-1].w);
		// update means and pos
		// we are using inefficient algorithm to work around a bug in Mars
		int * keys=(int*)output.keys;
		int * vals=(int*)output.vals;
		for(int i=0;i<output.count;i++){
			int4 idx=output.index[i];
			int key=keys[i];
			if(key<num_means){	// this is a new mean
				memcpy(means+key*dim, vals+idx.z, sizeof(int)*dim);
			}
			else{
				int point=key-num_means;
				if(pos[point]!=vals[idx.z]){
					changed=true;
					pos[point]=vals[idx.z];
				}
			}
		}
		UPDATE_GLOBAL_ARRAY(means);

	}

	cout<<"map_reduce: "<<mapreduce_time<<endl;

	for(int i=0;i<10 && i<num_points;i++){
		printf("(%d,%d)\n",i,pos[i]);
	}

	return 0;
}
