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

//--------------------------------------------------
//generate data
//--------------------------------------------------
static float *GenMatrix(int M_ROW_COUNT, int M_COL_COUNT)
{
	float *matrix = (float*)malloc(sizeof(float)*M_ROW_COUNT*M_COL_COUNT);

	srand(10);
	for (int i = 0; i < M_ROW_COUNT; i++)
		for (int j = 0; j < M_COL_COUNT; j++)
			matrix[i*M_COL_COUNT+j] = (float)(rand() % 100);

	return matrix;
}

void TranMatrix(float * m, int row){
	for(int i=0;i<row;i++){
		for(int j=i+1;j<row;j++){
			float tmp=m[i*row+j];
			m[i*row+j]=m[j*row+i];
			m[j*row+i]=tmp;
		}
	}
}

int main(int argc, char * argv[]){
double t1=get_time();

	int dim;
	if(!get_opt(argc,argv,"dim",dim)){
		cout<<"usage: "<<argv[0]<<" -dim dimmension"<<endl;
		return 1;
	}
	cout<<"using dim: "<<dim<<endl;

	pos_t * input=new pos_t[dim*dim];
	pos_t pos;
	for(int i=0;i<dim;i++){
		pos.x=i;
		for(int j=0;j<dim;j++){
			pos.y=j;
			input[i*dim+j]=pos;
		}
	}

	float * matrixA=GenMatrix(dim,dim);
	float * matrixB=GenMatrix(dim,dim);

	HMMSchedulerSpec args(argc,argv);
	args.input=input;
	args.input_size=dim*dim*sizeof(pos_t);
	args.unit_size=sizeof(pos_t);
	args.num_hash_buckets=dim*dim;

	global_data_t global_data;
	global_data.A=matrixA;
	global_data.B=matrixB;
	global_data.dim=dim;
	args.global_data = &global_data;

	DECLARE_GLOBAL_ARRAY(A, sizeof(float)*dim*dim);
	DECLARE_GLOBAL_ARRAY(B, sizeof(float)*dim*dim);


	HMMMainScheduler scheduler(args);
double t2=get_time();
cout<<"init: "<<t2-t1<<endl;
	
	scheduler.do_map_reduce();

double t3=get_time();
cout<<"mapreduce time: "<<t3-t2<<endl;

	OutputChunk output=scheduler.get_output();
	cout<<"total output: "<<output.count<<endl;
	for(int i=0;i<dim*dim && i<10;i++){
		int keyIdx=output.index[i].x;
		int valIdx=output.index[i].z;
		int key=*(int*)((char*)(output.keys)+keyIdx);
		float val=*(float*)((char*)(output.vals)+valIdx);
		cout<<"("<<key<<","<<val<<")"<<endl;
	}

	return 0;
}
