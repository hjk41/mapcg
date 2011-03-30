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

#define MAX_BUF_SIZE	255

int main(int argc, char **argv)
{   
	//----------------------------------------------
	//configuration
	//----------------------------------------------
	string filename;
	if(!get_opt(argc,argv, "i", filename)){
		cout<<"usage: "<<argv[0]<<" -i input_file_name"<<endl;
		return 1;
	}	

	void * rawbuf;
	unsigned int size;
	map_file(filename.c_str(), rawbuf, size);
	if(!rawbuf){
		cout<<"error opening file "<<filename<<endl;
		return 1;
	}
	char * filebuf=new char[size];
	memcpy(filebuf,rawbuf,size);

	vector<int> offsets;
	offsets.push_back(0);
	for(int i=0;i<size;i++){
		if(filebuf[i]!='\n')
			continue;
		filebuf[i]='\0';
		if(i+1<size)
			offsets.push_back(i+1);
	}

	HMMSchedulerSpec args(argc,argv);
	args.input=&offsets[0];
	args.input_size=sizeof(int)*offsets.size();
	args.unit_size=sizeof(int);
	args.PrintArgs();

	global_data_t global_data;
	global_data.content=filebuf;
	global_data.pass=1;
	args.global_data=&global_data;

	DECLARE_GLOBAL_ARRAY(content, size);

double mapreduce_time=0;
	HMMMainScheduler scheduler(args);
double t2=get_time();
	scheduler.do_map_reduce();
double t3=get_time();
mapreduce_time+=t3-t2;

	OutputChunk inter_output=scheduler.get_output();
	unsigned int count=inter_output.count;
	unsigned int total_key_size= inter_output.index[count-1].x+inter_output.index[count-1].y;
	void * inter_result=malloc(total_key_size);
	memcpy(inter_result, inter_output.keys, total_key_size);
	global_data.pass=2;

	cout<<"inter_output count: "<<count<<endl;
	args.input=inter_result;
	args.input_size=total_key_size;
	args.unit_size=sizeof(int2);

	HMMMainScheduler scheduler2(args);
t2=get_time();
	scheduler2.do_map_reduce();
t3=get_time();
mapreduce_time+=t3-t2;
cout<<"mapreduce time: "<<mapreduce_time<<endl;

	OutputChunk output=scheduler2.get_output();
	cout<<"total output: "<<output.count<<endl;
	for(int i=0;i<output.count && i<10;i++){
		int keyIdx=output.index[i].x;
		int valIdx=output.index[i].z;
		int2 key=*(int2*)((char*)(output.keys)+keyIdx);
		char * str=global_data.content+key.y;
		int val=*(int*)((char*)(output.vals)+valIdx);
		cout<<"("<<str<<","<<val<<")"<<endl;
	}

	delete[] filebuf;
	return 0;
}
