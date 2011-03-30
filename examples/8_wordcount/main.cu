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

int main(int argc, char * argv[]){
	//----------------------------------------------
	//get parameters
	//----------------------------------------------
	string filename;
	if(!get_opt(argc,argv,"f", filename)){
		cout<<"usage: "<<argv[0]<<" -f filename"<<endl;
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
	args.input_size=offsets.size()*sizeof(int);
	args.unit_size=sizeof(int);
	args.PrintArgs();

	global_data_t global_data;
	global_data.content=filebuf;
	args.global_data=&global_data;

	DECLARE_GLOBAL_ARRAY(content, size);
	
	HMMMainScheduler scheduler(args);
double t2=get_time();

	scheduler.do_map_reduce();
	cudaThreadSynchronize();

double t3=get_time();
cout<<"mapreduce time: "<<t3-t2<<endl;

	OutputChunk output=scheduler.get_output();
	cout<<"total output: "<<output.count<<endl;
	for(int i=0;i<output.count && i<10;i++){
		int keyIdx=output.index[i].x;
		int valIdx=output.index[i].z;
		char * start=(char*)(output.keys)+keyIdx;
		string link;
		link.assign(start,start+output.index[i].y);
		int val=*(int*)((char*)(output.vals)+valIdx);
		cout<<"("<<link<<","<<val<<")"<<endl;
	}

	return 0;
}
