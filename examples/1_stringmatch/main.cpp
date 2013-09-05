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
#include "HMM/HMMGlobalData.h"


#define MAX_WORD_SIZE	255

int main(int argc, char **argv){   
	//----------------------------------------------
	//get parameters
	//----------------------------------------------
	string keyword;
	string filename;
	if(!get_opt(argc,argv,"keyword",keyword) ||
		!get_opt(argc,argv,"f", filename)){
		cout<<"usage: "<<argv[0]<<" -f filename -keyword keyword"<<endl;
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
	unsigned int offset=0;
	FILE *fp = fopen(filename.c_str(), "r");
	char buf[1024];
	memset(buf,0,1024);
	while (fgets(buf, 1024, fp) != NULL)
	{
		offsets.push_back(offset);
		offset += strlen(buf);
		filebuf[offset-1] = '\0';
		memset(buf,0,1024);
	}

	HMMSchedulerSpec args(argc,argv);
	args.input=&offsets[0];
	args.input_size=sizeof(int)*offsets.size();
	args.unit_size=sizeof(int);

	args.PrintArgs();
	cout<<"using keyword: "<<keyword<<endl;

	global_data_t global_data;
	global_data.content=filebuf;
	char * keyword_buf=new char[keyword.size()+1];
	memcpy(keyword_buf, keyword.c_str(), keyword.size()+1);
	global_data.keyword=keyword_buf;
	args.global_data=&global_data;

	DECLARE_GLOBAL_ARRAY(content, size);
	DECLARE_GLOBAL_ARRAY(keyword, keyword.size()+1);

	double t1=get_time();
	HMMMainScheduler scheduler(args);
	scheduler.do_map_reduce();
	double t2=get_time();
	cout<<"==== total time: "<<t2-t1<<endl;

	OutputChunk output=scheduler.get_output();
	cout<<"number of output: "<<output.count<<endl;
	for(int i=0;i<output.count && i<10; i++){
		int keyIdx=output.index[i].x;
		int keySize=output.index[i].y;
		int valIdx=output.index[i].z;
		char * link=filebuf+*(int*)((char*)(output.keys)+keyIdx);
		int val=*(int*)((char*)(output.vals)+valIdx);
		cout<<"("<<val<<")"<<link<<endl;
	}

	delete[] filebuf;
	delete[] keyword_buf;

	return 0;
}
