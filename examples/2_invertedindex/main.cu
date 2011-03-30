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
double t1=get_time();
	if(argc<3){
		cout<<"usage: "<<argv[0]<<" -fl file_list_file"<<endl;
		return 1;
	}
	string listfile;
	if(!get_opt(argc,argv,"fl", listfile)){
		cout<<"usage: "<<argv[0]<<" -fl file_list_file"<<endl;
		return 1;
	}
	cout<<"using file list: "<<listfile<<endl;

	string all_content;
	vector<int> offsets;

	ifstream in(listfile.c_str());
	while (in.good())
	{
		string str;
		in>>str;
		if(str=="")
			continue;
		char * filename=(char *)(str.c_str());

		FILE *fp = fopen(filename, "r");
		char buf[1024];
		memset(buf,0,1024);
		while (fgets(buf, 1023, fp) != NULL)
		{
			int lineSize = strlen(buf);
			if(buf[lineSize-1]!='\n')
				lineSize++;
			buf[lineSize-1]='\0';
			if (lineSize < 14)
				continue;
			offsets.push_back(all_content.size());
			all_content.append(buf,lineSize);
			memset(buf,0,1024);
		}
		fclose(fp);
	}

	HMMSchedulerSpec args(argc,argv);
	args.input=&offsets[0];
	args.input_size=offsets.size()*sizeof(int);
	args.unit_size=sizeof(int);
	args.PrintArgs();

	global_data_t global_data;
	global_data.content=&(all_content[0]);
	args.global_data=&global_data;

	DECLARE_GLOBAL_ARRAY(content, all_content.size());
	
	HMMMainScheduler scheduler(args);
double t2=get_time();

	scheduler.do_map_reduce();

double t3=get_time();
cout<<"mapreduce time: "<<t3-t2<<endl;

	OutputChunk output=scheduler.get_output();
	cout<<"total output: "<<output.count<<endl;
	for(int i=0;i<output.count && i<10;i++){
		int keyIdx=output.index[i].x;
		int valIdx=output.index[i].z;
		char * link=(char*)(output.keys)+keyIdx;
		int val=*(int*)((char*)(output.vals)+valIdx);
		cout<<"("<<link<<","<<val<<")"<<endl;
	}

	return 0;
}
