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
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <cuda.h>
#include <sstream>

#include "DS.h"
using namespace std;

#if __CUDA_ARCH__ < 200         //Compute capability 1.x architectures
#define CUPRINTF cuPrintf
#else                                           //Compute capability 2.x architectures
#define CUPRINTF(fmt, ...) printf("[%d, %d]:\t" fmt, \
                                                                blockIdx.y*gridDim.x+blockIdx.x,\
                                                                threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
                                                                __VA_ARGS__)
#endif


// Setting
//============================//
unsigned int grid_dim = 256;
unsigned int block_dim = 256;
#define MAX_WORD_SIZE	255

struct G_State{
public:
	// input
	const void * input;
	unsigned int input_size;
	unsigned int unit_size;
	// global data
	int * job_pool;
	// mark pool
	int * mark_pool;	
};

double get_time(){
	cudaThreadSynchronize();
	timeval t;
	gettimeofday(&t, NULL);
	return (double)t.tv_sec+(double)t.tv_usec/1000000;
}

void map_file(const char * filename, void * & buf, unsigned int & size){
	int fp=open(filename,O_RDONLY);
	if(fp){
		struct stat filestat;
		fstat(fp, &filestat);
		size=filestat.st_size;
		buf=mmap(0,size,PROT_READ,MAP_PRIVATE,fp,0);
	}
	else
		buf=0;
}

inline bool get_opt(int argc, char * argv[], const char * option, string & output){
	using namespace std;
	bool opt_found=false;
	int i;
	for(i=0;i<argc;i++){
		if(argv[i][0]=='-'){
			string str(argv[i]+1);
			for(int j=0;j<str.size();j++)
				str[j]=tolower(str[j]);
			string opt(option);
			for(int j=0;j<opt.size();j++)
				opt[j]=tolower(opt[j]);
			if(str==option){
				opt_found=true;
				break;
			}
		}
	}
	if(opt_found){
		istringstream ss(argv[i+1]);
		ss>>output;
	}	
	return opt_found;
}

void solve(int *offset_base, int size, char *input, char * keyword, int * mark_set)
{
	//int threadIDd = blockIdx.x*blockDim.x+threadIdx.x;
	int threadID = 0;
	int * offsets = offset_base;
	for(int i=0;i<size;i++){
		char * line=input+offsets[i];
		cout<<offsets[i]<<':'<<line<<endl;
		char * pWord=line;
		while(*pWord!='\0'){
			char * curr=pWord;
			char * pkeyword=keyword;
			while(*curr==*pkeyword && *curr!='\0' && *pkeyword!='\0'){
				curr++;
				pkeyword++;
			}
			if(*pkeyword=='\0'){
				int pos=pWord-line;
				cout<<"Oh no! "<<i<<' '<<pos<<endl;
				mark_set[threadID+i] = mark_set[threadID+i] | 0x40000000 | pos;
				cout<<mark_set[threadID+i]<<endl;
				break;
			}
			pWord++;
		} 
	}
}

__global__ void dsolve(int *offset_base, int size, char *input, char * keyword, int * mark_set,int maxID)
{
	int threadID = blockIdx.x*blockDim.x+threadIdx.x;
	int * offsets = offset_base;
	for(int i=0;i<size;i++){
		if (threadID+i*65536>=maxID) break;
		//char * line=input+offsets[threadID*size+i];
		char * line=input+offsets[i*65536+threadID];
		char * pWord=line;
		while(*pWord!='\0'){
			char * curr=pWord;
			char * pkeyword=keyword;
			while(*curr==*pkeyword && *curr!='\0' && *pkeyword!='\0'){
				curr++;
				pkeyword++;
			}
			if(*pkeyword=='\0'){
				int pos=pWord-line;
//				CUPRINTF("threadID:%d\n",threadID);
				//mark_set[threadID*size+i] = mark_set[threadID*size+i] | pos;
				mark_set[i*65536+threadID] = mark_set[i*65536+threadID] | pos;
				break;
			}
			pWord++;
		} 
	}
}

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
	cout<<"---------------"<<endl;
	cout<<"filename:"<<filename<<'\n'<<"keyword:"<<keyword<<endl;
	cout<<"---------------"<<endl;

	void * rawbuf;
	unsigned int size;
	map_file(filename.c_str(), rawbuf, size);
	if(!rawbuf){
		cout<<"error opening file "<<filename<<endl;
		return 1;
	}
	char * filebuf=new char[size];
	memcpy(filebuf,rawbuf,size);
	char * keyword_buf=new char[keyword.size()+1];
	memcpy(keyword_buf, keyword.c_str(), keyword.size()+1);

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
	/*for(int i=0;i<offsets.size();++i)
	{
		cout<<offsets[i]<<' '<<endl;
	}*/
	int * mark_pool = new int[offsets.size()];
	memset(mark_pool,0,offsets.size()*sizeof(int));
	
	
	char * d_input;
	int * d_offsets;
	int * d_mark_pool;
	char * d_keyword;
	double t1=get_time();
	cudaMalloc((void**)&d_input,size);
	cudaMalloc((void**)&d_offsets,offsets.size()*sizeof(int));
	cudaMalloc((void**)&d_mark_pool,(offsets.size()/65536+1)*65536*sizeof(int));
	cudaMalloc((void**)&d_keyword,keyword.size()+1);
	cudaMemcpy(d_input,filebuf,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_offsets,&offsets[0],offsets.size()*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_keyword,keyword_buf,keyword.size()+1,cudaMemcpyHostToDevice);
	//cudaMemcpy(d_mark_pool,mark_pool,offsets.size()*sizeof(int),cudaMemcpyHostToDevice);
	double t2=get_time();
	
	cout<<"Still works well!"<<endl;
	
	//solve(&offsets[0], 4, filebuf, keyword_buf, mark_pool);
	dsolve<<<256,256>>>(d_offsets,offsets.size()/65536+1,d_input,d_keyword,d_mark_pool,offsets.size());
	cudaMemcpy(mark_pool,d_mark_pool,offsets.size()*sizeof(int),cudaMemcpyDeviceToHost);
	double t3=get_time();
	int u = 10;
	if(u>offsets.size()) u =offsets.size();
	for(int i=0; i<u;++i)
	{
		cout<<mark_pool[i]<<endl;
	}
	cout<<"Init time: "<<t2-t1<<endl;
	cout<<"Total time: "<<t3-t1<<endl;

	delete[] filebuf;
	delete[] keyword_buf;
	delete[] mark_pool;
	cudaFree(d_input);
	cudaFree(d_offsets);
	cudaFree(d_mark_pool);
	cudaFree(d_keyword);

	return 0;
}
