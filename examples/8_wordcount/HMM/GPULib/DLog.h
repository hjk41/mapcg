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

#ifndef DLOG_H
#define DLOG_H

#include <cuda.h>
#include <iostream>

#include "HMMUtilGPU.h"

namespace HMM_GPU{

const int SAFTY_BYTES=1024*1024;

enum DataType{
	CHAR,
	INT,
	STRING
};

struct LogPack{
	int size;
	DataType type;
	union{
		char c;
		int i;
		char str[];
	}data;
};

__device__ static int requiredLogSpace(const char*);
__device__ static int requiredLogSpace(const int);
__device__ static int requiredLogSpace(const unsigned int);
__device__ static int requiredLogSpace(const char);
__device__ static int requiredLogSpace(const void *, const unsigned int);

__device__ static void assignLog(LogPack *, const char c);
__device__ static void assignLog(LogPack *, const char * c);
__device__ static void assignLog(LogPack *, const int c);
__device__ static void assignLog(LogPack *, const unsigned int c);
__device__ static void assignLog(LogPack *, const void * p, const unsigned int s);

struct MyLog{
public:
	char empty_bytes[SAFTY_BYTES];
	char * buf;
	unsigned int curPos;
	char empty_bytes2[SAFTY_BYTES];
public:
	template <class T>
	__device__ MyLog & operator<<(const T & t){
		int packSize=minAlign(requiredLogSpace(t));
		int start=atomicAdd(&curPos, packSize);
		LogPack * pack=(LogPack *)(buf+start);
		assignLog(pack,t);
		pack->size=packSize;
		return *this;
	}
	__device__ void log(const void * p, const unsigned int s){
		int packSize=minAlign(requiredLogSpace(p,s));
		int start=atomicAdd(&curPos, packSize);
		LogPack * pack=(LogPack *)(buf+start);
		assignLog(pack, p, s);
		pack->size=packSize;
	}
};

__device__ MyLog DLog;



__device__ int requiredLogSpace(const int i){
	return sizeof(i)+sizeof(DataType)+sizeof(int);
}
__device__ int requiredLogSpace(const char * str){
	int len=0;
	while(*str++){
		len++;
	}
	return len+1+sizeof(DataType)+sizeof(int);
}
__device__ int requiredLogSpace(const void * ptr, const unsigned int len){
	return len+1+sizeof(DataType)+sizeof(int);
}
__device__ static int requiredLogSpace(const unsigned int i){
	return sizeof(i)+sizeof(DataType)+sizeof(int);
}
__device__ static int requiredLogSpace(const char c){
	return sizeof(c)+sizeof(DataType)+sizeof(int);
}

__device__ void assignLog(LogPack * pack, const char c){
	pack->data.c=c;
	pack->type=CHAR;
}
__device__ void assignLog(LogPack * pack, const int i){
	pack->data.i=i;
	pack->type=INT;
}
__device__ void assignLog(LogPack * pack, const unsigned int i){
	pack->data.i=i;
	pack->type=INT;
}
__device__ void assignLog(LogPack * pack, const char * str){
	int i=0;
	while(*str){
		pack->data.str[i++]=*str;
		str++;
	}
	pack->type=STRING;
}
__device__ void assignLog(LogPack * pack, const void * ptr, const unsigned int s){
	const char * str=(const char *)ptr;
	for(int i=0;i<s;i++){
		pack->data.str[i]=str[i];
	}
	pack->data.str[s]='\0';
}



__host__ void DLog_Init(unsigned int size){
	MyLog localLog;
	localLog.curPos=SAFTY_BYTES;
	cudaMalloc((void**)&localLog.buf, size+SAFTY_BYTES);
	cudaMemset(localLog.buf, 0, size);
	cudaMemcpyToSymbol(DLog, &localLog, sizeof(MyLog), 0, cudaMemcpyHostToDevice);
}

__host__ void DLog_Dump(){
using namespace std;
	cout<<"<<< --- begin dumping device log: "<<endl;
	MyLog localLog;
	cudaMemcpyFromSymbol(&localLog, DLog, sizeof(MyLog), 0, cudaMemcpyDeviceToHost);
	char * h_buf=new char[localLog.curPos+1];
	cudaMemcpy(h_buf, localLog.buf, localLog.curPos, cudaMemcpyDeviceToHost);
//	cudaFree(localLog.buf);

	char * ptr=h_buf+SAFTY_BYTES;
	while(ptr<h_buf+localLog.curPos){
		LogPack * pack=(LogPack *)ptr;
		if(pack->type==CHAR){
			cout<<pack->data.c<<endl;
		}
		else if(pack->type==INT){
			cout<<"0x"<<hex<<pack->data.i<<endl;
		}
		else{
			cout<<pack->data.str<<endl;
		}
		ptr+=pack->size;
	}

	cout<<"end dumping device log --- >>>"<<endl;

	delete[] h_buf;
}

__host__ void DLog_Destroy(){
	MyLog localLog;
	cudaMemcpyFromSymbol(&localLog, DLog, sizeof(MyLog), 0, cudaMemcpyDeviceToHost);
	cudaFree(localLog.buf);
}

};
#endif
