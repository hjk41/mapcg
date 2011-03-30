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

//0:38,11.17
//CPU 与 GPU框架基本均已完成，基本上所有函数都实现了两个版本，准备开始进行Debug

// Setting
//============================//
unsigned int grid_dim = 256;
unsigned int block_dim = 256;
#define MAX_WORD_SIZE	255

__device__ unsigned int default_hash(const void * key, const unsigned int keysize){
       unsigned long hash = 5381;
       char *str = (char *)key;
       for (int i = 0; i < keysize; i++)
       {
	       hash = ((hash << 5) + hash) + ((int)str[i]);
       }
       return hash;
}

/*unsigned int default_hash(const void * key, const unsigned int keysize){
       unsigned long hash = 5381;
       char *str = (char *)key;
       for (int i = 0; i < keysize; i++)
       {
	       hash = ((hash << 5) + hash) + ((int)str[i]);
       }
       return hash;
}*/

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


#define START		0x00
#define IN_TAG		0x01
#define IN_ATAG		 0x02
#define FOUND_HREF	  0x03
#define START_LINK	  0x04

__device__ int StrCmp(char * p1, char * p2, int size){
	for(int i=0;i<size;i++){
		if(p1[i]<p2[i])
			return -1;
		else if(p1[i]>p2[i])
			return 1;
	}
	return 0;
}

/*int StrCmp(char * p1, char * p2, int size){
	for(int i=0;i<size;i++){
		if(p1[i]<p2[i])
			return -1;
		else if(p1[i]>p2[i])
			return 1;
	}
	return 0;
}*/

__device__ char * StrChr(char * p, char c){
	while(*p!='\0' && *p!=c)
		p++;
	if(*p==c)
		return p;
	return NULL;
}

/*char * StrChr(char * p, char c){
	while(*p!='\0' && *p!=c)
		p++;
	if(*p==c)
		return p;
	return NULL;
}*/

//device memory

__constant__ char * memory_pool;			// memory pool
__device__ unsigned int memory_offset;		// the offset of the pool
__shared__ volatile char * warp_page[8];		// the local page in shared memory, one for each warp
__shared__ volatile unsigned int warp_offsets[8];	// the current offset for local page in shared memory, one for each warp
//__constant__ char * warp_start_page[2048];
//__constant__ unsigned int warp_start_offsets[2048];
__device__ char * warp_start_page[2048];
__device__ unsigned int warp_start_offsets[2048];


//Two level SMA
const unsigned int pageSize = 1024*4; // Bytes/page
const unsigned int totalSize = pageSize * 65535 * 2; //size of page pool

struct SMA{
public:
	//level1
	char * memory_pool;
	int memory_offset;
	//level2
	char * memory_warp_page[2048];
	int memory_warp_offsets[2048];
};

SMA sma;

//init d_sma
void smainit(){
	//cuda
		//char * toinitpool = new char[totalSize];
		//memset(toinitpool,0,sizeof(char)*totalSize);
	cudaMalloc((void**)&sma.memory_pool,totalSize);
		//cudaMemcpy(sma.memory_pool,toinitpool,totalSize,cudaMemcpyHostToDevice);
		//delete[] toinitpool;
	//pc
	//sma.memory_pool = new char[totalSize];
	// memory pool
	cudaMemcpyToSymbol(memory_pool,&sma.memory_pool,sizeof(char *));
	//cudaMemcpyToSymbol(&memory_pool,&sma.memory_pool,sizeof(char *));
	sma.memory_offset = 0;
	for(int i=0;i<2048;i++)
	{
		sma.memory_warp_page[i] = (char*)(sma.memory_pool + sma.memory_offset);
		sma.memory_warp_offsets[i] = 0;
		sma.memory_offset += pageSize;
	}
	
	//改成cuda版本时将如下注释去掉即可
	cudaMemcpyToSymbol( memory_pool , &(sma.memory_pool),sizeof(char *));
	cudaMemcpyToSymbol( memory_offset, &sma.memory_offset,sizeof(int));
	cout<<"Device Memory Pool:"<<(long int)sma.memory_pool<<endl;
	cout<<"Device Memory init offset:"<<sma.memory_offset<<endl;
	cudaMemcpyToSymbol(warp_start_page,&(sma.memory_warp_page[0]),2048*sizeof(char *));
	cudaMemcpyToSymbol(warp_start_offsets,&(sma.memory_warp_offsets[0]),2048*sizeof(int));
}

__device__ char * dmalloc(int size)//, int threadNum)
{
	int offset;
	char * pageStart;
	bool ret = false;	
	while(!ret)
	{
		pageStart = (char*)warp_page[threadIdx.x/32];
		offset = atomicAdd((unsigned int *)(&warp_offsets[threadIdx.x/32]),size);
		if(offset+size<=pageSize) ret = true;
		else if(offset<=pageSize && offset+size>pageSize)
		{
			pageStart = memory_pool+atomicAdd((unsigned int *)(&memory_offset),pageSize);
			warp_page[threadIdx.x/32]= pageStart;
			warp_offsets[threadIdx.x/32]=size;
			offset = 0;
			ret=true;
		}
	}		
	return pageStart+offset;
}

/*
char * warp_page[8];
int warp_offsets[8];

char * hmalloc(int size, int &threadNum)
{
	cout<<"hmalloc"<<endl;
	int offset;
	char * pageStart;
	bool ret = false;	
	while(!ret)
	{
		pageStart = warp_page[threadNum/32];
		offset = warp_offsets[threadNum/32];
		warp_offsets[threadNum/32] += size;
		if(offset+size<=pageSize) ret = true;
		else if(offset<=pageSize && offset+size>pageSize)
		{
			pageStart = sma.memory_pool+sma.memory_offset;
			sma.memory_offset += pageSize;
			warp_page[threadNum/32]= pageStart;
			warp_offsets[threadNum/32]=size;
			offset = 0;
			ret=true;
		}
	}		
	return pageStart+offset;
}
*/




//#pragma pack(push,4)
//Three level keylist
struct KeyListNode{
public:
	int sum;
	KeyListNode * next;
	//char * data;
	int keySize;
	char data[];
};
//#pragma pack(pop)

struct KeyList{
	KeyListNode * head;
	int hk;
};	// KeyList

struct Buckets{
	int num_buckets;
	KeyList *buckets;
};

Buckets h_buckets;
__device__ Buckets d_buckets;

void initBuckets(){
	//PC
	/*
	h_buckets.num_buckets = 30;
	h_buckets.buckets = new KeyList[h_buckets.num_buckets];
	for(int i=0; i<h_buckets.num_buckets; ++i)
	{
		h_buckets.buckets[i].head = NULL;
	}*/
	
	h_buckets.num_buckets = 3367;
	KeyList temp[h_buckets.num_buckets];
	for(int i=0; i<h_buckets.num_buckets; ++i)
	{
		temp[i].head = NULL;
		temp[i].hk = 7;
	}
	cudaMalloc((void**)&h_buckets.buckets,h_buckets.num_buckets*sizeof(KeyList));
	
	cudaMemcpy(h_buckets.buckets,temp,h_buckets.num_buckets*sizeof(KeyList),cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(d_buckets,&h_buckets,sizeof(Buckets));
	cout<<"h_buckets.num_buckets:"<<h_buckets.num_buckets<<endl;
	return;
}


__device__ char * newkeyFunc(char * key, int keySize){
	char * lk = dmalloc(keySize+sizeof(KeyListNode));
	KeyListNode * tmp=(KeyListNode *)lk;
	tmp->sum = 1;
	tmp->keySize = keySize;
	tmp->next = NULL;
	for(int i=0; i<keySize; i++)
		(tmp->data)[i] = key[i];
	return lk;
}

/*void newkeyFunc(KeyListNode * &newKey,char * &key, int keySize, int &threadNum){
	cout<<"newkeyFunc"<<endl;
	newKey = (KeyListNode *)hmalloc(keySize+sizeof(KeyListNode),threadNum);
	newKey->sum = 1;
	newKey->keySize = keySize;
	newKey->next = NULL;
	newKey->data = (char*)newKey+sizeof(KeyListNode);
	for(int i=0; i<keySize; ++i)
		(newKey->data)[i] = key[i];
}*/

//To compare the two key
__device__ bool keycmp(char * key1, int keySize1, char * key2, int keySize2)
{
	if(keySize1 != keySize2) return false;
	for(int i=0; i<keySize1; i++)
		if(key1[i]!=key2[i]) return false;
	return true;
}

/*bool keycmp(char * &key1, int &keySize1, char * &key2, int &keySize2)
{
	if(keySize1 != keySize2) return false;
	for(int i=0; i<keySize1; i++)
		if(key1[i]!=key2[i]) return false;
	return true;
}*/


template<class T1, class T2, class T3>
__device__ bool CAS64(T1 * addr, T2 old_val, T3 new_val){
	return *(unsigned long long *)(&old_val)==atomicCAS((unsigned long long *)addr, *(unsigned long long *)(&old_val), *(unsigned long long *)(&new_val));
}

template<class T1, class T2, class T3>
__device__ bool CAS32(T1 * addr, T2 old_val, T3 new_val){
	return *(uint32_t *)(&old_val)==atomicCAS((uint32_t *)addr, *(uint32_t *)(&old_val), *(uint32_t *)(&new_val));
}

#define LONG_PTR

template<class T1, class T2, class T3>
__device__ bool CASPTR(T1 * addr, T2 old_val, T3 new_val){
#ifdef LONG_PTR
	return CAS64(addr, old_val, new_val);
#else
	return CAS32(addr, old_val, new_val);
#endif
}

template<class T>
__device__ T minAlign(T n){
	return ((uint32_t)n&(4-1))==NULL ? n : n+4-((uint32_t)n&(4-1));
}

template<class T>
T miAlign(T n){
	return ((uint32_t)n&(4-1))==NULL ? n : n+4-((uint32_t)n&(4-1));
}

unsigned int RoundToAlign(unsigned int size){
	return (size+4-1)&(~(4-1));
}

__device__ unsigned int SMA_RoundToAlign(unsigned int size){
	return (size+4-1)&(~(4-1));
}


__device__ unsigned int sizeofKeyListNode(int keySize){
	//return minAlign(sizeof(KeyListNode *)+minAlign(sizeof(int))*2+keySize);
	return minAlign(24+keySize);
}


//To insert a key into Buckets, or just add a 1
__device__ void dinsert(char * key, int keySize, int * d_mark){
	int bucketsNum = default_hash(key,keySize)%d_buckets.num_buckets;
	KeyListNode * curr = d_buckets.buckets[bucketsNum].head;
	KeyListNode * newKey = NULL;
	if(curr==NULL)
	{
		newKey = (KeyListNode *)dmalloc(SMA_RoundToAlign(sizeofKeyListNode(keySize)));
		newKey->next = NULL;
		newKey->keySize = keySize;
		newKey->sum = 1;
		for(int i=0; i<keySize; i++)
			(newKey->data)[i] = key[i];
		if(CASPTR(&d_buckets.buckets[bucketsNum].head,NULL,newKey))
		{
			//atomicAdd((unsigned int *)(&d_mark[0]),1);
			return;	
		}				
	}
	curr = d_buckets.buckets[bucketsNum].head;
	while(1)
	{
		if(keycmp(curr->data,curr->keySize,key,keySize))
		{
			atomicAdd((unsigned int *)&(curr->sum), 1);
			return;
		}
		if(curr->next == NULL)
		{
			if( newKey == NULL )
			{
				newKey = (KeyListNode *)dmalloc(SMA_RoundToAlign(sizeofKeyListNode(keySize)));
				newKey->next = NULL;
				newKey->sum = 1;
				newKey->keySize = keySize;
				for(int i=0; i<keySize; i++)
					(newKey->data)[i] = key[i];
			}
			if(CASPTR(&(curr->next),NULL,newKey))
			{
				//atomicAdd((unsigned int *)(&d_mark[0]),1);
				return;
			}
		}
		//atomicAdd((unsigned int *)(&d_mark[0]),1);
		curr=curr->next;
	}
}

/*
//To insert a key into Buckets, or just add a 1
void hinsert(char * &key, int keySize, int threadNum){
	cout<<"hinsert";
	int bucketsNum = default_hash(key,keySize)%h_buckets.num_buckets;
	cout<<bucketsNum<<endl;
	//int bucketsNum = 0;
	KeyListNode * curr = h_buckets.buckets[bucketsNum].head;
	KeyListNode * newKey = NULL;
	if(curr==NULL)
	{
		newkeyFunc(newKey,key,keySize,threadNum);
		h_buckets.buckets[bucketsNum].head = newKey;
		return;
	}
	curr = h_buckets.buckets[bucketsNum].head;
	while(1)
	{
		cout<<"After Head!"<<endl;
		if(keycmp(curr->data,curr->keySize,key,keySize))
		{
			cout<<"	the same!"<<endl;
			curr->sum ++;
			return;
		}
		if(curr->next == NULL)
		{
			cout<<"	next null"<<endl;
			if( newKey == NULL )
				newkeyFunc(newKey,key,keySize,threadNum);
			curr->next = newKey;
			return;
		}
		curr=curr->next;
	}
}
*/

__global__ void dsolve(int *offset_base, int size, char *input, int maxsize, int * d_mark){
	int threadID = blockIdx.x*blockDim.x+threadIdx.x;
	if( threadIdx.x%32 == 0)
	{
		warp_page[threadIdx.x/32] = warp_start_page[threadID/32];
		warp_offsets[threadIdx.x/32] = warp_start_offsets[threadID/32];
	}
	char *linebuf;
	char * link_start;
	char *link_end;
	int state = START;
	char href[5];
	href[0] = 'h';
	href[1] = 'r';
	href[2] = 'e';
	href[3] = 'f';
	href[4] = '\0';
	for(int task=0; task<size; task++){
		if(task+threadID*size>=maxsize) break;
		linebuf = (char*)(&input[offset_base[task+threadID*size]]);
		state = START;
		for (char *p = linebuf; *p != '\0'; p++){
			switch (state){
				case START:
					if (*p == '<') state = IN_TAG;
					break;
				case IN_TAG:
					if (*p == 'a') state = IN_ATAG;
					else if (*p == ' ') state = IN_TAG;
					else state = START;
					break;
				case IN_ATAG:
					if (*p == 'h'){
						if (StrCmp(p, href, 4) == 0){
							state = FOUND_HREF;
							p += 4;
						}
						else state = START;
					}
					else if (*p == ' ') state = IN_ATAG;
					else state = START;
					break;
				case FOUND_HREF:
					if (*p == ' ') state = FOUND_HREF;
					else if (*p == '=') state = FOUND_HREF;
					else if (*p == '\"') state  = START_LINK;
					else state = START;
					break;
				case START_LINK:
					link_start=p;
					link_end = StrChr(p, '\"');
					if (link_end != NULL)
					{
						*link_end='\0';
						dinsert(link_start, link_end-link_start+1, d_mark);
						//atomicAdd((unsigned int *)(&d_mark[3]),1);
						//atomicAdd((unsigned int *)(&d_mark[4]),link_end-link_start+1);
						p =link_end;
					}
					state = START;
					break;
			}//switch
		}//for
	}
	//}
}

/*
void solve(int *offset_base, int size, char *input){
	warp_page[0] = sma.memory_warp_page[0];
	warp_offsets[0] = sma.memory_warp_offsets[0];
	char *linebuf;
	char * link_start;
	char *link_end;
	int state = START;
	char href[5];
	href[0] = 'h';
	href[1] = 'r';
	href[2] = 'e';
	href[3] = 'f';
	href[4] = '\0';
	for(int task=0; task<size; task++){
		cout<<task<<endl;
		//cout<<(int)(warp_page[0])<<endl;
		cout<<"warp_offsets:"<<warp_offsets[0]<<endl;
		linebuf = (char*)(&input[offset_base[task]]);
		//cout<<linebuf<<endl;
		state = START;
		for (char *p = linebuf; *p != '\0'; p++){
			switch (state){
				case START:
					if (*p == '<') state = IN_TAG;
					break;
				case IN_TAG:
					if (*p == 'a') state = IN_ATAG;
					else if (*p == ' ') state = IN_TAG;
					else state = START;
					break;
				case IN_ATAG:
					if (*p == 'h'){
						if (StrCmp(p, href, 4) == 0){
							state = FOUND_HREF;
							p += 4;
						}
						else state = START;
					}
					else if (*p == ' ') state = IN_ATAG;
					else state = START;
					break;
				case FOUND_HREF:
					if (*p == ' ') state = FOUND_HREF;
					else if (*p == '=') state = FOUND_HREF;
					else if (*p == '\"') state  = START_LINK;
					else state = START;
					break;
				case START_LINK:
					link_start=p;
					link_end = StrChr(p, '\"');
					if (link_end != NULL)
					{
						*link_end='\0';
						printf("emit_inter(%s,%d)\n",link_start,1);
						hinsert(link_start, link_end-link_start+1, 0);
						p =link_end;
					}
					state = START;
					break;
			}//switch
		}//for
	}
}
*/

__global__ void d_out(int * d_mark)
{
	int threadID = blockIdx.x*blockDim.x+threadIdx.x;
	KeyListNode * dio = d_buckets.buckets[0].head;
	int u = 1;
	if(threadID == 0){
	while(dio!=NULL)
	{
		d_mark[1] += (dio->keySize)*(dio->sum);
		dio = dio->next;
	}
	}
}

struct OutLoad
{
	int num;
	int len;
	char val[];
};

__constant__ char * g_ptr;
__device__ int g_out_offset;

__global__ void d_reduce(int * d_mark,int maxSize){
	int threadID = blockIdx.x*blockDim.x+threadIdx.x;
	OutLoad * ptr_start;
	int size = 0;
	
	if(threadID < maxSize)
	{
		KeyListNode * dio = d_buckets.buckets[threadID].head;
		//printf("g_ptr:%ld\n",(long int)g_ptr);
		while(dio!=NULL)
		{			
			ptr_start = (OutLoad *)(g_ptr+atomicAdd((unsigned int *)(&g_out_offset),minAlign(1 + dio->keySize+sizeof(int)*2)));
			ptr_start->num = dio->sum;
			size = dio->keySize;
			ptr_start->len = size;
			for(int i=0; i<size; i++)
				(ptr_start->val)[i] = dio->data[i];
			(ptr_start->val)[size] = '\0';
			//printf("threadID:%d,,,sum:%d,,,val:%s\n",threadID,ptr_start->num,ptr_start->val);
			dio = dio->next;
		}
	}
	
}


int main(int argc, char * argv[]){
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
		while (fgets(buf, 1024, fp) != NULL)
		{
			int lineSize = strlen(buf)+1;
			buf[lineSize-1]='\0';
			if (lineSize < 14)
				continue;
			offsets.push_back(all_content.size());
			all_content.append(buf,lineSize);
			memset(buf,0,1024);
		}
		fclose(fp);
	}


	double t1=get_time();
	smainit();
	initBuckets();
	//solve(&offsets[0], 100, &all_content[0]);
	

	int * d_offsets;
	char * ffinput;
	
	//用于输出
	//--------------------------------------------------------
	OutLoad * outptr;
	char * outputLoad = new char[1024*1024*256];
	char * d_outputLoad;
	cudaMalloc((void**)&d_outputLoad,1024*1024*256*sizeof(char));
	memset(outputLoad,0,sizeof(char)*1024*1024*256);
	cudaMemcpy(d_outputLoad,outputLoad,1024*1024*256*sizeof(char),cudaMemcpyHostToDevice);
	int h_out_offset = 0;
	cudaMemcpyToSymbol(g_out_offset,&h_out_offset,sizeof(int));
	cudaMemcpyToSymbol(g_ptr,&d_outputLoad,sizeof(char*));
	cout<<"g_outputLoad:"<<(long int)d_outputLoad<<endl;
	
	//--------------------------------------------------------
	
	//用于调试
	//--------------------------------------------------------
	int * d_mark;
	int * mark = new int[2048];
	cudaMalloc((void**)&d_mark,2048*sizeof(int));
	memset(mark,0,sizeof(int)*2048);
	cudaMemcpy(d_mark,mark,2048*sizeof(int),cudaMemcpyHostToDevice);
	//--------------------------------------------------------

	////////////////////
	
	cudaMalloc((void**)&ffinput,all_content.size()*sizeof(char));
	cudaMalloc((void**)&d_offsets,offsets.size()*sizeof(int));
	cudaMemcpy(d_offsets,&offsets[0],offsets.size()*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(ffinput,&all_content[0],all_content.size(),cudaMemcpyHostToDevice);
	printf("offsets.size()==%d\n",offsets.size());
	printf("minAlign:%d\n",(int)miAlign(11));
	printf("RoundToAlign:%d\n",(int)RoundToAlign(23));
	
	double t2=get_time();
	dsolve<<<256,256>>>(d_offsets,((int)offsets.size()/65536)+1,ffinput,offsets.size(),d_mark);
	//d_out<<<1,1>>>(d_mark);
	
	d_reduce<<<256,256>>>(d_mark, 3367); //其中3367是buckets的数量
	double t3=get_time();

	cudaMemcpy(&sma.memory_warp_offsets[0],d_mark,2048*sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(outputLoad,d_outputLoad,256*1024*1024,cudaMemcpyDeviceToHost);
	
	for(int j =0; j<10; j++)
	{
		cout<<"  "<<j<<"	:"<<sma.memory_warp_offsets[j]<<endl;
	}
	
	int offsetsu = 0;
	for(int j=0;j<17; j++)
	{
		OutLoad * nowe = (OutLoad *)(outputLoad+offsetsu);
		offsetsu += miAlign(sizeof(int)*2+nowe->len+1);
		cout<<"sum:"<<nowe->num<<",,len:"<<nowe->len<<",,val:"<<nowe->val<<endl;
	}
	
	cout<<"Init time: "<<t2-t1<<endl;
	cout<<"Total time: "<<t3-t1<<endl;
	//cout<<"offsets.size:"<<offsets.size()<<endl;
	
	cudaFree(ffinput);
	cudaFree(d_offsets);
	cudaFree(d_mark);
	cudaFree(sma.memory_pool);
	cudaFree(d_outputLoad);
	delete[] mark;
	delete[] outputLoad;
	return 0;
}
