#ifndef HMMSORTGPU_H
#define HMMSORTGPU_H

/**
 *This is the source code for Mars, a MapReduce framework on graphics
 *processors.
 *Author: Wenbin Fang (HKUST), Bingsheng He (HKUST)
 *Mentor: Naga K. Govindaraju (Microsoft Corp.), Qiong Luo (HKUST), Tuyong
 *Wang (Sina.com).
 *If you have any question on the code, please contact us at {saven,
 *wenbin, luo}@cse.ust.hk.
 *The copyright is held by HKUST. Mars is provided "as is" without any 
 *guarantees of any kind.
 */

#include "HMMScan.h"
#include "../../sort_compare.h"

namespace HMM_GPU{

//==============================================================
//GPU sort
//==============================================================
#ifdef __DEVICE_EMULATION__
	#define NUM_BLOCK_PER_CHUNK_BITONIC_SORT 2
	#define SHARED_MEM_INT2 2
	#define NUM_BLOCKS_CHUNK 2
	#define	NUM_THREADS_CHUNK 2
#else
	#define NUM_BLOCK_PER_CHUNK_BITONIC_SORT 512//b256
	#define SHARED_MEM_INT2 256
	#define NUM_BLOCKS_CHUNK 512
	#define	NUM_THREADS_CHUNK 256//(256)
#endif
#define CHUNK_SIZE (NUM_BLOCKS_CHUNK*NUM_THREADS_CHUNK)
#define NUM_CHUNKS_R (NUM_RECORDS_R/CHUNK_SIZE)


__device__ int getCompareValue(void *d_keys, void * d_vals, int4 index1, int4 index2)
{
	int v1=index1.x;
	int v2=index2.x;
	int compareValue=0;
	if((v1==-1) || (v2==-1))
	{
		if(v1==v2)
			compareValue=0;
		else
			if(v1==-1)
				compareValue=-1;
			else
				compareValue=1;
	}
	else{
		compareValue=less_than((char*)d_keys+index1.x, index1.y,	// key1, keysize1
			(char*)d_vals+index1.z, index1.w,	// val1, valsize1
			(char*)d_keys+index2.x, index2.y,	// key2, keysize2
			(char*)d_vals+index2.z, index2.w	// val2, valsize2
		); 
	}
	return compareValue;
}

void * s_qsRawData=NULL;

__global__ void
partBitonicSortKernel( void* d_keys, void * d_vals, int4* d_index, unsigned int numRecords, int chunkIdx, int unitSize)
{
	__shared__ int4 shared[NUM_THREADS_CHUNK];

	int tx = threadIdx.x;
	int bx = blockIdx.x;

	//load the data
	int dataIdx = chunkIdx*CHUNK_SIZE+bx*blockDim.x+tx;
	int unitIdx = ((NUM_BLOCKS_CHUNK*chunkIdx + bx)/unitSize)&1;
	shared[tx] = d_index[dataIdx];
	__syncthreads();
	int ixj=0;
	int a=0;
	int4 temp1;
	int4 temp2;
	int k = NUM_THREADS_CHUNK;

	if(unitIdx == 0)
	{
		for (int j = (k>>1); j>0; j =(j>>1))
		{
			ixj = tx ^ j;
			//a = (shared[tx].y - shared[ixj].y);				
			temp1=shared[tx];
			temp2= shared[ixj];
			if (ixj > tx) {
				//a=temp1.y-temp2.y;
				//a=compareString((void*)(((char4*)d_rawData)+temp1.x),(void*)(((char4*)d_rawData)+temp2.x)); 
				a=getCompareValue(d_keys, d_vals, temp1, temp2);
				if ((tx & k) == 0) {
					if ( (a>0)) {
						shared[tx]=temp2;
						shared[ixj]=temp1;
					}
				}
				else {
					if ( (a<0)) {
						shared[tx]=temp2;
						shared[ixj]=temp1;
					}
				}
			}
				
			__syncthreads();
		}
	}
	else
	{
		for (int j = (k>>1); j>0; j =(j>>1))
		{
			ixj = tx ^ j;
			temp1=shared[tx];
			temp2= shared[ixj];
			
			if (ixj > tx) {					
				//a=temp1.y-temp2.y;					
				//a=compareString((void*)(((char4*)d_rawData)+temp1.x),(void*)(((char4*)d_rawData)+temp2.x));
				a=getCompareValue(d_keys, d_vals, temp1, temp2);
				if ((tx & k) == 0) {
					if( (a<0))
					{
						shared[tx]=temp2;
						shared[ixj]=temp1;
					}
				}
				else {
					if( (a>0))
					{
						shared[tx]=temp2;
						shared[ixj]=temp1;
					}
				}
			}
			
			__syncthreads();
		}
	}

	d_index[dataIdx] = shared[tx];
}

__global__ void
unitBitonicSortKernel(void* d_keys, void* d_vals, int4* d_index, unsigned int numRecords, int chunkIdx )
{
	__shared__ int4 shared[NUM_THREADS_CHUNK];

	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int unitIdx = (NUM_BLOCKS_CHUNK*chunkIdx + bx)&1;

	//load the data
	int dataIdx = chunkIdx*CHUNK_SIZE+bx*blockDim.x+tx;
	shared[tx] = d_index[dataIdx];
	__syncthreads();

	int4 temp1;
	int4 temp2;
	int ixj=0;
	int a=0;
	if(unitIdx == 0)
	{
		for (int k = 2; k <= NUM_THREADS_CHUNK; (k =k<<1))
		{
			// bitonic merge:
			for (int j = (k>>1); j>0; (j=j>>1))
			{
				ixj = tx ^ j;	
				temp1=shared[tx];
				temp2= shared[ixj];
				if (ixj > tx) {					
					//a=temp1.y-temp2.y;
					//a=compareString((void*)(((char4*)d_rawData)+temp1.x),(void*)(((char4*)d_rawData)+temp2.x));
					a=getCompareValue(d_keys, d_vals, temp1, temp2);
					if ((tx & k) == 0) {
						if ( (a>0)) {
							shared[tx]=temp2;
							shared[ixj]=temp1;
						}
					}
					else {
						if ( (a<0)) {
							shared[tx]=temp2;
							shared[ixj]=temp1;
						}
					}
				}
				
				__syncthreads();
			}
		}
	}
	else
	{
		for (int k = 2; k <= NUM_THREADS_CHUNK; (k =k<<1))
		{
			// bitonic merge:
			for (int j = (k>>1); j>0; (j=j>>1))
			{
				ixj = tx ^ j;
				temp1=shared[tx];
				temp2= shared[ixj];
				if (ixj > tx) {					
					//a=temp1.y-temp2.y;
					//a=compareString((void*)(((char4*)d_rawData)+temp1.x),(void*)(((char4*)d_rawData)+temp2.x));
					a=getCompareValue(d_keys, d_vals, temp1, temp2);
					if ((tx & k) == 0) {
						if( (a<0))
						{
							shared[tx]=temp2;
							shared[ixj]=temp1;
						}
					}
					else {
						if( (a>0))
						{
							shared[tx]=temp2;
							shared[ixj]=temp1;
						}
					}
				}
				
				__syncthreads();
			}
		}

	}

	d_index[dataIdx] = shared[tx];
}

__global__ void
bitonicKernel( void* d_keys, void * d_vals, int4* d_index, unsigned int numRecords, int k, int j)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tid = threadIdx.x;
	int dataIdx = by*gridDim.x*blockDim.x + bx*blockDim.x + tid;

	int ixj = dataIdx^j;

	if( ixj > dataIdx )
	{
		int4 tmpR = d_index[dataIdx];
		int4 tmpIxj = d_index[ixj];
		if( (dataIdx&k) == 0 )
		{
			//if( tmpR.y > tmpIxj.y )
			//if(compareString((void*)(((char4*)d_rawData)+tmpR.x),(void*)(((char4*)d_rawData)+tmpIxj.x))==1) 
			if(getCompareValue(d_keys, d_vals, tmpR, tmpIxj)==1)
			{
				d_index[dataIdx] = tmpIxj;
				d_index[ixj] = tmpR;
			}
		}
		else
		{
			//if( tmpR.y < tmpIxj.y )
			//if(compareString((void*)(((char4*)d_rawData)+tmpR.x),(void*)(((char4*)d_rawData)+tmpIxj.x))==-1) 
			if(getCompareValue(d_keys, d_vals, tmpR, tmpIxj)==-1)
			{
				d_index[dataIdx] = tmpIxj;
				d_index[ixj] = tmpR;
			}
		}
	}
}

__device__ inline void swap(int4 & a, int4 & b)
{
	// Alternative swap doesn't use a temporary register:
	// a ^= b;
	// b ^= a;
	// a ^= b;
	
	int4 tmp = a;
	a = b;
	b = tmp;
}

__global__ void bitonicSortSingleBlock_kernel(void* d_keys, void* d_vals, int4 * d_index, int rLen, int4* d_output)
{
	__shared__ int4 bs_cmpbuf[SHARED_MEM_INT2];
	
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	
	if(tid<rLen)
	{
		bs_cmpbuf[tid] = d_index[tid];
	}
	else
	{
		bs_cmpbuf[tid].x =-1;
	}

	__syncthreads();

	// Parallel bitonic sort.
	int compareValue=0;
	for (int k = 2; k <= SHARED_MEM_INT2; k *= 2)
	{
	// Bitonic merge:
	for (int j = k / 2; j>0; j /= 2)
	{
		int ixj = tid ^ j;
		
		if (ixj > tid)
		{
		if ((tid & k) == 0)
		{
			compareValue=getCompareValue(d_keys, d_vals, bs_cmpbuf[tid], bs_cmpbuf[ixj]);
			//if (shared[tid] > shared[ixj])
			if(compareValue>0)
			{
				swap(bs_cmpbuf[tid], bs_cmpbuf[ixj]);
			}
		}
		else
		{
			compareValue=getCompareValue(d_keys, d_vals, bs_cmpbuf[tid], bs_cmpbuf[ixj]);
			//if (shared[tid] < shared[ixj])
			if(compareValue<0)
			{
				swap(bs_cmpbuf[tid], bs_cmpbuf[ixj]);
			}
		}
		}
		
		__syncthreads();
	}
	}

	// Write result.
	int startCopy=SHARED_MEM_INT2-rLen;
	if(tid>=startCopy)
	{
		d_output[tid-startCopy]=bs_cmpbuf[tid];
	}
}

__global__ void bitonicSortMultipleBlocks_kernel(void* d_keys, void* d_vals, int4 * d_index, int* d_bound, int startBlock, int numBlock, int4 *d_output)
{
	__shared__ int bs_pStart;
	__shared__ int bs_pEnd;
	__shared__ int bs_numElement;
	__shared__ int4 bs_shared[SHARED_MEM_INT2];
	

	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	if(bid>=numBlock) return;

	if(tid==0)
	{
		bs_pStart=d_bound[(bid+startBlock)<<1];
		bs_pEnd=d_bound[((bid+startBlock)<<1)+1];
		bs_numElement=bs_pEnd-bs_pStart;
	}
	__syncthreads();
	// Copy input to shared mem.
	if(tid<bs_numElement)
	{
		bs_shared[tid] = d_index[tid+bs_pStart];
	}
	else
	{
		bs_shared[tid].x =-1;
	}

	__syncthreads();

	// Parallel bitonic sort.
	int compareValue=0;
	for (int k = 2; k <= SHARED_MEM_INT2; k *= 2)
	{
	// Bitonic merge:
	for (int j = k / 2; j>0; j /= 2)
	{
		int ixj = tid ^ j;
		
		if (ixj > tid)
		{
		if ((tid & k) == 0)
		{
			compareValue=getCompareValue(d_keys, d_vals, bs_shared[tid], bs_shared[ixj]);
			//if (shared[tid] > shared[ixj])
			if(compareValue>0)
			{
				swap(bs_shared[tid], bs_shared[ixj]);
			}
		}
		else
		{
			compareValue=getCompareValue(d_keys, d_vals, bs_shared[tid], bs_shared[ixj]);
			//if (shared[tid] < shared[ixj])
			if(compareValue<0)
			{
				swap(bs_shared[tid], bs_shared[ixj]);
			}
		}
		}
		
		__syncthreads();
	}
	}

	// Write result.
	if(tid>=bs_numElement)
	{
		d_output[tid-bs_numElement]=bs_shared[tid];
	}
}


__global__ void initialize_kernel(int4* d_data, int startPos, int rLen, int4 value)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	d_data[pos]=value;
}
void bitonicSortMultipleBlocks(void* d_keys, void * d_vals, int4 * d_index, int* d_bound, int numBlock, int4 * d_output)
{
	int numThreadsPerBlock_x=SHARED_MEM_INT2;
	int numThreadsPerBlock_y=1;
	int numBlock_x=NUM_BLOCK_PER_CHUNK_BITONIC_SORT;
	int numBlock_y=1;
	int numChunk=numBlock/numBlock_x;
	if(numBlock%numBlock_x!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*numBlock_x;
		end=start+numBlock_x;
		if(end>numBlock)
			end=numBlock;
		bitonicSortMultipleBlocks_kernel<<<grid,thread>>>(d_keys, d_vals, d_index, d_bound, start, end-start, d_output);
		cudaThreadSynchronize();
	}
}


void bitonicSortSingleBlock(void* d_keys, void * d_vals, int4 * d_index, int rLen, int4 * d_output)
{
	int numThreadsPerBlock_x=SHARED_MEM_INT2;
	int numThreadsPerBlock_y=1;
	int numBlock_x=1;
	int numBlock_y=1;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	bitonicSortSingleBlock_kernel<<<grid,thread>>>(d_keys, d_vals, d_index, rLen, d_output);
	cudaThreadSynchronize();
}



void initialize(int4 *d_data, int rLen, int4 value)
{
#ifdef __DEVICE_EMULATION__
	int numThreadsPerBlock_x=2;
	int numThreadsPerBlock_y=1;
	int numBlock_x=2;
	int numBlock_y=1;
#else
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
#endif
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		initialize_kernel<<<grid,thread>>>(d_data, start, rLen, value);
	} 
	cudaThreadSynchronize();
}
void bitonicSortGPU(void* d_keys, void * d_vals, int4* d_indexIn, int rLen, int4 *d_indexOut)
{
	unsigned int numRecordsR;

	unsigned int size = rLen;
	unsigned int level = 0;
	while( size != 1 )
	{
		size = size/2;
		level++;
	}

	if( (1<<level) < rLen )
	{
		level++;
	}

	numRecordsR = (1<<level);
	if(rLen<=NUM_THREADS_CHUNK)
	{
		bitonicSortSingleBlock(d_keys, d_vals, d_indexIn, rLen, d_indexOut);
	}
	else
	if( rLen <= 256*1024 )
	{
		unsigned int numThreadsSort = NUM_THREADS_CHUNK;
		if(numRecordsR<NUM_THREADS_CHUNK)
			numRecordsR=NUM_THREADS_CHUNK;
		unsigned int numBlocksXSort = numRecordsR/numThreadsSort;
		unsigned int numBlocksYSort = 1;
		dim3 gridSort( numBlocksXSort, numBlocksYSort );		
		unsigned int memSizeRecordsR = sizeof( int4 ) * numRecordsR;
		//copy the <offset, length> pairs.
		int4* d_index = (int4 *)dMalloc(memSizeRecordsR);
		int4 tempValue;
		tempValue.x=tempValue.y=-1;
		initialize(d_index, numRecordsR, tempValue);
		CE( cudaMemcpy( d_index, d_indexIn, rLen*sizeof(int4), cudaMemcpyDeviceToDevice) );
	

		for( int k = 2; k <= numRecordsR; k *= 2 )
		{
			for( int j = k/2; j > 0; j /= 2 )
			{
				bitonicKernel<<<gridSort, numThreadsSort>>>(d_keys, d_vals, d_index, numRecordsR, k, j);
			}
		}
		CE( cudaMemcpy( d_indexOut, d_index+(numRecordsR-rLen), sizeof(int4)*rLen, cudaMemcpyDeviceToDevice) );
		dFree(d_index, memSizeRecordsR);
		cudaThreadSynchronize();
	}
	else
	{
	CUT_CHECK_ERROR("prescanArrayRecursive before kernels");
		unsigned int numThreadsSort = NUM_THREADS_CHUNK;
		unsigned int numBlocksYSort = 1;
		unsigned int numBlocksXSort = (numRecordsR/numThreadsSort)/numBlocksYSort;
		if(numBlocksXSort>=(1<<16))
		{
			numBlocksXSort=(1<<15);
			numBlocksYSort=(numRecordsR/numThreadsSort)/numBlocksXSort;			
		}
		unsigned int numBlocksChunk = NUM_BLOCKS_CHUNK;
		unsigned int numThreadsChunk = NUM_THREADS_CHUNK;
		
		unsigned int chunkSize = numBlocksChunk*numThreadsChunk;
		unsigned int numChunksR = numRecordsR/chunkSize;

		dim3 gridSort( numBlocksXSort, numBlocksYSort );
		unsigned int memSizeRecordsR = sizeof( int4 ) * numRecordsR;
	CUT_CHECK_ERROR("prescanArrayRecursive before kernels");
		int4* d_index = (int4 *)dMalloc(memSizeRecordsR);
		int4 tempValue;
		tempValue.x=tempValue.y=-1;
	CUT_CHECK_ERROR("prescanArrayRecursive before kernels");

		initialize(d_index, numRecordsR, tempValue);
	CUT_CHECK_ERROR("prescanArrayRecursive before kernels");

		CE( cudaMemcpy( d_index, d_indexIn, rLen*sizeof(int4), cudaMemcpyDeviceToDevice) );

		for( int chunkIdx = 0; chunkIdx < numChunksR; chunkIdx++ )
		{
			unitBitonicSortKernel<<< numBlocksChunk, numThreadsChunk>>>( d_keys, d_vals, d_index, numRecordsR, chunkIdx );
		}

		int j;
		for( int k = numThreadsChunk*2; k <= numRecordsR; k *= 2 )
		{
			for( j = k/2; j > numThreadsChunk/2; j /= 2 )
			{
				bitonicKernel<<<gridSort, numThreadsSort>>>( d_keys, d_vals, d_index, numRecordsR, k, j);
			}

			for( int chunkIdx = 0; chunkIdx < numChunksR; chunkIdx++ )
			{
				partBitonicSortKernel<<< numBlocksChunk, numThreadsChunk>>>(d_keys, d_vals, d_index, numRecordsR, chunkIdx, k/numThreadsSort );
			}
		}

		CE( cudaMemcpy( d_indexOut, d_index+(numRecordsR-rLen), sizeof(int4)*rLen, cudaMemcpyDeviceToDevice) );
		dFree (d_index, memSizeRecordsR);
		cudaThreadSynchronize();
	}
}

__global__ void getIntYArray_kernel(int2* d_input, int startPos, int rLen, int* d_output)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		int2 value=d_input[pos];
		d_output[pos]=value.y;
	}
}


__global__ void getXYArray_kernel(int4* d_input, int startPos, int rLen, int2* d_output)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		int4 value=d_input[pos];
		d_output[pos].x=value.x;
		d_output[pos].y=value.y;
	}
}

__global__ void getZWArray_kernel(int4* d_input, int startPos, int rLen, int2* d_output)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		int4 value=d_input[pos];
		d_output[pos].x=value.z;
		d_output[pos].y=value.w;
	}
}


__global__ void setXYArray_kernel(int4* d_input, int startPos, int rLen, int2* d_value)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		int4 value=d_input[pos];
		value.x=d_value[pos].x;
		value.y=d_value[pos].y;
		d_input[pos]=value;
	}
}

__global__ void setZWArray_kernel(int4* d_input, int startPos, int rLen, int2* d_value)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	if(pos<rLen)
	{
		int4 value=d_input[pos];
		value.z=d_value[pos].x;
		value.w=d_value[pos].y;
		d_input[pos]=value;
	}
}

void getIntYArray(int2 *d_data, int rLen, int* d_output)
{
#ifdef __DEVICE_EMULATION__
	int numThreadsPerBlock_x=2;
	int numThreadsPerBlock_y=1;
	int numBlock_x=2;
	int numBlock_y=1;
#else
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
#endif
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		getIntYArray_kernel<<<grid,thread>>>(d_data, start, rLen, d_output);
	} 
	cudaThreadSynchronize();
}

void getXYArray(int4 *d_data, int rLen, int2* d_output)
{
#ifdef __DEVICE_EMULATION__
	int numThreadsPerBlock_x=2;
	int numThreadsPerBlock_y=1;
	int numBlock_x=2;
	int numBlock_y=1;
#else
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
#endif
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		getXYArray_kernel<<<grid,thread>>>(d_data, start, rLen, d_output);
	} 
	cudaThreadSynchronize();
}

void getZWArray(int4 *d_data, int rLen, int2* d_output)
{
#ifdef __DEVICE_EMULATION__
	int numThreadsPerBlock_x=2;
	int numThreadsPerBlock_y=1;
	int numBlock_x=2;
	int numBlock_y=1;
#else
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
#endif

	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		getZWArray_kernel<<<grid,thread>>>(d_data, start, rLen, d_output);
	} 
	cudaThreadSynchronize();
}

void setXYArray(int4 *d_data, int rLen, int2* d_value)
{
#ifdef __DEVICE_EMULATION__
	int numThreadsPerBlock_x=2;
	int numThreadsPerBlock_y=1;
	int numBlock_x=2;
	int numBlock_y=1;
#else
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
#endif
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		setXYArray_kernel<<<grid,thread>>>(d_data, start, rLen, d_value);
	} 
	cudaThreadSynchronize();
}

void setZWArray(int4 *d_data, int rLen, int2* d_value)
{
#ifdef __DEVICE_EMULATION__
	int numThreadsPerBlock_x=2;
	int numThreadsPerBlock_y=1;
	int numBlock_x=2;
	int numBlock_y=1;
#else
	int numThreadsPerBlock_x=512;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
#endif
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		setZWArray_kernel<<<grid,thread>>>(d_data, start, rLen, d_value);
	} 
	cudaThreadSynchronize();
}
__global__ void copyChunks_kernel(void *d_source, int startPos, int2* d_Rin, int rLen, int *d_sum, void *d_dest)
{
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;	
	const int tid=tx+ty*blockDim.x;
	const int bid=bx+by*gridDim.x;
	const int numThread=blockDim.x;
	const int resultID=(bid)*numThread+tid;
	int pos=startPos+resultID;
	
	if(pos<rLen)
	{
		int2 value=d_Rin[pos];
		int offset=value.x;
		int size=value.y;
		int startWritePos=d_sum[pos];
		int i=0;
		char *source=(char*)d_source;
		char *dest=(char*)d_dest;
		for(i=0;i<size;i++)
		{
			dest[i+startWritePos]=source[i+offset];
		}
		value.x=startWritePos;
		d_Rin[pos]=value;
	}
}

void copyChunks(void *d_source, int2* d_Rin, int rLen, void *d_dest)
{
	//extract the size information for each chunk
	int* d_size = (int *)dMalloc (sizeof(int) * rLen);
	getIntYArray(d_Rin, rLen, d_size);
	//compute the prefix sum for the output positions.
	int* d_sum = (int *)dMalloc (sizeof(int)*rLen);
	prescanArray(d_sum,d_size,rLen);
	dFree(d_size, sizeof(int) * rLen);
	//output
#ifdef __DEVICE_EMULATION__
	int numThreadsPerBlock_x=2;
	int numThreadsPerBlock_y=1;
	int numBlock_x=2;
	int numBlock_y=1;
#else
	int numThreadsPerBlock_x=128;
	int numThreadsPerBlock_y=1;
	int numBlock_x=512;
	int numBlock_y=1;
#endif
	int chunkSize=numBlock_x*numThreadsPerBlock_x;
	int numChunk=rLen/chunkSize;
	if(rLen%chunkSize!=0)
		numChunk++;

	dim3  thread( numThreadsPerBlock_x, numThreadsPerBlock_y, 1);
	dim3  grid( numBlock_x, numBlock_y , 1);
	int i=0;
	int start=0;
	int end=0;
	CUT_CHECK_ERROR("prescanArrayRecursive before kernels");

	for(i=0;i<numChunk;i++)
	{
		start=i*chunkSize;
		end=start+chunkSize;
		if(end>rLen)
			end=rLen;
		copyChunks_kernel<<<grid,thread>>>(d_source, start, d_Rin, rLen, d_sum, d_dest);
	} 
	cudaThreadSynchronize();
	
	dFree(d_sum, sizeof(int)*rLen);
}

void GPUBitonicSortMem (void * d_inputKeyArray, int totalKeySize, void * d_inputValArray, int totalValueSize, 
		  int4 * d_inputPointerArray, int rLen, 
		  void * d_outputKeyArray, void * d_outputValArray, 
		  int4 * d_outputPointerArray
		  )
{

	bitonicSortGPU(d_inputKeyArray, d_inputValArray, d_inputPointerArray, rLen, d_outputPointerArray);

	//!we first scatter the values and then the keys. so that we can reuse d_PA. 
	int2 *d_PA = (int2 *)dMalloc( sizeof(int2)*rLen);

	//scatter the values.
	if(d_inputValArray!=NULL)
	{
		getZWArray(d_outputPointerArray, rLen, d_PA);
		copyChunks(d_inputValArray, d_PA, rLen, d_outputValArray);
		setZWArray(d_outputPointerArray, rLen, d_PA);
	}
	
	//scatter the keys.
	if(d_inputKeyArray!=NULL)
	{
		getXYArray(d_outputPointerArray, rLen, d_PA);
		copyChunks(d_inputKeyArray, d_PA, rLen, d_outputKeyArray);	
		setXYArray(d_outputPointerArray, rLen, d_PA);
	}

	dFree(d_PA,  sizeof(int2)*rLen);
}


};
#endif
