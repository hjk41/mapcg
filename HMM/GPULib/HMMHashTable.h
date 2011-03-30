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

#ifndef HMMHASHTABLEGPU_H
#define HMMHASHTABLEGPU_H

#include <stdint.h>

namespace HMM_GPU{

// --------------------------------------------------
// KeyT and ValueT
// wraps the addr and size of the keys and values
// --------------------------------------------------
struct VarLenT{
public:
	unsigned int size;
	char val[];
	__device__ void init(const void * v, const unsigned int s);
};

typedef VarLenT KeyT;
typedef VarLenT ValueT;

// --------------------------------------------------
// ValueList
// a list that only allows appending at the head, no delete, no insert at other places
// --------------------------------------------------
struct ValueListNode{
public:
	ValueListNode * next;	
	ValueT data;
	// non-static functions
	__device__ void init(const void * value, const unsigned int len);
	__device__ void getValue(const void * & val, unsigned int & valsize) const;
};

struct ValueList{
public:
	ValueListNode * head;
	// non-static funtions
	__device__ void init();
	__device__ unsigned int size() const;
	__device__ void insert(const void * val, const unsigned int size);
};

// --------------------------------------------------
// KeyList
// a list containing a key and the corresponding value list
// --------------------------------------------------
struct KeyValueList{
public:
	ValueList vlist;
	KeyT key;
	// non-static functions
	__device__ void init(const void * k, const unsigned int keysize);
};

struct KeyListNode{
public:
	KeyListNode * next;
	KeyValueList data;
	// non-static functions
	__device__ void init(const void * key, const unsigned int keysize);
	__device__ void getKey(const void * & key, unsigned int & keysize) const;
};

struct KeyList{
	KeyListNode * head;
	// functions
	__device__ void init();
	__device__ int KeyListSize() const;
	__device__ ValueList & getValueList(const void * key, const unsigned int keysize);
};	// KeyList

// --------------------------------------------------
// HashMap
// a hash map that allows concurrent insert and read, but no delete
// user should specify a memory pool, from which the nodes will be allocated
// the memory pool should be large enough to hold all the nodes
// --------------------------------------------------
struct KeysIter;
struct ValsIter;
// --------------------------------------------------
// Functions for traversing through the key/values
// --------------------------------------------------
struct ValsIter{
	ValueListNode * ptr;
	// constructors and copy operator
	__device__ ValsIter();
	__device__ ValsIter(ValueListNode * node);
	__device__ ValsIter operator=(ValueListNode * node);
	// whether the list ends
	__device__ operator bool() const;
	__device__ bool end() const;
	// prefix and postfix increment
	__device__ ValsIter operator++();
	__device__ ValsIter operator++(int);
	// get the value
	__device__ void getValue(const void * & val, unsigned int & valsize) const;
};

struct KeysIter{
	KeyListNode * ptr;
	// constructors and copy operator
	__device__ KeysIter();
	__device__ KeysIter(KeyListNode * node);
	__device__ KeysIter operator=(KeyListNode * node);
	// whether this list ends
	__device__ operator bool() const;
	__device__ bool end() const;
	// prefix and postfix increment
	__device__ KeysIter operator++();
	__device__ KeysIter operator++(int);
	// get the key
	__device__ void getKey(const void * & key, unsigned int & keysize) const;
	// get the value iterator
	__device__ ValsIter getValues() const;
	
	// used in reduce emit
	// restriction: only one thread can do this to a specific KeysIter
	//	after this call, the intermediate values will be lost
	__device__ void setValue(const void * val, const unsigned int size);
};

struct HashMultiMap{
	unsigned int num_buckets;
	KeyList * buckets;
	// non-static functions
	__device__ void insert(const void * key, const unsigned int keysize, const void * val, const unsigned int valsize);
	__device__ KeysIter getBucket(const unsigned int n) const;
	// get pointer to value by key, only the first value returned
	__device__ void* findVal(const void * key, const unsigned int keysize,const void *val, const unsigned int valsize);
};

__host__ HashMultiMap * newHashMultiMap(const unsigned int numBuckets);

// a key, along with the associated keys
struct HMMKVs_t{
	// data
	KeysIter kit;
	ValsIter vit;
	// functions
	__device__ HMMKVs_t(KeysIter ki);
	__device__ void get_key(const void * &, unsigned int &) const;
	__device__ void get_val(const void * &, unsigned int &) const;
	__device__ void next_val();
	__device__ bool end() const;
};

__device__ unsigned int default_hash(const void * key, const unsigned int keysize);
__device__ bool inter_key_equal(const void * key1, const unsigned int keysize1, const void * key2, const unsigned int keysize2);
};
#endif
