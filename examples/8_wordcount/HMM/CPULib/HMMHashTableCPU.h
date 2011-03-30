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

#ifndef HMMHASHTABLECPU_H
#define HMMHASHTABLECPU_H

#include <stdint.h>

namespace HMM_CPU{

// --------------------------------------------------
// KeyT and ValueT
// wraps the addr and size of the keys and values
// --------------------------------------------------
struct VarLenT{
public:
	unsigned int size;
	char val[];
	void init(const void * v, const unsigned int s);
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
	void init(const void * value, const unsigned int len);
	void getValue(const void * & val, unsigned int & valsize) const;
};

struct ValueList{
public:
	ValueListNode * head;
	// non-static funtions
	void init();
	unsigned int size() const;
	void insert(const void * val, const unsigned int size);
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
	void init(const void * k, const unsigned int keysize);
};

struct KeyListNode{
public:
	KeyListNode * next;
	KeyValueList data;
	// non-static functions
	void init(const void * key, const unsigned int keysize);
	void getKey(const void * & key, unsigned int & keysize) const;
};

struct KeyList{
	KeyListNode * head;
	// functions
	void init();
	int KeyListSize() const;
	ValueList & getValueList(const void * key, const unsigned int keysize);
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
	ValsIter();
	ValsIter(ValueListNode * node);
	ValsIter operator=(ValueListNode * node);
	// whether the list ends
	operator bool() const;
	bool end() const;
	// prefix and postfix increment
	ValsIter operator++();
	ValsIter operator++(int);
	// get the value
	void getValue(const void * & val, unsigned int & valsize) const;
};

struct KeysIter{
	KeyListNode * ptr;
	// constructors and copy operator
	KeysIter();
	KeysIter(KeyListNode * node);
	KeysIter operator=(KeyListNode * node);
	// whether this list ends
	operator bool() const;
	bool end() const;
	// prefix and postfix increment
	KeysIter operator++();
	KeysIter operator++(int);
	// get the key
	void getKey(const void * & key, unsigned int & keysize) const;
	// get the value iterator
	ValsIter getValues() const;
	
	// used in reduce emit
	// restriction: only one thread can do this to a specific KeysIter
	//	after this call, the intermediate values will be lost
	void setValue(const void * val, const unsigned int size);
};

struct HashMultiMap{
	unsigned int num_buckets;
	KeyList * buckets;
	// non-static functions
	void insert(const void * key, const unsigned int keysize, const void * val, const unsigned int valsize);
	KeysIter getBucket(const unsigned int n) const;
	// get pointer to value by key, only the first value returned
	bool findVal(const void * key, const unsigned int keysize, const void *&val, const unsigned int valsize);
};

HashMultiMap * newHashMultiMap(const unsigned int numBuckets);
void delHashMultiMap(HashMultiMap * map);


// --------------------------------------------------
// iterator for reduce()
// --------------------------------------------------
// a key, along with the associated values
struct HMMKVs_t{
	// data
	KeysIter kit;
	ValsIter vit;
	// functions
	HMMKVs_t(KeysIter ki);
	void get_key(const void * & ptr, unsigned int & size) const;
	void get_val(const void * & ptr, unsigned int & size) const;
	void next_val();
	bool end() const;
};

bool inter_key_equal(const void * key1, const unsigned int keysize1, const void * key2, const unsigned int keysize2);
unsigned int default_hash(const void * key, const unsigned int keysize);
};
#endif
