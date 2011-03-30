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

#ifndef HMMHASHTABLE_CU
#define HMMHASHTABLE_CU


#include <stdint.h>
#include "HMMHashTable.h"
#include "HMMSMA.h"
#include "HMMUtilGPU.h"
#include "DLog.h"
#include "../../hash.h"


namespace HMM_GPU{
// --------------------------------------------------
// KeyT and ValueT
// wraps the addr and size of the keys and values
// --------------------------------------------------
// VarLenT
__device__ void VarLenT::init(const void * v, const unsigned int s){
	size=s;
	copyVal(val, v, s);
}
__device__ unsigned int VarLenTRequiredSpace(const unsigned int len){
	return minAlign(sizeof(unsigned int)+len);
}
__device__ VarLenT * newVarLenT(const void * val, const unsigned int len){
	VarLenT * newVar=(VarLenT *)SMA_Malloc(VarLenTRequiredSpace(len));
	newVar->init(val, len);
	return newVar;
}

// --------------------------------------------------
// ValueList
// a list that only allows appending at the head, no delete, no insert at other places
// --------------------------------------------------
// ValueListNode
__device__ void ValueListNode::init(const void * value, const unsigned int len){
	data.init(value,len);
	next=NULL;
}
__device__ void ValueListNode::getValue(const void * & val, unsigned int & valsize) const{
	val=data.val;
	valsize=data.size;
}
__device__ unsigned int ValueListNodeRequiredSpace(const unsigned int len){
	// return the required space for holding this object
	return minAlign(sizeof(ValueListNode*)+VarLenTRequiredSpace(len));
}
__device__ ValueListNode * newValueListNode(const void * val, const unsigned int len){
	// get a new value list node
	ValueListNode * newNode=(ValueListNode *)SMA_Malloc(sizeof(ValueListNode *)+ValueListNodeRequiredSpace(len));
	newNode->init(val, len);
	return newNode;
}

// ValueList
__device__ void ValueList::init(){
	head=NULL;
}
__device__ unsigned int ValueList::size() const{
	// return size, make sure the list is not being modified at the same time
	ValueListNode * curr=head;
	int s=0;
	while(curr!=NULL){
		s++;
		curr=curr->next;
	}
	return s;
}
__device__ void ValueList::insert(const void * val, const unsigned int size){
	// insert a node
	ValueListNode * node=newValueListNode(val, size);
	while(1){
		ValueListNode * local_head=head;
		node->next=local_head;
		if(CASPTR(&(head),local_head,node))
			return;
	}
}
__device__ unsigned int ValueListRequiredSpace(const unsigned int len=0){
	// return the required space for holding this object
	return minAlign(sizeof(ValueListNode *));
}

// --------------------------------------------------
// KeyList
// a list containing a key and the corresponding value list
// --------------------------------------------------
// KeyValueList
__device__ void KeyValueList::init(const void * k, const unsigned int keysize){
	key.init(k,keysize);
	vlist.init();
}
__device__ unsigned int KeyValueListRequiredSpace(const unsigned int keysize){
	// return required space for holding this obj
	return minAlign(ValueListRequiredSpace(keysize)+VarLenTRequiredSpace(keysize));
}
__device__ KeyValueList * newKeyValueList(const void * key, const unsigned int keysize){
	// return a new kye value list
	KeyValueList * newList=(KeyValueList *)SMA_Malloc(KeyValueListRequiredSpace(keysize));
	newList->init(key,keysize);
	return newList;
}

// KeyListNode
__device__ void KeyListNode::init(const void * key, const unsigned int keysize){
	data.init(key,keysize);
	next=NULL;
}
__device__ void KeyListNode::getKey(const void * & key, unsigned int & keysize) const{
	key=data.key.val;
	keysize=data.key.size;
}
__device__ unsigned int KeyListNodeRequiredSpace(const unsigned int keysize){
	// return required space for holding this obj
	return minAlign(KeyValueListRequiredSpace(keysize)+sizeof(KeyListNode*));
}
__device__ KeyListNode * newKeyListNode(const void * key, const unsigned int keysize){
	// get a new key list node
	KeyListNode * newNode=(KeyListNode *)SMA_Malloc(KeyListNodeRequiredSpace(keysize));
	newNode->init(key, keysize);
	return newNode;
}

// KeyList
__device__ void KeyList::init(){
	head=NULL;
}
__device__ int KeyList::KeyListSize() const{
	// --------------------------------------------------
	// return the number of different keys in this list
	// should be used only when there are no other threads writing this list
	// --------------------------------------------------
	KeyListNode * curr=head;
	int s=0;
	while(curr!=NULL){
		s++;
		curr=curr->next;
	}
	return s;
}
__device__ ValueList & KeyList::getValueList(const void * key, const unsigned int keysize){
	// --------------------------------------------------
	// get the valuelist corresponding to the key
	// if not exist, add a new valuelist
	// gurrantees there is no two KeyValueList with the same key
	// --------------------------------------------------
	KeyListNode * curr=head;
	KeyListNode * new_node=NULL;
	// if empty list
	if(curr==NULL){
		new_node=newKeyListNode(key,keysize);
		if(CASPTR(&head,NULL,new_node))
			return (new_node->data).vlist;
	}
	// not empty now
	curr=head;
	while(1){
		if( inter_key_equal((curr->data).key.val, (curr->data).key.size, key, keysize) ){
			//**************** we should free this now here
			// if(new_node!=NULL)
			//	SMA_Free(new_node);
			return (curr->data).vlist;
		}
		if(curr->next==NULL){
			if(new_node==NULL)
				new_node=newKeyListNode(key,keysize);
			if(CASPTR(&(curr->next),NULL,new_node))
				return (new_node->data).vlist;
		}
		// program falls here when curr->next!=NULL or 
		//	curr->next was NULL and then someone inserts a new key list
		curr=curr->next;
	}
}

// --------------------------------------------------
// HashMap
// a hash map that allows concurrent insert and read, but no delete
// user should specify a memory pool, from which the nodes will be allocated
// the memory pool should be large enough to hold all the nodes
// --------------------------------------------------
// --------------------------------------------------
// Functions for traversing through the key/values
// --------------------------------------------------
// ValsIter
__device__ ValsIter::ValsIter():ptr(NULL){};
__device__ ValsIter::ValsIter(ValueListNode * node):ptr(node){};
__device__ ValsIter ValsIter::operator=(ValueListNode * node){
	ptr=node;
	return *this;
};
__device__ ValsIter::operator bool() const{
	return ptr;
}
__device__ bool ValsIter::end() const{
	return ptr==NULL;
}
__device__ ValsIter ValsIter::operator++(){
	ptr=ptr->next;
	return *this;
}
__device__ ValsIter ValsIter::operator++(int){
	ValsIter old(ptr);
	++(*this);
	return old;
}
__device__ void ValsIter::getValue(const void * & val, unsigned int & valsize) const{
	ptr->getValue(val,valsize);
}

// KeysIter
__device__ KeysIter::KeysIter():ptr(NULL){};
__device__ KeysIter::KeysIter(KeyListNode * node):ptr(node){};
__device__ KeysIter KeysIter::operator=(KeyListNode * node){
	ptr=node; 
	return *this;
}
__device__ KeysIter::operator bool() const{
	return ptr;
}
__device__ bool KeysIter::end() const{
	return ptr==NULL;
}
__device__ KeysIter KeysIter::operator++(){
	ptr=ptr->next;
	return *this;
}
__device__ KeysIter KeysIter::operator++(int){
	KeysIter old(ptr);
	++(*this);
	return old;
}
__device__ void KeysIter::getKey(const void * & key, unsigned int & keysize) const{
	ptr->getKey(key,keysize);
}
__device__ ValsIter KeysIter::getValues() const{
	return ptr->data.vlist.head;
}
__device__ void KeysIter::setValue(const void * val, const unsigned int size){
	ValueListNode * node=newValueListNode(val, size);
	ptr->data.vlist.head=node;
}
// HashMultiMap
__device__ void HashMultiMap::insert(const void * key, const unsigned int keysize, const void * val, const unsigned int valsize){
	// --------------------------------------------------
	// insert a key,value pair into the map
	// --------------------------------------------------
	// hash
	unsigned int index=hash_inter_key(key, keysize)%(num_buckets);
	// get corresponding valuelist
	ValueList & vlist=buckets[index].getValueList(key, keysize);
	// insert into the list
	vlist.insert(val, valsize);
}
__device__ void* HashMultiMap::findVal(const void * key, const unsigned int keysize, const void * val, const unsigned int valsize){
	unsigned int index=hash_inter_key(key, keysize)%(num_buckets);
	// get corresponding valuelist
	ValueList & vlist=buckets[index].getValueList(key, keysize);
	if(vlist.head==NULL){
		vlist.insert(val, valsize);
		return NULL;
	}
	else{
		return vlist.head->data.val;
	}
}


__device__ KeysIter HashMultiMap::getBucket(const unsigned int n) const{
	return buckets[n].head;
}
// functions
__host__ HashMultiMap * newHashMultiMap(const unsigned int numBuckets){
	HashMultiMap h_map;
	HashMultiMap * d_map=(HashMultiMap *)SMA_Malloc_From_Host(sizeof(HashMultiMap));
	h_map.num_buckets=numBuckets;
	h_map.buckets=(KeyList*)SMA_Malloc_From_Host(sizeof(KeyList)*numBuckets);
	CE(cudaMemset(h_map.buckets, 0, sizeof(KeyList)*numBuckets));
	memcpyH2D(d_map, &h_map, sizeof(HashMultiMap));
	return d_map;
}

// HMMKVs_t
__device__ HMMKVs_t::HMMKVs_t(KeysIter ki):kit(ki){
	vit=kit.getValues();
}
__device__ void HMMKVs_t::get_key(const void *& ptr, unsigned int & size) const{
	kit.getKey(ptr, size);
}
__device__ void HMMKVs_t::get_val(const void *& ptr, unsigned int & size) const{
	vit.getValue(ptr, size);
}
__device__ void HMMKVs_t::next_val(){
	++vit;
}

__device__ bool HMMKVs_t::end() const{
	return !vit;
}

__device__ unsigned int default_hash(const void * key, const unsigned int keysize){
       unsigned long hash = 5381;
       char *str = (char *)key;
       for (int i = 0; i < keysize; i++)
       {
	       hash = ((hash << 5) + hash) + ((int)str[i]); //  hash * 33 + c 
       }
       return hash;
}

};

#endif
