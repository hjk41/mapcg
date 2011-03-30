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

#include <stdint.h>
#include <string.h>
#include "HMMHashTableCPU.h"
#include "HMMUtilCPU.h"
#include "HMMSMACPU.h"
#include "../../hash.h"

namespace HMM_CPU{

unsigned int default_hash(const void * key, const unsigned int keysize){
	unsigned long hash = 5381;
	char *str = (char *)key;
	for (int i = 0; i < keysize; i++)
	{
		hash = ((hash << 5) + hash) + ((int)str[i]); /* hash * 33 + c */
	}
	return hash;
}

// --------------------------------------------------
// KeyT and ValueT
// wraps the addr and size of the keys and values
// --------------------------------------------------
// VarLenT
void VarLenT::init(const void * v, const unsigned int s){
	size=s;
	copyVal(val, v, s);
}
unsigned int VarLenTRequiredSpace(const unsigned int len){
	return minAlign(sizeof(unsigned int)+len);
}
VarLenT * newVarLenT(const void * val, const unsigned int len){
	VarLenT * newVar=(VarLenT *)SMA_Malloc(VarLenTRequiredSpace(len));
	newVar->init(val, len);
	return newVar;
}

// --------------------------------------------------
// ValueList
// a list that only allows appending at the head, no delete, no insert at other places
// --------------------------------------------------
// ValueListNode
void ValueListNode::init(const void * value, const unsigned int len){
	data.init(value,len);
	next=NULL;
}
void ValueListNode::getValue(const void * & val, unsigned int & valsize) const{
	val=data.val;
	valsize=data.size;
}
unsigned int ValueListNodeRequiredSpace(const unsigned int len){
	// return the required space for holding this object
	return minAlign(sizeof(ValueListNode*)+VarLenTRequiredSpace(len));
}
ValueListNode * newValueListNode(const void * val, const unsigned int len){
	// get a new value list node
	ValueListNode * newNode=(ValueListNode *)SMA_Malloc(sizeof(ValueListNode *)+ValueListNodeRequiredSpace(len));
	newNode->init(val, len);
	return newNode;
}

// ValueList
void ValueList::init(){
	head=NULL;
}
unsigned int ValueList::size() const{
	// return size, make sure the list is not being modified at the same time
	ValueListNode * curr=head;
	int s=0;
	while(curr!=NULL){
		s++;
		curr=curr->next;
	}
	return s;
}
void ValueList::insert(const void * val, const unsigned int size){
	// insert a node
	ValueListNode * node=newValueListNode(val, size);
	node->next=head;
	head=node;
}
unsigned int ValueListRequiredSpace(const unsigned int len=0){
	// return the required space for holding this object
	return minAlign(sizeof(ValueListNode *));
}

// --------------------------------------------------
// KeyList
// a list containing a key and the corresponding value list
// --------------------------------------------------
// KeyValueList
void KeyValueList::init(const void * k, const unsigned int keysize){
	key.init(k,keysize);
	vlist.init();
}
unsigned int KeyValueListRequiredSpace(const unsigned int keysize){
	// return required space for holding this obj
	return minAlign(ValueListRequiredSpace(keysize)+VarLenTRequiredSpace(keysize));
}
KeyValueList * newKeyValueList(const void * key, const unsigned int keysize){
	// return a new kye value list
	KeyValueList * newList=(KeyValueList *)SMA_Malloc(KeyValueListRequiredSpace(keysize));
	newList->init(key,keysize);
	return newList;
}

// KeyListNode
void KeyListNode::init(const void * key, const unsigned int keysize){
	data.init(key,keysize);
	next=NULL;
}
void KeyListNode::getKey(const void * & key, unsigned int & keysize) const{
	key=data.key.val;
	keysize=data.key.size;
}
unsigned int KeyListNodeRequiredSpace(const unsigned int keysize){
	// return required space for holding this obj
	return minAlign(KeyValueListRequiredSpace(keysize)+sizeof(KeyListNode*));
}
KeyListNode * newKeyListNode(const void * key, const unsigned int keysize){
	// get a new key list node
	KeyListNode * newNode=(KeyListNode *)SMA_Malloc(KeyListNodeRequiredSpace(keysize));
	newNode->init(key, keysize);
	return newNode;
}

// KeyList
void KeyList::init(){
	head=NULL;
}
int KeyList::KeyListSize() const{
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
ValueList & KeyList::getValueList(const void * key, const unsigned int keysize){
	// --------------------------------------------------
	// get the valuelist corresponding to the key
	// if not exist, add a new valuelist
	// gurrantees there is no two KeyValueList with the same key
	// --------------------------------------------------
	// if empty list
	if(head==NULL){
		head=newKeyListNode(key,keysize);
		return (head->data).vlist;
	}
	// not empty now
	KeyListNode * curr=head;
	while(1){
		if( inter_key_equal((curr->data).key.val, (curr->data).key.size, key, keysize) ){
			return (curr->data).vlist;
		}
		if(curr->next==NULL){
			KeyListNode * new_node=newKeyListNode(key,keysize);
			curr->next=new_node;
			return (new_node->data).vlist;
		}
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
ValsIter::ValsIter():ptr(NULL){};
ValsIter::ValsIter(ValueListNode * node):ptr(node){};
ValsIter ValsIter::operator=(ValueListNode * node){
	ptr=node;
	return *this;
};
ValsIter::operator bool() const{
	return ptr;
}
bool ValsIter::end() const{
	return ptr==NULL;
}
ValsIter ValsIter::operator++(){
	ptr=ptr->next;
	return *this;
}
ValsIter ValsIter::operator++(int){
	ValsIter old(ptr);
	++(*this);
	return old;
}
void ValsIter::getValue(const void * & val, unsigned int & valsize) const{
	ptr->getValue(val,valsize);
}

// KeysIter
KeysIter::KeysIter():ptr(NULL){};
KeysIter::KeysIter(KeyListNode * node):ptr(node){};
KeysIter KeysIter::operator=(KeyListNode * node){
	ptr=node; 
	return *this;
}
KeysIter::operator bool() const{
	return ptr;
}
bool KeysIter::end() const{
	return ptr==NULL;
}
KeysIter KeysIter::operator++(){
	ptr=ptr->next;
	return *this;
}
KeysIter KeysIter::operator++(int){
	KeysIter old(ptr);
	++(*this);
	return old;
}
void KeysIter::getKey(const void * & key, unsigned int & keysize) const{
	ptr->getKey(key,keysize);
}
ValsIter KeysIter::getValues() const{
	return ptr->data.vlist.head;
}
void KeysIter::setValue(const void * val, const unsigned int size){
	ValueListNode * node=newValueListNode(val, size);
	ptr->data.vlist.head=node;
}

// HashMultiMap
void HashMultiMap::insert(const void * key, const unsigned int keysize, const void * val, const unsigned int valsize){
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
bool HashMultiMap::findVal(const void * key, const unsigned int keysize,const void *&val, const unsigned int valsize){
	unsigned int index=hash_inter_key(key, keysize)%(num_buckets);
	// get corresponding valuelist
	ValueList & vlist=buckets[index].getValueList(key, keysize);
	if(vlist.head==NULL){
		vlist.insert(val, valsize);
		return false;
	}
	else{
		val=vlist.head->data.val;
		return true;
	}
}


KeysIter HashMultiMap::getBucket(const unsigned int n) const{
	return buckets[n].head;
}

// functions
HashMultiMap * newHashMultiMap(const unsigned int numBuckets){
	HashMultiMap * map=new HashMultiMap;
	map->num_buckets=numBuckets;
	map->buckets=new KeyList[numBuckets];
	memset(map->buckets, 0, sizeof(KeyList)*numBuckets);
	return map;
}

void delHashMultiMap(HashMultiMap * map){
	delete[] map->buckets;
	delete map;
}

// HMMKVs_t
HMMKVs_t::HMMKVs_t(KeysIter ki):kit(ki){
	vit=kit.getValues();
}
void HMMKVs_t::get_key(const void * & ptr, unsigned int & size) const{
	kit.getKey(ptr, size);
}

void HMMKVs_t::get_val(const void *& ptr, unsigned int & size) const{
	vit.getValue(ptr, size);
}

void HMMKVs_t::next_val(){
	++vit;
}

bool HMMKVs_t::end() const{
	return !vit;
}

};
