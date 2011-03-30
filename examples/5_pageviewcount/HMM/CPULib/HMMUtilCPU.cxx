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
#include "HMMUtilCPU.h"

namespace HMM_CPU{

uint32_t CASUint32(uint32_t * ptr, const uint32_t o, const uint32_t n){
	uint32_t prev;
	__asm__ __volatile__("LOCK cmpxchgl %k1,%2"
			     : "=a"(prev)
			     : "r"(n), "m"(*(ptr)), "0"(o)
			     : "memory");
	return prev; 
}

uint32_t ADDUint32(volatile uint32_t * ptr, const uint32_t inc){
	uint32_t prev;
	__asm__ __volatile__("LOCK xaddl %0,%1"
			     : "=a"(prev)
			     : "m"(*(ptr)), "r"(inc)
			     : "memory");
	return prev; 
}



#ifndef LONG_PTR
#else
uint64_t CASUint64(uint64_t * ptr, const uint64_t o, const uint64_t n){
	uint64_t prev;
	__asm__ __volatile__("LOCK cmpxchgq %k1,%2"
			     : "=a"(prev)
			     : "r"(n), "m"(*(ptr)), "0"(o)
			     : "memory");
	return prev; 
}
#endif

void copyVal(void * dst, const void * src, const unsigned int size){
	memcpy(dst,src,size);
}

};// namespace HMM_CPU

