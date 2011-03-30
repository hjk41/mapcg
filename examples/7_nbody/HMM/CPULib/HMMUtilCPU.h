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

#ifndef CPU_HMMUTIL_H
#define CPU_HMMUTIL_H

#include <stdint.h>

namespace HMM_CPU{


//===================================
// alignment
//===================================
const int MIN_ALIGN=4;	// minimum alignment, in bytes
template<class T>
T HMMAlign(T n, uint32_t b){
	return ((uint32_t)n&(b-1))==0 ? n : n+b-((uint32_t)n&(b-1));
}

template<class T>
T minAlign(T n){
	return ((uint32_t)n&(MIN_ALIGN-1))==0 ? n : n+MIN_ALIGN-((uint32_t)n&(MIN_ALIGN-1));
}

//===================================
// atomic operations
//===================================
uint32_t CASUint32(uint32_t * ptr, const uint32_t o, const uint32_t n);

template<class T1, class T2, class T3>
bool CAS32(T1 * addr, T2 old_val, T3 new_val){
	return *(uint32_t *)(&old_val)==CASUint32((uint32_t *)addr, *(uint32_t *)(&old_val), *(uint32_t *)(&new_val));
}

#ifndef LONG_PTR
template<class T1, class T2, class T3>
bool CASPTR(T1 * addr, T2 old_val, T3 new_val){
	return CAS32(addr, old_val, new_val);
}
#else
uint64_t CASUint64(uint64_t * ptr, const uint64_t o, const uint64_t n);

template<class T1, class T2, class T3>
bool CAS64(T1 * addr, T2 old_val, T3 new_val){
	return *(uint64_t *)(&old_val)==CASUint64((uint64_t *)addr, *(uint64_t *)(&old_val), *(uint64_t *)(&new_val));
}

template<class T1, class T2, class T3>
bool CASPTR(T1 * addr, T2 old_val, T3 new_val){
	return CAS64(addr, old_val, new_val);
}
#endif

uint32_t ADDUint32(volatile uint32_t * ptr, const uint32_t inc);

//===================================
// copy value
//===================================
void copyVal(void * dst, const void * src, const unsigned int);

};// namespace HMM_CPU

#endif	// define CPU_HMMUTIL_H
