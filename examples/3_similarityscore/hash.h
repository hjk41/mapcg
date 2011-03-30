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

#if defined(HMMHASHTABLECPU_H)
#define HMM_FUNC 
namespace HMM_CPU{
#elif defined(HMMHASHTABLEGPU_H)
#define HMM_FUNC __device__
namespace HMM_GPU{
#endif

#include "DS.h"

HMM_FUNC unsigned hash_inter_key(const void * key, const unsigned int keysize){
	return *(unsigned int*)key;
}

HMM_FUNC bool inter_key_equal(const void * key1, const unsigned int keysize1, const void * key2, const unsigned int keysize2){
	return *(unsigned int*)key1==*(unsigned int *)key2;
}

};
