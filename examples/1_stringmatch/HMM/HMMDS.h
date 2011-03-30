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

#ifndef HMMDS_H
#define HMMDS_H

#include <map>

#ifndef __CUDACC__
	struct __attribute__((aligned(16))) int4{
		int x;
		int y;
		int z;
		int w;
	};
	struct __attribute__((aligned(8))) int2{
		int x;
		int y;
	};
#else
	#include <cuda.h>
#endif

struct OutputChunk{
	char * keys;
	char * vals;
	int4 * index;
	unsigned int count;
	int2 * buckets;		// x: index of the first key in bucket i; y: the number of keys in this bucket
	unsigned int num_buckets;
};

struct GlobalDataListTuple{
	unsigned int size;
	enum _type{array, variable} type;
	GlobalDataListTuple():size(0),type(variable){};
	GlobalDataListTuple(unsigned int s, _type t):size(s),type(t){};
};
typedef std::map<unsigned int, GlobalDataListTuple> GlobalDataList;

typedef unsigned long long large_size_t;

#endif
