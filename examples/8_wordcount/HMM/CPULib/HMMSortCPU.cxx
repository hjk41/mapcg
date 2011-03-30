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
#include <string.h>
#include <algorithm>
using namespace std;

#include "HMMSortCPU.h"
#include "../../sort_compare.h"

namespace HMM_CPU{
	class Helper{
	private:
		const char * keys;
		const char * vals;
	public:
		Helper(const char * k, const char * v):keys(k),vals(v){};
		bool operator()(const int4 & lhs, const int4 & rhs) const{
			return less_than(keys+lhs.x, lhs.y, vals+lhs.z, lhs.w, keys+rhs.x, rhs.y, vals+rhs.z, rhs.w);
		}
	};

	void sort_chunk(void * v_keys, void * v_vals, int4 * index, unsigned int num_pairs){
		char * keys=(char*)v_keys;
		char * vals=(char*)v_vals;
		int4 tmp=index[num_pairs-1];
		unsigned int total_key_size=tmp.x+tmp.y;
		unsigned int total_val_size=tmp.z+tmp.w;
		char * new_keys=new char[total_key_size];
		char * new_vals=new char[total_val_size];
		int4 * new_index=new int4[num_pairs];		

		sort(index, index+num_pairs, Helper(keys, vals));

		unsigned int key_idx=0;
		unsigned int val_idx=0;
		for(int i=0;i<num_pairs;i++){
			int4 tmp=index[i];

			int4 & new_idx=new_index[i];
			new_idx.x=key_idx;
			new_idx.y=tmp.y;
			new_idx.z=val_idx;
			new_idx.w=tmp.w;

			memcpy(new_keys+key_idx, keys+tmp.x, tmp.y);
			key_idx+=tmp.y;
			memcpy(new_vals+val_idx, vals+tmp.z, tmp.w);
			val_idx+=tmp.w;
		}
		memcpy(keys, new_keys, total_key_size);
		memcpy(vals, new_vals, total_val_size);
		memcpy(index, new_index, num_pairs*sizeof(int4));
		delete[] new_keys;
		delete[] new_vals;
		delete[] new_index;
	}
};	// namespace HMM_CPU

