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

#ifndef HMMGLOBALDATA_H
#define HMMGLOBALDATA_H

#include "../DS.h"

void declare_global_array(unsigned int offset,unsigned int size);
void mark_array_as_dirty(unsigned int offset);
void mark_all_as_clean();

// macros
#define DECLARE_GLOBAL_ARRAY(field,size) do{global_data_t tmp; int offset=(char*)&tmp.field-(char*)&tmp; declare_global_array(offset, size);}while(0)
#define UPDATE_GLOBAL_ARRAY(field) do{global_data_t tmp; int offset=(char*)&tmp.field-(char*)&tmp; mark_array_as_dirty(offset);}while(0)

// manipulate global data on GPU
void sync_global_data(global_data_t * h_ptr, global_data_t * d_ptr);
global_data_t * malloc_gpu_global_data(global_data_t * h_global_data);
void free_gpu_global_data(global_data_t * d_global_data);
unsigned int get_global_data_size();
#endif
