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

#ifndef DS_H
#define DS_H

#ifndef __CUDACC__
#ifndef float3
	struct float3{
		float x;
		float y;
		float z;
	};
#endif

#ifndef uint2
	struct uint2{
		unsigned int x;
		unsigned int y;
	};
#endif
#endif

struct body_t{
	float pos_x;
	float pos_y;
	float pos_z;
	float vel_x;
	float vel_y;
	float vel_z;
	float mass;
};

typedef float3 acc_t;

typedef int task_t;

struct global_data_t{
	body_t * bodies;
	float delta;
	float soften;
	int num_body;
};

#endif
