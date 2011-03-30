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

#ifndef MAPREDUCE_H
#define MAPREDUCE_H

#include <stdio.h>
#include <math.h>

#if defined(HMMHASHTABLECPU_H)
#define HMM_FUNC HMMSchedulerCPU:: 
namespace HMM_CPU{
#elif defined(HMMHASHTABLEGPU_H)
#define HMM_FUNC __device__
namespace HMM_GPU{
#endif


void HMM_FUNC slice(const void * input_array, const unsigned int data_size, 
				const unsigned int unit_size, const unsigned int index, 
				const void * &ret_ptr, unsigned int & ret_size){
	default_slice(input_array, data_size, unit_size, index, ret_ptr, ret_size);
}

void HMM_FUNC map(const void * ptr, const unsigned int size, const global_data_t & global_data){
	const float G=9.8;
	const float delta=0.0001;
	const float max_dist=100000000;
	const float soften=0.0001;
	const int num_body=global_data.num_body;
	const body_t * src=global_data.bodies;

//	for(int i=0;i<size/sizeof(task_t);i++){
		const int x=*(int*)ptr;
		body_t body1=src[x];
		acc_t total_acc;
		total_acc.x=0;
		total_acc.y=0;
		total_acc.z=0;
		for(int y=0;y<num_body;y++){
			if(x==y)
				continue;
			const body_t body2=src[y];
			// calculate r/(r^2+soften)^1.5
			float r1=body1.pos_x-body2.pos_x;
			float r2=body1.pos_y-body2.pos_y;
			float r3=body1.pos_z-body2.pos_z;

			float rr=r1*r1+r2*r2+r3*r3;
			// ignore bodies that are too far
			if(rr>=max_dist)
				continue;
			float r_soften=rr+soften;
			r_soften=sqrt(r_soften);
			r_soften=r_soften*r_soften*r_soften;
			float invert=1/r_soften;

			r1*=invert;
			r2*=invert;
			r3*=invert;
			total_acc.x+=0-r1*body2.mass;
			total_acc.y+=0-r2*body2.mass;
			total_acc.z+=0-r3*body2.mass;
		}

		// update pos and speed
		total_acc.x*=G;
		total_acc.y*=G;
		total_acc.z*=G;
		body1.vel_x+=total_acc.x*delta;
		body1.vel_y+=total_acc.y*delta;
		body1.vel_z+=total_acc.z*delta;

		body1.pos_x+=delta*body1.vel_x;
		body1.pos_y+=delta*body1.vel_y;
		body1.pos_z+=delta*body1.vel_z;
		emit_intermediate(&x, sizeof(int), &body1, sizeof(body_t));
//	}
}

void HMM_FUNC reduce(HMMKVs_t & kv, const global_data_t & global_data){
	const void * val;
	unsigned int valsize;
	kv.get_val(val,valsize);
	emit(kv, val, valsize);
}


void HMM_FUNC combine(void * old_val, const void * new_val){

}

};
#endif
