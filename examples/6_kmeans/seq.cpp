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

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <list>
#include "time.h"
#include "math.h"
using namespace std;

#include "DS.h"
#include "HMM/UtilLib/HMMCommUtilTemplate.h"

#define VECTOR_SPACE    1000

//--------------------------------------------------
//generate data
//--------------------------------------------------
int *GenPoints(int num_points, int dim)
{
	srand(1024);
	int *buf = (int*)malloc(sizeof(int)*num_points*dim);

	for (int i = 0; i < num_points; i++)
		for (int j = 0; j < dim; j++)
			buf[i*dim+j] = rand()%VECTOR_SPACE;
//			buf[i*dim+j] = (i*dim+j)%VECTOR_SPACE;

	return buf;
}

int *GenMeans(int num_means, int dim)
{
	srand(1024);
	int *buf = (int*)malloc(dim*num_means*sizeof(int));
	for (int i = 0; i < num_means; i++)
		for (int j = 0; j < dim; j++)
//			buf[i*dim+j] = (i*dim+j)%VECTOR_SPACE;
			buf[i*dim + j] = rand()%VECTOR_SPACE;
	return buf;
}

inline unsigned int get_sq_dist(int* point, int* mean, int dim)
{
	int i;
	unsigned int sum = 0;
	for(i = 0; i < dim; i++)
	{
		sum += (point[i] - mean[i])*(point[i] - mean[i]);
	}
	return sum;
}

int main(int argc, char * argv[]){
	int num_points;
	int dim;
	int num_means;
	if (!get_opt(argc, argv, "vn", num_points) ||
	 !get_opt(argc, argv, "mn", num_means) ||
	 !get_opt(argc, argv, "dim", dim)){
		 printf("Usage: %s -vn INT -mn INT -dim INT\n"
			"\t-vn vector number\n"
			"\t-mn mean number\n"
			"\t-dim vector dimension\n", argv[0]);
		return 1;
	}

	int * points=GenPoints(num_points, dim);
	int * means=GenMeans(num_means, dim);
	int * index=new int[num_points];
	for(int i=0;i<num_points;i++)
		index[i]=i;
	int * pos=new int[num_points];
	for(int i=0;i<num_points;i++)
		pos[i]=-1;

	global_data_t global_data;
	global_data.dim = dim;
	global_data.num_means = num_means;
	global_data.points=points;
	global_data.means = means;

	double t2,t3;
	double time=0;
	bool changed=true;
	int * new_pos=new int[num_points];
	list<int> * clusters=new list<int>[num_means];
	int iter=0;
	while(changed){
		cout<<"iter: "<<iter++<<endl;
		changed=false;
		// assign points to differnt clusters
		cout<<"map"<<endl;
		for(int i=0;i<num_points;i++){
			int index = i;
			int dim = global_data.dim;
			int * point=global_data.points+dim*index;
			int num_means = global_data.num_means;
			int* means = global_data.means;

			unsigned int min_dist, cur_dist;
			min_dist = get_sq_dist(point, means, dim);
			int min_idx = 0;
			for(int j = 1; j < num_means; j++)
			{
				cur_dist = get_sq_dist(point, &means[j*dim], dim);
				if(cur_dist < min_dist)
				{
					min_dist = cur_dist;
					min_idx = j;
				}
			}
			new_pos[i]=min_idx;
			clusters[min_idx].push_back(i);
		}
		// recalculate means
		cout<<"reduce"<<endl;
		for(int i=0;i<num_means;i++){
			int index=i;
			int dim = global_data.dim;
			int num_means = global_data.num_means;
			int* means = &(global_data.means[index*dim]);
			unsigned int val_num = clusters[i].size();

			for(int j = 0; j < dim; j++)
			{
				means[j] = 0;
			}
			for(list<int>::iterator it=clusters[i].begin();it!=clusters[i].end();++it){
				int pIdx = *it;
				int * point=global_data.points+pIdx*dim;
				for(int j = 0; j < dim; j++){
					means[j] += point[j];
				}
			}
			clusters[i].clear();
			if(val_num!=0){
				for(int j = 0; j < dim; j++)
				{
					means[j] /= val_num;
				}
			}
		}
		// check if any of the points has moved
		cout<<"post process"<<endl;
		for(int i=0;i<num_points;i++){
			if(new_pos[i]!=pos[i]){
				changed=true;
				break;
			}
		}
		memcpy(pos,new_pos,sizeof(int)*num_points);
	}

	cout<<"total time: "<<time<<endl;

	for(int i=0;i<10 && i<num_points;i++){
		printf("(%d,%d)\n",i,pos[i]);
	}

	return 0;
}
