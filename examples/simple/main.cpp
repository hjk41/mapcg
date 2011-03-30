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
using namespace std;

#include <stdlib.h>
#include "HMM/HMMMain.h"

const int MODE=10;

void gen_ints(int * ints, unsigned int n){
	srand(1000);
	for(unsigned int i=0;i<n;i++){
//		ints[i]=rand()%MODE;
		ints[i]=i%MODE;
	}
}

int main(int argc, char * argv[]){
	if(argc<2){
		cout<<"usage: "<<argv[0]<<" <num_ints>"<<endl;
		return 1;
	}

	int NUM_INTS=atoi(argv[1]);

	int * ints=new int[NUM_INTS];
	gen_ints(ints, NUM_INTS);

	HMMSchedulerSpec args(argc,argv);
	args.input=ints;
	args.input_size=NUM_INTS*sizeof(int);
	args.unit_size=sizeof(int);
	args.num_hash_buckets=MODE;

	HMMMainScheduler scheduler(args);

	scheduler.do_map_reduce();

	OutputChunk output=scheduler.get_output();
	string str;
	for(int i=0;i<output.count && i<10;i++){
		int keyIdx=output.index[i].x;
		int valIdx=output.index[i].z;
		int key=*(const int *)((const char * )output.keys+keyIdx);
		int val=*(const int *)((const char * )output.vals+valIdx);
		cout<<"("<< key <<","<< val <<")"<<endl;
	}

	delete[] ints;
	return 0;
}
