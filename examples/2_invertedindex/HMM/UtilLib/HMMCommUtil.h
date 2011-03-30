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

#ifndef HMMCOMMUTIL_H
#define HMMCOMMUTIL_H

#include <stdio.h>

#define __LOG__
#define __TIMING__

//=============================
// CUDA related code
//=============================
#  define CUT_CHECK_ERROR(errorMessage) do {                             \
    cudaError_t err = cudaGetLastError();                                   \
    if( cudaSuccess != err) {                                           \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                               \
    }                                                                   \
    err = cudaThreadSynchronize();                                         \
    if( cudaSuccess != err) {                                           \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                               \
    } } while (0)

#  define CE(call) do {                                \
        call;CUT_CHECK_ERROR("------- Error ------\n"); \
     } while (0)

int get_num_gpus();
void set_gpu_num(int num);

// map a file into the memory
void map_file(const char * filename, void * & buf, unsigned int & size);

//=============================
// Timing and Logging
//=============================
double get_time();	// get the current time, implicit cudaThreadSynchronize
void init_timer();	// initialize timer
double time_elapsed();
#ifdef __TIMING__
	#define printtime(...) printf(__VA_ARGS__)
#else
	#define printtime(...)
#endif

#ifdef __LOG__
	#define printlog(...) printf(__VA_ARGS__)
#else
	#define printlog(...)
#endif

// print error
#define print_error(...) {printf(__VA_ARGS__);exit(1);}


#endif
