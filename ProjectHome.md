This is the open source implementation of the framework presented in the paper "MapCG: writing parallel program portable between CPU and GPU".

MapCG is a programming framework based on MapReduce. It enables programmers to write parallel programs on CPU and GPU with relative ease. Currently, the CPU runtime library is based on OpenMP, and the one on GPU is based on CUDA.

It is tested on Fedora 12 x86\_64 with CUDA 3.2. Other platforms are yet to be tested.