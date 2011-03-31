OPT=-O3 -g
DEVICE_EMU=#-deviceemu
CFLAGS=$(OPT) -fopenmp
NVFLAGS=$(OPT) -arch=sm_13 $(DEVICE_EMU)
LDFLAGS=$(OPT)  -lgomp -lpthread
#LDFLAGS=$(OPT) -liomp5 -lpthread -lirc -L/opt/intel/Compiler/11.0/074/lib/ia32
CC=g++
NVCC=nvcc -DLONG_PTR
LD=nvcc $(DEVICE_EMU)
OBJS=main.o \
	HMM/HMMCPUScheduler.o HMM/HMMGPUScheduler.o HMM/HMMMain.o HMM/HMMSchedulerSpec.o HMM/HMMGlobalData.o\
	HMM/CPULib/HMMHashTableCPU.o HMM/CPULib/HMMSMACPU.o HMM/CPULib/HMMSchedulerCPU.o HMM/CPULib/HMMSortCPU.o HMM/CPULib/HMMUtilCPU.o \
	HMM/GPULib/HMMSchedulerGPU.o \
	HMM/UtilLib/HMMCommUtil.o


.PHONY: test

test: $(OBJS)
	$(LD) $(LDFLAGS) -o test $(OBJS)

%.o : %.cxx
	$(CC) $(CFLAGS) -c $< -o $*.o
.cpp.o :
	$(NVCC) $(NVFLAGS) -c $< -o $*.o
%.o : %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $*.o 

clean:
	-rm test $(OBJS)
