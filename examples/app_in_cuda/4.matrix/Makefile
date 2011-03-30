OPT=-O3 -g -arch=sm_20
NVCC=nvcc

.PHONY: test

test:
	$(NVCC) $(OPT) *.cu -o test

clean:
	-rm test
