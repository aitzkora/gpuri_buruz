.SUFFIXES: .cu .x

SM_NUM ?= 35
CUDA_PATH ?= /opt/cuda

CUDA=nvcc
CFLAGS= -I$(CUDA_PATH)/samples/common/inc -arch=sm_$(SM_NUM) -Wno-deprecated-gpu-targets  #--ptxas-options=-v
LDFLAGS=-L$(CUDA_PATH)/lib64 -lcurand
all: monteCarlo.x

.cu.x: 
	$(CUDA) $(CFLAGS) $(LDFLAGS) -o $@ $< 

clean: 
	rm -fr *.x
