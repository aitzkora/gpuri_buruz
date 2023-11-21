.SUFFIXES: .cu .x

SM_NUM ?= 35
OACC_PATH ?= /home/fuentes/OpenACC/
CUDA=nvcc 
CFLAGS=-g -G -I$(OACC_PATH)/SDK/include -arch=sm_$(SM_NUM) -Wno-deprecated-gpu-targets 

.cu.x: 
	$(CUDA) $(CFLAGS) -o $@ $<

clean: 
	rm -fr *.x
