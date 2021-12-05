.SUFFIXES: .cu .x

SM_NUM ?= 35
CUDA_PATH ?= /opt/cuda

CUDA=nvcc
CFLAGS= -I$(CUDA_PATH)/samples/common/inc -arch=sm_$(SM_NUM) -Wno-deprecated-gpu-targets 

all:: heatGPU.x heatCPU.x heatTest.x

heatGPU.x: mainGPU.cu heatGPU.o heatCPU.o output.o
	$(CUDA) $(CFLAGS) -o $@ mainGPU.cu heatGPU.o heatCPU.o output.o

heatCPU.x: mainCPU.cu heatCPU.o output.o
	$(CUDA) $(CFLAGS) -o $@ mainCPU.cu  heatCPU.o output.o

heatTest.x: heatTest.cu heatGPU.o heatCPU.o 
	$(CUDA) $(CFLAGS) -o $@ heatTest.cu heatGPU.o heatCPU.o 

output.o : output.cu output.h
	$(CUDA) $(CFLAGS) -c $<

heatGPU.o : heatGPU.cu heatGPU.h heatCPU.h
	$(CUDA) $(CFLAGS) -c $<

heatCPU.o : heatCPU.cu heatCPU.h
	$(CUDA) $(CFLAGS) -c $<

clean:
	@echo  Cleaning...
	rm -f *.o *.mod 
clean_vtk: 
	@echo  Cleaning vtk files
	rm -fr test*.vtk 
cleanall: clean clean_vtk
	@echo  Cleaning executables and img files
	rm -fr *.vtk *.x
