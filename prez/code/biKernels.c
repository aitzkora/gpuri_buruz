partialSum<<<256,512,512*sizeof(int)>>>(xyDevice, partial, N);
finalSum<<<1,256,256*sizeof(int)>>>(partial, interiorGPU);
