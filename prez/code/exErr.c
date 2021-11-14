CudaSafeCall( cudaMemcpy(&sDevice, s, sizeS, cudaMemcpyHostToDevice ));
mykernel<<<1, N>>>(s);
CudaCheckError();
