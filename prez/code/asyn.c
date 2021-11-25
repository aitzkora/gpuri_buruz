__global__ void kernel(float * a , int offset)
{
  float x, c, s;
  int i = offset + threadIdx.x+ blockIdx.x * blockDim.x;
  x = 1.0 * i; s = sin(x); c = cos(x);
  a[i] += sqrt(s*s + c*c);
}
int main()
{
  ...
  // sequential
  cudaMemcpy(aDevice, aPinned, sizeTot, cudaMemcpyHostToDevice );
  kernel<<<n/blockSize, blockSize>>>(aDevice, 0);
  cudaMemcpy(aPinned, aDevice, sizeTot, cudaMemcpyDeviceToHost );
  ... 
  // Asynchronous
  for(int s = 0 ; s < nStreams; ++s) 
  {
    int offset = s * streamSize;
    cudaMemcpyAsync(aDevice + offset, aPinned + offset, streamSize * sizeof(float), cudaMemcpyHostToDevice, streams[s]);
    kernel<<<streamSize/blockSize, blockSize, 0, streams[s]>>>(aDevice, offset);
    cudaMemcpyAsync(aPinned + offset, aDevice + offset, streamSize * sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
  }
  ...
}

