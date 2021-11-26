__global__ void kernelShared(float * a)
{
  ...
  extern __shared__ int tile[];
  ...
}
kernelShared<<<grid, tBlock, sizeof(int) * sizeShared>>>(aDevice);
