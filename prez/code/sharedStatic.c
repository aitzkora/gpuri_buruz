__global__ void kernelShared(float * a)
{
  ...
  __shared__ int tile[GPU_TILE];
  ...
}
kernelShared<<<grid, tBlock>>>(aDevice);
