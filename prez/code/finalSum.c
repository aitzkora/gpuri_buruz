__global__ void finalSum(int * partial, int * total)
{
  extern __shared__ int psum[];
  int i = threadIdx.x;
  psum[i] = partial[i];
  __syncthreads();
  int  iNext = blockDim.x/2;
  while (iNext > 0)
  {
    if (i < iNext)
      psum[i] += psum[i+iNext];
    iNext >>= 1;
    __syncthreads();
  }
  if (i == 0) *total = psum[0];
}

