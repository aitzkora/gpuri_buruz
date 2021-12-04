__global__ void partialSum(scalar * input, int * partial, const int N) 
{
  extern __shared__ int psum[];
  int idX  = threadIdx.x + blockDim.x * blockIdx.x;
  int interior = 0;
  for (int i = idX; i < N ; i+= gridDim.x * blockDim.x)
    if ((input[i]*input[i]+input[i+N]*input[i+N]) <= 1.0)
       interior++;
  idX = threadIdx.x;
  psum[idX] = interior;
  __syncthreads();
  int iNext = blockDim.x / 2;
  while (iNext > 0)
  {
    if (idX < iNext)
      psum[idX] += psum[idX+iNext];
    iNext >>= 1;
    __syncthreads();
  }
  if (idX == 0) partial[blockIdx.x] = psum[0];
}
