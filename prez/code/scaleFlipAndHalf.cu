__global__ void scaleFlipAndHalf(float * vec, float k, int size)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if ( i % 2 == 1)
    i = size-i;
  vec[i] *=k;
}
