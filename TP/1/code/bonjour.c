#include <cstdio>
__global__ void bonjour()
{
  printf("bonjour de la part de %d sur le GPU!\n", threadIdx.x);
}
int main()
{
  bonjour<<<1, 8>>>();
  cudaDeviceSynchronize();
  return 0;
}

