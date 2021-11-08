#include <vector>
#include <cassert>
#define N 713
#define BLOCK_SIZE 256

__global__ void kernelXOR(int * a) 
{ 
  int  currentIndex = threadIdx.x + blockDim.x * blockIdx.x; 
  if (currentIndex < N) 
    a[currentIndex] ^= 0xFF;
}

int main()
{
  std::vector<int> tab(N, 1);
  int * tabGPU;
  cudaMalloc(&tabGPU, N * sizeof(int));
  cudaMemcpy(tabGPU, tab.data(), N * sizeof(int), cudaMemcpyHostToDevice);
  kernelXOR<<<(N-1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(tabGPU);
  cudaMemcpy(tab.data(), tabGPU, N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(tabGPU);
  for(int i=0; i < N; ++i) 
    assert (tab[i] == 0xFE) ;
  return 0;
}
