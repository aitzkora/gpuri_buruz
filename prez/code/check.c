//...
#include "helper_cuda.h"

int main(int argc, char * argv[])
{
  //...

  checkCudaErrors(cudaMalloc(&tab_d, sizeCpy));
  checkCudaErrors(cudaMemcpy(tab_d, tab, sizeCpy, cudaMemcpyHostToDevice ));
  increment<<<1,N>>>(tab_d, N, 3);
  checkCudaError(cudaMemcpy(tab, tab_d, sizeCpy, cudaMemcpyDeviceToHost ));
  checkCudaError(cudaFree(tab_d));
  //..
  return 0;
}
