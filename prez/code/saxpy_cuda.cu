#include <stdlib.h> 
#include <math.h>
#define N 8192

__global__ void  saxpy(float *x, float * y, float alpha){
  int  i= threadIdx.x + blockIdx.x * blockDim.x;
  y[i] = y[i] + alpha * x[i];
}

int main()  {
  int i;
  size_t sizeTot = N * sizeof(float);
  float * x = (float *) malloc(sizeTot);
  float * y = (float *) malloc(sizeTot);
  float * xd, *yd;

  for(i = 0; i < N ; i ++) { x[i] = 1.0; y[i] = 2.0; }

  cudaMalloc(&xd, sizeTot); cudaMalloc(&yd, sizeTot);
  cudaMemcpy(xd, x, sizeTot, cudaMemcpyHostToDevice);
  cudaMemcpy(yd, y, sizeTot, cudaMemcpyHostToDevice);
  saxpy<<<N/512, 512 >>>(xd, yd, 2.0);
  cudaMemcpy(y, yd, sizeTot, cudaMemcpyDeviceToHost);
  cudaFree(x); cudaFree(y);

  for(i = 0; i < N ; i ++) 
    if (fabs(x[i] - 4.0) > 1e-6) 
       exit(-1);

  return 0;
}
