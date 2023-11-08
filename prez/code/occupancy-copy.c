#define  ILP  4 
#define  N  1024*1024

#include <math.h>
#include <algorithm> 
#include <cuda_runtime.h> 
#include <stdio.h>

__global__ void copy(float * odata, float * idata) {
     float tmp;
     int  i;
     i = blockIdx.x * blockDim.x + threadIdx.x;
     tmp = idata[i];
     odata[i] = tmp;
}

__global__ void copy_ILP(float * odata, float * idata) {
    float tmp[ILP];
    int i, j;
    i = blockIdx.x * blockDim.x * ILP + threadIdx.x;
    for(j=0; j < ILP; j++ ) {
      tmp[j] = idata[i+j * blockDim.x];
    }
    for(j=0; j < ILP; j++ ) {
      odata[i+j * blockDim.x] = tmp[j];
    }
}

int main(int argc, char *argv[])
{
   int blockSize, i, numKer, smBytes, j;
   cudaDeviceProp  prop;
   cudaEvent_t  startEvent, stopEvent;
   const size_t sizeTot = sizeof(float) * N;
   float  time;
   float * x, * y , *x_d, * y_d ;
   if (argc > 1) 
     numKer = atoi(argv[1]);
   else
     numKer = 1;
   cudaGetDeviceProperties(&prop, 0);
   smBytes = prop.sharedMemPerBlock;
   x = (float *) malloc(sizeTot);
   y = (float *) malloc(sizeTot);
 
   cudaMalloc(&x_d, sizeTot);
   cudaMalloc(&y_d, sizeTot);

   for(i = 0; i < N; i++)
      x[i] = 1.0 * i;

   cudaMemcpy(x_d, x, sizeTot, cudaMemcpyHostToDevice);
   cudaEventCreate(&startEvent);
   cudaEventCreate(&stopEvent);
   
   for (j = 32; j < prop.maxThreadsPerBlock; j+= 32) {
     if (N % j == 0 ) {
     cudaEventRecord(startEvent, 0);
     blockSize = j;
     if (numKer == 1)  
       copy<<<N/blockSize, blockSize, 0.9 * smBytes>>>(y_d, x_d);
     else
       copy_ILP<<<N/(blockSize*ILP), blockSize, 0.9 * smBytes>>>(y_d, x_d);
     cudaEventRecord(stopEvent, 0);
     cudaEventSynchronize(stopEvent);
     cudaEventElapsedTime(&time, startEvent, stopEvent);
     cudaMemcpy(y, y_d, sizeTot, cudaMemcpyDeviceToHost);

     float maxVal = 0.0;
     for(i = 0; i < N ; ++i)
       maxVal = std::max(fabsf(y[i]-x[i]), maxVal);
     if ( maxVal < 1e-6 ) 
       printf("%5d %.3f\n", j, 2. * sizeTot*1.0e-6 / time);
     else 
       printf("****** test failed ******");
     }
  }
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
  
  cudaFree(x_d);
  cudaFree(y_d);
  free(x);
  free(y);
}
