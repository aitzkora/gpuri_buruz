#include <cstdio>
#include "heatCPU.h"
#include "heatGPU.h"
#include "output.h"
#include "helper_cuda.h"

int main(int argc, char * argv[])
{
   dim3 grid, tBlock;
   dim3 n;
   float err;
   float * uIn, 
         * uOut, 
         * uCheck;


   float * uInDevice, 
         * uOutDevice;

   size_t size_tot;

   float dt = 1;

   if (argc < 2)
   {
     n = { 8, 8, 8 };
   }
   else
   {
     unsigned int nfix;
     nfix = atoi(argv[1]);
     n = {nfix, nfix, nfix};
   }

   size_tot = sizeof(float) * n.x * n.y * n.z;

   uIn    = (float *) malloc(size_tot);
   uOut   = (float *) malloc(size_tot);
   uCheck = (float *) malloc(size_tot);

   checkCudaErrors(cudaMalloc(&uInDevice, size_tot));
   checkCudaErrors(cudaMalloc(&uOutDevice, size_tot));

   putValOnBoundary(uIn, n, 1.0);
   putValOnBoundary(uOut, n, 1.0);
   putValOnBoundary(uCheck, n, 1.0);

   checkCudaErrors(cudaMemcpy(uInDevice, uIn, size_tot, cudaMemcpyHostToDevice));
  
   tBlock = { 8, 8, 8 };

   grid.x = n.x / tBlock.x + (n.x % tBlock.x ? 1 : 0 ) ;
   grid.y = n.y / tBlock.y + (n.y % tBlock.y ? 1 : 0 ) ;
   grid.z = n.z / tBlock.z + (n.z % tBlock.z ? 1 : 0 ) ;


   printf("n : %d %d %d\n", n.x, n.y, n.z);
   printf("grid : %d %d %d\n", grid.x, grid.y, grid.z);
   printf("block : %d %d %d\n", tBlock.x, tBlock.y, tBlock.z);
   heatGPU<<<grid, tBlock>>>(uInDevice, uOutDevice, n, dt);
   checkCudaErrors(cudaGetLastError());
   checkCudaErrors(cudaDeviceSynchronize());
   checkCudaErrors(cudaMemcpy(uCheck, uOutDevice, size_tot, cudaMemcpyDeviceToHost));
   heatCPU(uIn, uOut, n, dt);
   err = computeError(uCheck, uOut, n);
   if (err > 1e-12) 
   {
      printf("******%s******\n", "Program failed");
   }
   else
   {
     printf("err = %.3e\n", err);
     if (n.x <= 16)
     {
       printf("uCheck = \n");
       printTensor(uCheck, n);
       printf("\n uOut = \n");
       printTensor(uOut, n);
     }
      
   }
   free(uIn);
   free(uOut);
   free(uCheck);

   checkCudaErrors(cudaFree(uInDevice));
   checkCudaErrors(cudaFree(uOutDevice));
   return 0;
}
