#include <stdio.h>
#include "heatCPU.h"
#include "output.h"

int main(int argc, char *argv[])
{
   float err, hMax;
   float * uIn, 
         * uOut, 
         * uTmp;

   float T=1;
   float dt;
   int it, itMax;
   size_t sizeTot;
   dim3 n ;

   if (argc < 2)
   {
     n = { 8, 8, 8 };
   }
   if (argc >=2)
   {
     unsigned int nfix;
     nfix = atoi(argv[1]);
     n = {nfix, nfix, nfix};
     itMax = 100;
   }
   if (argc >=3)
   {
     itMax = atoi(argv[2]);
   }
   
   sizeTot = sizeof(float) * n.x * n.y * n.z;

   uIn    = (float *) malloc(sizeTot);
   uOut   = (float *) malloc(sizeTot);

   putValOnBoundary(uIn, n, 1.0);
   putValOnBoundary(uOut, n, 1.0);


   hMax = 1./ (MAX(MAX(n.x, n.y), n.z));
   dt = hMax*hMax / 6.; 
   it = 0;
   for (float t = 0; t < T ; t+=dt, it++)
   {
      heatCPU(uIn, uOut, n, dt);
      if (it % 10 == 0) 
      {
        char name[120];
        err = computeError(uIn, uOut, n);
        snprintf(name, 120, "test_%03d.vtk", it);
        printf("it = %d, %f\n", it, err);
        saveVTK(name, uOut,  n);
      }
      uTmp = uIn;
      uIn= uOut;
      uOut= uTmp;
      if (it > itMax)  break;
        
   }

   free(uIn);
   free(uOut);

}
