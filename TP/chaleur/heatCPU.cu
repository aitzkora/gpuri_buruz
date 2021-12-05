#include "heatCPU.h"
#include <stdio.h>

void heatCPU(const float *uIn, float * uOut, const dim3 n, const float dt)
{
    float hx = 1./n.x;
    float hy = 1./n.y;
    float hz = 1./n.z;

    for(int i=1; i < n.x - 1; ++i)
      for(int j=1; j < n.y - 1; ++j)
        for(int k=1; k < n.z - 1; ++k)
          getElem(uOut, n, i, j, k) = getElem(uIn, n, i, j, k) + dt * (
                                      (getElem(uIn, n, i-1, j, k) + 
                                       getElem(uIn, n, i+1, j, k) ) / (hx * hx) +
                                      (getElem(uIn, n, i, j-1, k) + 
                                       getElem(uIn, n, i, j+1, k) ) / (hy * hy) +
                                      (getElem(uIn, n, i, j, k-1) + 
                                       getElem(uIn, n, i, j, k+1) ) / (hz * hz) 
                                      -getElem(uIn, n, i, j, k) * 2.0 / (hx *hx)
                                      -getElem(uIn, n, i, j, k) * 2.0 / (hy *hy)
                                      -getElem(uIn, n, i, j, k) * 2.0 / (hz *hz));

}

void putValOnBoundary(float * u, const dim3 n, const float val)
{    
  // first boundary
  for(int j = 0; j < n.y ; ++j) {
    for (int k = 0; k < n.z ; ++ k) {
      getElem(u, n, 0, j, k) = val;
      getElem(u, n, n.x-1, j, k) = val;
    }
  }
  // second boundary
  for(int i = 0; i < n.x; ++i) {
    for (int k = 0; k < n.z ; ++k) {
      getElem(u, n, i, 0, k) = val;
      getElem(u, n, i, n.y-1, k) = val;
    }
  }
  // third boundary
  for(int i = 0; i < n.x; ++i) {
    for (int j = 0; j < n.y; ++j) {
      getElem(u, n, i, j, 0) = val;
      getElem(u, n, i, j, n.z-1) = val;
    }
  }
}

float computeError(const float * u1, const float * u2, const dim3 n)
{
  float err = 0;
  for (int i = 1; i < n.x - 1; ++i)
    for (int j = 1; j < n.y - 1; ++j)
      for (int k = 1; k < n.z - 1; ++k)
        err += abs(getElem(u1, n, i, j, k) - getElem(u2, n, i, j, k));
  return err;
}

void printTensor(const float * u, const dim3 n)
{
  for (int i = 1; i < n.x - 1; ++i) {
	for (int j = 1; j < n.y - 1; ++j) {
	  for (int k = 1; k < n.z - 1; ++k) {
		printf("%08.2f ", getElem(u, n, i, j, k));
	  }
	  printf("\n");
	}
	printf("\n\n");
  }
}
