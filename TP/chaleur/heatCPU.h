#ifndef __HEATCPU_H
#define __HEATCPU_H

#define MAX(x, y) ((x) > (y) ? (x) : (y))

#define getElem(A, n, i, j, k) A[(i)*n.z*n.y+(j)*n.z+(k)]

void heatCPU(const float *uIn, float * uOut, const dim3 n, const float dt);
void putValOnBoundary(float * u, const dim3 n, const float val);
float computeError(const float * u1, const float * u2, const dim3 n);
void printTensor(const float * u, const dim3 n);
#endif
