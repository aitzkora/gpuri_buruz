#ifndef __HEATGPU_H
#define __HEATGPU_H

#include "heatCPU.h"

__global__ void heatGPU(const float * uIn, float * uOut, const dim3 n, const float dt);

#endif
