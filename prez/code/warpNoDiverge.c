__global__ kernel(float * x)
{
 int i = blockDim.x * blockIdx.x + threadIdx.x;
 if ((i/warpSize) % 2 == 0)
   x[i] = 0.
 else
   x[i] = 1.
}
