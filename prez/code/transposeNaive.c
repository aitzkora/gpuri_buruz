__global__ void transpose(const float * mIn, float * mOut, dim3 n)
{
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   int j = blockDim.y * blockIdx.y + threadIdx.y;
   mOut[i * n.x + j]  = mIn[j * n.y + i];
}
