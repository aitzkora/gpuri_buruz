__global__ void scaleFlip(float * vec, float k, int size)
{
    int i  = size - 1 - (blockDim.x * blockIdx.x + threadIdx.x);
      vec[i] *=k;
}
