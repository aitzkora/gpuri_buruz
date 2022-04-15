__global__ void scale(float * vec, float k, int size)
{
  int i  = blockDim.x * blockIdx.x + threadIdx.x;
  vec[i] *= k;
}
