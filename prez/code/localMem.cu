__global__ void k(float * a, int j, int k)
{
  float b[2];
  int i  = blockDim.x * blockIdx.x + threadIdx.x;
  b[j] = 1.;
  a[i] = b[k];
}
