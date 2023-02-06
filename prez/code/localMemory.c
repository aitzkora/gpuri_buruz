__global__ void k1(float * a)
{
  float b[2];
  int i  = blockDim.x * blockIdx.x + threadIdx.x;
  b[0] = 1.;
  b[1] = 2.;
  a[i] = b[2];
}

__global__ void k2(float * a, int j, int k)
{
  float b[2];
  int i  = blockDim.x * blockIdx.x + threadIdx.x;
  b[j] = 1.;
  a[i] = b[k];
}

__global__ void k3(float * a)
{
  float b[512];
  int i  = blockDim.x * blockIdx.x + threadIdx.x;
  for(int w = 0; w < 512;  ++w) b[w] = 1.0;
  a[i] = b[8];
}
