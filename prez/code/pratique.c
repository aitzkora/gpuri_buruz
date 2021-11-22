__global__ void base(float * a, float *b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] = exp(1.+sin(b[i]));
}

__global__ void memory(float * a, float *b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] = b[i];
}

__global__ void math(float * a, float b, int flag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = exp(1.+sin(b));
      if (v*flag == 1) a[i] = v;
}
