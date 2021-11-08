#define N 713
#define BLOCK_SIZE 256

__global__ void kernel(int * a) 
{ 
  int  i = threadIdx.x + blockDim.x * blockIdx.x; 
  if (i < N) a[i] ^= 0xFF; 
}

int main()
{
  ...
  kernel<<<(N-1)/BLOCK_SIZE + 1, BLOCK_SIZE>>>(tabGPU);
  ...
}
