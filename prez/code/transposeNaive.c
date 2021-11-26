#define getElem(m, ny, i, j)  m[(i)* ny + (j)]

__global__ void transposeNaive(const float * mIn, scalar * mOut, dim3 dims)
{
   int x = blockDim.x * blockIdx.x+ threadIdx.x;
   int y = blockDim.y * blockIdx.y+ threadIdx.y;
   int nx = dims.x;
   int ny = dims.y;
     getElem(mOut, nx, y, x) = getElem(mIn, ny, x, y);
}
