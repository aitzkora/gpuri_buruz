__global__ void transposeShared(const float *mIn, float *mOut, dim3 n)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int i = blockIdx.x * TILE_DIM + threadIdx.x;
    int j = blockIdx.y * TILE_DIM + threadIdx.y;

    tile[threadIdx.x][threadIdx.y] = mIn[ j * n.y + i];

    __syncthreads(); // ne pas oublier
  
    // les ... seront vu en TP ;-)
    mOut[...] = tile[threadIdx.y][threadIdx.x]; 
}
