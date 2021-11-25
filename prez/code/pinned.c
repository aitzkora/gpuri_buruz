int main()
{
   float * pinnedTab;
   float * tabDevice;
   size_t sizeTot = N * sizeof(scalar);

   cudaHostAlloc(&pinnedTab, sizeTot, cudaHostAllocDefault);
   // on peut aussi utiliser 
   cudaMalloc(&tabDevice, sizeTot);
  
   cudaMemcpy(tabDevice, pinnedTab, sizeTot,  cudaMemcpyHostToDevice);

   cudaFree(tabDevice);
   cudaFreeHost(pinnedTab);
}
