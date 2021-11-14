...
  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent, 0);
  cudaEventCreate(&stopEvent, 0);

  float time;
  cudaEventRecord (startEvent, 0);
  myKernel<<<(N-1)/blockSize + 1, blockSize>>>(array, N);
  cudaEventRecord (stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&time, startEvent, stopEvent);
  printf("time : %04.4fms\n", time);
 
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
...
