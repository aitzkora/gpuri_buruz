#include <stdlib.h>
#include <math.h>
#include <cublas.h>
#define N 8192

int main()
{

  int i;
  float alpha = 2.0f;
  float * x[N], y[N];
  float * xd, *yd;

  cublasInit();
  cublasSetVector(N,  sizeof(x[0]), x, 1, xd, 1);
  cublasSetVector(N,  sizeof(y[0]), y, 1, yd, 1);

  cublasSaxpy(N, alpha, xd, 1, yd, 1);
  
  cublasGetVector(N,  sizeof(y[0]), yd, 1, y, 1);
  cublasShutdown();

  for(i = 0; i < N; i ++ )
    if (fabsf(y[i] - 4.0) > 1e-6) exit (-1);
  
  return 0;
}
