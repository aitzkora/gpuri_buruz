#include <stdlib.h>
#include <math.h>
#define N 8192

int main()
{

  int i;
  float alpha = 2.0f;
  float * x   = (float *) malloc (N * sizeof (float));
  float * y   = (float *) malloc (N * sizeof (float));

  for (i = 0; i < N; i ++ ) {
    x[i]   = 1.0f; y[i] = 2.0f;
  }

  #pragma omp target map(tofrom:y[:N]) map(to:x[:N])
  {
  #pragma omp loop
  for(i = 0; i < N; i ++ )
    y[i] = y[i] + alpha * x[i];
  }  

  for(i = 0; i < N; i ++ )
    if (fabsf(y[i] - 4.0) > 1e-6) exit (-1);
  
  free(x); free(y);
  return 0;
}
