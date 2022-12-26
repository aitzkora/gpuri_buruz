#include <stdio.h>
#include <stdlib.h>
#define N (2<<22)
int main()
{
  float s = 0;
  int * t = 
    (int *) malloc(N*sizeof(int)); // alloc ?
  t[1] = 1;
  for(int i = 0; i < N ; ++i) 
    s+= t[i];
  printf("%f", s);
  free(t);
  return 0;
}
