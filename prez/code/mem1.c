#include <stdio.h>
#define N (2<<22)
int main()
{
  int t[N]={0, 1}; // alloc ?
  float s = 0;
  for(int i = 0; i < N ; ++i) 
    s+= t[i];
  printf("%f", s);
  return 0;
}
