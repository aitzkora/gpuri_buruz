#include <stdio.h>
#include <stdlib.h>
#define N 256

void increment(int * u, const int n, int val) {
  for(int i = 0; i < n ; ++i) u[i] += val;
}

int main()
{
  int tab[N];
  for(int i = 0; i < N; ++i)   tab[i] = 1;
  increment(tab, N, 3);
  for (int i = 0; i < N; ++i) { if (i[tab] != 4) abort();}
  return 0;
}
