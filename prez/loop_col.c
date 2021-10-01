#include <stdio.h>
#include <math.h>
#define N 10000
int main() { double matA[N][N]; 
 double s= 0;
 for (int j=0; j < N ; ++j) 
   for(int i=0; i < N ; ++i) 
   matA[i][j] = sin(i*M_PI/N)
               +cos(j*M_PI/N);
 for (int j=0; j < N ; ++j) 
   for(int i=0; i < N ; ++i) 
   s += fabs(matA[i][j] - 1.);
 printf("%f", s);
 return 0;
}
