#include <stdio.h>
#include <math.h>
#define N 10000
double matA[N][N]; 
int main() { 
 double s = 0;
   for(int i=0; i < N ; ++i) 
 for (int j=0; j < N ; ++j) 
   matA[i][j] = sin(i*M_PI/N)
               +cos(j*M_PI/N);
   for(int i=0; i < N ; ++i) 
 for (int j=0; j < N ; ++j) 
   s += fabs(matA[i][j] - 1.);
 printf("%f", s);
 return 0;
}
