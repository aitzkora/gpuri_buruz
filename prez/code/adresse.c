#include <stdio.h>

void f(int * p) { 
  ++(*p); 
  printf("p=%d,",*p);
}
int main() { 
  int x= 2; 
  f(&x); 
  printf("x=%d\n",x); 
}
