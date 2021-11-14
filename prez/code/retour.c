#define N 1024*1024
int * pInt = (int *)malloc(sizeof(int) * N);
if (pInt == 0)
{
  fprintf(stderr, "cannot allocate pInt\n");
  exit (-1);
}
...
