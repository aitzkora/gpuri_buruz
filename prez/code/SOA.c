#define N 1024
struct Elements {
float * x, *y, *z;
float * dx, *dy, *dz;
};

Elements elems;
elems.x = (float *) = malloc(N*sizeof(float));
...
