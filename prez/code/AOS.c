#define N 1014
struct Element{
   float x, y, z;
   float dx, dy, dz;
};

Element * elements = (Element *) malloc(N*sizeof(Element));
...
