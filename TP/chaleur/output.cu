#include <stdio.h>
#include "heatCPU.h"

void saveVTK(const char * filename, float * u, const dim3 size)
{
  FILE * fid;
  fid = fopen(filename, "w");
  fprintf(fid,  "# vtk DataFile Version 3.0\n");
  fprintf(fid,  "cell\n");
  fprintf(fid,  "ASCII\n");
  fprintf(fid,  "DATASET STRUCTURED_POINTS\n");
  fprintf(fid,  "DIMENSIONS %d %d %d\n", size.x, size.y, size.z);
  fprintf(fid,  "ORIGIN %d %d %d\n", 0, 0, 0);
  fprintf(fid,  "SPACING %f %f %f\n", 1./size.x, 1./size.y, 1./size.z);
  fprintf(fid,  "POINT_DATA %d\n", size.x * size.y * size.z);
  fprintf(fid,  "SCALARS cell float\n");
  fprintf(fid,  "LOOKUP_TABLE default\n");
  for(int i = 0; i < size.x ; ++i) 
    for(int j = 0; j < size.y ; ++j) 
      for(int k = 0; k < size.z ; ++k) 
         fprintf(fid, "%f ", getElem(u, size, i, j, k));
  fclose(fid);
}
