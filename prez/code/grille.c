// bloc 2D : 512 threads
dim3 db(32, 16); 
// grille de 512 blocs
dim3 dg(512);
// 512 * 512 = 262.144 threads!
myKernel<<<dg,db>>>(...);
