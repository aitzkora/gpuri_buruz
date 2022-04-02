type(dim3) :: db(32, 16) ! bloc 2D : 512 threads
type(dim3) dg(512) ! grille de 512 blocs
call myKernel<<<dg,db>>>(...) ! 512 * 512 = 262.144 threads!
