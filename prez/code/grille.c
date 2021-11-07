// decoupage des threads
dim3 dg, db;
// bloc 2D 
db.x = 32;
db.y = 16;
db.z = 1;
// grille de 512 blocs
dg.x = 512;
dg.y = 1;
dg.z = 1;
// 262.144 threads!
