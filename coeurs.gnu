set term eps
set output "nb_coeurs_gpu.eps"
set label 'Tesla C1060' at 2006, 500
set label 'Fermi 2070' at 2010, 451
set label 'Kepler K40'  at 2010, 2880
set label 'Maxwell K40' at 2012, 3272
set label 'Pascal P100' at 2014, 3840
set label 'Volta V100' at 2015, 5120
set label 'Turing T4' at 2016, 2560
set ylabel '# cœurs'
set xlabel 'année de sortie'
set key left
p "coeurs.dat" u 1:2  w lp title "# cœurs"
