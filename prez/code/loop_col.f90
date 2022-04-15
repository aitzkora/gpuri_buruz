program loop_col
 integer, parameter :: N = 10000
 real(kind=8) ::  matA(N, N)
 real(kind=8) :: s, pi = 4.d0 * atan(1.d0)
 integer :: i, j 
 
 s = 0.d0
 do j= 1, N
   do i=1, N
     matA(i, j) =  sin(i*pi/N) + cos(j*pi/N)
   end do
 end do 
 do j=1, N
   do i=1, N
     s = s + abs(matA(i, j) - 1.d0)
   end do
 end do 
 print '(f12.2)', s
end program loop_col
