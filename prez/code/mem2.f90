program alloc 
   integer, parameter  :: N = ishft(2,22)
   real :: t(N)
   t(1:2) = [0,1]
   print *, sum(t)
end program alloc
