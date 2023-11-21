program alloc
  integer, parameter :: N = ishft(2,22)
  call compute_sum()
contains
  subroutine compute_sum()
    real :: t(N)
    t(1:2) = [0,1]
    print *, sum(t)
  end subroutine compute_sum 
end program alloc
