program saxpy
  implicit none
  integer, parameter :: N = 8192
  integer :: i
  real(4), allocatable :: x(:)   
  real(4), allocatable :: y(:)   

  allocate(x(N), y(N))

  x = 1.0
  y = 2.0

  !$omp target map(tofrom:y(1:N)) map(to:x(1:N))
  !$omp loop
  do i=1, N
    y(i) = y(i) + 2.0 * x(i)
  end do
  !$omp end loop
  !$omp end target
  if (any(y /= 4.0)) stop -1

  deallocate(x, y)

end program saxpy
