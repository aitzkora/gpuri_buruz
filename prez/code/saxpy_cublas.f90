program saxpy_cublas
  use cublas
  use iso_c_binding
  implicit none
  integer, parameter :: N = 8192
  type(cublasHandle) :: handle
  real(c_float), allocatable :: x(:) 
  real(c_float), allocatable :: y(:)   
  real(c_float), device :: xd(N), yd(N)
  real(c_float), device :: alpha = 2.0
  integer :: ierr, elemsize
  allocate(x(N), y(N))

  x = 1.0
  y = 2.0
  elemsize = sizeof(x(1))
  ierr = cublasCreate(handle)
  ierr = cublasSetVector(N, elemsize, x, 1, xd, 1)
  ierr = cublasSetVector(N, elemsize, y, 1, yd, 1)
  call cublasSaxpy(N, 2.0, xd, 1, yd, 1)
  ierr = cublasGetVector(N, elemsize, yd, 1, y, 1)
  ierr = cublasDestroy(handle)
  if (any(y /= 4.0)) stop -1

  deallocate(x, y)

end program saxpy_cublas
