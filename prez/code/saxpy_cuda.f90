module saxpy_cuda
contains 
   attributes(global) subroutine saxpy(x, y, alpha)
     real(4), intent(in)    :: x(:)
     real(4), intent(inout) :: y(:)
     real(4), value :: alpha
     integer :: i
     i = threadIdx%x + (blockIdx%x-1)  * blockDim%x
     y(i) = y(i) + alpha * x(i)
   end subroutine saxpy

end module saxpy_cuda
program saxpy
  use saxpy_cuda
  use cudafor
  implicit none
  integer , parameter :: N = 8192
  integer ::i
  real(4), device :: x(N)   
  real(4), device :: y(N)   
  real(4) :: y_c(N)   

  x = 1.0
  y = 2.0

  call saxpy<<<N/512, 512 >>>(x, y, 2.0)

  y_c = y
  if (any(y_c /= 4.0)) stop -1
 
end program saxpy

