program incr
  implicit none
  integer, parameter :: N = 256
  integer :: tab(N), i

  tab = 1 
  call increment(tab, 3)
  if (any(tab(:) /= 4)) stop "error"
contains
  subroutine increment(u, val)
   integer, intent(inout) :: u(:)
   integer, intent(in) :: val
   u(:) = u(:) + val
  end subroutine increment
end program incr
