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
   integer :: i
   do i=1, size(u, 1) ! modern => write u(:) = u(:) + val w/o loop
     u(i) = u(i) + val 
   end do
  end subroutine increment
end program incr
