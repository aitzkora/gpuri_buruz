program adresse
  integer :: x 
  x = 2
  call f(x)
  print *, "x=", x
contains 
  subroutine f(p) 
    integer :: p
    p = p+1;
    print *, "p=", p
  end subroutine f
end program adresse
