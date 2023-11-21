program valeur
  integer :: x 
  x = 2
  call f(x)
  print *, "x=", x
contains 
  subroutine f(p) 
    integer, value :: p
    p = p+1;
    print *, "p=", p
  end subroutine f
end program valeur
