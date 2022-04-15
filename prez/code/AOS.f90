integer, parameter ::  N = 1024

type(Element)
   real(kind=4) :: x, y, z
   real(kind=4) :: dx, dy, dz
end type Element

type(Element), allocatable :: elements(:)
allocate(elements(N))
...
