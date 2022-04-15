integer, parameter :: N = 1024
type Elements
real(kind=4) ::  x(:), y(:), z(:)
real(kind=4) :: dx(:), dy(:), dz(:)
end type Elements

type(Elements) :: elems
allocate(elems%x(N))
...
