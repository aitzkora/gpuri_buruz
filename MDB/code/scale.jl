using CUDA
using BenchmarkTools
using Printf
using Test

# noyau homothetie a ← α ⋅ a
function scale_gpu!(a, α)
  i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
  if (i <= size(a, 1))  # garde-fou
    a[i] *= α 
  end      
  return # ne pas oublier !
end    

# appel du noyau
x = CUDA.ones(4096)
@cuda threads=512 blocks=cld(4096, 512) scale_gpu!(x, 4.0f0)

# vérification
@test sum((x .- 4.0f0).^2) < 1e-12

# perfs
t = @benchmark @cuda threads=512 blocks=cld(4096, 512) scale_gpu!($x, 0.001f0) # ne pas oublier d'interpoler x !
println(@sprintf "tps noyau = %3.5fμs" mean(t.times))
