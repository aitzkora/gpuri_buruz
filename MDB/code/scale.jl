using CUDA
using BenchmarkTools
# exemple noyau d'homothetie a ← α ⋅ a
function scale_gpu!(a, α)
  i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
  if (i <= size(a, 1))  
    a[i] *= α 
  end      
  return
end    

#appel du noyau
x= CUDA.ones(4096)
t = @btime @cuda threads=512 blocks=cld(4096, 512) scale_gpu!(x, 4.0f0)
println(t)
#@test sum((x .- 4.0f0).^2) < 1e-12
