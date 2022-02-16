using CUDA
using BenchmarkTools

function jacobi_gpu!(ap, a)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if ((i >= 2) && (i <= (size(a,1)-1)) && (j >= 2) && (j <= (size(a,2)-1)))
      ap[i,j] = 0.2f0 * (a[i-1,j]  + a[i+1,j]   + a[i,j-1]  + a[i,j+1])  +
                0.05f0 * (a[i-1,j-1]+ a[i+1,j-1] + a[i-1,j+1] + a[i+1,j+1])  
    end
    return
end


function init_sol!(a)
    a .= 0.0f0
    m = size(a,1)
    y₀ = sin.(π*[0:m-1;] ./ (m))
    a[:,1] = y₀
    a[:,end]= y₀ * exp(-π)
end    

const N = 2048

function solve_gpu(N, int_max, prec)
  a = CuArray{Float32}(undef, N, N);
  ap = similar(a)
  init_sol!(a);
  init_sol!(ap);
  nThreads = 32
  for i = 1:int_max
     @cuda threads=(nThreads,nThreads) blocks=(cld(N,nThreads), cld(N, nThreads)) jacobi_gpu!(ap,a)
     error = maximum(abs.(ap-a))   
     if (error<=prec) 
       break
     end 
     a[:,:] = ap[:,:]
  end
end

@benchmark solve_gpu(N, 300, 10.f0^(-3))

function jacobi_cpu!(ap, a)
    m,n = size(a)
    for i=2:m-1
        for j=2:n-1
            ap[i,j] = 0.2f0 * (a[i-1,j]  + a[i+1,j]   + a[i,j-1]  + a[i,j+1])  +
                      0.05f0 * (a[i-1,j-1]+ a[i+1,j-1] + a[i-1,j+1] + a[i+1,j+1])  
        end 
    end
    return
end

function solve_cpu(N, int_max, prec::Float32)
  b = Array{Float32}(undef, N,N)
  c = similar(b)
  init_sol!(b)
  init_sol!(c);
  for i = 1:int_max
     jacobi_cpu!(c,b)
     error = maximum(abs.(c-b))   
     if (error<=prec)
       break
     end 
     b[:,:] = c[:,:]
  end
end
 
@benchmark solve_cpu(N, 300, 10.0f0^(-3))
