# verifier la config
using CUDA
CUDA.versioninfo()

#version GPU
using BenchmarkTools
N = 2^10*32
A = CuArray([1:N;])
B = CuArray([0:N-1;])
@benchmark z = reduce(+, A.^3+B.^2-2 * A .* B)

# version CPU
A = [1:N;]
B = [0:N-1;]
@benchmark z = reduce(+, A.^2+B.^2-2 * A .* B)

A = CuArray([1:1000;])
s = 0
#CUDA.allowscalar(false) tweak that!
for i =1:1000
   s += A[i]
end
s

# exemple noyau d'homothetie a ← α ⋅ a
function scale_gpu!(a, α)
  i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
  if (i <= size(a, 1))  
    a[i] *= α 
  end      
  return
end    

    

# appel du noyau
using Test
x= CUDA.ones(4096)
@cuda threads=512 blocks=cld(4096, 512) scale_gpu!(x, 4.0f0)
@test sum((x .- 4.0f0).^2) < 1e-12

CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)

noyau = @cuda launch=false scale_gpu!(x, 4.0f0)

config = CUDA.launch_configuration(noyau.fun)

@show nThreads = min(length(x), config.threads)
@show nBlocks = cld(length(x), nThreads)
x = CUDA.ones(4096)
noyau(x, 4.0f0; threads=nThreads, blocks=nBlocks)
@test sum((x .- 4.0f0).^2) < 1e-12

CUDA.occupancy(noyau.fun, nThreads)

N2 = 32 * 1024 * 1024 
x = CUDA.ones(N2)
@cuda threads=1024 blocks=cld(N, 1024) scale_gpu!(x, 4.0f0)
@test sum((x .- 4.0f0).^2) < 1e-12

x = CUDA.ones(N2)
function scale_gpu2!(a, α)
  i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
  for j=i:gridDim().x * blockDim().x: size(a, 1) # maintenant on gère des données plus grosses que la grille
    a[j] *= α 
  end      
  return
end    
@cuda threads=1024 blocks=cld(N, 1024) scale_gpu2!(x, 4.0f0)
@test sum((x .- 4.0f0).^2) < 1e-12

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

N = 4096
a = CuArray{Float32}(undef, N, N)
ap = similar(a);
const BLOCK_X = 32
const BLOCK_Y = 16

function chrono_gpu!(a, ap, aff)
  init_sol!(a);
  init_sol!(ap);
  nThreads = 32
  for i = 1:100
     @cuda threads=(BLOCK_X,BLOCK_Y) blocks=(cld(N,BLOCK_X), cld(N, BLOCK_Y)) jacobi_gpu!(ap,a)
     error = reduce(max, abs.(ap-a))   
     if (aff != 0 && (i % aff) == 0)
          println("i =", i, " error = ", error)
     end 
     if (error<=1e-3) 
       break
     end 
     a = copy(ap)
  end

end 
chrono_gpu!(a, ap, 20 )

@benchmark chrono_gpu!($a, $ap, 0)

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

function chrono_cpu!(b,c, aff)
  init_sol!(b)
  init_sol!(c);
  for i = 1:100
     jacobi_cpu!(c,b)
     error = maximum(abs.(c-b))   
     if (aff != 0) && (i % aff) == 0
          println("i =", i, " error = ", error)
     end 
     if (error<=1e-3) 
       break
     end 
     b = copy(c)
  end
end


b = Array{Float32}(undef, N,N)
c = similar(b)
chrono_cpu!(b, c, 20)

#@benchmark chrono_cpu!($b, $c, 0)

function jacobi_gpu_shared!(a, ap)
    tile = @cuStaticSharedMem(Float32, (BLOCK_X+2, BLOCK_Y+2))  
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    is = threadIdx().x
    js = threadIdx().y
    nx = size(a, 1) 
    ny = size(a, 2) 
    
    if ( i > 1 && j > 1) 
      tile[is, js] = a[i-1, j-1]
    end    
    if ( i > 1 && j < ny && js > BLOCK_Y-2)  
      tile[is,  js+2] = a[i-1, j+1]
    end 
    if ( j > 1 && i < nx && is > BLOCK_X-2) 
      tile[is+2,  js] = a[i+1, j-1]
    end    
    if ( i < nx && j < ny && is > BLOCK_X-2 && js > BLOCK_Y - 2) 
      tile[is+2,js+2] = a[i+1, j+1]
    end
    
    sync_threads()

    if (i > 1 && i < nx && j > 1 && j < ny) 
      ap[i,j] = 0.2f0  * (tile[is, js+1]  + tile[is+2, js+1]  + 
                              tile[is+1, js]  + tile[is+1, js+2]) + 
                0.05f0 * (tile[is, js] + tile[is, js+2] + 
                              tile[is+2, js] + tile[is+2, js+2]) 
    end
    return
end

function chrono_shared!(a, ap, aff)
  init_sol!(a)
  init_sol!(ap)
  for i = 1:100
     @cuda threads=(BLOCK_X,BLOCK_Y) blocks=(cld(N,BLOCK_X), cld(N, BLOCK_Y)) jacobi_gpu_shared!(a,ap)
     error = reduce(max,abs.(a-ap))   
     if (aff != 0) && (i % aff) == 0
          println("i =", i, " error = ",error)
     end 
     if (error<=1e-3) 
       break
     end 
     a = copy(ap)
  end
end

chrono_shared!(a, ap, 20)

@benchmark chrono_shared!($a, $ap, 0)
