using CUDA
using BenchmarkTools
using Printf 

const BLOCK_X = 32
const BLOCK_Y = 16

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

function chrono_gpu!(a, ap, aff)
  init_sol!(a);
  init_sol!(ap);
  for i = 1:100
     @cuda threads=(BLOCK_X,BLOCK_Y) blocks=(cld(N,BLOCK_X), cld(N, BLOCK_Y)) jacobi_gpu!(ap,a)
     error = reduce(max,abs.(ap-a))   
     if (aff != 0) && (i % aff) == 0
          println("i =", i, " error = ", error)
     end 
     if (error<=1e-3) 
       break
     end 
     copyto!(a, ap)
  end
end

#t1 = @benchmark chrono_gpu!($a, $ap, 0)
#println(@sprintf "gpu naive %3.3f ms" mean(t1.times)/10^6)
t1 = @elapsed chrono_gpu!(a, ap, 0)
println(@sprintf "gpu naive %3.3f ms" t1)

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
     error = reduce(max, abs.(a-ap))   
     if (aff != 0) &&  (i % aff) == 0
       println("i =", i, " error = ",error)
     end 
     if (error<=1e-3) 
       break
     end 
     copyto!(a, ap)
  end
end

t2 = @elapsed chrono_shared!(a, ap, 0)
println(@sprintf "gpu shared %3.3f ms" t2)
#t2 = @benchmark chrono_shared!($a, $ap, 0)
#println(@sprintf "gpu shared %3.3f ms" mean(t2.times) /10^6)
