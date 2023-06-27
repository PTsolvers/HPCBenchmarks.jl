N         = 4096
C_SAMPLES = 500

function diffusion_kernel!(A_new,A,h)
    ix = (blockIdx().x-1i32)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1i32)*blockDim().y + threadIdx().y
    if ix ∈ axes(A_new,1) && iy ∈ axes(A_new,2)
        @inbounds A_new[ix,iy] = A[ix+1,iy+1] + h*(A[ix,iy+1] + A[ix+2,iy+1] + A[ix+1,iy] + A[ix+1,iy+2] - 4.0*A[ix+1,iy+1])
    end
    return
end

function run_julia_benchmarks(n)
    A_new = CuArray{Float64}(undef,n-2,n-2)
    A     = CuArray{Float64}(undef,n  ,n  )
    h     = 1/5
    nthreads = (32,8)
    nblocks  = cld.(size(A_new),nthreads)

    bm = @benchmark begin
        CUDA.@sync @cuda blocks=$nblocks threads=$nthreads diffusion_kernel!($A_new,$A,$h)
    end

    CUDA.unsafe_free!(A_new)
    CUDA.unsafe_free!(A)

    return bm
end

function run_c_benchmarks(lib,nsamples,n)
    trial = make_c_trial(nsamples)

    CUDA.reclaim()

    sym = Libdl.dlsym(lib,:run_benchmark)
    @ccall $sym(trial.times::Ptr{Cdouble},nsamples::Cint,n::Cint)::Cvoid

    return trial
end

group = BenchmarkGroup()
group["julia"] = run_julia_benchmarks(N)

# Add baseline C benchmark
libext  = Sys.iswindows() ? "dll" : "so"
libname = "diffusion_2d." * libext
run(`nvcc -O3 -o $libname --shared diffusion_2d.cu`)
Libdl.dlopen("./$libname") do lib
    group["reference"] = run_c_benchmarks(lib,C_SAMPLES,N)
end

RESULTS["diffusion-2d"] = group

