INPUTS["diffusion-3d"] = (
    c_samples=2000,
    n_range=2 .^ (8:10),
)

function diffusion_kernel!(A_new, A, h)
    ix = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1i32) * blockDim().y + threadIdx().y
    iz = (blockIdx().z - 1i32) * blockDim().z + threadIdx().z
    if ix ∈ axes(A_new, 1) && iy ∈ axes(A_new, 2) && iz ∈ axes(A_new, 3)
        @inbounds A_new[ix, iy, iz] = A[ix+1, iy+1, iz+1] + h * (A[ix, iy+1, iz+1] + A[ix+2, iy+1, iz+1]
                                                                 + A[ix+1, iy, iz+1] + A[ix+1, iy+2, iz+1]
                                                                 + A[ix+1, iy+1, iz] + A[ix+1, iy+1, iz+2] - 6.0 * A[ix+1, iy+1, iz+1])
    end
    return
end

@kernel function diffusion_kernel_ka!(A_new, A, h)
    ix, iy, iz = @index(Global, NTuple)
    if ix ∈ axes(A_new, 1) && iy ∈ axes(A_new, 2) && iz ∈ axes(A_new, 3)
        @inbounds A_new[ix, iy, iz] = A[ix+1, iy+1, iz+1] + h * (A[ix, iy+1, iz+1] + A[ix+2, iy+1, iz+1]
                                                                 + A[ix+1, iy, iz+1] + A[ix+1, iy+2, iz+1]
                                                                 + A[ix+1, iy+1, iz] + A[ix+1, iy+1, iz+2] - 6.0 * A[ix+1, iy+1, iz+1])
    end
end

function run_julia_benchmarks(n)
    A_new = CuArray{Float64}(undef, n - 2, n - 2, n - 2)
    A = CuArray{Float64}(undef, n, n, n)
    h = 1 / 7
    nthreads = (32, 8, 1)
    nblocks = cld.(size(A_new), nthreads)

    bm = @benchmark begin
        CUDA.@sync @cuda blocks=$nblocks threads=$nthreads diffusion_kernel!($A_new, $A, $h)
    end

    bm_ka = @benchmark begin
        diffusion_kernel_ka!(CUDABackend(), 256)($A_new, $A, $h; ndrange=($n, $n, $n))
        KernelAbstractions.synchronize(CUDABackend())
    end

    CUDA.unsafe_free!(A_new)
    CUDA.unsafe_free!(A)

    return (bm, bm_ka)
end

function run_c_benchmarks(lib, nsamples, n)
    trial = make_c_trial(nsamples)

    sym = Libdl.dlsym(lib, :run_benchmark)
    @ccall $sym(trial.times::Ptr{Cdouble}, nsamples::Cint, n::Cint)::Cvoid

    return trial
end

group = BenchmarkGroup()

# Compile C benchmark
libext = Sys.iswindows() ? "dll" : "so"
libname = "diffusion_3d." * libext
run(`nvcc -O3 -o $libname --shared -Xcompiler -fPIC diffusion_3d.cu`)

Libdl.dlopen("./$libname") do lib
    for N in INPUTS["diffusion-3d"].n_range
        @info "N = $N"
        group_n = BenchmarkGroup()
        group_n["julia"] = run_julia_benchmarks(N)[1]
        group_n["julia-ka"] = run_julia_benchmarks(N)[2]
        group_n["reference"] = run_c_benchmarks(lib, INPUTS["diffusion-3d"].c_samples, N)
        group[N] = group_n
        display(group_n)
    end
end

RESULTS["diffusion-3d"] = group
