using CUDA
using KernelAbstractions
using BenchmarkTools
using Libdl

# function make_c_trial(nsamples)
#     c_times = zeros(Float64, nsamples)
#     c_gctimes = zeros(Float64, nsamples)
#     c_memory = 0::Int64
#     c_allocs = 0::Int64
#     c_params = BenchmarkTools.DEFAULT_PARAMETERS
#     c_params.samples = nsamples
#     return BenchmarkTools.Trial(c_params, c_times, c_gctimes, c_memory, c_allocs)
# end

# INPUTS = Dict()

# INPUTS["atomic"] = (
#     c_samples=2000,
# )

function cuda_atomic_add!(target1, target2, source, indices)
    i = threadIdx().x + (blockIdx().x - 1) * gridDim().x
    i1, i2, i3, i4 = indices[i, 1], indices[i, 2], indices[i, 3], indices[i, 4]
    v = source[i]
    CUDA.@atomic target1[i1] += v
    CUDA.@atomic target1[i2] += v
    CUDA.@atomic target1[i3] += v
    CUDA.@atomic target1[i4] += v
    CUDA.@atomic target2[i1] += v
    CUDA.@atomic target2[i2] += v
    CUDA.@atomic target2[i3] += v
    CUDA.@atomic target2[i4] += v
    return
end

@kernel function ka_atomic_add!(target1, target2, source, indices)
    i = @index(Global, Linear)
    i1, i2, i3, i4 = indices[i, 1], indices[i, 2], indices[i, 3], indices[i, 4]
    v = source[i]
    KernelAbstractions.@atomic target1[i1] += v
    KernelAbstractions.@atomic target1[i2] += v
    KernelAbstractions.@atomic target1[i3] += v
    KernelAbstractions.@atomic target1[i4] += v
    KernelAbstractions.@atomic target2[i1] += v
    KernelAbstractions.@atomic target2[i2] += v
    KernelAbstractions.@atomic target2[i3] += v
    KernelAbstractions.@atomic target2[i4] += v
end

function run_julia_benchmarks(::Type{DAT}) where DAT
    n, bins = 1024, 64
    target1 = CuArray(zeros(DAT, bins))
    target2 = CuArray(zeros(DAT, bins))
    source  = CuArray(rand(DAT, n))
    indices = CuArray(rand(1:bins, n, 4))

    nthreads = 256
    nblocks = cld.(n, nthreads)

    bm = @benchmark begin
        @cuda threads=$nthreads blocks=$nblocks cuda_atomic_add!($target1, $target2, $source, $indices)
        CUDA.synchronize()
    end

    bm_ka = @benchmark begin
        ka_atomic_add!(CUDABackend(), 256, $n)($target1, $target2, $source, $indices)
        KernelAbstractions.synchronize(CUDABackend())
    end

    CUDA.unsafe_free!(source)
    CUDA.unsafe_free!(indices)
    CUDA.unsafe_free!(target1)
    CUDA.unsafe_free!(target2)

    return (bm, bm_ka)
end

function run_c_benchmarks(lib, nsamples)
    trial = make_c_trial(nsamples)

    sym = Libdl.dlsym(lib, :run_benchmark)
    @ccall $sym(trial.times::Ptr{Cdouble}, nsamples::Cint)::Cvoid

    return trial
end

# Compile C benchmark
libext = Sys.iswindows() ? "dll" : "so"
libname = "atomic." * libext
run(`hipcc -O3 -o $libname --shared -fPIC atomic.cu`)

Libdl.dlopen("./$libname") do lib
    group_n = BenchmarkGroup()
    jb = run_julia_benchmarks(Float32)
    group_n["julia"] = jb[1]
    group_n["julia-ka"] = jb[2]
    group_n["reference"] = run_c_benchmarks(lib, INPUTS["atomic"].c_samples)
    display(group_n)
end


