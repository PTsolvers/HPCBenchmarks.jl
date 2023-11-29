INPUTS["memcopy"] = (
    n_range=2 .^ (16:2:28),
    c_samples=2000,
)

function memcopy_kernel!(dst, src)
    ix = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    if ix <= length(dst)
        @inbounds dst[ix] = src[ix]
    end
    return
end

function run_julia_benchmarks(nbytes)
    dst = CuArray{UInt8}(undef, nbytes)
    src = CuArray{UInt8}(undef, nbytes)
    nthreads = 256
    nblocks = cld(length(dst), nthreads)

    bm = @benchmark begin
        CUDA.@sync @cuda blocks=$nblocks threads=$nthreads memcopy_kernel!($dst, $src)
    end

    CUDA.unsafe_free!(dst)
    CUDA.unsafe_free!(src)

    return bm
end

function run_c_benchmarks(lib, nsamples, nbytes)
    trial = make_c_trial(nsamples)

    CUDA.reclaim()

    sym = Libdl.dlsym(lib, :run_benchmark)
    @ccall $sym(trial.times::Ptr{Cdouble}, nsamples::Cint, nbytes::Cint)::Cvoid

    return trial
end

group = BenchmarkGroup()

# Compile C benchmark
libext = Sys.iswindows() ? "dll" : "so"
libname = "memcopy." * libext
run(`nvcc -O3 -o $libname --shared -Xcompiler -fPIC memcopy.cu`)

Libdl.dlopen("./$libname") do lib
    for N in INPUTS["memcopy"].n_range
        @info "N = $N"
        group_n = BenchmarkGroup()
        group_n["julia"] = run_julia_benchmarks(N)
        group_n["reference"] = run_c_benchmarks(lib, INPUTS["memcopy"].c_samples, N)
        group[N] = group_n
        display(group_n)
    end
end

RESULTS["memcopy"] = group
