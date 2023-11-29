INPUTS["memcopy"] = (
    n_range=2 .^ (16:2:28),
    c_samples=2000,
)

function memcopy_kernel!(dst, src)
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    if ix <= length(dst)
        @inbounds dst[ix] = src[ix]
    end
    return
end

function run_julia_benchmarks(n)
    dst = ROCArray{Float64}(undef, n)
    src = ROCArray{Float64}(undef, n)
    nthreads = 256
    nblocks = cld(length(dst), nthreads)

    bm = @benchmark begin
        @roc gridsize=$nblocks groupsize=$nthreads memcopy_kernel!($dst, $src)
        AMDGPU.synchronize()
    end

    AMDGPU.unsafe_free!(dst)
    AMDGPU.unsafe_free!(src)

    return bm
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
libname = "memcopy." * libext
run(`hipcc -O3 -o $libname --shared -fPIC memcopy.cu`)

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
