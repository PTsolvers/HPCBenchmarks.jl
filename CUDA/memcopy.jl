NBYTES    = 10^8
C_SAMPLES = 500

function memcopy_kernel!(dst,src)
    ix = (blockIdx().x-1i32)*blockDim().x + threadIdx().x
    if ix <= length(dst)
        @inbounds dst[ix] = src[ix]
    end
    return
end

function run_julia_benchmarks(nbytes)
    dst = CuArray{UInt8}(undef,nbytes)
    src = CuArray{UInt8}(undef,nbytes)
    nthreads = 256
    nblocks  = cld(length(dst),nthreads)

    @benchmark begin
        CUDA.@sync @cuda blocks=$nblocks threads=$nthreads memcopy_kernel!($dst,$src)
    end
end

function run_c_benchmarks(lib,nsamples,nbytes)
    trial = make_c_trial(nsamples)

    sym = CUDA.Libdl.dlsym(lib,:run_benchmark)
    @ccall $sym(trial.times::Ptr{Cdouble},nsamples::Cint,nbytes::Cint)::Cvoid
    CUDA.device_reset!()

    return trial
end

group = BenchmarkGroup()
group["julia"] = run_julia_benchmarks(NBYTES)

# Add baseline C benchmark
libext  = Sys.iswindows() ? "dll" : "so"
libname = "memcopy." * libext
run(`nvcc -O3 -o $libname --shared memcopy.cu`)
Libdl.dlopen("./$libname") do lib
    group["reference"] = run_c_benchmarks(lib,C_SAMPLES,NBYTES)
end

RESULTS["memcopy"] = group

