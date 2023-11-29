N_MILLISEC = 2
C_SAMPLES = 2000

function sleep_kernel(ncycles)
    start = CUDA.clock(UInt64)
    while CUDA.clock(UInt64) - start < ncycles
        sync_threads()
    end
    return
end

function run_julia_benchmarks(ncycles)
    suite = BenchmarkGroup()

    suite["nonblocking"] = @benchmarkable begin
        @cuda sleep_kernel($ncycles)
        CUDA.synchronize()
    end

    suite["blocking"] = @benchmarkable begin
        @cuda sleep_kernel($ncycles)
        CUDA.cuStreamSynchronize(stream())
    end

    warmup(suite)
    return run(suite)
end

function run_c_benchmarks(lib, nsamples, ncycles)
    trial = make_c_trial(nsamples)

    CUDA.reclaim()

    sym = Libdl.dlsym(lib, :run_benchmark)
    @ccall $sym(trial.times::Ptr{Cdouble}, nsamples::Cint, ncycles::Cint)::Cvoid

    return trial
end

clock_rate = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_CLOCK_RATE)
ncycles = N_MILLISEC * clock_rate

group = run_julia_benchmarks(ncycles)

# Add baseline C benchmark
libext = Sys.iswindows() ? "dll" : "so"
libname = "host_overhead." * libext
run(`nvcc -O3 -o $libname --shared host_overhead.cu`)
Libdl.dlopen("./$libname") do lib
    group["reference"] = run_c_benchmarks(lib, C_SAMPLES, ncycles)
end

RESULTS["host-overhead"] = group
