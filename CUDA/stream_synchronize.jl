using CUDA
using BenchmarkTools
using Statistics
using Libdl
using AxisKeys

const NSAMPLES = 2000
const NSPINS   = 20_000

function busy_kernel(nspins)
    ispin = 0
    while ispin < nspins
        ispin += 1
        sync_threads()
    end
    return
end

function run_c_benchmark(lib)
    c_times   = zeros(Float64,NSAMPLES)
    c_gctimes = zeros(Float64,NSAMPLES)
    c_memory  = 0::Int64
    c_allocs  = 0::Int64
    c_params  = BenchmarkTools.DEFAULT_PARAMETERS
    c_params.samples = NSAMPLES

    sym = CUDA.Libdl.dlsym(lib,:run_benchmark)
    @ccall $sym(c_times::Ptr{Cdouble},NSAMPLES::Cint,NSPINS::Cint)::Cvoid
    CUDA.device_reset!()

    return BenchmarkTools.Trial(c_params,c_times,c_gctimes,c_memory,c_allocs)
end

suite = BenchmarkGroup()

suite["nonblocking"] = @benchmarkable begin
    @cuda busy_kernel(NSPINS)
    CUDA.synchronize()
end

suite["blocking"] = @benchmarkable begin
    @cuda busy_kernel(NSPINS)
    CUDA.cuStreamSynchronize(stream())
end

results = run(suite; samples=NSAMPLES)

# Add baseline C benchmark
libext  = Sys.iswindows() ? "dll" : "so"
libname = "stream_synchronize." * libext
run(`nvcc -o $libname --shared stream_synchronize.cu`)
Libdl.dlopen("./$libname") do lib
    results["reference"] = run_c_benchmark(lib)
end

function judge_map(estimates)
    ek = keys(estimates) |> collect
    ev = values(estimates)
    jm = [judge(e1,e2) for e1 in ev, e2 in ev]
    return KeyedArray(jm, to=ek, from=ek)
end