include("common.jl")

@info "host overhead"
include("host_overhead.jl")

@info "memcopy"
include("memcopy.jl")

@info "diffusion"
include("diffusion_2d.jl")

abstract type HPCBenchmark end

_BENCHMARKS = Dict{Symbol, HPCBenchmark}


function runbenchmarks(benchmarks=:all)
    if benchmarks == :all
        benchmarks = collect(keys(_BENCHMARKS))
    end
end