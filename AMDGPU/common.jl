using AMDGPU
using KernelAbstractions
using BenchmarkTools
using Statistics
using Libdl
using AxisKeys
using Plots

function judge_map(estimates)
    ek = keys(estimates) |> collect
    ev = values(estimates)
    jm = [judge(e1, e2) for e1 in ev, e2 in ev]
    return KeyedArray(jm, to=ek, from=ek)
end

function make_c_trial(nsamples)
    c_times = zeros(Float64, nsamples)
    c_gctimes = zeros(Float64, nsamples)
    c_memory = 0::Int64
    c_allocs = 0::Int64
    c_params = BenchmarkTools.DEFAULT_PARAMETERS
    c_params.samples = nsamples
    return BenchmarkTools.Trial(c_params, c_times, c_gctimes, c_memory, c_allocs)
end

RESULTS = BenchmarkGroup()
INPUTS = Dict()
