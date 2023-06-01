using CUDA
import CUDA: i32

using BenchmarkTools
using Statistics
using Libdl

RESULTS = BenchmarkGroup()

include("common.jl")

@info "host overhead"
include("host_overhead.jl")

@info "memcopy"
include("memcopy.jl")