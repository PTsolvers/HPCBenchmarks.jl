include("common.jl")

# @info "host overhead"
# include("host_overhead.jl")

@info "memcopy"
include("memcopy.jl")

@info "diffusion"
include("diffusion_2d.jl")

@info "diffusion 3D"
include("diffusion_3d.jl")
