INPUTS["diffusion-2d"] = (
    c_samples=2000,
    n_range=2 .^ (8:2:14),
)

function diffusion_kernel!(A_new, A, h)
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    if ix ∈ axes(A_new, 1) && iy ∈ axes(A_new, 2)
        @inbounds A_new[ix, iy] = A[ix+1, iy+1] + h * (A[ix, iy+1] + A[ix+2, iy+1] + A[ix+1, iy] + A[ix+1, iy+2] - 4.0 * A[ix+1, iy+1])
    end
    return
end

@kernel function diffusion_kernel_ka!(A_new, A, h)
    ix, iy = @index(Global, NTuple)
    if ix ∈ axes(A_new, 1) && iy ∈ axes(A_new, 2)
        @inbounds A_new[ix, iy] = A[ix+1, iy+1] + h * (A[ix, iy+1] + A[ix+2, iy+1] + A[ix+1, iy] + A[ix+1, iy+2] - 4.0 * A[ix+1, iy+1])
    end
end

function run_julia_benchmarks(n)
    A_new = ROCArray{Float64}(undef, n - 2, n - 2)
    A = ROCArray{Float64}(undef, n, n)
    h = 1 / 5
    nthreads = (128, 2)
    nblocks = cld.(size(A_new), nthreads)

    bm = @benchmark begin
        @roc gridsize=$nblocks groupsize=$nthreads diffusion_kernel!($A_new, $A, $h)
        AMDGPU.synchronize()
    end

    bm_ka = @benchmark begin
        diffusion_kernel_ka!(ROCBackend(), 256)($A_new, $A, $h; ndrange=($n, $n))
        KernelAbstractions.synchronize(ROCBackend())
    end

    AMDGPU.unsafe_free!(A_new)
    AMDGPU.unsafe_free!(A)

    return (bm, bm_ka)
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
libname = "diffusion_2d." * libext
run(`hipcc -O3 -o $libname --shared -fPIC diffusion_2d.cu`)

Libdl.dlopen("./$libname") do lib
    for N in INPUTS["diffusion-2d"].n_range
        @info "N = $N"
        group_n = BenchmarkGroup()
        group_n["julia"] = run_julia_benchmarks(N)[1]
        group_n["julia-ka"] = run_julia_benchmarks(N)[2]
        group_n["reference"] = run_c_benchmarks(lib, INPUTS["diffusion-2d"].c_samples, N)
        group[N] = group_n
        display(group_n)
    end
end

RESULTS["diffusion-2d"] = group
