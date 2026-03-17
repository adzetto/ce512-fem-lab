# CE 512 HW3 - Rayleigh-Ritz Convergence Study
# Normalized: P=1, EA=1, L=1. Outputs .dat files.

using LinearAlgebra, Printf

const HAS_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end
if HAS_CUDA
    println("CUDA available - GPU offloading enabled")
    print("Warming up CUDA JIT compilation... ")
    _k = CuArray(rand(5, 5)); _f = CuArray(rand(5))
    _c = _k \ _f; _ = Array(_c)
    println("done.")
end

const OUTDIR = @__DIR__

# Exact solutions
u_exact_v(xs) = ifelse.(xs .<= 0.5, 2.0 .* xs, 0.5 .+ xs)
sigma_exact_v(xs) = ifelse.(xs .< 0.5, 2.0, ifelse.(xs .> 0.5, 1.0, 1.5))

# Vectorized assembly & solve
function solve_rr(n::Int)
    ks = collect(2:n+1)
    iv = Float64.(ks)
    K = (iv * iv') ./ (iv .+ iv' .- 1.0)
    F = (1.0 ./ (2.0 .^ iv)) .+ 1.0

    if HAS_CUDA && n >= 256
        a = Array(CuArray(K) \ CuArray(F))
    else
        a = K \ F
    end
    return ks, a
end

# Vectorized evaluation
function eval_rr_vec(xs::AbstractVector{Float64}, ks::Vector{Int}, a::Vector{Float64})
    logx = log.(max.(xs, 1e-300))
    kf   = Float64.(ks)'
    V = exp.(logx .* kf)
    W = kf .* exp.(logx .* (kf .- 1.0))
    return V * a, W * a
end

x_fine   = collect(range(0.0, 1.0, length=500))
n_values = [1, 2, 3, 4, 5, 8, 12, 20, 30, 50]

# Displacement Data
open(joinpath(OUTDIR, "data_displacement.dat"), "w") do io
    print(io, "x\texact")
    for n in n_values; print(io, "\tn$n"); end
    println(io)

    ue = u_exact_v(x_fine)
    sols = [solve_rr(n) for n in n_values]
    u_cols = [eval_rr_vec(x_fine, ks, a)[1] for (ks, a) in sols]

    for p in eachindex(x_fine)
        @printf(io, "%.6f\t%.6f", x_fine[p], ue[p])
        for col in u_cols
            @printf(io, "\t%.6f", col[p])
        end
        println(io)
    end
end

# Stress Data
open(joinpath(OUTDIR, "data_stress.dat"), "w") do io
    print(io, "x\texact")
    for n in n_values; print(io, "\tn$n"); end
    println(io)

    se = sigma_exact_v(x_fine)
    sols = [solve_rr(n) for n in n_values]
    sigma_cols = [eval_rr_vec(x_fine, ks, a)[2] for (ks, a) in sols]

    for p in eachindex(x_fine)
        @printf(io, "%.6f\t%.6f", x_fine[p], se[p])
        for col in sigma_cols
            @printf(io, "\t%.6f", col[p])
        end
        println(io)
    end
end

# Error Data
n_range = 1:60
x_quad  = collect(range(0.0, 1.0, length=2000))
dx      = x_quad[2] - x_quad[1]
ue_quad = u_exact_v(x_quad)
se_quad = sigma_exact_v(x_quad)

open(joinpath(OUTDIR, "data_errors.dat"), "w") do io
    println(io, "n\tu_L2\tu_Linf\tsigma_L2\tsigma_Linf\tu_tip_pct")

    for n in n_range
        ks, a = solve_rr(n)
        u_rr, s_rr = eval_rr_vec(x_quad, ks, a)

        du = abs.(u_rr .- ue_quad)
        ds = abs.(s_rr .- se_quad)

        u_L2   = sqrt(sum(du .^ 2) * dx)
        u_Linf = maximum(du)
        s_L2   = sqrt(sum(ds .^ 2) * dx)
        s_Linf = maximum(ds)
        tip_pct  = abs(u_rr[end] - 1.5) / 1.5 * 100

        @printf(io, "%d\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\n",
                n, u_L2, u_Linf, s_L2, s_Linf, tip_pct)
    end
end

# Print Summary
println("=" ^ 65)
println("CE 512 HW3 - Rayleigh-Ritz Convergence (vectorized, $(HAS_CUDA ? "GPU" : "CPU"))")
println("=" ^ 65)
println()
@printf("%-5s  %10s  %10s  %10s\n", "n", "u(L)", "u_err(%)", "||u||_L2")
println("-" ^ 45)

for n in n_values
    ks, a = solve_rr(n)
    u_rr, _ = eval_rr_vec(x_quad, ks, a)
    u_L2 = sqrt(sum(abs.(u_rr .- ue_quad) .^ 2) * dx)

    @printf("%-5d  %10.6f  %10.4f  %10.6f\n",
            n, u_rr[end], abs(u_rr[end] - 1.5) / 1.5 * 100, u_L2)
end

println("Done.")
