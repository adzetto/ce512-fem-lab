#=
CE 512 HW3 - Rayleigh-Ritz Convergence Study  (fully vectorized + optional CUDA)

General n-term polynomial:  u(x) = a_2*x^2 + a_3*x^3 + ... + a_{n+1}*x^{n+1}
(a_1*x omitted as required by the homework)

Stiffness matrix:  K[a,b] = i*j / (i+j-1),  i=a+1, j=b+1   <- Hilbert-like
Force vector:      F[b]   = (1/2)^(b+1) + 1

Assembly, evaluation, and error quadrature are **loop-free** (broadcasting +
matrix-vector products).  When CUDA.jl is available the stiffness solve and the
dense evaluation are offloaded to the GPU for very large n.

Outputs .dat files for pgfplots import in LaTeX.
Normalized: P=1, EA=1, L=1.
=#

using LinearAlgebra, Printf

# -- Try to load CUDA ----------------------------------------------------------
const HAS_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end
if HAS_CUDA
    println("CUDA available - GPU offloading enabled")
    print("Warming up CUDA JIT compilation... ")
    _k = CuArray(rand(5, 5))
    _f = CuArray(rand(5))
    _c = _k \ _f
    _ = Array(_c)
    println("done.")
end

const OUTDIR = @__DIR__

# -- Exact solution (normalized: P=EA=L=1) -------------------------------------
u_exact_v(xs) = ifelse.(xs .<= 0.5, 2.0 .* xs, 0.5 .+ xs)          # vectorized
sigma_exact_v(xs) = ifelse.(xs .< 0.5, 2.0, ifelse.(xs .> 0.5, 1.0, 1.5))

# -- Vectorized assembly -------------------------------------------------------
"""
    solve_rr(n) -> (ks, a)

Assemble the n x n stiffness matrix K and force vector F **without loops**,
then solve Ka = F.  Uses GPU if available and n is large enough.
"""
function solve_rr(n::Int)
    ks = collect(2:n+1)                       # exponents [2, 3, ..., n+1]
    iv = Float64.(ks)                         # row/col indices as floats

    # K[a,b] = i*j / (i+j-1) - outer product divided element-wise
    K = (iv * iv') ./ (iv .+ iv' .- 1.0)      # pure broadcasting, no loop

    # F[b] = 1/2^j + 1
    F = (1.0 ./ (2.0 .^ iv)) .+ 1.0           # pure broadcast

    # Solve - offload to GPU for large n
    if HAS_CUDA && n >= 256
        Kd = CuArray(K)
        Fd = CuArray(F)
        ad = Kd \ Fd
        a  = Array(ad)
    else
        a = K \ F
    end

    return ks, a
end

# -- Vectorized evaluation -----------------------------------------------------
"""
    eval_rr_vec(xs, ks, a) -> (u_vec, sigma_vec)

Evaluate displacement u(x) and stress sigma(x) at **all** points in `xs`
simultaneously using a single matrix-vector product (no loops).

    V[p, m] = xs[p]^ks[m]           ->  u  = V * a
    W[p, m] = ks[m] * xs[p]^(ks[m]-1)  ->  sigma = W * a
"""
function eval_rr_vec(xs::AbstractVector{Float64}, ks::Vector{Int}, a::Vector{Float64})
    # Vandermonde-like matrix: V[p,m] = x_p ^ k_m   (broadcasting, no loop)
    # log-exp trick avoids element-wise power loop:
    #   x^k = exp(k * log(x)),  with x=0 handled by replacing log(0)->-Inf
    logx = log.(max.(xs, 1e-300))              # column vector (npts,)
    kf   = Float64.(ks)'                       # row vector    (1, n)

    V = exp.(logx .* kf)                       # (npts, n) - all powers at once
    W = kf .* exp.(logx .* (kf .- 1.0))        # (npts, n) - derivative powers

    u_vec = V * a                              # matrix-vector, no loop
    sigma_vec = W * a
    return u_vec, sigma_vec
end

# ==============================================================================
#  CONFIGURATION
# ==============================================================================
x_fine   = collect(range(0.0, 1.0, length=500))
n_values = [1, 2, 3, 4, 5, 8, 12, 20, 30, 50]

# ==============================================================================
#  DISPLACEMENT DATA
# ==============================================================================
open(joinpath(OUTDIR, "data_displacement.dat"), "w") do io
    # header
    print(io, "x\texact")
    for n in n_values; print(io, "\tn$n"); end
    println(io)

    ue = u_exact_v(x_fine)                               # exact (vectorized)
    sols = [solve_rr(n) for n in n_values]              # precompute all
    u_cols = [eval_rr_vec(x_fine, ks, a)[1] for (ks, a) in sols]

    # write - single pass
    for p in eachindex(x_fine)
        @printf(io, "%.6f\t%.6f", x_fine[p], ue[p])
        for col in u_cols
            @printf(io, "\t%.6f", col[p])
        end
        println(io)
    end
end

# ==============================================================================
#  STRESS DATA
# ==============================================================================
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

# ==============================================================================
#  ERROR DATA  (vectorized quadrature - no inner loop over x)
# ==============================================================================
n_range = 1:60
x_quad  = collect(range(0.0, 1.0, length=2000))
dx      = x_quad[2] - x_quad[1]
ue_quad = u_exact_v(x_quad)
se_quad = sigma_exact_v(x_quad)

open(joinpath(OUTDIR, "data_errors.dat"), "w") do io
    println(io, "n\tu_L2\tu_Linf\tsigma_L2\tsigma_Linf\tu_tip_pct")

    for n in n_range
        ks, a = solve_rr(n)
        u_rr, s_rr = eval_rr_vec(x_quad, ks, a)    # all 2000 pts, no loop

        du = abs.(u_rr .- ue_quad)
        ds = abs.(s_rr .- se_quad)

        u_L2   = sqrt(sum(du .^ 2) * dx)            # vectorized norms
        u_Linf = maximum(du)
        s_L2   = sqrt(sum(ds .^ 2) * dx)
        s_Linf = maximum(ds)

        u_tip    = u_rr[end]                          # x=1 is last point
        tip_pct  = abs(u_tip - 1.5) / 1.5 * 100

        @printf(io, "%d\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\n",
                n, u_L2, u_Linf, s_L2, s_Linf, tip_pct)
    end
end

# ==============================================================================
#  PRINT SUMMARY
# ==============================================================================
println("=" ^ 65)
println("CE 512 HW3 - Rayleigh-Ritz Convergence (vectorized, $(HAS_CUDA ? "GPU" : "CPU"))")
println("=" ^ 65)
println()
@printf("%-5s  %10s  %10s  %10s\n", "n", "u(L)", "u_err(%)", "||u||_L2")
println("-" ^ 45)

for n in n_values
    ks, a = solve_rr(n)
    u_rr, _ = eval_rr_vec(x_quad, ks, a)

    du   = abs.(u_rr .- ue_quad)
    u_L2 = sqrt(sum(du .^ 2) * dx)
    u_tip = u_rr[end]

    @printf("%-5d  %10.6f  %10.4f  %10.6f\n",
            n, u_tip, abs(u_tip - 1.5) / 1.5 * 100, u_L2)
end

println()
println("Files written:")
println("  data_displacement.dat")
println("  data_stress.dat")
println("  data_errors.dat")
