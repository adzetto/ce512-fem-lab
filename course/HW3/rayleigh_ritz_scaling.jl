#=
CE 512 HW3 - Large-Scale Rayleigh-Ritz: n x n System Scaling
(fully vectorized + CUDA GPU offloading)

Assembly, solve, and evaluation are **loop-free**.
* K assembled via outer-product broadcasting: K = iv*iv' ./ (iv.+iv'.-1)
* u(x) evaluated as a matrix-vector product  V*c  (Vandermonde approach)
* GPU offloading via CUDA.jl for n >= 256

Outputs data_scaling.dat for pgfplots.
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

# Use all available BLAS threads on CPU
nt = Sys.CPU_THREADS
BLAS.set_num_threads(nt)
println("BLAS threads: $nt")

const OUTDIR = @__DIR__

# -- Exact solution (vectorized) -----------------------------------------------
u_exact_vec(xs) = ifelse.(xs .<= 0.5, 2.0 .* xs, 0.5 .+ xs)
sigma_exact_vec(xs) = ifelse.(xs .< 0.5, 2.0, ifelse.(xs .> 0.5, 1.0, 1.5))

# -- Vectorized assembly (no loops) -------------------------------------------
"""
    assemble_vec(n) -> (K, F)    [CPU arrays]

Build K and F using pure broadcasting - zero explicit loops.
"""
function assemble_vec(n::Int)
    iv = Float64.(2:n+1)                          # [2, 3, ..., n+1]
    K  = (iv * iv') ./ (iv .+ iv' .- 1.0)         # outer product + broadcast
    F  = (1.0 ./ (2.0 .^ iv)) .+ 1.0              # broadcast
    return K, F
end

# -- Vectorized solve (CPU or GPU) ---------------------------------------------
"""
    solve_system(K, F, n) -> (c, t_solve)

Solve Kc = F.  Offloads to GPU when CUDA is available and n >= 256.
Returns coefficient vector on CPU.
"""
function solve_system(K::Matrix{Float64}, F::Vector{Float64}, n::Int)
    if HAS_CUDA && n >= 256
        # GPU path
        t = @elapsed begin
            Kd = CuArray(K)
            Fd = CuArray(F)
            cd = Kd \ Fd
            c  = Array(cd)
        end
        return c, t
    else
        # CPU path - Cholesky (SPD) -> fallback LU -> fallback NaN
        local c
        t = @elapsed begin
            try
                c = cholesky(Symmetric(K)) \ F
            catch
                try
                    c = lu(K) \ F
                catch
                    c = fill(NaN, n)
                end
            end
        end
        return c, t
    end
end

# -- Vectorized evaluation (Vandermonde, no loops) -----------------------------
"""
    eval_u_vec(xs, n, c) -> u_vec

Evaluate u(x) = Sum[ c[a]*x^(a+1) ] at all points using a matrix-vector product.
"""
function eval_u_vec(xs::AbstractVector{Float64}, n::Int, c::Vector{Float64})
    ks   = Float64.(2:n+1)'                        # (1, n) row vector
    logx = log.(max.(xs, 1e-300))                   # (npts,) - safe log
    V    = exp.(logx .* ks)                         # (npts, n) Vandermonde-like
    return V * c                                    # matrix-vector, no loop
end

"""
    eval_sigma_vec(xs, n, c) -> sigma_vec

Evaluate sigma(x) = du/dx = Sum[ c[a]*k*x^(k-1) ] at all points.
"""
function eval_sigma_vec(xs::AbstractVector{Float64}, n::Int, c::Vector{Float64})
    ks   = Float64.(2:n+1)'
    logx = log.(max.(xs, 1e-300))
    W    = ks .* exp.(logx .* (ks .- 1.0))          # derivative Vandermonde
    return W * c
end

# ==============================================================================
#  CONFIGURATION
# ==============================================================================
n_vals = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50,
          75, 100, 150, 200, 300, 500, 750, 1000,
          1500, 2000, 3000, 5000, 7500, 10000]

const COND_MAX_N  = 2000   # cond() via SVD up to this n
const ERROR_MAX_N = 500    # L2 error via quadrature up to this n

# Quadrature grid - allocated once
x_quad = collect(range(0.0, 1.0, length=2000))
dx_q   = x_quad[2] - x_quad[1]
ue_q   = u_exact_vec(x_quad)
se_q   = sigma_exact_vec(x_quad)

# ==============================================================================
#  MAIN LOOP
# ==============================================================================
println("=" ^ 70)
println("CE 512 HW3 - Large-Scale Rayleigh-Ritz Scaling (vectorized, $(HAS_CUDA ? "GPU" : "CPU"))")
println("=" ^ 70)
println()

open(joinpath(OUTDIR, "data_scaling.dat"), "w") do io
    println(io, "n\tasm_s\tsolve_s\tcond_K\tresid\tu_tip\tu_tip_err\tu_L2\tsigma_L2")

    for n in n_vals
        @printf("n = %5d ... ", n)

        # -- Assembly (vectorized, timed) --
        local K, F
        t_asm = @elapsed begin
            K, F = assemble_vec(n)
        end

        # -- Condition number --
        kn = n <= COND_MAX_N ? cond(Symmetric(K)) : NaN

        # -- Solve --
        c, t_solve = solve_system(K, F, n)

        # -- Residual (vectorized) --
        resid = norm(K * c .- F) / norm(F)

        # -- Tip displacement u(1) = sum(c) since 1^k = 1 --
        u_tip   = sum(c)                              # no eval needed at x=1
        tip_err = abs(u_tip - 1.5) / 1.5 * 100

        # -- L2 errors (vectorized, no inner loop) --
        if n <= ERROR_MAX_N && isfinite(u_tip) && abs(u_tip) < 1e10
            ur = eval_u_vec(x_quad, n, c)
            sr = eval_sigma_vec(x_quad, n, c)

            du   = abs.(ur .- ue_q)
            ds   = abs.(sr .- se_q)
            u_l2 = sqrt(sum(du .^ 2) * dx_q)
            s_l2 = sqrt(sum(ds .^ 2) * dx_q)
        else
            u_l2 = NaN
            s_l2 = NaN
        end

        @printf(io, "%d\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\n",
                n, t_asm, t_solve, kn, resid, u_tip, tip_err, u_l2, s_l2)

        @printf("asm=%.4fs  solve=%.4fs  cond=%.2e  resid=%.2e  u(L)=%.6f\n",
                t_asm, t_solve, kn, resid, u_tip)
    end
end

println()
println("File written: data_scaling.dat")
println("Done.")
