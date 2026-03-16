using LinearAlgebra
using DelimitedFiles

outdir = length(ARGS) >= 1 ? ARGS[1] : ""

A = [1.0, 1.0, 1.0]
E = [64.0, 64.0, 64.0]
L = [4.0, 4.0, 6.0]
alpha = [acos(3.0 / 4.0), -acos(3.0 / 4.0), 0.0]

Dvec = [
    1 2 5 6
    5 6 3 4
    1 2 3 4
]

N = maximum(Dvec)
I6 = Matrix{Float64}(I, N, N)
S1 = I6[Dvec[1, :], :]
S2 = I6[Dvec[2, :], :]
S3 = I6[Dvec[3, :], :]

kref = [
    1.0 0.0 -1.0 0.0
    0.0 0.0 0.0 0.0
    -1.0 0.0 1.0 0.0
    0.0 0.0 0.0 0.0
]

c = cos.(alpha)
s = sin.(alpha)

T1 = [
    c[1] s[1] 0.0 0.0
    -s[1] c[1] 0.0 0.0
    0.0 0.0 c[1] s[1]
    0.0 0.0 -s[1] c[1]
]
T2 = [
    c[2] s[2] 0.0 0.0
    -s[2] c[2] 0.0 0.0
    0.0 0.0 c[2] s[2]
    0.0 0.0 -s[2] c[2]
]
T3 = [
    c[3] s[3] 0.0 0.0
    -s[3] c[3] 0.0 0.0
    0.0 0.0 c[3] s[3]
    0.0 0.0 -s[3] c[3]
]

k1 = A[1] * E[1] / L[1] * kref
k2 = A[2] * E[2] / L[2] * kref
k3 = A[3] * E[3] / L[3] * kref

kg1 = transpose(T1) * k1 * T1
kg2 = transpose(T2) * k2 * T2
kg3 = transpose(T3) * k3 * T3

K = transpose(S1) * kg1 * S1 + transpose(S2) * kg2 * S2 + transpose(S3) * kg3 * S3

G = [
    1.0 0.0 0.0 0.0 0.0 0.0
    0.0 1.0 0.0 0.0 0.0 0.0
    0.0 0.0 -sin(deg2rad(60.0)) cos(deg2rad(60.0)) 0.0 0.0
]
Q = zeros(Float64, size(G, 1))
a_G = maximum(K)
Gbar = a_G * G
Qbar = a_G * Q

P = zeros(Float64, N)
P[6] = -10.0

AL = [K transpose(Gbar); Gbar zeros(Float64, size(G, 1), size(G, 1))]
solution = AL \ vcat(P, Qbar)

U = solution[1:N]
Lag = solution[(N + 1):end] * a_G
R = K[1:4, :] * U

ueg1 = U[Dvec[1, :]]
ueg2 = U[Dvec[2, :]]
ueg3 = U[Dvec[3, :]]

ue1 = T1 * ueg1
ue2 = T2 * ueg2
ue3 = T3 * ueg3

local_displacements = hcat(ue1, ue2, ue3)
member_forces = hcat(k1 * ue1, k2 * ue2, k3 * ue3)
constraint_residual = G * U - Q

if outdir != ""
    mkpath(outdir)
    writedlm(joinpath(outdir, "U.tsv"), reshape(U, :, 1), '\t')
    writedlm(joinpath(outdir, "Lag.tsv"), reshape(Lag, :, 1), '\t')
    writedlm(joinpath(outdir, "R.tsv"), reshape(R, :, 1), '\t')
    writedlm(joinpath(outdir, "member_forces.tsv"), member_forces, '\t')
    writedlm(joinpath(outdir, "local_displacements.tsv"), local_displacements, '\t')
    writedlm(joinpath(outdir, "constraint_residual.tsv"), reshape(constraint_residual, :, 1), '\t')
else
    println("U = ", U)
    println("Lag = ", Lag)
    println("R = ", R)
    println("local_displacements = ", local_displacements)
    println("member_forces = ", member_forces)
end
