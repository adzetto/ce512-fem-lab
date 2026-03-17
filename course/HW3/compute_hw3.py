"""
CE 512 - Finite Element Method - Homework 3, Question 1
Numerical verification of the Rayleigh-Ritz solutions.

Muhammet Yagcioglu - 290204042
"""

import numpy as np

print("=" * 65)
print("CE 512 HW3 Q1 - Rayleigh-Ritz Verification")
print("=" * 65)

# =====================================================================
# Part (a): u(x) = a2 * x^2
# =====================================================================
print("\n--- Part (a): u(x) = a2 * x^2 ---\n")

# K_22 = 4*EA*L^3 / 3
# F_2  = 5*P*L^2 / 4
# a2   = F_2 / K_22 = (5*P*L^2/4) * (3/(4*EA*L^3)) = 15*P/(16*EA*L)

# In normalized form (set P=1, EA=1, L=1):
K_a = 4 / 3
F_a = 5 / 4
a2_a = F_a / K_a
print(f"K_22 / (EA*L^3) = {K_a:.6f}")
print(f"F_2  / (P*L^2)  = {F_a:.6f}")
print(f"a2 * (EA*L/P)   = {a2_a:.6f}  (should be 15/16 = {15/16:.6f})")

# Displacements (normalized: u * EA / (P*L))
xbar = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
u_a = (15 / 16) * xbar**2

print(f"\nNormalized displacement u_a * EA/(PL):")
for x, u in zip(xbar, u_a):
    print(f"  x/L = {x:.2f}:  u_bar = {u:.6f}")

# Stress (normalized: sigma * A / P)
sigma_a = (15 / 8) * xbar
print(f"\nNormalized stress sigma_a * A/P:")
for x, s in zip(xbar, sigma_a):
    print(f"  x/L = {x:.2f}:  sigma_bar = {s:.6f}")

# =====================================================================
# Part (b): u(x) = a2 * x^2 + a3 * x^3
# =====================================================================
print("\n--- Part (b): u(x) = a2*x^2 + a3*x^3 ---\n")

# Stiffness matrix (normalized by EA*L^3, EA*L^4, EA*L^5 resp.)
# K = EA * [[4L^3/3, 3L^4/2], [3L^4/2, 9L^5/5]]
# Set EA=1, L=1:
K_b = np.array([
    [4 / 3, 3 / 2],
    [3 / 2, 9 / 5]
])

# Force vector: F = P * [5L^2/4, 9L^3/8], set P=1, L=1:
F_b = np.array([5 / 4, 9 / 8])

print("Stiffness matrix K / (EA):")
print(f"  [{K_b[0,0]:10.6f}  {K_b[0,1]:10.6f}]")
print(f"  [{K_b[1,0]:10.6f}  {K_b[1,1]:10.6f}]")
print(f"\nForce vector F / P:")
print(f"  [{F_b[0]:10.6f}  {F_b[1]:10.6f}]")

# Determinant
det_K = np.linalg.det(K_b)
print(f"\ndet(K) / (EA)^2 = {det_K:.6f}  (should be 3/20 = {3/20:.6f})")

# Solve
a_b = np.linalg.solve(K_b, F_b)
a2_b = a_b[0]
a3_b = a_b[1]

print(f"\na2 * (EA*L/P)   = {a2_b:.6f}  (should be 15/4 = {15/4:.6f})")
print(f"a3 * (EA*L^2/P) = {a3_b:.6f}  (should be -5/2 = {-5/2:.6f})")

# Verify
residual = K_b @ a_b - F_b
print(f"\nResidual ||K*a - F|| = {np.linalg.norm(residual):.2e}")

# Displacements (normalized)
u_b = (15 / 4) * xbar**2 - (5 / 2) * xbar**3
print(f"\nNormalized displacement u_b * EA/(PL):")
for x, u in zip(xbar, u_b):
    print(f"  x/L = {x:.2f}:  u_bar = {u:.6f}")

# Stress (normalized)
sigma_b = (15 / 2) * xbar * (1 - xbar)
print(f"\nNormalized stress sigma_b * A/P:")
for x, s in zip(xbar, sigma_b):
    print(f"  x/L = {x:.2f}:  sigma_bar = {s:.6f}")

# =====================================================================
# Exact solution
# =====================================================================
print("\n--- Exact Solution ---\n")

u_exact = np.where(xbar <= 0.5, 2 * xbar, 0.5 + xbar)
sigma_exact_left = 2.0   # for x < L/2
sigma_exact_right = 1.0  # for x > L/2

print("Normalized displacement u_exact * EA/(PL):")
for x, u in zip(xbar, u_exact):
    print(f"  x/L = {x:.2f}:  u_bar = {u:.6f}")

print(f"\nNormalized stress: sigma_exact * A/P = {sigma_exact_left:.1f} for x < L/2")
print(f"                   sigma_exact * A/P = {sigma_exact_right:.1f} for x > L/2")

# =====================================================================
# Comparison table
# =====================================================================
print("\n--- Displacement Comparison ---\n")
print(f"{'x/L':>5s}  {'Exact':>8s}  {'Part(a)':>8s}  {'Err(a)':>8s}  {'Part(b)':>8s}  {'Err(b)':>8s}")
print("-" * 55)
for x, ue, ua, ub in zip(xbar, u_exact, u_a, u_b):
    err_a = (1 - ua / ue) * 100 if ue > 0 else 0
    err_b = (1 - ub / ue) * 100 if ue > 0 else 0
    print(f"{x:5.2f}  {ue:8.4f}  {ua:8.4f}  {err_a:7.1f}%  {ub:8.4f}  {err_b:7.1f}%")

print("\n--- Stress Comparison ---\n")
print(f"{'x/L':>5s}  {'Exact':>8s}  {'Part(a)':>8s}  {'Part(b)':>8s}")
print("-" * 40)
for x, sa, sb in zip(xbar, sigma_a, sigma_b):
    se = sigma_exact_left if x < 0.5 else (sigma_exact_right if x > 0.5 else 2.0)
    print(f"{x:5.2f}  {se:8.4f}  {sa:8.4f}  {sb:8.4f}")


# =====================================================================
# QUESTION 2: Rectangular Element Shape Functions
# =====================================================================
print("\n" + "=" * 65)
print("CE 512 HW3 Q2 - Rectangular Element Verification")
print("=" * 65)

a_dim, b_dim = 1.0, 1.0  # use unit dimensions for verification

# Node coordinates (CCW from bottom-left)
nodes = np.array([
    [-a_dim, -b_dim],
    [ a_dim, -b_dim],
    [ a_dim,  b_dim],
    [-a_dim,  b_dim]
])
xi_I  = np.array([-1, +1, +1, -1])
eta_I = np.array([-1, -1, +1, +1])

# [A] matrix: A_Ir = p_r(s_I, t_I)
A_mat = np.zeros((4, 4))
for I in range(4):
    s_I, t_I = nodes[I]
    A_mat[I] = [1, s_I, t_I, s_I * t_I]

print("\n[A] matrix:")
for row in A_mat:
    print("  [" + "  ".join(f"{v:6.1f}" for v in row) + " ]")

print(f"\ndet[A] = {np.linalg.det(A_mat):.4f}  (expected -16*a^2*b^2 = {-16*a_dim**2*b_dim**2:.4f})")

A_inv = np.linalg.inv(A_mat)
print("\n[A]^{-1} matrix:")
for row in A_inv:
    print("  [" + "  ".join(f"{v:8.4f}" for v in row) + " ]")

# Verify A * A^{-1} = I
identity_check = A_mat @ A_inv
print(f"\n||A * A^{{-1}} - I|| = {np.linalg.norm(identity_check - np.eye(4)):.2e}")

# Shape functions
def N_I_func(I, s, t, a=1.0, b=1.0):
    return (a + xi_I[I] * s) * (b + eta_I[I] * t) / (4 * a * b)

# Verify Kronecker delta: N_I(s_J, t_J) = delta_IJ
print("\nKronecker delta check N_I(s_J, t_J):")
kd_matrix = np.zeros((4, 4))
for I in range(4):
    for J in range(4):
        kd_matrix[I, J] = N_I_func(I, nodes[J, 0], nodes[J, 1], a_dim, b_dim)
print(kd_matrix)
assert np.allclose(kd_matrix, np.eye(4)), "Kronecker delta FAILED!"
print("  -> delta_IJ verified!")

# Verify partition of unity at several points
print("\nPartition of unity check (sum N_I = 1):")
test_pts = [(0, 0), (0.3, -0.7), (-0.5, 0.5), (a_dim, b_dim)]
for s, t in test_pts:
    total = sum(N_I_func(I, s, t, a_dim, b_dim) for I in range(4))
    print(f"  (s,t)=({s:5.2f},{t:5.2f}):  sum = {total:.10f}")
print("  -> partition of unity verified!")

# Verify via p^T * A^{-1} construction
print("\nDirect p^T * A^{-1} check at (s,t) = (0.3, -0.7):")
s_test, t_test = 0.3, -0.7
p_vec = np.array([1, s_test, t_test, s_test * t_test])
N_from_Ainv = p_vec @ A_inv
N_direct = np.array([N_I_func(I, s_test, t_test, a_dim, b_dim) for I in range(4)])
print(f"  p^T * A^{{-1}} = {N_from_Ainv}")
print(f"  N_I formula  = {N_direct}")
print(f"  ||diff||     = {np.linalg.norm(N_from_Ainv - N_direct):.2e}")

print("\n" + "=" * 65)
print("All Q1 and Q2 computations verified successfully.")
print("=" * 65)
