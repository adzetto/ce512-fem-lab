"""
CE 512 HW3 - FemLab Validation for Q1 (1-D bar) and Q2 (rectangular element)
Uses FemLab-Python from C:\\Users\\lenovo\\Documents\\6_FemLab-Python
"""

import sys
sys.path.insert(0, r"C:\Users\lenovo\Documents\6_FemLab-Python\src")

import numpy as np
import femlab

# =====================================================================
# QUESTION 1: 1-D Bar Validation
# =====================================================================
print("=" * 70)
print("CE 512 HW3 - FemLab Validation")
print("=" * 70)

# Parameters (normalized)
P_val = 1.0
E_val = 1.0
A_val = 1.0
L_val = 1.0
n_elem = 20
dx = L_val / n_elem
n_nodes = n_elem + 1

print(f"\nQ1: 1-D Bar with {n_elem} elements, dx = {dx}")

# Node coordinates (1D problem modeled as 2D with y=0)
X = np.zeros((n_nodes, 1), dtype=float)
for i in range(n_nodes):
    X[i, 0] = i * dx

# Topology: [node1, node2, material_group] (1-based indexing)
T = np.zeros((n_elem, 3), dtype=int)
for i in range(n_elem):
    T[i] = [i + 1, i + 2, 1]

# Material properties: [A, E]
G = np.array([[A_val, E_val]], dtype=float)

# Initialize
K, p, q = femlab.init(n_nodes, 1, use_sparse=False)

# Assemble stiffness
K = femlab.kbar(K, T, X, G)

# Apply loads
mid_node = n_elem // 2 + 1  # node at x = L/2
P_loads = np.array([
    [mid_node, P_val],
    [n_nodes, P_val]
], dtype=float)
p = femlab.setload(p, P_loads)

# Apply boundary conditions: fix node 1 (dof 1)
C = np.array([[1, 1, 0.0]], dtype=float)
K, p, _ = femlab.setbc(K, p, C, 1)

# Solve
u = np.linalg.solve(K, p)

# Results
x_arr = X[:, 0]
u_arr = u[:, 0]

# Exact solution
def u_exact(x):
    return np.where(x <= L_val / 2, 2 * P_val * x / (E_val * A_val),
                    P_val * (L_val / 2 + x) / (E_val * A_val))

def u_rr_a(x):
    return (15 * P_val / (16 * E_val * A_val * L_val)) * x**2

def u_rr_b(x):
    a2 = 15 * P_val / (4 * E_val * A_val * L_val)
    a3 = -5 * P_val / (2 * E_val * A_val * L_val**2)
    return a2 * x**2 + a3 * x**3

u_ex = u_exact(x_arr)

print("\n--- Q1 Displacement Comparison ---\n")
print(f"{'x/L':>5s}  {'Exact':>8s}  {'FemLab':>8s}  {'Part(a)':>8s}  {'Part(b)':>8s}  {'FEM err':>8s}")
print("-" * 60)

key_x = [0.0, 0.25, 0.5, 0.75, 1.0]
for xv in key_x:
    idx = int(round(xv * n_elem))
    ue = u_exact(np.array([xv]))[0]
    uf = u_arr[idx]
    ua = u_rr_a(np.array([xv]))[0]
    ub = u_rr_b(np.array([xv]))[0]
    err_fem = abs(uf - ue) / ue * 100 if ue > 0 else 0
    print(f"{xv:5.2f}  {ue:8.4f}  {uf:8.4f}  {ua:8.4f}  {ub:8.4f}  {err_fem:7.3f}%")

max_err = np.max(np.abs(u_arr - u_ex))
print(f"\nMax |u_FemLab - u_exact| = {max_err:.2e}")

# =====================================================================
# QUESTION 2: Rectangular Element Shape Function Verification
# =====================================================================
print("\n" + "=" * 70)
print("Q2: Rectangular Element Shape Functions (FemLab keq4e)")
print("=" * 70)

# Create a single Q4 element centered at origin, size 2a x 2b
a_dim = 1.0
b_dim = 1.0

# Nodes: 4 corners (1-based in FemLab), CCW from bottom-left
X_q = np.array([
    [-a_dim, -b_dim],
    [ a_dim, -b_dim],
    [ a_dim,  b_dim],
    [-a_dim,  b_dim]
], dtype=float)

# Topology: [n1, n2, n3, n4, material_group]
T_q = np.array([[1, 2, 3, 4, 1]], dtype=int)

# Material: [E, nu, thickness] for plane stress
E_q = 100.0
nu_q = 0.3
h_q = 1.0
G_q = np.array([[E_q, nu_q, h_q]], dtype=float)

# Compute element stiffness directly using keq4e (element-level function)
# G format: [E, nu] for plane stress (default), [E, nu, 2] for plane strain
G_elem = np.array([E_q, nu_q], dtype=float)
K_q = femlab.keq4e(X_q, G_elem)

print(f"\nElement dimensions: 2a = {2*a_dim}, 2b = {2*b_dim}")
print(f"Material: E = {E_q}, nu = {nu_q}, h = {h_q}")

# Analytical stiffness for comparison
# For plane stress: C = E/(1-nu^2) * [[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]]
C_ps = (E_q / (1 - nu_q**2)) * np.array([
    [1,    nu_q, 0],
    [nu_q, 1,    0],
    [0,    0,    (1 - nu_q) / 2]
])

# Analytical shape function gradients and stiffness via 2x2 Gauss quadrature
xi_I = np.array([-1, +1, +1, -1])
eta_I = np.array([-1, -1, +1, +1])

def B_matrix(zeta, eta, a, b):
    """3x8 strain-displacement matrix for Q4 at (zeta, eta)."""
    B = np.zeros((3, 8))
    for I_node in range(4):
        dN_ds = xi_I[I_node] * (1 + eta_I[I_node] * eta) / (4 * a)
        dN_dt = eta_I[I_node] * (1 + xi_I[I_node] * zeta) / (4 * b)
        col = 2 * I_node
        B[0, col]     = dN_ds
        B[1, col + 1] = dN_dt
        B[2, col]     = dN_dt
        B[2, col + 1] = dN_ds
    return B

# 2x2 Gauss quadrature
gp = 1 / np.sqrt(3)
gauss_pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
K_analytical = np.zeros((8, 8))
for zeta_g, eta_g in gauss_pts:
    B = B_matrix(zeta_g, eta_g, a_dim, b_dim)
    K_analytical += B.T @ C_ps @ B * a_dim * b_dim * h_q  # w=1 for each Gauss point

# Compare
diff = np.linalg.norm(K_q - K_analytical)
print(f"\n||K_FemLab - K_analytical|| = {diff:.2e}")
print(f"||K_FemLab|| = {np.linalg.norm(K_q):.4f}")
print(f"Relative error = {diff / np.linalg.norm(K_q):.2e}")

# Eigenvalue check: rank should be 5 (3 rigid body modes)
eigvals = np.linalg.eigvalsh(K_analytical)
print(f"\nEigenvalues of K (8x8):")
for i, ev in enumerate(eigvals):
    print(f"  lambda_{i+1} = {ev:12.6f}  {'(zero - RBM)' if abs(ev) < 1e-8 else ''}")

n_zero = np.sum(np.abs(eigvals) < 1e-8)
n_pos = np.sum(eigvals > 1e-8)
print(f"\nZero eigenvalues (RBMs): {n_zero}")
print(f"Positive eigenvalues: {n_pos}")
print(f"Rank = {n_pos} (expected 5)")

# Verify partition of unity: apply uniform displacement d = [1,0,1,0,1,0,1,0]
# Should give zero internal force (rigid body translation)
d_rigid = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
f_rigid = K_analytical @ d_rigid
print(f"\nRigid body check: ||K * d_rigid|| = {np.linalg.norm(f_rigid):.2e} (should be ~0)")

print("\n" + "=" * 70)
print("FemLab validation complete.")
print("=" * 70)
