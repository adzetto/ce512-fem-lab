"""
CE 512 HW3 - OpenSees Validation for Question 1
1-D bar: fixed at x=0, loads P at x=L/2 and x=L.
Compare exact solution and Rayleigh-Ritz approximations with FEM.
"""

import openseespy.opensees as ops
import numpy as np

# =====================================================================
# Parameters (normalized: P=1, EA=1, L=1)
# =====================================================================
P = 1.0
E = 1.0
A = 1.0
L = 1.0

# =====================================================================
# OpenSees model: 1D bar with fine mesh
# =====================================================================
ops.wipe()
ops.model('basic', '-ndm', 1, '-ndf', 1)

n_elem = 20  # fine mesh for accuracy
dx = L / n_elem
mid_node = n_elem // 2 + 1  # node at x = L/2

# Create nodes
for i in range(n_elem + 1):
    ops.node(i + 1, i * dx)

# Fix left end
ops.fix(1, 1)

# Define material and elements (Truss element)
ops.uniaxialMaterial('Elastic', 1, E)
for i in range(n_elem):
    ops.element('Truss', i + 1, i + 1, i + 2, A, 1)

# Apply loads
ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)
ops.load(mid_node, P)           # P at x = L/2
ops.load(n_elem + 1, P)         # P at x = L

# Analysis
ops.system('BandSPD')
ops.numberer('RCM')
ops.constraints('Plain')
ops.integrator('LoadControl', 1.0)
ops.algorithm('Linear')
ops.analysis('Static')
ops.analyze(1)

# =====================================================================
# Extract results
# =====================================================================
print("=" * 70)
print("CE 512 HW3 - OpenSees Validation (1-D Bar)")
print("=" * 70)

# Displacement at all nodes
x_os = np.array([ops.nodeCoord(i + 1, 1) for i in range(n_elem + 1)])
u_os = np.array([ops.nodeDisp(i + 1, 1) for i in range(n_elem + 1)])

# Exact solution
def u_exact(x):
    return np.where(x <= L / 2, 2 * P * x / (E * A), P * (L / 2 + x) / (E * A))

# Rayleigh-Ritz Part (a): u = 15P/(16EAL) * x^2
def u_rr_a(x):
    a2 = 15 * P / (16 * E * A * L)
    return a2 * x**2

# Rayleigh-Ritz Part (b): u = 15P/(4EAL)*x^2 - 5P/(2EAL^2)*x^3
def u_rr_b(x):
    a2 = 15 * P / (4 * E * A * L)
    a3 = -5 * P / (2 * E * A * L**2)
    return a2 * x**2 + a3 * x**3

u_ex = u_exact(x_os)
u_a = u_rr_a(x_os)
u_b = u_rr_b(x_os)

# =====================================================================
# Displacement comparison at key points
# =====================================================================
key_x = [0.0, 0.25, 0.5, 0.75, 1.0]
print("\n--- Displacement Comparison (normalized: u * EA / (PL)) ---\n")
print(f"{'x/L':>5s}  {'Exact':>8s}  {'OpenSees':>8s}  {'Part(a)':>8s}  {'Part(b)':>8s}  {'FEM err':>8s}")
print("-" * 60)

for xv in key_x:
    idx = int(round(xv * n_elem))
    ue = u_exact(np.array([xv]))[0]
    uo = u_os[idx]
    ua = u_rr_a(np.array([xv]))[0]
    ub = u_rr_b(np.array([xv]))[0]
    err_fem = abs(uo - ue) / ue * 100 if ue > 0 else 0
    print(f"{xv:5.2f}  {ue:8.4f}  {uo:8.4f}  {ua:8.4f}  {ub:8.4f}  {err_fem:7.3f}%")

# =====================================================================
# Stress comparison
# =====================================================================
print("\n--- Stress Comparison (normalized: sigma * A / P) ---\n")
print(f"{'x/L':>5s}  {'Exact':>8s}  {'OpenSees':>8s}  {'Part(a)':>8s}  {'Part(b)':>8s}")
print("-" * 50)

# Stress from Rayleigh-Ritz
def sigma_rr_a(x):
    return 15 * P * x / (8 * A * L)

def sigma_rr_b(x):
    return 15 * P * x * (L - x) / (2 * A * L**2)

# Stress from OpenSees (element forces)
for xv in key_x:
    idx = int(round(xv * n_elem))
    # Exact stress
    if xv < 0.5:
        se = 2 * P / A
    elif xv > 0.5:
        se = P / A
    else:
        se = 2 * P / A  # left limit

    # OpenSees stress: from element axial force / A
    if idx > 0:
        elem_id = idx  # element to the left of node
        force = ops.eleResponse(elem_id, 'axialForce')
        if force:
            so = abs(force[0]) / A
        else:
            # Try basic force
            so = abs(ops.eleForce(elem_id, 1)) / A
    else:
        # At x=0, use first element
        force = ops.eleResponse(1, 'axialForce')
        if force:
            so = abs(force[0]) / A
        else:
            so = abs(ops.eleForce(1, 1)) / A

    sa = sigma_rr_a(np.array([xv]))[0]
    sb = sigma_rr_b(np.array([xv]))[0]
    print(f"{xv:5.2f}  {se:8.4f}  {so:8.4f}  {sa:8.4f}  {sb:8.4f}")

# =====================================================================
# Key values for LaTeX table
# =====================================================================
print("\n--- Values for LaTeX booktabs table ---\n")

# Displacements at x=L/2 and x=L
for xv, label in [(0.5, "L/2"), (1.0, "L")]:
    idx = int(round(xv * n_elem))
    ue = u_exact(np.array([xv]))[0]
    uo = u_os[idx]
    ua = u_rr_a(np.array([xv]))[0]
    ub = u_rr_b(np.array([xv]))[0]
    print(f"u({label}): exact={ue:.6f}, OpenSees={uo:.6f}, RR(a)={ua:.6f}, RR(b)={ub:.6f}")

# Check: OpenSees should match exact for this linear problem
max_err = np.max(np.abs(u_os - u_ex))
print(f"\nMax |u_FEM - u_exact| = {max_err:.2e}")
print("(Should be ~0 since FEM with linear elements is exact for piecewise-linear solutions)")

ops.wipe()
print("\nOpenSees validation complete.")
