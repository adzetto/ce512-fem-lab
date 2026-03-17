"""
CE 512 Homework 2 — Full Numerical Computation
4-element truss structure
"""
import math
import numpy as np

np.set_printoptions(precision=8, linewidth=200, suppress=False)

# ============================================================
# GIVEN DATA
# ============================================================
nodes_m = {1: (0.0, 0.0), 2: (2.5, 0.0), 3: (2.5, 3.0), 4: (4.5, 2.2)}
elements = {1: (1, 3), 2: (2, 3), 3: (2, 4), 4: (3, 4)}
E = 20000.0      # MPa  (N/mm^2)
A = 375.0        # mm^2
sigma_y = 300.0  # MPa
b = h = 40.0     # mm
t = 2.5          # mm
bi = b - 2.0*t
hi = h - 2.0*t
I_hollow = (b*h**3 - bi*hi**3) / 12.0

# Convert node coords to mm
nodes = {n: (x*1000, y*1000) for n, (x, y) in nodes_m.items()}

print("=" * 70)
print("  CE 512 HOMEWORK 2 — NUMERICAL RESULTS")
print("=" * 70)

# ============================================================
#  ELEMENT GEOMETRY
# ============================================================
print("\n1. ELEMENT GEOMETRY")
print("-" * 70)

elem_data = {}
for e, (ni, nj) in elements.items():
    xi, yi = nodes[ni]
    xj, yj = nodes[nj]
    dx = xj - xi
    dy = yj - yi
    Le = math.sqrt(dx**2 + dy**2)
    ce = dx / Le
    se = dy / Le
    theta = math.degrees(math.atan2(dy, dx))
    EAL = E * A / Le
    
    elem_data[e] = {
        'ni': ni, 'nj': nj,
        'dx': dx, 'dy': dy,
        'L': Le, 'c': ce, 's': se,
        'theta': theta, 'EAL': EAL
    }
    
    print(f"\n  Element {e}: Node {ni} -> Node {nj}")
    print(f"    Δx = {dx:.1f} mm,  Δy = {dy:.1f} mm")
    print(f"    L_{e} = √({dx:.1f}² + {dy:.1f}²) = {Le:.4f} mm")
    print(f"    c_{e} = cos θ_{e} = {dx:.1f}/{Le:.4f} = {ce:.10f}")
    print(f"    s_{e} = sin θ_{e} = {dy:.1f}/{Le:.4f} = {se:.10f}")
    print(f"    θ_{e} = {theta:.4f}°")
    print(f"    c²  = {ce**2:.10f}")
    print(f"    s²  = {se**2:.10f}")
    print(f"    cs  = {ce*se:.10f}")
    print(f"    EA/L_{e} = {E}×{A}/{Le:.4f} = {EAL:.6f} N/mm")

# ============================================================
#  ELEMENT STIFFNESS MATRICES (local)
# ============================================================
print("\n\n2. ELEMENT STIFFNESS MATRICES IN LOCAL COORDINATES")
print("-" * 70)
print("  k_local = (EA/L) * [1  -1; -1  1]")

# ============================================================
#  TRANSFORMATION (ROTATION) MATRICES
# ============================================================
print("\n\n3. TRANSFORMATION MATRICES  T_e = [c  s  0  0; 0  0  c  s]")
print("-" * 70)

for e, d in elem_data.items():
    c, s = d['c'], d['s']
    print(f"\n  Element {e} (θ = {d['theta']:.4f}°):")
    print(f"    T_{e} = [{c:12.8f}  {s:12.8f}  {0:12.8f}  {0:12.8f}]")
    print(f"          [{0:12.8f}  {0:12.8f}  {c:12.8f}  {s:12.8f}]")

# ============================================================
#  ELEMENT STIFFNESS IN GLOBAL COORDINATES
# ============================================================
print("\n\n4. ELEMENT STIFFNESS MATRICES IN GLOBAL COORDINATES")
print("-" * 70)
print("  K_e = (EA/L) * [c²   cs  -c²  -cs ]")
print("                 [cs   s²  -cs  -s² ]")
print("                 [-c²  -cs  c²   cs ]")
print("                 [-cs  -s²  cs   s² ]")

for e, d in elem_data.items():
    c, s, EAL = d['c'], d['s'], d['EAL']
    ke = EAL * np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s]
    ])
    d['ke'] = ke
    
    ni, nj = d['ni'], d['nj']
    dof_labels = [f'u{ni}', f'v{ni}', f'u{nj}', f'v{nj}']
    
    print(f"\n  Element {e} (Nodes {ni}-{nj}), EA/L_{e} = {EAL:.6f} N/mm:")
    print(f"    DOFs: {dof_labels}")
    for i in range(4):
        row = "    [" + "  ".join([f"{ke[i,j]:12.4f}" for j in range(4)]) + " ]"
        print(row)

# ============================================================
#  GLOBAL STIFFNESS ASSEMBLY
# ============================================================
print("\n\n5. GLOBAL STIFFNESS MATRIX ASSEMBLY")
print("-" * 70)

ndof = 8
K_global = np.zeros((ndof, ndof))

for e, d in elem_data.items():
    ni, nj = d['ni'], d['nj']
    ke = d['ke']
    dofs = [2*(ni-1), 2*(ni-1)+1, 2*(nj-1), 2*(nj-1)+1]
    d['global_dofs'] = dofs
    
    print(f"  Element {e}: local DOFs -> global DOFs {[x+1 for x in dofs]}")
    
    for r in range(4):
        for col in range(4):
            K_global[dofs[r], dofs[col]] += ke[r, col]

labels = ['u1', 'v1', 'u2', 'v2', 'u3', 'v3', 'u4', 'v4']
print(f"\n  Global K ({ndof}×{ndof}):")
header = "         " + "".join([f"{l:>12s}" for l in labels])
print(header)
for i in range(ndof):
    row = f"  {labels[i]:>5s}  " + "".join([f"{K_global[i,j]:12.4f}" for j in range(ndof)])
    print(row)

# ============================================================
#  BOUNDARY CONDITIONS & PARTITIONING
# ============================================================
print("\n\n6. BOUNDARY CONDITIONS AND PARTITIONING")
print("-" * 70)
print("  Fixed DOFs (pinned at N1 and N2):")
print("    u1 = v1 = u2 = v2 = 0  (DOFs 1,2,3,4)")
print("  Free DOFs: u3, v3, u4, v4  (DOFs 5,6,7,8)")

free_dofs  = [4, 5, 6, 7]   # 0-indexed
fixed_dofs = [0, 1, 2, 3]

K_ff = K_global[np.ix_(free_dofs, free_dofs)]
K_rf = K_global[np.ix_(fixed_dofs, free_dofs)]

free_labels = ['u3', 'v3', 'u4', 'v4']
print(f"\n  K_ff (4×4):")
header = "         " + "".join([f"{l:>14s}" for l in free_labels])
print(header)
for i in range(4):
    row = f"  {free_labels[i]:>5s}  " + "".join([f"{K_ff[i,j]:14.4f}" for j in range(4)])
    print(row)

# ============================================================
#  SOLVING FOR DISPLACEMENTS
# ============================================================
print("\n\n7. SOLVING FOR DISPLACEMENTS  K_ff · d_f = F_f")
print("-" * 70)

F_f = np.array([0.0, 0.0, 0.0, -1.0])  # per unit P
print("  Force vector (per unit P):")
print(f"    {{F_f}} = P × {{ 0, 0, 0, -1 }}^T")

K_ff_inv = np.linalg.inv(K_ff)
print(f"\n  K_ff^(-1) (×10^-4):")
for i in range(4):
    row = "    [" + "  ".join([f"{K_ff_inv[i,j]*1e4:14.6f}" for j in range(4)]) + " ]"
    print(row)

d_f = K_ff_inv @ F_f
print(f"\n  Nodal displacements (per unit P, mm/N):")
for i, l in enumerate(free_labels):
    print(f"    {l} = {d_f[i]:+.10e} × P  mm")

d_full = np.zeros(ndof)
d_full[free_dofs] = d_f

# ============================================================
#  ELEMENT STRAINS
# ============================================================
print("\n\n8. ELEMENT STRAINS")
print("-" * 70)
print("  ε_e = (1/L_e) [-c_e  -s_e  c_e  s_e] {d_e}")

strain_data = {}
for e, d in elem_data.items():
    ni, nj = d['ni'], d['nj']
    c, s, Le = d['c'], d['s'], d['L']
    
    idx_i = [2*(ni-1), 2*(ni-1)+1]
    idx_j = [2*(nj-1), 2*(nj-1)+1]
    
    ui, vi = d_full[idx_i[0]], d_full[idx_i[1]]
    uj, vj = d_full[idx_j[0]], d_full[idx_j[1]]
    
    strain = (-c*ui - s*vi + c*uj + s*vj) / Le
    stress = E * strain
    force  = stress * A
    
    strain_data[e] = {
        'strain': strain,
        'stress': stress,
        'force': force,
        'ui': ui, 'vi': vi, 'uj': uj, 'vj': vj
    }
    
    print(f"\n  Element {e} (Nodes {ni}-{nj}):")
    print(f"    d_e = [{ui:+.10e}, {vi:+.10e}, {uj:+.10e}, {vj:+.10e}]^T  (×P)")
    print(f"    ε_{e} = {strain:+.10e} × P")
    print(f"    σ_{e} = E·ε = {stress:+.10e} × P  MPa")
    print(f"    N_{e} = σ·A = {force:+.10e} × P  N")
    if force > 0:
        print(f"    --> TENSION")
    else:
        print(f"    --> COMPRESSION")

# ============================================================
#  REACTIONS
# ============================================================
print("\n\n9. SUPPORT REACTIONS (per unit P)")
print("-" * 70)

R = K_rf @ d_f
fixed_labels = ['R_x1', 'R_y1', 'R_x2', 'R_y2']
for i, l in enumerate(fixed_labels):
    print(f"  {l} = {R[i]:+.10f} × P  N")

print(f"\n  Equilibrium check:")
print(f"    ΣFx = {R[0]+R[2]:+.2e}")
print(f"    ΣFy = {R[1]+R[3]+(-1.0):+.2e}")
Mz1 = R[2]*0 + R[3]*2500 + 0*0 + (-1.0)*4500
print(f"    ΣM₁ = R_y2×2500 + (-P)×4500 = {R[3]*2500:.4f} + {(-1.0)*4500:.4f} = {Mz1:.6f}")

# ============================================================
#  PART (d): FIRST YIELD
# ============================================================
print("\n\n10. PART (d): FIRST YIELD")
print("-" * 70)
print(f"  σ_y = {sigma_y} MPa")
print(f"  |σ_e| = |stress_per_P| × P = σ_y  =>  P_y = σ_y / |stress_per_P|")

min_Py = float('inf')
yield_elem = 0
for e, sd in strain_data.items():
    s_per_P = abs(sd['stress'])
    if s_per_P > 1e-15:
        Py = sigma_y / s_per_P
        nature = "tension" if sd['force'] > 0 else "compression"
        print(f"  Element {e}: |σ/P| = {s_per_P:.10f} MPa/N,  P_y = {Py:.4f} N = {Py/1000:.4f} kN  ({nature})")
        if Py < min_Py:
            min_Py = Py
            yield_elem = e

print(f"\n  *** Element {yield_elem} yields first at P = {min_Py:.4f} N = {min_Py/1000:.4f} kN ***")

# ============================================================
#  PART (e): FIRST BUCKLING
# ============================================================
print("\n\n11. PART (e): FIRST BUCKLING (Euler, pinned-pinned)")
print("-" * 70)

# Cross-section: 40x40x2.5 mm hollow square section
print(f"  Hollow square section: {b:.1f}×{h:.1f} mm, thickness = {t:.1f} mm")
print(f"    A = {A:.4f} mm²")
print(f"    I = (b·h³ - b_i·h_i³)/12 = {I_hollow:.4f} mm⁴")
print(f"  Euler critical load: P_cr = π²EI/L²  (pinned-pinned, K=1)")

min_Pb = float('inf')
buckle_elem = 0
for e, d in elem_data.items():
    Le = d['L']
    sd = strain_data[e]
    F_per_P = sd['force']
    
    P_euler = math.pi**2 * E * I_hollow / Le**2
    
    if F_per_P < -1e-10:  # compression
        P_buckle = P_euler / abs(F_per_P)
        print(f"\n  Element {e} (COMPRESSION):")
        print(f"    L_{e} = {Le:.4f} mm")
        print(f"    P_cr,euler = π²×{E}×{I_hollow:.4f}/{Le:.4f}² = {P_euler:.4f} N")
        print(f"    |N_{e}/P| = {abs(F_per_P):.6f}")
        print(f"    P_buckle = P_cr / |N/P| = {P_euler:.4f} / {abs(F_per_P):.6f} = {P_buckle:.4f} N = {P_buckle/1000:.4f} kN")
        if P_buckle < min_Pb:
            min_Pb = P_buckle
            buckle_elem = e
    else:
        print(f"\n  Element {e}: TENSION (N/P = {F_per_P:+.6f}), no buckling possible")

print(f"\n  *** Element {buckle_elem} buckles first at P = {min_Pb:.4f} N = {min_Pb/1000:.4f} kN ***")

# ============================================================
#  NUMERICAL RESULTS FOR P = P_yield
# ============================================================
print("\n\n12. NUMERICAL DISPLACEMENTS AT P = P_yield")
print("-" * 70)
P_val = min_Py
print(f"  P = {P_val:.4f} N")
for i, l in enumerate(free_labels):
    print(f"    {l} = {d_f[i]*P_val:+.6f} mm")

print("\n  Stresses at P_yield:")
for e, sd in strain_data.items():
    print(f"    Element {e}: σ = {sd['stress']*P_val:+.4f} MPa, N = {sd['force']*P_val:+.4f} N")

print("\n\nDONE.")
