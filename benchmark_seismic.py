"""
Seismic Time History Benchmark: femlabpy vs OpenSeesPy vs CalculiX
==================================================================

Problem: 2D plane-stress column under horizontal earthquake excitation.
  - Height H = 8 m, Width W = 1 m, Thickness t = 1.0 m
  - E = 30 GPa, nu = 0.2, rho = 2500 kg/m^3
  - Fixed base, horizontal (X) excitation using the PEER BOL090.AT2 record

Three-way comparison:
  1. femlabpy   -- seismic_load (effective force) + solve_newmark (implicit)
  2. OpenSeesPy -- UniformExcitation pattern + Transient analysis
  3. CalculiX   -- *DLOAD with GRAV (equivalent body force) + *DYNAMIC
"""

import sys, os, time
import numpy as np

# ============================================================================
# PART 0: PARSE AT2 FILE
# ============================================================================


def read_at2(filepath):
    """Parse PEER NGA .AT2 file. Returns (dt, npts, accel_array_in_g)"""
    with open(filepath, "r") as f:
        lines = f.readlines()

    # 4th line contains NPTS and DT
    header4 = lines[3]
    parts = header4.replace(",", " ").split()
    npts = int(parts[1])
    dt = float(parts[3])

    # Read data from line 5 onwards
    accel = []
    for line in lines[4:]:
        vals = [float(v) for v in line.split()]
        accel.extend(vals)

    return dt, npts, np.array(accel[:npts])


eq_file = r"C:\Users\lenovo\Downloads\BOL090.AT2"
if not os.path.exists(eq_file):
    print(f"Error: Could not find AT2 file at {eq_file}")
    sys.exit(1)

dt_eq, npts_eq, accel_g = read_at2(eq_file)
g = 9.80665  # m/s^2
accel_ms2 = accel_g * g

print("=" * 72)
print("SEISMIC TIME HISTORY BENCHMARK")
print("=" * 72)
print(f"  Record: {os.path.basename(eq_file)}")
print(f"  Points: {npts_eq}, Time step: {dt_eq} s, Total time: {npts_eq * dt_eq:.2f} s")
print(f"  PGA: {np.max(np.abs(accel_g)):.4f} g")

# ============================================================================
# PART 1: PROBLEM PARAMETERS & MESH
# ============================================================================

W = 1.0  # width (m)
H = 8.0  # height (m)
thick = 1.0  # thickness (m)
E_val = 30e9  # Young's modulus (Pa)
nu_val = 0.2  # Poisson's ratio
rho_val = 2500.0  # density (kg/m^3)

nx, ny = 4, 32  # mesh divisions

print()
print("-" * 72)
print("PART 1: Building Gmsh Mesh (Column)")
print("-" * 72)

import gmsh

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("column")

p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
p2 = gmsh.model.geo.addPoint(W, 0.0, 0.0)
p3 = gmsh.model.geo.addPoint(W, H, 0.0)
p4 = gmsh.model.geo.addPoint(0.0, H, 0.0)

l1 = gmsh.model.geo.addLine(p1, p2)  # bottom
l2 = gmsh.model.geo.addLine(p2, p3)  # right
l3 = gmsh.model.geo.addLine(p3, p4)  # top
l4 = gmsh.model.geo.addLine(p4, p1)  # left

cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
surf = gmsh.model.geo.addPlaneSurface([cl])

gmsh.model.geo.synchronize()

# Transfinite (structured) meshing
gmsh.model.geo.mesh.setTransfiniteCurve(l1, nx + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(l3, nx + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(l2, ny + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(l4, ny + 1)
gmsh.model.geo.mesh.setTransfiniteSurface(surf)
gmsh.model.geo.mesh.setRecombine(2, surf)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)

mesh_path = os.path.join(
    os.path.dirname(__file__) if "__file__" in dir() else ".",
    "column_seismic.msh",
)
mesh_path = os.path.abspath(mesh_path)
gmsh.write(mesh_path)

node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
node_coords = node_coords.reshape(-1, 3)
node_map = {int(tag): i for i, tag in enumerate(node_tags)}
nn_gmsh = len(node_tags)

elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(dim=2)
quad_nodes_raw = None
for et, etags, enodes in zip(elem_types, elem_tags_list, elem_node_tags_list):
    etype_name = gmsh.model.mesh.getElementProperties(et)[0]
    if "Quadrilateral" in etype_name or et == 3:
        quad_nodes_raw = enodes.reshape(-1, 4)
        break

gmsh.finalize()

X_mesh = np.zeros((nn_gmsh, 2), dtype=float)
for tag, idx in node_map.items():
    row = np.where(node_tags == tag)[0][0]
    X_mesh[idx] = node_coords[row, :2]

nel_mesh = quad_nodes_raw.shape[0]
T_mesh_0 = np.zeros((nel_mesh, 4), dtype=int)
for e in range(nel_mesh):
    for j in range(4):
        T_mesh_0[e, j] = node_map[int(quad_nodes_raw[e, j])]

print(f"  Gmsh mesh saved: {mesh_path}")
print(f"  Nodes: {nn_gmsh}, Quads: {nel_mesh}")

tol = 1e-8
fixed_nodes_0 = np.where(np.abs(X_mesh[:, 1]) < tol)[0]  # y=0 (bottom)
top_nodes_0 = np.where(np.abs(X_mesh[:, 1] - H) < tol)[0]  # y=H (top)
top_mid_0 = top_nodes_0[len(top_nodes_0) // 2]  # top center node

print(f"  Fixed-base nodes: {len(fixed_nodes_0)}")
print(f"  Monitoring roof node: {top_mid_0} (y={X_mesh[top_mid_0, 1]} m)")

# Newmark params
beta, gamma = 0.25, 0.5

# ============================================================================
# PART 2A: OPENSEESPY REFERENCE SOLVER
# ============================================================================

print()
print("-" * 72)
print("PART 2A: OpenSeesPy Reference Solver")
print("-" * 72)

import openseespy.opensees as ops

ops.wipe()
ops.model("basic", "-ndm", 2, "-ndf", 2)

for i in range(nn_gmsh):
    ops.node(i + 1, float(X_mesh[i, 0]), float(X_mesh[i, 1]))

for n0 in fixed_nodes_0:
    ops.fix(int(n0 + 1), 1, 1)

mat_tag = 1
ops.nDMaterial("ElasticIsotropic", mat_tag, E_val, nu_val, rho_val)

t0_ops = time.perf_counter()
for e in range(nel_mesh):
    n1, n2, n3, n4 = T_mesh_0[e] + 1
    ops.element(
        "quad", e + 1, int(n1), int(n2), int(n3), int(n4), thick, "PlaneStress", mat_tag
    )

# Eigen analysis first
n_eigen = 3
eigenvalues = ops.eigen(n_eigen)
freq_ops = np.array([np.sqrt(ev) / (2.0 * np.pi) for ev in eigenvalues])
print(f"  OpenSees natural frequencies (Hz): {freq_ops}")

# Add 5% Rayleigh damping
from femlabpy.damping import rayleigh_coefficients

alphaM_ops, betaK_ops = rayleigh_coefficients(freq_ops[0], freq_ops[1], 0.05, 0.05)
ops.rayleigh(float(alphaM_ops), 0.0, float(betaK_ops), 0.0)

# Base excitation
ts_tag = 1
ops.timeSeries("Path", ts_tag, "-dt", dt_eq, "-values", *accel_ms2.tolist())
ops.pattern("UniformExcitation", 1, 1, "-accel", ts_tag)  # 1 = X direction

ops.system("BandGeneral")
ops.numberer("RCM")
ops.constraints("Plain")
ops.integrator("Newmark", gamma, beta)
ops.algorithm("Linear")
ops.analysis("Transient")

t_ops_hist = np.zeros(npts_eq + 1)
u_roof_ops = np.zeros(npts_eq + 1)

top_mid_tag = int(top_mid_0 + 1)

t0_dyn = time.perf_counter()
for step in range(npts_eq):
    ops.analyze(1, dt_eq)
    t_ops_hist[step + 1] = ops.getTime()
    u_roof_ops[step + 1] = ops.nodeDisp(top_mid_tag, 1)  # disp in X

ops_time = time.perf_counter() - t0_dyn
print(f"  OpenSees THA solve time ({npts_eq} steps): {ops_time:.4f} s")
print(f"  OpenSees roof max disp (X): {np.max(np.abs(u_roof_ops)):.6e} m")

ops.wipe()

# ============================================================================
# PART 2B: CALCULIX REFERENCE SOLVER
# ============================================================================

print()
print("-" * 72)
print("PART 2B: CalculiX Reference Solver")
print("-" * 72)

inp_path = "column_seismic_ccx.inp"
with open(inp_path, "w") as f:
    f.write("*HEADING\nColumn Seismic THA\n")
    f.write("*NODE\n")
    for i in range(nn_gmsh):
        f.write(f"{i + 1}, {X_mesh[i, 0]}, {X_mesh[i, 1]}, 0.0\n")

    f.write("*ELEMENT, TYPE=CPS4, ELSET=EALL\n")
    for e in range(nel_mesh):
        n1, n2, n3, n4 = T_mesh_0[e] + 1
        f.write(f"{e + 1}, {n1}, {n2}, {n3}, {n4}\n")

    f.write("*MATERIAL, NAME=CONCRETE\n")
    f.write("*ELASTIC\n")
    f.write(f"{E_val}, {nu_val}\n")
    f.write("*DENSITY\n")
    f.write(f"{rho_val}\n")
    f.write(f"*DAMPING, ALPHA={alphaM_ops:.6f}, BETA={betaK_ops:.6f}\n")

    f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=CONCRETE\n")
    f.write(f"{thick}\n")

    f.write("*BOUNDARY\n")
    for n0 in fixed_nodes_0:
        f.write(f"{n0 + 1}, 1, 2, 0.0\n")

    f.write("*NSET, NSET=NTOPMID\n")
    f.write(f"{top_mid_0 + 1},\n")

    # Amplitude table (t, accel in g)
    f.write("*AMPLITUDE, NAME=EQACCEL\n")
    amp_lines = []
    for i in range(npts_eq):
        t_val = i * dt_eq
        a_val = accel_g[i]
        amp_lines.append(f"{t_val:.6f}, {a_val:.6e}")
        if len(amp_lines) == 4:
            f.write(", ".join(amp_lines) + "\n")
            amp_lines = []
    if len(amp_lines) > 0:
        f.write(", ".join(amp_lines) + "\n")

    # Dynamic step (DIRECT forces fixed step)
    f.write("*STEP, INC=100000\n")
    f.write(f"*DYNAMIC, DIRECT, ALPHA=0.0, BETA={beta}, GAMMA={gamma}\n")
    f.write(f"{dt_eq}, {npts_eq * dt_eq}, {dt_eq}, {dt_eq}\n")

    # Apply effective seismic load as body force: GRAV in X direction.
    # The effective force is p = -m * ag.
    # GRAV magnitude is g=9.80665. Direction is (-1, 0, 0)
    f.write("*DLOAD, AMPLITUDE=EQACCEL\n")
    f.write(f"EALL, GRAV, {g}, -1., 0., 0.\n")

    f.write("*NODE FILE\nU\n")
    f.write("*NODE PRINT, NSET=NTOPMID\nU\n")
    f.write("*END STEP\n")

import subprocess

t0_ccx = time.perf_counter()

u_roof_ccx = np.zeros(npts_eq + 1)
try:
    subprocess.run(["ccx", "-i", "column_seismic_ccx"], check=True, capture_output=True)
    ccx_time = time.perf_counter() - t0_ccx
    print(f"  CalculiX THA solve time: {ccx_time:.4f} s")

    # Parse the .dat file
    ccx_dat = "column_seismic_ccx.dat"
    with open(ccx_dat, "r") as f:
        lines = f.readlines()

    step_idx = 1
    for line in lines:
        if "displacements (vx,vy,vz)" in line:
            continue
        parts = line.split()
        if len(parts) == 4 and parts[0].isdigit():
            node_id = int(parts[0])
            if node_id == top_mid_0 + 1:
                if step_idx <= npts_eq:
                    u_roof_ccx[step_idx] = float(parts[1])  # Ux
                    step_idx += 1

    print(f"  CalculiX roof max disp (X): {np.max(np.abs(u_roof_ccx)):.6e} m")
except Exception as e:
    print(f"  Warning: Failed to run/parse CalculiX THA. Error: {e}")

# ============================================================================
# PART 2C: FEMLABPY SOLVER
# ============================================================================

print()
print("-" * 72)
print("PART 2C: femlabpy Solver")
print("-" * 72)

from femlabpy import (
    kq4e,
    solve_newmark,
    solve_modal,
)
from femlabpy.elements.quads import mq4e
from femlabpy.io import load_gmsh
from femlabpy.dynamics import seismic_load

# Load mesh
mesh = load_gmsh(mesh_path)
T_fl = mesh.quads.astype(int)  # 1-based
X_fl = mesh.positions[:, :2]
nn_fl = X_fl.shape[0]
dof_fl = 2
ndof_fl = nn_fl * dof_fl

G_fl = np.array([[E_val, nu_val, 1.0, thick, rho_val]])

t0_asm = time.perf_counter()
K_fl = np.zeros((ndof_fl, ndof_fl), dtype=float)
M_fl = np.zeros((ndof_fl, ndof_fl), dtype=float)
K_fl = kq4e(K_fl, T_fl, X_fl, G_fl)
M_fl = mq4e(M_fl, T_fl, X_fl, G_fl)
print(f"  femlabpy assembly time: {time.perf_counter() - t0_asm:.4f} s")

# BCs: fix base (y ~ 0)
C_bc_rows = []
fixed_nodes_fl = np.where(np.abs(X_fl[:, 1]) < tol)[0] + 1
for n in fixed_nodes_fl:
    C_bc_rows.append([n, 1, 0.0])
    C_bc_rows.append([n, 2, 0.0])
C_bc_fl = np.array(C_bc_rows, dtype=float)

# Modal solve
modal_fl = solve_modal(K_fl, M_fl, n_modes=3, C_bc=C_bc_fl, dof=dof_fl)
print(f"  femlabpy natural frequencies (Hz): {modal_fl.freq_hz}")

# Add 5% Rayleigh damping
from femlabpy.damping import rayleigh_damping, rayleigh_coefficients

a_fl, b_fl = rayleigh_coefficients(modal_fl.freq_hz[0], modal_fl.freq_hz[1], 0.05, 0.05)
C_damp_fl = rayleigh_damping(M_fl, K_fl, float(a_fl), float(b_fl))

# Seismic Load
# Build influence vector: 1 for X-DOFs, 0 for Y-DOFs
inf_vec = np.zeros(ndof_fl)
inf_vec[0::2] = 1.0

p_eff = seismic_load(M_fl, inf_vec, accel_ms2, dt_eq)

u0_fl = np.zeros((ndof_fl, 1))
v0_fl = np.zeros((ndof_fl, 1))

t0_fl = time.perf_counter()
result_fl = solve_newmark(
    M_fl,
    C_damp_fl,
    K_fl,
    p_eff,
    u0_fl,
    v0_fl,
    dt=dt_eq,
    nsteps=npts_eq,
    beta=beta,
    gamma=gamma,
    C_bc=C_bc_fl,
    dof=dof_fl,
)
fl_time = time.perf_counter() - t0_fl
print(f"  femlabpy THA solve time ({npts_eq} steps): {fl_time:.4f} s")

# Extract roof displacement (X)
top_mid_fl_dof = top_mid_0 * 2  # X-DOF
u_roof_fl = result_fl.u[:, top_mid_fl_dof]
print(f"  femlabpy roof max disp (X): {np.max(np.abs(u_roof_fl)):.6e} m")

# ============================================================================
# PART 3: COMPARISON
# ============================================================================

print()
print("=" * 72)
print("TIME HISTORY ANALYSIS COMPARISON")
print("=" * 72)

max_ops = np.max(np.abs(u_roof_ops))
max_ccx = np.max(np.abs(u_roof_ccx))
max_fl = np.max(np.abs(u_roof_fl))

print(f"  Max absolute roof displacement (X):")
print(f"    OpenSeesPy : {max_ops:.6e} m")
print(f"    CalculiX   : {max_ccx:.6e} m")
print(f"    femlabpy   : {max_fl:.6e} m")

# RMS differences
rms_os_fl = np.sqrt(np.mean((u_roof_ops - u_roof_fl) ** 2))
rms_ccx_fl = np.sqrt(np.mean((u_roof_ccx - u_roof_fl) ** 2))
os_norm = np.sqrt(np.mean(u_roof_ops**2))
ccx_norm = np.sqrt(np.mean(u_roof_ccx**2))

print()
print(f"  Time History Agreement (RMS over {npts_eq} steps):")
print(
    f"    femlabpy vs OpenSeesPy : {rms_os_fl:.4e} m ({rms_os_fl / os_norm * 100:.6f}%)"
)
if ccx_norm > 0:
    print(
        f"    femlabpy vs CalculiX   : {rms_ccx_fl:.4e} m ({rms_ccx_fl / ccx_norm * 100:.6f}%)"
    )
print("=" * 72)
