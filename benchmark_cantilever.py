"""
Cantilever Beam Benchmark: femlabpy vs OpenSeesPy vs Analytical
================================================================

Problem: Cantilever beam (plane stress Q4 elements)
  - Length L = 4 m, Height H = 0.5 m, Thickness t = 0.1 m
  - E = 210 GPa, nu = 0.3, rho = 7850 kg/m^3
  - Fixed left end, tip load P = -1000 N (downward) at right end

Three-way comparison:
  1. femlabpy   -- uses load_gmsh + kq4e/mq4e + solve_newmark + solve_modal
  2. OpenSeesPy -- independent FEM tool (quad elements, eigen, Newmark)
  3. Analytical  -- Euler-Bernoulli beam theory closed-form solutions

Validates:
  - Static: tip deflection
  - Modal: first 3 natural frequencies
  - Dynamic: tip displacement time history under suddenly-applied load
"""

import sys, os, time
import numpy as np

# ============================================================================
# PART 0: PROBLEM PARAMETERS
# ============================================================================

L = 4.0  # beam length (m)
H = 0.5  # beam height (m)
thick = 0.1  # thickness (m)
E_val = 210e9  # Young's modulus (Pa)
nu_val = 0.3  # Poisson's ratio
rho_val = 7850.0  # density (kg/m^3)
P_tip = -1000.0  # tip load (N, downward)
nx, ny = 32, 4  # mesh divisions

I_beam = thick * H**3 / 12.0  # second moment of area
A_beam = H * thick  # cross-section area
EI = E_val * I_beam
rhoA = rho_val * A_beam

print("=" * 72)
print("CANTILEVER BEAM BENCHMARK")
print("=" * 72)
print(f"  L={L} m, H={H} m, t={thick} m")
print(f"  E={E_val:.3e} Pa, nu={nu_val}, rho={rho_val} kg/m^3")
print(f"  I={I_beam:.6e} m^4, A={A_beam:.4e} m^2, EI={EI:.6e} N.m^2")
print(f"  Mesh: {nx}x{ny} Q4 elements ({(nx + 1) * (ny + 1)} nodes)")
print(f"  Tip load: P = {P_tip} N")
print()

# ============================================================================
# PART 1: BUILD GMSH MESH
# ============================================================================

print("-" * 72)
print("PART 1: Building Gmsh Mesh")
print("-" * 72)

import gmsh

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("cantilever")

# Define geometry
p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
p2 = gmsh.model.geo.addPoint(L, 0.0, 0.0)
p3 = gmsh.model.geo.addPoint(L, H, 0.0)
p4 = gmsh.model.geo.addPoint(0.0, H, 0.0)

l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
surf = gmsh.model.geo.addPlaneSurface([cl])

# Physical groups
gmsh.model.geo.synchronize()
gmsh.model.addPhysicalGroup(2, [surf], tag=1, name="beam")
gmsh.model.addPhysicalGroup(1, [l4], tag=2, name="fixed_end")
gmsh.model.addPhysicalGroup(1, [l2], tag=3, name="loaded_end")

# Transfinite (structured) meshing for Q4
gmsh.model.geo.mesh.setTransfiniteCurve(l1, nx + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(l3, nx + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(l2, ny + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(l4, ny + 1)
gmsh.model.geo.mesh.setTransfiniteSurface(surf)
gmsh.model.geo.mesh.setRecombine(2, surf)  # quads

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)

mesh_path = os.path.join(
    os.path.dirname(__file__) if "__file__" in dir() else ".",
    "cantilever_benchmark.msh",
)
mesh_path = os.path.abspath(mesh_path)
gmsh.write(mesh_path)

# Extract mesh data directly from gmsh API
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
node_coords = node_coords.reshape(-1, 3)
node_map = {int(tag): i for i, tag in enumerate(node_tags)}
nn_gmsh = len(node_tags)

# Get quad elements
elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(dim=2)
quad_nodes_raw = None
for et, etags, enodes in zip(elem_types, elem_tags_list, elem_node_tags_list):
    etype_name = gmsh.model.mesh.getElementProperties(et)[0]
    if "Quadrilateral" in etype_name or et == 3:
        quad_nodes_raw = enodes.reshape(-1, 4)
        break

gmsh.finalize()

assert quad_nodes_raw is not None, "No quad elements found!"

# Build X (coordinates, 0-based index)
X_mesh = np.zeros((nn_gmsh, 2), dtype=float)
for tag, idx in node_map.items():
    row = np.where(node_tags == tag)[0][0]
    X_mesh[idx] = node_coords[row, :2]

# Build T (topology, 0-based nodes)
nel_mesh = quad_nodes_raw.shape[0]
T_mesh_0 = np.zeros((nel_mesh, 4), dtype=int)
for e in range(nel_mesh):
    for j in range(4):
        T_mesh_0[e, j] = node_map[int(quad_nodes_raw[e, j])]

print(f"  Gmsh mesh saved: {mesh_path}")
print(f"  Nodes: {nn_gmsh}, Quads: {nel_mesh}")

# Identify BC nodes from coordinates
tol = 1e-8
fixed_nodes_0 = np.where(np.abs(X_mesh[:, 0]) < tol)[0]  # x=0 (left)
tip_nodes_0 = np.where(np.abs(X_mesh[:, 0] - L) < tol)[0]  # x=L (right)

print(f"  Fixed-end nodes: {len(fixed_nodes_0)}, Tip nodes: {len(tip_nodes_0)}")

# ============================================================================
# PART 2A: ANALYTICAL SOLUTIONS (Euler-Bernoulli Beam Theory)
# ============================================================================

print()
print("-" * 72)
print("PART 2A: Analytical Solutions (Euler-Bernoulli)")
print("-" * 72)

# Static: tip deflection under point load P at free end
delta_analytical = P_tip * L**3 / (3.0 * EI)  # negative = downward
print(f"  Static tip deflection: {delta_analytical:.6e} m")

# Modal: cantilever natural frequencies
beta_L_vals = [1.87510, 4.69409, 7.85476, 10.99554, 14.13717]
freq_analytical = []
for bl in beta_L_vals:
    omega_n = bl**2 * np.sqrt(EI / (rhoA * L**4))
    f_n = omega_n / (2.0 * np.pi)
    freq_analytical.append(f_n)
freq_analytical = np.array(freq_analytical)
print(f"  Natural frequencies (Hz): {freq_analytical[:3]}")

omega_1_analytical = beta_L_vals[0] ** 2 * np.sqrt(EI / (rhoA * L**4))
print(f"  omega_1 = {omega_1_analytical:.4f} rad/s")

# ============================================================================
# PART 2B: CALCULIX REFERENCE SOLVER (Static)
# ============================================================================

print()
print("-" * 72)
print("PART 2B: CalculiX Reference Solver")
print("-" * 72)

inp_path = "cantilever_calculix.inp"
with open(inp_path, "w") as f:
    f.write("*HEADING\nCantilever benchmark\n")
    f.write("*NODE\n")
    for i in range(nn_gmsh):
        f.write(f"{i + 1}, {X_mesh[i, 0]}, {X_mesh[i, 1]}, 0.0\n")

    f.write("*ELEMENT, TYPE=CPS4, ELSET=EALL\n")
    for e in range(nel_mesh):
        n1, n2, n3, n4 = T_mesh_0[e] + 1
        f.write(f"{e + 1}, {n1}, {n2}, {n3}, {n4}\n")

    f.write("*MATERIAL, NAME=STEEL\n")
    f.write("*ELASTIC\n")
    f.write(f"{E_val}, {nu_val}\n")
    f.write("*DENSITY\n")
    f.write(f"{rho_val}\n")

    f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL\n")
    f.write(f"{thick}\n")

    f.write("*BOUNDARY\n")
    for n0 in fixed_nodes_0:
        f.write(f"{n0 + 1}, 1, 2, 0.0\n")

    f.write("*STEP\n")
    f.write("*STATIC\n")
    f.write("*CLOAD\n")
    load_per_node_ccx = P_tip / len(tip_nodes_0)
    for n0 in tip_nodes_0:
        f.write(f"{n0 + 1}, 2, {load_per_node_ccx}\n")

    f.write("*NSET, NSET=NTIP\n")
    for n0 in tip_nodes_0:
        f.write(f"{n0 + 1},\n")

    f.write("*NODE PRINT, NSET=NTIP\nU\n")
    f.write("*END STEP\n")

import subprocess

t0_ccx = time.perf_counter()
try:
    # Run ccx (assume it is in PATH from conda environment)
    # The command is usually 'ccx -i jobname' where jobname is without .inp
    subprocess.run(
        ["ccx", "-i", "cantilever_calculix"], check=True, capture_output=True
    )
    ccx_solve_time = time.perf_counter() - t0_ccx

    # Parse the .dat file
    delta_ccx_list = []
    with open("cantilever_calculix.dat", "r") as f:
        lines = f.readlines()
        in_disp_block = False
        for line in lines:
            if "displacements (vx,vy,vz)" in line:
                in_disp_block = True
                continue
            if in_disp_block:
                if line.strip() == "":
                    continue
                parts = line.split()
                if len(parts) == 4 and parts[0].isdigit():
                    # Format: node, ux, uy, uz
                    node_id = int(parts[0])
                    if node_id - 1 in tip_nodes_0:
                        delta_ccx_list.append(float(parts[2]))

    if len(delta_ccx_list) > 0:
        delta_ccx = np.mean(delta_ccx_list)
        print(f"  Static solve time: {ccx_solve_time:.4f} s")
        print(f"  Static tip deflection: {delta_ccx:.6e} m")
    else:
        print("  Warning: Failed to parse displacements from CalculiX .dat file")
        delta_ccx = float("nan")
except Exception as e:
    print(f"  Warning: Failed to run CalculiX. Make sure it is installed. Error: {e}")
    delta_ccx = float("nan")

# Run CalculiX for Modal Analysis
inp_modal_path = "cantilever_calculix_modal.inp"
with open(inp_modal_path, "w") as f:
    f.write("*HEADING\nCantilever modal benchmark\n")
    f.write("*NODE\n")
    for i in range(nn_gmsh):
        f.write(f"{i + 1}, {X_mesh[i, 0]}, {X_mesh[i, 1]}, 0.0\n")

    f.write("*ELEMENT, TYPE=CPS4, ELSET=EALL\n")
    for e in range(nel_mesh):
        n1, n2, n3, n4 = T_mesh_0[e] + 1
        f.write(f"{e + 1}, {n1}, {n2}, {n3}, {n4}\n")

    f.write("*MATERIAL, NAME=STEEL\n")
    f.write("*ELASTIC\n")
    f.write(f"{E_val}, {nu_val}\n")
    f.write("*DENSITY\n")
    f.write(f"{rho_val}\n")

    f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL\n")
    f.write(f"{thick}\n")

    f.write("*BOUNDARY\n")
    for n0 in fixed_nodes_0:
        f.write(f"{n0 + 1}, 1, 2, 0.0\n")

    f.write("*STEP\n")
    f.write("*FREQUENCY\n")
    f.write("5\n")
    f.write("*END STEP\n")

t0_ccx_modal = time.perf_counter()
freq_ccx = []
try:
    subprocess.run(
        ["ccx", "-i", "cantilever_calculix_modal"], check=True, capture_output=True
    )
    ccx_modal_time = time.perf_counter() - t0_ccx_modal

    with open("cantilever_calculix_modal.dat", "r") as f:
        lines = f.readlines()
        in_freq_block = False
        for line in lines:
            if "E I G E N V A L U E   O U T P U T" in line:
                in_freq_block = True
                continue
            if in_freq_block:
                if "P A R T I C I P A T I O N   F A C T O R S" in line:
                    break
                if (
                    "MODE NO" in line
                    or "(RAD/TIME)" in line
                    or "REAL PART" in line
                    or line.strip() == ""
                ):
                    continue
                parts = line.split()
                if len(parts) >= 4 and parts[0].isdigit():
                    # Format: 1 0.2727151E+05 0.1651409E+03 0.2628299E+02 0.0000000E+00
                    # CYCLES/TIME is index 3
                    freq_ccx.append(float(parts[3]))
                if len(freq_ccx) == 5:
                    break
    if len(freq_ccx) > 0:
        print(f"  Modal solve time: {ccx_modal_time:.4f} s")
        print(f"  Natural frequencies (Hz): {freq_ccx[:3]}")
    else:
        print("  Warning: Failed to parse frequencies from CalculiX .dat file")
except Exception as e:
    print(f"  Warning: Failed to run CalculiX modal. Error: {e}")

# ============================================================================
# PART 2C: OPENSEESPY REFERENCE SOLVER
# ============================================================================

print()
print("-" * 72)
print("PART 2B: OpenSeesPy Reference Solver")
print("-" * 72)

import openseespy.opensees as ops

ops.wipe()
ops.model("basic", "-ndm", 2, "-ndf", 2)

# -- Create nodes (1-based tags in OpenSees) --
for i in range(nn_gmsh):
    ops.node(i + 1, float(X_mesh[i, 0]), float(X_mesh[i, 1]))

# -- Fix left-end nodes --
for n0 in fixed_nodes_0:
    ops.fix(int(n0 + 1), 1, 1)

# -- Material: nDMaterial ElasticIsotropic --
mat_tag = 1
ops.nDMaterial("ElasticIsotropic", mat_tag, E_val, nu_val, rho_val)

# -- Build Q4 elements (quad, PlaneStress, thickness) --
t0_ops = time.perf_counter()
for e in range(nel_mesh):
    n1, n2, n3, n4 = T_mesh_0[e] + 1  # 1-based
    ops.element(
        "quad", e + 1, int(n1), int(n2), int(n3), int(n4), thick, "PlaneStress", mat_tag
    )
ops_assembly_time = time.perf_counter() - t0_ops
print(f"  Model build time: {ops_assembly_time:.4f} s")

# ---- STATIC ANALYSIS ----
ops.timeSeries("Linear", 1)
ops.pattern("Plain", 1, 1)
load_per_node_ops = P_tip / len(tip_nodes_0)
for n0 in tip_nodes_0:
    ops.load(int(n0 + 1), 0.0, float(load_per_node_ops))

ops.system("BandGeneral")
ops.numberer("RCM")
ops.constraints("Plain")
ops.integrator("LoadControl", 1.0)
ops.algorithm("Linear")
ops.analysis("Static")
ops.analyze(1)

# Tip deflection
tip_y_ops = []
for n0 in tip_nodes_0:
    tip_y_ops.append(ops.nodeDisp(int(n0 + 1), 2))
delta_ops = np.mean(tip_y_ops)
print(f"  Static tip deflection: {delta_ops:.6e} m")

# ---- MODAL ANALYSIS ----
n_eigen = 5
t0_eig = time.perf_counter()
eigenvalues = ops.eigen(n_eigen)
ops_modal_time = time.perf_counter() - t0_eig

freq_ops = np.array([np.sqrt(ev) / (2.0 * np.pi) for ev in eigenvalues])
print(f"  Modal solve time: {ops_modal_time:.4f} s")
print(f"  Natural frequencies (Hz): {freq_ops[:3]}")

# ---- DYNAMIC ANALYSIS (Newmark, average acceleration) ----
# Reset for transient
ops.wipeAnalysis()
ops.loadConst("-time", 0.0)

# Remove the static load pattern, apply step load as constant in transient
ops.timeSeries("Constant", 2)
ops.pattern("Plain", 2, 2)
for n0 in tip_nodes_0:
    ops.load(int(n0 + 1), 0.0, float(load_per_node_ops))

dt_dyn = 0.0001
T_end = 0.05
nsteps_dyn = int(T_end / dt_dyn)

ops.system("BandGeneral")
ops.numberer("RCM")
ops.constraints("Plain")
ops.integrator("Newmark", 0.5, 0.25)  # gamma=0.5, beta=0.25 (avg accel)
ops.algorithm("Linear")
ops.analysis("Transient")

# Pick a tip mid-node for monitoring
tip_mid_0 = tip_nodes_0[len(tip_nodes_0) // 2]
tip_mid_tag = int(tip_mid_0 + 1)

t_ops_hist = np.zeros(nsteps_dyn + 1)
u_tip_ops_hist = np.zeros(nsteps_dyn + 1)
# At t=0, displacement from static analysis is already present;
# we want to track incremental from zero. Record the static offset.
u_static_offset = ops.nodeDisp(tip_mid_tag, 2)

t0_dyn = time.perf_counter()
for step in range(nsteps_dyn):
    ops.analyze(1, dt_dyn)
    t_ops_hist[step + 1] = ops.getTime()
    # The transient displacement includes the static offset from the first
    # load pattern (loadConst). The "step load" dynamic problem starts from
    # rest with load applied at t=0. Since we used loadConst, the static
    # displacement is frozen. The new pattern applies the same load again,
    # so the total dynamic displacement = nodeDisp - static_offset gives
    # the transient part only. But actually we want the FULL response to
    # the step load (static + dynamic), which is what the reference and
    # femlabpy compute. However loadConst freezes the static loads as
    # constant, and pattern 2 adds the SAME load again. So effectively
    # the structure sees 2x the load. Let me reconsider...
    # Actually: loadConst freezes the load but keeps displacement.
    # Pattern 2 adds another P. So total load = 2P after loadConst.
    # We need to subtract the static contribution and scale properly.
    # The simplest fix: record the full nodeDisp and subtract static offset.
    u_tip_ops_hist[step + 1] = ops.nodeDisp(tip_mid_tag, 2) - u_static_offset

ops_dynamic_time = time.perf_counter() - t0_dyn
print(f"  Dynamic solve time ({nsteps_dyn} steps): {ops_dynamic_time:.4f} s")
print(f"  Dynamic tip max deflection: {np.min(u_tip_ops_hist):.6e} m")

ops.wipe()

# ============================================================================
# PART 2C: FEMLABPY SOLVER
# ============================================================================

print()
print("-" * 72)
print("PART 2C: femlabpy Solver")
print("-" * 72)

from femlabpy import (
    init,
    kq4e,
    setbc,
    setload,
    solve_newmark,
    solve_modal,
    constant_load,
)
from femlabpy.elements.quads import mq4e
from femlabpy.io import load_gmsh

# Load mesh
mesh = load_gmsh(mesh_path)
T_fl = mesh.quads.astype(int)  # (nel, 5): [n1, n2, n3, n4, tag] 1-based
X_fl = mesh.positions[:, :2]  # (nn, 2)
nn_fl = X_fl.shape[0]
dof_fl = 2
ndof_fl = nn_fl * dof_fl

# Material: [E, nu, type(1=plane stress), thickness, rho]
G_fl = np.array([[E_val, nu_val, 1.0, thick, rho_val]])

# Assemble K and M
t0 = time.perf_counter()
K_fl = np.zeros((ndof_fl, ndof_fl), dtype=float)
M_fl = np.zeros((ndof_fl, ndof_fl), dtype=float)
K_fl = kq4e(K_fl, T_fl, X_fl, G_fl)
M_fl = mq4e(M_fl, T_fl, X_fl, G_fl)
fl_assembly_time = time.perf_counter() - t0
print(f"  Assembly time: {fl_assembly_time:.4f} s")

# BCs: fix left end (x ~ 0)
fixed_nodes_fl = np.where(np.abs(X_fl[:, 0]) < tol)[0] + 1  # 1-based
C_bc_rows = []
for n in fixed_nodes_fl:
    C_bc_rows.append([n, 1, 0.0])
    C_bc_rows.append([n, 2, 0.0])
C_bc_fl = np.array(C_bc_rows, dtype=float)

# Load: tip force distributed at x ~ L
tip_nodes_fl = np.where(np.abs(X_fl[:, 0] - L) < tol)[0] + 1  # 1-based
load_per_node_fl = P_tip / len(tip_nodes_fl)
P_rows = []
for n in tip_nodes_fl:
    P_rows.append([n, 0.0, load_per_node_fl])  # [node, Fx, Fy]
P_load_fl = np.array(P_rows, dtype=float)

# Static solve
K_fl2, p_fl, q_fl = init(nn_fl, dof_fl, use_sparse=False)
K_fl2 = kq4e(K_fl2, T_fl, X_fl, G_fl)
p_fl = setload(p_fl, P_load_fl)
K_bc, p_bc, _ = setbc(K_fl2, p_fl, C_bc_fl, dof_fl)
u_fl_static = np.linalg.solve(K_bc, p_bc)

# Tip deflection
tip_y_dofs_fl = [(n - 1) * 2 + 1 for n in tip_nodes_fl]
delta_fl = np.mean(u_fl_static[tip_y_dofs_fl])
print(f"  Static tip deflection: {delta_fl:.6e} m")

# Modal solve
t0 = time.perf_counter()
modal_fl = solve_modal(K_fl, M_fl, n_modes=5, C_bc=C_bc_fl, dof=dof_fl)
fl_modal_time = time.perf_counter() - t0
print(f"  Modal solve time: {fl_modal_time:.4f} s")
print(f"  Natural frequencies (Hz): {modal_fl.freq_hz[:3]}")

# Dynamic: suddenly-applied step load (Newmark)
p_dyn_fl = np.zeros((ndof_fl, 1), dtype=float)
for n in tip_nodes_fl:
    p_dyn_fl[(n - 1) * 2 + 1] = load_per_node_fl

C_damp_fl = np.zeros((ndof_fl, ndof_fl))
u0_fl = np.zeros((ndof_fl, 1))
v0_fl = np.zeros((ndof_fl, 1))

t0 = time.perf_counter()
result_fl = solve_newmark(
    M_fl,
    C_damp_fl,
    K_fl,
    constant_load(p_dyn_fl.ravel()),
    u0_fl,
    v0_fl,
    dt=dt_dyn,
    nsteps=nsteps_dyn,
    C_bc=C_bc_fl,
    dof=dof_fl,
    compute_energy=True,
)
fl_dynamic_time = time.perf_counter() - t0
print(f"  Dynamic solve time ({nsteps_dyn} steps): {fl_dynamic_time:.4f} s")

# Tip DOF in femlabpy
tip_mid_fl = tip_nodes_fl[len(tip_nodes_fl) // 2]
tip_dof_fl = (tip_mid_fl - 1) * 2 + 1
print(f"  Dynamic tip max deflection: {np.min(result_fl.u[:, tip_dof_fl]):.6e} m")

# ============================================================================
# PART 3: COMPARISON
# ============================================================================

print()
print("=" * 72)
print("COMPARISON RESULTS")
print("=" * 72)

# --- Static ---
print()
print("1. STATIC ANALYSIS -- Tip Deflection under P = {} N".format(P_tip))
print("-" * 72)
print(f"  {'Method':<30s} {'Tip Defl. (m)':<20s} {'Error vs Analytical':<20s}")
print(f"  {'-' * 30} {'-' * 20} {'-' * 20}")
print(f"  {'Analytical (E-B)':<30s} {delta_analytical:<20.6e} {'--':<20s}")
err_ops_s = abs(delta_ops - delta_analytical) / abs(delta_analytical) * 100
err_fl_s = abs(delta_fl - delta_analytical) / abs(delta_analytical) * 100
if not np.isnan(delta_ccx):
    err_ccx_s = abs(delta_ccx - delta_analytical) / abs(delta_analytical) * 100
    print(f"  {'CalculiX':<30s} {delta_ccx:<20.6e} {err_ccx_s:<19.2f}%")

print(f"  {'OpenSeesPy':<30s} {delta_ops:<20.6e} {err_ops_s:<19.2f}%")
print(f"  {'femlabpy':<30s} {delta_fl:<20.6e} {err_fl_s:<19.2f}%")

err_cross_s = abs(delta_fl - delta_ops) / abs(delta_ops) * 100
print(f"  {'femlabpy vs OpenSeesPy':<30s} {'':20s} {err_cross_s:<19.4f}%")
if not np.isnan(delta_ccx):
    err_cross_ccx = abs(delta_fl - delta_ccx) / abs(delta_ccx) * 100
    print(f"  {'femlabpy vs CalculiX':<30s} {'':20s} {err_cross_ccx:<19.4f}%")

# --- Modal ---
print()
print("2. MODAL ANALYSIS -- Natural Frequencies (Hz)")
print("-" * 72)
print(
    f"  {'Mode':<6s} {'Analytical':<14s} {'CalculiX':<14s} {'OpenSees':<14s} {'femlabpy':<14s} "
    f"{'CCX err%':<10s} {'OS err%':<10s} {'FL err%':<10s} {'FL-CCX err%':<12s}"
)
print(
    f"  {'-' * 6} {'-' * 14} {'-' * 14} {'-' * 14} {'-' * 14} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 12}"
)

n_compare = min(3, len(freq_analytical), len(freq_ops), len(modal_fl.freq_hz))
for i in range(n_compare):
    fa = freq_analytical[i]
    fo = freq_ops[i]
    fc = freq_ccx[i] if i < len(freq_ccx) else float("nan")
    ff = modal_fl.freq_hz[i]

    e_ops = abs(fo - fa) / fa * 100
    e_fl = abs(ff - fa) / fa * 100
    e_ccx = abs(fc - fa) / fa * 100 if not np.isnan(fc) else float("nan")
    e_cross = abs(ff - fc) / fc * 100 if fc > 0 else float("nan")

    fc_str = f"{fc:<14.4f}" if not np.isnan(fc) else f"{'--':<14s}"
    e_ccx_str = f"{e_ccx:<10.3f}" if not np.isnan(e_ccx) else f"{'--':<10s}"
    e_cross_str = f"{e_cross:<12.6f}" if not np.isnan(e_cross) else f"{'--':<12s}"

    print(
        f"  {i + 1:<6d} {fa:<14.4f} {fc_str} {fo:<14.4f} {ff:<14.4f} "
        f"{e_ccx_str} {e_ops:<10.3f} {e_fl:<10.3f} {e_cross_str}"
    )

# --- Dynamic ---
print()
print("3. DYNAMIC ANALYSIS -- Step Load, Undamped, Newmark (avg accel)")
print(f"   dt = {dt_dyn} s, {nsteps_dyn} steps, T_end = {T_end} s")
print("-" * 72)

# Analytical dynamic (first-mode approx)
t_anal = np.arange(nsteps_dyn + 1) * dt_dyn
u_tip_anal = delta_analytical * (1.0 - np.cos(omega_1_analytical * t_anal))

check_times = [0.005, 0.01, 0.02, 0.03, 0.05]
print(
    f"  {'Time (s)':<12s} {'Analytical':<16s} {'OpenSees':<16s} {'femlabpy':<16s} {'FL-OS diff':<14s}"
)
print(f"  {'-' * 12} {'-' * 16} {'-' * 16} {'-' * 16} {'-' * 14}")

for tc in check_times:
    idx = int(round(tc / dt_dyn))
    if idx >= len(t_anal):
        continue
    ua = u_tip_anal[idx]
    uo = u_tip_ops_hist[idx] if idx < len(u_tip_ops_hist) else float("nan")
    uf = result_fl.u[idx, tip_dof_fl]
    diff = abs(uf - uo) if not np.isnan(uo) else float("nan")
    print(f"  {tc:<12.4f} {ua:<16.6e} {uo:<16.6e} {uf:<16.6e} {diff:<14.2e}")

# RMS difference femlabpy vs OpenSees
common_len = min(len(u_tip_ops_hist), result_fl.u.shape[0])
rms_diff = np.sqrt(
    np.mean((u_tip_ops_hist[:common_len] - result_fl.u[:common_len, tip_dof_fl]) ** 2)
)
max_diff = np.max(
    np.abs(u_tip_ops_hist[:common_len] - result_fl.u[:common_len, tip_dof_fl])
)
ops_rms = np.sqrt(np.mean(u_tip_ops_hist[:common_len] ** 2))
print()
print(f"  RMS difference (femlabpy vs OpenSees): {rms_diff:.6e}")
print(f"  Max difference (femlabpy vs OpenSees): {max_diff:.6e}")
if ops_rms > 0:
    print(f"  Relative RMS error: {rms_diff / ops_rms * 100:.6f}%")

# --- Energy ---
print()
print("4. ENERGY CONSERVATION (femlabpy, undamped)")
print("-" * 72)
if result_fl.energy is not None:
    E_total = result_fl.energy["total"]
    # Skip first entry if zero (no energy at t=0)
    E_nonzero = E_total[E_total > 0]
    if len(E_nonzero) > 0:
        E0 = E_nonzero[0]
        E_var = np.max(np.abs(E_nonzero - E0))
        E_rel = E_var / E0 * 100 if E0 > 0 else 0
        print(f"  Total energy at t=dt: {E0:.6e}")
        print(f"  Max energy variation: {E_var:.6e}")
        print(f"  Relative variation:   {E_rel:.8f}%")
    else:
        print("  (no positive energy values)")
else:
    print("  (energy not computed)")

# --- Timing ---
print()
print("5. PERFORMANCE")
print("-" * 72)
print(f"  {'Operation':<40s} {'OpenSeesPy':<16s} {'femlabpy':<16s}")
print(f"  {'-' * 40} {'-' * 16} {'-' * 16}")
print(
    f"  {'Assembly / model build':<40s} {ops_assembly_time:<16.4f} {fl_assembly_time:<16.4f}"
)
print(f"  {'Modal solve':<40s} {ops_modal_time:<16.4f} {fl_modal_time:<16.4f}")
print(
    f"  {'Dynamic solve (' + str(nsteps_dyn) + ' steps)':<40s} {ops_dynamic_time:<16.4f} {fl_dynamic_time:<16.4f}"
)

print()
print("=" * 72)
print("CONCLUSION")
print("=" * 72)
if ops_rms > 0:
    rel_rms_pct = rms_diff / ops_rms * 100
else:
    rel_rms_pct = 0.0
print(f"  femlabpy matches OpenSeesPy to ~{max_diff:.2e} m max difference")
print(f"  (relative RMS = {rel_rms_pct:.6f}%) over {nsteps_dyn} time steps.")
print(f"  Static deflections agree within {err_cross_s:.4f}%.")
print(f"  Both FEM tools differ from E-B beam theory by ~{err_fl_s:.1f}%")
print(f"  (expected: 2D plane-stress FEM vs 1D beam is not exact).")
if result_fl.energy is not None:
    E_nonzero2 = E_total[E_total > 0]
    if len(E_nonzero2) > 0:
        E02 = E_nonzero2[0]
        E_var2 = np.max(np.abs(E_nonzero2 - E02))
        E_rel2 = E_var2 / E02 * 100 if E02 > 0 else 0
        print(f"  Energy conservation: {E_rel2:.8f}% variation (Newmark avg. accel).")
print("=" * 72)
