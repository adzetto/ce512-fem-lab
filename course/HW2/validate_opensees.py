"""
CE 512 Homework 2 — OpenSees Analysis & Data Export
Solves the truss, exports all TSV data for pgfplots, validates results.
Run: conda run -n opensees python validate_opensees.py
"""
import openseespy.opensees as ops
import numpy as np
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-10

# Problem data
E = 20000.0; A = 375.0; sigma_y = 300.0
b = h = 40.0; t = 2.5
bi = b - 2*t; hi = h - 2*t
I = (b*h**3 - bi*hi**3)/12.0
nodes = {1:(0.0,0.0), 2:(2500.0,0.0), 3:(2500.0,3000.0), 4:(4500.0,2200.0)}
elems = {1:(1,3), 2:(2,3), 3:(2,4), 4:(3,4)}

def build_model():
    ops.wipe()
    ops.model('basic','-ndm',2,'-ndf',2)
    for nid,(x,y) in nodes.items(): ops.node(nid,x,y)
    ops.fix(1,1,1); ops.fix(2,1,1)
    ops.uniaxialMaterial('Elastic',1,E)
    for eid,(ni,nj) in elems.items(): ops.element('Truss',eid,ni,nj,A,1)

def run_static(P):
    """Apply P downward at node 4, solve, return displacements and element results."""
    build_model()
    ops.timeSeries('Linear',1)
    ops.pattern('Plain',1,1)
    ops.load(4, 0.0, -P)
    ops.system('BandSPD'); ops.numberer('RCM'); ops.constraints('Plain')
    ops.integrator('LoadControl',1.0); ops.algorithm('Linear')
    ops.analysis('Static'); ops.analyze(1)
    
    d = {}
    for nid in nodes:
        d[nid] = (ops.nodeDisp(nid,1), ops.nodeDisp(nid,2))
    
    ops.reactions()
    R = {}
    for nid in [1,2]:
        R[nid] = (ops.nodeReaction(nid,1), ops.nodeReaction(nid,2))
    
    res = {}
    for eid,(ni,nj) in elems.items():
        f = ops.eleResponse(eid,'axialForce')
        N = f[0] if f and len(f)>0 else 0.0
        xi,yi = nodes[ni]; xj,yj = nodes[nj]
        Le = np.sqrt((xj-xi)**2+(yj-yi)**2)
        stress = N/A; strain = stress/E
        res[eid] = {'N':N, 'stress':stress, 'strain':strain, 'L':Le}
    
    ops.wipe()
    return d, R, res

def save_tsv(name, header, data):
    fp = os.path.join(OUTDIR, name)
    np.savetxt(fp, data, delimiter='\t', header='\t'.join(header), comments='')
    print(f"  {name} ({data.shape[0]} rows)")

# ============================================================
#  1. Unit load solution
# ============================================================
print("="*60)
print("  OpenSees FEM Analysis — CE 512 HW2")
print("="*60)

d1, R1, r1 = run_static(1.0)

print("\nUnit load (P=1 N) displacements:")
for nid in sorted(nodes):
    u,v = d1[nid]
    print(f"  Node {nid}: u={u:+.10e}, v={v:+.10e}")

print("\nUnit load reactions:")
for nid in [1,2]:
    rx,ry = R1[nid]
    print(f"  Node {nid}: Rx={rx:+.10e}, Ry={ry:+.10e}")

print("\nUnit load element results:")
for eid in sorted(r1):
    r = r1[eid]
    nat = "T" if r['N']>0 else "C"
    print(f"  Elem {eid}: N={r['N']:+.10e}, sigma={r['stress']:+.10e}, eps={r['strain']:+.10e} ({nat})")

# ============================================================
#  2. Critical loads
# ============================================================
Py = {}; Pb = {}
for eid,r in r1.items():
    if abs(r['stress'])>EPS: Py[eid] = sigma_y/abs(r['stress'])
    if r['N']<-EPS: Pb[eid] = (np.pi**2*E*I/r['L']**2)/abs(r['N'])

P_yield = min(Py.values()); e_yield = min(Py, key=Py.get)
P_buckle = min(Pb.values()); e_buckle = min(Pb, key=Pb.get)
print(f"\nFirst yield:  Elem {e_yield}, P_y = {P_yield:.6f} N = {P_yield/1000:.3f} kN")
print(f"First buckle: Elem {e_buckle}, P_cr = {P_buckle:.6f} N = {P_buckle/1000:.6f} kN")

# ============================================================
#  3. Export TSV data for pgfplots
# ============================================================
print("\n--- Exporting TSV data ---")

# Pushover
v4pp = abs(d1[4][1])
Parr = np.linspace(0, P_yield*1.05, 200)
save_tsv('data_pushover.tsv', ['v4_mm','P_N','P_kN'],
         np.column_stack([v4pp*Parr, Parr, Parr/1000]))

# Time history: ramp up to P_yield, ramp down
nstep = 200
t = np.linspace(0, 2.0, nstep)
Ph = np.where(t<=1.0, P_yield*t, P_yield*(2.0-t))
rows = []
for i in range(nstep):
    d_i, _, r_i = run_static(Ph[i])
    v4 = d_i[4][1]
    s1,s2,s3,s4 = r_i[1]['stress'], r_i[2]['stress'], r_i[3]['stress'], r_i[4]['stress']
    rows.append([t[i], Ph[i]/1000, v4, s1, s2, s3, s4])
data_th = np.array(rows)
save_tsv('data_timehistory.tsv', ['t','P_kN','v4_mm','sigma1','sigma2','sigma3','sigma4'], data_th)

# Load-displacement path
save_tsv('data_loaddisplacement.tsv', ['absv4_mm','P_kN'],
         np.column_stack([-data_th[:,2], data_th[:,1]]))

# Buckling zoom
Pz = np.linspace(0, P_buckle*2, 100)
save_tsv('data_buckling_zoom.tsv', ['v4_mm','P_N'],
         np.column_stack([v4pp*Pz, Pz]))

# Validation comparison
with open(os.path.join(OUTDIR,'data_validation_opensees.tsv'),'w') as f:
    f.write("Quantity\tOpenSees\n")
    for nid in [3,4]:
        u,v = d1[nid]
        f.write(f"u{nid}\t{u:.10e}\n")
        f.write(f"v{nid}\t{v:.10e}\n")
    for eid in sorted(r1):
        f.write(f"eps{eid}\t{r1[eid]['strain']:.10e}\n")
    f.write(f"P_yield\t{P_yield:.6f}\n")
    f.write(f"P_buckle\t{P_buckle:.6f}\n")
print("  data_validation_opensees.tsv")

print("\n" + "="*60)
print("  ALL DATA EXPORTED SUCCESSFULLY")
print("="*60)
