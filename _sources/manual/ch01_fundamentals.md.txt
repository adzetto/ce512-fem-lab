# Chapter 1: Fundamentals & Data Structures

## 1.1 The femlabpy Philosophy

`femlabpy` is a pure-Python finite element library that inherits the array-based, vectorized philosophy of the MATLAB/Scilab FemLab toolboxes. Unlike many modern object-oriented finite element frameworks that represent nodes, elements, and materials as deeply nested class instances, `femlabpy` relies entirely on flat, contiguous `numpy` arrays. This design choice has several profound implications:

1. **Performance:** By storing data in NumPy arrays, we leverage highly optimized C and Fortran backends for matrix operations, assembly, and slicing.
2. **Transparency:** The mathematical mapping from theory to code is direct. A stiffness matrix $K_e$ is simply a 2D array; a displacement vector $u$ is a 1D array.
3. **Didactic Clarity:** Students and researchers can inspect the entire state of the model at any point by simply printing an array. There are no hidden states or complex getters/setters.

## 1.2 Core Data Structures

A finite element model in `femlabpy` is completely described by five core matrices. Understanding the mathematical and structural shape of these arrays is the fundamental prerequisite for using the library.

### 1.2.1 Node Coordinates Matrix (`X`)

The matrix `X` stores the spatial coordinates of all nodes in the global coordinate system. For a 2D problem with $N$ nodes, `X` is an $N \times 2$ matrix. The row index corresponds to the global node ID.

$$
\mathbf{X} = \begin{bmatrix}
x_1 & y_1 \\
x_2 & y_2 \\
\vdots & \vdots \\
x_N & y_N
\end{bmatrix}
$$

*Note: `femlabpy` strictly uses 1-based indexing for node references in topology arrays to maintain compatibility with legacy meshes and mathematical conventions, although the Python arrays themselves are 0-indexed under the hood.*

### 1.2.2 Topology Matrix (`T`)

The topology matrix `T` defines the connectivity of the elements. For a mesh of 4-node quadrilaterals (Q4) with $E$ elements, `T` is an $E \times 5$ integer array. The first 4 columns contain the 1-based global node IDs that make up the element, ordered counter-clockwise. The final column is the material property ID.

$$
\mathbf{T} = \begin{bmatrix}
n_{1,1} & n_{1,2} & n_{1,3} & n_{1,4} & \text{prop}_1 \\
\vdots & \vdots & \vdots & \vdots & \vdots \\
n_{E,1} & n_{E,2} & n_{E,3} & n_{E,4} & \text{prop}_E
\end{bmatrix}
$$

### 1.2.3 Material Properties (`G`)

The material matrix `G` contains the physical properties assigned to the elements. For 2D plane stress/strain problems, a row typically takes the form: `[E, \nu, \text{type}, t, \rho]`, where:
- $E$: Young's Modulus
- $\nu$: Poisson's Ratio
- $\text{type}$: 1 for Plane Stress, 2 for Plane Strain
- $t$: Thickness (for Plane Stress)
- $\rho$: Density (for dynamic mass matrices)

### 1.2.4 Boundary Constraints (`C`)

Dirichlet boundary conditions (prescribed displacements) are stored in the `C` array. Each row specifies a constrained node, the degree of freedom (DOF) index (1 for $u_x$, 2 for $u_y$), and the prescribed value $\bar{u}$.

$$
\mathbf{C} = \begin{bmatrix}
\text{node\_id}_1 & \text{dof}_1 & \bar{u}_1 \\
\vdots & \vdots & \vdots
\end{bmatrix}
$$

### 1.2.5 Point Loads (`P`)

Neumann boundary conditions (point loads) are stored in the `P` array, following a similar format to `C`: `[node_id, dof, force_value]`.

## 1.3 The Finite Element Analysis Sequence

The standard linear static analysis in `femlabpy` follows a rigorous mathematical sequence:

1. **Initialization:** The global stiffness matrix $\mathbf{K}$ and load vector $\mathbf{p}$ are initialized to zero.
   ```python
   K, p = fp.init(nn, dof)
   ```
2. **Assembly:** Element stiffness matrices $\mathbf{K}_e$ are computed and assembled into $\mathbf{K}$.
   $$ \mathbf{K} = \sum_{e=1}^{E} \mathbf{L}_e^T \mathbf{K}_e \mathbf{L}_e $$
   ```python
   K = fp.kq4e(K, T, X, G)
   ```
3. **Load Application:** Point loads are mapped into the global load vector $\mathbf{p}$.
   ```python
   p = fp.setload(p, P, dof)
   ```
4. **Boundary Conditions:** Constraints are applied using the penalty method, modifying $\mathbf{K}$ and $\mathbf{p}$ in place.
   ```python
   K_bc, p_bc, _ = fp.setbc(K, p, C, dof)
   ```
5. **Solution:** The algebraic system $\mathbf{K}_{bc} \mathbf{u} = \mathbf{p}_{bc}$ is solved for the nodal displacements $\mathbf{u}$.
   ```python
   u = np.linalg.solve(K_bc, p_bc)
   ```
6. **Internal Forces Recovery:** Stresses and strains are computed at the element Gauss points.
   ```python
   q, S, E = fp.qq4e(np.zeros_like(p), T, X, G, u)
   ```

## 1.4 Step-by-Step Tutorial: Array Manipulation and Workflow

Let us dive deeply into a hands-on example to solidify these concepts. We will construct a simple single-element problem, exploring precisely how `femlabpy` expects you to manage and query its arrays. 

### 1.4.1 Constructing the Arrays

First, we import `numpy` and `femlabpy`.

```{code-block} python
import numpy as np
import femlabpy as fp
```

#### The Node Coordinates Matrix (`X`)
We define a single 4-node quadrilateral element of size 2x2. 

```{code-block} python
X = np.array([
    [0.0, 0.0],  # Node 1
    [2.0, 0.0],  # Node 2
    [2.0, 2.0],  # Node 3
    [0.0, 2.0]   # Node 4
])
```
* **Line-by-line breakdown:** We instantiate `X` as a 2D `numpy.ndarray`. The index in Python corresponds to the node ID offset by 1. For instance, `X[0]` gives `[0.0, 0.0]`, which represents global Node 1. 

#### The Topology Matrix (`T`)
Next, we define how these nodes are connected to form an element.

```{code-block} python
T = np.array([
    [1, 2, 3, 4, 1]  # Element 1: Nodes 1-2-3-4, Material Property 1
])
```
* **Line-by-line breakdown:** `T` is defined with five columns. The first four values `1, 2, 3, 4` are the **1-based global node IDs** ordered counter-clockwise. The last value `1` points to the first row of our material properties matrix `G`. *Crucially, `femlabpy` assumes user-facing topology uses 1-based indices to match traditional engineering literature.*

#### The Material Matrix (`G`)
We assign standard steel properties to our plane-stress problem.

```{code-block} python
G = np.array([
    [200e9, 0.3, 1, 0.1, 7850]  # Prop 1: E, nu, Plane Stress (1), t=0.1, rho
])
```
* **Line-by-line breakdown:** `G` represents physical constants. Since element 1 references property ID `1` in matrix `T`, the solver will read `G[0]` to obtain Young's modulus, Poisson's ratio, and element thickness.

#### Boundary Constraints (`C`) and Loads (`P`)
We clamp the left edge (Nodes 1 and 4) in both directions, and apply a 10 kN downward load on Node 3.

```{code-block} python
C = np.array([
    [1, 1, 0.0],  # Node 1, u_x = 0
    [1, 2, 0.0],  # Node 1, u_y = 0
    [4, 1, 0.0],  # Node 4, u_x = 0
    [4, 2, 0.0]   # Node 4, u_y = 0
])

P = np.array([
    [3, 2, -10000.0] # Node 3, F_y = -10 kN
])
```

### 1.4.2 Slicing and Querying the Arrays

To understand how `femlabpy` accesses node coordinates internally, let's explore slicing the topology array.

If we want to extract the coordinates of the nodes belonging to the first element, we might be tempted to use `X[T[0]]`. However, because `T` includes the material ID in the last column, we must first **slice** the array to exclude it.

```{code-block} python
element_0_nodes = T[0, :4] 
# returns array([1, 2, 3, 4])
```
* **Line-by-line breakdown:** `T[0, :4]` accesses the first row (index `0`) and slices the columns from index `0` up to, but not including, `4`. This isolates the node IDs.

Because `T` uses **1-based** node numbering (as is standard in finite element literature), but Python arrays are **0-indexed**, we must subtract `1` before querying `X`.

```{code-block} python
zero_indexed_nodes = element_0_nodes - 1
# returns array([0, 1, 2, 3])

element_coords = X[zero_indexed_nodes]
# returns array([[0., 0.], [2., 0.], [2., 2.], [0., 2.]])
```
* **Line-by-line breakdown:** Advanced `numpy` indexing allows us to pass the array `[0, 1, 2, 3]` directly into `X`. `numpy` elegantly returns a sub-array containing exactly the spatial coordinates for the nodes of Element 1. This pattern—`X[T[e, :4]-1]`—is the exact mathematical maneuver `femlabpy` employs internally to compute element Jacobian matrices.

### 1.4.3 Executing the Solution Workflow

Now, let us examine the analysis phase. Each function manipulates the system sequentially.

#### 1. Initialization (`init`)
```{code-block} python
nn = X.shape[0]  # Number of nodes = 4
dof = 2          # Degrees of freedom per node
K, p = fp.init(nn, dof)
```
* **Why it is called:** Before we can assemble our system, we need containers of the correct size. The global stiffness matrix must be $N_{\text{dof}} \times N_{\text{dof}}$, where $N_{\text{dof}} = 4 \times 2 = 8$. 
* **What it does:** `fp.init` creates an $8 \times 8$ zero matrix for `K` and an $8 \times 1$ zero vector for `p`.

#### 2. Element Assembly (`kq4e`)
```{code-block} python
K = fp.kq4e(K, T, X, G)
```
* **Why it is called:** The global stiffness matrix currently contains zeros. We must calculate the $8 \times 8$ element stiffness matrix $K_e$ for our quadrilateral and add it to the global matrix.
* **What it does:** `fp.kq4e` iterates over the rows of `T`. For each element, it looks up the coordinates in `X` and the material in `G`, performs Gaussian quadrature to integrate the stiffness terms, and scatters the values into the global `K` matrix.

#### 3. Loading (`setload`)
```{code-block} python
p = fp.setload(p, P, dof)
```
* **Why it is called:** External forces must be registered in the right-hand side vector.
* **What it does:** It iterates over `P`. For the load at Node 3, direction $y$ (DOF 2), it calculates the global index: `(3 - 1) * 2 + (2 - 1) = 5`. It then adds `-10000.0` to `p[5]`.

#### 4. Boundary Conditions (`setbc`)
```{code-block} python
K_bc, p_bc, _ = fp.setbc(K, p, C, dof)
```
* **Why it is called:** A stiffness matrix without supports is singular (its determinant is zero, meaning rigid body motion is possible). We cannot invert it. We must enforce the clamp at the left edge.
* **What it does:** `fp.setbc` uses the penalty method. For each constrained DOF in `C`, it places an enormously large number (e.g., $10^{15}$) on the corresponding diagonal entry of `K`. It also sets the corresponding entry in `p` to $\text{penalty} \times \bar{u}$. This mathematically coerces the solver to yield exactly $\bar{u}$ for that DOF without changing the matrix size.

#### 5. Solving the System
```{code-block} python
u = np.linalg.solve(K_bc, p_bc)
```
* **Why it is called:** With a properly conditioned `K_bc` and an applied load vector `p_bc`, the system of linear algebraic equations $\mathbf{K}\mathbf{u} = \mathbf{p}$ is complete.
* **What it does:** We utilize `numpy.linalg.solve`, an efficient LAPACK-based routine, to invert the system and find the nodal displacement vector `u`.