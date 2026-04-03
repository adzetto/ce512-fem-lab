function run_solver_case(case_name, repo_root, output_root, matlab_root)
%RUN_SOLVER_CASE Run one benchmark case headlessly and export TSV artifacts.

if nargin < 1 || strlength(string(case_name)) == 0
    error("run_solver_case:CaseRequired", "A benchmark case name is required.");
end

script_dir = fileparts(mfilename("fullpath"));
if nargin < 2 || strlength(string(repo_root)) == 0
    repo_root = fileparts(fileparts(script_dir));
end
if nargin < 3 || strlength(string(output_root)) == 0
    output_root = fullfile(repo_root, 'benchmarks', 'matlab_regression');
end
if nargin < 4 || strlength(string(matlab_root)) == 0
    matlab_root = resolve_matlab_root(repo_root);
end

repo_root = char(repo_root);
case_name = char(case_name);
comparison_root = char(output_root);
inputs_dir = fullfile(comparison_root, 'inputs', case_name);
results_dir = fullfile(comparison_root, 'raw', 'matlab', case_name);
matlab_root = char(matlab_root);
examples_root = fullfile(matlab_root, 'examples');

ensure_dir(inputs_dir);
ensure_dir(results_dir);

addpath(script_dir);
addpath(matlab_root);
addpath(examples_root);
set(0, 'DefaultFigureVisible', 'off');

data = load_case_data(case_name, repo_root);
export_inputs(inputs_dir, data);
result = run_case_solver(case_name, data);
export_results(results_dir, result);
end

function data = load_case_data(case_name, repo_root)
switch case_name
    case "cantilever_q4"
        canti;
        data = struct("X", X, "T", T, "G", G, "C", C, "P", P, "dof", dof);
    case "gmsh_triangle_t3"
        mesh = load_gmsh(resolve_mesh_path(repo_root));
        triangles = mesh.TRIANGLES(1:mesh.nbTriangles, :);
        refs = triangles(:, 4);
        [~, ~, props] = unique(refs, "stable");
        data = struct( ...
            "X", mesh.POS(:, 1:2), ...
            "T", [triangles(:, 1:3), props], ...
            "G", [2.0e8, 0.3, 1.0; 0.7e8, 0.23, 1.0], ...
            "C", [5,1,0; 7,1,0; 8,1,0; 8,2,0; 9,2,0; 10,2,0; 9,1,0; 10,1,0; 11,1,0; 6,1,0], ...
            "P", [25,0,-0.05; 24,0,-0.1; 22,0,-0.05], ...
            "dof", 2);
    case "flow_q4"
        flow;
        data = struct("X", X, "T", T1, "G", G, "C", C, "dof", dof);
    case "flow_t3"
        flow;
        data = struct("X", X, "T", T2, "G", G, "C", C, "dof", dof);
    case {"bar01_nlbar", "bar02_nlbar", "bar03_nlbar"}
        run(strrep(case_name, "_nlbar", ".m"));
        data = struct( ...
            "X", X, "T", T, "G", G, "C", C, "P", P, ...
            "no_loadsteps", no_loadsteps, "i_max", i_max, ...
            "i_d", i_d, "TOL", TOL, "plotdof", plotdof);
    case {"square_plastps", "square_plastpe"}
        square;
        data = struct( ...
            "X", X, "T", T, "G", G, "C", C, "P", P, "dof", dof, ...
            "no_loadsteps", no_loadsteps, "i_max", i_max, ...
            "i_d", i_d, "TOL", TOL, "plotdof", plotdof);
    case {"hole_plastps", "hole_plastpe"}
        hole;
        data = struct( ...
            "X", X, "T", T, "G", G, "C", C, "P", P, "dof", dof, ...
            "no_loadsteps", no_loadsteps, "i_max", i_max, ...
            "i_d", i_d, "TOL", TOL, "plotdof", plotdof);
    otherwise
        error("run_solver_case:UnknownCase", "Unknown benchmark case '%s'.", case_name);
end
end

function result = run_case_solver(case_name, data)
switch case_name
    case "cantilever_q4"
        result = solve_elastic_q4(data);
    case "gmsh_triangle_t3"
        result = solve_triangle_t3(data);
    case "flow_q4"
        result = solve_flow_q4(data);
    case "flow_t3"
        result = solve_flow_t3(data);
    case {"bar01_nlbar", "bar02_nlbar", "bar03_nlbar"}
        result = solve_nlbar(data);
    case {"square_plastps", "hole_plastps"}
        result = solve_plastic(data, "ps");
    case {"square_plastpe", "hole_plastpe"}
        result = solve_plastic(data, "pe");
    otherwise
        error("run_solver_case:UnknownSolver", "No solver defined for '%s'.", case_name);
end
end

function result = solve_elastic_q4(data)
[K, p, q] = init(size(data.X, 1), data.dof);
K = kq4e(K, data.T, data.X, data.G);
p = setload(p, data.P);
[K, p] = setbc(K, p, data.C, data.dof);
u = K \ p;
[q, S, E] = qq4e(q, data.T, data.X, data.G, u);
R = constraint_reactions(q, data.C, data.dof);
result = struct("u", u, "q", q, "S", S, "E", E, "R", R);
end

function result = solve_triangle_t3(data)
[K, p, q] = init(size(data.X, 1), data.dof);
K = kt3e(K, data.T, data.X, data.G);
p = setload(p, data.P);
u = solve_lag_local(K, p, data.C, data.dof);
[q, S, E] = qt3e(q, data.T, data.X, data.G, u);
R = constraint_reactions(q, data.C, data.dof);
result = struct("u", u, "q", q, "S", S, "E", E, "R", R);
end

function result = solve_flow_q4(data)
[K, p, q] = init(size(data.X, 1), 1);
K = kq4p(K, data.T, data.X, data.G);
if isfield(data, "P")
    p = setload(p, data.P);
end
[K, p] = setbc(K, p, data.C);
u = K \ p;
[q, S, E] = qq4p(q, data.T, data.X, data.G, u);
R = constraint_reactions(q, data.C, 1);
result = struct("u", u, "q", q, "S", S, "E", E, "R", R);
end

function result = solve_flow_t3(data)
[K, p, q] = init(size(data.X, 1), 1);
K = kt3p(K, data.T, data.X, data.G);
if isfield(data, "P")
    p = setload(p, data.P);
end
[K, p] = setbc(K, p, data.C);
u = K \ p;
[q, S, E] = qt3p(q, data.T, data.X, data.G, u);
R = constraint_reactions(q, data.C, 1);
result = struct("u", u, "q", q, "S", S, "E", E, "R", R);
end

function result = solve_nlbar(data)
dof = size(data.X, 2);
ndof = size(data.X, 1) * dof;
u = zeros(ndof, 1);
du = zeros(ndof, 1);
f = zeros(ndof, 1);
df = zeros(ndof, 1);
df = setload(df, data.P);
U = [];
F = [];

n = 1;
i = data.i_d;
guard = 0;

while n <= data.no_loadsteps
    guard = guard + 1;
    if guard > 1000
        error("run_solver_case:NonlinearGuard", "Nonlinear bar solver exceeded the restart guard.");
    end

    if i < data.i_max
        K = zeros(ndof);
        K = kbar(K, data.T, data.X, data.G, u);
        [Kt, df] = setbc(K, df, data.C, dof);
        du0 = Kt \ df;
        if du' * du0 < 0
            df = -df;
            du0 = -du0;
        end

        if n == 1
            l0 = norm(du0);
            l = l0;
            l_max = 2 * l0;
        else
            l = norm(du);
            l0 = norm(du0);
        end
    end

    if data.i_d <= i && i < data.i_max
        du = min(l / l0, l_max / l0) * du0;
    elseif i < data.i_d
        du = min(2 * l / l0, l_max / l0) * du0;
    else
        du0 = 0.5 * du0;
        du = du0;
    end

    for i = 1:data.i_max
        q = zeros(ndof, 1);
        [q, S, E] = qbar(q, data.T, data.X, data.G, u + du);
        dq = q - f;
        xi = (dq' * du) / (df' * du);
        r = -dq + xi * df;
        if rnorm(r, data.C, dof) < data.TOL * rnorm(df, data.C, dof)
            break
        else
            [Kt, r] = setbc(K, r, data.C, dof);
            delta_u = Kt \ r;
            du = du + delta_u;
        end
    end

    if i < data.i_max
        f = f + xi * df;
        u = u + du;
        U(n + 1, 1) = u(data.plotdof); %#ok<AGROW>
        F(n + 1, 1) = f(data.plotdof); %#ok<AGROW>
        n = n + 1;
    end
end

q = zeros(ndof, 1);
[q, S, E] = qbar(q, data.T, data.X, data.G, u);
R = constraint_reactions(q, data.C, dof);
result = struct("u", u, "q", q, "S", S, "E", E, "R", R, "f", f, "path_u", U, "path_f", F);
end

function result = solve_plastic(data, mode)
dof = data.dof;
ndof = size(data.X, 1) * dof;
f = zeros(ndof, 1);
df = zeros(ndof, 1);
df = setload(df, data.P);
u = zeros(ndof, 1);
du = zeros(ndof, 1);
U = [];
F = [];

nelem = size(data.T, 1);
S = zeros(nelem, 1);
E = zeros(nelem, 1);
mattype = 1;
n = 1;
i = data.i_d;
i_tot = 0;
guard = 0;

while n <= data.no_loadsteps
    guard = guard + 1;
    if guard > 2000
        error("run_solver_case:PlasticGuard", "Plastic solver exceeded the restart guard.");
    end

    if i < data.i_max
        K = zeros(ndof);
        if strcmp(mode, "ps")
            K = kq4eps(K, data.T, data.X, data.G, S, E, mattype);
        else
            K = kq4epe(K, data.T, data.X, data.G, S, E, mattype);
        end
        K = sparse(K);
        [Kt, df] = setbc(K, df, data.C, dof);
        du0 = Kt \ df;
        if du' * du0 < 0
            df = -df;
            du0 = -du0;
        end
        if n == 1
            l0 = norm(du0);
            l = l0;
            l_max = 2 * l0;
        else
            l = norm(du);
            l0 = norm(du0);
        end
    end

    if data.i_d <= i && i < data.i_max
        du = min(l / l0, l_max / l0) * du0;
    elseif i < data.i_d
        du = min(2 * l / l0, l_max / l0) * du0;
    else
        du0 = 0.5 * du0;
        du = du0;
    end

    for i = 1:data.i_max
        q = zeros(ndof, 1);
        if strcmp(mode, "ps")
            [q, Sn, En] = qq4eps(q, data.T, data.X, data.G, u + du, S, E, mattype);
        else
            [q, Sn, En] = qq4epe(q, data.T, data.X, data.G, u + du, S, E, mattype);
        end
        dq = q - f;
        xi = (dq' * du) / (df' * du);
        r = -dq + xi * df;

        if rnorm(r, data.C, dof) < data.TOL * rnorm(df, data.C, dof)
            break
        else
            [Kt, r] = setbc(K, r, data.C, dof);
            delta_u = Kt \ r;
            du = du + delta_u;
        end
    end

    if i < data.i_max
        f = f + xi * df;
        u = u + du;
        S = Sn;
        E = En;
        U(n + 1, 1) = u(data.plotdof); %#ok<AGROW>
        F(n + 1, 1) = f(data.plotdof); %#ok<AGROW>
        i_tot = i_tot + i;
        n = n + 1;
    end
end

q = zeros(ndof, 1);
if strcmp(mode, "ps")
    [q, S, E] = qq4eps(q, data.T, data.X, data.G, u, S, E, mattype);
else
    [q, S, E] = qq4epe(q, data.T, data.X, data.G, u, S, E, mattype);
end
R = constraint_reactions(q, data.C, dof);
result = struct("u", u, "q", q, "S", S, "E", E, "R", R, "f", f, "path_u", U, "path_f", F, "i_tot", i_tot);
end

function export_inputs(root_dir, data)
fields = fieldnames(data);
for k = 1:numel(fields)
    name = fields{k};
    write_tsv(fullfile(root_dir, [name, '.tsv']), data.(name));
end
end

function export_results(root_dir, result)
fields = fieldnames(result);
for k = 1:numel(fields)
    name = fields{k};
    write_tsv(fullfile(root_dir, [name, '.tsv']), result.(name));
end
end

function R = constraint_reactions(q, C, dof)
if dof == 1
    indices = C(:, 1);
    R = [C(:, 1), ones(size(C, 1), 1), q(indices)];
else
    indices = (C(:, 1) - 1) * dof + C(:, 2);
    R = [C(:, 1:2), q(indices)];
end
end

function u = solve_lag_local(K, p, C, dof)
n_constraints = size(C, 1);
system_size = size(K, 1);
ks = 1.0e-2 * max(abs(diag(K)));
if ks == 0
    ks = 1.0;
end

G = zeros(n_constraints, system_size);
Q = ks * C(:, end);
for i = 1:n_constraints
    index = (C(i, 1) - 1) * dof + C(i, 2);
    G(i, index) = ks;
end

Kbar = [K, G'; G, zeros(n_constraints)];
pbar = [p; Q];
ubar = Kbar \ pbar;
u = ubar(1:system_size);
end

function write_tsv(pathname, value)
pathname = char(pathname);
value = full(double(value));
fid = fopen(pathname, 'w');
if fid < 0
    error("run_solver_case:WriteFailed", "Could not open '%s' for writing.", pathname);
end
cleanup = onCleanup(@() fclose(fid));
if isempty(value)
    return
end

% Only reshape true column-like vectors; preserve row vectors (1xN matrices
% like G = [E, nu, ptype]) so that downstream consumers see the correct
% matrix shape.
if iscolumn(value) || isscalar(value)
    % already fine
elseif isvector(value) && size(value,1) ~= 1
    value = reshape(value, [], 1);
end

[nr, nc] = size(value);
for i = 1:nr
    for j = 1:nc
        if j > 1
            fprintf(fid, "\t");
        end
        fprintf(fid, "%.16g", value(i, j));
    end
    fprintf(fid, "\n");
end
end

function ensure_dir(pathname)
pathname = char(pathname);
if ~exist(pathname, "dir")
    mkdir(pathname);
end
end

function matlab_root = resolve_matlab_root(repo_root)
candidates = {
    getenv('FEMLAB_MATLAB_ROOT')
    fullfile(fileparts(repo_root), 'HWs', 'HW4', 'tmp_femlab_matlab', 'FemLab_matlab', 'M_Files')
    'C:\Users\lenovo\Downloads\FemLab_matlab\FemLab_matlab\M_Files'
};
for k = 1:numel(candidates)
    candidate = char(candidates{k});
    if strlength(string(candidate)) == 0
        continue
    end
    if exist(candidate, 'dir')
        matlab_root = candidate;
        return
    end
end
error("run_solver_case:MissingMatlabRoot", "Could not locate the original FemLab MATLAB M_Files directory.");
end

function mesh_path = resolve_mesh_path(repo_root)
candidates = {
    fullfile(repo_root, 'src', 'femlabpy', 'data', 'meshes', 'deneme.msh')
    fullfile(repo_root, 'mesh', 'deneme.msh')
};
for k = 1:numel(candidates)
    candidate = char(candidates{k});
    if exist(candidate, 'file')
        mesh_path = candidate;
        return
    end
end
error("run_solver_case:MissingMesh", "Could not locate deneme.msh in the repository.");
end
